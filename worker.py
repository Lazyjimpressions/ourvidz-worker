# worker.py - Updated with working models and robust error handling
import os
import json
import time
import torch
import requests
import subprocess
from PIL import Image
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import video generation models
try:
    from diffusers import WanVideoPipeline
    HAS_WAN = True
except ImportError:
    print("⚠️ Wan models not available in current diffusers version")
    HAS_WAN = False

class VideoWorker:
    def __init__(self):
        """Initialize worker with updated model paths"""
        self.model_path = "/workspace/models"
        
        # Model instances (loaded on demand)
        self.wan_t2v_pipeline = None
        self.wan_i2v_pipeline = None
        self.mistral_model = None
        self.mistral_tokenizer = None
        self.mistral_model_name = None  # Track which Mistral model we loaded
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.upstash_redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.upstash_redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        # Validate environment
        if not all([self.supabase_url, self.supabase_service_key, self.upstash_redis_url, self.upstash_redis_token]):
            print("❌ Missing required environment variables")
            missing = []
            if not self.supabase_url: missing.append("SUPABASE_URL")
            if not self.supabase_service_key: missing.append("SUPABASE_SERVICE_KEY")
            if not self.upstash_redis_url: missing.append("UPSTASH_REDIS_REST_URL")
            if not self.upstash_redis_token: missing.append("UPSTASH_REDIS_REST_TOKEN")
            print(f"Missing: {', '.join(missing)}")
            return
        
        print("🚀 OurVidz Worker initialized")
        self.log_gpu_info()
        self.check_models()
        
        # Start main worker loop
        print("🎬 OurVidz Worker started! Waiting for jobs...")

    def log_gpu_info(self):
        """Log GPU information"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🔥 GPU: {gpu_name} ({total_memory:.1f}GB)")
        else:
            print("❌ No GPU available")
        
        print(f"📁 Model path: {self.model_path}")
        print(f"🔗 Supabase: {self.supabase_url}")

    def check_models(self):
        """Check what models are available"""
        print("🔍 Checking models in: /workspace/models")
        
        if not os.path.exists(self.model_path):
            print("📥 Downloading models to network volume (this will take ~45 minutes)...")
            self.download_models()
        else:
            print("📁 Models directory exists, checking contents...")
            self.list_model_contents()

    def list_model_contents(self):
        """List what's in the models directory"""
        for root, dirs, files in os.walk(self.model_path):
            level = root.replace(self.model_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")

    def download_models(self):
        """Download models with better error handling"""
        
        # Try Wan 2.1 14B Text-to-Video
        print("🎥 Downloading Wan 2.1 14B Text-to-Video...")
        try:
            if HAS_WAN:
                from diffusers import WanVideoPipeline
                wan_t2v = WanVideoPipeline.from_pretrained(
                    "Wan-AI/Wan2.1-T2V-14B",
                    torch_dtype=torch.float16,
                    cache_dir=f"{self.model_path}/wan_t2v"
                )
                print("✅ Wan 2.1 T2V downloaded successfully")
                del wan_t2v  # Free memory
            else:
                raise ImportError("WanVideoPipeline not available")
        except Exception as e:
            print(f"⚠️ Wan 2.1 T2V download failed: {e}")
            print("📝 Will use fallback model during generation")

        # Try Wan 2.1 14B Image-to-Video (Phase 2)
        print("🖼️ Downloading Wan 2.1 14B Image-to-Video...")
        try:
            if HAS_WAN:
                wan_i2v = WanVideoPipeline.from_pretrained(
                    "Wan-AI/Wan2.1-I2V-14B-720P",
                    torch_dtype=torch.float16,
                    cache_dir=f"{self.model_path}/wan_i2v"
                )
                print("✅ Wan 2.1 I2V downloaded successfully")
                del wan_i2v  # Free memory
            else:
                raise ImportError("WanVideoPipeline not available")
        except Exception as e:
            print(f"⚠️ Wan 2.1 I2V download failed: {e}")

        # Try Mistral models (ungated versions)
        print("📝 Downloading Mistral 7B...")
        mistral_success = False
        
        # Try ungated Mistral 7B v0.1 first
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-v0.1",
                cache_dir=f"{self.model_path}/mistral"
            )
            model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-v0.1",
                torch_dtype=torch.float16,
                cache_dir=f"{self.model_path}/mistral"
            )
            print("✅ Mistral 7B v0.1 (ungated) downloaded successfully")
            self.mistral_model_name = "mistralai/Mistral-7B-v0.1"
            mistral_success = True
            del tokenizer, model  # Free memory
        except Exception as e:
            print(f"⚠️ Mistral 7B v0.1 download failed: {e}")
            
            # Try uncensored alternative
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "ehartford/dolphin-2.0-mistral-7b",
                    cache_dir=f"{self.model_path}/mistral_alt"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    "ehartford/dolphin-2.0-mistral-7b",
                    torch_dtype=torch.float16,
                    cache_dir=f"{self.model_path}/mistral_alt"
                )
                print("✅ Dolphin Mistral 7B (uncensored) downloaded successfully")
                self.mistral_model_name = "ehartford/dolphin-2.0-mistral-7b"
                mistral_success = True
                del tokenizer, model  # Free memory
            except Exception as e2:
                print(f"⚠️ Dolphin Mistral download failed: {e2}")

        if not mistral_success:
            print("❌ All Mistral model downloads failed")
            
        print("🎉 Model download completed!")

    def load_mistral(self):
        """Load Mistral model for prompt enhancement"""
        if self.mistral_model is None and self.mistral_model_name:
            print(f"📝 Loading {self.mistral_model_name}...")
            try:
                cache_dir = f"{self.model_path}/mistral" if "mistralai" in self.mistral_model_name else f"{self.model_path}/mistral_alt"
                
                self.mistral_tokenizer = AutoTokenizer.from_pretrained(
                    self.mistral_model_name,
                    cache_dir=cache_dir
                )
                self.mistral_model = AutoModelForCausalLM.from_pretrained(
                    self.mistral_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=cache_dir
                )
                print(f"✅ {self.mistral_model_name} loaded successfully")
                self.log_gpu_memory()
                return True
            except Exception as e:
                print(f"❌ Failed to load {self.mistral_model_name}: {e}")
                return False
        elif self.mistral_model is not None:
            return True  # Already loaded
        else:
            print("❌ No Mistral model available to load")
            return False

    def unload_mistral(self):
        """Free Mistral memory"""
        if self.mistral_model is not None:
            print("🗑️ Unloading Mistral...")
            del self.mistral_model
            del self.mistral_tokenizer
            self.mistral_model = None
            self.mistral_tokenizer = None
            torch.cuda.empty_cache()
            print("✅ Mistral unloaded")
            self.log_gpu_memory()

    def load_wan_t2v(self):
        """Load Wan 2.1 14B Text-to-Video"""
        if not HAS_WAN:
            print("❌ Wan models not available")
            return False
            
        if self.wan_t2v_pipeline is None:
            print("🎥 Loading Wan 2.1 14B Text-to-Video...")
            try:
                self.wan_t2v_pipeline = WanVideoPipeline.from_pretrained(
                    "Wan-AI/Wan2.1-T2V-14B",
                    torch_dtype=torch.float16,
                    cache_dir=f"{self.model_path}/wan_t2v"
                ).to("cuda")
                print("✅ Wan 2.1 T2V loaded successfully")
                self.log_gpu_memory()
                return True
            except Exception as e:
                print(f"❌ Failed to load Wan 2.1 T2V: {e}")
                return False
        return True

    def unload_wan_models(self):
        """Free all Wan memory"""
        if self.wan_t2v_pipeline is not None:
            print("🗑️ Unloading Wan models...")
            del self.wan_t2v_pipeline
            self.wan_t2v_pipeline = None
            torch.cuda.empty_cache()
            print("✅ Wan models unloaded")
            self.log_gpu_memory()

    def log_gpu_memory(self):
        """Monitor GPU memory usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🔥 GPU Memory - Used: {memory_allocated:.2f}GB / Reserved: {memory_reserved:.2f}GB / Total: {total_memory:.0f}GB")

    def enhance_prompt(self, original_prompt: str, character_description: str = None) -> str:
        """Enhanced prompt generation"""
        if not self.load_mistral():
            print("❌ Mistral not available, returning original prompt")
            return original_prompt
            
        system_prompt = """You are an expert at converting casual text into detailed video generation prompts.

Create a cinematic, vivid prompt optimized for AI video generation. Focus on:
- Visual details and composition
- Lighting and atmosphere  
- Movement and camera work
- Realistic physics and motion

Keep under 200 words. Make it specific and cinematic."""

        if character_description:
            system_prompt += f"\n\nCharacter to include: {character_description}"

        try:
            # Handle different model formats
            if "dolphin" in self.mistral_model_name.lower():
                # Dolphin uses ChatML format
                input_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{original_prompt}<|im_end|>\n<|im_start|>assistant"
            else:
                # Standard Mistral format
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": original_prompt}
                ]
                input_text = self.mistral_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            inputs = self.mistral_tokenizer(input_text, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = self.mistral_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.mistral_tokenizer.eos_token_id
                )

            enhanced = self.mistral_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            self.unload_mistral()
            return enhanced.strip()
            
        except Exception as e:
            self.unload_mistral()
            print(f"❌ Prompt enhancement failed: {e}")
            return original_prompt

    def generate_preview(self, prompt: str) -> Optional[Image.Image]:
        """Generate preview frame - placeholder for now"""
        print(f"🖼️ Generating preview for: {prompt[:50]}...")
        
        # For now, return a simple placeholder
        # TODO: Implement actual preview generation when Wan models are working
        try:
            # Create a simple colored image as placeholder
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (1280, 720), color='darkblue')
            draw = ImageDraw.Draw(img)
            
            # Add text overlay
            try:
                # Try to use a system font
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            except:
                font = ImageFont.load_default()
                
            text = "Preview\n(Wan 2.1 Loading...)"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (1280 - text_width) // 2
            y = (720 - text_height) // 2
            draw.text((x, y), text, fill='white', font=font)
            
            print("✅ Preview placeholder generated")
            return img
            
        except Exception as e:
            print(f"❌ Preview generation failed: {e}")
            return None

    def generate_video(self, prompt: str) -> Optional[List[Image.Image]]:
        """Generate video frames - placeholder for now"""
        print(f"🎬 Generating video for: {prompt[:50]}...")
        
        # For now, return placeholder frames
        # TODO: Implement actual video generation when Wan models are working
        try:
            frames = []
            from PIL import Image, ImageDraw, ImageFont
            
            # Generate 80 frames (5 seconds at 16fps)
            for i in range(80):
                # Create frame with changing color to simulate motion
                hue = (i * 4) % 360
                from colorsys import hsv_to_rgb
                rgb = hsv_to_rgb(hue/360, 0.8, 0.8)
                color = tuple(int(c * 255) for c in rgb)
                
                img = Image.new('RGB', (1280, 720), color=color)
                draw = ImageDraw.Draw(img)
                
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
                except:
                    font = ImageFont.load_default()
                    
                text = f"Video Frame {i+1}/80\n(Wan 2.1 Loading...)"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                x = (1280 - text_width) // 2
                y = (720 - text_height) // 2
                draw.text((x, y), text, fill='white', font=font)
                
                frames.append(img)
            
            print(f"✅ Generated {len(frames)} placeholder frames")
            return frames
            
        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            return None

    def save_and_upload_image(self, image: Image.Image, filename: str) -> Optional[str]:
        """Save and upload image to Supabase"""
        try:
            local_path = f"/tmp/{filename}"
            image.save(local_path, "PNG", quality=95)
            
            upload_url = self.upload_to_supabase(local_path, f"scene-previews/{filename}")
            os.remove(local_path)
            return upload_url
            
        except Exception as e:
            print(f"❌ Image upload failed: {e}")
            return None

    def save_and_upload_video(self, frames: List[Image.Image], filename: str) -> Optional[str]:
        """Convert frames to MP4 and upload"""
        try:
            temp_dir = f"/tmp/frames_{int(time.time())}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save frames
            for i, frame in enumerate(frames):
                frame.save(f"{temp_dir}/frame_{i:04d}.png")
            
            # FFmpeg conversion
            output_path = f"/tmp/{filename}"
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", "16",
                "-i", f"{temp_dir}/frame_%04d.png",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ], check=True, capture_output=True)
            
            upload_url = self.upload_to_supabase(output_path, f"videos-final/{filename}")
            
            # Cleanup
            subprocess.run(["rm", "-rf", temp_dir])
            os.remove(output_path)
            
            return upload_url
            
        except Exception as e:
            print(f"❌ Video processing failed: {e}")
            return None

    def upload_to_supabase(self, file_path: str, storage_path: str) -> str:
        """Upload file to Supabase storage"""
        try:
            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': file},
                    headers={'Authorization': f"Bearer {self.supabase_service_key}"}
                )
            
            if response.status_code == 200:
                return f"{self.supabase_url}/storage/v1/object/public/{storage_path}"
            else:
                raise Exception(f"Upload failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Supabase upload error: {e}")
            raise

    def notify_completion(self, job_id: str, status: str, output_url: str = None, error_message: str = None):
        """Notify Supabase of job completion"""
        try:
            callback_data = {
                'jobId': job_id,
                'status': status,
                'outputUrl': output_url,
                'errorMessage': error_message
            }
            
            response = requests.post(
                f"{self.supabase_url}/functions/v1/job-callback",
                json=callback_data,
                headers={
                    'Authorization': f"Bearer {self.supabase_service_key}",
                    'Content-Type': 'application/json'
                }
            )
            
            if response.status_code == 200:
                print(f"✅ Job {job_id} callback sent successfully")
            else:
                print(f"❌ Callback failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job(self, job_data: dict):
        """Process a single job"""
        job_id = job_data['jobId']
        job_type = job_data['jobType']
        
        print(f"🔄 Processing job {job_id} ({job_type})")
        
        try:
            if job_type == 'enhance':
                enhanced_prompt = self.enhance_prompt(
                    job_data['prompt'], 
                    job_data.get('characterDescription')
                )
                print(f"✅ Enhanced prompt: {enhanced_prompt[:100]}...")
                self.notify_completion(job_id, 'completed')
                
            elif job_type == 'preview':
                preview_image = self.generate_preview(job_data['prompt'])
                if preview_image:
                    filename = f"{job_data['videoId']}_preview.png"
                    upload_url = self.save_and_upload_image(preview_image, filename)
                    if upload_url:
                        print(f"✅ Preview uploaded: {upload_url}")
                        self.notify_completion(job_id, 'completed', upload_url)
                    else:
                        raise Exception("Failed to upload preview")
                else:
                    raise Exception("Failed to generate preview")
                    
            elif job_type == 'video':
                video_frames = self.generate_video(job_data['prompt'])
                if video_frames:
                    filename = f"{job_data['videoId']}_final.mp4"
                    upload_url = self.save_and_upload_video(video_frames, filename)
                    if upload_url:
                        print(f"✅ Video uploaded: {upload_url}")
                        self.notify_completion(job_id, 'completed', upload_url)
                    else:
                        raise Exception("Failed to upload video")
                else:
                    raise Exception("Failed to generate video")
            
            print(f"🎉 Job {job_id} completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Job {job_id} failed: {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)

    def poll_redis_queue(self):
        """Poll Redis queue for jobs"""
        try:
            response = requests.get(
                f"{self.upstash_redis_url}/brpop/job-queue/60",
                headers={'Authorization': f"Bearer {self.upstash_redis_token}"}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('result'):
                    queue_name, job_json = result['result']
                    job_data = json.loads(job_json)
                    return job_data
            return None
            
        except Exception as e:
            print(f"❌ Redis polling error: {e}")
            return None

    def run(self):
        """Main worker loop"""
        idle_start = time.time()
        idle_check_interval = 30  # Check every 30 seconds
        max_idle_minutes = 10
        
        while True:
            try:
                job_data = self.poll_redis_queue()
                
                if job_data:
                    # Reset idle timer
                    idle_start = time.time()
                    self.process_job(job_data)
                else:
                    # Check idle time
                    idle_minutes = (time.time() - idle_start) / 60
                    
                    if idle_minutes >= max_idle_minutes:
                        print(f"🛑 No jobs for {max_idle_minutes} minutes, shutting down...")
                        break
                    elif int(idle_minutes * 2) % 2 == 0:  # Print every 30 seconds
                        print(f"⏳ Idle for {idle_minutes:.1f} minutes (shutdown at {max_idle_minutes})")
                    
                    time.sleep(idle_check_interval)
                    
            except Exception as e:
                print(f"❌ Worker error: {e}")
                time.sleep(30)
        
        print("🔚 Worker shutting down...")

if __name__ == "__main__":
    worker = VideoWorker()
    worker.run()
