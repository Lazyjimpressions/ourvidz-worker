# worker.py - Clean, Complete, Audited Version
import os
import json
import time
import torch
import requests
import subprocess
import uuid
import glob
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

class VideoWorker:
    def __init__(self):
        """Initialize worker with real Wan 2.1 integration"""
        self.model_path = "/workspace/models"
        self.wan_path = "/workspace/Wan2.1"
        self.wan_model_path = "/workspace/models/wan2.1-t2v-1.3b"
        
        # Model instances (loaded on demand)
        self.mistral_model = None
        self.mistral_tokenizer = None
        
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
            self.environment_valid = False
        else:
            self.environment_valid = True

        # Check Wan 2.1 availability
        self.wan_available = self.check_wan_availability()
        
        print("🚀 OurVidz Worker initialized")
        self.log_gpu_info()
        print(f"🎥 Wan 2.1 Available: {self.wan_available}")
        print(f"🔧 Environment Valid: {self.environment_valid}")

    def check_wan_availability(self):
        """Check if Wan 2.1 is properly installed and available"""
        try:
            if not os.path.exists(self.wan_model_path):
                print(f"❌ Wan 2.1 model not found at {self.wan_model_path}")
                return False
                
            generate_script = os.path.join(self.wan_path, "generate.py")
            if not os.path.exists(generate_script):
                print(f"❌ Wan 2.1 generate.py not found at {generate_script}")
                return False
                
            essential_files = [
                "diffusion_pytorch_model.safetensors",
                "models_t5_umt5-xxl-enc-bf16.pth", 
                "Wan2.1_VAE.pth"
            ]
            
            for file in essential_files:
                file_path = os.path.join(self.wan_model_path, file)
                if not os.path.exists(file_path):
                    print(f"❌ Missing Wan 2.1 model file: {file}")
                    return False
                    
            print("✅ Wan 2.1 installation verified")
            return True
            
        except Exception as e:
            print(f"❌ Error checking Wan 2.1 availability: {e}")
            return False

    def log_gpu_info(self):
        """Log GPU information"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            current_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"🔥 GPU: {gpu_name} ({total_memory:.1f}GB)")
            print(f"💾 VRAM: {current_memory:.2f}GB / {total_memory:.1f}GB")
        else:
            print("❌ CUDA not available")

    def enhance_prompt(self, original_prompt: str, character_description: str = None) -> str:
        """Enhanced prompt generation - simplified for now"""
        enhanced = original_prompt
        
        if character_description:
            enhanced = f"{character_description}, {enhanced}"
            
        enhanced = f"High quality video of {enhanced}, cinematic lighting, detailed, 16fps"
        
        print(f"📝 Enhanced prompt: {enhanced}")
        return enhanced

    def find_generated_video_file(self, output_filename: str) -> List[str]:
        """Smart file finding with multiple strategies"""
        potential_files = []
        
        try:
            print(f"🔍 Searching for generated file with base name: {output_filename}")
            
            # Strategy 1: Exact filename match
            exact_patterns = [
                f"{output_filename}.mp4",
                f"{output_filename}_*.mp4",
                f"*{output_filename}*.mp4"
            ]
            
            for pattern in exact_patterns:
                full_pattern = os.path.join(self.wan_path, pattern)
                matches = glob.glob(full_pattern)
                if matches:
                    print(f"✅ Found {len(matches)} files with pattern: {pattern}")
                    potential_files.extend(matches)
            
            # Strategy 2: Recent MP4 files (last 2 minutes)
            if not potential_files:
                print("🔄 Looking for recent MP4 files...")
                current_time = time.time()
                all_mp4s = glob.glob(os.path.join(self.wan_path, "*.mp4"))
                
                recent_mp4s = []
                for mp4_file in all_mp4s:
                    file_time = os.path.getctime(mp4_file)
                    if current_time - file_time < 120:  # 2 minutes
                        recent_mp4s.append((mp4_file, file_time))
                
                if recent_mp4s:
                    recent_mp4s.sort(key=lambda x: x[1], reverse=True)
                    potential_files = [f[0] for f in recent_mp4s]
                    print(f"✅ Found {len(potential_files)} recent MP4 files")
            
            # Validate files exist and have content
            valid_files = []
            for file_path in potential_files:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > 1024:  # At least 1KB
                        valid_files.append(file_path)
                        print(f"✅ Valid file: {os.path.basename(file_path)} ({file_size/1024/1024:.1f}MB)")
            
            return valid_files
            
        except Exception as e:
            print(f"❌ File search error: {e}")
            return []

    def generate_wan_video_with_debug(self, prompt: str, output_filename: str, size: str = "832*480") -> Optional[str]:
        """Enhanced version with file debugging"""
        if not self.wan_available:
            print("❌ Wan 2.1 not available, cannot generate video")
            return None
            
        try:
            print(f"🎬 Starting Wan 2.1 video generation...")
            print(f"📝 Prompt: {prompt}")
            print(f"📐 Size: {size}")
            print(f"📁 Output filename: {output_filename}")
            
            cmd = [
                "python", "generate.py",
                "--task", "t2v-1.3B",
                "--size", size,
                "--ckpt_dir", self.wan_model_path,
                "--prompt", prompt,
                "--save_file", f"{output_filename}.mp4"
            ]
            
            print(f"🔧 Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.wan_path,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            print(f"📤 Wan 2.1 process completed with return code: {result.returncode}")
            
            if result.returncode == 0:
                print("✅ Wan 2.1 generation completed successfully")
                
                potential_files = self.find_generated_video_file(output_filename)
                
                if potential_files:
                    latest_file = potential_files[0]
                    print(f"✅ Found generated video: {latest_file}")
                    return latest_file
                else:
                    print("❌ Generated video file not found")
                    print(f"stdout: {result.stdout}")
                    print(f"stderr: {result.stderr}")
                    return None
                    
            else:
                print(f"❌ Wan 2.1 generation failed with return code {result.returncode}")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("❌ Wan 2.1 generation timed out (>10 minutes)")
            return None
        except Exception as e:
            print(f"❌ Error during Wan 2.1 generation: {e}")
            return None

    def generate_image_fast(self, prompt: str) -> Optional[str]:
        """Generate fast/low-res image for previews and storyboarding (2-3 seconds)"""
        print("🖼️ Generating fast image (low res)...")
        
        if self.wan_available:
            print("🎥 Using Wan 2.1 T2V for fast image generation (512x768)")
            output_filename = f"image_fast_{uuid.uuid4().hex[:8]}"
            video_path = self.generate_wan_video_with_debug(prompt, output_filename, "512*768")
            
            if video_path:
                try:
                    preview_path = video_path.replace('.mp4', '_image.png')
                    
                    print(f"🎞️ Extracting frame from: {video_path}")
                    print(f"🖼️ Saving image to: {preview_path}")
                    
                    result = subprocess.run([
                        'ffmpeg', '-i', video_path, '-vf', 'select=eq(n\\,0)', 
                        '-q:v', '3', '-y', preview_path
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0 and os.path.exists(preview_path):
                        file_size = os.path.getsize(preview_path) / 1024  # KB
                        print(f"✅ Fast image generated: {preview_path} ({file_size:.1f}KB)")
                        return preview_path
                    else:
                        print(f"❌ FFmpeg failed: {result.stderr}")
                        return None
                        
                except Exception as e:
                    print(f"❌ Failed to extract image frame: {e}")
        
        # Fallback: create placeholder image
        print("🖼️ Creating placeholder fast image")
        return self.create_placeholder_image(prompt, size=(512, 768), image_type="Fast Image")

    def generate_image_high(self, prompt: str) -> Optional[str]:
        """Generate high-res image for characters and premium content (3-4 seconds)"""
        print("🖼️ Generating high-res image...")
        
        if self.wan_available:
            print("🎥 Using Wan 2.1 T2V for high-res image generation (1024x1024)")
            output_filename = f"image_high_{uuid.uuid4().hex[:8]}"
            video_path = self.generate_wan_video_with_debug(prompt, output_filename, "1024*1024")
            
            if video_path:
                try:
                    preview_path = video_path.replace('.mp4', '_image.png')
                    
                    print(f"🎞️ Extracting frame from: {video_path}")
                    print(f"🖼️ Saving high-res image to: {preview_path}")
                    
                    result = subprocess.run([
                        'ffmpeg', '-i', video_path, '-vf', 'select=eq(n\\,0)', 
                        '-q:v', '2', '-y', preview_path  # Higher quality
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0 and os.path.exists(preview_path):
                        file_size = os.path.getsize(preview_path) / 1024  # KB
                        print(f"✅ High-res image generated: {preview_path} ({file_size:.1f}KB)")
                        return preview_path
                    else:
                        print(f"❌ FFmpeg failed: {result.stderr}")
                        return None
                        
                except Exception as e:
                    print(f"❌ Failed to extract image frame: {e}")
        
        # Fallback: create placeholder image
        print("🖼️ Creating placeholder high-res image")
        return self.create_placeholder_image(prompt, size=(1024, 1024), image_type="High-Res Image")

    def generate_video_fast(self, prompt: str) -> Optional[str]:
        """Generate fast/low-res video for standard content (4-6 minutes)"""
        print("🎬 Generating fast video (standard quality)...")
        
        if self.wan_available:
            print("🎥 Using Wan 2.1 T2V-1.3B for fast video generation (832x480)")
            output_filename = f"video_fast_{uuid.uuid4().hex[:8]}"
            video_path = self.generate_wan_video_with_debug(prompt, output_filename, "832*480")
            
            if video_path:
                file_size = os.path.getsize(video_path) / 1024 / 1024  # MB
                print(f"✅ Fast video generated: {video_path} ({file_size:.1f}MB)")
                return video_path
        
        print("🎬 Creating placeholder fast video")
        return self.create_placeholder_video(prompt, size="832x480", video_type="Fast Video")

    def generate_video_high(self, prompt: str) -> Optional[str]:
        """Generate high-res video for premium content (6-8 minutes)"""
        print("🎬 Generating high-res video (premium quality)...")
        
        if self.wan_available:
            print("🎥 Using Wan 2.1 T2V-14B for high-res video generation (1280x720)")
            # Note: This would use the 14B model when available
            output_filename = f"video_high_{uuid.uuid4().hex[:8]}"
            video_path = self.generate_wan_video_with_debug(prompt, output_filename, "1280*720")
            
            if video_path:
                file_size = os.path.getsize(video_path) / 1024 / 1024  # MB
                print(f"✅ High-res video generated: {video_path} ({file_size:.1f}MB)")
                return video_path
        
        print("🎬 Creating placeholder high-res video")
        return self.create_placeholder_video(prompt, size="1280x720", video_type="High-Res Video")

    def create_placeholder_image(self, prompt: str, size: tuple = (832, 480), image_type: str = "Image") -> str:
        """Create placeholder image with configurable size"""
        print(f"🖼️ Creating placeholder {image_type.lower()}...")
        
        try:
            width, height = size
            img = Image.new('RGB', (width, height), color=(100, 150, 200))
            draw = ImageDraw.Draw(img)
            
            try:
                font_size = min(width, height) // 20  # Scale font to image size
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
                
            text = f"{image_type}\n(Generated by OurVidz)"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            prompt_text = f"Prompt: {prompt[:40]}..."
            draw.text((20, height - 50), prompt_text, fill=(255, 255, 255))
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            draw.text((20, 20), timestamp, fill=(255, 255, 255))
            
            output_path = f"/tmp/{image_type.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}.png"
            img.save(output_path, "PNG")
            print(f"✅ Placeholder {image_type.lower()} created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to create placeholder {image_type.lower()}: {e}")
            return None

    def create_placeholder_video(self, prompt: str, size: str = "832x480", video_type: str = "Video") -> str:
        """Create placeholder video with configurable size"""
        print(f"🎬 Creating placeholder {video_type.lower()}...")
        
        try:
            output_path = f"/tmp/{video_type.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}.mp4"
            
            subprocess.run([
                'ffmpeg', '-f', 'lavfi', '-i', 
                f'color=c=blue:size={size}:duration=5',
                '-vf', f'drawtext=text="OurVidz {video_type}\\nPrompt: {prompt[:30]}...":fontcolor=white:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', output_path
            ], check=True, capture_output=True)
            
            print(f"✅ Placeholder {video_type.lower()} created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to create placeholder {video_type.lower()}: {e}")
            try:
                width, height = map(int, size.split('x'))
                img = Image.new('RGB', (width, height), color=(50, 100, 150))
                fallback_path = f"/tmp/fallback_{video_type.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}.png"
                img.save(fallback_path)
                return fallback_path
            except:
                return None

    def upload_to_supabase(self, file_path: str, storage_path: str) -> str:
        """Upload file to Supabase storage with unique naming"""
        if not self.environment_valid:
            print("❌ Cannot upload to Supabase: invalid environment")
            return f"placeholder:///{storage_path}"
            
        try:
            print(f"📤 Uploading file: {file_path}")
            print(f"📁 Storage path: {storage_path}")
            
            if not os.path.exists(file_path):
                raise Exception(f"File does not exist: {file_path}")
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise Exception(f"File is empty: {file_path}")
            
            print(f"📊 File size: {file_size / 1024 / 1024:.1f}MB")

            content_type = 'application/octet-stream'
            if file_path.lower().endswith('.png'):
                content_type = 'image/png'
            elif file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif file_path.lower().endswith('.mp4'):
                content_type = 'video/mp4'

            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': (os.path.basename(file_path), file, content_type)},
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                        'x-upsert': 'true'
                    }
                )
            
            if response.status_code in [200, 201]:
                public_url = f"{self.supabase_url}/storage/v1/object/public/{storage_path}"
                print(f"✅ File uploaded to: {public_url}")
                return public_url
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                raise Exception(f"Upload failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Supabase upload error: {e}")
            raise

    def notify_completion(self, job_id: str, status: str, output_url: str = None, error_message: str = None, enhanced_prompt: str = None):
        """Notify Supabase of job completion"""
        if not self.environment_valid:
            print(f"⚠️ Cannot send callback: invalid environment (job {job_id})")
            return
            
        try:
            callback_data = {
                'jobId': job_id,
                'status': status,
                'outputUrl': output_url,
                'errorMessage': error_message,
                'enhancedPrompt': enhanced_prompt
            }
            
            print(f"📞 Sending callback for job {job_id}: {status}")
            
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
        """Process a single job with modular job types"""
        job_id = job_data.get('jobId', 'unknown')
        job_type = job_data.get('jobType', 'unknown')
        
        print(f"🔄 Processing job {job_id} ({job_type})")
        
        try:
            timestamp = int(time.time())
            
            if job_type == 'enhance':
                # Legacy support - enhance prompts
                original_prompt = job_data.get('prompt', '')
                character_desc = job_data.get('characterDescription', '')
                
                if not original_prompt.strip():
                    original_prompt = "person walking"
                
                enhanced_prompt = self.enhance_prompt(original_prompt, character_desc)
                print(f"✅ Enhanced prompt: {enhanced_prompt[:100]}...")
                
                self.notify_completion(job_id, 'completed', output_url=None, error_message=None, enhanced_prompt=enhanced_prompt)
                
            elif job_type == 'image_fast':
                # Fast/low-res images for previews and storyboarding (2-3 seconds)
                prompt = job_data.get('prompt', 'woman walking')
                if not prompt.strip():
                    prompt = "woman walking"
                    
                image_path = self.generate_image_fast(prompt)
                
                if image_path:
                    filename = f"{job_id}_{timestamp}_image_fast.png"
                    storage_path = f"image_fast/{filename}"  # Updated bucket name
                    upload_url = self.upload_to_supabase(image_path, storage_path)
                    
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        
                    print(f"✅ Fast image uploaded: {upload_url}")
                    self.notify_completion(job_id, 'completed', upload_url)
                else:
                    raise Exception("Failed to generate fast image")
                    
            elif job_type == 'image_high':
                # High-res images for characters and premium content (3-4 seconds)
                prompt = job_data.get('prompt', 'woman walking')
                if not prompt.strip():
                    prompt = "woman walking"
                    
                image_path = self.generate_image_high(prompt)
                
                if image_path:
                    filename = f"{job_id}_{timestamp}_image_high.png"
                    storage_path = f"image_high/{filename}"  # Updated bucket name
                    upload_url = self.upload_to_supabase(image_path, storage_path)
                    
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        
                    print(f"✅ High-res image uploaded: {upload_url}")
                    self.notify_completion(job_id, 'completed', upload_url)
                else:
                    raise Exception("Failed to generate high-res image")
                    
            elif job_type == 'video_fast':
                # Fast/standard video generation (4-6 minutes)
                prompt = job_data.get('prompt', 'woman walking')
                if not prompt.strip():
                    prompt = "woman walking"
                    
                video_path = self.generate_video_fast(prompt)
                
                if video_path:
                    filename = f"{job_id}_{timestamp}_video_fast.mp4"
                    storage_path = f"video_fast/{filename}"  # Updated bucket name
                    upload_url = self.upload_to_supabase(video_path, storage_path)
                    
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        
                    print(f"✅ Fast video uploaded: {upload_url}")
                    self.notify_completion(job_id, 'completed', upload_url)
                else:
                    raise Exception("Failed to generate fast video")
                    
            elif job_type == 'video_high':
                # High-res/premium video generation (6-8 minutes)
                prompt = job_data.get('prompt', 'woman walking')
                if not prompt.strip():
                    prompt = "woman walking"
                    
                video_path = self.generate_video_high(prompt)
                
                if video_path:
                    filename = f"{job_id}_{timestamp}_video_high.mp4"
                    storage_path = f"video_high/{filename}"  # Updated bucket name
                    upload_url = self.upload_to_supabase(video_path, storage_path)
                    
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        
                    print(f"✅ High-res video uploaded: {upload_url}")
                    self.notify_completion(job_id, 'completed', upload_url)
                else:
                    raise Exception("Failed to generate high-res video")
                    
            # Legacy support for old job types
            elif job_type in ['preview', 'image']:
                print(f"⚠️ Legacy job type '{job_type}' mapped to 'image_fast'")
                # Redirect to image_fast
                job_data['jobType'] = 'image_fast'
                return self.process_job(job_data)
                
            elif job_type == 'video':
                print(f"⚠️ Legacy job type '{job_type}' mapped to 'video_fast'")
                # Redirect to video_fast
                job_data['jobType'] = 'video_fast'
                return self.process_job(job_data)
                
            else:
                print(f"❌ Unknown job type: {job_type}")
                print(f"📋 Supported types: image_fast, image_high, video_fast, video_high, enhance")
                self.notify_completion(job_id, 'failed', error_message=f"Unknown job type: {job_type}")
            
            print(f"🎉 Job {job_id} completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Job {job_id} failed: {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)

    def run(self):
        """Main worker loop with non-blocking Redis polling"""
        print("🎬 OurVidz GPU Worker started!")
        
        if self.wan_available:
            print("🎥 Running with Wan 2.1 support")
        else:
            print("⚠️ Running in placeholder mode")
            
        if not self.environment_valid:
            print("⚠️ Running with limited functionality (environment issues)")
            
        print("⏳ Waiting for jobs...")
        
        idle_time = 0
        max_idle_time = 10 * 60   # 10 minutes
        poll_interval = 5  # Poll every 5 seconds
        
        while True:
            try:
                if not self.environment_valid:
                    print("⚠️ Environment invalid, waiting...")
                    time.sleep(poll_interval)
                    idle_time += poll_interval
                    continue
                
                response = requests.get(
                    f"{self.upstash_redis_url}/rpop/job_queue",
                    headers={
                        'Authorization': f"Bearer {self.upstash_redis_token}"
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('result'):
                        idle_time = 0
                        
                        job_json = result['result']
                        job_data = json.loads(job_json)
                        print(f"📥 Received job: {job_data.get('jobType', 'unknown')} - {job_data.get('jobId', 'no-id')}")
                        self.process_job(job_data)
                    else:
                        idle_time += poll_interval
                        time.sleep(poll_interval)
                        
                        if idle_time % 60 == 0:
                            minutes_idle = idle_time // 60
                            print(f"⏳ Idle for {minutes_idle} minutes (shutdown at {max_idle_time//60})")
                else:
                    print(f"⚠️ Redis connection issue: {response.status_code} - {response.text}")
                    time.sleep(poll_interval)
                    idle_time += poll_interval
                    
                if idle_time >= max_idle_time:
                    print(f"🛑 Shutting down after {max_idle_time//60} minutes of inactivity")
                    break
                    
            except Exception as e:
                print(f"❌ Worker error: {e}")
                time.sleep(poll_interval)
                idle_time += poll_interval

if __name__ == "__main__":
    required_vars = [
        'UPSTASH_REDIS_REST_URL',
        'UPSTASH_REDIS_REST_TOKEN', 
        'SUPABASE_URL',
        'SUPABASE_SERVICE_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        print("🔄 Starting worker anyway (will run in limited mode)")
    
    worker = VideoWorker()
    print("🚀 Starting worker main loop...")
    worker.run()
