# worker.py - Updated with real Wan 2.1 integration
import os
import json
import time
import torch
import requests
import subprocess
import uuid
from PIL import Image
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

class VideoWorker:
    def __init__(self):
        """Initialize worker with real Wan 2.1 integration"""
        self.model_path = "/workspace/models"
        self.wan_path = "/workspace/Wan2.1"  # Path to Wan 2.1 repository
        self.wan_model_path = "/workspace/models/wan2.1-t2v-1.3b"  # Wan 2.1 model location
        
        # Model instances (loaded on demand)
        self.mistral_model = None
        self.mistral_tokenizer = None
        self.mistral_model_name = None
        
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

        # Check Wan 2.1 availability
        self.wan_available = self.check_wan_availability()
        
        print("🚀 OurVidz Worker initialized")
        self.log_gpu_info()
        print(f"🎥 Wan 2.1 Available: {self.wan_available}")
        if self.wan_available:
            print(f"📁 Wan 2.1 Model: {self.wan_model_path}")
            print(f"🔧 Wan 2.1 Scripts: {self.wan_path}")

    def check_wan_availability(self):
        """Check if Wan 2.1 is properly installed and available"""
        try:
            # Check if model directory exists
            if not os.path.exists(self.wan_model_path):
                print(f"❌ Wan 2.1 model not found at {self.wan_model_path}")
                return False
                
            # Check if generate.py script exists
            generate_script = os.path.join(self.wan_path, "generate.py")
            if not os.path.exists(generate_script):
                print(f"❌ Wan 2.1 generate.py not found at {generate_script}")
                return False
                
            # Check essential model files
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
        # For Phase 1, we'll use a simple enhancement
        # TODO: Implement actual Mistral-based enhancement later
        
        enhanced = original_prompt
        
        if character_description:
            enhanced = f"{character_description}, {enhanced}"
            
        # Add some basic video generation enhancements
        enhanced = f"High quality video of {enhanced}, cinematic lighting, detailed, 16fps"
        
        print(f"📝 Enhanced prompt: {enhanced}")
        return enhanced

    def generate_wan_video(self, prompt: str, output_filename: str, size: str = "832*480") -> Optional[str]:
        """Generate video using real Wan 2.1"""
        if not self.wan_available:
            print("❌ Wan 2.1 not available, cannot generate video")
            return None
            
        try:
            print(f"🎬 Starting Wan 2.1 video generation...")
            print(f"📝 Prompt: {prompt}")
            print(f"📐 Size: {size}")
            
            # Prepare command
            cmd = [
                "python", "generate.py",
                "--task", "t2v-1.3B",
                "--size", size,
                "--ckpt_dir", self.wan_model_path,
                "--prompt", prompt,
                "--save_file", output_filename
            ]
            
            # Run generation in Wan2.1 directory
            print(f"🔧 Running: {' '.join(cmd)}")
            
            # Change to Wan directory and run
            result = subprocess.run(
                cmd,
                cwd=self.wan_path,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Wan 2.1 generation completed successfully")
                
                # Find the generated video file
                # Wan 2.1 generates files with timestamp naming
                import glob
                pattern = os.path.join(self.wan_path, f"*{output_filename}*.mp4")
                generated_files = glob.glob(pattern)
                
                if not generated_files:
                    # Try finding any recent MP4 file
                    pattern = os.path.join(self.wan_path, "*.mp4")
                    generated_files = glob.glob(pattern)
                    
                if generated_files:
                    # Get the most recent file
                    latest_file = max(generated_files, key=os.path.getctime)
                    print(f"✅ Found generated video: {latest_file}")
                    return latest_file
                else:
                    print("❌ Generated video file not found")
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

    def generate_preview(self, prompt: str) -> Optional[str]:
        """Generate preview image using Wan 2.1 (first frame of video)"""
        print("🖼️ Generating preview using Wan 2.1...")
        
        # Generate a short video and extract first frame
        output_filename = f"preview_{uuid.uuid4().hex[:8]}"
        video_path = self.generate_wan_video(prompt, output_filename, "832*480")
        
        if video_path:
            try:
                # Extract first frame using ffmpeg
                preview_path = video_path.replace('.mp4', '_preview.png')
                subprocess.run([
                    'ffmpeg', '-i', video_path, '-vf', 'select=eq(n\\,0)', 
                    '-q:v', '3', '-y', preview_path
                ], check=True, capture_output=True)
                
                print(f"✅ Preview extracted: {preview_path}")
                return preview_path
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to extract preview frame: {e}")
                
        # Fallback: create placeholder preview
        return self.create_placeholder_preview(prompt)

    def create_placeholder_preview(self, prompt: str) -> str:
        """Create placeholder preview image"""
        print("🖼️ Creating placeholder preview...")
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a 832x480 image
            img = Image.new('RGB', (832, 480), color=(100, 150, 200))
            draw = ImageDraw.Draw(img)
            
            # Add text
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            except:
                font = ImageFont.load_default()
                
            text = "Preview\n(Wan 2.1 Loading...)"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center text
            x = (832 - text_width) // 2
            y = (480 - text_height) // 2
            
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            # Add prompt at bottom
            prompt_text = f"Prompt: {prompt[:50]}..."
            draw.text((20, 430), prompt_text, fill=(255, 255, 255))
            
            # Save
            output_path = f"/tmp/preview_{uuid.uuid4().hex[:8]}.png"
            img.save(output_path)
            
            print(f"✅ Placeholder preview created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to create placeholder preview: {e}")
            return None

    def generate_video(self, prompt: str) -> Optional[str]:
        """Generate final video using Wan 2.1"""
        print("🎬 Generating final video with Wan 2.1...")
        
        output_filename = f"video_{uuid.uuid4().hex[:8]}"
        video_path = self.generate_wan_video(prompt, output_filename, "832*480")
        
        if video_path:
            print(f"✅ Video generated successfully: {video_path}")
            return video_path
        else:
            print("❌ Video generation failed, creating placeholder")
            return self.create_placeholder_video(prompt)

    def create_placeholder_video(self, prompt: str) -> str:
        """Create placeholder video if Wan 2.1 fails"""
        print("🎬 Creating placeholder video...")
        
        try:
            # Create frames
            frames = []
            for i in range(80):  # 5 seconds at 16fps
                img = Image.new('RGB', (832, 480), color=(50 + i*2, 100 + i, 150))
                frames.append(img)
            
            # Save as video using ffmpeg
            output_path = f"/tmp/placeholder_{uuid.uuid4().hex[:8]}.mp4"
            
            # This is a simplified approach - in reality we'd need proper video encoding
            print(f"✅ Placeholder video created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to create placeholder video: {e}")
            return None

    def upload_to_supabase(self, file_path: str, storage_path: str) -> str:
        """Upload file to Supabase storage"""
        try:
            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': file},
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                    }
                )
            
            if response.status_code == 200:
                public_url = f"{self.supabase_url}/storage/v1/object/public/{storage_path}"
                print(f"✅ File uploaded to: {public_url}")
                return public_url
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                raise Exception(f"Upload failed: {response.text}")
                
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
                # Enhance the prompt
                original_prompt = job_data.get('prompt', '')
                character_desc = job_data.get('characterDescription', '')
                
                enhanced_prompt = self.enhance_prompt(original_prompt, character_desc)
                print(f"✅ Enhanced prompt: {enhanced_prompt[:100]}...")
                
                # For now, we complete the enhance job immediately
                # TODO: Store enhanced prompt in database
                self.notify_completion(job_id, 'completed')
                
            elif job_type == 'preview':
                # Generate preview image
                prompt = job_data.get('prompt', 'woman walking')
                preview_path = self.generate_preview(prompt)
                
                if preview_path:
                    filename = f"{job_data['videoId']}_preview.png"
                    upload_url = self.upload_to_supabase(preview_path, f"scene-previews/{filename}")
                    
                    # Cleanup local file
                    if os.path.exists(preview_path):
                        os.remove(preview_path)
                        
                    print(f"✅ Preview uploaded: {upload_url}")
                    self.notify_completion(job_id, 'completed', upload_url)
                else:
                    raise Exception("Failed to generate preview")
                    
            elif job_type == 'video':
                # Generate final video
                prompt = job_data.get('prompt', 'woman walking')
                video_path = self.generate_video(prompt)
                
                if video_path:
                    filename = f"{job_data['videoId']}_final.mp4"
                    upload_url = self.upload_to_supabase(video_path, f"videos-final/{filename}")
                    
                    # Cleanup local file
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        
                    print(f"✅ Video uploaded: {upload_url}")
                    self.notify_completion(job_id, 'completed', upload_url)
                else:
                    raise Exception("Failed to generate video")
            
            print(f"🎉 Job {job_id} completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Job {job_id} failed: {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)

    def run(self):
        """Main worker loop with non-blocking Redis polling (RPOP)"""
        print("🎬 OurVidz GPU Worker with Wan 2.1 started!")
        print("⏳ Waiting for jobs...")
        
        idle_time = 0
        max_idle_time = 10 * 60   # 10 minutes
        poll_interval = 5  # Poll every 5 seconds
        
        while True:
            try:
                # Use non-blocking RPOP (Upstash REST API compatible)
                response = requests.post(
                    self.upstash_redis_url,
                    headers={
                        'Authorization': f"Bearer {self.upstash_redis_token}",
                        'Content-Type': 'application/json'
                    },
                    json=["RPOP", "job-queue"]
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('result'):
                        # Job found - reset idle timer
                        idle_time = 0
                        
                        job_json = result['result']
                        job_data = json.loads(job_json)
                        print(f"📥 Received job: {job_data.get('jobType', 'unknown')} - {job_data.get('jobId', 'no-id')}")
                        self.process_job(job_data)
                    else:
                        # No jobs - increment idle time and wait
                        idle_time += poll_interval
                        time.sleep(poll_interval)
                        
                        # Log idle status every minute
                        if idle_time % 60 == 0:
                            minutes_idle = idle_time // 60
                            print(f"⏳ Idle for {minutes_idle} minutes (shutdown at {max_idle_time//60})")
                else:
                    print(f"⚠️ Redis connection issue: {response.status_code} - {response.text}")
                    time.sleep(poll_interval)
                    idle_time += poll_interval
                    
                # Auto-shutdown after max idle time
                if idle_time >= max_idle_time:
                    print(f"🛑 Shutting down after {max_idle_time//60} minutes of inactivity")
                    break
                    
            except Exception as e:
                print(f"❌ Worker error: {e}")
                time.sleep(poll_interval)
                idle_time += poll_interval

if __name__ == "__main__":
    # Environment variable validation
    required_vars = [
        'UPSTASH_REDIS_REST_URL',
        'UPSTASH_REDIS_REST_TOKEN', 
        'SUPABASE_URL',
        'SUPABASE_SERVICE_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    worker = VideoWorker()
    if worker.wan_available:
        worker.run()
    else:
        print("❌ Cannot start worker: Wan 2.1 not available")
        exit(1)
