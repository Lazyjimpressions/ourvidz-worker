# worker.py - Complete Updated Version with Fixed Filenames and All Legacy Code
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
            # Don't exit - continue with limited functionality
            self.environment_valid = False
        else:
            self.environment_valid = True

        # Check Wan 2.1 availability
        self.wan_available = self.check_wan_availability()
        
        print("🚀 OurVidz Worker initialized")
        self.log_gpu_info()
        print(f"🎥 Wan 2.1 Available: {self.wan_available}")
        print(f"🔧 Environment Valid: {self.environment_valid}")
        
        if self.wan_available:
            print(f"📁 Wan 2.1 Model: {self.wan_model_path}")
            print(f"🔧 Wan 2.1 Scripts: {self.wan_path}")
        else:
            print("⚠️ Running in placeholder mode - will create test content")

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

    def debug_generated_files(self, output_filename: str, before_generation: bool = True):
        """Debug what files exist before and after generation"""
        timestamp = "BEFORE" if before_generation else "AFTER"
        print(f"🔍 {timestamp} GENERATION - File Debug:")
        
        # List all files in Wan2.1 directory
        try:
            all_files = os.listdir(self.wan_path)
            mp4_files = [f for f in all_files if f.endswith('.mp4')]
            
            print(f"📁 Wan2.1 directory: {self.wan_path}")
            print(f"📊 Total files: {len(all_files)}")
            print(f"🎬 MP4 files: {len(mp4_files)}")
            
            if mp4_files:
                print("🎥 MP4 files found:")
                for mp4 in mp4_files[:10]:  # Show first 10
                    file_path = os.path.join(self.wan_path, mp4)
                    file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                    file_time = os.path.getctime(file_path)
                    print(f"   📄 {mp4} ({file_size:.1f}MB, created: {time.ctime(file_time)})")
            else:
                print("❌ No MP4 files found")
                
            # Show recent files (last 5 minutes)
            recent_files = []
            current_time = time.time()
            for file in all_files:
                file_path = os.path.join(self.wan_path, file)
                if os.path.isfile(file_path):
                    file_time = os.path.getctime(file_path)
                    if current_time - file_time < 300:  # 5 minutes
                        recent_files.append((file, file_time))
            
            if recent_files:
                print(f"⏰ Recent files (last 5 min): {len(recent_files)}")
                for file, file_time in recent_files[:5]:
                    print(f"   📄 {file} (created: {time.ctime(file_time)})")
                    
        except Exception as e:
            print(f"❌ Debug error: {e}")

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
                    print(f"✅ Strategy 1 found {len(matches)} files with pattern: {pattern}")
                    potential_files.extend(matches)
            
            # Strategy 2: Recent MP4 files (last 2 minutes)
            if not potential_files:
                print("🔄 Strategy 2: Looking for recent MP4 files...")
                current_time = time.time()
                all_mp4s = glob.glob(os.path.join(self.wan_path, "*.mp4"))
                
                recent_mp4s = []
                for mp4_file in all_mp4s:
                    file_time = os.path.getctime(mp4_file)
                    if current_time - file_time < 120:  # 2 minutes
                        recent_mp4s.append((mp4_file, file_time))
                
                if recent_mp4s:
                    # Sort by creation time (newest first)
                    recent_mp4s.sort(key=lambda x: x[1], reverse=True)
                    potential_files = [f[0] for f in recent_mp4s]
                    print(f"✅ Strategy 2 found {len(potential_files)} recent MP4 files")
            
            # Strategy 3: Any MP4 file (last resort)
            if not potential_files:
                print("🔄 Strategy 3: Looking for any MP4 files...")
                all_mp4s = glob.glob(os.path.join(self.wan_path, "*.mp4"))
                if all_mp4s:
                    # Sort by modification time (newest first)
                    all_mp4s.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    potential_files = all_mp4s[:3]  # Take 3 most recent
                    print(f"✅ Strategy 3 found {len(potential_files)} MP4 files")
            
            # Validate files exist and have content
            valid_files = []
            for file_path in potential_files:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > 1024:  # At least 1KB
                        valid_files.append(file_path)
                        print(f"✅ Valid file: {os.path.basename(file_path)} ({file_size/1024/1024:.1f}MB)")
                    else:
                        print(f"⚠️ File too small: {os.path.basename(file_path)} ({file_size}B)")
                else:
                    print(f"❌ File not found: {file_path}")
            
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
            print(f"🎬 Starting Wan 2.1 video generation with debug...")
            print(f"📝 Prompt: {prompt}")
            print(f"📐 Size: {size}")
            print(f"📁 Output filename: {output_filename}")
            
            # Debug BEFORE generation
            self.debug_generated_files(output_filename, before_generation=True)
            
            # Prepare command
            cmd = [
                "python", "generate.py",
                "--task", "t2v-1.3B",
                "--size", size,
                "--ckpt_dir", self.wan_model_path,
                "--prompt", prompt,
                "--save_file", f"{output_filename}.mp4"
            ]
            
            # Run generation in Wan2.1 directory
            print(f"🔧 Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.wan_path,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            print(f"📤 Wan 2.1 process completed with return code: {result.returncode}")
            
            # Debug AFTER generation
            self.debug_generated_files(output_filename, before_generation=False)
            
            if result.returncode == 0:
                print("✅ Wan 2.1 generation completed successfully")
                
                # Enhanced file finding logic
                potential_files = self.find_generated_video_file(output_filename)
                
                if potential_files:
                    latest_file = potential_files[0]  # Already sorted by creation time
                    print(f"✅ Found generated video: {latest_file}")
                    return latest_file
                else:
                    print("❌ Generated video file not found despite successful generation")
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

    def generate_preview(self, prompt: str) -> Optional[str]:
        """Generate preview image using Wan 2.1 with enhanced debugging"""
        print("🖼️ Generating preview...")
        
        if self.wan_available:
            print("🎥 Using Wan 2.1 for preview generation")
            output_filename = f"preview_{uuid.uuid4().hex[:8]}"
            video_path = self.generate_wan_video_with_debug(prompt, output_filename, "832*480")
            
            if video_path:
                try:
                    # Extract first frame using ffmpeg
                    preview_path = video_path.replace('.mp4', '_preview.png')
                    
                    print(f"🎞️ Extracting frame from: {video_path}")
                    print(f"🖼️ Saving preview to: {preview_path}")
                    
                    result = subprocess.run([
                        'ffmpeg', '-i', video_path, '-vf', 'select=eq(n\\,0)', 
                        '-q:v', '3', '-y', preview_path
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0 and os.path.exists(preview_path):
                        file_size = os.path.getsize(preview_path) / 1024  # KB
                        print(f"✅ Preview extracted successfully: {preview_path} ({file_size:.1f}KB)")
                        return preview_path
                    else:
                        print(f"❌ FFmpeg failed: {result.stderr}")
                        return None
                        
                except Exception as e:
                    print(f"❌ Failed to extract preview frame: {e}")
        
        # Fallback: create placeholder preview
        print("🖼️ Creating placeholder preview")
        return self.create_placeholder_preview(prompt)

    def create_placeholder_preview(self, prompt: str) -> str:
        """Create placeholder preview image"""
        print("🖼️ Creating placeholder preview...")
        
        try:
            # Create a 832x480 image
            img = Image.new('RGB', (832, 480), color=(100, 150, 200))
            draw = ImageDraw.Draw(img)
            
            # Add text
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            except:
                font = ImageFont.load_default()
                
            text = "Preview\n(Generated by OurVidz)"
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
            
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            draw.text((20, 20), timestamp, fill=(255, 255, 255))
            
            # Save
            output_path = f"/tmp/preview_{uuid.uuid4().hex[:8]}.png"
            img.save(output_path, "PNG")
            print(f"📊 Placeholder file size: {os.path.getsize(output_path)} bytes")
            
            print(f"✅ Placeholder preview created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to create placeholder preview: {e}")
            return None

    def generate_video(self, prompt: str) -> Optional[str]:
        """Generate final video using Wan 2.1 with enhanced debugging"""
        print("🎬 Generating final video...")
        
        if self.wan_available:
            print("🎥 Using Wan 2.1 for video generation")
            output_filename = f"video_{uuid.uuid4().hex[:8]}"
            video_path = self.generate_wan_video_with_debug(prompt, output_filename, "832*480")
            
            if video_path:
                file_size = os.path.getsize(video_path) / 1024 / 1024  # MB
                print(f"✅ Video generated successfully: {video_path} ({file_size:.1f}MB)")
                return video_path
        
        # Fallback: create placeholder video
        print("🎬 Creating placeholder video")
        return self.create_placeholder_video(prompt)

    def create_placeholder_video(self, prompt: str) -> str:
        """Create placeholder video if Wan 2.1 fails"""
        print("🎬 Creating placeholder video...")
        
        try:
            # Create a simple MP4 using ffmpeg
            output_path = f"/tmp/placeholder_{uuid.uuid4().hex[:8]}.mp4"
            
            # Create 5-second placeholder video with text
            subprocess.run([
                'ffmpeg', '-f', 'lavfi', '-i', 
                f'color=c=blue:size=832x480:duration=5',
                '-vf', f'drawtext=text="OurVidz Placeholder\\nPrompt: {prompt[:30]}...":fontcolor=white:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', output_path
            ], check=True, capture_output=True)
            
            print(f"✅ Placeholder video created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to create placeholder video: {e}")
            # Create a minimal static image as fallback
            try:
                img = Image.new('RGB', (832, 480), color=(50, 100, 150))
                fallback_path = f"/tmp/fallback_{uuid.uuid4().hex[:8]}.png"
                img.save(fallback_path)
                return fallback_path
            except:
                return None

    def upload_to_supabase(self, file_path: str, storage_path: str) -> str:
        """Upload file to Supabase storage with unique naming and upsert support"""
        if not self.environment_valid:
            print("❌ Cannot upload to Supabase: invalid environment")
            return f"placeholder:///{storage_path}"
            
        try:
            print(f"📤 Uploading file: {file_path}")
            print(f"📁 Storage path: {storage_path}")
            
            # Verify file exists and has content
            if not os.path.exists(file_path):
                raise Exception(f"File does not exist: {file_path}")
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise Exception(f"File is empty: {file_path}")
            
            print(f"📊 File size: {file_size / 1024 / 1024:.1f}MB")

            # Determine content type based on file extension
            content_type = 'application/octet-stream'  # Default
            if file_path.lower().endswith('.png'):
                content_type = 'image/png'
            elif file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif file_path.lower().endswith('.mp4'):
                content_type = 'video/mp4'
            
            print(f"📋 Content-Type: {content_type}")

            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': (os.path.basename(file_path), file, content_type)},
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                        'upsert': 'true'  # Allow overwriting existing files
                    }
                )
            
            if response.status_code in [200, 201]:  # Accept both success codes
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
            if enhanced_prompt:
                print(f"📝 Enhanced prompt in callback: {enhanced_prompt[:100]}...")
            
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
        """Process a single job with unique filename generation"""
        job_id = job_data.get('jobId', 'unknown')
        job_type = job_data.get('jobType', 'unknown')
        video_id = job_data.get('videoId', 'unknown')
        
        print(f"🔄 Processing job {job_id} ({job_type})")
        
        try:
            # Generate unique timestamp for filenames
            timestamp = int(time.time())
            
            if job_type == 'enhance':
                # Enhance the prompt
                original_prompt = job_data.get('prompt', '')
                character_desc = job_data.get('characterDescription', '')
                
                if not original_prompt.strip():
                    print("⚠️ Empty prompt detected, using fallback")
                    original_prompt = "person walking"
                
                enhanced_prompt = self.enhance_prompt(original_prompt, character_desc)
                print(f"✅ Enhanced prompt: {enhanced_prompt[:100]}...")
                
                # Send enhanced prompt in callback
                self.notify_completion(job_id, 'completed', output_url=None, error_message=None, enhanced_prompt=enhanced_prompt)
                
            elif job_type == 'preview':
                # Generate preview image with unique filename
                prompt = job_data.get('prompt', 'woman walking')
                if not prompt.strip():
                    prompt = "woman walking"
                    
                preview_path = self.generate_preview(prompt)
                
                if preview_path:
                    # Create unique filename: jobId_timestamp_preview.png
                    filename = f"{job_id}_{timestamp}_preview.png"
                    storage_path = f"scene-previews/{filename}"
                    upload_url = self.upload_to_supabase(preview_path, storage_path)
                    
                    # Cleanup local file
                    if os.path.exists(preview_path):
                        os.remove(preview_path)
                        
                    print(f"✅ Preview uploaded: {upload_url}")
                    self.notify_completion(job_id, 'completed', upload_url)
                else:
                    raise Exception("Failed to generate preview")
                    
            elif job_type == 'video':
                # Generate final video with unique filename
                prompt = job_data.get('prompt', 'woman walking')
                if not prompt.strip():
                    prompt = "woman walking"
                    
                video_path = self.generate_video(prompt)
                
                if video_path:
                    # Create unique filename: jobId_timestamp_final.mp4
                    filename = f"{job_id}_{timestamp}_final.mp4"
                    storage_path = f"videos-final/{filename}"
                    upload_url = self.upload_to_supabase(video_path, storage_path)
                    
                    # Cleanup local file
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        
                    print(f"✅ Video uploaded: {upload_url}")
                    self.notify_completion(job_id, 'completed', upload_url)
                else:
                    raise Exception("Failed to generate video")
            else:
                print(f"⚠️ Unknown job type: {job_type}")
                self.notify_completion(job_id, 'failed', error_message=f"Unknown job type: {job_type}")
            
            print(f"🎉 Job {job_id} completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Job {job_id} failed: {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)

    def run(self):
        """Main worker loop with non-blocking Redis polling (RPOP)"""
        print("🎬 OurVidz GPU Worker started!")
        
        if self.wan_available:
            print("🎥 Running with Wan 2.1 support and enhanced file debugging")
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
                    # If environment is invalid, just wait and log
                    print("⚠️ Environment invalid, waiting...")
                    time.sleep(poll_interval)
                    idle_time += poll_interval
                    continue
                
                # Use non-blocking RPOP (Upstash REST API compatible)
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
                # worker.py - Complete Updated Version with Fixed Filenames
import os
import json
import time
import torch
import requests
import subprocess
from PIL import Image
from typing import Optional, List
from pathlib import Path

class VideoWorker:
    def __init__(self):
        """Initialize worker with Wan 2.1 models"""
        self.model_path = "/workspace/models"
        
        # Model instances (loaded on demand)
        self.current_model = None
        self.current_model_type = None
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        print("🚀 OurVidz Worker initialized")
        self.log_gpu_memory()
        self.check_models()

    def log_gpu_memory(self):
        """Monitor RTX 4090 24GB VRAM usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🔥 GPU Memory - Used: {memory_allocated:.2f}GB / {total_memory:.0f}GB")
        else:
            print("❌ No GPU detected")

    def check_models(self):
        """Check available models"""
        print("📋 Model Availability Check:")
        
        models = {
            "wan2.1-t2v-1.3b": "Fast video/image generation",
            "wan2.1-t2v-14b": "Premium video generation"
        }
        
        for model_name, description in models.items():
            model_path = Path(f"{self.model_path}/{model_name}")
            status = "✅ Available" if model_path.exists() else "❌ Missing"
            print(f"  {model_name:20} -> {status} ({description})")

    def generate_image_preview(self, prompt: str, job_id: str) -> Optional[str]:
        """Generate single frame image using Wan 2.1 T2V (2-3 seconds)"""
        print(f"🖼️ Generating image preview for job {job_id}")
        
        try:
            # Use Wan 2.1 T2V for single frame generation
            cmd = [
                "python", "/workspace/Wan2.1/scripts/inference.py",
                "--model_path", f"{self.model_path}/wan2.1-t2v-1.3b",
                "--prompt", prompt,
                "--num_frames", "1",  # Single frame for image
                "--height", "512",
                "--width", "768", 
                "--steps", "20",
                "--output_dir", "/tmp"
            ]
            
            print(f"⚡ Generating image (ETA: 2-3 seconds)")
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd="/workspace/Wan2.1"
            )
            
            generation_time = time.time() - start_time
            print(f"✅ Image generated in {generation_time:.1f}s")
            
            # Find the generated image
            output_files = list(Path("/tmp").glob("*.png"))
            if output_files:
                latest_file = max(output_files, key=lambda f: f.stat().st_mtime)
                return str(latest_file)
            else:
                raise Exception("Generated image file not found")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Image generation failed: {e.stderr}")
            raise Exception(f"Image generation failed: {e.stderr}")

    def generate_video(self, prompt: str, job_id: str, premium: bool = False) -> Optional[str]:
        """Generate 5-second video using Wan 2.1 T2V"""
        model_name = "wan2.1-t2v-14b" if premium else "wan2.1-t2v-1.3b"
        print(f"🎥 Generating video with {model_name} for job {job_id}")
        
        try:
            cmd = [
                "python", "/workspace/Wan2.1/scripts/inference.py",
                "--model_path", f"{self.model_path}/{model_name}",
                "--prompt", prompt,
                "--num_frames", "80",  # 5 seconds at 16fps
                "--height", "480" if not premium else "720",
                "--width", "832" if not premium else "1280",
                "--steps", "25",
                "--output_dir", "/tmp"
            ]
            
            eta = "6-8 minutes" if premium else "4-6 minutes"
            print(f"⚡ Generating video (ETA: {eta})")
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd="/workspace/Wan2.1"
            )
            
            generation_time = time.time() - start_time
            print(f"✅ Video generated in {generation_time/60:.1f} minutes")
            
            # Find the generated video
            output_files = list(Path("/tmp").glob("*.mp4"))
            if output_files:
                latest_file = max(output_files, key=lambda f: f.stat().st_mtime)
                return str(latest_file)
            else:
                raise Exception("Generated video file not found")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Video generation failed: {e.stderr}")
            raise Exception(f"Video generation failed: {e.stderr}")

    def upload_to_supabase(self, file_path: str, storage_path: str) -> str:
        """Upload file to Supabase storage with unique naming"""
        try:
            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': file},
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                        'upsert': 'true'  # Allow overwriting existing files
                    }
                )
            
            if response.status_code in [200, 201]:
                public_url = f"{self.supabase_url}/storage/v1/object/public/{storage_path}"
                print(f"📤 File uploaded: {public_url}")
                return public_url
            else:
                print(f"Upload response: {response.status_code} - {response.text}")
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
        """Process a single job with proper filename generation"""
        job_id = job_data['jobId']
        job_type = job_data.get('jobType', 'unknown')
        video_id = job_data.get('videoId', 'unknown')
        prompt = job_data.get('prompt', '')
        
        print(f"\n🔄 Processing job {job_id} ({job_type})")
        print(f"📝 Prompt: {prompt[:100]}...")
        
        try:
            output_path = None
            storage_path = None
            
            # Generate unique filename based on job_id and type
            timestamp = int(time.time())
            
            if job_type in ['preview', 'image', 'image_preview']:
                # Image generation (2-3 seconds)
                output_path = self.generate_image_preview(prompt, job_id)
                filename = f"{job_id}_{timestamp}_preview.png"
                storage_path = f"scene-previews/{filename}"
                
            elif job_type in ['video', 'video_fast']:
                # Standard video generation (4-6 minutes)
                output_path = self.generate_video(prompt, job_id, premium=False)
                filename = f"{job_id}_{timestamp}_video.mp4"
                storage_path = f"videos-final/{filename}"
                
            elif job_type in ['video_premium', 'video_hd']:
                # Premium video generation (6-8 minutes)
                output_path = self.generate_video(prompt, job_id, premium=True)
                filename = f"{job_id}_{timestamp}_premium.mp4"
                storage_path = f"videos-final/{filename}"
                
            else:
                raise ValueError(f"Unknown job type: {job_type}")
            
            if not output_path or not Path(output_path).exists():
                raise Exception("Generated file not found")
            
            # Upload to Supabase with unique path
            upload_url = self.upload_to_supabase(output_path, storage_path)
            
            # Clean up local file
            Path(output_path).unlink(missing_ok=True)
            
            print(f"✅ Job {job_id} completed successfully")
            self.notify_completion(job_id, 'completed', upload_url)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Job {job_id} failed: {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)

    def run(self):
        """Main worker loop with Redis polling"""
        print("\n🎬 OurVidz GPU Worker Started!")
        print("⏳ Waiting for jobs...")
        
        while True:
            try:
                # Poll Redis queue via REST API
                response = requests.get(
                    f"{os.getenv('UPSTASH_REDIS_REST_URL')}/brpop/job-queue/5",
                    headers={
                        'Authorization': f"Bearer {os.getenv('UPSTASH_REDIS_REST_TOKEN')}"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('result'):
                        queue_name, job_json = result['result']
                        job_data = json.loads(job_json)
                        self.process_job(job_data)
                    else:
                        print("💤 No jobs, waiting...")
                else:
                    print(f"⚠️ Redis connection issue: {response.status_code}")
                    time.sleep(30)
                    
            except Exception as e:
                print(f"❌ Worker error: {e}")
                time.sleep(30)

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
    worker.run()
