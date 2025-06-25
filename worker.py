# worker.py - OPTIMIZED for Wan 2.1 1.3B (ALL JOBS)
import os
import json
import time
import requests
import subprocess
import uuid
import shutil
from pathlib import Path
from PIL import Image
import cv2

class VideoWorker:
    def __init__(self):
        """Initialize optimized OurVidz worker using 1.3B model for all jobs"""
        print("🚀 OurVidz Worker initialized (1.3B OPTIMIZED)")
        
        # Check dependencies
        self.ffmpeg_available = shutil.which('ffmpeg') is not None
        print(f"🔧 FFmpeg Available: {self.ffmpeg_available}")
        
        # GPU detection
        self.detect_gpu()
        
        # Check for Wan 2.1 installation
        self.wan_available = self.check_wan_installation()
        print(f"🎥 Wan 2.1 Available: {self.wan_available}")
        
        # OPTIMIZED job configurations using 1.3B model for ALL jobs
        self.job_configs = {
            'image_fast': {
                'task': 't2v-1.3B',         # Use 1.3B for single frame generation
                'size': '832*480',          # Fast tier: Standard definition
                'model_path': '/workspace/models/wan2.1-t2v-1.3b',
                'frame_num': 1,             # Single frame for image
                'sample_steps': 8,          # Fast generation
                'sample_guide_scale': 6.0,  # Balanced guidance
                'offload_model': False,     # Keep in VRAM for speed
                'expected_time': '15-30 seconds'
            },
            'image_high': {
                'task': 't2v-1.3B',         # Use 1.3B for single frame generation
                'size': '1280*720',         # High tier: HD quality
                'model_path': '/workspace/models/wan2.1-t2v-1.3b',
                'frame_num': 1,             # Single frame for image
                'sample_steps': 20,         # Quality generation
                'sample_guide_scale': 7.5,  # Higher guidance for quality
                'offload_model': False,     # 1.3B fits in VRAM
                'expected_time': '45-90 seconds'
            },
            'video_fast': {
                'task': 't2v-1.3B',         # Native video generation
                'size': '832*480',          # Fast tier: Standard definition
                'model_path': '/workspace/models/wan2.1-t2v-1.3b',
                'frame_num': 17,            # 1 second video (4n+1 = 17 frames)
                'sample_steps': 12,         # Balanced speed/quality
                'sample_guide_scale': 6.0,  # Standard guidance
                'offload_model': False,     # Keep in VRAM
                'expected_time': '60-120 seconds'
            },
            'video_high': {
                'task': 't2v-1.3B',         # Native video generation
                'size': '1280*720',         # High tier: HD quality
                'model_path': '/workspace/models/wan2.1-t2v-1.3b',
                'frame_num': 33,            # 2 second video (4n+1 = 33 frames)
                'sample_steps': 25,         # Quality generation
                'sample_guide_scale': 7.5,  # High guidance for quality
                'offload_model': False,     # 1.3B reliable in VRAM
                'expected_time': '3-6 minutes'
            }
        }
        
        print("⚡ 1.3B OPTIMIZED configurations (Reliable + Fast):")
        for job_type, config in self.job_configs.items():
            size_display = config['size'].replace('*', 'x')
            frames = config['frame_num']
            content_type = "image" if frames == 1 else f"{frames}-frame video"
            print(f"   🎯 {job_type}: {config['task']} | {size_display} | {content_type} | {config['expected_time']}")
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        print("🎬 1.3B OPTIMIZED OurVidz Worker started!")

    def detect_gpu(self):
        """Detect GPU"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                gpu_name = gpu_info[0]
                gpu_memory = gpu_info[1]
                print(f"🔥 GPU: {gpu_name} ({gpu_memory}GB)")
        except Exception as e:
            print(f"⚠️ GPU detection failed: {e}")

    def check_wan_installation(self):
        """Check Wan 2.1 1.3B model installation"""
        try:
            wan_dir = Path("/workspace/Wan2.1")
            generate_script = wan_dir / "generate.py"
            model_1_3b = Path("/workspace/models/wan2.1-t2v-1.3b")
            
            if not wan_dir.exists():
                print("❌ Wan2.1 directory not found")
                return False
                
            if not generate_script.exists():
                print("❌ generate.py not found")
                return False
                
            if not model_1_3b.exists():
                print("❌ 1.3B model directory not found")
                return False
                
            print("✅ Wan 2.1 1.3B model verified")
            return True
                
        except Exception as e:
            print(f"❌ Error checking Wan installation: {e}")
            return False

    def generate_optimized(self, prompt, job_type):
        """OPTIMIZED generation using 1.3B model for all jobs"""
        if job_type not in self.job_configs:
            print(f"❌ Unknown job type: {job_type}")
            return None
            
        config = self.job_configs[job_type]
        job_id = str(uuid.uuid4())[:8]
        
        try:
            # Clear CUDA cache for clean generation
            print("🧹 Clearing CUDA cache...")
            import torch
            torch.cuda.empty_cache()
            
            print(f"⚡ 1.3B OPTIMIZED {job_type} generation:")
            print(f"   🎯 Task: {config['task']}")
            print(f"   📁 Model: 1.3B (reliable)")
            print(f"   📐 Size: {config['size'].replace('*', 'x')}")
            print(f"   🎞️ Frames: {config['frame_num']} ({'image' if config['frame_num'] == 1 else 'video'})")
            print(f"   🔧 Steps: {config['sample_steps']}")
            print(f"   ⏱️ Expected time: {config['expected_time']}")
            print(f"   📝 Prompt: {prompt}")
            
            # Always generate as video (even for images)
            output_filename = f"{job_type}_{job_id}.mp4"
            
            # Build optimized command for 1.3B model
            cmd = [
                "python", "generate.py",
                "--task", config['task'],
                "--size", config['size'],
                "--ckpt_dir", config['model_path'],
                "--prompt", prompt,
                "--save_file", output_filename,
                "--sample_steps", str(config['sample_steps']),
                "--sample_guide_scale", str(config['sample_guide_scale']),
                "--frame_num", str(config['frame_num'])
            ]
            
            # Add memory optimization only if needed (1.3B usually doesn't need it)
            if config.get('offload_model'):
                cmd.extend(["--offload_model", "True"])
            
            print(f"🔧 1.3B Command: {' '.join(cmd)}")
            
            # Run generation with appropriate timeout
            os.chdir("/workspace/Wan2.1")
            start_time = time.time()
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            generation_time = time.time() - start_time
            print(f"📤 Generation completed in {generation_time:.1f}s (return code: {result.returncode})")
            
            if result.returncode != 0:
                print(f"❌ Generation failed:")
                print(f"   stderr: {result.stderr}")
                print(f"   stdout: {result.stdout}")
                return None
            
            # Find generated video
            video_path = f"/workspace/Wan2.1/{output_filename}"
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / (1024 * 1024)
                print(f"✅ Generated video: {video_path} ({file_size:.1f}MB in {generation_time:.1f}s)")
                
                # For image jobs, extract frame from single-frame video
                if 'image' in job_type:
                    image_path = self.extract_frame_from_video(video_path, job_id, job_type)
                    if image_path:
                        # Clean up intermediate video file
                        try:
                            os.remove(video_path)
                            print(f"🧹 Cleaned up intermediate video: {video_path}")
                        except:
                            pass
                        return image_path
                    else:
                        print("⚠️ Frame extraction failed, returning video")
                        return video_path
                else:
                    # Return video directly for video jobs
                    return video_path
            else:
                print(f"❌ Video not found: {video_path}")
                # Search for alternative files
                try:
                    files = [f for f in os.listdir("/workspace/Wan2.1") if job_id in f]
                    print(f"   📂 Found files with job_id: {files}")
                    if files:
                        fallback_path = f"/workspace/Wan2.1/{files[0]}"
                        print(f"   🔄 Using fallback: {fallback_path}")
                        return fallback_path
                except Exception as e:
                    print(f"   ❌ Directory listing failed: {e}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Generation timed out after 10 minutes")
            return None
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None

    def extract_frame_from_video(self, video_path, job_id, job_type):
        """Extract frame from single-frame video for image jobs"""
        try:
            image_path = f"/workspace/Wan2.1/{job_type}_{job_id}.png"
            
            print(f"🎞️ Extracting frame for image job...")
            
            if self.ffmpeg_available:
                # Method 1: FFmpeg (preferred)
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vf', 'select=eq(n\\,0)',  # First frame
                    '-vframes', '1',
                    '-q:v', '2',  # High quality
                    '-y', image_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(image_path):
                    file_size = os.path.getsize(image_path) / 1024  # KB
                    print(f"✅ FFmpeg extracted image: {image_path} ({file_size:.0f}KB)")
                    return image_path
                else:
                    print(f"❌ FFmpeg failed: {result.stderr}")
            
            # Method 2: OpenCV fallback
            print("🔄 Trying OpenCV fallback...")
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print("❌ OpenCV could not open video")
                return None
            
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                # Convert BGR to RGB and save
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image.save(image_path, "PNG", optimize=True)
                
                if os.path.exists(image_path):
                    file_size = os.path.getsize(image_path) / 1024
                    print(f"✅ OpenCV extracted image: {image_path} ({file_size:.0f}KB)")
                    return image_path
            
            print("❌ All frame extraction methods failed")
            return None
                
        except Exception as e:
            print(f"❌ Frame extraction error: {e}")
            return None

    def upload_to_supabase(self, file_path, job_type, user_id, job_id):
        """Upload to Supabase storage with user-scoped paths"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ File does not exist: {file_path}")
                return None
                
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"📤 Uploading {file_size:.1f}MB to Supabase...")
            
            # Determine bucket and extension based on job type
            if 'image' in job_type:
                bucket = job_type  # image_fast or image_high
                extension = "png"
            else:
                bucket = job_type  # video_fast or video_high
                extension = "mp4"
            
            # Create user-scoped filename with timestamp
            timestamp = int(time.time())
            filename = f"job_{job_id}_{timestamp}_{job_type}.{extension}"
            user_path = f"{user_id}/{filename}"
            storage_path = f"{bucket}/{user_path}"
            
            print(f"🗂️ Bucket: {bucket}")
            print(f"📁 User path: {user_path}")
            
            # Determine content type
            content_type = 'application/octet-stream'
            if file_path.lower().endswith('.png'):
                content_type = 'image/png'
            elif file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif file_path.lower().endswith('.mp4'):
                content_type = 'video/mp4'
            
            # Upload to Supabase
            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': (os.path.basename(file_path), file, content_type)},
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                        'x-upsert': 'true'
                    },
                    timeout=120
                )
            
            if response.status_code in [200, 201]:
                print(f"✅ Uploaded to: {storage_path}")
                return user_path
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return None

    def notify_completion(self, job_id, status, file_path=None, error_message=None):
        """Send completion callback to Supabase"""
        try:
            callback_data = {
                'jobId': job_id,
                'status': status,
                'filePath': file_path,
                'errorMessage': error_message
            }
            
            print(f"📞 Sending callback for job {job_id}: {status}")
            if file_path:
                print(f"📁 File path: {file_path}")
            
            response = requests.post(
                f"{self.supabase_url}/functions/v1/job-callback",
                json=callback_data,
                headers={
                    'Authorization': f"Bearer {self.supabase_service_key}",
                    'Content-Type': 'application/json'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"✅ Callback sent successfully for job {job_id}")
            else:
                print(f"❌ Callback failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job(self, job_data):
        """Process job with 1.3B optimization"""
        job_id = job_data.get('jobId')
        job_type = job_data.get('jobType')
        prompt = job_data.get('prompt', 'person walking')
        user_id = job_data.get('userId')
        
        print(f"\n📥 1.3B OPTIMIZED Processing: {job_type} - {job_id}")
        print(f"👤 User: {user_id}")
        
        # Validate required fields
        if not job_id or not job_type or not user_id:
            error_msg = f"Missing required fields: jobId={job_id}, jobType={job_type}, userId={user_id}"
            print(f"❌ {error_msg}")
            self.notify_completion(job_id or 'unknown', 'failed', error_message=error_msg)
            return
        
        # Show expected performance
        if job_type in self.job_configs:
            expected_time = self.job_configs[job_type]['expected_time']
            frames = self.job_configs[job_type]['frame_num']
            content_type = "image" if frames == 1 else f"{frames}-frame video"
            print(f"⏱️ Expected completion: {expected_time}")
            print(f"🎬 Generating: {content_type}")
        else:
            error_msg = f"Unknown job type: {job_type}. Supported: {list(self.job_configs.keys())}"
            print(f"❌ {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)
            return
        
        try:
            if self.wan_available:
                output_path = self.generate_optimized(prompt, job_type)
            else:
                print("❌ Wan 2.1 1.3B not available")
                raise Exception("Wan 2.1 1.3B not available")
            
            if output_path and os.path.exists(output_path):
                # Upload to Supabase with user-scoped path
                file_path = self.upload_to_supabase(output_path, job_type, user_id, job_id)
                
                if file_path:
                    print(f"🎉 1.3B OPTIMIZED job {job_id} completed successfully!")
                    self.notify_completion(job_id, 'completed', file_path)
                else:
                    raise Exception("Upload to Supabase failed")
                
                # Cleanup local file
                try:
                    os.remove(output_path)
                    print(f"🧹 Cleaned up: {output_path}")
                except:
                    pass
            else:
                raise Exception("Generation failed - no output file created")
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Job {job_id} failed: {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)

    def poll_queue(self):
        """Poll Redis queue for jobs (FIXED: job_queue not job-queue)"""
        try:
            # FIXED: Use job_queue (underscore) instead of job-queue (hyphen)
            response = requests.get(
                f"{self.redis_url}/rpop/job_queue",
                headers={'Authorization': f"Bearer {self.redis_token}"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('result'):
                    return json.loads(result['result'])
                else:
                    return None
            else:
                print(f"⚠️ Redis error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Queue polling error: {e}")
            return None

    def run(self):
        """Main worker loop with idle shutdown"""
        print("⏳ Waiting for 1.3B OPTIMIZED jobs...")
        
        if self.wan_available:
            print("⚡ Running with 1.3B model for ALL job types:")
            print("   🎯 image_fast: 1.3B → 832x480 single frame → PNG (15-30s)")
            print("   🎯 image_high: 1.3B → 1280x720 single frame → PNG (45-90s)")
            print("   🎯 video_fast: 1.3B → 832x480 17-frame video → MP4 (60-120s)")
            print("   🎯 video_high: 1.3B → 1280x720 33-frame video → MP4 (3-6min)")
            print("   ✅ Expected success rate: 99%+")
        else:
            print("⚠️ Wan 2.1 1.3B not available - worker will fail jobs")
        
        idle_time = 0
        max_idle_time = 10 * 60  # 10 minutes
        poll_interval = 5  # 5 seconds
        
        while True:
            try:
                job_data = self.poll_queue()
                
                if job_data:
                    # Reset idle timer when job received
                    idle_time = 0
                    self.process_job(job_data)
                    print("⏳ Waiting for next 1.3B job...")
                else:
                    # Increment idle time
                    idle_time += poll_interval
                    
                    # Log idle status every minute
                    if idle_time % 60 == 0 and idle_time > 0:
                        minutes_idle = idle_time // 60
                        max_minutes = max_idle_time // 60
                        print(f"💤 Idle for {minutes_idle}/{max_minutes} minutes...")
                    
                    # Shutdown after max idle time
                    if idle_time >= max_idle_time:
                        print(f"🛑 Shutting down after {max_idle_time//60} minutes of inactivity")
                        break
                    
                    time.sleep(poll_interval)
                    
            except KeyboardInterrupt:
                print("👋 Worker stopped by user")
                break
            except Exception as e:
                print(f"❌ Worker error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    # Validate environment variables
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_SERVICE_KEY', 
        'UPSTASH_REDIS_REST_URL',
        'UPSTASH_REDIS_REST_TOKEN'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        print("🔄 Worker will start but may have limited functionality")
    
    print("🚀 Starting 1.3B OPTIMIZED OurVidz Worker...")
    worker = VideoWorker()
    worker.run()
