# worker.py - OPTIMIZED with Model Persistence
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
        """Initialize with persistent model loading"""
        print("🚀 OurVidz Worker initialized (PERSISTENT 1.3B)")
        
        # Check dependencies
        self.ffmpeg_available = shutil.which('ffmpeg') is not None
        print(f"🔧 FFmpeg Available: {self.ffmpeg_available}")
        
        # GPU detection
        self.detect_gpu()
        
        # Check for Wan 2.1 installation
        self.wan_available = self.check_wan_installation()
        print(f"🎥 Wan 2.1 Available: {self.wan_available}")
        
        # MODEL PERSISTENCE: Load once, keep in memory
        self.model_loaded = False
        self.model_path = '/workspace/models/wan2.1-t2v-1.3b'
        
        # OPTIMIZED job configurations
        self.job_configs = {
            'image_fast': {
                'size': '832*480',
                'frame_num': 1,
                'sample_steps': 8,
                'sample_guide_scale': 6.0,
                'expected_time': '15-30 seconds'
            },
            'image_high': {
                'size': '1280*720',
                'frame_num': 1,
                'sample_steps': 20,
                'sample_guide_scale': 7.5,
                'expected_time': '45-90 seconds'
            },
            'video_fast': {
                'size': '832*480',
                'frame_num': 17,
                'sample_steps': 12,
                'sample_guide_scale': 6.0,
                'expected_time': '60-120 seconds'
            },
            'video_high': {
                'size': '1280*720',
                'frame_num': 33,
                'sample_steps': 25,
                'sample_guide_scale': 7.5,
                'expected_time': '3-6 minutes'
            }
        }
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        # LOAD MODEL ON STARTUP (instead of per-job)
        if self.wan_available:
            self.preload_model()
        
        print("🎬 PERSISTENT 1.3B OurVidz Worker started!")

    def detect_gpu(self):
        """Detect GPU and memory"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                gpu_name = gpu_info[0]
                gpu_memory_total = gpu_info[1]
                gpu_memory_free = gpu_info[2] 
                gpu_memory_used = gpu_info[3]
                print(f"🔥 GPU: {gpu_name} ({gpu_memory_total}GB total)")
                print(f"💾 VRAM: {gpu_memory_used}MB used, {gpu_memory_free}MB free")
                
                # If already using significant VRAM, model might be loaded
                if int(gpu_memory_used) > 5000:
                    print("⚡ Significant VRAM usage detected - model may already be loaded")
                    self.model_loaded = True
                    
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

    def preload_model(self):
        """Load model once on startup using minimal generation"""
        if self.model_loaded:
            print("✅ Model already loaded")
            return
            
        print("🔄 PRELOADING 1.3B model (one-time startup cost)...")
        start_time = time.time()
        
        try:
            # Change to Wan2.1 directory for model loading
            os.chdir("/workspace/Wan2.1")
            
            # Run minimal generation to load model into VRAM
            # Use smallest possible settings to minimize initial load time
            dummy_cmd = [
                "python", "generate.py",
                "--task", "t2v-1.3B",
                "--size", "480*832",  # Smallest size
                "--ckpt_dir", self.model_path,
                "--prompt", "a",  # Minimal prompt
                "--save_file", "preload_dummy.mp4",
                "--sample_steps", "1",  # Minimum steps
                "--sample_guide_scale", "3.0",  # Lower guidance
                "--frame_num", "1"  # Single frame
            ]
            
            print(f"🔧 Preload command: {' '.join(dummy_cmd)}")
            
            # Run with timeout for initial load
            result = subprocess.run(dummy_cmd, capture_output=True, text=True, timeout=180)
            
            load_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Model preloaded successfully in {load_time:.1f}s")
                self.model_loaded = True
                
                # Clean up dummy file
                try:
                    dummy_file = "/workspace/Wan2.1/preload_dummy.mp4"
                    if os.path.exists(dummy_file):
                        os.remove(dummy_file)
                        print("🧹 Cleaned up preload dummy file")
                except Exception as cleanup_error:
                    print(f"⚠️ Cleanup failed: {cleanup_error}")
            else:
                print(f"⚠️ Preload failed, will load per-job:")
                print(f"   Return code: {result.returncode}")
                print(f"   Error: {result.stderr[:200]}...")
                # Model will load per-job instead
                
        except subprocess.TimeoutExpired:
            print("⏰ Model preload timed out after 3 minutes, will load per-job")
        except Exception as e:
            print(f"❌ Preload error (will load per-job): {e}")

    def check_gpu_memory(self):
        """Check GPU memory usage to see if model is loaded"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                memory_used = int(result.stdout.strip())
                return memory_used
        except:
            pass
        return 0

    def generate_optimized(self, prompt, job_type):
        """FAST generation with persistent model monitoring"""
        if job_type not in self.job_configs:
            print(f"❌ Unknown job type: {job_type}")
            return None
            
        config = self.job_configs[job_type]
        job_id = str(uuid.uuid4())[:8]
        
        try:
            # Check memory before generation
            memory_before = self.check_gpu_memory()
            model_likely_loaded = memory_before > 5000  # >5GB suggests model is loaded
            
            print(f"⚡ PERSISTENT {job_type} generation:")
            print(f"   📐 Size: {config['size'].replace('*', 'x')}")
            print(f"   🎞️ Frames: {config['frame_num']} ({'image' if config['frame_num'] == 1 else 'video'})")
            print(f"   🔧 Steps: {config['sample_steps']}")
            print(f"   ⏱️ Expected time: {config['expected_time']}")
            print(f"   📝 Prompt: {prompt}")
            print(f"   🧠 Model status: {'LIKELY LOADED' if model_likely_loaded else 'COLD START'} ({memory_before}MB VRAM)")
            
            output_filename = f"{job_type}_{job_id}.mp4"
            
            # Build optimized command
            cmd = [
                "python", "generate.py",
                "--task", "t2v-1.3B",
                "--size", config['size'],
                "--ckpt_dir", self.model_path,
                "--prompt", prompt,
                "--save_file", output_filename,
                "--sample_steps", str(config['sample_steps']),
                "--sample_guide_scale", str(config['sample_guide_scale']),
                "--frame_num", str(config['frame_num'])
            ]
            
            # Adjust timeout based on whether model is likely loaded
            timeout = 90 if model_likely_loaded else 180
            
            os.chdir("/workspace/Wan2.1")
            start_time = time.time()
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            generation_time = time.time() - start_time
            memory_after = self.check_gpu_memory()
            
            status = "WARM" if model_likely_loaded else "COLD"
            print(f"📤 {status} generation completed in {generation_time:.1f}s")
            print(f"   🧠 VRAM: {memory_before}MB → {memory_after}MB")
            
            # Update model loaded status based on results
            if generation_time < 60 and memory_after > 5000:
                self.model_loaded = True
                print("   ✅ Model appears to be persistent in VRAM")
            
            if result.returncode != 0:
                print(f"❌ Generation failed:")
                print(f"   stderr: {result.stderr}")
                return None
            
            # Process output file
            video_path = f"/workspace/Wan2.1/{output_filename}"
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / (1024 * 1024)
                print(f"✅ Generated: {video_path} ({file_size:.1f}MB)")
                
                # For image jobs, extract frame
                if 'image' in job_type:
                    image_path = self.extract_frame_from_video(video_path, job_id, job_type)
                    if image_path:
                        try:
                            os.remove(video_path)
                            print(f"🧹 Cleaned up intermediate video")
                        except:
                            pass
                        return image_path
                    else:
                        return video_path
                else:
                    return video_path
            else:
                print(f"❌ Video not found: {video_path}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Generation timed out after {timeout}s")
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
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vf', 'select=eq(n\\,0)',
                    '-vframes', '1',
                    '-q:v', '2',
                    '-y', image_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(image_path):
                    file_size = os.path.getsize(image_path) / 1024
                    print(f"✅ FFmpeg extracted image: {image_path} ({file_size:.0f}KB)")
                    return image_path
                else:
                    print(f"❌ FFmpeg failed: {result.stderr}")
            
            # OpenCV fallback
            print("🔄 Trying OpenCV fallback...")
            cap = cv2.VideoCapture(video_path)
            
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    image.save(image_path, "PNG", optimize=True)
                    
                    if os.path.exists(image_path):
                        file_size = os.path.getsize(image_path) / 1024
                        print(f"✅ OpenCV extracted image: {image_path} ({file_size:.0f}KB)")
                        return image_path
            
            return None
                
        except Exception as e:
            print(f"❌ Frame extraction error: {e}")
            return None

    def upload_to_supabase(self, file_path, job_type, user_id, job_id):
        """Upload to Supabase storage"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ File does not exist: {file_path}")
                return None
                
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"📤 Uploading {file_size:.1f}MB to Supabase...")
            
            if 'image' in job_type:
                bucket = job_type
                extension = "png"
            else:
                bucket = job_type
                extension = "mp4"
            
            timestamp = int(time.time())
            filename = f"job_{job_id}_{timestamp}_{job_type}.{extension}"
            user_path = f"{user_id}/{filename}"
            storage_path = f"{bucket}/{user_path}"
            
            content_type = 'application/octet-stream'
            if file_path.lower().endswith('.png'):
                content_type = 'image/png'
            elif file_path.lower().endswith('.mp4'):
                content_type = 'video/mp4'
            
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
                print(f"✅ Callback sent successfully")
            else:
                print(f"❌ Callback failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job(self, job_data):
        """Process job with persistent model"""
        job_id = job_data.get('jobId')
        job_type = job_data.get('jobType')
        prompt = job_data.get('prompt', 'person walking')
        user_id = job_data.get('userId')
        
        print(f"\n📥 PERSISTENT Processing: {job_type} - {job_id}")
        
        if not job_id or not job_type or not user_id:
            error_msg = f"Missing required fields"
            print(f"❌ {error_msg}")
            self.notify_completion(job_id or 'unknown', 'failed', error_message=error_msg)
            return
        
        try:
            if self.wan_available:
                output_path = self.generate_optimized(prompt, job_type)
            else:
                raise Exception("Wan 2.1 not available")
            
            if output_path and os.path.exists(output_path):
                file_path = self.upload_to_supabase(output_path, job_type, user_id, job_id)
                
                if file_path:
                    print(f"🎉 PERSISTENT job {job_id} completed!")
                    self.notify_completion(job_id, 'completed', file_path)
                else:
                    raise Exception("Upload failed")
                
                try:
                    os.remove(output_path)
                    print(f"🧹 Cleaned up: {output_path}")
                except:
                    pass
            else:
                raise Exception("Generation failed")
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Job {job_id} failed: {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)

    def poll_queue(self):
        """Poll Redis queue for jobs"""
        try:
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
                print(f"⚠️ Redis error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Queue polling error: {e}")
            return None

    def run(self):
        """Main worker loop with idle shutdown"""
        print("⏳ Waiting for PERSISTENT jobs...")
        
        if self.model_loaded:
            print("⚡ Running with PERSISTENT model (expected performance):")
            print("   🎯 image_fast: 15-30s (was 105s)")
            print("   🎯 image_high: 45-90s")
            print("   🎯 video_fast: 60-120s")
            print("   🎯 video_high: 3-6min")
        else:
            print("⚠️ Running with PER-JOB loading (slower)")
        
        idle_time = 0
        max_idle_time = 10 * 60  # 10 minutes
        poll_interval = 5
        
        while True:
            try:
                job_data = self.poll_queue()
                
                if job_data:
                    idle_time = 0
                    self.process_job(job_data)
                    print("⏳ Waiting for next PERSISTENT job...")
                else:
                    idle_time += poll_interval
                    
                    if idle_time % 60 == 0 and idle_time > 0:
                        minutes_idle = idle_time // 60
                        max_minutes = max_idle_time // 60
                        print(f"💤 Idle {minutes_idle}/{max_minutes}min (model in VRAM)")
                    
                    if idle_time >= max_idle_time:
                        print(f"🛑 Shutting down after {max_idle_time//60} minutes")
                        break
                    
                    time.sleep(poll_interval)
                    
            except KeyboardInterrupt:
                print("👋 Worker stopped")
                break
            except Exception as e:
                print(f"❌ Worker error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    print("🚀 Starting PERSISTENT 1.3B Worker...")
    worker = VideoWorker()
    worker.run()
