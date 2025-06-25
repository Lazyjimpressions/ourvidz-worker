# worker.py - SUPER FAST optimized version
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
        """Initialize optimized OurVidz worker"""
        print("🚀 OurVidz Worker initialized (OPTIMIZED)")
        
        # Check dependencies
        self.ffmpeg_available = shutil.which('ffmpeg') is not None
        print(f"🔧 FFmpeg Available: {self.ffmpeg_available}")
        
        # GPU detection
        self.detect_gpu()
        
        # Check for Wan 2.1 installation
        self.wan_available = self.check_wan_installation()
        print(f"🎥 Wan 2.1 Available: {self.wan_available}")
        
        # Optimized sizes and parameters for each job type
        self.job_configs = {
            'image_fast': {
                'task': 't2i-14B',           # Direct image generation (14B for quality)
                'size': '480*832',           # Portrait, small
                'sample_steps': 10,          # FAST: Minimal steps
                'offload_model': True,       # Memory optimization
                'expected_time': '10-30 seconds'
            },
            'image_high': {
                'task': 't2i-14B',           # Direct image generation (14B for best quality)
                'size': '1024*1024',         # Square, high-res
                'sample_steps': 25,          # QUALITY: More steps for premium
                'offload_model': True,       # Memory optimization
                'expected_time': '45-90 seconds'
            },
            'video_fast': {
                'task': 't2v-1.3B',          # FAST video generation (1.3B for speed)
                'size': '832*480',           # Landscape, standard
                'frame_num': 17,             # 1 second (4n+1 = 17 frames)
                'sample_steps': 15,          # FAST: Fewer steps
                'offload_model': True,       # Memory optimization
                'expected_time': '1-2 minutes'
            },
            'video_high': {
                'task': 't2v-14B',           # PREMIUM video generation (14B for quality)
                'size': '1280*720',          # HD landscape
                'frame_num': 33,             # 2 seconds (4n+1 = 33 frames)
                'sample_steps': 30,          # QUALITY: More steps for premium
                'offload_model': True,       # Memory optimization
                'expected_time': '8-12 minutes'
            }
        }
        
        print("⚡ OPTIMIZED configurations:")
        for job_type, config in self.job_configs.items():
            print(f"   🎯 {job_type}: {config['task']} | {config['size']} | {config['expected_time']}")
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        print("🎬 SUPER FAST OurVidz Worker started!")

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
        """Check Wan 2.1 installation"""
        try:
            wan_dir = Path("/workspace/Wan2.1")
            generate_script = wan_dir / "generate.py"
            model_dir = Path("/workspace/models/wan2.1-t2v-1.3b")
            
            if wan_dir.exists() and generate_script.exists() and model_dir.exists():
                return True
            else:
                print("❌ Wan 2.1 installation incomplete")
                return False
                
        except Exception as e:
            print(f"❌ Error checking Wan installation: {e}")
            return False

    def generate_optimized(self, prompt, job_type):
        """OPTIMIZED generation based on job type"""
        if job_type not in self.job_configs:
            print(f"❌ Unknown job type: {job_type}")
            return None
            
        config = self.job_configs[job_type]
        job_id = str(uuid.uuid4())[:8]
        
        try:
            print(f"⚡ OPTIMIZED {job_type} generation:")
            print(f"   🎯 Task: {config['task']}")
            print(f"   📐 Size: {config['size']}")
            print(f"   ⏱️ Expected time: {config['expected_time']}")
            print(f"   📝 Prompt: {prompt}")
            
            # Determine output file extension based on task
            if config['task'].startswith('t2i'):
                output_filename = f"{job_type}_{job_id}.png"  # Direct image output
            else:
                output_filename = f"{job_type}_{job_id}.mp4"  # Video output
            
            # Build optimized command
            cmd = [
                "python", "generate.py",
                "--task", config['task'],
                "--size", config['size'],
                "--ckpt_dir", "/workspace/models/wan2.1-t2v-1.3b",
                "--prompt", prompt,
                "--save_file", output_filename,
                "--sample_steps", str(config['sample_steps'])
            ]
            
            # Add frame_num for video tasks
            if 'frame_num' in config:
                cmd.extend(["--frame_num", str(config['frame_num'])])
            
            # Add memory optimization
            if config.get('offload_model'):
                cmd.extend(["--offload_model", "True"])
            
            print(f"🔧 Command: {' '.join(cmd)}")
            
            # Run generation with timing
            os.chdir("/workspace/Wan2.1")
            start_time = time.time()
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            generation_time = time.time() - start_time
            print(f"📤 Generation completed in {generation_time:.1f}s (return code: {result.returncode})")
            
            if result.returncode != 0:
                print(f"❌ Generation failed:")
                print(f"   stderr: {result.stderr}")
                print(f"   stdout: {result.stdout}")
                return None
            
            # Find and validate output
            output_path = f"/workspace/Wan2.1/{output_filename}"
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"✅ Generated: {output_path} ({file_size:.1f}MB in {generation_time:.1f}s)")
                return output_path
            else:
                print(f"❌ Output file not found: {output_path}")
                # List files for debugging
                try:
                    files = [f for f in os.listdir("/workspace/Wan2.1") if job_id in f]
                    print(f"   📂 Found files with job_id: {files}")
                    if files:
                        # Return the first matching file
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

    def upload_to_supabase(self, file_path, job_type, user_id, job_id):
        """Upload to Supabase storage"""
        try:
            if not os.path.exists(file_path):
                return None
                
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"📤 Uploading {file_size:.1f}MB...")
            
            # Determine bucket and extension
            if 'image' in job_type:
                bucket = job_type
                extension = "png"
            else:
                bucket = job_type
                extension = "mp4"
            
            # Create filename
            timestamp = int(time.time())
            filename = f"job_{job_id}_{timestamp}_{job_type}.{extension}"
            user_path = f"{user_id}/{filename}"
            storage_path = f"{bucket}/{user_path}"
            
            # Upload
            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': file},
                    headers={'Authorization': f"Bearer {self.supabase_service_key}"},
                    timeout=60
                )
            
            if response.status_code == 200:
                print(f"✅ Uploaded: {storage_path}")
                return user_path
            else:
                print(f"❌ Upload failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return None

    def notify_completion(self, job_id, status, file_path=None, error_message=None):
        """Send completion callback"""
        try:
            callback_data = {
                'jobId': job_id,
                'status': status,
                'filePath': file_path,
                'errorMessage': error_message
            }
            
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
                print(f"✅ Callback sent for job {job_id}")
            else:
                print(f"❌ Callback failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job(self, job_data):
        """Process a job with optimized generation"""
        job_id = job_data.get('jobId')
        job_type = job_data.get('jobType')
        prompt = job_data.get('prompt')
        user_id = job_data.get('userId')
        
        print(f"\n📥 FAST Processing: {job_type} - {job_id}")
        
        # Show expected performance
        if job_type in self.job_configs:
            expected_time = self.job_configs[job_type]['expected_time']
            print(f"⏱️ Expected completion: {expected_time}")
        
        try:
            if self.wan_available:
                output_path = self.generate_optimized(prompt, job_type)
            else:
                print("❌ Wan 2.1 not available")
                raise Exception("Wan 2.1 not available")
            
            if output_path and os.path.exists(output_path):
                file_path = self.upload_to_supabase(output_path, job_type, user_id, job_id)
                
                if file_path:
                    print(f"🎉 FAST job {job_id} completed!")
                    self.notify_completion(job_id, 'completed', file_path)
                else:
                    raise Exception("Upload failed")
                
                # Cleanup
                try:
                    os.remove(output_path)
                except:
                    pass
            else:
                raise Exception("Generation failed")
                
        except Exception as e:
            print(f"❌ Job {job_id} failed: {e}")
            self.notify_completion(job_id, 'failed', error_message=str(e))

    def poll_queue(self):
        """Poll for jobs"""
        try:
            response = requests.get(
                f"{self.redis_url}/rpop/job-queue",
                headers={'Authorization': f"Bearer {self.redis_token}"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('result'):
                    return json.loads(result['result'])
            return None
                
        except Exception as e:
            print(f"❌ Queue error: {e}")
            return None

    def run(self):
        """Main loop"""
        print("⏳ Waiting for FAST jobs...")
        
        while True:
            try:
                job_data = self.poll_queue()
                
                if job_data:
                    self.process_job(job_data)
                else:
                    print("💤 No jobs...")
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Worker error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    worker = VideoWorker()
    worker.run()
