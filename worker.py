# worker.py - Simple Fixed Version (No Syntax Errors)
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
import torch
import numpy as np

# Clean environment
for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
    if key in os.environ:
        del os.environ[key]

# GPU optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class SimpleWorker:
    def __init__(self):
        print("🚀 OurVidz SIMPLE WORKER - RTX 6000 ADA")
        print("🔧 Diagnosis Mode: Will capture full error output")
        
        # Paths
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_path = "/workspace/Wan2.1"
        
        # Job configurations
        self.job_type_mapping = {
            'image_fast': {
                'content_type': 'image',
                'file_extension': 'png',
                'sample_steps': 4,
                'sample_guide_scale': 3.0,
                'size': '480*832',
                'frame_num': 1,
                'storage_bucket': 'image_fast',
                'expected_time': 60
            },
            'video_fast': {
                'content_type': 'video',
                'file_extension': 'mp4',
                'sample_steps': 4,
                'sample_guide_scale': 3.0,
                'size': '480*832',
                'frame_num': 81,
                'storage_bucket': 'video_fast',
                'expected_time': 120
            }
        }
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')

        # Check if Wan 2.1 needs fixing
        self.check_wan_installation()
        
        print("🔧 Simple worker ready!")

    def check_wan_installation(self):
        """Check if Wan 2.1 installation needs fixing"""
        print("\n🔍 Checking Wan 2.1 installation...")
        
        # Check if key files exist
        wan_exists = os.path.exists(self.wan_path)
        generate_exists = os.path.exists(os.path.join(self.wan_path, "generate.py"))
        
        print(f"   Wan 2.1 directory: {wan_exists}")
        print(f"   generate.py: {generate_exists}")
        
        if not wan_exists or not generate_exists:
            print("❌ Wan 2.1 installation incomplete - will need manual fix")
        else:
            print("✅ Wan 2.1 appears to be installed")

    def log_gpu_memory(self, context=""):
        """Log GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - allocated
            
            status = "🔥" if allocated > 0.1 else "❄️"
            
            print(f"{status} GPU {context}: {allocated:.2f}GB used, {free:.2f}GB free / {total:.1f}GB total")

    def generate_with_diagnosis(self, prompt, job_type):
        """Generate with full error diagnosis"""
        config = self.job_type_mapping[job_type]
        job_id = str(uuid.uuid4())[:8]
        
        print(f"\n🎬 === GENERATION DIAGNOSIS ===")
        print(f"📝 Job Type: {job_type}")
        print(f"📝 Prompt: {prompt}")
        print(f"📁 Wan Path: {self.wan_path}")
        print(f"📁 Model Path: {self.model_path}")
        
        # Create temp directories
        temp_base = Path("/tmp/ourvidz")
        temp_base.mkdir(exist_ok=True)
        temp_processing = temp_base / "processing"
        temp_processing.mkdir(exist_ok=True)
        
        # Output file
        temp_video_filename = f"{job_type}_{job_id}.mp4"
        temp_video_path = temp_processing / temp_video_filename
        
        # Build command
        cmd = [
            "python", "generate.py",
            "--task", "t2v-1.3B",
            "--ckpt_dir", self.model_path,
            "--offload_model", "False",
            "--size", config['size'],
            "--sample_steps", str(config['sample_steps']),
            "--sample_guide_scale", str(config['sample_guide_scale']),
            "--frame_num", str(config['frame_num']),
            "--prompt", prompt,
            "--save_file", str(temp_video_path.absolute())
        ]
        
        print(f"📋 Command: {' '.join(cmd)}")
        
        # Environment
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'PYTHONUNBUFFERED': '1'
        })
        
        original_cwd = os.getcwd()
        
        try:
            os.chdir(self.wan_path)
            start_time = time.time()
            self.log_gpu_memory("before generation")
            
            print("🚀 Starting generation with full error capture...")
            
            # Run subprocess with complete output capture
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            generation_time = time.time() - start_time
            return_code = result.returncode
            
            print(f"⏱️ Generation completed in {generation_time:.1f}s")
            print(f"🔧 Return code: {return_code}")
            
            self.log_gpu_memory("after generation")
            
            # Print ALL output for diagnosis
            print("\n📋 === COMPLETE SUBPROCESS OUTPUT ===")
            
            if result.stdout:
                print("STDOUT:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        print(f"   {line}")
            else:
                print("STDOUT: (empty)")
            
            if result.stderr:
                print("\nSTDERR:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        print(f"   {line}")
            else:
                print("STDERR: (empty)")
            
            print("📋 === END OUTPUT ===\n")
            
            # Check if file was created
            if return_code == 0 and temp_video_path.exists():
                file_size = temp_video_path.stat().st_size / 1024
                print(f"✅ Output file created: {file_size:.0f}KB")
                return str(temp_video_path)
            else:
                print(f"❌ Generation failed")
                return None
                
        except subprocess.TimeoutExpired:
            print("❌ Generation timed out (5 minutes)")
            return None
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return None
        finally:
            os.chdir(original_cwd)

    def extract_frame_from_video(self, video_path, job_id, job_type):
        """Extract frame for image jobs"""
        temp_processing = Path("/tmp/ourvidz/processing")
        image_path = temp_processing / f"{job_type}_{job_id}.png"
        
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.save(str(image_path), "PNG", optimize=True, quality=95)
                
                # Clean up video file
                try:
                    os.remove(video_path)
                except:
                    pass
                    
                return str(image_path)
            else:
                print("❌ Failed to read frame from video")
                return None
        except Exception as e:
            print(f"❌ Frame extraction error: {e}")
            return None

    def upload_to_supabase(self, file_path, job_type, user_id, job_id):
        """Upload to Supabase storage"""
        if not os.path.exists(file_path):
            print(f"❌ File not found for upload: {file_path}")
            return None
            
        config = self.job_type_mapping[job_type]
        storage_bucket = config['storage_bucket']
        content_type = config['content_type']
        file_extension = config['file_extension']
        
        timestamp = int(time.time())
        filename = f"job_{job_id}_{timestamp}_{job_type}.{file_extension}"
        full_path = f"{storage_bucket}/{user_id}/{filename}"
        
        mime_type = 'image/png' if content_type == 'image' else 'video/mp4'
        
        print(f"📤 Uploading to Supabase: {storage_bucket}/{filename}")
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
                file_size = len(file_data) / 1024
                
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{full_path}",
                    data=file_data,
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                        'Content-Type': mime_type,
                        'x-upsert': 'true'
                    },
                    timeout=60
                )
                
                if response.status_code in [200, 201]:
                    print(f"✅ Upload successful: {file_size:.0f}KB")
                    return f"{user_id}/{filename}"
                else:
                    print(f"❌ Upload failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return None
        finally:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass

    def notify_completion(self, job_id, status, file_path=None, error_message=None):
        """Notify completion via callback"""
        data = {
            'jobId': job_id,
            'status': status,
            'filePath': file_path,
            'errorMessage': error_message
        }
        
        try:
            response = requests.post(
                f"{self.supabase_url}/functions/v1/job-callback",
                json=data,
                headers={
                    'Authorization': f"Bearer {self.supabase_service_key}",
                    'Content-Type': 'application/json'
                },
                timeout=15
            )
            
            if response.status_code == 200:
                print(f"✅ Callback sent: {status}")
            else:
                print(f"❌ Callback failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job(self, job_data):
        """Process job with diagnosis"""
        job_id = job_data.get('jobId')
        job_type = job_data.get('jobType')
        prompt = job_data.get('prompt')
        user_id = job_data.get('userId')
        
        if not all([job_id, job_type, user_id, prompt]):
            error_msg = "Missing required fields"
            print(f"❌ {error_msg}")
            self.notify_completion(job_id or 'unknown', 'failed', error_message=error_msg)
            return

        print(f"\n🎯 Processing job: {job_id} ({job_type})")
        print(f"👤 User: {user_id}")
        print(f"📝 Prompt: {prompt}")
        
        start_time = time.time()
        
        try:
            # Generate with full diagnosis
            output_path = self.generate_with_diagnosis(prompt, job_type)
            
            if output_path:
                # Handle image vs video
                config = self.job_type_mapping[job_type]
                if config['content_type'] == 'image':
                    output_path = self.extract_frame_from_video(output_path, job_id, job_type)
                
                if output_path:
                    # Upload to Supabase
                    supa_path = self.upload_to_supabase(output_path, job_type, user_id, job_id)
                    if supa_path:
                        duration = time.time() - start_time
                        print(f"🎉 Job completed successfully in {duration:.1f}s")
                        self.notify_completion(job_id, 'completed', supa_path)
                        return
            
            print("❌ Job failed")
            self.notify_completion(job_id, 'failed', error_message="Generation failed")
            
        except Exception as e:
            print(f"❌ Job processing error: {e}")
            self.notify_completion(job_id, 'failed', error_message=str(e))

    def poll_queue(self):
        """Poll Redis queue"""
        try:
            response = requests.get(
                f"{self.redis_url}/rpop/job_queue",
                headers={'Authorization': f"Bearer {self.redis_token}"},
                timeout=5
            )
            if response.status_code == 200 and response.json().get('result'):
                return json.loads(response.json()['result'])
        except Exception as e:
            print(f"❌ Poll error: {e}")
        return None

    def run(self):
        """Main loop"""
        print("\n🎬 SIMPLE WORKER READY!")
        print("🔍 Will provide complete error diagnosis for generation issues")
        print("⏳ Waiting for jobs...\n")
        
        job_count = 0
        
        while True:
            job = self.poll_queue()
            if job:
                job_count += 1
                print(f"🚀 === PROCESSING JOB #{job_count} ===")
                self.process_job(job)
                print("=" * 80)
            else:
                time.sleep(5)

if __name__ == "__main__":
    print("🚀 Starting OurVidz SIMPLE WORKER")
    
    # Verify environment
    required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        exit(1)
    
    try:
        worker = SimpleWorker()
        worker.run()
    except Exception as e:
        print(f"❌ Worker failed: {e}")
        exit(1)
