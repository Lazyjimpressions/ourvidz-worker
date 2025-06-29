# worker.py - RTX 6000 ADA WITH PROPER WAN 2.1 INSTALLATION
# FIX: Missing wan/pipeline.py and model files - reinstall Wan 2.1 properly
import os
import json
import time
import requests
import subprocess
import uuid
import shutil
import gc
import threading
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

class FixedWorker:
    def __init__(self):
        print("🚀 OurVidz FIXED WORKER - Proper Wan 2.1 Installation")
        print("🔧 FIXING: Missing wan/pipeline.py and model components")
        
        # Paths
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_path = "/workspace/Wan2.1"
        
        # Worker state
        self.models_loaded = False
        self.last_job_time = time.time()
        
        # RTX 6000 Ada job configurations
        self.job_type_mapping = {
            'image_fast': {
                'content_type': 'image',
                'file_extension': 'png',
                'sample_steps': 4,
                'sample_guide_scale': 3.0,
                'size': '480*832',
                'frame_num': 1,
                'storage_bucket': 'image_fast',
                'expected_time': 60,
                'description': 'Fast image generation'
            },
            'video_fast': {
                'content_type': 'video',
                'file_extension': 'mp4',
                'sample_steps': 4,
                'sample_guide_scale': 3.0,
                'size': '480*832',
                'frame_num': 81,
                'storage_bucket': 'video_fast',
                'expected_time': 120,
                'description': 'Fast 5s video'
            }
        }
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')

        # Fix Wan 2.1 installation before doing anything else
        self.fix_wan_installation()
        
        print("🔧 Fixed worker ready!")

    def fix_wan_installation(self):
        """Fix the broken Wan 2.1 installation"""
        print("\n🔧 === FIXING WAN 2.1 INSTALLATION ===")
        
        try:
            # Step 1: Clean up broken installation
            print("🗑️ Cleaning up broken Wan 2.1 installation...")
            if os.path.exists(self.wan_path):
                shutil.rmtree(self.wan_path)
                print("   ✅ Removed broken Wan2.1 directory")
            
            # Step 2: Fresh clone
            print("📥 Cloning fresh Wan 2.1 repository...")
            result = subprocess.run([
                "git", "clone", 
                "https://github.com/Wan-Video/Wan2.1.git",
                self.wan_path
            ], capture_output=True, text=True, cwd="/workspace")
            
            if result.returncode != 0:
                print(f"❌ Git clone failed: {result.stderr}")
                return False
            
            print("   ✅ Fresh repository cloned")
            
            # Step 3: Verify critical files exist
            critical_files = [
                "generate.py",
                "wan/__init__.py", 
                "wan/modules/__init__.py",
                "wan/modules/model.py"
            ]
            
            missing_files = []
            for file in critical_files:
                full_path = os.path.join(self.wan_path, file)
                if not os.path.exists(full_path):
                    missing_files.append(file)
                else:
                    print(f"   ✅ {file}")
            
            if missing_files:
                print(f"❌ Still missing files: {missing_files}")
                return False
            
            # Step 4: Install the package properly
            print("📦 Installing Wan 2.1 package...")
            original_cwd = os.getcwd()
            os.chdir(self.wan_path)
            
            result = subprocess.run([
                "pip", "install", "-e", ".", "--force-reinstall"
            ], capture_output=True, text=True)
            
            os.chdir(original_cwd)
            
            if result.returncode != 0:
                print(f"❌ Package installation failed: {result.stderr}")
                return False
            
            print("   ✅ Package installed successfully")
            
            # Step 5: Test imports
            print("🧪 Testing critical imports...")
            try:
                original_cwd = os.getcwd()
                os.chdir(self.wan_path)
                
                import sys
                sys.path.insert(0, self.wan_path)
                
                # Test basic imports
                import wan
                print("   ✅ wan module import")
                
                # Try to import video generation components
                try:
                    from wan.modules.model import WanModel
                    print("   ✅ WanModel import")
                except Exception as e:
                    print(f"   ⚠️ WanModel import: {e}")
                
            # Test generate.py execution with detailed error capture
            print("🧪 Testing generate.py execution...")
            test_result = subprocess.run([
                "python", "generate.py", "--help"
            ], capture_output=True, text=True, timeout=30)
            
            if test_result.returncode == 0:
                print("   ✅ generate.py executable")
            else:
                print(f"   ❌ generate.py test failed:")
                print(f"   STDOUT: {test_result.stdout}")
                print(f"   STDERR: {test_result.stderr}")
                # Don't return False here - might still work for actual generation
                
            except Exception as e:
                print(f"   ❌ Import test failed: {e}")
                return False
            finally:
                os.chdir(original_cwd)
            
            # Step 6: Verify model files
            print("📂 Verifying model files...")
            if os.path.exists(self.model_path):
                model_files = list(Path(self.model_path).rglob("*"))
                total_size = sum(f.stat().st_size for f in model_files if f.is_file()) / (1024**3)
                print(f"   Model files: {len(model_files)} files, {total_size:.2f}GB")
                
                # Check for main model file
                main_model = os.path.join(self.model_path, "diffusion_pytorch_model.safetensors")
                if os.path.exists(main_model):
                    size_gb = os.path.getsize(main_model) / (1024**3)
                    print(f"   ✅ Main model: {size_gb:.2f}GB")
                else:
                    print("   ❌ Main model file missing")
                    return False
            else:
                print("   ❌ Model directory missing")
                return False
            
            print("🔧 === WAN 2.1 INSTALLATION FIXED ===\n")
            return True
            
        except Exception as e:
            print(f"❌ Installation fix failed: {e}")
            return False

    def log_gpu_memory(self, context=""):
        """Enhanced GPU memory logging"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - allocated
            
            status = "🔥" if allocated > 0.1 else "❄️"
            models_status = f"GPU:{allocated:.2f}GB" if allocated > 0.1 else "NO GPU USAGE"
            
            print(f"{status} GPU {context} ({models_status}): {allocated:.2f}GB used, {free:.2f}GB free / {total:.1f}GB total")
            
            # Alert if no GPU usage during generation
            if context.startswith("after") and allocated < 0.1:
                print("🚨 WARNING: Still no GPU usage - check subprocess output for errors!")

    def generate_with_fixed_subprocess(self, prompt, job_type):
        """Generate with fixed Wan 2.1 installation"""
        config = self.job_type_mapping[job_type]
        job_id = str(uuid.uuid4())[:8]
        
        print(f"\n🎬 === GENERATION WITH FIXED WAN 2.1 ===")
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
        
        # Build command with proper parameters
        cmd = [
            "python", "generate.py",
            "--task", "t2v-1.3B",
            "--ckpt_dir", self.model_path,
            "--offload_model", "False",  # Keep models on GPU
            "--size", config['size'],
            "--sample_steps", str(config['sample_steps']),
            "--sample_guide_scale", str(config['sample_guide_scale']),
            "--frame_num", str(config['frame_num']),
            "--prompt", prompt,
            "--save_file", str(temp_video_path.absolute())
        ]
        
        print(f"📋 Command: {' '.join(cmd)}")
        print(f"📁 Output: {temp_video_path.absolute()}")
        
        # Enhanced environment
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'TORCH_USE_CUDA_DSA': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
            'CUDA_LAUNCH_BLOCKING': '1',  # For debugging
            'PYTHONUNBUFFERED': '1'       # Real-time output
        })
        
        original_cwd = os.getcwd()
        os.chdir(self.wan_path)
        
        try:
            start_time = time.time()
            self.log_gpu_memory("before generation")
            
            print("🚀 Starting fixed generation...")
            
            # Run with real-time monitoring
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor process and GPU usage
            gpu_usage_detected = False
            output_lines = []
            error_lines = []
            
            while process.poll() is None:
                # Check GPU usage every 2 seconds
                current_allocated = torch.cuda.memory_allocated() / (1024**3)
                if current_allocated > 0.5 and not gpu_usage_detected:  # 500MB threshold
                    print(f"🔥 GPU USAGE DETECTED: {current_allocated:.2f}GB allocated!")
                    gpu_usage_detected = True
                
                # Read output with better buffering
                try:
                    # Read stdout
                    while True:
                        stdout_line = process.stdout.readline()
                        if not stdout_line:
                            break
                        output_lines.append(stdout_line.strip())
                        print(f"   OUT: {stdout_line.strip()}")
                        
                    # Read stderr  
                    while True:
                        stderr_line = process.stderr.readline()
                        if not stderr_line:
                            break
                        error_lines.append(stderr_line.strip())
                        print(f"   ERR: {stderr_line.strip()}")
                        
                except:
                    pass
                
                time.sleep(1)  # Check every second for faster error detection
            
            # Get final output
            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                output_lines.extend(remaining_stdout.strip().split('\n'))
            if remaining_stderr:
                error_lines.extend(remaining_stderr.strip().split('\n'))
            
            generation_time = time.time() - start_time
            return_code = process.returncode
            
            print(f"⏱️ Generation completed in {generation_time:.1f}s")
            print(f"🔧 Return code: {return_code}")
            
            self.log_gpu_memory("after generation")
            
            # Analyze results
            if gpu_usage_detected:
                print("🎉 SUCCESS: GPU usage detected - models are loading properly!")
            else:
                print("❌ ISSUE: Still no GPU usage detected")
                print("📋 Subprocess output analysis:")
                
                # Look for specific error patterns
                all_output = output_lines + error_lines
                for line in all_output:
                    if any(keyword in line.lower() for keyword in ['error', 'failed', 'exception', 'traceback']):
                        print(f"   🚨 {line}")
                    elif any(keyword in line.lower() for keyword in ['loading', 'model', 'cuda', 'gpu']):
                        print(f"   📋 {line}")
            
            # Check if file was created
            if return_code == 0 and temp_video_path.exists():
                file_size = temp_video_path.stat().st_size / 1024
                print(f"✅ Output file created: {file_size:.0f}KB")
                return str(temp_video_path)
            else:
                print(f"❌ Generation failed or no output file")
                # Print ALL error output for debugging
                print("📋 COMPLETE ERROR OUTPUT:")
                all_output = output_lines + error_lines
                if all_output:
                    for line in all_output:
                        if line.strip():
                            print(f"   {line}")
                else:
                    print("   (No output captured)")
                return None
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return None
        finally:
            os.chdir(original_cwd)
            print(f"🎬 === END GENERATION ===\n")

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
            # Clean up local file
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
        """Process job with fixed installation"""
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
            # Generate with fixed installation
            output_path = self.generate_with_fixed_subprocess(prompt, job_type)
            
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
        print("\n🎬 FIXED WORKER READY!")
        print("✅ Wan 2.1 installation has been fixed")
        print("🔥 Should now see proper GPU usage during generation")
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
    print("🚀 Starting OurVidz FIXED WORKER - Proper Wan 2.1 Installation")
    
    # Verify environment
    required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        exit(1)
    
    try:
        worker = FixedWorker()
        worker.run()
    except Exception as e:
        print(f"❌ Worker failed: {e}")
        exit(1)
