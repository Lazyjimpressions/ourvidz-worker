# worker.py - RTX 6000 ADA WITH COMPREHENSIVE GPU DEBUGGING
# ISSUE: GPU showing 0.00GB usage indicates models not loading to GPU
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

# Clean environment - no distributed training needed
for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
    if key in os.environ:
        del os.environ[key]

# GPU optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class DebugWorker:
    def __init__(self):
        print("🚀 OurVidz DEBUG WORKER - RTX 6000 ADA GPU INVESTIGATION")
        print("🔍 DEBUGGING: Why GPU shows 0.00GB usage during generation")
        
        # Comprehensive CUDA debugging
        self.debug_cuda_environment()
        
        # Paths
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_path = "/workspace/Wan2.1"
        
        # Worker state
        self.models_loaded = False
        self.wan_pipeline = None
        self.last_job_time = time.time()
        self.model_lock = threading.Lock()
        
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

        # Verify Wan 2.1 installation
        self.verify_wan_installation()
        
        print("🔧 Debug worker ready for GPU investigation!")

    def debug_cuda_environment(self):
        """Comprehensive CUDA environment debugging"""
        print("\n🔍 === CUDA ENVIRONMENT DEBUG ===")
        
        # Basic CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA Available: {cuda_available}")
        
        if not cuda_available:
            print("❌ CUDA not available - this is the problem!")
            return
            
        # GPU details
        device_count = torch.cuda.device_count()
        print(f"   GPU Count: {device_count}")
        
        if device_count > 0:
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Current device
        current_device = torch.cuda.current_device()
        print(f"   Current Device: {current_device}")
        
        # Force device selection and test
        try:
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            
            # Test tensor creation on GPU
            test_tensor = torch.rand(1000, 1000, device='cuda')
            print(f"   GPU Test Tensor: {test_tensor.device} ✅")
            del test_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   GPU Test Failed: {e} ❌")
        
        # Environment variables
        print(f"   CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
        print(f"   PYTORCH_CUDA_ALLOC_CONF: {os.getenv('PYTORCH_CUDA_ALLOC_CONF')}")
        
        print("🔍 === END CUDA DEBUG ===\n")

    def verify_wan_installation(self):
        """Verify Wan 2.1 installation and model files"""
        print("\n🔍 === WAN 2.1 INSTALLATION DEBUG ===")
        
        # Check Wan 2.1 repository
        wan_repo_exists = os.path.exists(self.wan_path)
        print(f"   Wan 2.1 Repository: {wan_repo_exists}")
        
        if wan_repo_exists:
            generate_script = os.path.join(self.wan_path, "generate.py")
            print(f"   generate.py exists: {os.path.exists(generate_script)}")
            
            # Check for key files
            key_files = [
                "generate.py", 
                "wan/__init__.py",
                "wan/model.py",
                "wan/pipeline.py"
            ]
            
            for file in key_files:
                full_path = os.path.join(self.wan_path, file)
                exists = os.path.exists(full_path)
                print(f"   {file}: {exists}")
        
        # Check model files
        model_exists = os.path.exists(self.model_path)
        print(f"   Model Directory: {model_exists}")
        
        if model_exists:
            model_files = list(Path(self.model_path).rglob("*"))
            total_size = sum(f.stat().st_size for f in model_files if f.is_file()) / (1024**3)
            print(f"   Model Files Count: {len(model_files)}")
            print(f"   Total Model Size: {total_size:.2f}GB")
            
            # Check for critical model files
            critical_files = [
                "diffusion_pytorch_model.safetensors",
                "model_index.json",
                "scheduler/scheduler_config.json"
            ]
            
            for file in critical_files:
                full_path = os.path.join(self.model_path, file)
                exists = os.path.exists(full_path)
                if exists:
                    size_mb = os.path.getsize(full_path) / (1024**2)
                    print(f"   {file}: ✅ ({size_mb:.1f}MB)")
                else:
                    print(f"   {file}: ❌ MISSING")
        
        # Test Python import
        try:
            original_cwd = os.getcwd()
            os.chdir(self.wan_path)
            
            # Try importing Wan modules
            import sys
            sys.path.insert(0, self.wan_path)
            
            try:
                import wan
                print("   Wan module import: ✅")
            except Exception as e:
                print(f"   Wan module import: ❌ {e}")
            
            try:
                from wan.pipeline import WanVideoPipeline
                print("   WanVideoPipeline import: ✅")
            except Exception as e:
                print(f"   WanVideoPipeline import: ❌ {e}")
            
        except Exception as e:
            print(f"   Import test failed: {e}")
        finally:
            os.chdir(original_cwd)
        
        print("🔍 === END WAN DEBUG ===\n")

    def log_gpu_memory(self, context=""):
        """Enhanced GPU memory logging with debugging"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - allocated
            
            status = "🔥" if allocated > 0.1 else "❄️"  # Flag if >100MB allocated
            models_status = f"GPU:{allocated:.2f}GB" if allocated > 0.1 else "NO GPU USAGE"
            
            print(f"{status} GPU {context} ({models_status}): {allocated:.2f}GB used, {free:.2f}GB free / {total:.1f}GB total")
            
            # Alert if no GPU usage during generation
            if context.startswith("after") and allocated < 0.1:
                print("🚨 WARNING: No GPU memory usage detected - models may not be loading to GPU!")

    def test_gpu_tensor_operation(self):
        """Test basic GPU tensor operations"""
        print("\n🔍 === GPU TENSOR TEST ===")
        try:
            # Create tensor on GPU
            test_tensor = torch.rand(1000, 1000, device='cuda')
            self.log_gpu_memory("with test tensor")
            
            # Perform operation
            result = torch.matmul(test_tensor, test_tensor)
            print(f"   Matrix multiplication result shape: {result.shape}")
            
            # Clean up
            del test_tensor, result
            torch.cuda.empty_cache()
            self.log_gpu_memory("after cleanup")
            print("   GPU tensor test: ✅")
            
        except Exception as e:
            print(f"   GPU tensor test: ❌ {e}")
        print("🔍 === END GPU TENSOR TEST ===\n")

    def generate_with_debug_subprocess(self, prompt, job_type):
        """Enhanced subprocess generation with GPU debugging"""
        config = self.job_type_mapping[job_type]
        job_id = str(uuid.uuid4())[:8]
        
        print(f"\n🔍 === SUBPROCESS GENERATION DEBUG ===")
        print(f"📝 Job Type: {job_type}")
        print(f"📝 Prompt: {prompt}")
        print(f"📁 Model Path: {self.model_path}")
        print(f"📁 Wan Path: {self.wan_path}")
        
        # Pre-generation GPU test
        self.test_gpu_tensor_operation()
        
        # Create temp directories
        temp_base = Path("/tmp/ourvidz")
        temp_base.mkdir(exist_ok=True)
        temp_processing = temp_base / "processing"
        temp_processing.mkdir(exist_ok=True)
        
        # Output file
        temp_video_filename = f"{job_type}_{job_id}.mp4"
        temp_video_path = temp_processing / temp_video_filename
        
        # Build command with debugging
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
            self.log_gpu_memory("before subprocess")
            
            print("🚀 Starting subprocess generation...")
            print("   (Watching for GPU memory usage...)")
            
            # Start subprocess with real-time output
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
                # Check GPU usage every 5 seconds
                current_allocated = torch.cuda.memory_allocated() / (1024**3)
                if current_allocated > 0.1 and not gpu_usage_detected:
                    print(f"🔥 GPU USAGE DETECTED: {current_allocated:.2f}GB allocated!")
                    gpu_usage_detected = True
                
                # Read any available output
                try:
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        output_lines.append(stdout_line.strip())
                        print(f"   STDOUT: {stdout_line.strip()}")
                        
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        error_lines.append(stderr_line.strip())
                        print(f"   STDERR: {stderr_line.strip()}")
                        
                except:
                    pass
                
                time.sleep(1)  # Check every second
            
            # Get final output
            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                output_lines.extend(remaining_stdout.strip().split('\n'))
            if remaining_stderr:
                error_lines.extend(remaining_stderr.strip().split('\n'))
            
            generation_time = time.time() - start_time
            return_code = process.returncode
            
            print(f"⏱️ Subprocess completed in {generation_time:.1f}s")
            print(f"🔧 Return code: {return_code}")
            
            self.log_gpu_memory("after subprocess")
            
            if not gpu_usage_detected:
                print("🚨 CRITICAL: NO GPU USAGE DETECTED DURING GENERATION!")
                print("   This indicates the model is not loading to GPU properly")
            
            # Print subprocess output for debugging
            if output_lines:
                print("\n📋 SUBPROCESS STDOUT:")
                for line in output_lines[-10:]:  # Last 10 lines
                    if line.strip():
                        print(f"   {line}")
            
            if error_lines:
                print("\n📋 SUBPROCESS STDERR:")
                for line in error_lines[-10:]:  # Last 10 lines
                    if line.strip():
                        print(f"   {line}")
            
            # Check if generation succeeded
            if return_code == 0 and temp_video_path.exists():
                file_size = temp_video_path.stat().st_size / 1024
                print(f"✅ Generation succeeded: {file_size:.0f}KB")
                print(f"🔍 === END SUBPROCESS DEBUG ===\n")
                return str(temp_video_path)
            else:
                print(f"❌ Generation failed: return code {return_code}")
                if not temp_video_path.exists():
                    print("❌ Output file not created")
                print(f"🔍 === END SUBPROCESS DEBUG ===\n")
                return None
                
        except Exception as e:
            print(f"❌ Subprocess error: {e}")
            print(f"🔍 === END SUBPROCESS DEBUG ===\n")
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
        """Process job with comprehensive debugging"""
        job_id = job_data.get('jobId')
        job_type = job_data.get('jobType')
        prompt = job_data.get('prompt')
        user_id = job_data.get('userId')
        
        if not all([job_id, job_type, user_id, prompt]):
            error_msg = "Missing required fields"
            print(f"❌ {error_msg}")
            self.notify_completion(job_id or 'unknown', 'failed', error_message=error_msg)
            return

        print(f"\n🎯 === JOB PROCESSING DEBUG ===")
        print(f"📥 Job ID: {job_id}")
        print(f"📥 Job Type: {job_type}")
        print(f"👤 User: {user_id}")
        print(f"📝 Prompt: {prompt}")
        
        start_time = time.time()
        
        try:
            # Generate with debugging
            output_path = self.generate_with_debug_subprocess(prompt, job_type)
            
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
        finally:
            print(f"🎯 === END JOB PROCESSING ===\n")

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
        """Main loop with comprehensive debugging"""
        print("\n🔍 DEBUG WORKER READY!")
        print("🎯 Primary Goal: Identify why GPU shows 0.00GB usage")
        print("🔧 Will monitor GPU memory during each generation")
        print("📋 Will capture subprocess output for analysis")
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
    print("🚀 Starting OurVidz DEBUG WORKER - GPU Investigation Mode")
    
    # Verify environment
    required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        exit(1)
    
    try:
        worker = DebugWorker()
        worker.run()
    except Exception as e:
        print(f"❌ Worker failed: {e}")
        exit(1)
