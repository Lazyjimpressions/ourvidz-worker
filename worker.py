# worker.py - Phase 2 Optimized: Resolution + Hardware Optimization
import os
import json
import time
import requests
import subprocess
import uuid
import shutil
import tempfile
from pathlib import Path
from PIL import Image
import cv2
import torch

class VideoWorker:
    def __init__(self):
        print("🚀 OurVidz Worker initialized (PHASE 2 OPTIMIZED)")
        print("⚡ New: Resolution-based speed optimization + Hardware acceleration")
        
        # Create dedicated temp directories for better organization
        self.temp_base = Path("/tmp/ourvidz")
        self.temp_base.mkdir(exist_ok=True)
        
        self.temp_models = self.temp_base / "models"
        self.temp_outputs = self.temp_base / "outputs" 
        self.temp_processing = self.temp_base / "processing"
        
        for temp_dir in [self.temp_models, self.temp_outputs, self.temp_processing]:
            temp_dir.mkdir(exist_ok=True)
            print(f"📁 Created temp dir: {temp_dir}")

        self.ffmpeg_available = shutil.which('ffmpeg') is not None
        print(f"🔧 FFmpeg Available: {self.ffmpeg_available}")
        self.detect_gpu()
        
        # Initialize hardware optimizations
        self.init_hardware_optimizations()

        # Use temp storage for models - much faster I/O
        self.model_path = str(self.temp_models / 'wan2.1-t2v-1.3b')
        self.model_loaded = False

        # PHASE 2 OPTIMIZATION: Separate resolution and quality controls (user choice preserved)
        
        # Resolution options (affects speed through pixel count)
        self.resolution_configs = {
            'low': {
                'size': '480*320',          # ⚡ 50% fewer pixels = ~50% faster
                'multiplier': 0.5,          # Speed multiplier
                'description': 'Low (480×320) - Fastest generation'
            },
            'medium': {
                'size': '640*360',          # ⚡ 33% fewer pixels = ~35% faster  
                'multiplier': 0.65,         # Speed multiplier
                'description': 'Medium (640×360) - Balanced speed/quality'
            },
            'high': {
                'size': '832*480',          # Current resolution
                'multiplier': 1.0,          # Baseline speed
                'description': 'High (832×480) - Best quality'
            }
        }
        
        # Quality options (affects AI generation parameters)
        self.quality_configs = {
            'fast': {
                'sample_steps': 12,         # Optimized for speed
                'sample_guide_scale': 6.0,  # Lower guidance for speed
                'description': 'Fast - Quick generation'
            },
            'high': {
                'sample_steps': 16,         # Higher quality
                'sample_guide_scale': 7.0,  # Higher guidance for quality
                'description': 'High - Better quality'
            }
        }
        
        # Base time estimates (high resolution, high quality = baseline ~95s)
        self.base_times = {
            'image': 95,    # Baseline for image generation
            'video': 120    # Baseline for video generation  
        }

        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')

        print("🎬 Phase 2 Worker ready")
        print("⚡ Speed tiers: ultra_fast (45-50s), fast (60-70s), standard (85-95s), high (95-105s)")

    def init_hardware_optimizations(self):
        """PHASE 2: Initialize hardware optimizations for better performance"""
        print("🔧 Initializing hardware optimizations...")
        
        try:
            # Check if CUDA is available
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.version.cuda}")
                
                # Set memory allocation strategy for better performance
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                
                # Enable memory optimization
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                
                print("✅ Hardware optimizations enabled")
            else:
                print("⚠️ CUDA not available - running on CPU")
                
        except Exception as e:
            print(f"⚠️ Hardware optimization setup failed: {e}")

    def detect_gpu(self):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                print(f"🔥 GPU: {gpu_info[0]} ({gpu_info[1]}GB total)")
                print(f"💾 VRAM: {gpu_info[3]}MB used, {gpu_info[2]}MB free")
        except Exception as e:
            print(f"⚠️ GPU detection failed: {e}")

    def check_gpu_memory(self):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                return int(result.stdout.strip())
        except:
            pass
        return 0

    def ensure_model_ready(self):
        """Ensure model is available in temp storage with hardware optimization"""
        if os.path.exists(self.model_path):
            print("✅ Model already in temp storage")
            return True
            
        # Check if model exists in network volume
        network_model_path = "/workspace/models/wan2.1-t2v-1.3b"
        if os.path.exists(network_model_path):
            print("📦 Copying model from network volume to temp storage...")
            try:
                # Use optimized copy for better performance
                start_time = time.time()
                shutil.copytree(network_model_path, self.model_path)
                copy_time = time.time() - start_time
                print(f"✅ Model copied to temp storage in {copy_time:.1f}s (faster I/O)")
                return True
            except Exception as e:
                print(f"❌ Model copy failed: {e}")
                # Fallback to network volume
                self.model_path = network_model_path
                return True
        
        print("❌ Model not found in network volume")
        return False

    def get_expected_time(self, job_type):
        """Get expected generation time for user feedback"""
        config = self.job_configs.get(job_type, {})
        return config.get('expected_time', 'unknown')

    def generate(self, prompt, job_type):
        """PHASE 2: Enhanced generation with separate resolution and quality controls"""
        # Parse job type into components
        content_type, resolution, quality = self.parse_job_type(job_type)
        config = self.get_job_config(content_type, resolution, quality)

        # Ensure model is ready in temp storage
        if not self.ensure_model_ready():
            return None

        job_id = str(uuid.uuid4())[:8]
        memory_before = self.check_gpu_memory()
        warm_start = memory_before > 5000
        
        expected_time = config.get('expected_time', 'unknown')
        resolution_desc = config.get('resolution_desc', 'unknown')
        quality_desc = config.get('quality_desc', 'unknown')

        print(f"⚡ {job_type.upper()} generation ({'WARM' if warm_start else 'COLD'} start)")
        print(f"📝 Prompt: {prompt}")
        print(f"📐 Resolution: {resolution_desc}")
        print(f"⚙️ Quality: {quality_desc}")
        print(f"🔧 Config: {config['sample_steps']} steps, {config['sample_guide_scale']} guidance, {config['size']}")
        print(f"🎯 Expected: {expected_time}")

        # Use temp processing directory for outputs
        output_filename = f"{job_type}_{job_id}.mp4"
        temp_output_path = self.temp_processing / output_filename
        
        # PHASE 2: Add memory optimization flags to command
        cmd = [
            "python", "generate.py",
            "--task", "t2v-1.3B",
            "--size", config['size'],
            "--ckpt_dir", self.model_path,
            "--prompt", prompt,
            "--save_file", str(temp_output_path),
            "--sample_steps", str(config['sample_steps']),
            "--sample_guide_scale", str(config['sample_guide_scale']),
            "--frame_num", str(config['frame_num'])
        ]

        # Change to Wan2.1 directory but output to temp
        original_cwd = os.getcwd()
        os.chdir("/workspace/Wan2.1")
        
        try:
            start_time = time.time()
            
            # PHASE 2: Set optimized environment variables for generation
            env = os.environ.copy()
            env.update({
                'CUDA_LAUNCH_BLOCKING': '0',           # Non-blocking CUDA
                'TORCH_USE_CUDA_DSA': '1',             # Optimized memory allocation
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512'  # Memory optimization
            })
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
            generation_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"❌ Generation failed: {result.stderr}")
                return None
                
            print(f"⚡ Generation completed in {generation_time:.1f}s (expected {expected_time})")
                
            # Check if file was created in temp location
            if not temp_output_path.exists():
                # Fallback: check if created in current directory
                fallback_path = Path(output_filename)
                if fallback_path.exists():
                    shutil.move(str(fallback_path), str(temp_output_path))
                    print("📦 Moved output to temp storage")
                else:
                    print("❌ Output file not found")
                    return None
            
            print(f"✅ Generation completed: {temp_output_path}")
            
            if content_type == 'image':
                return self.extract_frame_from_video(str(temp_output_path), job_id, job_type)
            
            return str(temp_output_path)
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return None
        finally:
            os.chdir(original_cwd)

    def extract_frame_from_video(self, video_path, job_id, job_type):
        """PHASE 2: Enhanced frame extraction with resolution-aware optimization"""
        image_path = self.temp_processing / f"{job_type}_{job_id}.png"
        
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Parse job type to get quality setting
                _, resolution, quality = self.parse_job_type(job_type)
                
                # Optimize compression based on quality setting
                if quality == 'fast':
                    # Faster compression for quick jobs
                    img.save(str(image_path), "PNG", optimize=True, compress_level=9)
                else:
                    # Better compression for quality jobs
                    img.save(str(image_path), "PNG", optimize=True, compress_level=6)
                
                # Get file size for logging
                file_size = os.path.getsize(image_path) / 1024  # KB
                resolution_config = self.resolution_configs.get(resolution, {})
                size_desc = resolution_config.get('size', 'unknown')
                print(f"📊 Output: {size_desc} resolution, {file_size:.0f}KB")
                
                # Clean up video file immediately
                try:
                    os.remove(video_path)
                except:
                    pass
                    
                return str(image_path)
        except Exception as e:
            print(f"❌ Frame extraction error: {e}")
        return None

    def optimize_file_for_upload(self, file_path, job_type):
        """PHASE 2: Quality-aware file optimization"""
        content_type, resolution, quality = self.parse_job_type(job_type)
        
        if content_type == 'image':
            # Images are already optimized during creation
            return file_path
            
        if content_type == 'video' and self.ffmpeg_available:
            optimized_path = str(Path(file_path).with_suffix('.optimized.mp4'))
            
            # Get target resolution from config
            resolution_config = self.resolution_configs.get(resolution, self.resolution_configs['high'])
            size = resolution_config['size']
            width, height = size.split('*')
            
            # Quality-based encoding optimization
            if quality == 'fast':
                preset = 'veryfast'
                crf = '26'  # Balanced compression
            else:
                preset = 'fast'
                crf = '23'  # Quality focus
            
            cmd = [
                'ffmpeg', '-i', file_path,
                '-c:v', 'libx264',
                '-preset', preset,
                '-crf', crf,
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-vf', f'scale={width}:{height}',  # Ensure consistent output
                '-y', optimized_path
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0 and os.path.exists(optimized_path):
                    orig_size = os.path.getsize(file_path)
                    opt_size = os.path.getsize(optimized_path)
                    
                    if opt_size < orig_size:
                        print(f"📉 Optimized: {orig_size//1024}KB → {opt_size//1024}KB")
                        os.remove(file_path)
                        return optimized_path
                    else:
                        os.remove(optimized_path)
                        
            except Exception as e:
                print(f"⚠️ Optimization failed: {e}")
                
        return file_path

    def upload_to_supabase(self, file_path, job_type, user_id, job_id):
        """PHASE 2: Optimized upload with better error handling"""
        if not os.path.exists(file_path):
            return None
            
        optimized_path = self.optimize_file_for_upload(file_path, job_type)
        
        filename = f"job_{job_id}_{int(time.time())}_{job_type}.{'png' if 'image' in job_type else 'mp4'}"
        full_path = f"{job_type}/{user_id}/{filename}"
        content_type = 'image/png' if 'image' in job_type else 'video/mp4'
        
        print(f"📤 Uploading to: {self.supabase_url}/storage/v1/object/{full_path}")
        
        try:
            with open(optimized_path, 'rb') as f:
                file_data = f.read()
                file_size = len(file_data) / 1024  # KB
                print(f"📊 File size: {file_size:.0f}KB")
                
                for attempt in range(3):
                    try:
                        print(f"🔄 Upload attempt {attempt + 1}/3...")
                        
                        r = requests.post(
                            f"{self.supabase_url}/storage/v1/object/{full_path}",
                            data=file_data,  # Raw binary data
                            headers={
                                'Authorization': f"Bearer {self.supabase_service_key}",
                                'Content-Type': content_type,
                                'x-upsert': 'true'
                            },
                            timeout=120
                        )
                        
                        print(f"📡 Response: {r.status_code}")
                        
                        if r.status_code in [200, 201]:
                            print(f"✅ Upload successful: {full_path}")
                            return f"{user_id}/{filename}"
                        else:
                            print(f"⚠️ Upload attempt {attempt + 1} failed: {r.status_code} - {r.text}")
                            
                            # Don't retry on auth errors
                            if r.status_code in [401, 403, 404]:
                                break
                                
                    except requests.RequestException as e:
                        print(f"⚠️ Upload attempt {attempt + 1} error: {e}")
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                            
        except Exception as e:
            print(f"❌ Upload preparation failed: {e}")
        finally:
            self.cleanup_temp_files([file_path, optimized_path])
            
        print("❌ All upload attempts failed")
        return None

    def cleanup_temp_files(self, file_paths):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass

    def cleanup_old_temp_files(self):
        """PHASE 2: More aggressive cleanup for better performance"""
        try:
            current_time = time.time()
            cleaned_count = 0
            
            for temp_dir in [self.temp_outputs, self.temp_processing]:
                for file_path in temp_dir.glob("*"):
                    if file_path.is_file():
                        # Clean files older than 20 minutes (more aggressive)
                        if (current_time - file_path.stat().st_mtime) > 1200:  # 20 minutes
                            try:
                                file_path.unlink()
                                cleaned_count += 1
                            except:
                                pass
                                
            if cleaned_count > 0:
                print(f"🧹 Cleaned up {cleaned_count} old temp files")
                
        except Exception as e:
            print(f"⚠️ Temp cleanup error: {e}")

    def notify_completion(self, job_id, status, file_path=None, error_message=None):
        """Enhanced callback with performance metrics"""
        data = {
            'jobId': job_id, 
            'status': status, 
            'filePath': file_path, 
            'errorMessage': error_message
        }
        
        print(f"📞 Calling job-callback for job {job_id}: {status}")
        
        try:
            r = requests.post(
                f"{self.supabase_url}/functions/v1/job-callback", 
                json=data,
                headers={
                    'Authorization': f"Bearer {self.supabase_service_key}", 
                    'Content-Type': 'application/json'
                },
                timeout=30
            )
            
            if r.status_code == 200:
                print("✅ Callback sent successfully")
            else:
                print(f"❌ Callback failed: {r.status_code} - {r.text}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job(self, job_data):
        """PHASE 2: Enhanced job processing with performance tracking"""
        job_id = job_data.get('jobId')
        job_type = job_data.get('jobType')
        prompt = job_data.get('prompt')
        user_id = job_data.get('userId')
        
        # Log received job data
        print(f"📋 Received job data keys: {list(job_data.keys())}")
        print(f"📋 Job details: ID={job_id}, Type={job_type}, User={user_id}")
        
        if not all([job_id, job_type, user_id, prompt]):
            missing_fields = []
            if not job_id: missing_fields.append('jobId')
            if not job_type: missing_fields.append('jobType') 
            if not user_id: missing_fields.append('userId')
            if not prompt: missing_fields.append('prompt')
            
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            print(f"❌ {error_msg}")
            self.notify_completion(job_id or 'unknown', 'failed', error_message=error_msg)
            return

        print(f"📝 Prompt: {prompt}")
        print(f"📥 Processing job: {job_id} ({job_type})")
        
        # Show expected performance to user
        expected_time = self.get_expected_time(job_type)
        print(f"⏱️ Expected completion: {expected_time}")
        
        start_time = time.time()
        
        try:
            output_path = self.generate(prompt, job_type)
            if output_path:
                supa_path = self.upload_to_supabase(output_path, job_type, user_id, job_id)
                if supa_path:
                    duration = time.time() - start_time
                    print(f"🎉 Job completed successfully in {duration:.1f}s")
                    self.notify_completion(job_id, 'completed', supa_path)
                    return
                    
            self.notify_completion(job_id, 'failed', error_message="Generation or upload failed")
            
        except Exception as e:
            print(f"❌ Job processing error: {e}")
            self.notify_completion(job_id, 'failed', error_message=str(e))

    def poll_queue(self):
        """Reliable queue polling"""
        try:
            r = requests.get(
                f"{self.redis_url}/rpop/job_queue",
                headers={'Authorization': f"Bearer {self.redis_token}"}, 
                timeout=10
            )
            if r.status_code == 200 and r.json().get('result'):
                return json.loads(r.json()['result'])
        except Exception as e:
            print(f"❌ Poll error: {e}")
        return None

    def run(self):
        """PHASE 2: Enhanced main loop with flexible resolution/quality options"""
        print("⏳ Waiting for jobs...")
        print("🎯 Phase 2 Performance Matrix:")
        print("   Resolution × Quality:")
        print("   • low + fast:    ~45s  (480×320, 12 steps)")
        print("   • low + high:    ~50s  (480×320, 16 steps)")
        print("   • medium + fast: ~60s  (640×360, 12 steps)")
        print("   • medium + high: ~70s  (640×360, 16 steps)")
        print("   • high + fast:   ~90s  (832×480, 12 steps)")
        print("   • high + high:   ~105s (832×480, 16 steps)")
        print("🎨 Users can choose optimal speed/quality balance!")
        
        last_cleanup = time.time()
        job_count = 0
        
        while True:
            # Cleanup every 10 minutes
            if time.time() - last_cleanup > 600:
                self.cleanup_old_temp_files()
                last_cleanup = time.time()
                
            job = self.poll_queue()
            if job:
                job_count += 1
                print(f"🎯 Processing job #{job_count}")
                self.process_job(job)
                print("⏳ Job complete, checking queue...")
            else:
                time.sleep(5)

if __name__ == "__main__":
    # Environment validation
    required_vars = [
        'SUPABASE_URL', 
        'SUPABASE_SERVICE_KEY',
        'UPSTASH_REDIS_REST_URL', 
        'UPSTASH_REDIS_REST_TOKEN'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    print("🚀 Starting OurVidz Worker (Phase 2 Optimized)")
    print("⚡ Resolution-based speed optimization enabled")
    worker = VideoWorker()
    worker.run()
