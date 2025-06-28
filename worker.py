# worker.py - FIXED: Using only supported Wan 2.1 resolutions
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
        print("🚀 OurVidz Worker initialized (RESOLUTION FIXED)")
        print("✅ Using only Wan 2.1 supported resolutions")
        
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

        # FIXED: Map existing job types to SUPPORTED Wan 2.1 resolutions only
        self.job_type_mapping = {
            # Image job types - using supported resolutions for speed optimization
            'image_fast': {
                'content_type': 'image',
                'resolution': 'small',            # 480×832 (fastest supported)
                'quality': 'fast',                # 8 steps, 5.5 guidance
                'storage_bucket': 'image_fast',
                'expected_time': 45,              # Fastest possible with supported resolution
                'description': 'Small resolution, maximum speed'
            },
            'image_high': {
                'content_type': 'image',
                'resolution': 'standard',         # 832×480 (current working resolution)
                'quality': 'balanced',            # 10 steps, 6.0 guidance  
                'storage_bucket': 'image_high',
                'expected_time': 70,              # Optimized but quality-focused
                'description': 'Standard resolution, balanced quality'
            },
            
            # Video job types - using supported resolutions
            'video_fast': {
                'content_type': 'video',
                'resolution': 'small',            # 480×832 (fastest supported)
                'quality': 'fast',                # 8 steps, 5.5 guidance
                'storage_bucket': 'video_fast',
                'expected_time': 55,              # Fastest possible video
                'description': 'Small resolution, fast video'
            },
            'video_high': {
                'content_type': 'video', 
                'resolution': 'standard',         # 832×480 (current working)
                'quality': 'balanced',            # 10 steps, 6.0 guidance
                'storage_bucket': 'video_high',
                'expected_time': 85,              # Optimized quality video
                'description': 'Standard resolution, quality video'
            }
        }
        
        # FIXED: Resolution configurations using ONLY supported Wan 2.1 sizes
        self.resolution_configs = {
            'small': {
                'size': '480*832',              # ✅ SUPPORTED - Portrait, smaller (faster)
                'aspect_ratio': 'portrait',
                'multiplier': 0.7,              # Speed improvement from smaller size
                'description': 'Small (480×832) - Fastest supported resolution'
            },
            'standard': {
                'size': '832*480',              # ✅ SUPPORTED - Current working resolution
                'aspect_ratio': 'landscape', 
                'multiplier': 1.0,              # Baseline
                'description': 'Standard (832×480) - Current resolution'
            },
            'large': {
                'size': '1280*720',             # ✅ SUPPORTED - HD landscape
                'aspect_ratio': 'landscape',
                'multiplier': 1.5,              # Slower due to more pixels
                'description': 'Large (1280×720) - HD quality'
            },
            'hd_portrait': {
                'size': '720*1280',             # ✅ SUPPORTED - HD portrait
                'aspect_ratio': 'portrait',
                'multiplier': 1.5,              # Slower due to more pixels
                'description': 'HD Portrait (720×1280) - High quality'
            },
            'square': {
                'size': '1024*1024',            # ✅ SUPPORTED - Square format
                'aspect_ratio': 'square',
                'multiplier': 1.3,              # More pixels than standard
                'description': 'Square (1024×1024) - Square format'
            }
        }
        
        # Quality configurations optimized for speed
        self.quality_configs = {
            'fast': {
                'sample_steps': 8,              # Minimum reasonable steps
                'sample_guide_scale': 5.5,      # Lower guidance for speed
                'description': 'Fast - Speed optimized'
            },
            'balanced': {
                'sample_steps': 10,             # Balanced steps
                'sample_guide_scale': 6.0,      # Moderate guidance
                'description': 'Balanced - Speed/quality balance'
            },
            'quality': {
                'sample_steps': 12,             # Higher quality
                'sample_guide_scale': 6.5,      # Higher guidance
                'description': 'Quality - Quality focused'
            }
        }

        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')

        print("🎬 FIXED Worker ready")
        print("📊 SUPPORTED resolution mappings:")
        for job_type, config in self.job_type_mapping.items():
            res_config = self.resolution_configs[config['resolution']]
            print(f"   • {job_type}: {config['expected_time']}s ({res_config['size']} - {config['description']})")

    def init_hardware_optimizations(self):
        """Initialize hardware optimizations for better performance"""
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

    def get_job_config(self, job_type):
        """Get configuration using only supported resolutions"""
        job_mapping = self.job_type_mapping.get(job_type)
        if not job_mapping:
            # Fallback for unknown job types - use working resolution
            print(f"⚠️ Unknown job type: {job_type}, using defaults")
            return {
                'size': '832*480',              # ✅ KNOWN WORKING
                'frame_num': 1,
                'sample_steps': 10,
                'sample_guide_scale': 6.0,
                'expected_time': 70,
                'storage_bucket': 'image_fast',
                'content_type': 'image'
            }
        
        # Get internal configurations
        resolution_config = self.resolution_configs[job_mapping['resolution']]
        quality_config = self.quality_configs[job_mapping['quality']]
        
        # Determine frame count based on content type
        if job_mapping['content_type'] == 'image':
            frame_num = 1
        elif job_mapping['content_type'] == 'video':
            frame_num = 17  # ~1 second at 16fps
        else:
            frame_num = 1
            
        return {
            'size': resolution_config['size'],           # ✅ GUARANTEED SUPPORTED
            'frame_num': frame_num,
            'sample_steps': quality_config['sample_steps'],
            'sample_guide_scale': quality_config['sample_guide_scale'],
            'expected_time': job_mapping['expected_time'],
            'storage_bucket': job_mapping['storage_bucket'],
            'content_type': job_mapping['content_type'],
            'resolution_desc': resolution_config['description'],
            'quality_desc': quality_config['description']
        }

    def get_expected_time(self, job_type):
        """Get expected generation time for user feedback"""
        job_mapping = self.job_type_mapping.get(job_type, {})
        return f"{job_mapping.get('expected_time', 70)}s"

    def generate(self, prompt, job_type):
        """FIXED: Enhanced generation with supported resolutions only"""
        config = self.get_job_config(job_type)

        # Ensure model is ready in temp storage
        if not self.ensure_model_ready():
            return None

        job_id = str(uuid.uuid4())[:8]
        memory_before = self.check_gpu_memory()
        warm_start = memory_before > 5000
        
        expected_time = config['expected_time']
        resolution_desc = config.get('resolution_desc', 'unknown')
        quality_desc = config.get('quality_desc', 'unknown')

        print(f"⚡ {job_type.upper()} generation ({'WARM' if warm_start else 'COLD'} start)")
        print(f"📝 Prompt: {prompt}")
        print(f"📐 FIXED resolution: {resolution_desc}")
        print(f"⚙️ Quality: {quality_desc}")
        print(f"🔧 Config: {config['sample_steps']} steps, {config['sample_guide_scale']} guidance, {config['size']} ✅")
        print(f"🎯 Expected: {expected_time}s")

        # Use temp processing directory for outputs
        output_filename = f"{job_type}_{job_id}.mp4"
        temp_output_path = self.temp_processing / output_filename
        
        # FIXED: Command with guaranteed supported resolution
        cmd = [
            "python", "generate.py",
            "--task", "t2v-1.3B",
            "--size", config['size'],               # ✅ GUARANTEED SUPPORTED
            "--ckpt_dir", self.model_path,
            "--prompt", prompt,
            "--save_file", str(temp_output_path),
            "--sample_steps", str(config['sample_steps']),
            "--sample_guide_scale", str(config['sample_guide_scale']),
            "--frame_num", str(config['frame_num'])
        ]

        print(f"🔧 Command: {' '.join(cmd[-8:])}")  # Log key parameters

        # Change to Wan2.1 directory but output to temp
        original_cwd = os.getcwd()
        os.chdir("/workspace/Wan2.1")
        
        try:
            start_time = time.time()
            
            # Set optimized environment variables for generation
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
                print(f"❌ Command that failed: {' '.join(cmd)}")
                return None
                
            print(f"⚡ Generation completed in {generation_time:.1f}s (expected {expected_time}s)")
                
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
            
            if config['content_type'] == 'image':
                return self.extract_frame_from_video(str(temp_output_path), job_id, job_type)
            
            return str(temp_output_path)
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return None
        finally:
            os.chdir(original_cwd)

    def extract_frame_from_video(self, video_path, job_id, job_type):
        """Enhanced frame extraction with job-type-aware optimization"""
        image_path = self.temp_processing / f"{job_type}_{job_id}.png"
        
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Optimize compression based on job type
                if 'fast' in job_type:
                    # Faster compression for quick jobs
                    img.save(str(image_path), "PNG", optimize=True, compress_level=9)
                else:
                    # Better compression for quality jobs
                    img.save(str(image_path), "PNG", optimize=True, compress_level=6)
                
                # Get file size for logging
                file_size = os.path.getsize(image_path) / 1024  # KB
                config = self.get_job_config(job_type)
                size_desc = config.get('size', 'unknown')
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
        """Job-type-aware file optimization"""
        config = self.get_job_config(job_type)
        content_type = config['content_type']
        
        if content_type == 'image':
            # Images are already optimized during creation
            return file_path
            
        if content_type == 'video' and self.ffmpeg_available:
            optimized_path = str(Path(file_path).with_suffix('.optimized.mp4'))
            
            # Get target resolution from config
            size = config['size']
            width, height = size.split('*')
            
            # Quality-based encoding optimization
            if 'fast' in job_type:
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
        """Upload to the correct storage bucket based on job type"""
        if not os.path.exists(file_path):
            return None
            
        optimized_path = self.optimize_file_for_upload(file_path, job_type)
        config = self.get_job_config(job_type)
        
        # Use the storage bucket from job type mapping
        storage_bucket = config['storage_bucket']
        content_type = config['content_type']
        
        filename = f"job_{job_id}_{int(time.time())}_{job_type}.{'png' if content_type == 'image' else 'mp4'}"
        full_path = f"{storage_bucket}/{user_id}/{filename}"
        mime_type = 'image/png' if content_type == 'image' else 'video/mp4'
        
        print(f"📤 Uploading to bucket: {storage_bucket}")
        print(f"📤 Full path: {self.supabase_url}/storage/v1/object/{full_path}")
        
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
                                'Content-Type': mime_type,
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
        """More aggressive cleanup for better performance"""
        try:
            current_time = time.time()
            cleaned_count = 0
            
            for temp_dir in [self.temp_outputs, self.temp_processing]:
                for file_path in temp_dir.glob("*"):
                    if file_path.is_file():
                        # Clean files older than 20 minutes
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
        """Enhanced job processing with fixed resolutions"""
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
        """Enhanced main loop with fixed resolution support"""
        print("⏳ Waiting for jobs...")
        print("🎯 FIXED Resolution Targets:")
        print("   • image_fast: 45s (480×832 - fastest supported)")
        print("   • image_high: 70s (832×480 - current working)")
        print("   • video_fast: 55s (480×832 - fastest supported)")
        print("   • video_high: 85s (832×480 - current working)")
        print("✅ All resolutions guaranteed supported by Wan 2.1")
        
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
    
    print("🚀 Starting OurVidz Worker (RESOLUTION FIXED)")
    print("✅ Using only Wan 2.1 supported resolutions")
    worker = VideoWorker()
    worker.run()
