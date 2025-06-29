# worker.py - Infrastructure-Matched Production Worker
import os
import json
import time
import requests
import subprocess
import uuid
import shutil
import tempfile
import threading
from pathlib import Path
from PIL import Image
import cv2
import torch

class VideoWorker:
    def __init__(self):
        print("🚀 OurVidz Worker initialized (INFRASTRUCTURE MATCHED)")
        print("⚡ Conforming to existing frontend/edge/redis/supabase setup")
        
        # Create dedicated temp directories (matching original)
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
        
        # GPU monitoring state (from original)
        self.gpu_monitoring_active = False
        self.gpu_monitor_thread = None
        self.generation_active = False
        self.last_gpu_stats = {}
        
        # Initialize GPU management
        self.detect_gpu()
        self.force_gpu_activation()
        self.init_hardware_optimizations()
        self.start_enhanced_gpu_management()

        # Use persistent model path (corrected)
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_script_path = "/workspace/Wan2.1"
        
        # 4 JOB TYPES CONFIGURATION (exactly matching original)
        self.job_type_mapping = {
            'image_fast': {
                'content_type': 'image',
                'resolution': 'small',            
                'quality': 'fast',                
                'storage_bucket': 'image_fast',
                'expected_time': 90,
                'description': 'Small resolution, fastest available speed'
            },
            'image_high': {
                'content_type': 'image',
                'resolution': 'standard',         
                'quality': 'balanced',            
                'storage_bucket': 'image_high',
                'expected_time': 100,
                'description': 'Standard resolution, balanced quality'
            },
            'video_fast': {
                'content_type': 'video',
                'resolution': 'small',            
                'quality': 'fast',                
                'storage_bucket': 'video_fast',
                'expected_time': 95,  # Matching original timing
                'description': 'Small resolution, fast video'
            },
            'video_high': {
                'content_type': 'video', 
                'resolution': 'standard',         
                'quality': 'balanced',            
                'storage_bucket': 'video_high',
                'expected_time': 110,  # Matching original timing
                'description': 'Standard resolution, quality video'
            }
        }
        
        # Resolution configurations using ONLY supported Wan 2.1 sizes
        self.resolution_configs = {
            'small': {
                'size': '480*832',              
                'multiplier': 0.7,              
                'description': 'Small (480×832) - Fastest supported'
            },
            'standard': {
                'size': '832*480',              
                'multiplier': 1.0,              
                'description': 'Standard (832×480) - Current working'
            }
        }
        
        # Quality configurations optimized for speed (matching original)
        self.quality_configs = {
            'fast': {
                'sample_steps': 8,              
                'sample_guide_scale': 5.5,      
                'description': 'Fast - Speed optimized'
            },
            'balanced': {
                'sample_steps': 10,             
                'sample_guide_scale': 6.0,      
                'description': 'Balanced - Speed/quality balance'
            }
        }

        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')

        print("🎬 Infrastructure-matched worker ready")

    def detect_gpu(self):
        """GPU detection (simplified from original)"""
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"🔥 GPU: {gpu_name} ({total_memory:.1f}GB)")
            else:
                print("❌ CUDA not available")
        except Exception as e:
            print(f"⚠️ GPU detection failed: {e}")

    def force_gpu_activation(self):
        """GPU activation (simplified from original)"""
        try:
            if torch.cuda.is_available():
                print("🔥 GPU activation...")
                warmup_tensor = torch.ones((1000, 1000), device='cuda')
                for _ in range(5):
                    result = torch.matmul(warmup_tensor, warmup_tensor)
                    torch.cuda.synchronize()
                del warmup_tensor, result
                torch.cuda.empty_cache()
                print("✅ GPU activation complete")
        except Exception as e:
            print(f"⚠️ GPU activation failed: {e}")

    def init_hardware_optimizations(self):
        """Hardware optimizations (from original)"""
        try:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True  
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                os.environ.update({
                    'CUDA_LAUNCH_BLOCKING': '0',
                    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
                    'TORCH_USE_CUDA_DSA': '1',
                })
                print("✅ Hardware optimizations applied")
        except Exception as e:
            print(f"⚠️ Hardware optimization failed: {e}")

    def start_enhanced_gpu_management(self):
        """Start GPU management thread (simplified from original)"""
        def gpu_management_thread():
            try:
                while True:
                    if torch.cuda.is_available():
                        # Keep GPU warm
                        warmup = torch.ones((100, 100), device='cuda')
                        torch.matmul(warmup, warmup)
                        torch.cuda.synchronize()
                        del warmup
                    time.sleep(30)
            except Exception as e:
                print(f"⚠️ GPU management error: {e}")
        
        management_thread = threading.Thread(target=gpu_management_thread, daemon=True)
        management_thread.start()

    def get_job_config(self, job_type):
        """Get configuration with realistic timing expectations (from original)"""
        job_mapping = self.job_type_mapping.get(job_type)
        if not job_mapping:
            return {
                'size': '832*480',
                'frame_num': 1,
                'sample_steps': 10,
                'sample_guide_scale': 6.0,
                'expected_time': 90,
                'storage_bucket': 'image_fast',
                'content_type': 'image'
            }
        
        resolution_config = self.resolution_configs[job_mapping['resolution']]
        quality_config = self.quality_configs[job_mapping['quality']]
        
        frame_num = 1 if job_mapping['content_type'] == 'image' else 17  # Matching original
            
        return {
            'size': resolution_config['size'],
            'frame_num': frame_num,
            'sample_steps': quality_config['sample_steps'],
            'sample_guide_scale': quality_config['sample_guide_scale'],
            'expected_time': job_mapping['expected_time'],
            'storage_bucket': job_mapping['storage_bucket'],
            'content_type': job_mapping['content_type'],
            'resolution_desc': resolution_config['description'],
            'quality_desc': quality_config['description']
        }

    def generate_with_gpu_monitoring(self, prompt, job_type):
        """Enhanced generation with GPU monitoring (matching original method name)"""
        config = self.get_job_config(job_type)
        self.generation_active = True

        job_id = str(uuid.uuid4())[:8]
        expected_time = config['expected_time']

        print(f"⚡ {job_type.upper()} generation")
        print(f"📝 Prompt: {prompt}")
        print(f"📐 Resolution: {config.get('resolution_desc', 'unknown')}")
        print(f"⚙️ Quality: {config.get('quality_desc', 'unknown')}")
        print(f"🔧 Config: {config['sample_steps']} steps, {config['sample_guide_scale']} guidance, {config['size']}")
        print(f"🎯 Expected: {expected_time}s")

        # Use temp processing directory (from original)
        output_filename = f"{job_type}_{job_id}.mp4"
        temp_output_path = self.temp_processing / output_filename
        
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

        original_cwd = os.getcwd()
        os.chdir("/workspace/Wan2.1")
        
        try:
            start_time = time.time()
            
            env = os.environ.copy()
            env.update({
                'CUDA_LAUNCH_BLOCKING': '0',
                'OMP_NUM_THREADS': '8',
                'MKL_NUM_THREADS': '8',
            })
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
            generation_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"❌ Generation failed: {result.stderr}")
                return None
                
            print(f"⚡ Generation completed in {generation_time:.1f}s (expected {expected_time}s)")
                
            if not temp_output_path.exists():
                fallback_path = Path(output_filename)
                if fallback_path.exists():
                    shutil.move(str(fallback_path), str(temp_output_path))
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
            self.generation_active = False
            os.chdir(original_cwd)

    def extract_frame_from_video(self, video_path, job_id, job_type):
        """Enhanced frame extraction (from original)"""
        image_path = self.temp_processing / f"{job_type}_{job_id}.png"
        
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Optimize compression based on job type (from original)
                if 'fast' in job_type:
                    img.save(str(image_path), "PNG", optimize=True, compress_level=9)
                else:
                    img.save(str(image_path), "PNG", optimize=True, compress_level=6)
                
                file_size = os.path.getsize(image_path) / 1024
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

    def upload_to_supabase(self, file_path, job_type, user_id, job_id):
        """Upload matching original structure"""
        if not os.path.exists(file_path):
            return None
            
        config = self.get_job_config(job_type)
        storage_bucket = config['storage_bucket']
        content_type = config['content_type']
        
        filename = f"job_{job_id}_{int(time.time())}_{job_type}.{'png' if content_type == 'image' else 'mp4'}"
        full_path = f"{storage_bucket}/{user_id}/{filename}"
        mime_type = 'image/png' if content_type == 'image' else 'video/mp4'
        
        print(f"📤 Uploading to bucket: {storage_bucket}")
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
                file_size = len(file_data) / 1024
                print(f"📊 File size: {file_size:.0f}KB")
                
                r = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{full_path}",
                    data=file_data,
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                        'Content-Type': mime_type,
                        'x-upsert': 'true'
                    },
                    timeout=120
                )
                
                if r.status_code in [200, 201]:
                    print(f"✅ Upload successful: {full_path}")
                    return f"{user_id}/{filename}"  # Return path, not full URL (matching original)
                else:
                    print(f"❌ Upload failed: {r.status_code} - {r.text}")
                    
        except Exception as e:
            print(f"❌ Upload preparation failed: {e}")
        finally:
            # Cleanup temp files (from original)
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
            
        print("❌ Upload failed")
        return None

    def notify_completion(self, job_id, status, file_path=None, error_message=None):
        """Callback matching original structure"""
        data = {
            'jobId': job_id, 
            'status': status, 
            'filePath': file_path,  # Original uses filePath, not outputUrl
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
        """Job processing matching original structure"""
        job_id = job_data.get('jobId')
        job_type = job_data.get('jobType')
        prompt = job_data.get('prompt')
        user_id = job_data.get('userId')
        
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
        
        start_time = time.time()
        
        try:
            # Use enhanced generation method (matching original)
            output_path = self.generate_with_gpu_monitoring(prompt, job_type)
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
        """Queue polling matching original (CRITICAL: job_queue not job-queue)"""
        try:
            r = requests.get(
                f"{self.redis_url}/rpop/job_queue",  # ORIGINAL USES UNDERSCORE
                headers={'Authorization': f"Bearer {self.redis_token}"}, 
                timeout=10
            )
            if r.status_code == 200 and r.json().get('result'):
                return json.loads(r.json()['result'])
        except Exception as e:
            print(f"❌ Poll error: {e}")
        return None

    def cleanup_old_temp_files(self):
        """Cleanup from original"""
        try:
            current_time = time.time()
            cleaned_count = 0
            
            for temp_dir in [self.temp_outputs, self.temp_processing]:
                for file_path in temp_dir.glob("*"):
                    if file_path.is_file():
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

    def run(self):
        """Main loop matching original structure"""
        print("⏳ Waiting for jobs...")
        print("🎯 Supported Job Types:")
        for job_type, config in self.job_type_mapping.items():
            print(f"   • {job_type}: {config['description']} (~{config['expected_time']}s)")
        
        last_cleanup = time.time()
        job_count = 0
        
        while True:
            # Cleanup every 10 minutes
            if time.time() - last_cleanup > 600:
                self.cleanup_old_temp_files()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                last_cleanup = time.time()
                
            # Poll for jobs
            job = self.poll_queue()
            if job:
                job_count += 1
                print(f"🎯 Processing job #{job_count}")
                self.process_job(job)
            else:
                time.sleep(5)

if __name__ == "__main__":
    print("🚀 Starting OurVidz Infrastructure-Matched Worker")
    
    try:
        worker = VideoWorker()
        worker.run()
    except Exception as e:
        print(f"❌ Worker failed to start: {e}")
        exit(1)
