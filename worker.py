# worker.py - GPU Optimized with Model Persistence
import os
import json
import time
import requests
import uuid
import shutil
import threading
from pathlib import Path
from PIL import Image
import cv2
import torch

# FORCE GPU USAGE - CRITICAL FOR PERFORMANCE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

class VideoWorker:
    def __init__(self):
        print("🚀 OurVidz Worker initialized (GPU OPTIMIZED WITH MODEL PERSISTENCE)")
        
        # Force GPU setup
        if not torch.cuda.is_available():
            print("❌ CUDA not available - cannot continue")
            exit(1)
            
        torch.cuda.set_device(0)
        self.device = torch.device('cuda:0')
        print(f"🔥 GPU FORCED: {torch.cuda.get_device_name(0)}")
        
        # Create temp directories
        self.temp_base = Path("/tmp/ourvidz")
        self.temp_base.mkdir(exist_ok=True)
        
        self.temp_models = self.temp_base / "models"
        self.temp_outputs = self.temp_base / "outputs" 
        self.temp_processing = self.temp_base / "processing"
        
        for temp_dir in [self.temp_models, self.temp_outputs, self.temp_processing]:
            temp_dir.mkdir(exist_ok=True)

    def extract_frame_from_video(self, video_path, job_id, job_type):
        """Extract frame for image jobs"""
        image_path = self.temp_processing / f"{job_type}_{job_id}.png"
        
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.save(str(image_path), "PNG", optimize=True)
                
                file_size = os.path.getsize(image_path) / 1024
                print(f"📊 Output: {file_size:.0f}KB")
                
                # Clean up video file
                try:
                    os.remove(video_path)
                except:
                    pass
                    
                return str(image_path)
        except Exception as e:
            print(f"❌ Frame extraction error: {e}")
        return None
        
        # GPU optimization
        self.init_gpu_optimizations()
        
        # OPTIMIZED job configurations with realistic timing
        self.job_type_mapping = {
            'image_fast': {
                'content_type': 'image',
                'sample_steps': 4,
                'sample_guide_scale': 3.0,
                'size': '480*832',
                'frame_num': 1,
                'storage_bucket': 'image_fast',
                'expected_time': 15,  # With pre-loaded model
                'description': 'Ultra fast image generation'
            },
            'image_high': {
                'content_type': 'image',
                'sample_steps': 6,
                'sample_guide_scale': 5.0,
                'size': '832*480',
                'frame_num': 1,
                'storage_bucket': 'image_high',
                'expected_time': 20,  # With pre-loaded model
                'description': 'High quality image generation'
            },
            'video_fast': {
                'content_type': 'video',
                'sample_steps': 4,
                'sample_guide_scale': 3.0,
                'size': '480*832',
                'frame_num': 17,
                'storage_bucket': 'video_fast',
                'expected_time': 25,  # With pre-loaded model
                'description': 'Fast video generation'
            },
            'video_high': {
                'content_type': 'video',
                'sample_steps': 6,
                'sample_guide_scale': 5.0,
                'size': '832*480',
                'frame_num': 17,
                'storage_bucket': 'video_high',
                'expected_time': 35,  # With pre-loaded model
                'description': 'High quality video generation'
            }
        }
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')

        print("🎬 GPU-optimized worker ready")

    def init_gpu_optimizations(self):
        """Initialize GPU optimizations for maximum performance"""
        try:
            # Enable all performance optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True  
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            
            # Memory optimization
            torch.cuda.empty_cache()
            
            print("✅ GPU optimizations applied")
        except Exception as e:
            print(f"⚠️ GPU optimization failed: {e}")

    def generate_with_gpu_forced(self, prompt, job_type):
        """Generate using the working generate.py method with GPU forced"""
        config = self.job_type_mapping.get(job_type, self.job_type_mapping['image_fast'])
        
        print(f"⚡ {job_type.upper()} generation (GPU FORCED)")
        print(f"📝 Prompt: {prompt}")
        print(f"🔧 Config: {config['sample_steps']} steps, {config['sample_guide_scale']} guidance, {config['size']}")
        print(f"🎯 Expected: {config['expected_time']}s")
        
        job_id = str(uuid.uuid4())[:8]
        output_filename = f"{job_type}_{job_id}.mp4"
        temp_output_path = self.temp_processing / output_filename
        
        # Build command exactly like our successful test
        cmd = [
            "python", "-c",
            """
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)
exec(open('generate.py').read())
""",
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
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            generation_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"❌ Generation failed: {result.stderr}")
                print(f"❌ STDOUT: {result.stdout}")
                return None
                
            print(f"⚡ Generation completed in {generation_time:.1f}s")
                
            if not temp_output_path.exists():
                # Check if file was created in current directory
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
            os.chdir(original_cwd)

    def generate_with_loaded_model(self, prompt, job_type):
        """Generate using the working generate.py method - FAST GENERATION"""
        config = self.job_type_mapping.get(job_type, self.job_type_mapping['image_fast'])
        
        print(f"⚡ {job_type.upper()} generation (USING WORKING METHOD)")
        print(f"📝 Prompt: {prompt}")
        print(f"🔧 Config: {config['sample_steps']} steps, {config['sample_guide_scale']} guidance, {config['size']}")
        print(f"🎯 Expected: {config['expected_time']}s")
        
        return self.generate_with_gpu_forced(prompt, job_type)

    def upload_to_supabase(self, file_path, job_type, user_id, job_id):
        """Upload file to Supabase storage"""
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
            
        config = self.job_type_mapping.get(job_type, self.job_type_mapping['image_fast'])
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
                print(f"📊 Upload size: {file_size:.0f}KB")
                
                r = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{full_path}",
                    data=file_data,
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                        'Content-Type': mime_type,
                        'x-upsert': 'true'
                    },
                    timeout=60
                )
                
                if r.status_code in [200, 201]:
                    print(f"✅ Upload successful: {full_path}")
                    return f"{user_id}/{filename}"
                else:
                    print(f"❌ Upload failed: {r.status_code} - {r.text}")
                    
        except Exception as e:
            print(f"❌ Upload error: {e}")
        finally:
            # Clean up temp file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️ Cleaned up: {file_path}")
            except:
                pass
                
        return None

    def notify_completion(self, job_id, status, file_path=None, error_message=None):
        """Notify Supabase of job completion"""
        data = {
            'jobId': job_id,
            'status': status,
            'filePath': file_path,  # Using filePath as per original infrastructure
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
                timeout=15
            )
            
            if r.status_code == 200:
                print("✅ Callback sent successfully")
            else:
                print(f"❌ Callback failed: {r.status_code} - {r.text}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job(self, job_data):
        """Process job with optimized pre-loaded model"""
        job_id = job_data.get('jobId')
        job_type = job_data.get('jobType')
        prompt = job_data.get('prompt')
        user_id = job_data.get('userId')
        
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
        print(f"📥 Processing OPTIMIZED job: {job_id} ({job_type})")
        
        total_start_time = time.time()
        
        try:
            # Generate using working method
            output_path = self.generate_with_loaded_model(prompt, job_type)
            
            if output_path:
                # Upload to Supabase
                supa_path = self.upload_to_supabase(output_path, job_type, user_id, job_id)
                
                if supa_path:
                    total_duration = time.time() - total_start_time
                    config = self.job_type_mapping.get(job_type, self.job_type_mapping['image_fast'])
                    expected = config['expected_time']
                    
                    if total_duration < expected:
                        speedup = ((expected - total_duration) / expected) * 100
                        print(f"🎉 Job completed in {total_duration:.1f}s - {speedup:.1f}% FASTER than expected!")
                    else:
                        print(f"🎉 Job completed in {total_duration:.1f}s")
                    
                    self.notify_completion(job_id, 'completed', supa_path)
                    return
                else:
                    print("❌ Upload failed")
            else:
                print("❌ Generation failed")
                    
            self.notify_completion(job_id, 'failed', error_message="Generation or upload failed")
            
        except Exception as e:
            print(f"❌ Job processing error: {e}")
            import traceback
            traceback.print_exc()
            self.notify_completion(job_id, 'failed', error_message=str(e))

    def poll_queue(self):
        """Poll Redis queue for jobs"""
        try:
            r = requests.get(
                f"{self.redis_url}/rpop/job_queue",
                headers={'Authorization': f"Bearer {self.redis_token}"},
                timeout=5
            )
            if r.status_code == 200 and r.json().get('result'):
                return json.loads(r.json()['result'])
        except Exception as e:
            print(f"❌ Poll error: {e}")
        return None

    def cleanup_old_temp_files(self):
        """Clean up old temporary files"""
        try:
            current_time = time.time()
            cleaned_count = 0
            
            for temp_dir in [self.temp_outputs, self.temp_processing]:
                for file_path in temp_dir.glob("*"):
                    if file_path.is_file():
                        if (current_time - file_path.stat().st_mtime) > 600:  # 10 minutes old
                            try:
                                file_path.unlink()
                                cleaned_count += 1
                            except:
                                pass
                                
            if cleaned_count > 0:
                print(f"🧹 Cleaned up {cleaned_count} old temp files")
                
            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

    def run(self):
        """Main worker loop"""
        print("⏳ Waiting for jobs (USING WORKING GENERATE.PY METHOD)...")
        print("🎯 Job Types:")
        for job_type, config in self.job_type_mapping.items():
            print(f"   • {job_type}: {config['description']} (~{config['expected_time']}s)")
        
        last_cleanup = time.time()
        job_count = 0
        
        while True:
            # Periodic cleanup
            if time.time() - last_cleanup > 300:  # Every 5 minutes
                self.cleanup_old_temp_files()
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
    print("🚀 Starting OurVidz FIXED Worker (Using Working Generate.py Method)")
    
    try:
        worker = VideoWorker()
        worker.run()
    except Exception as e:
        print(f"❌ Worker failed to start: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
