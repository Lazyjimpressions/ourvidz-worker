# worker.py - Fixed Upload Logic and Job Processing
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

class VideoWorker:
    def __init__(self):
        print("🚀 OurVidz Worker initialized (FIXED UPLOAD LOGIC)")
        
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

        # Use temp storage for models - much faster I/O
        self.model_path = str(self.temp_models / 'wan2.1-t2v-1.3b')
        self.model_loaded = False

        self.job_configs = {
            'image_fast': {'size': '832*480', 'frame_num': 1, 'sample_steps': 8, 'sample_guide_scale': 6.0},
            'image_high': {'size': '1280*720', 'frame_num': 1, 'sample_steps': 20, 'sample_guide_scale': 7.5},
            'video_fast': {'size': '832*480', 'frame_num': 17, 'sample_steps': 12, 'sample_guide_scale': 6.0},
            'video_high': {'size': '1280*720', 'frame_num': 33, 'sample_steps': 25, 'sample_guide_scale': 7.5}
        }

        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')

        print("🎬 Worker ready (upload logic fixed)")

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
        """Ensure model is available in temp storage (copy from network volume if needed)"""
        if os.path.exists(self.model_path):
            print("✅ Model already in temp storage")
            return True
            
        # Check if model exists in network volume
        network_model_path = "/workspace/models/wan2.1-t2v-1.3b"
        if os.path.exists(network_model_path):
            print("📦 Copying model from network volume to temp storage...")
            try:
                shutil.copytree(network_model_path, self.model_path)
                print("✅ Model copied to temp storage (faster I/O)")
                return True
            except Exception as e:
                print(f"❌ Model copy failed: {e}")
                # Fallback to network volume
                self.model_path = network_model_path
                return True
        
        print("❌ Model not found in network volume")
        return False

    def generate(self, prompt, job_type):
        config = self.job_configs.get(job_type)
        if not config:
            print(f"❌ Unknown job type: {job_type}")
            return None

        # Ensure model is ready in temp storage
        if not self.ensure_model_ready():
            return None

        job_id = str(uuid.uuid4())[:8]
        memory_before = self.check_gpu_memory()
        warm_start = memory_before > 5000

        print(f"⚡ {job_type.upper()} generation ({'WARM' if warm_start else 'COLD'} start)")
        print(f"📝 Prompt: {prompt}")

        # Use temp processing directory for outputs - much faster writes
        output_filename = f"{job_type}_{job_id}.mp4"
        temp_output_path = self.temp_processing / output_filename
        
        cmd = [
            "python", "generate.py",
            "--task", "t2v-1.3B",
            "--size", config['size'],
            "--ckpt_dir", self.model_path,
            "--prompt", prompt,
            "--save_file", str(temp_output_path),  # Write directly to temp
            "--sample_steps", str(config['sample_steps']),
            "--sample_guide_scale", str(config['sample_guide_scale']),
            "--frame_num", str(config['frame_num'])
        ]

        # Change to Wan2.1 directory but output to temp
        original_cwd = os.getcwd()
        os.chdir("/workspace/Wan2.1")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"❌ Generation failed: {result.stderr}")
                return None
                
            # Check if file was created in temp location
            if not temp_output_path.exists():
                # Fallback: check if created in current directory
                fallback_path = Path(output_filename)
                if fallback_path.exists():
                    # Move to temp location
                    shutil.move(str(fallback_path), str(temp_output_path))
                    print("📦 Moved output to temp storage")
                else:
                    print("❌ Output file not found")
                    return None
            
            if 'image' in job_type:
                return self.extract_frame_from_video(str(temp_output_path), job_id, job_type)
            
            return str(temp_output_path)
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return None
        finally:
            os.chdir(original_cwd)

    def extract_frame_from_video(self, video_path, job_id, job_type):
        """Extract frame using temp storage for faster I/O"""
        image_path = self.temp_processing / f"{job_type}_{job_id}.png"
        
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Use PIL with optimization for smaller file size and faster upload
                img = Image.fromarray(frame_rgb)
                img.save(str(image_path), "PNG", optimize=True, compress_level=6)
                
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
        """Optimize files before upload for faster transfer"""
        if 'image' in job_type:
            # Already optimized during creation
            return file_path
            
        if 'video' in job_type and self.ffmpeg_available:
            # Optimize video for web streaming and smaller size
            optimized_path = str(Path(file_path).with_suffix('.optimized.mp4'))
            
            cmd = [
                'ffmpeg', '-i', file_path,
                '-c:v', 'libx264',
                '-preset', 'fast',  # Faster encoding
                '-crf', '23',       # Good quality/size balance
                '-movflags', '+faststart',  # Web optimization
                '-pix_fmt', 'yuv420p',
                '-y', optimized_path
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0 and os.path.exists(optimized_path):
                    # Check if optimized version is actually smaller
                    orig_size = os.path.getsize(file_path)
                    opt_size = os.path.getsize(optimized_path)
                    
                    if opt_size < orig_size:
                        print(f"📉 Optimized: {orig_size//1024}KB → {opt_size//1024}KB")
                        os.remove(file_path)  # Remove original
                        return optimized_path
                    else:
                        os.remove(optimized_path)  # Remove if not better
                        
            except Exception as e:
                print(f"⚠️ Optimization failed: {e}")
                
        return file_path

    def upload_to_supabase(self, file_path, job_type, user_id, job_id):
        """FIXED: Upload with correct Supabase Storage API format"""
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
            
        # Optimize file before upload
        optimized_path = self.optimize_file_for_upload(file_path, job_type)
        
        filename = f"job_{job_id}_{int(time.time())}_{job_type}.{'png' if 'image' in job_type else 'mp4'}"
        full_path = f"{job_type}/{user_id}/{filename}"
        
        # Determine correct content type
        content_type = 'image/png' if 'image' in job_type else 'video/mp4'
        
        print(f"📤 Uploading to: {self.supabase_url}/storage/v1/object/{full_path}")
        
        try:
            with open(optimized_path, 'rb') as f:
                file_data = f.read()  # Read entire file into memory
                file_size = len(file_data)
                print(f"📊 File size: {file_size // 1024}KB")
                
                # Upload with retry logic for better reliability
                for attempt in range(3):
                    try:
                        print(f"🔄 Upload attempt {attempt + 1}/3...")
                        
                        r = requests.post(
                            f"{self.supabase_url}/storage/v1/object/{full_path}",
                            data=file_data,  # ✅ RAW BINARY DATA (not files=)
                            headers={
                                'Authorization': f"Bearer {self.supabase_service_key}",
                                'Content-Type': content_type,  # ✅ REQUIRED HEADER
                                'x-upsert': 'true',  # ✅ PREVENT 400 ASSET EXISTS
                                'Content-Length': str(file_size)  # Optional but helpful
                            },
                            timeout=120  # Longer timeout for large files
                        )
                        
                        print(f"📡 Response: {r.status_code}")
                        
                        if r.status_code in [200, 201]:
                            print(f"✅ Upload successful: {full_path}")
                            return f"{user_id}/{filename}"
                        else:
                            print(f"⚠️ Upload attempt {attempt + 1} failed: {r.status_code}")
                            print(f"📝 Response text: {r.text}")
                            
                            # Don't retry on certain errors
                            if r.status_code in [401, 403, 404]:
                                print("🚫 Auth/permission error - not retrying")
                                break
                                
                    except requests.RequestException as e:
                        print(f"⚠️ Upload attempt {attempt + 1} network error: {e}")
                        if attempt < 2:  # Don't sleep on last attempt
                            time.sleep(2 ** attempt)  # Exponential backoff
                            
        except Exception as e:
            print(f"❌ Upload preparation failed: {e}")
        finally:
            # Clean up temp files
            self.cleanup_temp_files([file_path, optimized_path])
            
        print("❌ All upload attempts failed")
        return None

    def cleanup_temp_files(self, file_paths):
        """Clean up temporary files to keep temp storage lean"""
        for file_path in file_paths:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass

    def cleanup_old_temp_files(self):
        """Periodic cleanup of old temp files"""
        try:
            current_time = time.time()
            for temp_dir in [self.temp_outputs, self.temp_processing]:
                for file_path in temp_dir.glob("*"):
                    if file_path.is_file() and (current_time - file_path.stat().st_mtime) > 3600:  # 1 hour
                        try:
                            file_path.unlink()
                            print(f"🧹 Cleaned up old temp file: {file_path.name}")
                        except:
                            pass
        except Exception as e:
            print(f"⚠️ Temp cleanup error: {e}")

    def notify_completion(self, job_id, status, file_path=None, error_message=None):
        """Notify job completion to callback function"""
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
        """FIXED: Process job with proper field extraction"""
        job_id = job_data.get('jobId')
        job_type = job_data.get('jobType')
        prompt = job_data.get('prompt')  # ✅ No fallback - prompt is required
        user_id = job_data.get('userId')
        
        # Log received job data for debugging
        print(f"📋 Received job data keys: {list(job_data.keys())}")
        print(f"📋 Job details: ID={job_id}, Type={job_type}, User={user_id}")
        print(f"📝 Prompt: {prompt}")
        
        # Validate all required fields
        if not all([job_id, job_type, user_id, prompt]):
            missing = [field for field, value in [
                ('jobId', job_id), ('jobType', job_type), 
                ('userId', user_id), ('prompt', prompt)
            ] if not value]
            error_msg = f"Missing required fields: {', '.join(missing)}"
            print(f"❌ {error_msg}")
            self.notify_completion(job_id or 'unknown', 'failed', error_message=error_msg)
            return

        print(f"📥 Processing job: {job_id} ({job_type})")
        start_time = time.time()
        
        try:
            # Generate content
            output_path = self.generate(prompt, job_type)
            if output_path:
                print(f"✅ Generation completed: {output_path}")
                
                # Upload to Supabase
                supa_path = self.upload_to_supabase(output_path, job_type, user_id, job_id)
                if supa_path:
                    duration = time.time() - start_time
                    print(f"🎉 Job completed successfully in {duration:.1f}s")
                    self.notify_completion(job_id, 'completed', supa_path)
                    return
                else:
                    print("❌ Upload failed")
            else:
                print("❌ Generation failed")
                    
            self.notify_completion(job_id, 'failed', error_message="Generation or upload failed")
            
        except Exception as e:
            print(f"❌ Job processing error: {e}")
            self.notify_completion(job_id, 'failed', error_message=str(e))

    def poll_queue(self):
        """Poll Redis queue for new jobs"""
        try:
            r = requests.get(
                f"{self.redis_url}/rpop/job_queue",
                headers={'Authorization': f"Bearer {self.redis_token}"}, 
                timeout=10
            )
            
            if r.status_code == 200:
                result = r.json().get('result')
                if result:
                    return json.loads(result)
            elif r.status_code != 200:
                print(f"⚠️ Redis poll error: {r.status_code}")
                
        except Exception as e:
            print(f"❌ Poll error: {e}")
        return None

    def run(self):
        """Main worker loop"""
        print("⏳ Waiting for jobs...")
        last_cleanup = time.time()
        
        while True:
            # Periodic cleanup every 30 minutes
            if time.time() - last_cleanup > 1800:
                self.cleanup_old_temp_files()
                last_cleanup = time.time()
                
            job = self.poll_queue()
            if job:
                self.process_job(job)
                print("⏳ Job complete, checking queue...")
            else:
                time.sleep(5)

if __name__ == "__main__":
    # Validate environment variables
    required_vars = [
        'SUPABASE_URL', 'SUPABASE_SERVICE_KEY',
        'UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    print("🔧 Environment variables validated")
    
    worker = VideoWorker()
    worker.run()
