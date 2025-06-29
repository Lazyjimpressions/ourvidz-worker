# worker.py - FIXED GPU MEMORY MANAGEMENT FOR VIDEO GENERATION
# KEY FIX: Aggressive memory cleanup between jobs to prevent OOM on video generation
import os
import json
import time
import requests
import subprocess
import uuid
import shutil
import gc
from pathlib import Path
from PIL import Image
import cv2
import torch

# Clean environment - no distributed training needed
for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
    if key in os.environ:
        del os.environ[key]

# GPU optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# CRITICAL: Memory management for video generation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class VideoWorker:
    def __init__(self):
        print("🚀 OurVidz Worker initialized - GPU MEMORY MANAGEMENT FIXED")
        print("🔧 KEY FIX: Aggressive memory cleanup to prevent video generation OOM")
        
        # Verify CUDA availability
        if not torch.cuda.is_available():
            print("❌ CUDA not available - exiting")
            exit(1)
        
        # Force GPU setup
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        
        # Log GPU status
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Create temp directories
        self.temp_base = Path("/tmp/ourvidz")
        self.temp_base.mkdir(exist_ok=True)
        self.temp_processing = self.temp_base / "processing"
        self.temp_processing.mkdir(exist_ok=True)
        
        # Paths
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_path = "/workspace/Wan2.1"
        
        # Verify critical paths
        if not Path(self.wan_path).exists():
            print(f"❌ Wan2.1 path missing: {self.wan_path}")
            exit(1)
        
        if not Path(self.model_path).exists():
            print(f"❌ Model path missing: {self.model_path}")
            exit(1)
        
        # Updated job configurations - REDUCED frame counts for memory management
        self.job_type_mapping = {
            'image_fast': {
                'content_type': 'image',
                'file_extension': 'png',
                'sample_steps': 4,
                'sample_guide_scale': 3.0,
                'size': '480*832',
                'frame_num': 1,
                'storage_bucket': 'image_fast',
                'expected_time': 78,
                'description': 'Fast image generation (1 frame extraction)'
            },
            'image_high': {
                'content_type': 'image',
                'file_extension': 'png',
                'sample_steps': 6,
                'sample_guide_scale': 4.0,
                'size': '832*480',
                'frame_num': 1,
                'storage_bucket': 'image_high',
                'expected_time': 80,
                'description': 'High quality image (1 frame extraction)'
            },
            'video_fast': {
                'content_type': 'video',
                'file_extension': 'mp4',
                'sample_steps': 4,
                'sample_guide_scale': 3.0,
                'size': '480*832',
                'frame_num': 9,  # REDUCED: 0.5 second instead of 1 second
                'storage_bucket': 'video_fast',
                'expected_time': 85,
                'description': 'Fast 0.5-second video (9 frames)'
            },
            'video_high': {
                'content_type': 'video',
                'file_extension': 'mp4',
                'sample_steps': 6,
                'sample_guide_scale': 4.0,
                'size': '640*480',  # REDUCED: Smaller resolution for memory
                'frame_num': 13,     # REDUCED: 0.75 second instead of 1 second
                'storage_bucket': 'video_high',
                'expected_time': 90,
                'description': 'High quality 0.75-second video (13 frames)'
            }
        }
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')

        print("🎬 Worker ready - MEMORY OPTIMIZED FOR VIDEO GENERATION!")
        print("🔧 Memory management fixes:")
        for job_type, config in self.job_type_mapping.items():
            print(f"   • {job_type}: {config['description']}")

    def log_gpu_memory(self, context=""):
        """Enhanced GPU memory logging"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - allocated
            print(f"🔥 GPU {context}: {allocated:.2f}GB used, {free:.2f}GB free / {total:.1f}GB total")
            
            # Warning if memory usage is high
            if allocated > 20.0:  # More than 20GB used
                print(f"⚠️ HIGH MEMORY USAGE: {allocated:.2f}GB - cleanup recommended")

    def aggressive_cleanup(self):
        """Aggressive GPU memory cleanup between jobs"""
        print("🧹 Performing aggressive GPU cleanup...")
        
        # Python garbage collection
        gc.collect()
        
        # PyTorch cache cleanup
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Force synchronization
        torch.cuda.synchronize()
        
        self.log_gpu_memory("after cleanup")

    def generate_with_optimized_settings(self, prompt, job_type):
        """Generate with ENHANCED memory management for video generation"""
        config = self.job_type_mapping.get(job_type, self.job_type_mapping['image_fast'])
        
        print(f"⚡ {job_type.upper()} generation")
        print(f"📝 Prompt: {prompt}")
        print(f"🔧 Config: {config['sample_steps']} steps, {config['sample_guide_scale']} guidance")
        print(f"📺 Frames: {config['frame_num']} frames ({config['frame_num']/16:.2f}s video)")
        print(f"🎯 Expected: {config['expected_time']}s")
        
        # PRE-GENERATION: Aggressive cleanup
        self.aggressive_cleanup()
        
        job_id = str(uuid.uuid4())[:8]
        print(f"📁 Job ID: {job_id}")
        
        # Always generate as MP4 first (Wan 2.1 only outputs MP4)
        temp_video_filename = f"{job_type}_{job_id}.mp4"
        temp_video_path = self.temp_processing / temp_video_filename
        
        # Build command with memory-optimized parameters
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
        
        print(f"📁 Generating to: {temp_video_path.absolute()}")
        print(f"🔧 Memory-optimized command with {config['frame_num']} frames")
        
        # Clean environment
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'TORCH_USE_CUDA_DSA': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'  # Memory fragmentation fix
        })
        
        # Remove distributed training variables
        for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
            env.pop(key, None)
        
        original_cwd = os.getcwd()
        os.chdir(self.wan_path)
        
        try:
            start_time = time.time()
            self.log_gpu_memory("before generation")
            
            # EXTENDED TIMEOUT for video generation
            timeout = 180 if config['content_type'] == 'video' else 120
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            generation_time = time.time() - start_time
            print(f"⚡ Generation completed in {generation_time:.1f}s")
            self.log_gpu_memory("after generation")
            
            # Print generate.py output for debugging
            print(f"📝 generate.py stdout length: {len(result.stdout) if result.stdout else 0}")
            print(f"⚠️ generate.py stderr length: {len(result.stderr) if result.stderr else 0}")
            
            if result.stdout:
                print(f"📝 Last 500 chars of stdout: {result.stdout[-500:]}")
            if result.stderr:
                print(f"⚠️ Last 500 chars of stderr: {result.stderr[-500:]}")
            
            print(f"🔍 Return code: {result.returncode}")
            
            # Check for OOM errors specifically
            if "CUDA out of memory" in str(result.stderr):
                print("❌ CUDA OUT OF MEMORY ERROR DETECTED")
                print("🔧 Try reducing frame count or resolution")
                self.aggressive_cleanup()
                return None
            
            if result.returncode != 0:
                print(f"❌ Generation failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error details: {result.stderr[-1000:]}")
                self.aggressive_cleanup()
                return None
            
            # Enhanced file detection
            output_candidates = [
                temp_video_path,
                Path(self.wan_path) / temp_video_filename,
                Path(temp_video_filename),
                Path(f"{job_type}_{job_id}_temp.mp4"),
                Path(f"output.mp4"),
                Path(f"generated.mp4")
            ]
            
            actual_output_path = None
            
            # Try expected file names first
            for candidate in output_candidates:
                if candidate.exists():
                    actual_output_path = candidate
                    print(f"✅ Found output file: {candidate}")
                    break
            
            # If not found, look for newest MP4 file
            if not actual_output_path:
                print("🔍 Looking for newest MP4 file...")
                try:
                    mp4_files = list(Path('.').glob('*.mp4'))
                    if mp4_files:
                        newest_mp4 = max(mp4_files, key=lambda x: x.stat().st_mtime)
                        file_age = time.time() - newest_mp4.stat().st_mtime
                        if file_age < 60:  # Within last minute
                            actual_output_path = newest_mp4
                            print(f"✅ Found newest MP4: {newest_mp4} (created {file_age:.1f}s ago)")
                except Exception as e:
                    print(f"❌ Error finding newest MP4: {e}")
            
            if not actual_output_path:
                print("❌ Output file not found")
                self.aggressive_cleanup()
                return None
            
            # Move to expected location if needed
            if actual_output_path != temp_video_path:
                shutil.move(str(actual_output_path), str(temp_video_path))
                print(f"📁 Moved output from {actual_output_path} to {temp_video_path}")
            
            # Get file size
            file_size = temp_video_path.stat().st_size / 1024
            print(f"📊 Generated file: {file_size:.0f}KB")
            
            # POST-GENERATION: Immediate cleanup
            self.aggressive_cleanup()
            
            # Handle image extraction vs video output
            if config['content_type'] == 'image':
                print(f"🖼️ Extracting image frame from video...")
                return self.extract_frame_from_video(str(temp_video_path), job_id, job_type)
            else:
                print(f"🎥 Returning video file: {temp_video_path}")
                return str(temp_video_path)
            
        except subprocess.TimeoutExpired:
            print(f"❌ Generation timed out (>{timeout}s)")
            print("🔧 This usually indicates insufficient memory for video generation")
            self.aggressive_cleanup()
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            self.aggressive_cleanup()
            return None
        finally:
            os.chdir(original_cwd)
            # Final cleanup
            self.aggressive_cleanup()

    def extract_frame_from_video(self, video_path, job_id, job_type):
        """Extract frame for image jobs and save as PNG"""
        image_path = self.temp_processing / f"{job_type}_{job_id}.png"
        
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                # Convert BGR to RGB and save as PNG
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.save(str(image_path), "PNG", optimize=True, quality=95)
                
                file_size = os.path.getsize(image_path) / 1024
                print(f"✅ Frame extracted to PNG: {file_size:.0f}KB")
                
                # Clean up temporary video file
                try:
                    os.remove(video_path)
                    print("🗑️ Temporary video file cleaned up")
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
            
        config = self.job_type_mapping.get(job_type, self.job_type_mapping['image_fast'])
        storage_bucket = config['storage_bucket']
        content_type = config['content_type']
        file_extension = config['file_extension']
        
        timestamp = int(time.time())
        filename = f"job_{job_id}_{timestamp}_{job_type}.{file_extension}"
        full_path = f"{storage_bucket}/{user_id}/{filename}"
        
        mime_type = 'image/png' if content_type == 'image' else 'video/mp4'
        
        print(f"📤 Uploading to Supabase:")
        print(f"   Bucket: {storage_bucket}")
        print(f"   Path: {full_path}")
        print(f"   MIME: {mime_type}")
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
                file_size = len(file_data) / 1024
                print(f"📊 File size: {file_size:.0f}KB")
                
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
                
                print(f"📡 Upload response: {response.status_code}")
                if response.status_code not in [200, 201]:
                    print(f"❌ Upload error details: {response.text}")
                
                if response.status_code in [200, 201]:
                    print(f"✅ Upload successful to {storage_bucket}")
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
                    print("🗑️ Local file cleaned up")
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")

    def notify_completion(self, job_id, status, file_path=None, error_message=None):
        """Notify completion via callback"""
        data = {
            'jobId': job_id,
            'status': status,
            'filePath': file_path,
            'errorMessage': error_message
        }
        
        print(f"📞 Calling job-callback for job {job_id}: {status}")
        if file_path:
            print(f"   File path: {file_path}")
        
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
            
            print(f"📡 Callback response: {response.status_code}")
            if response.status_code != 200:
                print(f"❌ Callback error details: {response.text}")
            
            if response.status_code == 200:
                print("✅ Callback sent successfully")
            else:
                print(f"❌ Callback failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job(self, job_data):
        """Process job with enhanced memory management"""
        job_id = job_data.get('jobId')
        job_type = job_data.get('jobType')
        prompt = job_data.get('prompt')
        user_id = job_data.get('userId')
        
        if not all([job_id, job_type, user_id, prompt]):
            error_msg = "Missing required fields"
            print(f"❌ {error_msg}")
            self.notify_completion(job_id or 'unknown', 'failed', error_message=error_msg)
            return

        print(f"📥 Processing job: {job_id} ({job_type})")
        print(f"👤 User: {user_id}")
        print(f"📝 Prompt: {prompt[:50]}...")
        
        # PRE-JOB: Memory status
        self.log_gpu_memory("pre-job")
        
        start_time = time.time()
        
        try:
            # Generate content with enhanced memory management
            output_path = self.generate_with_optimized_settings(prompt, job_type)
            if output_path:
                print(f"✅ Generation successful: {Path(output_path).name}")
                
                # Upload to Supabase
                supa_path = self.upload_to_supabase(output_path, job_type, user_id, job_id)
                if supa_path:
                    duration = time.time() - start_time
                    expected = self.job_type_mapping[job_type]['expected_time']
                    
                    if duration <= expected * 2:
                        print(f"🎉 Job completed in {duration:.1f}s (expected {expected}s) ✅")
                    else:
                        print(f"⚠️ Job completed in {duration:.1f}s (expected {expected}s) - slower than expected")
                    
                    self.notify_completion(job_id, 'completed', supa_path)
                    return
                else:
                    print("❌ Upload to Supabase failed")
            else:
                print("❌ Generation failed")
                    
            self.notify_completion(job_id, 'failed', error_message="Generation or upload failed")
            
        except Exception as e:
            print(f"❌ Job processing error: {e}")
            self.notify_completion(job_id, 'failed', error_message=str(e))
        finally:
            # POST-JOB: Aggressive cleanup
            self.aggressive_cleanup()

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
        """Main loop with memory management"""
        print("⏳ Waiting for jobs...")
        print("🚀 MEMORY-OPTIMIZED WORKER:")
        print("🔧 KEY IMPROVEMENTS:")
        print("   • Aggressive GPU cleanup between jobs")
        print("   • Reduced frame counts for video generation")
        print("   • Enhanced OOM detection and handling")
        print("   • Memory fragmentation fixes")
        
        for job_type, config in self.job_type_mapping.items():
            print(f"   • {job_type}: {config['description']}")
        
        print("\n🔥 System ready for memory-intensive video generation!")
        
        job_count = 0
        
        while True:
            job = self.poll_queue()
            if job:
                job_count += 1
                print(f"\n🎯 Processing job #{job_count}")
                self.process_job(job)
                
                # CRITICAL: Aggressive cleanup after each job
                print("🧹 Post-job cleanup...")
                self.aggressive_cleanup()
                print("=" * 60)
            else:
                time.sleep(5)

if __name__ == "__main__":
    print("🚀 Starting OurVidz MEMORY-OPTIMIZED Worker")
    print("🔧 KEY FIX: GPU memory management for video generation")
    
    # Verify environment
    required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        exit(1)
    
    print(f"🔍 Environment check:")
    print(f"   CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"   PYTORCH_CUDA_ALLOC_CONF: {os.getenv('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"   Mode: Memory-optimized video generation")
    
    try:
        worker = VideoWorker()
        worker.run()
    except Exception as e:
        print(f"❌ Worker failed: {e}")
        exit(1)
