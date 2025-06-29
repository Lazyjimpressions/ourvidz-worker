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

        # Model persistence - KEY OPTIMIZATION
        self.wan_pipeline = None
        self.model_loaded = False
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        
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

    def load_wan_model_once(self):
        """Load Wan 2.1 model once and keep in memory - KEY OPTIMIZATION"""
        if self.model_loaded:
            print("✅ Model already loaded in memory")
            return True
            
        print("🔄 Loading Wan 2.1 model (one-time setup, ~90 seconds)...")
        load_start = time.time()
        
        try:
            # Add Wan2.1 to Python path
            import sys
            sys.path.insert(0, '/workspace/Wan2.1')
            
            # Import Wan modules
            from wan.pipelines import WanVideoPipeline
            from wan.utils import setup_seed
            
            # Setup reproducible generation
            setup_seed(42)
            
            # Load model directly on GPU
            print(f"📁 Loading from: {self.model_path}")
            self.wan_pipeline = WanVideoPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map='cuda:0'
            ).to(self.device)
            
            load_time = time.time() - load_start
            self.model_loaded = True
            
            print(f"✅ Wan 2.1 model loaded in {load_time:.1f}s and ready for fast generation")
            print(f"🔥 Model device: {next(self.wan_pipeline.parameters()).device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_with_loaded_model(self, prompt, job_type):
        """Generate using pre-loaded model - FAST GENERATION"""
        if not self.model_loaded:
            print("🔄 Model not loaded, loading now...")
            if not self.load_wan_model_once():
                print("❌ Failed to load model")
                return None
                
        config = self.job_type_mapping.get(job_type, self.job_type_mapping['image_fast'])
        
        print(f"⚡ {job_type.upper()} generation (PRE-LOADED MODEL)")
        print(f"📝 Prompt: {prompt}")
        print(f"🔧 Config: {config['sample_steps']} steps, {config['sample_guide_scale']} guidance, {config['size']}")
        print(f"🎯 Expected: {config['expected_time']}s (model already loaded)")
        
        generation_start = time.time()
        
        try:
            # Force GPU context
            with torch.cuda.device(0):
                # Generate with pre-loaded pipeline
                result = self.wan_pipeline(
                    prompt=prompt,
                    height=int(config['size'].split('*')[1]),
                    width=int(config['size'].split('*')[0]),
                    num_frames=config['frame_num'],
                    num_inference_steps=config['sample_steps'],
                    guidance_scale=config['sample_guide_scale'],
                    generator=torch.Generator(device='cuda').manual_seed(42)
                )
            
            generation_time = time.time() - generation_start
            expected_time = config['expected_time']
            
            if generation_time < expected_time:
                speedup = ((expected_time - generation_time) / expected_time) * 100
                print(f"🚀 Generation completed in {generation_time:.1f}s ({speedup:.1f}% faster than expected!)")
            else:
                print(f"⚡ Generation completed in {generation_time:.1f}s")
            
            # Save result
            job_id = str(uuid.uuid4())[:8]
            
            if config['content_type'] == 'image':
                # Save as PNG for images
                output_file = self.temp_processing / f"{job_type}_{job_id}.png"
                result.frames[0][0].save(str(output_file), "PNG", optimize=True)
                print(f"📊 Image saved: {output_file}")
            else:
                # Save as MP4 for videos
                output_file = self.temp_processing / f"{job_type}_{job_id}.mp4"
                import imageio
                with imageio.get_writer(str(output_file), fps=16) as writer:
                    for frame in result.frames[0]:
                        import numpy as np
                        writer.append_data(np.array(frame))
                print(f"📊 Video saved: {output_file}")
            
            file_size = os.path.getsize(output_file) / 1024
            print(f"📊 File size: {file_size:.0f}KB")
            
            return str(output_file)
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            return None

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
            # Generate with pre-loaded model (fast!)
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
        """Main worker loop with model pre-loading"""
        print("🚀 Starting optimized worker with model pre-loading...")
        
        # Load model once at startup
        print("🔄 Pre-loading Wan 2.1 model for fast generation...")
        if not self.load_wan_model_once():
            print("❌ Failed to load model at startup - exiting")
            return
            
        print("⏳ Waiting for jobs (MODEL PRE-LOADED FOR FAST GENERATION)...")
        print("🎯 OPTIMIZED Job Types:")
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
                print(f"🎯 Processing optimized job #{job_count}")
                self.process_job(job)
            else:
                time.sleep(5)

if __name__ == "__main__":
    print("🚀 Starting OurVidz GPU-OPTIMIZED Worker with Model Persistence")
    
    try:
        worker = VideoWorker()
        worker.run()
    except Exception as e:
        print(f"❌ Worker failed to start: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
