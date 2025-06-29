# worker.py - Production OurVidz GPU Worker (Fixed for generate.py)
import os
import json
import time
import requests
import subprocess
import uuid
import shutil
from pathlib import Path
import torch

class VideoWorker:
    def __init__(self):
        print("🚀 OurVidz Production Worker initialized")
        
        # Paths
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_script_path = "/workspace/Wan2.1"
        self.output_dir = "/workspace/output"
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        # Validate environment
        self.validate_environment()
        self.setup_gpu_optimizations()
        
        print("✅ Worker ready for production")

    def validate_environment(self):
        """Validate all required components"""
        print("🔍 Validating environment...")
        
        # Check model
        if os.path.exists(self.model_path):
            print("✅ Model directory found")
        else:
            print("❌ Model directory not found")
            raise Exception("Model not found")
            
        # Check Wan 2.1 - use generate.py instead of scripts/sample_wan.py
        generate_script = f"{self.wan_script_path}/generate.py"
        if os.path.exists(generate_script):
            print("✅ Wan 2.1 generate.py found")
        else:
            print("❌ Wan 2.1 generate.py not found")
            raise Exception("Wan 2.1 generate.py not found")
            
        # Check environment variables
        required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN']
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            print(f"❌ Missing environment variables: {missing}")
            raise Exception(f"Missing required environment variables: {missing}")
        else:
            print("✅ All environment variables configured")
            
        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {gpu_name} ({total_memory:.1f}GB)")
        else:
            print("❌ CUDA not available")
            raise Exception("CUDA not available")

    def setup_gpu_optimizations(self):
        """Setup GPU optimizations for best performance"""
        print("🔧 Setting up GPU optimizations...")
        
        try:
            # PyTorch optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Environment optimizations
            os.environ.update({
                'CUDA_LAUNCH_BLOCKING': '0',
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
                'TORCH_USE_CUDA_DSA': '1',
            })
            
            # Warm up GPU
            if torch.cuda.is_available():
                warmup_tensor = torch.ones((1000, 1000), device='cuda')
                for _ in range(5):
                    result = torch.matmul(warmup_tensor, warmup_tensor)
                    torch.cuda.synchronize()
                del warmup_tensor, result
                torch.cuda.empty_cache()
                
            print("✅ GPU optimizations complete")
            
        except Exception as e:
            print(f"⚠️ GPU optimization warning: {e}")

    def enhance_prompt(self, original_prompt: str, character_description: str = None) -> str:
        """Simple prompt enhancement for Phase 1"""
        enhanced = f"High quality cinematic video, {original_prompt}"
        if character_description:
            enhanced = f"High quality cinematic video featuring {character_description}, {original_prompt}"
        
        # Add cinematic descriptors
        enhanced += ", professional lighting, smooth motion, detailed"
        
        print(f"📝 Enhanced prompt: {enhanced[:100]}...")
        return enhanced

    def generate_video(self, prompt: str, job_id: str) -> str:
        """Generate video using Wan 2.1 generate.py with optimized parameters"""
        print(f"🎬 Generating video for job {job_id}")
        print(f"📝 Prompt: {prompt}")
        
        output_filename = f"{job_id}_video.mp4"
        output_path = f"{self.output_dir}/{output_filename}"
        
        # Use generate.py with proven working parameters
        cmd = [
            "python", "generate.py",
            "--task", "t2v-1.3B",
            "--ckpt_dir", self.model_path,
            "--prompt", prompt,
            "--save_file", output_path,
            "--size", "832*480",  # Proven working landscape format
            "--frame_num", "80",  # 5 seconds at 16fps
            "--sample_steps", "25",  # Good quality/speed balance
            "--sample_guide_scale", "6.0",  # Optimal guidance
        ]
        
        print("🎥 Starting Wan 2.1 generation...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            
            # Run generation with optimized environment
            env = os.environ.copy()
            env.update({
                'CUDA_LAUNCH_BLOCKING': '0',
                'OMP_NUM_THREADS': '8',
                'MKL_NUM_THREADS': '8',
            })
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.wan_script_path,
                timeout=600,  # 10 minute timeout
                env=env
            )
            
            generation_time = time.time() - start_time
            
            if result.returncode == 0:
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                    print(f"✅ Video generated in {generation_time:.1f}s: {file_size:.1f}MB")
                    return output_path
                else:
                    print(f"❌ Generation completed but file not found")
                    print(f"stdout: {result.stdout}")
                    return None
            else:
                print(f"❌ Video generation failed: {result.stderr}")
                print(f"stdout: {result.stdout}")
                return None
                
        except Exception as e:
            print(f"❌ Video generation error: {e}")
            return None

    def upload_to_supabase(self, file_path: str, filename: str) -> str:
        """Upload file to Supabase storage"""
        storage_path = f"videos-final/{filename}"
        
        try:
            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': file},
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                    },
                    timeout=120
                )
            
            if response.status_code == 200:
                public_url = f"{self.supabase_url}/storage/v1/object/public/{storage_path}"
                print(f"✅ File uploaded: {public_url}")
                return public_url
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Supabase upload error: {e}")
            return None

    def notify_completion(self, job_id: str, status: str, output_url: str = None, error_message: str = None):
        """Notify Supabase of job completion"""
        try:
            callback_data = {
                'jobId': job_id,
                'status': status,
                'outputUrl': output_url,
                'errorMessage': error_message
            }
            
            response = requests.post(
                f"{self.supabase_url}/functions/v1/job-callback",
                json=callback_data,
                headers={
                    'Authorization': f"Bearer {self.supabase_service_key}",
                    'Content-Type': 'application/json'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"✅ Job {job_id} callback sent successfully")
            else:
                print(f"❌ Callback failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job(self, job_data: dict):
        """Process a single job"""
        job_id = job_data.get('jobId')
        job_type = job_data.get('jobType')
        prompt = job_data.get('prompt')
        user_id = job_data.get('userId')
        character_description = job_data.get('characterDescription')
        
        print(f"📋 Processing job: {job_id} ({job_type})")
        
        if not all([job_id, job_type, prompt, user_id]):
            error_msg = "Missing required job fields"
            print(f"❌ {error_msg}")
            self.notify_completion(job_id or 'unknown', 'failed', error_message=error_msg)
            return

        try:
            start_time = time.time()
            
            if job_type == 'enhance':
                # Phase 1: Simple enhancement
                enhanced_prompt = self.enhance_prompt(prompt, character_description)
                print(f"✅ Prompt enhanced")
                self.notify_completion(job_id, 'completed')
                
            elif job_type == 'preview':
                # Phase 1: Skip preview, go to video
                print(f"⏭️ Skipping preview for Phase 1")
                self.notify_completion(job_id, 'completed')
                
            elif job_type == 'video':
                # Generate final video
                enhanced_prompt = self.enhance_prompt(prompt, character_description)
                
                video_path = self.generate_video(enhanced_prompt, job_id)
                if video_path:
                    filename = f"{job_id}_final.mp4"
                    upload_url = self.upload_to_supabase(video_path, filename)
                    
                    # Cleanup local file
                    try:
                        os.remove(video_path)
                    except:
                        pass
                    
                    if upload_url:
                        duration = time.time() - start_time
                        print(f"🎉 Job {job_id} completed in {duration:.1f}s")
                        self.notify_completion(job_id, 'completed', upload_url)
                    else:
                        self.notify_completion(job_id, 'failed', error_message="Upload failed")
                else:
                    self.notify_completion(job_id, 'failed', error_message="Generation failed")
            else:
                self.notify_completion(job_id, 'failed', error_message=f"Unknown job type: {job_type}")
                
        except Exception as e:
            print(f"❌ Job processing error: {e}")
            self.notify_completion(job_id, 'failed', error_message=str(e))

    def poll_redis_queue(self):
        """Poll Redis queue for jobs"""
        try:
            response = requests.get(
                f"{self.redis_url}/rpop/job-queue",
                headers={'Authorization': f"Bearer {self.redis_token}"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('result'):
                    job_json = result['result']
                    job_data = json.loads(job_json)
                    return job_data
            
            return None
            
        except Exception as e:
            print(f"⚠️ Redis poll error: {e}")
            return None

    def cleanup_old_files(self):
        """Cleanup old output files"""
        try:
            current_time = time.time()
            cleaned = 0
            
            for file_path in Path(self.output_dir).glob("*"):
                if file_path.is_file():
                    if (current_time - file_path.stat().st_mtime) > 1800:  # 30 minutes
                        try:
                            file_path.unlink()
                            cleaned += 1
                        except:
                            pass
                            
            if cleaned > 0:
                print(f"🧹 Cleaned up {cleaned} old files")
                
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

    def run(self):
        """Main worker loop"""
        print("🎬 OurVidz Production Worker started!")
        print("⏳ Waiting for jobs from Redis queue...")
        
        job_count = 0
        last_cleanup = time.time()
        
        while True:
            try:
                # Periodic cleanup
                if time.time() - last_cleanup > 600:  # Every 10 minutes
                    self.cleanup_old_files()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    last_cleanup = time.time()
                
                # Poll for jobs
                job_data = self.poll_redis_queue()
                
                if job_data:
                    job_count += 1
                    print(f"🎯 Processing job #{job_count}")
                    self.process_job(job_data)
                else:
                    print("💤 No jobs, waiting...")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"❌ Worker error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    print("🚀 Starting OurVidz Production Worker")
    
    try:
        worker = VideoWorker()
        worker.run()
    except Exception as e:
        print(f"❌ Worker failed to start: {e}")
        exit(1)
