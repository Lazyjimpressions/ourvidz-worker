# simple_worker.py - Simplified worker for current environment
import os
import json
import time
import requests
import subprocess
import uuid
from pathlib import Path

class SimpleVideoWorker:
    def __init__(self):
        print("🚀 OurVidz Simple Worker initialized")
        
        # Paths
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_script_path = "/workspace/Wan2.1"
        self.output_dir = "/workspace/output"
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Environment variables (with fallback for testing)
        self.supabase_url = os.getenv('SUPABASE_URL', 'https://ulmdmzhcdwfadbvfpckt.supabase.co')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        print(f"🔧 Model path: {self.model_path}")
        print(f"🔧 Wan scripts: {self.wan_script_path}")
        print(f"🔧 Environment variables configured: {bool(self.supabase_service_key)}")
        
        self.verify_setup()

    def verify_setup(self):
        """Verify that everything is set up correctly"""
        print("🔍 Verifying setup...")
        
        # Check model exists
        if os.path.exists(self.model_path):
            print("✅ Model directory found")
        else:
            print("❌ Model directory not found")
            
        # Check Wan 2.1 scripts
        sample_script = f"{self.wan_script_path}/scripts/sample_wan.py"
        if os.path.exists(sample_script):
            print("✅ Wan 2.1 scripts found")
        else:
            print("❌ Wan 2.1 scripts not found")
            
        # Check environment variables
        if self.supabase_service_key:
            print("✅ Supabase credentials configured")
        else:
            print("⚠️ Supabase credentials missing - will run in test mode")
            
        if self.redis_url and self.redis_token:
            print("✅ Redis credentials configured")
        else:
            print("⚠️ Redis credentials missing - will run in test mode")

    def generate_video(self, prompt: str, output_filename: str) -> bool:
        """Generate video using Wan 2.1 with proven parameters"""
        print(f"🎬 Generating video: {prompt}")
        
        output_path = f"{self.output_dir}/{output_filename}"
        
        # Use the proven parameters from your successful test
        cmd = [
            "python", f"{self.wan_script_path}/scripts/sample_wan.py",
            "--model_dir", self.model_path,
            "--task", "t2v-1.3B",
            "--prompt", prompt,
            "--size", "832*480",  # Proven working size
            "--num_frames", "80",  # 5 seconds at 16fps
            "--num_inference_steps", "25",
            "--guidance_scale", "6.0",
            "--sample_shift", "8",
            "--offload_model", "True",
            "--t5_cpu",
            "--save_path", output_path
        ]
        
        print("🎥 Running Wan 2.1 generation...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.wan_script_path,
                timeout=600  # 10 minute timeout
            )
            
            generation_time = time.time() - start_time
            
            if result.returncode == 0:
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                    print(f"✅ Video generated in {generation_time:.1f}s: {file_size:.1f}MB")
                    return True
                else:
                    print(f"❌ Generation completed but file not found: {output_path}")
                    print(f"stdout: {result.stdout}")
                    return False
            else:
                print(f"❌ Video generation failed: {result.stderr}")
                print(f"stdout: {result.stdout}")
                return False
                
        except Exception as e:
            print(f"❌ Video generation error: {e}")
            return False

    def upload_to_supabase(self, file_path: str, storage_path: str) -> str:
        """Upload file to Supabase storage"""
        if not self.supabase_service_key:
            print("⚠️ No Supabase credentials - skipping upload")
            return file_path
            
        try:
            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': file},
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                    }
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
        if not self.supabase_service_key:
            print(f"⚠️ No Supabase credentials - would notify: {job_id} = {status}")
            return
            
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
                }
            )
            
            if response.status_code == 200:
                print(f"✅ Job {job_id} callback sent successfully")
            else:
                print(f"❌ Callback failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job(self, job_data: dict):
        """Process a single job"""
        job_id = job_data.get('jobId', str(uuid.uuid4())[:8])
        job_type = job_data.get('jobType', 'video')
        prompt = job_data.get('prompt', 'A woman walking')
        user_id = job_data.get('userId', 'test-user')
        
        print(f"📋 Processing job: {job_id}")
        print(f"📝 Prompt: {prompt}")
        
        try:
            # Generate video
            filename = f"{job_id}_video.mp4"
            success = self.generate_video(prompt, filename)
            
            if success:
                file_path = f"{self.output_dir}/{filename}"
                
                # Upload to Supabase
                storage_path = f"videos-final/{filename}"
                upload_url = self.upload_to_supabase(file_path, storage_path)
                
                if upload_url:
                    print(f"🎉 Job {job_id} completed successfully")
                    self.notify_completion(job_id, 'completed', upload_url)
                else:
                    print(f"❌ Job {job_id} upload failed")
                    self.notify_completion(job_id, 'failed', error_message="Upload failed")
            else:
                print(f"❌ Job {job_id} generation failed")
                self.notify_completion(job_id, 'failed', error_message="Generation failed")
                
        except Exception as e:
            print(f"❌ Job {job_id} processing error: {e}")
            self.notify_completion(job_id, 'failed', error_message=str(e))

    def poll_redis_queue(self):
        """Poll Redis queue for jobs"""
        if not (self.redis_url and self.redis_token):
            return None
            
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

    def test_generation(self):
        """Test video generation with a simple prompt"""
        print("🧪 Running test generation...")
        
        test_prompt = "A woman walking confidently down a busy street"
        test_filename = f"test_{int(time.time())}.mp4"
        
        success = self.generate_video(test_prompt, test_filename)
        
        if success:
            print("✅ Test generation successful!")
            return True
        else:
            print("❌ Test generation failed!")
            return False

    def run(self):
        """Main worker loop"""
        print("🎬 OurVidz Simple Worker started!")
        
        # Test generation first if no environment variables
        if not (self.supabase_service_key and self.redis_url):
            print("⚠️ Running in test mode - no queue integration")
            if self.test_generation():
                print("✅ Worker is ready for production with environment variables")
            else:
                print("❌ Worker needs troubleshooting")
            return
        
        print("⏳ Waiting for jobs from Redis queue...")
        
        while True:
            try:
                job_data = self.poll_redis_queue()
                
                if job_data:
                    self.process_job(job_data)
                else:
                    print("💤 No jobs, waiting...")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"❌ Worker error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    print("🚀 Starting OurVidz Simple Worker")
    
    worker = SimpleVideoWorker()
    worker.run()
