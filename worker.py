# worker.py - Simplified and Stable (No Startup Preload)
import os
import json
import time
import requests
import subprocess
import uuid
import shutil
from pathlib import Path
from PIL import Image
import cv2

class VideoWorker:
    def __init__(self):
        print("🚀 OurVidz Worker initialized (NO STARTUP PRELOAD)")

        self.ffmpeg_available = shutil.which('ffmpeg') is not None
        print(f"🔧 FFmpeg Available: {self.ffmpeg_available}")
        self.detect_gpu()

        self.model_path = '/tmp/models/wan2.1-t2v-1.3b'
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

        print("🎬 Worker ready (model will load on first job)")

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

    def generate(self, prompt, job_type):
        config = self.job_configs.get(job_type)
        if not config:
            print(f"❌ Unknown job type: {job_type}")
            return None

        job_id = str(uuid.uuid4())[:8]
        memory_before = self.check_gpu_memory()
        warm_start = memory_before > 5000

        print(f"⚡ {job_type.upper()} generation ({'WARM' if warm_start else 'COLD'} start)")
        print(f"📝 Prompt: {prompt}")

        output_filename = f"{job_type}_{job_id}.mp4"
        cmd = [
            "python", "generate.py",
            "--task", "t2v-1.3B",
            "--size", config['size'],
            "--ckpt_dir", self.model_path,
            "--prompt", prompt,
            "--save_file", output_filename,
            "--sample_steps", str(config['sample_steps']),
            "--sample_guide_scale", str(config['sample_guide_scale']),
            "--frame_num", str(config['frame_num'])
        ]

        os.chdir("/workspace/Wan2.1")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"❌ Generation failed: {result.stderr}")
                return None
            path = f"/workspace/Wan2.1/{output_filename}"
            if not os.path.exists(path):
                print("❌ Output file not found")
                return None
            if 'image' in job_type:
                return self.extract_frame_from_video(path, job_id, job_type)
            return path
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return None

    def extract_frame_from_video(self, video_path, job_id, job_type):
        image_path = f"/workspace/Wan2.1/{job_type}_{job_id}.png"
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(frame_rgb).save(image_path, "PNG", optimize=True)
                os.remove(video_path)
                return image_path
        except Exception as e:
            print(f"❌ Frame extraction error: {e}")
        return None

    def upload_to_supabase(self, file_path, job_type, user_id, job_id):
        if not os.path.exists(file_path): return None
        filename = f"job_{job_id}_{int(time.time())}_{job_type}.{'png' if 'image' in job_type else 'mp4'}"
        full_path = f"{job_type}/{user_id}/{filename}"
        with open(file_path, 'rb') as f:
            r = requests.post(
                f"{self.supabase_url}/storage/v1/object/{full_path}",
                files={'file': (filename, f)},
                headers={'Authorization': f"Bearer {self.supabase_service_key}", 'x-upsert': 'true'}
            )
            if r.status_code in [200, 201]:
                print(f"✅ Uploaded to Supabase: {full_path}")
                return f"{user_id}/{filename}"
        print("❌ Upload failed")
        return None

    def notify_completion(self, job_id, status, file_path=None, error_message=None):
        data = {'jobId': job_id, 'status': status, 'filePath': file_path, 'errorMessage': error_message}
        try:
            r = requests.post(f"{self.supabase_url}/functions/v1/job-callback", json=data,
                              headers={'Authorization': f"Bearer {self.supabase_service_key}", 'Content-Type': 'application/json'})
            print("✅ Callback sent" if r.status_code == 200 else "❌ Callback failed")
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job(self, job_data):
        job_id = job_data.get('jobId')
        job_type = job_data.get('jobType')
        prompt = job_data.get('prompt', 'person walking')
        user_id = job_data.get('userId')
        if not all([job_id, job_type, user_id]):
            self.notify_completion(job_id or 'unknown', 'failed', error_message="Missing fields")
            return

        print(f"📥 Processing job: {job_id} ({job_type})")
        output_path = self.generate(prompt, job_type)
        if output_path:
            supa_path = self.upload_to_supabase(output_path, job_type, user_id, job_id)
            if supa_path:
                self.notify_completion(job_id, 'completed', supa_path)
                try: os.remove(output_path)
                except: pass
                return
        self.notify_completion(job_id, 'failed', error_message="Generation or upload failed")

    def poll_queue(self):
        try:
            r = requests.get(f"{self.redis_url}/rpop/job_queue",
                             headers={'Authorization': f"Bearer {self.redis_token}"}, timeout=10)
            if r.status_code == 200 and r.json().get('result'):
                return json.loads(r.json()['result'])
        except Exception as e:
            print(f"❌ Poll error: {e}")
        return None

    def run(self):
        print("⏳ Waiting for jobs...")
        while True:
            job = self.poll_queue()
            if job:
                self.process_job(job)
                print("⏳ Job complete, checking queue...")
            else:
                time.sleep(5)

if __name__ == "__main__":
    worker = VideoWorker()
    worker.run()
