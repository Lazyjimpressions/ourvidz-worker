#!/usr/bin/env python3
# worker.py - OurVidz GPU Worker for RunPod

import os
import json
import time
import torch
import requests
import subprocess
from PIL import Image
from pathlib import Path
from typing import Optional, List
import traceback

class OurVidzWorker:
    def __init__(self):
        """Initialize the OurVidz worker"""
        self.model_path = Path("/workspace/models")
        
        # Model instances (loaded on demand)
        self.wan_t2v_pipeline = None
        self.wan_i2v_pipeline = None
        self.mistral_model = None
        self.mistral_tokenizer = None
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        # Auto-shutdown tracking
        self.idle_start_time = None
        self.max_idle_minutes = 10
        
        # Validate environment
        self._validate_environment()
        
        print("🚀 OurVidz Worker initialized")
        self._log_system_info()

    def _validate_environment(self):
        """Validate required environment variables"""
        required_vars = [
            'SUPABASE_URL', 'SUPABASE_SERVICE_KEY',
            'UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing environment variables: {missing}")

    def _log_system_info(self):
        """Log system information"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        print(f"📁 Model path: {self.model_path}")
        print(f"🔗 Supabase: {self.supabase_url}")

    def _log_gpu_memory(self, context=""):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🔥 GPU Memory {context}- Used: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.1f}GB")

    def load_mistral(self):
        """Load Mistral 7B for prompt enhancement"""
        if self.mistral_model is None:
            print("📝 Loading Mistral 7B...")
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                mistral_path = self.model_path / "mistral"
                
                self.mistral_tokenizer = AutoTokenizer.from_pretrained(
                    mistral_path if mistral_path.exists() else "mistralai/Mistral-7B-Instruct-v0.2",
                    cache_dir=str(mistral_path),
                    local_files_only=mistral_path.exists()
                )
                
                self.mistral_model = AutoModelForCausalLM.from_pretrained(
                    mistral_path if mistral_path.exists() else "mistralai/Mistral-7B-Instruct-v0.2",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=str(mistral_path),
                    local_files_only=mistral_path.exists()
                )
                
                print("✅ Mistral 7B loaded")
                self._log_gpu_memory("after Mistral load")
                
            except Exception as e:
                print(f"❌ Failed to load Mistral 7B: {e}")
                raise

    def unload_mistral(self):
        """Free Mistral memory"""
        if self.mistral_model is not None:
            print("🗑️ Unloading Mistral 7B...")
            del self.mistral_model
            del self.mistral_tokenizer
            self.mistral_model = None
            self.mistral_tokenizer = None
            torch.cuda.empty_cache()
            print("✅ Mistral 7B unloaded")
            self._log_gpu_memory("after Mistral unload")

    def load_wan_t2v(self):
        """Load Wan 2.1 14B Text-to-Video"""
        if self.wan_t2v_pipeline is None:
            print("🎥 Loading Wan 2.1 14B Text-to-Video...")
            try:
                from diffusers import DiffusionPipeline
                
                wan_path = self.model_path / "wan_t2v"
                
                self.wan_t2v_pipeline = DiffusionPipeline.from_pretrained(
                    wan_path if wan_path.exists() else "Wan-AI/Wan2.1-T2V-14B",
                    torch_dtype=torch.float16,
                    cache_dir=str(wan_path),
                    local_files_only=wan_path.exists()
                ).to("cuda")
                
                print("✅ Wan 2.1 T2V loaded")
                self._log_gpu_memory("after Wan T2V load")
                
            except Exception as e:
                print(f"❌ Failed to load Wan 2.1 T2V: {e}")
                # Fallback to Stable Diffusion for testing
                print("🔄 Loading fallback model...")
                from diffusers import StableDiffusionPipeline
                self.wan_t2v_pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16
                ).to("cuda")
                print("✅ Fallback model loaded")

    def unload_wan_models(self):
        """Free all Wan 2.1 memory"""
        if self.wan_t2v_pipeline is not None:
            print("🗑️ Unloading Wan 2.1 models...")
            del self.wan_t2v_pipeline
            self.wan_t2v_pipeline = None
            torch.cuda.empty_cache()
            print("✅ Wan 2.1 models unloaded")
            self._log_gpu_memory("after Wan unload")

    def enhance_prompt(self, original_prompt: str, character_description: str = None) -> str:
        """Enhance prompt using Mistral 7B"""
        print(f"📝 Enhancing prompt: {original_prompt[:50]}...")
        
        try:
            self.load_mistral()
            
            system_prompt = """You are an expert at converting casual text into detailed video generation prompts.

Create a cinematic, vivid prompt optimized for AI video generation. Focus on:
- Visual details and composition
- Lighting and atmosphere
- Movement and camera work
- Realistic physics and motion

Keep under 200 words. Make it specific and cinematic."""

            if character_description:
                system_prompt += f"\n\nCharacter to include: {character_description}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_prompt}
            ]

            input_text = self.mistral_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.mistral_tokenizer(input_text, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = self.mistral_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.mistral_tokenizer.eos_token_id
                )

            enhanced = self.mistral_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            self.unload_mistral()
            
            result = enhanced.strip()
            print(f"✅ Enhanced prompt: {result[:100]}...")
            return result
            
        except Exception as e:
            print(f"❌ Prompt enhancement failed: {e}")
            self.unload_mistral()
            return original_prompt

    def generate_preview(self, prompt: str) -> Optional[Image.Image]:
        """Generate preview frame"""
        print(f"🖼️ Generating preview for: {prompt[:50]}...")
        
        try:
            self.load_wan_t2v()
            
            # Generate single image
            if hasattr(self.wan_t2v_pipeline, '__call__'):
                result = self.wan_t2v_pipeline(
                    prompt=prompt,
                    height=720,
                    width=1280,
                    num_inference_steps=20,
                    guidance_scale=7.5
                )
                
                # Handle different pipeline types
                if hasattr(result, 'images'):
                    image = result.images[0]
                elif hasattr(result, 'frames'):
                    image = result.frames[0][0]
                else:
                    image = result[0]
            
            self.unload_wan_models()
            print("✅ Preview generated")
            return image
            
        except Exception as e:
            print(f"❌ Preview generation failed: {e}")
            self.unload_wan_models()
            return None

    def generate_video(self, prompt: str) -> Optional[List[Image.Image]]:
        """Generate 5-second video (placeholder for now)"""
        print(f"🎥 Generating video for: {prompt[:50]}...")
        
        try:
            # For now, generate multiple frames from single image
            preview = self.generate_preview(prompt)
            if preview:
                # Create 80 frames (5 seconds at 16fps) by duplicating
                frames = [preview] * 80
                print("✅ Video frames generated")
                return frames
            
            return None
            
        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            return None

    def save_and_upload_image(self, image: Image.Image, filename: str) -> Optional[str]:
        """Save and upload image to Supabase"""
        try:
            local_path = f"/tmp/{filename}"
            image.save(local_path, "PNG", quality=95)
            
            upload_url = self._upload_to_supabase(local_path, f"scene-previews/{filename}")
            
            os.remove(local_path)
            return upload_url
            
        except Exception as e:
            print(f"❌ Image upload failed: {e}")
            return None

    def save_and_upload_video(self, frames: List[Image.Image], filename: str) -> Optional[str]:
        """Convert frames to MP4 and upload"""
        try:
            temp_dir = f"/tmp/frames_{int(time.time())}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save frames
            for i, frame in enumerate(frames):
                frame.save(f"{temp_dir}/frame_{i:04d}.png")
            
            # Create MP4 with FFmpeg
            output_path = f"/tmp/{filename}"
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", "16",
                "-i", f"{temp_dir}/frame_%04d.png",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ], check=True)
            
            upload_url = self._upload_to_supabase(output_path, f"videos-final/{filename}")
            
            # Cleanup
            subprocess.run(["rm", "-rf", temp_dir])
            os.remove(output_path)
            
            return upload_url
            
        except Exception as e:
            print(f"❌ Video upload failed: {e}")
            return None

    def _upload_to_supabase(self, file_path: str, storage_path: str) -> str:
        """Upload file to Supabase storage"""
        try:
            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': file},
                    headers={'Authorization': f"Bearer {self.supabase_service_key}"}
                )
            
            if response.status_code == 200:
                return f"{self.supabase_url}/storage/v1/object/public/{storage_path}"
            else:
                raise Exception(f"Upload failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Supabase upload error: {e}")
            raise

    def _notify_completion(self, job_id: str, status: str, output_url: str = None, error_message: str = None):
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
                }
            )
            
            if response.status_code == 200:
                print(f"✅ Job {job_id} callback sent")
            else:
                print(f"❌ Callback failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def _get_next_job(self):
        """Get next job from Redis queue"""
        try:
            response = requests.get(
                f"{self.redis_url}/brpop/job-queue/5",
                headers={'Authorization': f"Bearer {self.redis_token}"}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('result'):
                    queue_name, job_json = result['result']
                    return json.loads(job_json)
            
            return None
            
        except Exception as e:
            print(f"❌ Redis error: {e}")
            return None

    def process_job(self, job_data: dict):
        """Process a single job"""
        job_id = job_data['jobId']
        job_type = job_data['jobType']
        
        print(f"🔄 Processing job {job_id} ({job_type})")
        
        try:
            if job_type == 'enhance':
                enhanced = self.enhance_prompt(
                    job_data['prompt'], 
                    job_data.get('characterDescription')
                )
                self._notify_completion(job_id, 'completed')
                
            elif job_type == 'preview':
                preview = self.generate_preview(job_data['prompt'])
                if preview:
                    filename = f"{job_data['videoId']}_preview.png"
                    upload_url = self.save_and_upload_image(preview, filename)
                    if upload_url:
                        self._notify_completion(job_id, 'completed', upload_url)
                    else:
                        raise Exception("Upload failed")
                else:
                    raise Exception("Generation failed")
                    
            elif job_type == 'video':
                frames = self.generate_video(job_data['prompt'])
                if frames:
                    filename = f"{job_data['videoId']}_final.mp4"
                    upload_url = self.save_and_upload_video(frames, filename)
                    if upload_url:
                        self._notify_completion(job_id, 'completed', upload_url)
                    else:
                        raise Exception("Upload failed")
                else:
                    raise Exception("Generation failed")
            
            print(f"🎉 Job {job_id} completed")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Job {job_id} failed: {error_msg}")
            traceback.print_exc()
            self._notify_completion(job_id, 'failed', error_message=error_msg)

    def run(self):
        """Main worker loop with auto-shutdown"""
        print("🎬 OurVidz Worker started! Waiting for jobs...")
        
        # Download models if needed
        try:
            from download_models import check_and_download_models
            check_and_download_models()
        except Exception as e:
            print(f"⚠️ Model download check failed: {e}")
        
        while True:
            try:
                job = self._get_next_job()
                
                if job:
                    self.idle_start_time = None
                    self.process_job(job)
                else:
                    # Track idle time for auto-shutdown
                    if self.idle_start_time is None:
                        self.idle_start_time = time.time()
                        print("💤 No jobs found, starting idle timer...")
                    
                    idle_minutes = (time.time() - self.idle_start_time) / 60
                    
                    if idle_minutes >= self.max_idle_minutes:
                        print(f"🛑 No jobs for {self.max_idle_minutes} minutes, shutting down...")
                        break
                    
                    print(f"⏳ Idle for {idle_minutes:.1f} minutes (shutdown at {self.max_idle_minutes})")
                    time.sleep(30)
                    
            except KeyboardInterrupt:
                print("🛑 Shutdown requested")
                break
            except Exception as e:
                print(f"❌ Worker error: {e}")
                traceback.print_exc()
                time.sleep(30)
        
        print("🔚 Worker shutting down...")

if __name__ == "__main__":
    worker = OurVidzWorker()
    worker.run()
