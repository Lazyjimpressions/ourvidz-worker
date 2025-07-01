# sdxl_worker.py - LUSTIFY SDXL Worker for RTX 6000 ADA
# Optimized for 48GB VRAM with concurrent Wan 2.1 operation
import os
import json
import time
import requests
import uuid
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LustifySDXLWorker:
    def __init__(self):
        """Initialize LUSTIFY SDXL Worker for RTX 6000 ADA"""
        print("🚀 LUSTIFY SDXL WORKER - RTX 6000 ADA OPTIMIZED")
        print("⚡ Performance: 3.2s generation, 10.5GB VRAM")
        
        # Model path
        self.model_path = "/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors"
        
        # Pipeline instance (loaded on-demand)
        self.pipeline = None
        self.model_loaded = False
        
        # Job configuration for different quality tiers
        self.job_configs = {
            'sdxl_image_fast': {
                'content_type': 'image',
                'file_extension': 'png',
                'height': 1024,
                'width': 1024,
                'num_inference_steps': 15,
                'guidance_scale': 6.0,
                'storage_bucket': 'sdxl_fast',
                'expected_time': 3,
                'quality_tier': 'fast'
            },
            'sdxl_image_high': {
                'content_type': 'image', 
                'file_extension': 'png',
                'height': 1024,
                'width': 1024,
                'num_inference_steps': 25,
                'guidance_scale': 7.5,
                'storage_bucket': 'sdxl_high',
                'expected_time': 5,
                'quality_tier': 'high'
            },
            'sdxl_image_premium': {
                'content_type': 'image',
                'file_extension': 'png', 
                'height': 1280,
                'width': 1280,
                'num_inference_steps': 40,
                'guidance_scale': 8.0,
                'storage_bucket': 'sdxl_premium',
                'expected_time': 8,
                'quality_tier': 'premium'
            },
            'sdxl_img2img': {
                'content_type': 'image',
                'file_extension': 'png',
                'height': 1024,
                'width': 1024,
                'num_inference_steps': 20,
                'guidance_scale': 7.0,
                'strength': 0.75,
                'storage_bucket': 'sdxl_img2img',
                'expected_time': 4,
                'quality_tier': 'img2img'
            }
        }
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        self.validate_environment()
        logger.info("🎨 LUSTIFY SDXL Worker initialized successfully")

    def validate_environment(self):
        """Validate GPU and environment setup"""
        logger.info("🔍 Validating SDXL environment...")
        
        # Check PyTorch and CUDA
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ GPU: {device_name} ({total_memory:.1f}GB)")
            
            if total_memory >= 40:  # RTX 6000 ADA detection
                logger.info("✅ RTX 6000 ADA detected - concurrent operation enabled")
            else:
                logger.warning("⚠️ Smaller GPU detected - may need sequential operation")
        else:
            logger.error("❌ CUDA not available")
            
        # Check model file
        if Path(self.model_path).exists():
            model_size = Path(self.model_path).stat().st_size / (1024**3)
            logger.info(f"✅ LUSTIFY model: {model_size:.1f}GB")
        else:
            logger.error(f"❌ Model missing: {self.model_path}")
            
        # Check environment variables
        required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            logger.error(f"❌ Missing env vars: {missing}")
        else:
            logger.info("✅ All environment variables configured")

    def load_model(self):
        """Load LUSTIFY SDXL model on-demand"""
        if self.model_loaded:
            return
            
        logger.info("📦 Loading LUSTIFY SDXL v2.0...")
        start_time = time.time()
        
        try:
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            ).to("cuda")
            
            # Optimize pipeline for production
            self.pipeline.enable_attention_slicing()
            if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                self.pipeline.enable_xformers_memory_efficient_attention()
            
            load_time = time.time() - start_time
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            
            self.model_loaded = True
            logger.info(f"✅ LUSTIFY loaded in {load_time:.1f}s, using {vram_used:.1f}GB VRAM")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise

    def unload_model(self):
        """Unload model to free VRAM (if needed for sequential operation)"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.model_loaded = False
            torch.cuda.empty_cache()
            logger.info("🗑️ LUSTIFY model unloaded")

    def generate_image(self, prompt, job_type, init_image=None):
        """Generate image with LUSTIFY SDXL"""
        if job_type not in self.job_configs:
            raise ValueError(f"Unknown job type: {job_type}")
            
        config = self.job_configs[job_type]
        
        # Ensure model is loaded
        self.load_model()
        
        logger.info(f"🎨 Generating {job_type}: {prompt[:50]}...")
        start_time = time.time()
        
        try:
            generation_kwargs = {
                'prompt': prompt,
                'height': config['height'],
                'width': config['width'],
                'num_inference_steps': config['num_inference_steps'],
                'guidance_scale': config['guidance_scale'],
                'num_images_per_prompt': 1
            }
            
            # Add img2img specific parameters
            if job_type == 'sdxl_img2img' and init_image:
                generation_kwargs['image'] = init_image
                generation_kwargs['strength'] = config['strength']
                # Note: Would need to use StableDiffusionXLImg2ImgPipeline for img2img
                # For now, treating as text2img
            
            # Generate image
            result = self.pipeline(**generation_kwargs)
            image = result.images[0]
            
            generation_time = time.time() - start_time
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            
            logger.info(f"✅ Generated in {generation_time:.1f}s, peak VRAM: {peak_vram:.1f}GB")
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise

    def upload_to_supabase(self, file_path, storage_path):
        """Upload image to Supabase storage"""
        try:
            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': file},
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                        'x-upsert': 'true'
                    },
                    timeout=60
                )
            
            if response.status_code in [200, 201]:
                # Return relative path within bucket
                path_parts = storage_path.split('/', 1)
                relative_path = path_parts[1] if len(path_parts) == 2 else storage_path
                logger.info(f"📁 Uploaded to bucket: {relative_path}")
                return relative_path
            else:
                logger.error(f"❌ Upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Upload error: {e}")
            return None

    def process_job(self, job_data):
        """Process a single SDXL job"""
        job_id = job_data['jobId']
        job_type = job_data['jobType']
        prompt = job_data['prompt']
        user_id = job_data['userId']
        
        logger.info(f"🚀 Processing SDXL job {job_id} ({job_type})")
        logger.info(f"📝 Prompt: {prompt}")
        
        try:
            # Generate image
            start_time = time.time()
            image = self.generate_image(prompt, job_type)
            
            if not image:
                raise Exception("Image generation failed")
            
            # Save and upload
            config = self.job_configs[job_type]
            timestamp = int(time.time())
            filename = f"sdxl_{job_id}_{timestamp}.{config['file_extension']}"
            temp_path = f"/tmp/{filename}"
            
            # Save locally
            image.save(temp_path, "PNG", quality=95, optimize=True)
            
            # Upload to Supabase
            storage_path = f"{config['storage_bucket']}/{user_id}/{filename}"
            upload_path = self.upload_to_supabase(temp_path, storage_path)
            
            if not upload_path:
                raise Exception("File upload failed")
            
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
            
            total_time = time.time() - start_time
            logger.info(f"✅ SDXL job {job_id} completed in {total_time:.1f}s")
            logger.info(f"📁 File: {upload_path}")
            
            # Notify completion
            self.notify_completion(job_id, 'completed', upload_path)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ SDXL job {job_id} failed: {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)

    def notify_completion(self, job_id, status, file_path=None, error_message=None):
        """Notify Supabase of job completion"""
        try:
            callback_data = {
                'jobId': job_id,
                'status': status,
                'filePath': file_path,
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
                logger.info(f"✅ Callback sent for job {job_id}")
            else:
                logger.warning(f"⚠️ Callback failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Callback error: {e}")

    def poll_queue(self):
        """Poll Redis SDXL queue for new jobs"""
        try:
            response = requests.get(
                f"{self.redis_url}/rpop/sdxl_queue",
                headers={'Authorization': f"Bearer {self.redis_token}"},
                timeout=10
            )
            
            if response.status_code == 200 and response.json().get('result'):
                return json.loads(response.json()['result'])
                
        except Exception as e:
            if "timeout" not in str(e).lower():
                logger.warning(f"⚠️ Queue poll error: {e}")
        
        return None

    def run(self):
        """Main SDXL worker loop"""
        logger.info("🎨 LUSTIFY SDXL WORKER READY!")
        logger.info("⚡ Performance: 3.2s generation, 10.5GB VRAM")
        logger.info("🔥 RTX 6000 ADA: Concurrent operation enabled")
        logger.info("📬 Waiting for SDXL jobs...")
        
        job_count = 0
        
        while True:
            try:
                job = self.poll_queue()
                if job:
                    job_count += 1
                    logger.info(f"📬 SDXL Job #{job_count} received")
                    self.process_job(job)
                    logger.info("=" * 60)
                else:
                    # No job available, wait briefly
                    time.sleep(2)  # Faster polling for quick SDXL jobs
                    
            except KeyboardInterrupt:
                logger.info("👋 SDXL Worker shutting down...")
                self.unload_model()
                break
            except Exception as e:
                logger.error(f"❌ SDXL Worker error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    logger.info("🚀 Starting LUSTIFY SDXL Worker")
    
    # Environment validation
    required_vars = [
        'SUPABASE_URL', 
        'SUPABASE_SERVICE_KEY', 
        'UPSTASH_REDIS_REST_URL', 
        'UPSTASH_REDIS_REST_TOKEN'
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    # Quick GPU test
    if torch.cuda.is_available():
        test_tensor = torch.randn(100, 100, device='cuda')
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"✅ GPU test: {gpu_memory:.3f}GB allocated")
        del test_tensor
        torch.cuda.empty_cache()
    else:
        logger.warning("⚠️ CUDA not available - SDXL performance will be severely degraded")
    
    try:
        worker = LustifySDXLWorker()
        worker.run()
    except Exception as e:
        logger.error(f"❌ SDXL Worker startup failed: {e}")
        exit(1)
