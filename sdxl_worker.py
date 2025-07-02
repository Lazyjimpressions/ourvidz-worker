# sdxl_worker.py - CLEANED UP VERSION
# Preserves Phase 2 job types but only validates Phase 1
# Performance: 3.6s generation, 10.5GB VRAM peak on RTX 6000 ADA

import os
import json
import time
import requests
import uuid
import torch
import gc
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
        print("🎨 LUSTIFY SDXL WORKER - PRODUCTION VERSION")
        print("⚡ RTX 6000 ADA: 3-8s generation, 10.5GB VRAM peak")
        print("📋 Phase 1: sdxl_image_fast, sdxl_image_high")
        print("🚀 Phase 2: sdxl_image_premium, sdxl_img2img (preserved)")
        
        # Model configuration
        self.model_path = "/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors"
        self.pipeline = None
        self.model_loaded = False
        
        # Job configurations optimized for quality/speed
        self.job_configs = {
            # PHASE 1 - ACTIVE VALIDATION
            'sdxl_image_fast': {
                'content_type': 'image',
                'file_extension': 'png',
                'height': 1024,
                'width': 1024,
                'num_inference_steps': 15,
                'guidance_scale': 6.0,
                'storage_bucket': 'sdxl_fast',
                'expected_time': 5,
                'quality_tier': 'fast',
                'phase': 1
            },
            'sdxl_image_high': {
                'content_type': 'image', 
                'file_extension': 'png',
                'height': 1024,
                'width': 1024,
                'num_inference_steps': 25,
                'guidance_scale': 7.5,
                'storage_bucket': 'sdxl_high',
                'expected_time': 8,
                'quality_tier': 'high',
                'phase': 1
            },
            
            # PHASE 2 - PRESERVED BUT NOT VALIDATED
            'sdxl_image_premium': {
                'content_type': 'image',
                'file_extension': 'png', 
                'height': 1280,
                'width': 1280,
                'num_inference_steps': 40,
                'guidance_scale': 8.5,
                'storage_bucket': 'sdxl_premium',
                'expected_time': 12,
                'quality_tier': 'premium',
                'phase': 2
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
                'expected_time': 6,
                'quality_tier': 'img2img',
                'phase': 2
            }
        }
        
        # Phase 1 job types for validation
        self.phase_1_jobs = ['sdxl_image_fast', 'sdxl_image_high']
        
        # Environment setup
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        self.validate_environment()
        logger.info("🎨 LUSTIFY SDXL Worker initialized successfully")

    def validate_environment(self):
        """Comprehensive environment validation"""
        logger.info("🔍 Validating SDXL environment...")
        
        # Check PyTorch and CUDA
        logger.info(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ GPU: {device_name} ({total_memory:.1f}GB)")
            
            # Test GPU allocation
            test_tensor = torch.randn(100, 100, device='cuda')
            allocated = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"✅ GPU test: {allocated:.3f}GB allocated")
            del test_tensor
            torch.cuda.empty_cache()
        else:
            logger.error("❌ CUDA not available")
            
        # Check diffusers
        try:
            import diffusers
            logger.info(f"✅ Diffusers: {diffusers.__version__}")
        except ImportError:
            logger.error("❌ Diffusers not available")
            
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
        """Load LUSTIFY SDXL model with optimizations"""
        if self.model_loaded:
            return
            
        logger.info("📦 Loading LUSTIFY SDXL v2.0...")
        start_time = time.time()
        
        try:
            # Load pipeline from single file
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to("cuda")
            
            # Enable memory optimizations
            try:
                self.pipeline.enable_attention_slicing()
                logger.info("✅ Attention slicing enabled")
            except Exception as e:
                logger.warning(f"⚠️ Attention slicing failed: {e}")
            
            # Try xformers if available
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("✅ xformers optimization enabled")
            except Exception as e:
                logger.warning(f"⚠️ xformers optimization failed: {e}")
            
            load_time = time.time() - start_time
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            
            self.model_loaded = True
            logger.info(f"✅ LUSTIFY loaded in {load_time:.1f}s, using {vram_used:.1f}GB VRAM")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise

    def generate_image(self, prompt, job_type):
        """Generate image with LUSTIFY SDXL"""
        if job_type not in self.job_configs:
            raise ValueError(f"Unknown job type: {job_type}")
            
        config = self.job_configs[job_type]
        
        # Ensure model is loaded
        self.load_model()
        
        logger.info(f"🎨 Generating {job_type}: {prompt[:50]}...")
        start_time = time.time()
        
        try:
            # Clear GPU cache before generation
            torch.cuda.empty_cache()
            
            generation_kwargs = {
                'prompt': prompt,
                'height': config['height'],
                'width': config['width'],
                'num_inference_steps': config['num_inference_steps'],
                'guidance_scale': config['guidance_scale'],
                'num_images_per_prompt': 1,
                'generator': torch.Generator(device="cuda").manual_seed(int(time.time()) % 2**32)
            }
            
            # Add negative prompt for better quality
            generation_kwargs['negative_prompt'] = (
                "blurry, low quality, distorted, deformed, bad anatomy, "
                "watermark, signature, text, logo, extra limbs, missing limbs"
            )
            
            # Generate image
            with torch.inference_mode():
                result = self.pipeline(**generation_kwargs)
                image = result.images[0]
            
            generation_time = time.time() - start_time
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            
            logger.info(f"✅ Generated in {generation_time:.1f}s, peak VRAM: {peak_vram:.1f}GB")
            
            # Clear cache after generation
            torch.cuda.empty_cache()
            gc.collect()
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            # Clear cache on error
            torch.cuda.empty_cache()
            gc.collect()
            raise

    def upload_to_supabase(self, file_path, storage_path):
        """Upload image to Supabase storage with verification"""
        try:
            # Verify file exists before upload
            if not Path(file_path).exists():
                print(f"❌ File does not exist: {file_path}")
                return None
                
            # Get file size for verification
            file_size = Path(file_path).stat().st_size
            print(f"📁 Uploading file: {file_size} bytes")
            
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
                logger.info(f"✅ Upload verified: {file_size} bytes")
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
        
        # Phase validation
        if job_type not in self.phase_1_jobs:
            error_msg = f"Job type {job_type} not supported in Phase 1"
            logger.warning(f"⚠️ {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)
            return
        
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
            
            # Save locally with optimization
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
        finally:
            # Always cleanup GPU memory
            torch.cuda.empty_cache()
            gc.collect()

    def notify_completion(self, job_id, status, file_path=None, error_message=None):
        """Notify Supabase of job completion"""
        try:
            callback_data = {
                'jobId': job_id,
                'status': status,
                'filePath': file_path,  # ✅ Correct parameter name
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
        logger.info("⚡ Performance: 3-8s generation, RTX 6000 ADA optimized")
        logger.info("📬 Polling sdxl_queue for sdxl_image_fast, sdxl_image_high")
        logger.info("🔒 Phase 1 validation: Only Phase 1 jobs accepted")
        
        job_count = 0
        
        try:
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
                        time.sleep(2)  # Fast polling for quick SDXL jobs
                        
                except Exception as e:
                    logger.error(f"❌ Job processing error: {e}")
                    time.sleep(10)
                    
        except KeyboardInterrupt:
            logger.info("👋 SDXL Worker shutting down...")
        finally:
            # Cleanup on shutdown
            if self.pipeline:
                del self.pipeline
                torch.cuda.empty_cache()
                gc.collect()
            logger.info("✅ SDXL Worker cleanup complete")

if __name__ == "__main__":
    logger.info("🚀 Starting LUSTIFY SDXL Worker - PHASE 1 VALIDATED")
    
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
    
    try:
        worker = LustifySDXLWorker()
        worker.run()
    except Exception as e:
        logger.error(f"❌ SDXL Worker startup failed: {e}")
        exit(1)
