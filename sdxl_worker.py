# sdxl_worker.py - BATCH GENERATION VERSION - FIXED CALLBACK PARAMETERS
# NEW: Supports 6-image batch generation for better user experience
# FIXED: Correct callback parameter names for job-callback compatibility
# Performance: 3.6s per image, ~22s for 6 images on RTX 6000 ADA

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
        """Initialize LUSTIFY SDXL Worker with batch generation support"""
        print("🎨 LUSTIFY SDXL WORKER - BATCH GENERATION VERSION - FIXED CALLBACKS")
        print("⚡ RTX 6000 ADA: 3-8s per image, supports 6-image batches")
        print("📋 Phase 1: sdxl_image_fast, sdxl_image_high")
        print("🚀 NEW: 6-image batch generation for improved UX")
        print("🔧 FIXED: Callback parameter consistency with job-callback function")
        
        # Model configuration
        self.model_path = "/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors"
        self.pipeline = None
        self.model_loaded = False
        
        # Job configurations with batch support
        self.job_configs = {
            'sdxl_image_fast': {
                'content_type': 'image',
                'file_extension': 'png',
                'height': 1024,
                'width': 1024,
                'num_inference_steps': 15,
                'guidance_scale': 6.0,
                'storage_bucket': 'sdxl_image_fast',
                'expected_time_per_image': 4,
                'quality_tier': 'fast',
                'phase': 1,
                'supports_batch': True
            },
            'sdxl_image_high': {
                'content_type': 'image', 
                'file_extension': 'png',
                'height': 1024,
                'width': 1024,
                'num_inference_steps': 25,
                'guidance_scale': 7.5,
                'storage_bucket': 'sdxl_image_high',
                'expected_time_per_image': 8,
                'quality_tier': 'high',
                'phase': 1,
                'supports_batch': True
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
        logger.info("🎨 LUSTIFY SDXL Worker with batch generation initialized")

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

    def generate_images_batch(self, prompt, job_type, num_images=6):
        """Generate multiple images in a single batch for efficiency"""
        if job_type not in self.job_configs:
            raise ValueError(f"Unknown job type: {job_type}")
            
        config = self.job_configs[job_type]
        
        # Ensure model is loaded
        self.load_model()
        
        logger.info(f"🎨 Generating {num_images} images for {job_type}: {prompt[:50]}...")
        start_time = time.time()
        
        try:
            # Clear GPU cache before generation
            torch.cuda.empty_cache()
            
            generation_kwargs = {
                'prompt': [prompt] * num_images,  # Replicate prompt for batch
                'height': config['height'],
                'width': config['width'],
                'num_inference_steps': config['num_inference_steps'],
                'guidance_scale': config['guidance_scale'],
                'num_images_per_prompt': 1,  # Generate 1 image per prompt in batch
                'generator': [torch.Generator(device="cuda").manual_seed(int(time.time()) + i) for i in range(num_images)]  # Different seeds for variety
            }
            
            # Add negative prompt for better quality
            generation_kwargs['negative_prompt'] = [
                "blurry, low quality, distorted, deformed, bad anatomy, "
                "watermark, signature, text, logo, extra limbs, missing limbs"
            ] * num_images
            
            # Generate batch of images
            with torch.inference_mode():
                result = self.pipeline(**generation_kwargs)
                images = result.images  # List of PIL Images
            
            generation_time = time.time() - start_time
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            
            logger.info(f"✅ Generated {len(images)} images in {generation_time:.1f}s, peak VRAM: {peak_vram:.1f}GB")
            logger.info(f"📊 Average time per image: {generation_time/len(images):.1f}s")
            
            # Clear cache after generation
            torch.cuda.empty_cache()
            gc.collect()
            
            return images
            
        except Exception as e:
            logger.error(f"❌ Batch generation failed: {e}")
            # Clear cache on error
            torch.cuda.empty_cache()
            gc.collect()
            raise

    def upload_images_batch(self, images, job_id, user_id, config):
        """Upload multiple images and return array of URLs"""
        upload_urls = []
        timestamp = int(time.time())
        
        logger.info(f"📁 Uploading {len(images)} images...")
        
        for i, image in enumerate(images):
            try:
                # Create unique filename for each image
                filename = f"sdxl_{job_id}_{timestamp}_{i+1}.{config['file_extension']}"
                temp_path = f"/tmp/{filename}"
                
                # Save locally with optimization
                image.save(temp_path, "PNG", quality=95, optimize=True)
                logger.info(f"💾 Image {i+1} saved locally: {temp_path}")
                
                # Upload to Supabase
                storage_path = f"{config['storage_bucket']}/{user_id}/{filename}"
                upload_path = self.upload_to_supabase(temp_path, storage_path)
                
                if upload_path:
                    upload_urls.append(upload_path)
                    logger.info(f"✅ Image {i+1} uploaded: {upload_path}")
                else:
                    logger.error(f"❌ Image {i+1} upload failed")
                    upload_urls.append(None)  # Placeholder for failed upload
                
                # Cleanup temp file
                Path(temp_path).unlink(missing_ok=True)
                
            except Exception as e:
                logger.error(f"❌ Image {i+1} processing failed: {e}")
                upload_urls.append(None)
        
        # Filter out failed uploads
        successful_uploads = [url for url in upload_urls if url is not None]
        logger.info(f"📊 Upload summary: {len(successful_uploads)}/{len(images)} images successful")
        
        return successful_uploads

    def upload_to_supabase(self, file_path, storage_path):
        """Upload image to Supabase storage with proper Content-Type"""
        try:
            # Verify file exists before upload
            if not Path(file_path).exists():
                logger.error(f"❌ File does not exist: {file_path}")
                return None
                
            # Get file size for verification
            file_size = Path(file_path).stat().st_size
            
            # Use proper binary upload with explicit Content-Type
            with open(file_path, 'rb') as file:
                file_data = file.read()
            
            headers = {
                'Authorization': f"Bearer {self.supabase_service_key}",
                'Content-Type': 'image/png',  # ✅ Explicit PNG content type
                'x-upsert': 'true'
            }
            
            response = requests.post(
                f"{self.supabase_url}/storage/v1/object/{storage_path}",
                data=file_data,  # ✅ Raw binary data
                headers=headers,
                timeout=60
            )
            
            if response.status_code in [200, 201]:
                # Return relative path within bucket
                path_parts = storage_path.split('/', 1)
                relative_path = path_parts[1] if len(path_parts) == 2 else storage_path
                return relative_path
            else:
                logger.error(f"❌ Upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Upload error: {e}")
            return None

    def process_job(self, job_data):
        """Process a single SDXL job with FIXED payload structure"""
        # FIXED: Use correct field names from edge function
        job_id = job_data['id']           # ✅ Edge function sends 'id'
        job_type = job_data['type']       # ✅ Edge function sends 'type'
        prompt = job_data['prompt']
        user_id = job_data['user_id']     # ✅ Edge function sends 'user_id'
        
        # Optional fields with defaults
        image_id = job_data.get('image_id', f"image_{int(time.time())}")
        config = job_data.get('config', {})
        
        # Extract num_images from config (default to 6 for batch generation)
        num_images = config.get('num_images', 6)
        
        logger.info(f"🚀 Processing SDXL job {job_id} ({job_type})")
        logger.info(f"📝 Prompt: {prompt}")
        logger.info(f"🖼️ Generating {num_images} images")
        logger.info(f"👤 User ID: {user_id}")
        
        # Phase validation
        if job_type not in self.phase_1_jobs:
            error_msg = f"Job type {job_type} not supported in Phase 1"
            logger.warning(f"⚠️ {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)
            return
        
        try:
            # Generate batch of images
            start_time = time.time()
            images = self.generate_images_batch(prompt, job_type, num_images)
            
            if not images:
                raise Exception("Image generation failed")
            
            # Upload all images
            job_config = self.job_configs[job_type]
            upload_urls = self.upload_images_batch(images, job_id, user_id, job_config)
            
            if not upload_urls:
                raise Exception("All image uploads failed")
            
            total_time = time.time() - start_time
            logger.info(f"✅ SDXL job {job_id} completed in {total_time:.1f}s")
            logger.info(f"📁 Generated {len(upload_urls)} images")
            
            # FIXED: Notify completion with correct parameter names
            self.notify_completion(job_id, 'completed', assets=upload_urls)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ SDXL job {job_id} failed: {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)
        finally:
            # Always cleanup GPU memory
            torch.cuda.empty_cache()
            gc.collect()

    def notify_completion(self, job_id, status, assets=None, error_message=None):
        """FIXED: Notify Supabase of job completion with correct callback format"""
        try:
            # FIXED: Use correct callback format that matches job-callback edge function expectations
            callback_data = {
                'job_id': job_id,        # ✅ Use job_id (snake_case) as expected by callback function
                'status': status,
                'assets': assets if assets else [],  # ✅ Use 'assets' array format
                'error_message': error_message
            }
            
            logger.info(f"📞 Sending FIXED callback for job {job_id}:")
            logger.info(f"   Status: {status}")
            logger.info(f"   Assets count: {len(assets) if assets else 0}")
            logger.info(f"   Error: {error_message}")
            logger.info(f"   Using job_id parameter: {job_id}")
            
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
                logger.info(f"✅ FIXED Callback sent successfully for job {job_id}")
                if assets:
                    logger.info(f"📊 Sent {len(assets)} asset URLs")
            else:
                logger.warning(f"⚠️ Callback failed: {response.status_code} - {response.text}")
                logger.error(f"❌ Callback payload was: {callback_data}")
                
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
        logger.info("⚡ Performance: 3-8s per image, ~22s for 6-image batch")
        logger.info("📬 Polling sdxl_queue for sdxl_image_fast, sdxl_image_high")
        logger.info("🖼️ NEW: 6-image batch generation support")
        logger.info("🔧 FIXED: Proper callback parameter consistency (job_id + assets)")
        
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
                        time.sleep(2)  # Fast polling for SDXL jobs
                        
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
    logger.info("🚀 Starting LUSTIFY SDXL Worker - BATCH GENERATION VERSION - FIXED CALLBACKS")
    
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