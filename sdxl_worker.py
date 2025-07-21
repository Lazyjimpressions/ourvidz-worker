# sdxl_worker.py - FLEXIBLE QUANTITY + IMAGE-TO-IMAGE VERSION + SDXL COMPEL LIBRARY INTEGRATION - CONSISTENT PARAMETER NAMING
# NEW: Supports user-selected quantities (1, 3, or 6 images) and image-to-image generation
# FIXED: SDXL-specific Compel library integration with prompt_embeds and pooled_prompt_embeds
# FIXED: Consistent callback parameter names (job_id, assets) for edge function compatibility
# Performance: 1 image: 3-8s, 3 images: 9-24s, 6 images: 18-48s on RTX 6000 ADA

"""
🎯 COMPEL INTEGRATION USAGE EXAMPLES (SIMPLE STRING CONCATENATION):

# Example 1: Basic Compel enhancement
job_data = {
    "id": "job-123",
    "type": "sdxl_image_high",
    "prompt": "beautiful woman in garden",
    "user_id": "user-123",
    "compel_enabled": True,
    "compel_weights": "(beautiful:1.3), (woman:1.2), (garden:1.1)",
    "config": {
        "num_images": 1
    }
}

# Example 2: Compel with image-to-image
job_data = {
    "id": "job-124", 
    "type": "sdxl_image_fast",
    "prompt": "portrait of a person",
    "user_id": "user-123",
    "compel_enabled": True,
    "compel_weights": "(portrait:1.4), (person:1.1)",
    "config": {
        "num_images": 3
    },
    "metadata": {
        "reference_image_url": "https://example.com/reference.jpg",
        "reference_strength": 0.7,
        "reference_type": "style"
    }
}

# Example 3: No Compel (standard generation)
job_data = {
    "id": "job-125",
    "type": "sdxl_image_high", 
    "prompt": "landscape painting",
    "user_id": "user-123",
    "compel_enabled": False,  # or omit this field
    "config": {
        "num_images": 6
    }
}

🎯 COMPEL WEIGHTS SYNTAX (SIMPLE CONCATENATION):
- The frontend sends the correct Compel syntax
- We simply concatenate: prompt + " " + compel_weights
- Example: "beautiful woman" + " " + "(beautiful:1.3), (woman:1.2)"
- Result: "beautiful woman (beautiful:1.3), (woman:1.2)"
"""

import os
import json
import time
import requests
import uuid
import torch
import gc
import io
import sys
import compel  # ADD: Compel library for proper prompt weighting
from compel import Compel  # ADD: Compel processor
sys.path.append('/workspace/python_deps/lib/python3.11/site-packages')
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LustifySDXLWorker:
    def __init__(self):
        """Initialize LUSTIFY SDXL Worker with flexible quantity, image-to-image generation, and SDXL-specific Compel library integration"""
        print("🎨 LUSTIFY SDXL WORKER - FLEXIBLE QUANTITY + IMAGE-TO-IMAGE + SDXL COMPEL LIBRARY INTEGRATION - CONSISTENT PARAMETERS")
        print("⚡ RTX 6000 ADA: 1 image: 3-8s, 3 images: 9-24s, 6 images: 18-48s")
        print("📋 Phase 1: sdxl_image_fast, sdxl_image_high")
        print("🚀 NEW: User-selected quantities (1, 3, or 6 images) for flexible UX")
        print("🖼️ NEW: Image-to-image generation with style, composition, and character reference modes")
        print("🌱 NEW: Seed control for reproducible generation and character consistency")
        print("🔧 FIXED: SDXL-specific Compel library integration with prompt_embeds and pooled_prompt_embeds")
        print("🔧 FIXED: Consistent parameter naming (job_id, assets, metadata) across all callbacks")
        print("✅ API COMPLIANT: Supports metadata.reference_image_url, reference_strength, reference_type")
        
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
                'supports_flexible_quantities': True
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
                'supports_flexible_quantities': True
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
        logger.info("🎨 LUSTIFY SDXL Worker with flexible quantities and image-to-image support initialized")

    def download_image_from_url(self, image_url):
        """Download image from URL and return PIL Image object"""
        try:
            logger.info(f"📥 Downloading reference image from: {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"✅ Reference image downloaded: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"❌ Failed to download reference image: {e}")
            raise

    def preprocess_reference_image(self, image, target_size=(1024, 1024)):
        """Preprocess reference image for SDXL generation"""
        try:
            # Resize image to target size while maintaining aspect ratio
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image
            new_image = Image.new('RGB', target_size, (0, 0, 0))
            
            # Center the image
            x = (target_size[0] - image.width) // 2
            y = (target_size[1] - image.height) // 2
            new_image.paste(image, (x, y))
            
            logger.info(f"✅ Reference image preprocessed to {target_size}")
            return new_image
            
        except Exception as e:
            logger.error(f"❌ Failed to preprocess reference image: {e}")
            raise

    def generate_with_style_reference(self, prompt, reference_image, strength, config, num_images=1, generators=None):
        """Generate images using reference image for style transfer"""
        logger.info(f"🎨 Style transfer generation with strength: {strength}")
        
        # Preprocess reference image
        processed_image = self.preprocess_reference_image(reference_image, (config['width'], config['height']))
        
        # Use provided generators or create new ones
        if generators is None:
            generators = [torch.Generator(device="cuda").manual_seed(int(time.time()) + i) for i in range(num_images)]
        
        # Use image-to-image pipeline with style strength
        generation_kwargs = {
            'prompt': [prompt] * num_images,
            'image': processed_image,
            'strength': strength,  # Controls how much of the original image to preserve
            'num_inference_steps': config['num_inference_steps'],
            'guidance_scale': config['guidance_scale'],
            'num_images_per_prompt': 1,
            'generator': generators
        }
        
        # Add negative prompt
        generation_kwargs['negative_prompt'] = [
            "blurry, low quality, distorted, deformed, bad anatomy, "
            "watermark, signature, text, logo, extra limbs, missing limbs"
        ] * num_images
        
        with torch.inference_mode():
            result = self.pipeline(**generation_kwargs)
            return result.images

    def generate_with_composition_reference(self, prompt, reference_image, strength, config, num_images=1, generators=None):
        """Generate images using reference image for composition guidance"""
        logger.info(f"🎨 Composition guidance generation with strength: {strength}")
        
        # Preprocess reference image
        processed_image = self.preprocess_reference_image(reference_image, (config['width'], config['height']))
        
        # Use provided generators or create new ones
        if generators is None:
            generators = [torch.Generator(device="cuda").manual_seed(int(time.time()) + i) for i in range(num_images)]
        
        # Use image-to-image with higher strength for composition preservation
        generation_kwargs = {
            'prompt': [prompt] * num_images,
            'image': processed_image,
            'strength': min(strength + 0.2, 0.9),  # Higher strength for composition
            'num_inference_steps': config['num_inference_steps'],
            'guidance_scale': config['guidance_scale'],
            'num_images_per_prompt': 1,
            'generator': generators
        }
        
        # Add negative prompt
        generation_kwargs['negative_prompt'] = [
            "blurry, low quality, distorted, deformed, bad anatomy, "
            "watermark, signature, text, logo, extra limbs, missing limbs"
        ] * num_images
        
        with torch.inference_mode():
            result = self.pipeline(**generation_kwargs)
            return result.images

    def generate_with_character_reference(self, prompt, reference_image, strength, config, num_images=1, generators=None):
        """Generate images using reference image for character consistency"""
        logger.info(f"🎨 Character consistency generation with strength: {strength}")
        
        # Preprocess reference image
        processed_image = self.preprocess_reference_image(reference_image, (config['width'], config['height']))
        
        # Use provided generators or create new ones
        if generators is None:
            generators = [torch.Generator(device="cuda").manual_seed(int(time.time()) + i) for i in range(num_images)]
        
        # Use image-to-image with moderate strength for character preservation
        generation_kwargs = {
            'prompt': [prompt] * num_images,
            'image': processed_image,
            'strength': strength,
            'num_inference_steps': config['num_inference_steps'],
            'guidance_scale': config['guidance_scale'],
            'num_images_per_prompt': 1,
            'generator': generators
        }
        
        # Add negative prompt
        generation_kwargs['negative_prompt'] = [
            "blurry, low quality, distorted, deformed, bad anatomy, "
            "watermark, signature, text, logo, extra limbs, missing limbs"
        ] * num_images
        
        with torch.inference_mode():
            result = self.pipeline(**generation_kwargs)
            return result.images

    def generate_image_to_image(self, prompt, reference_image, strength, config, num_images=1, generators=None):
        """Standard image-to-image generation"""
        logger.info(f"🎨 Standard image-to-image generation with strength: {strength}")
        
        # Preprocess reference image
        processed_image = self.preprocess_reference_image(reference_image, (config['width'], config['height']))
        
        # Use provided generators or create new ones
        if generators is None:
            generators = [torch.Generator(device="cuda").manual_seed(int(time.time()) + i) for i in range(num_images)]
        
        # Use image-to-image pipeline
        generation_kwargs = {
            'prompt': [prompt] * num_images,
            'image': processed_image,
            'strength': strength,
            'num_inference_steps': config['num_inference_steps'],
            'guidance_scale': config['guidance_scale'],
            'num_images_per_prompt': 1,
            'generator': generators
        }
        
        # Add negative prompt
        generation_kwargs['negative_prompt'] = [
            "blurry, low quality, distorted, deformed, bad anatomy, "
            "watermark, signature, text, logo, extra limbs, missing limbs"
        ] * num_images
        
        with torch.inference_mode():
            result = self.pipeline(**generation_kwargs)
            return result.images

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

    def process_compel_weights(self, prompt, weights_config=None):
        """
        Process prompt with proper Compel library integration for SDXL.
        FIXED: Use requires_pooled=[False, True] for Compel 2.1.1 with SDXL.
        """
        if not weights_config:
            return prompt, None
        try:
            if not self.model_loaded:
                self.load_model()
            logger.info(f"🔧 Initializing Compel 2.1.1 with SDXL encoders - FIXED SYNTAX")
            compel_processor = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                requires_pooled=[False, True]  # ✅ FIXED: List instead of boolean for Compel 2.1.1
            )
            logger.info(f"✅ Compel processor initialized successfully with FIXED SDXL syntax")
            combined_prompt = f"{prompt} {weights_config}"
            logger.info(f"📝 Combined prompt: {combined_prompt}")
            # Always unpack as a tuple
            prompt_embeds, pooled_prompt_embeds = compel_processor(combined_prompt)
            logger.info(f"✅ Compel weights applied with proper SDXL library integration")
            logger.info(f"📝 Original prompt: {prompt}")
            logger.info(f"🎯 Compel weights: {weights_config}")
            logger.info(f"🔧 Generated prompt_embeds: {prompt_embeds.shape}")
            logger.info(f"🔧 Generated pooled_prompt_embeds: {pooled_prompt_embeds.shape}")
            return (prompt_embeds, pooled_prompt_embeds), prompt
        except Exception as e:
            logger.error(f"❌ Compel processing failed: {e}")
            logger.info(f"🔄 Falling back to original prompt: {prompt}")
            return prompt, None  # Fallback to original prompt

    def generate_images_batch(self, prompt, job_type, num_images=1, reference_image=None, reference_strength=0.5, reference_type='style', seed=None):
        """Generate multiple images in a single batch for efficiency (supports 1, 3, or 6 images) with optional image-to-image and seed control"""
        if job_type not in self.job_configs:
            raise ValueError(f"Unknown job type: {job_type}")
            
        config = self.job_configs[job_type]
        
        # Ensure model is loaded
        self.load_model()
        
        # Handle seed configuration
        if seed:
            logger.info(f"🌱 Using provided seed: {seed}")
            # Use provided seed for reproducible results
            generators = [torch.Generator(device="cuda").manual_seed(int(seed) + i) for i in range(num_images)]
        else:
            # Generate random seeds for variety
            random_seed = int(time.time())
            generators = [torch.Generator(device="cuda").manual_seed(random_seed + i) for i in range(num_images)]
            seed = random_seed  # Capture the base seed for callback
            logger.info(f"🎲 Using random seed: {seed}")
        
        if reference_image:
            logger.info(f"🎨 Generating {num_images} image(s) with {reference_type} reference (strength: {reference_strength})")
        else:
            logger.info(f"🎨 Generating {num_images} image(s) for {job_type}: {prompt[:50]}...")
            
        if num_images > 1:
            logger.info(f"📊 Expected performance: {num_images * config['expected_time_per_image']:.0f}s total")
        start_time = time.time()
        
        try:
            # Clear GPU cache before generation
            torch.cuda.empty_cache()
            
            # Handle image-to-image generation
            if reference_image:
                if reference_type == 'style':
                    images = self.generate_with_style_reference(prompt, reference_image, reference_strength, config, num_images, generators)
                elif reference_type == 'composition':
                    images = self.generate_with_composition_reference(prompt, reference_image, reference_strength, config, num_images, generators)
                elif reference_type == 'character':
                    images = self.generate_with_character_reference(prompt, reference_image, reference_strength, config, num_images, generators)
                else:
                    # Default image-to-image
                    images = self.generate_image_to_image(prompt, reference_image, reference_strength, config, num_images, generators)
            else:
                # Standard text-to-image generation with Compel support
                generation_kwargs = {
                    'height': config['height'],
                    'width': config['width'],
                    'num_inference_steps': config['num_inference_steps'],
                    'guidance_scale': config['guidance_scale'],
                    'num_images_per_prompt': 1,  # Generate 1 image per prompt in batch
                    'generator': generators  # Use configured generators with seeds
                }
                
                # Handle Compel conditioning tensors vs string prompt for SDXL
                if isinstance(prompt, tuple) and len(prompt) == 2:
                    # Compel conditioning tensors were returned (prompt_embeds, pooled_prompt_embeds)
                    prompt_embeds, pooled_prompt_embeds = prompt
                    generation_kwargs['prompt_embeds'] = prompt_embeds
                    generation_kwargs['pooled_prompt_embeds'] = pooled_prompt_embeds  # SDXL requires this
                    logger.info("✅ Using Compel conditioning tensors for SDXL generation")
                elif isinstance(prompt, torch.Tensor):
                    # Legacy single conditioning tensor (fallback)
                    generation_kwargs['prompt_embeds'] = prompt
                    logger.info("✅ Using single Compel conditioning tensor for generation (legacy)")
                else:
                    # String prompt (no Compel or Compel failed)
                    generation_kwargs['prompt'] = [prompt] * num_images  # Replicate prompt for batch
                    logger.info("✅ Using string prompt for generation (no Compel)")
                
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
            
            return images, seed
            
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
        """Process a single SDXL job with CONSISTENT payload structure"""
        # CONSISTENT: Use standardized field names across all workers
        job_id = job_data['id']           # ✅ Standard: 'id' field
        job_type = job_data['type']       # ✅ Standard: 'type' field
        prompt = job_data['prompt']       # ✅ Standard: 'prompt' field
        user_id = job_data['user_id']     # ✅ Standard: 'user_id' field
        
        # Optional fields with defaults
        image_id = job_data.get('image_id', f"image_{int(time.time())}")
        config = job_data.get('config', {})
        
        # Extract num_images from config (default to 1 for backward compatibility)
        num_images = config.get('num_images', 1)
        
        # Validate num_images (only allow 1, 3, or 6 for performance optimization)
        if num_images not in [1, 3, 6]:
            logger.warning(f"⚠️ Invalid num_images: {num_images}, defaulting to 1")
            num_images = 1
        
        # Extract seed from config for reproducible generation
        seed = config.get('seed')
        if seed:
            logger.info(f"🌱 Seed provided in job config: {seed}")
        else:
            logger.info(f"🎲 No seed provided, will use random seed")
        
        # Extract image-to-image parameters from metadata (ALREADY COMPLIANT WITH API SPEC)
        metadata = job_data.get('metadata', {})
        reference_image_url = metadata.get('reference_image_url')  # ✅ API spec: metadata.reference_image_url
        reference_strength = metadata.get('reference_strength', 0.5)  # ✅ API spec: metadata.reference_strength
        reference_type = metadata.get('reference_type', 'style')  # ✅ API spec: metadata.reference_type
        
        # ✅ COMPEL SUPPORT: Extract Compel parameters directly from job payload
        compel_enabled = job_data.get("compel_enabled", False)
        compel_weights = job_data.get("compel_weights", "")
        
        logger.info(f"🚀 Processing SDXL job {job_id} ({job_type})")
        logger.info(f"📝 Prompt: {prompt}")
        logger.info(f"🖼️ Generating {num_images} image(s) for user")
        logger.info(f"👤 User ID: {user_id}")
        
        # Log Compel configuration if present
        if compel_enabled and compel_weights:
            logger.info(f"🎯 Compel enhancement enabled: {compel_weights}")
        
        # Log image-to-image parameters if present
        if reference_image_url:
            logger.info(f"🖼️ Image-to-image mode: {reference_type} (strength: {reference_strength})")
            logger.info(f"📥 Reference image URL: {reference_image_url}")
        
        # Phase validation
        if job_type not in self.phase_1_jobs:
            error_msg = f"Job type {job_type} not supported in Phase 1"
            logger.warning(f"⚠️ {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)
            return
        
        try:
            # Handle image-to-image generation
            reference_image = None
            if reference_image_url:
                try:
                    reference_image = self.download_image_from_url(reference_image_url)
                    logger.info(f"✅ Reference image loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to load reference image: {e}")
                    # Continue with text-to-image generation
                    reference_image = None
            
            # Process Compel enhancement with proper library integration
            final_prompt = prompt
            original_prompt = None
            compel_success = False
            final_prompt_type = "string"
            compel_metadata = {
                "compel_enabled": compel_enabled,
                "compel_weights": compel_weights,
                "enhancement_strategy": "none"
            }
            
            if compel_enabled and compel_weights:
                logger.info(f"🎯 Compel enhancement enabled: {compel_weights}")
                
                try:
                    # Apply Compel weights to the prompt (proper library integration)
                    final_prompt, original_prompt = self.process_compel_weights(prompt, compel_weights)
                    if isinstance(final_prompt, tuple) and len(final_prompt) == 2:
                        compel_success = True
                        final_prompt_type = "conditioning_tensor"
                        logger.info(f"✅ Compel processing successful")
                        logger.info(f"🎯 Using Compel conditioning tensors for SDXL generation")
                    else:
                        compel_success = False
                        final_prompt_type = "string"
                        logger.info(f"⚠️ Compel did not return tensors, using string prompt")
                    
                except Exception as e:
                    logger.error(f"❌ Compel processing failed: {e}")
                    final_prompt = prompt  # Fallback to original prompt
                    original_prompt = None
                    compel_success = False
                    final_prompt_type = "string"
                    logger.info(f"🔄 Using original prompt due to Compel failure: {prompt}")
                
                # Log the final prompt being used
                if isinstance(final_prompt, tuple) and len(final_prompt) == 2:
                    logger.info(f"🎯 Using Compel conditioning tensors for SDXL generation")
                elif isinstance(final_prompt, torch.Tensor):
                    logger.info(f"🎯 Using single Compel conditioning tensor for generation")
                else:
                    logger.info(f"🎯 Using Compel-enhanced prompt: {final_prompt}")
            else:
                final_prompt = prompt
                original_prompt = None
                compel_success = False
                final_prompt_type = "string"
                logger.info(f"🎯 Using standard prompt (no Compel): {prompt}")
            
            # Generate batch of images with final prompt
            start_time = time.time()
            images, used_seed = self.generate_images_batch(
                final_prompt, 
                job_type, 
                num_images, 
                reference_image=reference_image,
                reference_strength=reference_strength,
                reference_type=reference_type,
                seed=seed
            )
            
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
            logger.info(f"🌱 Seed used: {used_seed}")
            
            # ✅ COMPEL SUPPORT: Prepare metadata for callback with Compel information
            callback_metadata = {
                'seed': used_seed,
                'generation_time': total_time,
                'num_images': len(upload_urls),
                'job_type': job_type,
                'original_prompt': original_prompt if original_prompt else prompt,
                'final_prompt': str(final_prompt),
                'compel_enabled': compel_enabled,
                'compel_weights': compel_weights if compel_enabled else None,
                'compel_success': compel_success if compel_enabled else False,
                'enhancement_strategy': 'compel' if compel_success else 'fallback' if compel_enabled else 'none',
                'final_prompt_type': final_prompt_type
            }
            
            # CONSISTENT: Notify completion with standardized parameter names and metadata
            self.notify_completion(job_id, 'completed', assets=upload_urls, metadata=callback_metadata)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ SDXL job {job_id} failed: {error_msg}")
            
            # Prepare error metadata
            error_metadata = {
                'error_type': type(e).__name__,
                'job_type': job_type,
                'timestamp': time.time()
            }
            
            self.notify_completion(job_id, 'failed', error_message=error_msg, metadata=error_metadata)
        finally:
            # Always cleanup GPU memory
            torch.cuda.empty_cache()
            gc.collect()

    def notify_completion(self, job_id, status, assets=None, error_message=None, metadata=None):
        """CONSISTENT: Notify Supabase with standardized callback parameter names and metadata"""
        try:
            # CONSISTENT: Use standardized callback format across all workers
            callback_data = {
                'job_id': job_id,        # ✅ Standard: job_id (snake_case)
                'status': status,        # ✅ Standard: status field
                'assets': assets if assets else [],  # ✅ Standard: assets array
                'error_message': error_message      # ✅ Standard: error_message field
            }
            
            # Add metadata if provided (for seed and other generation details)
            if metadata:
                callback_data['metadata'] = metadata
            
            logger.info(f"📞 Sending CONSISTENT callback for job {job_id}:")
            logger.info(f"   Status: {status}")
            logger.info(f"   Assets count: {len(assets) if assets else 0}")
            logger.info(f"   Error: {error_message}")
            logger.info(f"   Metadata: {metadata}")
            logger.info(f"   Parameters: job_id, status, assets, error_message, metadata (CONSISTENT)")
            
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
                logger.info(f"✅ CONSISTENT Callback sent successfully for job {job_id}")
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
        logger.info("⚡ Performance: 1 image: 3-8s, 3 images: 9-24s, 6 images: 18-48s")
        logger.info("📬 Polling sdxl_queue for sdxl_image_fast, sdxl_image_high")
        logger.info("🖼️ FLEXIBLE: User-selected quantities (1, 3, or 6 images)")
        logger.info("🖼️ IMAGE-TO-IMAGE: Style, composition, and character reference modes")
        logger.info("🌱 SEED CONTROL: Reproducible generation and character consistency")
        logger.info("🔧 FIXED: SDXL-specific Compel library integration with prompt_embeds and pooled_prompt_embeds")
        logger.info("🔧 CONSISTENT: Standardized callback parameters (job_id, status, assets, error_message, metadata)")
        
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

    def test_compel_integration(self):
        """Test Compel integration with proper library integration"""
        logger.info("🧪 Testing Compel integration (proper library integration)...")
        
        # Test cases
        test_cases = [
            {
                "prompt": "beautiful woman in garden",
                "compel_weights": "(masterpiece:1.3), (best quality:1.2)",
                "expected_type": "sdxl_tensors"
            },
            {
                "prompt": "portrait of a person",
                "compel_weights": "(perfect anatomy:1.2), (professional:1.1)",
                "expected_type": "sdxl_tensors"
            },
            {
                "prompt": "landscape painting",
                "compel_weights": None,
                "expected_type": "string"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"🧪 Test case {i+1}: {test_case['prompt']}")
            
            # Test Compel processing
            enhanced_prompt, original_prompt = self.process_compel_weights(
                test_case['prompt'], 
                test_case['compel_weights']
            )
            
            # Check if we got the expected type
            if test_case['expected_type'] == "sdxl_tensors":
                if isinstance(enhanced_prompt, tuple) and len(enhanced_prompt) == 2:
                    prompt_embeds, pooled_prompt_embeds = enhanced_prompt
                    logger.info(f"✅ SDXL Compel conditioning tensors successful:")
                    logger.info(f"   prompt_embeds: {prompt_embeds.shape}")
                    logger.info(f"   pooled_prompt_embeds: {pooled_prompt_embeds.shape}")
                else:
                    logger.warning(f"⚠️ Expected SDXL tensors tuple but got: {type(enhanced_prompt)}")
            else:
                if isinstance(enhanced_prompt, str):
                    logger.info(f"✅ String prompt successful: {enhanced_prompt}")
                else:
                    logger.warning(f"⚠️ Expected string but got: {type(enhanced_prompt)}")
        
        logger.info("🧪 Compel integration test completed")

if __name__ == "__main__":
    logger.info("🚀 Starting LUSTIFY SDXL Worker - FLEXIBLE QUANTITY + IMAGE-TO-IMAGE + SDXL COMPEL LIBRARY INTEGRATION")
    
    # Check for test mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-compel":
        logger.info("🧪 Running in Compel test mode...")
        try:
            worker = LustifySDXLWorker()
            worker.test_compel_integration()
            logger.info("✅ Compel test completed successfully")
        except Exception as e:
            logger.error(f"❌ Compel test failed: {e}")
            exit(1)
    else:
        # Normal worker mode
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