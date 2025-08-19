# sdxl_worker.py - FLEXIBLE QUANTITY + IMAGE-TO-IMAGE VERSION + SDXL COMPEL LIBRARY INTEGRATION - CONSISTENT PARAMETER NAMING
# NEW: Supports user-selected quantities (1, 3, or 6 images) and image-to-image generation
# FIXED: SDXL-specific Compel library integration with prompt_embeds and pooled_prompt_embeds
# FIXED: Consistent callback parameter names (job_id, assets) for edge function compatibility
# NEW: I2I pipeline with denoise_strength and thumbnail generation
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
        "denoise_strength": 0.7,  # NEW: Use denoise_strength instead of reference_strength
        "exact_copy_mode": false  # NEW: Explicit mode flag
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

🎯 I2I PIPELINE CHANGES:
- Use StableDiffusionXLImg2ImgPipeline for all I2I operations
- Accept denoise_strength (0-1) instead of reference_strength
- Support exact_copy_mode for promptless exact copies
- Generate thumbnails for all images
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
import traceback  # ADD: Traceback for better error logging
import compel  # ADD: Compel library for proper prompt weighting
from compel import Compel  # ADD: Compel processor
from io import BytesIO  # ADD: BytesIO for image serialization
sys.path.append('/workspace/python_deps/lib/python3.11/site-packages')
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
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
        print("🖼️ NEW: Image-to-image generation with I2I pipeline and denoise_strength")
        print("🌱 NEW: Seed control for reproducible generation and character consistency")
        print("🔧 FIXED: SDXL-specific Compel library integration with prompt_embeds and pooled_prompt_embeds")
        print("🔧 FIXED: Consistent parameter naming (job_id, assets, metadata) across all callbacks")
        print("✅ API COMPLIANT: Supports metadata.reference_image_url, denoise_strength, exact_copy_mode")
        print("🖼️ NEW: Thumbnail generation for all images")
        
        # Model configuration
        self.model_path = "/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors"
        self.pipeline = None
        self.i2i_pipeline = None  # NEW: Separate I2I pipeline
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

    def generate_with_i2i_pipeline(self, prompt, reference_image, denoise_strength, exact_copy_mode, config, num_images=1, generators=None):
        """Generate images using StableDiffusionXLImg2ImgPipeline with denoise_strength"""
        logger.info(f"🎨 I2I pipeline generation with denoise_strength: {denoise_strength}, exact_copy_mode: {exact_copy_mode}")
        
        # Preprocess reference image to model size
        processed_image = self.preprocess_reference_image(reference_image, (config['width'], config['height']))
        
        # Use provided generators or create new ones
        if generators is None:
            generators = [torch.Generator(device="cuda").manual_seed(int(time.time()) + i) for i in range(num_images)]
        
        # Configure generation parameters based on mode
        if exact_copy_mode:
            # Promptless exact copy mode - Worker-side guard clamping
            denoise_strength = min(denoise_strength, 0.05)  # Clamp to ≤ 0.05
            guidance_scale = 1.0
            negative_prompt = None  # Omit negative prompt
            steps = min(max(6, int(denoise_strength * 100)), 10)  # 6-10 steps
            negative_prompt_used = False
            logger.info(f"📋 Exact copy mode: denoise_strength={denoise_strength}, guidance_scale={guidance_scale}, steps={steps}, negative_prompt_used={negative_prompt_used}")
        else:
            # Reference modify mode
            denoise_strength = max(0.10, min(denoise_strength, 0.25))  # Clamp to [0.10-0.25]
            guidance_scale = max(4.0, min(config['guidance_scale'], 7.0))  # Clamp to [4-7]
            steps = max(15, min(config['num_inference_steps'], 30))  # Clamp to [15-30]
            negative_prompt = [
                "blurry, low quality, distorted, deformed, bad anatomy, "
                "watermark, signature, text, logo, extra limbs, missing limbs"
            ] * num_images
            negative_prompt_used = True
            logger.info(f"📋 Reference modify mode: denoise_strength={denoise_strength}, guidance_scale={guidance_scale}, steps={steps}, negative_prompt_used={negative_prompt_used}")
        
        # Prepare generation kwargs
        generation_kwargs = {
            'image': processed_image,
            'strength': denoise_strength,
            'num_inference_steps': steps,
            'guidance_scale': guidance_scale,
            'num_images_per_prompt': 1,
            'generator': generators
        }
        
        # Handle prompt based on mode
        if exact_copy_mode:
            # Empty prompt for exact copy
            generation_kwargs['prompt'] = [""] * num_images
        else:
            # Use provided prompt for modification
            if isinstance(prompt, str):
                generation_kwargs['prompt'] = [prompt] * num_images
            elif isinstance(prompt, dict) and 'prompt_embeds' in prompt:
                # Handle Compel conditioning tensors
                prompt_embeds = prompt['prompt_embeds']
                pooled_prompt_embeds = prompt['pooled_prompt_embeds']
                negative_prompt_embeds = prompt['negative_prompt_embeds']
                negative_pooled_prompt_embeds = prompt['negative_pooled_prompt_embeds']
                
                # Replicate tensors for batch generation
                if num_images > 1:
                    prompt_embeds = prompt_embeds.repeat(num_images, 1, 1)
                    pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_images, 1)
                    negative_prompt_embeds = negative_prompt_embeds.repeat(num_images, 1, 1)
                    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(num_images, 1)
                
                generation_kwargs['prompt_embeds'] = prompt_embeds
                generation_kwargs['pooled_prompt_embeds'] = pooled_prompt_embeds
                generation_kwargs['negative_prompt_embeds'] = negative_prompt_embeds
                generation_kwargs['negative_pooled_prompt_embeds'] = negative_pooled_prompt_embeds
                logger.info("✅ Using Compel conditioning tensors for I2I generation")
            else:
                generation_kwargs['prompt'] = [str(prompt)] * num_images
        
        # Add negative prompt if not in exact copy mode
        if not exact_copy_mode and negative_prompt:
            generation_kwargs['negative_prompt'] = negative_prompt
        
        # Generate using I2I pipeline
        with torch.inference_mode():
            result = self.i2i_pipeline(**generation_kwargs)
            return result.images, negative_prompt_used

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
        """Load LUSTIFY SDXL model with optimizations for both text-to-image and image-to-image"""
        if self.model_loaded:
            return
            
        logger.info("📦 Loading LUSTIFY SDXL v2.0 (Text-to-Image + Image-to-Image)...")
        start_time = time.time()
        
        try:
            # Load text-to-image pipeline from single file
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to("cuda")
            
            # Load image-to-image pipeline from the same model
            self.i2i_pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to("cuda")
            
            # Enable memory optimizations for both pipelines
            for pipeline_name, pipeline in [("Text-to-Image", self.pipeline), ("Image-to-Image", self.i2i_pipeline)]:
                try:
                    pipeline.enable_attention_slicing()
                    logger.info(f"✅ Attention slicing enabled for {pipeline_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Attention slicing failed for {pipeline_name}: {e}")
                
                # Try xformers if available
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info(f"✅ xformers optimization enabled for {pipeline_name}")
                except Exception as e:
                    logger.warning(f"⚠️ xformers optimization failed for {pipeline_name}: {e}")
            
            load_time = time.time() - start_time
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            
            self.model_loaded = True
            logger.info(f"✅ LUSTIFY loaded in {load_time:.1f}s, using {vram_used:.1f}GB VRAM")
            logger.info(f"✅ Both text-to-image and image-to-image pipelines ready")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise

    def process_compel_weights(self, prompt, weights_config=None):
        """
        Process prompt with proper Compel library integration for SDXL
        FIXED: Proper prompt structure and weight balancing
        """
        if not weights_config:
            return prompt, None
        try:
            if not self.model_loaded:
                self.load_model()
            # Use safe version logging for Compel
            version = getattr(compel, '__version__', 'unknown')
            logger.info(f"🔧 Initializing Compel {version} with SDXL encoders")
            compel_processor = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                requires_pooled=[False, True]
            )
            logger.info(f"✅ Compel processor initialized successfully")
            
            # CRITICAL FIX: Clean up duplicate weights and balance the prompt
            cleaned_weights = self.clean_compel_weights(weights_config)
            
            # CRITICAL FIX: Proper prompt structure - subject first, then enhancement
            # PATCH: Do NOT wrap the prompt in parentheses or give it a weight!
            combined_prompt = f"{prompt}, {cleaned_weights}" if cleaned_weights else prompt
            logger.info(f"📝 Original prompt: {prompt}")
            logger.info(f"🎯 Cleaned Compel weights: {cleaned_weights}")
            logger.info(f"📝 Final combined prompt: {combined_prompt}")
            # Log token count for combined prompt
            try:
                token_count = len(self.pipeline.tokenizer(combined_prompt).input_ids)
                logger.info(f"🔢 Token count (prompt + compel): {token_count}")
            except Exception as e:
                logger.warning(f"⚠️ Could not compute token count for combined prompt: {e}")
            
            prompt_embeds, pooled_prompt_embeds = compel_processor(combined_prompt)
            
            # CRITICAL FIX: Generate negative conditioning as well
            negative_prompt = ("blurry, low quality, distorted, deformed, bad anatomy, "
                             "watermark, signature, text, logo, extra limbs, missing limbs")
            # Truncate long negatives to fit within 77 tokens
            try:
                neg_ids = self.pipeline.tokenizer(negative_prompt, truncation=True, max_length=77).input_ids
                negative_prompt = self.pipeline.tokenizer.decode(neg_ids, skip_special_tokens=True)
                logger.info(f"✂️ Trimmed negative prompt: {negative_prompt}")
                neg_token_count = len(neg_ids)
                logger.info(f"🔢 Negative prompt token count: {neg_token_count}")
            except Exception as e:
                logger.warning(f"⚠️ Could not trim or count tokens for negative prompt: {e}")
            
            negative_prompt_embeds, negative_pooled_prompt_embeds = compel_processor(negative_prompt)
            
            logger.info(f"✅ Compel weights applied successfully with balanced prompt structure")
            logger.info(f"🔧 Generated prompt_embeds: {prompt_embeds.shape}")
            logger.info(f"🔧 Generated pooled_prompt_embeds: {pooled_prompt_embeds.shape}")
            logger.info(f"🔧 Generated negative_prompt_embeds: {negative_prompt_embeds.shape}")
            logger.info(f"🔧 Generated negative_pooled_prompt_embeds: {negative_pooled_prompt_embeds.shape}")
            
            # Return both positive and negative conditioning
            return {
                'prompt_embeds': prompt_embeds,
                'pooled_prompt_embeds': pooled_prompt_embeds,
                'negative_prompt_embeds': negative_prompt_embeds,
                'negative_pooled_prompt_embeds': negative_pooled_prompt_embeds
            }, prompt
        except Exception as e:
            logger.error(f"❌ Compel processing failed: {e}")
            logger.info(f"🔄 Falling back to original prompt: {prompt}")
            return prompt, None  # Fallback to original prompt

    def clean_compel_weights(self, weights_config):
        """Clean and deduplicate Compel weights (case-insensitive, normalized)"""
        if not weights_config:
            return ""
        
        # Split by commas and clean each weight
        weights = [w.strip() for w in weights_config.split(',')]
        
        # Normalize weights for deduplication: strip, lowercase, remove extra spaces inside parentheses/colons
        def normalize_weight(w):
            w = w.strip().lower()
            # Remove spaces after '(' and before ')', and around ':'
            if w.startswith('(') and w.endswith(')'):
                w = w[1:-1].strip()
                if ':' in w:
                    parts = w.split(':', 1)
                    left = parts[0].strip()
                    right = parts[1].strip()
                    w = f"({left}:{right})"
                else:
                    w = f"({w})"
            return w
        
        seen = set()
        cleaned_weights = []
        for weight in weights:
            norm = normalize_weight(weight)
            if norm not in seen:
                cleaned_weights.append(weight.strip())  # Keep original formatting for output
                seen.add(norm)
            else:
                logger.info(f"🧹 Removed duplicate weight (normalized): {weight}")
        
        # Limit to reasonable number of weights (max 6 for token efficiency)
        if len(cleaned_weights) > 6:
            logger.info(f"🎯 Limiting weights from {len(cleaned_weights)} to 6 for token efficiency")
            cleaned_weights = cleaned_weights[:6]
        
        result = ", ".join(cleaned_weights)
        logger.info(f"🧹 Cleaned weights: {weights_config} → {result}")
        return result

    def generate_images_batch(self, prompt, job_type, num_images=1, reference_image=None, denoise_strength=0.5, exact_copy_mode=False, seed=None):
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
            if exact_copy_mode:
                logger.info(f"🎨 Generating {num_images} image(s) in exact copy mode (denoise_strength: {denoise_strength})")
            else:
                logger.info(f"🎨 Generating {num_images} image(s) with reference modification (denoise_strength: {denoise_strength})")
        else:
            if isinstance(prompt, str):
                logger.info(f"🎨 Generating {num_images} image(s) for {job_type}: {prompt[:50]}...")
            elif isinstance(prompt, dict) and 'prompt_embeds' in prompt:
                logger.info(f"🎨 Generating {num_images} image(s) for {job_type}: [Compel conditioning tensors]...")
            else:
                logger.info(f"🎨 Generating {num_images} image(s) for {job_type}: {str(prompt)[:50]}...")
            
        if num_images > 1:
            logger.info(f"📊 Expected performance: {num_images * config['expected_time_per_image']:.0f}s total")
        start_time = time.time()
        
        try:
            # Clear GPU cache before generation
            torch.cuda.empty_cache()
            
            # Handle image-to-image generation using I2I pipeline
            if reference_image:
                images, negative_prompt_used = self.generate_with_i2i_pipeline(prompt, reference_image, denoise_strength, exact_copy_mode, config, num_images, generators)
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
                if isinstance(prompt, dict) and 'prompt_embeds' in prompt:
                    # Compel conditioning dictionary was returned with negative embeddings
                    prompt_embeds = prompt['prompt_embeds']
                    pooled_prompt_embeds = prompt['pooled_prompt_embeds']
                    negative_prompt_embeds = prompt['negative_prompt_embeds']
                    negative_pooled_prompt_embeds = prompt['negative_pooled_prompt_embeds']
                    
                    # CRITICAL FIX: Replicate tensors for batch generation
                    if num_images > 1:
                        prompt_embeds = prompt_embeds.repeat(num_images, 1, 1)
                        pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_images, 1)
                        negative_prompt_embeds = negative_prompt_embeds.repeat(num_images, 1, 1)
                        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(num_images, 1)
                        logger.info(f"🔧 Replicated Compel tensors for {num_images} images")
                        logger.info(f"🔧 prompt_embeds shape: {prompt_embeds.shape}")
                        logger.info(f"🔧 pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")
                        logger.info(f"🔧 negative_prompt_embeds shape: {negative_prompt_embeds.shape}")
                        logger.info(f"🔧 negative_pooled_prompt_embeds shape: {negative_pooled_prompt_embeds.shape}")
                    
                    generation_kwargs['prompt_embeds'] = prompt_embeds
                    generation_kwargs['pooled_prompt_embeds'] = pooled_prompt_embeds
                    generation_kwargs['negative_prompt_embeds'] = negative_prompt_embeds
                    generation_kwargs['negative_pooled_prompt_embeds'] = negative_pooled_prompt_embeds
                    
                    logger.info("✅ Using Compel conditioning tensors with negative conditioning for SDXL")
                
                elif isinstance(prompt, torch.Tensor):
                    # Legacy single conditioning tensor (fallback)
                    if num_images > 1:
                        prompt = prompt.repeat(num_images, 1, 1)
                        logger.info(f"🔧 Replicated legacy tensor for {num_images} images")
                    generation_kwargs['prompt_embeds'] = prompt
                    logger.info("✅ Using single Compel conditioning tensor for generation (legacy)")
                    
                else:
                    # String prompt (no Compel or Compel failed)
                    generation_kwargs['prompt'] = [prompt] * num_images  # Replicate prompt for batch
                    
                    # Only add negative prompt for string prompts (not conditioning tensors)
                    generation_kwargs['negative_prompt'] = [
                        "blurry, low quality, distorted, deformed, bad anatomy, "
                        "watermark, signature, text, logo, extra limbs, missing limbs"
                    ] * num_images
                    
                    logger.info("✅ Using string prompt for generation (no Compel)")
                
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
            
            # Return appropriate tuple based on generation type
            if reference_image:
                return images, seed, negative_prompt_used
            else:
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

    def upload_to_supabase_storage(self, bucket, path, file_data, content_type='image/png'):
        """Upload file data to Supabase storage bucket with correct Content-Type"""
        try:
            supabase_url = os.environ.get('SUPABASE_URL')
            supabase_service_key = os.environ.get('SUPABASE_SERVICE_KEY')
            
            if not supabase_url or not supabase_service_key:
                logger.error("❌ Missing Supabase credentials")
                return None
            
            headers = {
                'Authorization': f"Bearer {supabase_service_key}",
                'Content-Type': content_type,  # ✅ Use correct Content-Type for PNG
                'x-upsert': 'true'
            }
            
            response = requests.post(
                f"{supabase_url}/storage/v1/object/{bucket}/{path}",
                data=file_data,
                headers=headers,
                timeout=60
            )
            
            if response.status_code in [200, 201]:
                return path
            else:
                logger.error(f"❌ Upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Upload error: {e}")
            logger.error(f"❌ Upload traceback: {traceback.format_exc()}")
            return None

    def generate_thumbnail(self, image, max_size=256):
        """Generate a thumbnail from the given image"""
        try:
            # Calculate new size maintaining aspect ratio
            width, height = image.size
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            
            # Resize image
            thumbnail = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to WEBP format
            thumbnail_buffer = BytesIO()
            thumbnail.save(thumbnail_buffer, format='WEBP', quality=85, optimize=True)
            thumbnail_buffer.seek(0)
            
            logger.info(f"✅ Generated thumbnail: {new_width}x{new_height} WEBP")
            return thumbnail_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"❌ Thumbnail generation failed: {e}")
            return None

    def upload_to_storage(self, images, job_id, user_id, used_seed, job_type, denoise_strength=None, negative_prompt_used=True, exact_copy_mode=False):
        """Upload images to workspace-temp bucket with thumbnails"""
        uploaded_assets = []
        
        for i, image in enumerate(images):
            # Simple path: workspace-temp/{user_id}/{job_id}/{index}.png
            storage_path = f"{user_id}/{job_id}/{i}.png"
            thumbnail_path = f"{user_id}/{job_id}/{i}.thumb.webp"
            logger.info(f"📤 Uploading image {i} to workspace-temp/{storage_path}")
            
            # Convert image to bytes with PNG optimization
            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG', optimize=True)  # ✅ Optimize PNG
            img_buffer.seek(0)  # ✅ Reset buffer position
            img_bytes = img_buffer.getvalue()
            
            # Generate thumbnail
            thumbnail_bytes = self.generate_thumbnail(image)
            
            # Upload original image to workspace-temp bucket
            upload_result = self.upload_to_supabase_storage(
                bucket='workspace-temp',
                path=storage_path,
                file_data=img_bytes,
                content_type='image/png'  # ✅ Explicitly set PNG Content-Type
            )
            
            # Upload thumbnail if generation was successful
            thumbnail_url = None
            if thumbnail_bytes and upload_result:
                thumbnail_upload_result = self.upload_to_supabase_storage(
                    bucket='workspace-temp',
                    path=thumbnail_path,
                    file_data=thumbnail_bytes,
                    content_type='image/webp'
                )
                if thumbnail_upload_result:
                    thumbnail_url = thumbnail_path
                    logger.info(f"✅ Successfully uploaded thumbnail {i} to workspace-temp/{thumbnail_path}")
                else:
                    logger.warning(f"⚠️ Failed to upload thumbnail {i} to workspace-temp/{thumbnail_path}")
            
            if upload_result:
                logger.info(f"✅ Successfully uploaded image {i} to workspace-temp/{storage_path} (Content-Type: image/png)")
                
                # Build metadata with denoise_strength if provided
                metadata = {
                    'width': image.width,
                    'height': image.height,
                    'format': 'png',
                    'batch_size': len(images),
                    'steps': self.job_configs[job_type]['num_inference_steps'],
                    'guidance_scale': self.job_configs[job_type]['guidance_scale'],
                    'seed': used_seed + i,  # Each image gets seed + index
                    'file_size_bytes': len(img_bytes),
                    'asset_index': i,
                    'negative_prompt_used': negative_prompt_used
                }
                
                # Add I2I-specific metadata if provided
                if denoise_strength is not None:
                    metadata['denoise_strength'] = denoise_strength
                    metadata['pipeline'] = 'img2img'
                    metadata['resize_policy'] = 'center_crop'
                
                asset = {
                    'type': 'image',
                    'url': storage_path,  # ✅ Use 'url' field as expected by edge function
                    'metadata': metadata
                }
                
                # Add thumbnail_url if available
                if thumbnail_url:
                    asset['thumbnail_url'] = thumbnail_url
                
                uploaded_assets.append(asset)
            else:
                logger.error(f"❌ Failed to upload image {i} to workspace-temp/{storage_path}")
        
        logger.info(f"📤 Upload complete: {len(uploaded_assets)}/{len(images)} images uploaded successfully")
        return uploaded_assets

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
        
        # Handle denoise_strength parameter (new) with fallback to reference_strength (deprecated)
        denoise_strength = metadata.get('denoise_strength')
        if denoise_strength is None:
            # Fallback to deprecated reference_strength
            reference_strength = metadata.get('reference_strength', 0.5)
            denoise_strength = 1.0 - reference_strength  # Convert reference_strength to denoise_strength
            logger.warning(f"⚠️ DEPRECATED: Using reference_strength={reference_strength}, converted to denoise_strength={denoise_strength}")
        else:
            logger.info(f"✅ Using denoise_strength: {denoise_strength}")
        
        # Extract exact_copy_mode flag
        exact_copy_mode = metadata.get('exact_copy_mode', False)
        if exact_copy_mode:
            logger.info(f"✅ Exact copy mode enabled")
        
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
            if exact_copy_mode:
                logger.info(f"🖼️ Image-to-image exact copy mode (denoise_strength: {denoise_strength})")
            else:
                logger.info(f"🖼️ Image-to-image reference modify mode (denoise_strength: {denoise_strength})")
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
                    if isinstance(final_prompt, dict) and 'prompt_embeds' in final_prompt:
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
                if isinstance(final_prompt, dict) and 'prompt_embeds' in final_prompt:
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
            
            # Logging Fix: Avoid dumping tensors in logs
            if isinstance(final_prompt, dict) and 'prompt_embeds' in final_prompt:
                logger.info(f"🎨 Generating {num_images} image(s) for {job_type}: [Compel conditioning tensors]...")
            elif isinstance(final_prompt, str):
                logger.info(f"🎨 Generating {num_images} image(s) for {job_type}: {final_prompt[:50]}...")
            else:
                logger.info(f"🎨 Generating {num_images} image(s) for {job_type}: {str(final_prompt)[:50]}...")
            
            # Generate batch of images with final prompt
            start_time = time.time()
            if reference_image:
                # I2I generation
                images, used_seed, negative_prompt_used = self.generate_images_batch(
                    final_prompt, 
                    job_type, 
                    num_images, 
                    reference_image=reference_image,
                    denoise_strength=denoise_strength,
                    exact_copy_mode=exact_copy_mode,
                    seed=seed
                )
            else:
                # Text-to-image generation
                images, used_seed = self.generate_images_batch(
                    final_prompt, 
                    job_type, 
                    num_images, 
                    reference_image=reference_image,
                    denoise_strength=denoise_strength,
                    exact_copy_mode=exact_copy_mode,
                    seed=seed
                )
                negative_prompt_used = True  # Always use negative prompt for text-to-image
            
            if not images:
                raise Exception("Image generation failed")
            
            # Upload all images to workspace-temp bucket with thumbnails
            uploaded_assets = self.upload_to_storage(
                images, job_id, user_id, used_seed, job_type, 
                denoise_strength=denoise_strength if reference_image else None,
                negative_prompt_used=negative_prompt_used,
                exact_copy_mode=exact_copy_mode
            )
            
            if not uploaded_assets:
                raise Exception("All image uploads failed")
            
            total_time = time.time() - start_time
            logger.info(f"✅ SDXL job {job_id} completed in {total_time:.1f}s")
            logger.info(f"📁 Generated {len(uploaded_assets)} images")
            logger.info(f"🌱 Seed used: {used_seed}")
            
            # Metadata Fix: Avoid dumping tensors in metadata
            callback_metadata = {
                'seed': used_seed,
                'generation_time': total_time,
                'num_images': len(uploaded_assets),
                'job_type': job_type,
                'original_prompt': original_prompt if original_prompt else prompt,
                'final_prompt': '[Compel conditioning tensors]' if isinstance(final_prompt, dict) and 'prompt_embeds' in final_prompt else str(final_prompt),
                'compel_enabled': compel_enabled,
                'compel_weights': compel_weights if compel_enabled else None,
                'compel_success': compel_success if compel_enabled else False,
                'enhancement_strategy': 'compel' if compel_success else 'fallback' if compel_enabled else 'none',
                'final_prompt_type': final_prompt_type
            }
            
            # CONSISTENT: Notify completion with standardized parameter names and metadata
            self.notify_completion(job_id, 'completed', assets=uploaded_assets, metadata=callback_metadata)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ SDXL job {job_id} failed: {error_msg}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
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
        """Main SDXL worker loop - handles both direct execution and module import"""
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

    def start(self):
        """Alternative start method for compatibility with different startup scripts"""
        logger.info("🚀 Starting SDXL Worker via start() method...")
        return self.run()

    def serve(self):
        """Alternative serve method for compatibility with different startup scripts"""
        logger.info("🚀 Starting SDXL Worker via serve() method...")
        return self.run()

    def launch(self):
        """Alternative launch method for compatibility with different startup scripts"""
        logger.info("🚀 Starting SDXL Worker via launch() method...")
        return self.run()

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
                if isinstance(enhanced_prompt, dict) and 'prompt_embeds' in enhanced_prompt:
                    prompt_embeds = enhanced_prompt['prompt_embeds']
                    pooled_prompt_embeds = enhanced_prompt['pooled_prompt_embeds']
                    negative_prompt_embeds = enhanced_prompt['negative_prompt_embeds']
                    negative_pooled_prompt_embeds = enhanced_prompt['negative_pooled_prompt_embeds']
                    logger.info(f"✅ SDXL Compel conditioning tensors successful:")
                    logger.info(f"   prompt_embeds: {prompt_embeds.shape}")
                    logger.info(f"   pooled_prompt_embeds: {pooled_prompt_embeds.shape}")
                    logger.info(f"   negative_prompt_embeds: {negative_prompt_embeds.shape}")
                    logger.info(f"   negative_pooled_prompt_embeds: {negative_pooled_prompt_embeds.shape}")
                else:
                    logger.warning(f"⚠️ Expected SDXL tensors dict but got: {type(enhanced_prompt)}")
            else:
                if isinstance(enhanced_prompt, str):
                    logger.info(f"✅ String prompt successful: {enhanced_prompt}")
                else:
                    logger.warning(f"⚠️ Expected string but got: {type(enhanced_prompt)}")
        
        logger.info("🧪 Compel integration test completed")


def main():
    """Main entry point for SDXL worker - handles both direct execution and module import"""
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

if __name__ == "__main__":
    main()