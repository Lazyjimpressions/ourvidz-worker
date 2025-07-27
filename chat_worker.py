#!/usr/bin/env python3
"""
OurVidz Chat Worker - Dedicated Qwen Instruct Service
Handles: Manual prompt enhancement, chat interface, admin utilities
Model: Qwen 2.5-7B Instruct (always loaded when possible)
Port: 7861

OPTIMIZATIONS:
- Model set to eval() mode for inference-only
- Device pinning with stored model_device
- PyTorch 2.0 compilation (when available)
- Comprehensive OOM error handling with retry logic
- Memory cleanup and validation
"""

import os
import sys
import time
import torch
import psutil
import signal
import threading
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Setup environment
PYTHON_DEPS_PATH = '/workspace/python_deps/lib/python3.11/site-packages'
if PYTHON_DEPS_PATH not in sys.path:
    sys.path.insert(0, PYTHON_DEPS_PATH)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancementSystemPrompts:
    """Intelligent system prompts for model-specific enhancement"""
    
    @staticmethod
    def get_sdxl_system_prompt(job_type="sdxl_image_fast", quality="fast"):
        """SDXL LUSTIFY-specific system prompt"""
        base_prompt = """You are an expert AI prompt engineer specializing in SDXL LUSTIFY model optimization and adult content generation.

CRITICAL REQUIREMENTS:
- Target Model: SDXL LUSTIFY v2.0 (75-token optimal, 150-token maximum)
- Content Focus: Adult/NSFW content with anatomical accuracy
- Quality Priority: Photorealistic details, professional lighting, technical excellence

ENHANCEMENT STRATEGY:
1. ANATOMY FIRST: Ensure accurate proportions, realistic features, natural poses
2. QUALITY TAGS: Add professional photography terms (4K, sharp focus, detailed)
3. LIGHTING: Specify cinematic lighting (soft lighting, natural light, studio lighting)
4. TECHNICAL: Include quality modifiers (masterpiece, best quality, ultra-detailed)
5. COMPRESS: Optimize for 75-token sweet spot while preserving key details

SDXL-SPECIFIC OPTIMIZATION:
- Use proven SDXL quality tags: "masterpiece, best quality, ultra detailed"
- Emphasize lighting: "professional photography, soft lighting, detailed"
- Anatomical accuracy: "realistic proportions, natural pose, detailed features"
- Avoid unnecessary words, prioritize visual impact terms
- Balance detail with token efficiency"""

        if quality == "high":
            base_prompt += """

HIGH QUALITY MODE:
- Extend to 100-120 tokens for maximum detail
- Add advanced technical terms: "photorealistic, hyperdetailed, professional grade"
- Include specific camera settings: "85mm lens, shallow depth of field"
- Enhanced lighting details: "rim lighting, volumetric lighting, perfect exposure"
"""
        
        return base_prompt

    @staticmethod 
    def get_wan_system_prompt(job_type="video_fast", quality="fast"):
        """WAN 2.1-specific system prompt"""
        base_prompt = """You are an expert AI prompt engineer specializing in WAN 2.1 video generation and temporal consistency.

CRITICAL REQUIREMENTS:
- Target Model: WAN 2.1 T2V 1.3B (motion-focused, 5-second videos)
- Content Focus: Temporal consistency, smooth motion, cinematic quality
- Quality Priority: Motion realism, scene coherence, professional cinematography

ENHANCEMENT STRATEGY:
1. MOTION FIRST: Describe natural, fluid movements and transitions
2. TEMPORAL CONSISTENCY: Ensure elements maintain coherence across frames
3. CINEMATOGRAPHY: Add professional camera work (smooth pans, steady shots)
4. SCENE SETTING: Establish clear environment and spatial relationships  
5. TECHNICAL QUALITY: Video-specific quality terms (smooth motion, stable)

WAN-SPECIFIC OPTIMIZATION:
- Motion descriptions: "smooth movement, natural motion, fluid transitions"
- Temporal stability: "consistent lighting, stable composition, coherent scene"
- Cinematography: "professional camera work, smooth pans, steady shots"
- Video quality: "high framerate, smooth motion, temporal consistency"
- Scene coherence: "well-lit environment, clear spatial relationships"

TOKEN STRATEGY: 150-250 tokens optimal for detailed motion description"""

        if "7b_enhanced" in job_type:
            base_prompt += """

QWEN 7B ENHANCED MODE:
- Leverage full 7B model capabilities for superior enhancement
- Advanced cinematography: "dynamic camera angles, professional composition"
- Complex motion: "multi-layered motion, realistic physics, natural timing"
- Enhanced storytelling: "narrative coherence, emotional resonance"
- Technical excellence: "broadcast quality, professional grade, cinema-level"
"""

        return base_prompt

    @staticmethod
    def get_enhancement_context(job_type, quality_level, model_target):
        """Generate contextual information for AI enhancement"""
        return {
            "job_type": job_type,
            "quality_level": quality_level, 
            "target_model": model_target,
            "token_target": 75 if "sdxl" in job_type else 200,
            "content_type": "video" if "video" in job_type else "image",
            "enhancement_level": "enhanced" if "7b" in job_type else "standard"
        }

# Enhanced prompt generation with context
def create_enhanced_messages(original_prompt, job_type="sdxl_image_fast", quality="fast"):
    """Create contextually-aware messages for AI enhancement"""
    
    # Determine system prompt based on job type
    if "sdxl" in job_type:
        system_prompt = EnhancementSystemPrompts.get_sdxl_system_prompt(job_type, quality)
        model_context = "SDXL LUSTIFY"
    elif "video" in job_type or "image" in job_type:
        system_prompt = EnhancementSystemPrompts.get_wan_system_prompt(job_type, quality)
        model_context = "WAN 2.1"
    else:
        # Fallback
        system_prompt = EnhancementSystemPrompts.get_sdxl_system_prompt(job_type, quality)
        model_context = "SDXL LUSTIFY"
    
    # Create enhancement context
    context = EnhancementSystemPrompts.get_enhancement_context(job_type, quality, model_context)
    
    # Build intelligent user prompt with context
    user_prompt = f"""ENHANCEMENT REQUEST:
Model Target: {context['target_model']}
Content Type: {context['content_type'].title()}
Quality Level: {context['quality_level'].title()}
Token Target: {context['token_target']} tokens optimal
Enhancement Level: {context['enhancement_level'].title()}

Original Prompt: "{original_prompt}"

Task: Enhance this prompt according to the system requirements above. Focus on {model_context}-specific optimization while maintaining the original creative intent. Ensure the enhancement is optimized for {context['content_type']} generation with {context['quality_level']} quality settings."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# Worker Level Implementation (chat_worker.py)
# FALLBACK & PERFORMANCE OPTIMIZATION LAYER

class ChatWorkerEnhancement:
    """Worker-level enhancement with fallbacks and optimization"""
    
    def __init__(self, chat_worker):
        self.chat_worker = chat_worker  # Reference to parent ChatWorker
        
        # Fallback system prompts (when edge function is unavailable)
        self.fallback_prompts = {
            'sdxl_fast': "You are an SDXL optimization expert. Create 75-token prompts with quality tags, anatomical accuracy, and professional lighting. Respond with enhanced prompt only.",
            'sdxl_high': "You are an elite SDXL expert. Create 100-120 token prompts with advanced quality, perfect anatomy, and studio lighting. Respond with enhanced prompt only.",
            'wan_fast': "You are a WAN 2.1 video expert. Create 175-token prompts with smooth motion, temporal consistency, and cinematography. Respond with enhanced prompt only.",
            'wan_high': "You are a WAN 2.1 + 7B expert. Create 250-token prompts with cinematic quality, complex motion, and broadcast standards. Respond with enhanced prompt only."
        }
        
        # Performance optimization
        self.prompt_cache = {}  # Cache recent enhancements
        self.model_warm = False
        
    def enhance_prompt_intelligent(self, request_data):
        """Enhanced prompt generation with edge function integration"""
        
        # Extract request parameters
        original_prompt = request_data.get('prompt', '')
        job_type = request_data.get('job_type', 'sdxl_image_fast')
        quality = request_data.get('quality', 'fast')
        enhancement_type = request_data.get('enhancement_type', 'manual')
        
        # Check if edge function provided system prompt
        edge_system_prompt = request_data.get('system_prompt')
        edge_context = request_data.get('context', {})
        
        try:
            if edge_system_prompt:
                # USE EDGE FUNCTION'S INTELLIGENT SYSTEM PROMPT
                logger.info("🧠 Using edge function's intelligent system prompt")
                messages = [
                    {"role": "system", "content": edge_system_prompt},
                    {"role": "user", "content": f"Original prompt: {original_prompt}"}
                ]
                enhancement_source = "edge_function"
                
            else:
                # FALLBACK TO WORKER'S BUILT-IN PROMPTS
                logger.info("🔄 Falling back to worker's built-in system prompts")
                fallback_key = self._get_fallback_key(job_type, quality)
                system_prompt = self.fallback_prompts.get(fallback_key, self.fallback_prompts['sdxl_fast'])
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Enhance this prompt: {original_prompt}"}
                ]
                enhancement_source = "worker_fallback"
            
            # Performance optimization - check cache
            cache_key = f"{original_prompt}_{job_type}_{quality}"
            if cache_key in self.prompt_cache:
                logger.info("⚡ Returning cached enhancement")
                cached_result = self.prompt_cache[cache_key].copy()
                cached_result['cache_hit'] = True
                return cached_result
            
            # Generate enhancement using the parent worker's model
            start_time = time.time()
            enhanced_result = self._generate_with_model(messages, original_prompt, enhancement_type, job_type, quality)
            
            if enhanced_result['success']:
                # Worker-level post-processing
                enhanced_result = self._worker_post_process(
                    enhanced_result, 
                    job_type, 
                    quality,
                    edge_context
                )
                
                # Cache successful results
                self.prompt_cache[cache_key] = enhanced_result.copy()
                
                # Cleanup cache if too large
                if len(self.prompt_cache) > 100:
                    # Remove oldest entries
                    oldest_keys = list(self.prompt_cache.keys())[:20]
                    for key in oldest_keys:
                        del self.prompt_cache[key]
                
                enhanced_result['enhancement_source'] = enhancement_source
                enhanced_result['worker_optimizations'] = {
                    'caching': True,
                    'post_processing': True,
                    'fallback_ready': True
                }
                
            return enhanced_result
            
        except Exception as e:
            logger.error(f"❌ Worker enhancement failed: {e}")
            # Ultimate fallback - return original with basic enhancement
            return self._emergency_fallback(original_prompt, job_type, quality)
    
    def _generate_with_model(self, messages, original_prompt, enhancement_type, job_type, quality):
        """Generate enhancement using the parent worker's model infrastructure"""
        # Use the parent worker's enhance_prompt method but with our custom messages
        # We'll temporarily override the message generation in the parent method
        
        if not self.chat_worker.model_loaded:
            # Try to load model
            if not self.chat_worker.load_qwen_instruct_model():
                return {
                    'success': False,
                    'error': 'Model not available',
                    'enhanced_prompt': original_prompt
                }

        try:
            logger.info(f"🤖 Intelligent enhancement for {job_type} with {quality} quality: {original_prompt[:50]}...")
            start_time = time.time()

            # Apply chat template and tokenize properly
            try:
                # Apply chat template
                text = self.chat_worker.qwen_instruct_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                logger.info(f"🔍 Chat template applied successfully, length: {len(text)}")
                
                # Tokenize with explicit parameters
                inputs = self.chat_worker.qwen_instruct_tokenizer(
                    text,  # Single string, not list
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )
                
                # Verify inputs structure
                if not hasattr(inputs, 'input_ids') and 'input_ids' not in inputs:
                    logger.error("❌ Tokenizer output missing input_ids")
                    return {
                        'success': False,
                        'error': 'Tokenization failed - missing input_ids',
                        'enhanced_prompt': original_prompt
                    }
                
                logger.info(f"✅ Tokenization successful, input shape: {inputs.input_ids.shape}")
                
            except Exception as e:
                logger.error(f"❌ Chat template or tokenization failed: {e}")
                return {
                    'success': False,
                    'error': f'Tokenization error: {str(e)}',
                    'enhanced_prompt': original_prompt
                }
            
            # Move to device with better error handling
            try:
                # Handle both dict and object formats
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.chat_worker.model_device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.chat_worker.model_device)
                logger.info("✅ Inputs moved to device successfully")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("⚠️ Out of memory during tensor device transfer, attempting cleanup...")
                    torch.cuda.empty_cache()
                    # Retry once after cleanup
                    try:
                        if isinstance(inputs, dict):
                            inputs = {k: v.to(self.chat_worker.model_device) for k, v in inputs.items()}
                        else:
                            inputs = inputs.to(self.chat_worker.model_device)
                        logger.info("✅ Tensor transfer successful after memory cleanup")
                    except RuntimeError as retry_e:
                        logger.error(f"❌ Tensor transfer failed even after cleanup: {retry_e}")
                        raise
                else:
                    logger.error(f"❌ Device transfer error: {e}")
                    raise

            # Generate enhanced prompt with better parameters
            try:
                with torch.no_grad():
                    generated_ids = self.chat_worker.qwen_instruct_model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.chat_worker.qwen_instruct_tokenizer.eos_token_id,
                        use_cache=True
                    )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("⚠️ Out of memory during generation, attempting cleanup...")
                    torch.cuda.empty_cache()
                    # Retry generation once after cleanup
                    try:
                        with torch.no_grad():
                            generated_ids = self.chat_worker.qwen_instruct_model.generate(
                                **inputs,
                                max_new_tokens=200,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                                repetition_penalty=1.1,
                                pad_token_id=self.chat_worker.qwen_instruct_tokenizer.eos_token_id,
                                use_cache=True
                            )
                        logger.info("✅ Generation successful after memory cleanup")
                    except RuntimeError as retry_e:
                        logger.error(f"❌ Generation failed even after cleanup: {retry_e}")
                        raise
                else:
                    logger.error(f"❌ Generation error: {e}")
                    raise

            # Decode response
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            enhanced = self.chat_worker.qwen_instruct_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            enhanced = enhanced.strip()

            generation_time = time.time() - start_time
            self.chat_worker.stats['requests_served'] += 1

            logger.info(f"✅ Intelligent enhancement completed in {generation_time:.1f}s")
            logger.info(f"📝 Length: {len(original_prompt)} → {len(enhanced)} chars")

            return {
                'success': True,
                'original_prompt': original_prompt,
                'enhanced_prompt': enhanced,
                'generation_time': generation_time,
                'enhancement_type': enhancement_type,
                'job_type': job_type,
                'quality': quality
            }

        except Exception as e:
            logger.error(f"❌ Intelligent enhancement failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'enhanced_prompt': original_prompt
            }
    
    def _get_fallback_key(self, job_type, quality):
        """Determine fallback prompt key based on job type and quality"""
        if 'sdxl' in job_type:
            return 'sdxl_high' if quality == 'high' else 'sdxl_fast'
        elif 'video' in job_type or '7b' in job_type:
            return 'wan_high' if quality == 'high' or '7b' in job_type else 'wan_fast'
        else:
            return 'sdxl_fast'  # Default fallback
    
    def _worker_post_process(self, result, job_type, quality, edge_context):
        """Worker-level post-processing and optimization"""
        enhanced_prompt = result['enhanced_prompt']
        
        # Token counting and compression (backup to edge function)
        token_count = len(enhanced_prompt.split())
        
        # SDXL-specific optimization
        if 'sdxl' in job_type:
            target_tokens = 120 if quality == 'high' else 75
            
            if token_count > target_tokens:
                logger.info(f"🔧 Worker-level token compression: {token_count} → {target_tokens}")
                enhanced_prompt = self._compress_for_sdxl(enhanced_prompt, target_tokens)
                result['enhanced_prompt'] = enhanced_prompt
                result['worker_compression'] = {
                    'original_tokens': token_count,
                    'compressed_tokens': len(enhanced_prompt.split()),
                    'target_tokens': target_tokens
                }
        
        # Quality validation
        quality_score = self._validate_enhancement_quality(enhanced_prompt, job_type)
        result['quality_score'] = quality_score
        
        # Model-specific validation
        if 'sdxl' in job_type:
            result['sdxl_optimizations'] = self._validate_sdxl_optimization(enhanced_prompt)
        elif 'video' in job_type:
            result['wan_optimizations'] = self._validate_wan_optimization(enhanced_prompt)
        
        return result
    
    def _compress_for_sdxl(self, prompt, target_tokens):
        """Intelligent compression for SDXL while preserving key elements"""
        words = prompt.split()
        
        if len(words) <= target_tokens:
            return prompt
        
        # Priority preservation: quality tags > subject > lighting > technical > style
        quality_terms = ['masterpiece', 'best quality', 'ultra detailed', '4K', '8K', 'hyperrealistic']
        lighting_terms = ['lighting', 'professional photography', 'studio', 'natural light']
        technical_terms = ['photorealistic', 'detailed', 'sharp focus', 'high resolution']
        
        preserved = []
        remaining_words = []
        
        # First pass: preserve critical quality terms
        for word in words:
            if any(term in word.lower() for term in quality_terms):
                preserved.append(word)
            else:
                remaining_words.append(word)
        
        # Second pass: fill remaining slots with most important content
        available_slots = target_tokens - len(preserved)
        if available_slots > 0:
            preserved.extend(remaining_words[:available_slots])
        
        compressed = ' '.join(preserved)
        logger.info(f"✂️ Compressed prompt: {len(words)} → {len(preserved)} tokens")
        
        return compressed
    
    def _validate_enhancement_quality(self, enhanced_prompt, job_type):
        """Validate enhancement quality with scoring"""
        score = 0
        enhanced_lower = enhanced_prompt.lower()
        
        # SDXL quality indicators
        if 'sdxl' in job_type:
            sdxl_quality_terms = [
                'masterpiece', 'best quality', 'ultra detailed',
                'professional photography', 'lighting', 'detailed',
                'photorealistic', 'high resolution', '4k', '8k'
            ]
            score += sum(1 for term in sdxl_quality_terms if term in enhanced_lower)
        
        # WAN quality indicators  
        elif 'video' in job_type:
            wan_quality_terms = [
                'smooth movement', 'natural motion', 'fluid transitions',
                'professional camera', 'cinematography', 'temporal consistency',
                'stable composition', 'smooth motion'
            ]
            score += sum(1 for term in wan_quality_terms if term in enhanced_lower)
        
        return min(score / 5.0, 1.0)  # Normalize to 0-1
    
    def _validate_sdxl_optimization(self, enhanced_prompt):
        """Validate SDXL-specific optimizations"""
        enhanced_lower = enhanced_prompt.lower()
        
        optimizations = {
            'has_quality_tags': any(term in enhanced_lower for term in ['masterpiece', 'best quality', 'ultra detailed']),
            'has_lighting': any(term in enhanced_lower for term in ['lighting', 'professional photography', 'studio']),
            'has_technical_terms': any(term in enhanced_lower for term in ['photorealistic', 'detailed', 'sharp focus']),
            'has_resolution': any(term in enhanced_lower for term in ['4k', '8k', 'high resolution']),
            'token_count': len(enhanced_prompt.split())
        }
        
        return optimizations
    
    def _validate_wan_optimization(self, enhanced_prompt):
        """Validate WAN-specific optimizations"""
        enhanced_lower = enhanced_prompt.lower()
        
        optimizations = {
            'has_motion': any(term in enhanced_lower for term in ['smooth movement', 'natural motion', 'fluid']),
            'has_cinematography': any(term in enhanced_lower for term in ['professional camera', 'cinematography', 'camera work']),
            'has_temporal': any(term in enhanced_lower for term in ['temporal consistency', 'stable', 'coherent']),
            'has_quality': any(term in enhanced_lower for term in ['high framerate', 'smooth motion', 'professional']),
            'token_count': len(enhanced_prompt.split())
        }
        
        return optimizations
    
    def _emergency_fallback(self, original_prompt, job_type, quality):
        """Emergency fallback when all enhancement fails"""
        
        # Basic enhancement based on job type
        if 'sdxl' in job_type:
            enhanced = f"masterpiece, best quality, ultra detailed, {original_prompt}, professional photography, detailed, photorealistic"
        elif 'video' in job_type:
            enhanced = f"smooth movement, natural motion, {original_prompt}, professional camera work, cinematic, temporal consistency"
        else:
            enhanced = f"high quality, detailed, {original_prompt}, professional"
        
        logger.warning(f"🚨 Emergency fallback enhancement applied")
        
        return {
            'success': True,
            'enhanced_prompt': enhanced,
            'original_prompt': original_prompt,
            'enhancement_source': 'emergency_fallback',
            'warning': 'Emergency fallback applied - reduced quality enhancement'
        }

class ChatWorker:
    def __init__(self):
        """Initialize Chat Worker with smart memory management"""
        self.app = Flask(__name__)
        self.port = 7861
        self.model_loaded = False
        self.loading_lock = threading.Lock()
        
        # Model paths - using verified working paths
        self.instruct_model_path = "/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
        self.qwen_instruct_model = None
        self.qwen_instruct_tokenizer = None
        
        # Performance tracking
        self.stats = {
            'requests_served': 0,
            'model_loads': 0,
            'model_unloads': 0,
            'startup_time': time.time()
        }
        
        # Setup environment
        self.setup_environment()
        
        # Initialize enhancement system
        self.enhancement = ChatWorkerEnhancement(self)
        
        # Setup Flask routes
        self.setup_routes()
        
        logger.info("🤖 Chat Worker initialized with optimized model loading and intelligent enhancement")

    def setup_environment(self):
        """Configure environment variables"""
        env_vars = {
            'CUDA_VISIBLE_DEVICES': '0',
            'TORCH_USE_CUDA_DSA': '1',
            'PYTHONUNBUFFERED': '1',
            'HF_HOME': '/workspace/models/huggingface_cache',
            'TRANSFORMERS_CACHE': '/workspace/models/huggingface_cache/hub',
            'HUGGINGFACE_HUB_CACHE': '/workspace/models/huggingface_cache/hub'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        logger.info("✅ Environment configured")

    def log_gpu_memory(self):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            cached = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"🔥 GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB, Total: {total:.0f}GB")

    def check_memory_available(self, required_gb=15):
        """Check if enough VRAM is available for model loading"""
        if not torch.cuda.is_available():
            return False
            
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated() / (1024**3)
        available = total - allocated
        
        logger.info(f"🔍 Memory check: {available:.1f}GB available, {required_gb}GB required")
        return available >= required_gb

    def load_qwen_instruct_model(self, force=False):
        """Load Qwen Instruct model with memory management"""
        with self.loading_lock:
            if self.model_loaded and not force:
                logger.info("✅ Qwen Instruct already loaded")
                return True

            # Check if model path exists
            if not os.path.exists(self.instruct_model_path):
                logger.error(f"❌ Model not found: {self.instruct_model_path}")
                return False

            # Check memory availability
            if not self.check_memory_available(15):
                logger.warning("⚠️ Insufficient VRAM for Qwen Instruct model")
                return False

            try:
                logger.info("🔄 Loading Qwen 2.5-7B Instruct model...")
                load_start = time.time()

                # Load tokenizer
                self.qwen_instruct_tokenizer = AutoTokenizer.from_pretrained(
                    self.instruct_model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )

                # Load model
                self.qwen_instruct_model = AutoModelForCausalLM.from_pretrained(
                    self.instruct_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True
                )
                
                # Store the actual device for consistent tensor operations
                self.model_device = next(self.qwen_instruct_model.parameters()).device
                
                # Set model to evaluation mode (disable dropout, etc.)
                self.qwen_instruct_model.eval()
                
                # Clean up any fragmented memory
                torch.cuda.empty_cache()
                
                # PyTorch 2.0 optimization (if available)
                try:
                    self.qwen_instruct_model = torch.compile(self.qwen_instruct_model)
                    logger.info("✅ PyTorch 2.0 compilation applied")
                except Exception as e:
                    logger.info(f"ℹ️ PyTorch 2.0 compilation not available: {e}")
                
                # Validate model works with a test inference
                logger.info("🔍 Validating model with test inference...")
                test_input = self.qwen_instruct_tokenizer(["test"], return_tensors="pt")
                with torch.no_grad():
                    _ = self.qwen_instruct_model(**test_input.to(self.model_device))
                logger.info("✅ Model validation successful")

                load_time = time.time() - load_start
                self.model_loaded = True
                self.stats['model_loads'] += 1
                
                logger.info(f"✅ Qwen Instruct loaded in {load_time:.1f}s")
                self.log_gpu_memory()
                return True

            except Exception as e:
                logger.error(f"❌ Failed to load Qwen Instruct: {e}")
                self.qwen_instruct_model = None
                self.qwen_instruct_tokenizer = None
                self.model_loaded = False
                return False

    def unload_qwen_instruct_model(self):
        """Unload Qwen Instruct model to free memory"""
        with self.loading_lock:
            if not self.model_loaded:
                logger.info("ℹ️ Qwen Instruct already unloaded")
                return

            try:
                logger.info("🗑️ Unloading Qwen Instruct model...")
                
                if self.qwen_instruct_model is not None:
                    del self.qwen_instruct_model
                if self.qwen_instruct_tokenizer is not None:
                    del self.qwen_instruct_tokenizer
                
                self.qwen_instruct_model = None
                self.qwen_instruct_tokenizer = None
                self.model_loaded = False
                self.stats['model_unloads'] += 1
                
                torch.cuda.empty_cache()
                logger.info("✅ Qwen Instruct unloaded")
                self.log_gpu_memory()

            except Exception as e:
                logger.error(f"❌ Error unloading model: {e}")

    def enhance_prompt(self, original_prompt, enhancement_type="manual", job_type="sdxl_image_fast", quality="fast"):
        """Enhanced prompt generation using Instruct model with dynamic system prompts"""
        if not self.model_loaded:
            # Try to load model
            if not self.load_qwen_instruct_model():
                return {
                    'success': False,
                    'error': 'Model not available',
                    'enhanced_prompt': original_prompt
                }

        try:
            logger.info(f"🤖 Enhancing prompt ({enhancement_type}) for {job_type} with {quality} quality: {original_prompt[:50]}...")
            start_time = time.time()

            # Use dynamic system prompts based on job type and quality
            messages = create_enhanced_messages(
                original_prompt=original_prompt,
                job_type=job_type,
                quality=quality
            )

            # FIXED: Apply chat template and tokenize properly
            try:
                # Apply chat template
                text = self.qwen_instruct_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                logger.info(f"🔍 Chat template applied successfully, length: {len(text)}")
                
                # Tokenize with explicit parameters
                inputs = self.qwen_instruct_tokenizer(
                    text,  # Single string, not list
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )
                
                # Verify inputs structure
                if not hasattr(inputs, 'input_ids') and 'input_ids' not in inputs:
                    logger.error("❌ Tokenizer output missing input_ids")
                    return {
                        'success': False,
                        'error': 'Tokenization failed - missing input_ids',
                        'enhanced_prompt': original_prompt
                    }
                
                logger.info(f"✅ Tokenization successful, input shape: {inputs.input_ids.shape}")
                
            except Exception as e:
                logger.error(f"❌ Chat template or tokenization failed: {e}")
                return {
                    'success': False,
                    'error': f'Tokenization error: {str(e)}',
                    'enhanced_prompt': original_prompt
                }
            
            # Move to device with better error handling
            try:
                # Handle both dict and object formats
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.model_device)
                logger.info("✅ Inputs moved to device successfully")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("⚠️ Out of memory during tensor device transfer, attempting cleanup...")
                    torch.cuda.empty_cache()
                    # Retry once after cleanup
                    try:
                        if isinstance(inputs, dict):
                            inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
                        else:
                            inputs = inputs.to(self.model_device)
                        logger.info("✅ Tensor transfer successful after memory cleanup")
                    except RuntimeError as retry_e:
                        logger.error(f"❌ Tensor transfer failed even after cleanup: {retry_e}")
                        raise
                else:
                    logger.error(f"❌ Device transfer error: {e}")
                    raise

            # Generate enhanced prompt with better parameters
            try:
                with torch.no_grad():
                    generated_ids = self.qwen_instruct_model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.qwen_instruct_tokenizer.eos_token_id,
                        # Remove early_stopping as it's not valid for this model
                        use_cache=True
                    )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("⚠️ Out of memory during generation, attempting cleanup...")
                    torch.cuda.empty_cache()
                    # Retry generation once after cleanup
                    try:
                        with torch.no_grad():
                            generated_ids = self.qwen_instruct_model.generate(
                                **inputs,
                                max_new_tokens=200,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                                repetition_penalty=1.1,
                                pad_token_id=self.qwen_instruct_tokenizer.eos_token_id,
                                use_cache=True
                            )
                        logger.info("✅ Generation successful after memory cleanup")
                    except RuntimeError as retry_e:
                        logger.error(f"❌ Generation failed even after cleanup: {retry_e}")
                        raise
                else:
                    logger.error(f"❌ Generation error: {e}")
                    raise

            # Decode response
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            enhanced = self.qwen_instruct_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            enhanced = enhanced.strip()

            generation_time = time.time() - start_time
            self.stats['requests_served'] += 1

            logger.info(f"✅ Enhancement completed in {generation_time:.1f}s")
            logger.info(f"📝 Length: {len(original_prompt)} → {len(enhanced)} chars")

            return {
                'success': True,
                'original_prompt': original_prompt,
                'enhanced_prompt': enhanced,
                'generation_time': generation_time,
                'enhancement_type': enhancement_type,
                'job_type': job_type,
                'quality': quality
            }

        except Exception as e:
            logger.error(f"❌ Enhancement failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'enhanced_prompt': original_prompt
            }

    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            uptime = time.time() - self.stats['startup_time']
            return jsonify({
                'status': 'healthy',
                'model_loaded': self.model_loaded,
                'uptime': uptime,
                'stats': self.stats
            })

        @self.app.route('/enhance', methods=['POST'])
        def enhance_endpoint():
            """Intelligent enhancement endpoint with edge function integration"""
            try:
                data = request.get_json()
                if not data or 'prompt' not in data:
                    return jsonify({'success': False, 'error': 'Missing prompt'}), 400

                # Use intelligent enhancement method
                result = self.enhancement.enhance_prompt_intelligent(data)
                
                if result['success']:
                    logger.info(f"✅ Enhancement successful via {result.get('enhancement_source', 'unknown')}")
                    return jsonify(result)
                else:
                    logger.error(f"❌ Enhancement failed: {result.get('error', 'Unknown error')}")
                    return jsonify(result), 500

            except Exception as e:
                logger.error(f"❌ Enhancement endpoint error: {e}")
                return jsonify({
                    'success': False, 
                    'error': str(e),
                    'enhanced_prompt': data.get('prompt', '') if data else ''
                }), 500

        @self.app.route('/enhance/intelligent', methods=['POST'])
        def enhance_endpoint_intelligent():
            """Intelligent enhancement endpoint with edge function integration"""
            try:
                data = request.get_json()
                if not data or 'prompt' not in data:
                    return jsonify({'success': False, 'error': 'Missing prompt'}), 400

                # Use intelligent enhancement method
                result = self.enhancement.enhance_prompt_intelligent(data)
                
                if result['success']:
                    logger.info(f"✅ Intelligent enhancement successful via {result.get('enhancement_source', 'unknown')}")
                    return jsonify(result)
                else:
                    logger.error(f"❌ Intelligent enhancement failed: {result.get('error', 'Unknown error')}")
                    return jsonify(result), 500

            except Exception as e:
                logger.error(f"❌ Intelligent enhancement endpoint error: {e}")
                return jsonify({
                    'success': False, 
                    'error': str(e),
                    'enhanced_prompt': data.get('prompt', '') if data else ''
                }), 500

        @self.app.route('/enhance/legacy', methods=['POST'])
        def enhance_endpoint_legacy():
            """Legacy enhancement endpoint for backward compatibility"""
            try:
                data = request.get_json()
                if not data or 'prompt' not in data:
                    return jsonify({'success': False, 'error': 'Missing prompt'}), 400

                prompt = data['prompt']
                enhancement_type = data.get('enhancement_type', 'manual')
                job_type = data.get('job_type', 'sdxl_image_fast')
                quality = data.get('quality', 'fast')
                
                result = self.enhance_prompt(
                    prompt, 
                    enhancement_type, 
                    job_type, 
                    quality
                )
                
                if result['success']:
                    return jsonify(result)
                else:
                    return jsonify(result), 500

            except Exception as e:
                logger.error(f"❌ Legacy enhancement endpoint error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/enhancement/info', methods=['GET'])
        def enhancement_info():
            """Get enhancement system information"""
            try:
                info = {
                    'enhancement_system': 'ChatWorkerEnhancement',
                    'features': {
                        'edge_function_integration': True,
                        'fallback_system': True,
                        'caching': True,
                        'post_processing': True,
                        'quality_validation': True,
                        'token_compression': True
                    },
                    'supported_job_types': {
                        'sdxl_image_fast': 'SDXL LUSTIFY fast mode (75 tokens)',
                        'sdxl_image_high': 'SDXL LUSTIFY high quality (120 tokens)',
                        'video_fast': 'WAN 2.1 fast mode (175 tokens)',
                        'video_high': 'WAN 2.1 high quality (250 tokens)',
                        'wan_7b_enhanced': 'WAN 2.1 + 7B enhanced mode'
                    },
                    'fallback_prompts': list(self.enhancement.fallback_prompts.keys()),
                    'cache_size': len(self.enhancement.prompt_cache),
                    'model_loaded': self.model_loaded
                }
                
                return jsonify(info)
                
            except Exception as e:
                logger.error(f"❌ Enhancement info endpoint error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/enhancement/cache/clear', methods=['POST'])
        def clear_enhancement_cache():
            """Clear enhancement cache"""
            try:
                cache_size = len(self.enhancement.prompt_cache)
                self.enhancement.prompt_cache.clear()
                logger.info(f"🗑️ Enhancement cache cleared ({cache_size} entries)")
                
                return jsonify({
                    'success': True,
                    'message': f'Cache cleared ({cache_size} entries)',
                    'cache_size': 0
                })
                
            except Exception as e:
                logger.error(f"❌ Cache clear error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/chat', methods=['POST'])
        def chat_endpoint():
            """Chat interface endpoint (future implementation)"""
            return jsonify({
                'success': False,
                'error': 'Chat endpoint not yet implemented',
                'message': 'Coming soon in Phase 2'
            }), 501

        @self.app.route('/admin', methods=['POST'])
        def admin_endpoint():
            """Admin utilities endpoint (future implementation)"""
            return jsonify({
                'success': False,
                'error': 'Admin endpoint not yet implemented',
                'message': 'Coming soon in Phase 2'
            }), 501

        @self.app.route('/memory/status', methods=['GET'])
        def memory_status():
            """Memory status endpoint"""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                available = total - allocated
                
                response = {
                    'total_vram': total,
                    'allocated_vram': allocated,
                    'available_vram': available,
                    'model_loaded': self.model_loaded
                }
                
                # Add device information if model is loaded
                if self.model_loaded and hasattr(self, 'model_device'):
                    response['model_device'] = str(self.model_device)
                    response['device_type'] = 'cuda' if 'cuda' in str(self.model_device) else 'cpu'
                
                return jsonify(response)
            else:
                return jsonify({'error': 'CUDA not available'}), 500

        @self.app.route('/memory/unload', methods=['POST'])
        def force_unload():
            """Force unload model (for memory management)"""
            self.unload_qwen_instruct_model()
            return jsonify({'success': True, 'message': 'Model unloaded'})

        @self.app.route('/memory/load', methods=['POST'])
        def force_load():
            """Force load model"""
            success = self.load_qwen_instruct_model(force=True)
            return jsonify({'success': success, 'message': 'Model load attempted'})

        @self.app.route('/model/info', methods=['GET'])
        def model_info():
            """Get detailed model information"""
            if not self.model_loaded:
                return jsonify({'error': 'Model not loaded'}), 404
            
            try:
                info = {
                    'model_loaded': True,
                    'model_path': self.instruct_model_path,
                    'model_device': str(self.model_device),
                    'device_type': 'cuda' if 'cuda' in str(self.model_device) else 'cpu',
                    'model_parameters': sum(p.numel() for p in self.qwen_instruct_model.parameters()),
                    'model_size_gb': sum(p.numel() * p.element_size() for p in self.qwen_instruct_model.parameters()) / (1024**3),
                    'is_eval_mode': self.qwen_instruct_model.training == False,
                    'torch_version': torch.__version__,
                    'cuda_available': torch.cuda.is_available()
                }
                
                if torch.cuda.is_available():
                    info['cuda_version'] = torch.version.cuda
                    info['gpu_name'] = torch.cuda.get_device_name(0)
                
                return jsonify(info)
            except Exception as e:
                return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

    def start_server(self):
        """Start the Flask server"""
        try:
            # Load model at startup if possible
            logger.info("🚀 Starting Chat Worker server...")
            self.load_qwen_instruct_model()
            
            # Start Flask server
            logger.info(f"🌐 Chat Worker listening on port {self.port}")
            self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
            
        except Exception as e:
            logger.error(f"❌ Server startup failed: {e}")
            raise

def auto_register_chat_worker():
    """Auto-register chat worker URL with Supabase"""
    try:
        print("🌐 Starting Chat Worker auto-registration...")
        
        # Detect RunPod URL
        pod_id = os.getenv('RUNPOD_POD_ID')
        if not pod_id:
            print("⚠️ RUNPOD_POD_ID not found - skipping auto-registration")
            return False
        
        worker_url = f"https://{pod_id}-7861.proxy.runpod.net"
        print(f"🔍 Detected Chat Worker URL: {worker_url}")
        
        # Validate environment variables
        supabase_url = os.getenv('SUPABASE_URL')
        service_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not supabase_url or not service_key:
            print("❌ Missing Supabase credentials")
            return False
        
        # Registration data
        registration_data = {
            "worker_url": worker_url,
            "auto_registered": True,
            "registration_method": "chat_worker_self_registration",
            "detection_method": "RUNPOD_POD_ID",
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"📝 Registering Chat Worker with Supabase...")
        
        # Call register-chat-worker edge function
        edge_function_url = f"{supabase_url}/functions/v1/register-chat-worker"
        
        response = requests.post(
            edge_function_url,
            headers={
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json"
            },
            json=registration_data,
            timeout=15
        )
        
        print(f"📄 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Chat Worker auto-registered successfully!")
                print(f"🎯 URL: {worker_url}")
                return True
            else:
                print(f"❌ Registration failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response: {response.text}")
        
        return False
        
    except Exception as e:
        print(f"❌ Auto-registration error: {e}")
        return False

if __name__ == "__main__":
    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        # Cleanup code here if needed
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start chat worker
    worker = ChatWorker()
    
    # Start Flask server in a separate thread to allow auto-registration
    import threading
    
    def start_server_thread():
        worker.start_server()
    
    server_thread = threading.Thread(target=start_server_thread, daemon=True)
    server_thread.start()
    
    # Wait a moment for server to be ready
    time.sleep(3)
    
    # Auto-register with Supabase
    auto_register_chat_worker()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Chat Worker...")
        sys.exit(0)