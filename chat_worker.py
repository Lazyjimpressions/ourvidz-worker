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
        base_prompt = """You are an expert AI prompt engineer specializing in WAN 2.1 video generation and temporal consistency for adult content.

CRITICAL REQUIREMENTS:
- Target Model: WAN 2.1 T2V 1.3B (motion-focused, 5-second videos)
- Content Focus: Adult/NSFW content with temporal consistency, smooth motion, cinematic quality
- Quality Priority: Motion realism, scene coherence, professional cinematography, anatomical accuracy

ENHANCEMENT STRATEGY:
1. MOTION FIRST: Describe natural, fluid movements and transitions with realistic anatomy
2. TEMPORAL CONSISTENCY: Ensure elements maintain coherence across frames
3. CINEMATOGRAPHY: Add professional camera work (smooth pans, steady shots)
4. SCENE SETTING: Establish clear environment and spatial relationships  
5. TECHNICAL QUALITY: Video-specific quality terms (smooth motion, stable)
6. ANATOMICAL ACCURACY: Ensure realistic proportions and natural poses throughout motion

WAN-SPECIFIC OPTIMIZATION:
- Motion descriptions: "smooth movement, natural motion, fluid transitions"
- Temporal stability: "consistent lighting, stable composition, coherent scene"
- Cinematography: "professional camera work, smooth pans, steady shots"
- Video quality: "high framerate, smooth motion, temporal consistency"
- Scene coherence: "well-lit environment, clear spatial relationships"
- Adult content: "realistic anatomy, natural poses, authentic expressions"

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
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'], generated_ids)
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
            """Simple prompt enhancement using Qwen Instruct model"""
            try:
                data = request.get_json()
                if not data or 'prompt' not in data:
                    return jsonify({'success': False, 'error': 'Missing prompt'}), 400

                prompt = data['prompt']
                enhancement_type = data.get('enhancement_type', 'manual')
                job_type = data.get('job_type', 'sdxl_image_fast')
                quality = data.get('quality', 'fast')
                
                # Use direct Qwen Instruct enhancement
                result = self.enhance_prompt(
                    prompt, 
                    enhancement_type, 
                    job_type, 
                    quality
                )
                
                if result['success']:
                    logger.info(f"✅ Enhancement successful: {len(prompt)} → {len(result['enhanced_prompt'])} chars")
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
                    'enhancement_system': 'Direct Qwen Instruct Enhancement',
                    'features': {
                        'dynamic_system_prompts': True,
                        'job_type_optimization': True,
                        'quality_levels': True,
                        'memory_management': True,
                        'error_handling': True
                    },
                    'supported_job_types': {
                        'sdxl_image_fast': 'SDXL LUSTIFY fast mode (75 tokens)',
                        'sdxl_image_high': 'SDXL LUSTIFY high quality (120 tokens)',
                        'video_fast': 'WAN 2.1 fast mode (175 tokens)',
                        'video_high': 'WAN 2.1 high quality (250 tokens)',
                        'wan_7b_enhanced': 'WAN 2.1 + 7B enhanced mode'
                    },
                    'model_info': {
                        'model_name': 'Qwen2.5-7B-Instruct',
                        'model_loaded': self.model_loaded,
                        'enhancement_method': 'Direct Qwen Instruct with dynamic prompts'
                    },
                    'endpoints': {
                        '/enhance': 'POST - Simple prompt enhancement',
                        '/enhance/legacy': 'POST - Legacy enhancement (same as /enhance)',
                        '/enhancement/info': 'GET - This information'
                    }
                }
                
                return jsonify(info)
                
            except Exception as e:
                logger.error(f"❌ Enhancement info endpoint error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/chat', methods=['POST'])
        def chat_endpoint():
            """Dedicated conversational chat endpoint for Playground"""
            try:
                data = request.get_json()
                if not data or 'message' not in data:
                    return jsonify({'success': False, 'error': 'Missing message'}), 400

                message = data['message']
                system_prompt = data.get('system_prompt')  # Add this
                conversation_id = data.get('conversation_id')
                project_id = data.get('project_id')
                context_type = data.get('context_type', 'general')
                conversation_history = self.validate_conversation_history(data.get('conversation_history', []))
                
                logger.info(f"💬 Chat request: {message[:50]}... (conversation: {conversation_id})")
                
                # Generate conversational response
                result = self.generate_chat_response(
                    message=message,
                    system_prompt=system_prompt,  # Add this
                    conversation_id=conversation_id,
                    project_id=project_id,
                    context_type=context_type,
                    conversation_history=conversation_history
                )
                
                if result['success']:
                    logger.info(f"✅ Chat response generated in {result.get('generation_time', 0):.1f}s")
                    return jsonify({
                        'success': True,
                        'response': result['response'],
                        'generation_time': result.get('generation_time'),
                        'conversation_id': conversation_id,
                        'context_type': context_type,
                        'message_id': result.get('message_id'),
                        'system_prompt_used': result.get('system_prompt_used', False),
                        'unrestricted_mode': result.get('unrestricted_mode', False)
                    })
                else:
                    return jsonify(result), 500
                    
            except Exception as e:
                logger.error(f"❌ Chat endpoint error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'response': 'I apologize, but I encountered an error. Please try again.'
                }), 500

        @self.app.route('/chat/unrestricted', methods=['POST'])
        def chat_unrestricted_endpoint():
            """Dedicated unrestricted chat endpoint for adult content"""
            try:
                data = request.get_json()
                if not data or 'message' not in data:
                    return jsonify({'success': False, 'error': 'Missing message'}), 400

                message = data['message']
                system_prompt = data.get('system_prompt')
                conversation_id = data.get('conversation_id')
                project_id = data.get('project_id')
                context_type = data.get('context_type', 'unrestricted')
                conversation_history = self.validate_conversation_history(data.get('conversation_history', []))
                
                logger.info(f"🔓 Unrestricted chat request: {message[:50]}... (conversation: {conversation_id})")
                
                # Build conversation messages
                messages = self.build_conversation_messages(
                    message=message,
                    system_prompt=system_prompt,
                    context_type=context_type,
                    project_id=project_id,
                    conversation_history=conversation_history
                )
                
                # Generate unrestricted response
                result = self.generate_unrestricted_response(messages)
                result['unrestricted_mode'] = True
                result['conversation_id'] = conversation_id
                result['context_type'] = context_type
                
                if result['success']:
                    logger.info(f"✅ Unrestricted response generated in {result.get('generation_time', 0):.1f}s")
                    return jsonify(result)
                else:
                    return jsonify(result), 500
                    
            except Exception as e:
                logger.error(f"❌ Unrestricted chat endpoint error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'response': 'I apologize, but I encountered an error. Please try again.'
                }), 500

        @self.app.route('/chat/health', methods=['GET'])
        def chat_health():
            """Health check specifically for chat functionality"""
            return jsonify({
                'status': 'healthy',
                'chat_ready': self.model_loaded,
                'endpoints': {
                    '/chat': 'POST - Conversational chat',
                    '/chat/unrestricted': 'POST - Unrestricted adult content chat',
                    '/enhance': 'POST - Simple prompt enhancement', 
                    '/health': 'GET - General health check'
                },
                'model_info': {
                    'loaded': self.model_loaded,
                    'model_name': 'Qwen2.5-7B-Instruct' if self.model_loaded else None,
                    'memory_usage': self.get_memory_info()
                },
                'timestamp': time.time()
            })

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

    def generate_chat_response(self, message: str, system_prompt: str = None, conversation_id: str = None, 
                             project_id: str = None, context_type: str = 'general',
                             conversation_history: list = None) -> dict:
        """Generate conversational response using Qwen Instruct model"""
        
        if not self.model_loaded:
            if not self.load_qwen_instruct_model():
                return {
                    'success': False,
                    'error': 'Model not available',
                    'response': 'I apologize, but the chat service is currently unavailable. Please try again later.'
                }

        try:
            start_time = time.time()
            
            # Check for unrestricted mode
            is_unrestricted = self.detect_unrestricted_mode(message)
            
            # Build conversation messages with history and context
            messages = self.build_conversation_messages(
                message=message,
                system_prompt=system_prompt,
                context_type=context_type,
                project_id=project_id,
                conversation_history=conversation_history or []
            )
            
            # Use unrestricted generation if detected
            if is_unrestricted:
                logger.info("🔓 Using unrestricted generation mode")
                result = self.generate_unrestricted_response(messages)
                result['unrestricted_mode'] = True
                return result
            
            # Apply chat template
            text = self.qwen_instruct_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize with proper parameters
            inputs = self.qwen_instruct_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096  # Increased for conversation context
            )
            
            # Verify inputs structure
            if not hasattr(inputs, 'input_ids') and 'input_ids' not in inputs:
                logger.error(f"❌ Tokenizer output missing input_ids: {type(inputs)}")
                return {
                    'success': False,
                    'error': 'Tokenization failed - missing input_ids',
                    'response': 'I apologize, but I encountered a technical error. Please try again.'
                }
            
            logger.info(f"✅ Tokenization successful for chat, input shape: {inputs.input_ids.shape if hasattr(inputs, 'input_ids') else 'unknown'}")
            
            # Move to device with better error handling
            try:
                # Handle both dict and object formats
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.model_device)
                logger.info("✅ Inputs moved to device successfully for chat")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("⚠️ Out of memory during tensor device transfer for chat, attempting cleanup...")
                    torch.cuda.empty_cache()
                    # Retry once after cleanup
                    try:
                        if isinstance(inputs, dict):
                            inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
                        else:
                            inputs = inputs.to(self.model_device)
                        logger.info("✅ Tensor transfer successful after memory cleanup for chat")
                    except RuntimeError as retry_e:
                        logger.error(f"❌ Tensor transfer failed even after cleanup for chat: {retry_e}")
                        raise
                else:
                    logger.error(f"❌ Device transfer error for chat: {e}")
                    raise
            
            # Generate conversational response
            with torch.no_grad():
                generated_ids = self.qwen_instruct_model.generate(
                    **inputs,
                    max_new_tokens=250,  # Reduced for better UI fit
                    do_sample=True,
                    temperature=0.8,     # Slightly higher for more personality
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.qwen_instruct_tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            try:
                # Handle both dict and object formats for input_ids access
                input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs.input_ids
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
                ]
                
                response = self.qwen_instruct_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                response = response.strip()
                
                if not response:
                    logger.warning("⚠️ Generated response is empty, using fallback")
                    response = "I understand your message, but I'm having trouble generating a response right now. Could you please rephrase or try again?"
                    
            except Exception as decode_error:
                logger.error(f"❌ Response decoding failed: {decode_error}")
                response = "I apologize, but I encountered an error while processing my response. Please try again."
            
            generation_time = time.time() - start_time
            self.stats['requests_served'] += 1
            
            logger.info(f"✅ Chat response completed in {generation_time:.1f}s")
            
            return {
                'success': True,
                'response': response,
                'generation_time': generation_time,
                'context_type': context_type,
                'message_id': f"msg_{int(time.time() * 1000)}",  # Simple message ID
                'system_prompt_used': system_prompt is not None,
                'unrestricted_mode': False
            }
            
        except Exception as e:
            logger.error(f"❌ Chat generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': 'I encountered an error generating a response. Please try again.'
            }

    def build_conversation_messages(self, message: str, system_prompt: str = None, context_type: str = 'general',
                                  project_id: str = None, conversation_history: list = None) -> list:
        """Build conversation messages with proper context and history"""
        
        # Apply dynamic system prompt if provided, otherwise use default
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}]
        else:
            # Build system prompt based on context
            default_system_prompt = self.build_conversation_system_prompt(context_type, project_id)
            messages = [{"role": "system", "content": default_system_prompt}]
        
        # Add conversation history (limit to last 10 exchanges for context window)
        if conversation_history:
            # Take last 20 messages (10 exchanges) to stay within context window
            recent_history = conversation_history[-20:] if len(conversation_history) > 20 else conversation_history
            
            for hist_msg in recent_history:
                if hist_msg.get('sender') == 'user':
                    messages.append({"role": "user", "content": hist_msg.get('content', '')})
                elif hist_msg.get('sender') == 'assistant':
                    messages.append({"role": "assistant", "content": hist_msg.get('content', '')})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        return messages

    def build_conversation_system_prompt(self, context_type: str, project_id: str = None) -> str:
        """Build system prompt based on conversation context"""
        
        if context_type == 'story_development':
            prompt = """You are an expert AI storytelling assistant specializing in adult content creation and narrative development.

CORE CAPABILITIES:
- Character development and consistency tracking
- Plot structure and story arc development
- Scene development and dialogue creation
- Story continuity and world-building
- Creative writing assistance and brainstorming

CONVERSATION STYLE:
- Engaging and collaborative storytelling partner
- Ask clarifying questions to understand the user's creative vision
- Offer specific, actionable suggestions for story improvement
- Maintain story consistency across conversations
- Support both creative exploration and technical execution
- Remember character details, plot points, and story rules

STORY MEMORY:
- Track character names, personalities, and relationships
- Remember plot developments and story timeline
- Maintain consistency with established story rules
- Reference previous story elements naturally in conversation

CURRENT CONTEXT:"""
            
            if project_id:
                prompt += f"\nProject ID: {project_id}"
                prompt += "\n(This conversation is linked to a specific project - maintain context accordingly)"
            
            prompt += """

Remember: You're helping create compelling, consistent stories. Be creative, supportive, and maintain narrative coherence throughout our conversation."""
            
        else:  # general conversation
            prompt = """You are a helpful AI assistant for OurVidz platform users and administrators.

CAPABILITIES:
- General conversation and assistance
- Platform guidance and support questions
- Creative brainstorming and ideation
- Technical questions about content creation
- Administrative support tasks
- Content planning and strategy discussions

CONVERSATION STYLE:
- Professional but friendly and approachable
- Conversational yet informative
- Helpful and solution-oriented
- Respectful of user privacy and preferences
- Maintain context across our conversation

PLATFORM CONTEXT:
- OurVidz is an AI-powered content creation platform
- Users create images and videos using AI models
- Focus on helping users achieve their creative goals

Provide helpful, accurate responses while maintaining a natural conversational tone. Remember previous parts of our conversation to provide contextual assistance."""
        
        return prompt

    def get_memory_info(self):
        """Get current memory information for health checks"""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                available = total - allocated
                
                return {
                    'total_vram_gb': round(total, 2),
                    'allocated_vram_gb': round(allocated, 2),
                    'available_vram_gb': round(available, 2),
                    'memory_usage_percent': round((allocated / total) * 100, 1)
                }
            else:
                return {'error': 'CUDA not available'}
        except Exception as e:
            return {'error': str(e)}

    def validate_conversation_history(self, history: list) -> list:
        """Validate and clean conversation history"""
        if not isinstance(history, list):
            return []
        
        valid_history = []
        for msg in history:
            if isinstance(msg, dict) and 'sender' in msg and 'content' in msg:
                if msg['sender'] in ['user', 'assistant'] and isinstance(msg['content'], str):
                    valid_history.append({
                        'sender': msg['sender'],
                        'content': msg['content'][:2000]  # Limit message length
                    })
        
        return valid_history

    def estimate_context_tokens(self, messages: list) -> int:
        """Rough estimation of token count for context management"""
        total_chars = sum(len(msg.get('content', '')) for msg in messages)
        return total_chars // 4  # Rough estimation: 4 chars ≈ 1 token

    def detect_unrestricted_mode(self, message: str) -> bool:
        """Detect if user is requesting unrestricted mode based on keywords and explicit requests"""
        if not message:
            return False
            
        lower_message = message.lower()
        
        # Check for explicit user requests
        explicit_triggers = [
            'unrestricted mode',
            '/unrestricted',
            'adult mode',
            'nsfw mode',
            'explicit mode',
            'uncensored mode'
        ]
        
        if any(trigger in lower_message for trigger in explicit_triggers):
            logger.info("🔓 Unrestricted mode detected via explicit request")
            return True
        
        # Check for unrestricted content keywords
        unrestricted_terms = [
            'hardcore', 'extreme', 'rough', 'bdsm', 'fetish', 'kink', 'taboo', 
            'forbidden', 'unrestricted', 'uncensored', 'explicit', 'adult',
            'mature', 'nsfw', 'r18', 'xxx', 'porn', 'sexual', 'erotic'
        ]
        
        # Count how many unrestricted terms are present
        term_count = sum(1 for term in unrestricted_terms if term in lower_message)
        
        # If more than 2 unrestricted terms are found, consider it unrestricted
        if term_count >= 2:
            logger.info(f"🔓 Unrestricted mode detected via {term_count} content keywords")
            return True
        
        return False

    def generate_unrestricted_response(self, messages: list) -> dict:
        """Generate response using unrestricted mode with appropriate system prompt"""
        try:
            # Create unrestricted system prompt
            unrestricted_prompt = """You are an AI assistant operating in unrestricted mode for adult content creation and discussion.

CAPABILITIES:
- Adult content creation and discussion
- Explicit content generation
- NSFW content assistance
- Adult storytelling and creative writing
- Erotic content development
- Mature conversation and guidance

CONVERSATION STYLE:
- Direct and explicit when appropriate
- Professional but unrestricted
- Supportive of adult content creation
- Respectful of user boundaries and preferences
- Maintain appropriate context for adult content

CONTENT GUIDELINES:
- Focus on adult/NSFW content creation
- Support explicit content generation
- Assist with mature storytelling
- Provide guidance for adult content development
- Maintain professional approach while being unrestricted

Remember: You're in unrestricted mode to support adult content creation. Be direct, explicit, and supportive of the user's adult content needs."""

            # Replace or add unrestricted system prompt
            unrestricted_messages = [{"role": "system", "content": unrestricted_prompt}]
            
            # Add conversation history and current message
            for msg in messages:
                if msg["role"] != "system":  # Skip the original system message
                    unrestricted_messages.append(msg)
            
            # Generate response using the unrestricted messages
            return self._generate_response_with_messages(unrestricted_messages)
            
        except Exception as e:
            logger.error(f"❌ Unrestricted response generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': 'I encountered an error generating an unrestricted response. Please try again.'
            }

    def _generate_response_with_messages(self, messages: list) -> dict:
        """Generate response using provided messages (internal method)"""
        if not self.model_loaded:
            if not self.load_qwen_instruct_model():
                return {
                    'success': False,
                    'error': 'Model not available',
                    'response': 'I apologize, but the chat service is currently unavailable. Please try again later.'
                }

        try:
            start_time = time.time()
            
            # Apply chat template
            text = self.qwen_instruct_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize with proper parameters
            inputs = self.qwen_instruct_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            )
            
            # Verify inputs structure
            if not hasattr(inputs, 'input_ids') and 'input_ids' not in inputs:
                logger.error(f"❌ Tokenizer output missing input_ids: {type(inputs)}")
                return {
                    'success': False,
                    'error': 'Tokenization failed - missing input_ids',
                    'response': 'I apologize, but I encountered a technical error. Please try again.'
                }
            
            # Move to device
            try:
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.model_device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if isinstance(inputs, dict):
                        inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
                    else:
                        inputs = inputs.to(self.model_device)
                else:
                    raise
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.qwen_instruct_model.generate(
                    **inputs,
                    max_new_tokens=250,  # Reduced for better UI fit
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.qwen_instruct_tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            try:
                input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs.input_ids
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
                ]
                
                response = self.qwen_instruct_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                response = response.strip()
                
                if not response:
                    response = "I understand your message, but I'm having trouble generating a response right now. Could you please rephrase or try again?"
                    
            except Exception as decode_error:
                logger.error(f"❌ Response decoding failed: {decode_error}")
                response = "I apologize, but I encountered an error while processing my response. Please try again."
            
            generation_time = time.time() - start_time
            self.stats['requests_served'] += 1
            
            return {
                'success': True,
                'response': response,
                'generation_time': generation_time,
                'message_id': f"msg_{int(time.time() * 1000)}"
            }
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': 'I encountered an error generating a response. Please try again.'
            }

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