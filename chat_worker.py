#!/usr/bin/env python3
"""
OurVidz Chat Worker - Pure Inference Engine
Handles: Direct model inference with provided system prompts
Model: Qwen 2.5-7B Instruct (always loaded when possible)
Port: 7861

CRITICAL: This worker is a PURE INFERENCE ENGINE
- NO hardcoded system prompts
- NO prompt modification or enhancement logic
- ONLY executes what the edge functions provide
- ALL intelligence comes from database templates via edge functions

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

class ChatWorker:
    def __init__(self):
        """Initialize Chat Worker as pure inference engine"""
        self.app = Flask(__name__)
        self.port = 7861
        self.model_loaded = False
        self.loading_lock = threading.Lock()
        
        # Set memory fraction limit for Chat worker (15GB out of 48GB)
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.31)  # 15GB / 48GB
            logger.info("🧠 Memory fraction set to 0.31 (15GB) for Chat worker")
        
        # Model paths - allow env override; verified defaults
        self.instruct_model_path = os.getenv(
            'QWEN_INSTRUCT_PATH',
            "/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
        )
        self.base_model_path = os.getenv(
            'QWEN_BASE_PATH',
            "/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"
        )

        # Instruct (safety-tuned) model
        self.qwen_instruct_model = None
        self.qwen_instruct_tokenizer = None
        self.instruct_model_device = None

        # Base (no extra safety) model
        self.qwen_base_model = None
        self.qwen_base_tokenizer = None
        self.base_model_device = None

        # Active model tracking
        self.active_model_type = 'instruct'  # one of: 'instruct' | 'base'
        self.sfw_filters_enabled_by_default = False
        
        # Performance tracking
        self.stats = {
            'requests_served': 0,
            'model_loads': 0,
            'model_unloads': 0,
            'startup_time': time.time(),
            'sfw_requests': 0,
            'base_model_uses': 0,
            'instruct_model_uses': 0
        }
        
        # Setup environment
        self.setup_environment()
        
        # Setup Flask routes
        self.setup_routes()
        
        logger.info("🤖 Chat Worker initialized as PURE INFERENCE ENGINE - no hardcoded prompts")

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

    def apply_sfw_filters(self, text: str) -> str:
        """Apply minimal SFW redaction when explicitly requested by the user.
        Keeps semantics but masks explicit terms to meet SFW mode requirements."""
        try:
            if not text:
                return text
            redactions = [
                'sex', 'sexual', 'explicit', 'porn', 'pornographic', 'nsfw',
                'cum', 'orgasm', 'anal', 'vagina', 'penis', 'breast', 'boobs',
                'blowjob', 'handjob', 'fuck', 'fucking', 'suck', 'lick', 'moan',
                'nude', 'naked'
            ]
            sanitized = text
            for term in redactions:
                for variant in (term, term.capitalize(), term.upper()):
                    sanitized = sanitized.replace(variant, '▇▇')
            return sanitized
        except Exception:
            return text

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
                self.instruct_model_device = next(self.qwen_instruct_model.parameters()).device
                
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
                    _ = self.qwen_instruct_model(**test_input.to(self.instruct_model_device))
                logger.info("✅ Model validation successful")

                load_time = time.time() - load_start
                # Mark as loaded
                self.model_loaded = True
                self.stats['model_loads'] += 1
                
                logger.info(f"✅ Qwen Instruct loaded in {load_time:.1f}s")
                self.log_gpu_memory()
                return True

            except Exception as e:
                logger.error(f"❌ Failed to load Qwen Instruct: {e}")
                self.qwen_instruct_model = None
                self.qwen_instruct_tokenizer = None
                # Keep model_loaded reflecting if base is present
                self.model_loaded = self.qwen_base_model is not None
                return False

    def unload_qwen_instruct_model(self):
        """Unload Qwen Instruct model to free memory"""
        with self.loading_lock:
            if self.qwen_instruct_model is None:
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
                # Reflect global state
                self.model_loaded = self.qwen_base_model is not None
                self.stats['model_unloads'] += 1
                
                torch.cuda.empty_cache()
                logger.info("✅ Qwen Instruct unloaded")
                self.log_gpu_memory()

            except Exception as e:
                logger.error(f"❌ Error unloading model: {e}")

    def load_qwen_base_model(self, force=False):
        """Load Qwen 2.5-7B Base model (reduced safety filtering)."""
        with self.loading_lock:
            if self.qwen_base_model is not None and not force:
                logger.info("✅ Qwen Base already loaded")
                return True

            if not os.path.exists(self.base_model_path):
                logger.error(f"❌ Base model not found: {self.base_model_path}")
                return False

            if not self.check_memory_available(15):
                logger.warning("⚠️ Insufficient VRAM for Qwen Base model")
                return False

            try:
                logger.info("🔄 Loading Qwen 2.5-7B Base model...")
                load_start = time.time()

                # Load tokenizer
                self.qwen_base_tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )

                # Ensure pad token exists for base models
                if self.qwen_base_tokenizer.pad_token is None:
                    self.qwen_base_tokenizer.pad_token = self.qwen_base_tokenizer.eos_token

                # Load model
                self.qwen_base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True
                )

                self.qwen_base_model.eval()
                self.base_model_device = next(self.qwen_base_model.parameters()).device

                torch.cuda.empty_cache()

                try:
                    self.qwen_base_model = torch.compile(self.qwen_base_model)
                    logger.info("✅ PyTorch 2.0 compilation applied (base)")
                except Exception as e:
                    logger.info(f"ℹ️ PyTorch 2.0 compilation not available (base): {e}")

                # Simple validation
                logger.info("🔍 Validating base model with test inference...")
                test_input = self.qwen_base_tokenizer(["test"], return_tensors="pt")
                with torch.no_grad():
                    _ = self.qwen_base_model(**test_input.to(self.base_model_device))
                logger.info("✅ Base model validation successful")

                load_time = time.time() - load_start
                self.model_loaded = True
                self.stats['model_loads'] += 1
                logger.info(f"✅ Qwen Base loaded in {load_time:.1f}s")
                self.log_gpu_memory()
                return True

            except Exception as e:
                logger.error(f"❌ Failed to load Qwen Base: {e}")
                self.qwen_base_model = None
                self.qwen_base_tokenizer = None
                # Reflect instruct state
                self.model_loaded = self.qwen_instruct_model is not None
                return False

    def unload_qwen_base_model(self):
        """Unload Qwen Base model to free memory."""
        with self.loading_lock:
            if self.qwen_base_model is None:
                logger.info("ℹ️ Qwen Base already unloaded")
                return

            try:
                logger.info("🗑️ Unloading Qwen Base model...")
                del self.qwen_base_model
                del self.qwen_base_tokenizer
                self.qwen_base_model = None
                self.qwen_base_tokenizer = None
                self.model_loaded = self.qwen_instruct_model is not None
                self.stats['model_unloads'] += 1
                torch.cuda.empty_cache()
                logger.info("✅ Qwen Base unloaded")
                self.log_gpu_memory()
            except Exception as e:
                logger.error(f"❌ Error unloading base model: {e}")

    def generate_inference(self, messages: list, max_tokens: int = 512, temperature: float = 0.7, 
                          top_p: float = 0.9, use_base_model: bool = False) -> dict:
        """
        Pure inference method - executes exactly what is provided
        NO MODIFICATION of system prompts or messages
        """
        # Ensure requested model is available
        if use_base_model:
            if self.qwen_base_model is None:
                if not self.load_qwen_base_model():
                    return {
                        'success': False,
                        'error': 'Base model not available',
                        'response': 'Model is currently unavailable. Please try again later.'
                    }
            self.active_model_type = 'base'
            self.stats['base_model_uses'] += 1
            model = self.qwen_base_model
            tokenizer = self.qwen_base_tokenizer
            device = self.base_model_device
        else:
            if self.qwen_instruct_model is None:
                if not self.load_qwen_instruct_model():
                    return {
                        'success': False,
                        'error': 'Instruct model not available',
                        'response': 'Model is currently unavailable. Please try again later.'
                    }
            self.active_model_type = 'instruct'
            self.stats['instruct_model_uses'] += 1
            model = self.qwen_instruct_model
            tokenizer = self.qwen_instruct_tokenizer
            device = self.instruct_model_device

        try:
            start_time = time.time()
            
            # Log the messages being processed (for debugging only)
            logger.info(f"🔄 Processing {len(messages)} messages")
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:100]
                logger.info(f"   [{i}] {role}: {content}{'...' if len(msg.get('content', '')) > 100 else ''}")
            
            # Apply chat template if available; otherwise, minimal, role-preserving fallback
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                logger.info("ℹ️ Chat template unavailable for selected tokenizer - using fallback formatting")
                parts = []
                for m in messages:
                    role = m.get('role', 'user')
                    content = m.get('content', '')
                    parts.append(f"{role}: {content}")
                parts.append("assistant:")
                text = "\n".join(parts)
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            )
            
            # Verify inputs
            if 'input_ids' not in inputs:
                logger.error("❌ Tokenization failed - missing input_ids")
                return {
                    'success': False,
                    'error': 'Tokenization failed',
                    'response': 'I encountered a technical error. Please try again.'
                }
            
            # Move to device with OOM handling
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("⚠️ OOM during tensor transfer, cleaning up...")
                    torch.cuda.empty_cache()
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                else:
                    raise
            
            # Generate response with OOM handling
            try:
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("⚠️ OOM during generation, cleaning up and retrying...")
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            repetition_penalty=1.1,
                            pad_token_id=tokenizer.eos_token_id,
                            use_cache=True
                        )
                else:
                    raise
            
            # Extract only the new tokens (response)
            input_length = inputs['input_ids'].shape[1]
            new_tokens = generated_ids[0][input_length:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean response (minimal cleanup only)
            response = response.strip()
            
            # Basic validation
            if not response:
                response = "I couldn't generate a proper response. Please try again."
            
            generation_time = time.time() - start_time
            self.stats['requests_served'] += 1
            
            logger.info(f"✅ Inference completed in {generation_time:.2f}s, response length: {len(response)}")
            
            return {
                'success': True,
                'response': response,
                'generation_time': generation_time,
                'tokens_generated': len(new_tokens),
                'model_info': {
                    'model_name': 'Qwen2.5-7B-Base' if use_base_model else 'Qwen2.5-7B-Instruct',
                    'inference_engine': 'pure',
                    'active_model_type': self.active_model_type
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': 'I encountered an error during inference. Please try again.'
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
                'stats': self.stats,
                'worker_type': 'pure_inference_engine',
                'no_hardcoded_prompts': True
            })

        @self.app.route('/chat', methods=['POST'])
        def chat_endpoint():
            """
            Pure inference endpoint for chat
            Expects: messages array with system/user/assistant roles
            NO prompt modification or enhancement
            """
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'success': False, 'error': 'No JSON data provided'}), 400

                # Extract messages - they should come from edge function with proper system prompt
                messages = data.get('messages', [])
                if not messages:
                    return jsonify({'success': False, 'error': 'No messages provided'}), 400

                # Extract generation parameters
                max_tokens = data.get('max_tokens', 512)
                temperature = data.get('temperature', 0.7)
                top_p = data.get('top_p', 0.9)

                # SFW mode is opt-in; default is unrestricted
                sfw_mode = bool(data.get('sfw_mode', False))
                if sfw_mode:
                    self.stats['sfw_requests'] += 1

                # Explicit model override (default instruct)
                requested_model = str(data.get('model', 'qwen_instruct')).lower()
                use_base = requested_model == 'qwen_base'
                logger.info(f"🧩 Model selection → base={use_base} (requested_model={requested_model}), sfw_mode={sfw_mode}")
                
                # Validate messages format
                if not isinstance(messages, list):
                    return jsonify({'success': False, 'error': 'Messages must be an array'}), 400
                
                for msg in messages:
                    if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                        return jsonify({'success': False, 'error': 'Invalid message format'}), 400
                    if msg['role'] not in ['system', 'user', 'assistant']:
                        return jsonify({'success': False, 'error': f'Invalid role: {msg["role"]}'}), 400
                
                logger.info(f"💬 Chat inference request with {len(messages)} messages")
                
                # Execute pure inference
                result = self.generate_inference(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    use_base_model=use_base
                )

                # Apply SFW filtering only if explicitly requested
                if result.get('success') and sfw_mode:
                    result['response'] = self.apply_sfw_filters(result['response'])
                    result['sfw_mode'] = True
                    if 'model_info' in result:
                        result['model_info']['sfw_mode'] = True
                
                if result['success']:
                    return jsonify(result)
                else:
                    return jsonify(result), 500
                    
            except Exception as e:
                logger.error(f"❌ Chat endpoint error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'response': 'I encountered an error. Please try again.'
                }), 500

        @self.app.route('/enhance', methods=['POST'])
        def enhance_endpoint():
            """
            Pure inference endpoint for enhancement
            Expects: messages array with enhancement system prompt from edge function
            NO hardcoded enhancement logic
            """
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'success': False, 'error': 'No JSON data provided'}), 400

                # Extract messages - they should come from edge function with proper enhancement system prompt
                messages = data.get('messages', [])
                if not messages:
                    return jsonify({'success': False, 'error': 'No messages provided'}), 400

                # Extract generation parameters optimized for enhancement
                max_tokens = data.get('max_tokens', 200)  # Shorter for prompts
                temperature = data.get('temperature', 0.7)
                top_p = data.get('top_p', 0.9)

                # Optional SFW and explicit model override
                sfw_mode = bool(data.get('sfw_mode', False))
                if sfw_mode:
                    self.stats['sfw_requests'] += 1
                requested_model = str(data.get('model', 'qwen_instruct')).lower()
                use_base = requested_model == 'qwen_base'
                
                logger.info(f"🎨 Enhancement inference request with {len(messages)} messages")
                
                # Execute pure inference
                result = self.generate_inference(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    use_base_model=use_base
                )
                
                if result['success']:
                    # For enhancement, also provide original prompt tracking
                    enhanced_prompt = result['response']
                    if sfw_mode:
                        enhanced_prompt = self.apply_sfw_filters(enhanced_prompt)
                    return jsonify({
                        'success': True,
                        'enhanced_prompt': enhanced_prompt,
                        'generation_time': result['generation_time'],
                        'tokens_generated': result['tokens_generated'],
                        'model_info': {**result['model_info'], 'sfw_mode': sfw_mode}
                    })
                else:
                    return jsonify(result), 500
                    
            except Exception as e:
                logger.error(f"❌ Enhancement endpoint error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'enhanced_prompt': 'Enhancement failed. Using original prompt.'
                }), 500

        @self.app.route('/generate', methods=['POST'])
        def generate_endpoint():
            """
            Generic inference endpoint
            Accepts any valid messages array and generation parameters
            """
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'success': False, 'error': 'No JSON data provided'}), 400

                messages = data.get('messages', [])
                if not messages:
                    return jsonify({'success': False, 'error': 'No messages provided'}), 400

                # Extract all generation parameters
                max_tokens = data.get('max_tokens', 512)
                temperature = data.get('temperature', 0.7)
                top_p = data.get('top_p', 0.9)

                # Optional SFW and explicit model override
                sfw_mode = bool(data.get('sfw_mode', False))
                if sfw_mode:
                    self.stats['sfw_requests'] += 1
                requested_model = str(data.get('model', 'qwen_instruct')).lower()
                use_base = requested_model == 'qwen_base'
                
                logger.info(f"⚡ Generic inference request with {len(messages)} messages")
                
                result = self.generate_inference(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    use_base_model=use_base
                )
                
                if result.get('success') and sfw_mode:
                    result['response'] = self.apply_sfw_filters(result['response'])
                    result['sfw_mode'] = True
                    if 'model_info' in result:
                        result['model_info']['sfw_mode'] = True

                return jsonify(result) if result['success'] else (jsonify(result), 500)
                    
            except Exception as e:
                logger.error(f"❌ Generate endpoint error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'response': 'Generation failed. Please try again.'
                }), 500

        @self.app.route('/worker/info', methods=['GET'])
        def worker_info():
            """Get worker information and capabilities"""
            return jsonify({
                'worker_type': 'pure_inference_engine',
                'model': 'Qwen2.5-7B-Instruct',
                'capabilities': {
                    'chat': True,
                    'enhancement': True,
                    'generation': True,
                    'hardcoded_prompts': False,
                    'prompt_modification': False,
                    'pure_inference': True
                },
                'models_loaded': {
                    'instruct_loaded': self.qwen_instruct_model is not None,
                    'base_loaded': self.qwen_base_model is not None,
                    'active_model_type': self.active_model_type
                },
                'model_paths': {
                    'instruct': self.instruct_model_path,
                    'base': self.base_model_path
                },
                'endpoints': {
                    '/chat': 'POST - Chat inference with messages array',
                    '/enhance': 'POST - Enhancement inference with messages array',
                    '/generate': 'POST - Generic inference with messages array',
                    '/health': 'GET - Health check',
                    '/worker/info': 'GET - This information',
                    '/debug/model': 'GET - Current model/debug status'
                },
                'model_loaded': self.model_loaded,
                'stats': self.stats,
                'message_format': {
                    'required': ['messages'],
                    'optional': ['max_tokens', 'temperature', 'top_p', 'sfw_mode', 'model'],
                    'example': {
                        'messages': [
                            {'role': 'system', 'content': 'System prompt from edge function'},
                            {'role': 'user', 'content': 'User message'}
                        ],
                        'max_tokens': 512,
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'sfw_mode': False,
                        'model': 'qwen_instruct'
                    }
                }
            })

        # Debug endpoint
        @self.app.route('/debug/model', methods=['GET'])
        def debug_model():
            """Return current model selection and NSFW-related diagnostics."""
            return jsonify({
                'active_model_type': self.active_model_type,
                'instruct_loaded': self.qwen_instruct_model is not None,
                'base_loaded': self.qwen_base_model is not None,
                'paths': {
                    'instruct': self.instruct_model_path,
                    'base': self.base_model_path
                },
                'devices': {
                    'instruct': str(self.instruct_model_device) if self.qwen_instruct_model is not None else None,
                    'base': str(self.base_model_device) if self.qwen_base_model is not None else None
                },
                'stats': self.stats
            })

        # Memory management endpoints
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
                    'model_loaded': self.model_loaded,
                    'instruct_loaded': self.qwen_instruct_model is not None,
                    'base_loaded': self.qwen_base_model is not None
                }
                
                if self.qwen_instruct_model is not None and self.instruct_model_device is not None:
                    response['instruct_device'] = str(self.instruct_model_device)
                if self.qwen_base_model is not None and self.base_model_device is not None:
                    response['base_device'] = str(self.base_model_device)
                
                return jsonify(response)
            else:
                return jsonify({'error': 'CUDA not available'}), 500

        @self.app.route('/memory/unload', methods=['POST'])
        def force_unload():
            """Force unload model (for memory management)"""
            which = request.args.get('which', 'all')
            if which in ('instruct', 'all'):
                self.unload_qwen_instruct_model()
            if which in ('base', 'all'):
                self.unload_qwen_base_model()
            return jsonify({'success': True, 'message': f"Unload executed for: {which}"})

        @self.app.route('/memory/load', methods=['POST'])
        def force_load():
            """Force load model"""
            which = request.args.get('which', 'instruct')
            if which == 'base':
                success = self.load_qwen_base_model(force=True)
            else:
                success = self.load_qwen_instruct_model(force=True)
            return jsonify({'success': success, 'message': f'{which} model load attempted'})

    def start_server(self):
        """Start the Flask server"""
        try:
            # Load instruct model at startup if possible
            logger.info("🚀 Starting Chat Worker as PURE INFERENCE ENGINE...")
            self.load_qwen_instruct_model()
            
            # Start Flask server
            logger.info(f"🌐 Pure Inference Chat Worker listening on port {self.port}")
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
            "registration_method": "pure_inference_chat_worker",
            "worker_type": "pure_inference_engine",
            "capabilities": {
                "hardcoded_prompts": False,
                "prompt_modification": False,
                "pure_inference": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"📝 Registering Pure Inference Chat Worker with Supabase...")
        
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
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Pure Inference Chat Worker registered successfully!")
                print(f"🎯 URL: {worker_url}")
                return True
            else:
                print(f"❌ Registration failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
        
        return False
        
    except Exception as e:
        print(f"❌ Auto-registration error: {e}")
        return False

if __name__ == "__main__":
    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start chat worker
    worker = ChatWorker()
    
    # Start Flask server in a separate thread
    import threading
    
    def start_server_thread():
        worker.start_server()
    
    server_thread = threading.Thread(target=start_server_thread, daemon=True)
    server_thread.start()
    
    # Wait for server to be ready
    time.sleep(3)
    
    # Auto-register with Supabase
    auto_register_chat_worker()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Pure Inference Chat Worker...")
        sys.exit(0)