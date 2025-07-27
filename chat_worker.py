#!/usr/bin/env python3
"""
OurVidz Chat Worker - Dedicated Qwen Instruct Service
Handles: Manual prompt enhancement, chat interface, admin utilities
Model: Qwen 2.5-7B Instruct (always loaded when possible)
Port: 7861
"""

import os
import sys
import time
import torch
import psutil
import signal
import threading
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
        """Initialize Chat Worker with smart memory management"""
        self.app = Flask(__name__)
        self.port = 7861
        self.model_loaded = False
        self.loading_lock = threading.Lock()
        
        # Model paths - using verified working paths
        self.instruct_model_path = "/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct"
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
        
        logger.info("🤖 Chat Worker initialized")

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

    def enhance_prompt(self, original_prompt, enhancement_type="manual"):
        """Enhanced prompt generation using Instruct model"""
        if not self.model_loaded:
            # Try to load model
            if not self.load_qwen_instruct_model():
                return {
                    'success': False,
                    'error': 'Model not available',
                    'enhanced_prompt': original_prompt
                }

        try:
            logger.info(f"🤖 Enhancing prompt ({enhancement_type}): {original_prompt[:50]}...")
            start_time = time.time()

            # Build conversation for instruct model
            system_prompt = """You are an expert AI prompt engineer specializing in cinematic and adult content generation.

Transform simple prompts into detailed, cinematic descriptions while maintaining anatomical accuracy and realism.

Focus on:
- High-quality visual details and realistic proportions
- Cinematic lighting and professional photography style  
- Specific poses, expressions, and scene composition
- Technical quality like 4K resolution and smooth motion

Respond with enhanced prompts that are detailed, specific, and optimized for AI generation."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Enhance this prompt for AI generation: {original_prompt}"}
            ]

            # Prepare chat input
            text = self.qwen_instruct_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = self.qwen_instruct_tokenizer([text], return_tensors="pt")
            
            # Move to device
            if hasattr(self.qwen_instruct_model, 'device'):
                inputs = {k: v.to(self.qwen_instruct_model.device) for k, v in inputs.items()}

            # Generate enhanced prompt
            with torch.no_grad():
                generated_ids = self.qwen_instruct_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.qwen_instruct_tokenizer.eos_token_id,
                    early_stopping=True
                )

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
                'enhancement_type': enhancement_type
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
            """Manual prompt enhancement endpoint"""
            try:
                data = request.get_json()
                if not data or 'prompt' not in data:
                    return jsonify({'success': False, 'error': 'Missing prompt'}), 400

                prompt = data['prompt']
                enhancement_type = data.get('enhancement_type', 'manual')
                
                result = self.enhance_prompt(prompt, enhancement_type)
                
                if result['success']:
                    return jsonify(result)
                else:
                    return jsonify(result), 500

            except Exception as e:
                logger.error(f"❌ Enhancement endpoint error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

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
                
                return jsonify({
                    'total_vram': total,
                    'allocated_vram': allocated,
                    'available_vram': available,
                    'model_loaded': self.model_loaded
                })
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
    worker.start_server()