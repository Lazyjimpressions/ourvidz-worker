# wan_worker.py - Enhanced WAN Worker with Qwen 7B Integration
# CRITICAL FIXES: Enhancement timeout, upload validation, graceful fallback, CALLBACK FORMAT
# Date: July 5, 2025

import os
import json
import time
import torch
import requests
import subprocess
import tempfile
import signal
import mimetypes
import fcntl        # ✅ CRITICAL: Import fcntl at module level (Unix systems)
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

class TimeoutException(Exception):
    """Custom exception for timeouts"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeouts"""
    raise TimeoutException("Operation timed out")

class EnhancedWanWorker:
    def __init__(self):
        """Initialize Enhanced WAN Worker with Qwen 7B integration"""
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_code_path = "/workspace/Wan2.1"
        
        # CRITICAL: Set environment variables immediately (VERIFIED FIX)
        os.environ['PYTHONPATH'] = '/workspace/python_deps/lib/python3.11/site-packages'
        os.environ['HF_HOME'] = '/workspace/models/huggingface_cache'
        os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/models/huggingface_cache/hub'
        
        # Updated HuggingFace cache configuration
        self.hf_cache_path = "/workspace/models/huggingface_cache"
        # CORRECTED PATH: 7B model is in root cache dir, not hub/ subdirectory
        self.qwen_model_path = f"{self.hf_cache_path}/models--Qwen--Qwen2.5-7B-Instruct"
        
        # Environment configuration
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        # Model instances (loaded on demand)
        self.qwen_model = None
        self.qwen_tokenizer = None
        
        # Enhancement settings
        self.enhancement_timeout = 60  # CRITICAL: 60 second timeout for Qwen enhancement
        self.max_enhancement_attempts = 2  # Allow 2 attempts before fallback
        
        # Job type configurations - ALL 8 TYPES SUPPORTED
        # UPDATED: Based on verified WAN 2.1 testing (50 steps default, 5.0 guidance)
        self.job_configs = {
            # Standard job types (no enhancement)
            'image_fast': {
                'size': '480*832',
                'sample_steps': 25,  # Fast: Half of default 50 steps
                'sample_guide_scale': 5.0,  # Verified working default
                'frame_num': 1,
                'enhance_prompt': False,
                'expected_time': 40,  # ~38s verified for 50 steps, 25 steps should be ~20s
                'content_type': 'image'
            },
            'image_high': {
                'size': '480*832',
                'sample_steps': 50,  # High: Full 50 steps (verified working)
                'sample_guide_scale': 5.0,  # Verified working default
                'frame_num': 1,
                'enhance_prompt': False,
                'expected_time': 40,  # ~38s verified for 50 steps
                'content_type': 'image'
            },
            'video_fast': {
                'size': '480*832',
                'sample_steps': 25,  # Fast: Half of default 50 steps
                'sample_guide_scale': 5.0,  # Verified working default
                'frame_num': 17,
                'enhance_prompt': False,
                'expected_time': 35,  # Faster than 51s verified for 50 steps
                'content_type': 'video'
            },
            'video_high': {
                'size': '480*832',
                'sample_steps': 50,  # High: Full 50 steps (verified working)
                'sample_guide_scale': 5.0,  # Verified working default
                'frame_num': 17,
                'enhance_prompt': False,
                'expected_time': 55,  # ~51s verified for 50 steps
                'content_type': 'video'
            },
            
            # Enhanced job types (with Qwen 7B enhancement)
            'image7b_fast_enhanced': {
                'size': '480*832',
                'sample_steps': 25,  # Fast: Half of default 50 steps
                'sample_guide_scale': 5.0,  # Verified working default
                'frame_num': 1,
                'enhance_prompt': True,
                'expected_time': 54,  # 40s + 14s enhancement
                'content_type': 'image'
            },
            'image7b_high_enhanced': {
                'size': '480*832',
                'sample_steps': 50,  # High: Full 50 steps (verified working)
                'sample_guide_scale': 5.0,  # Verified working default
                'frame_num': 1,
                'enhance_prompt': True,
                'expected_time': 54,  # 40s + 14s enhancement
                'content_type': 'image'
            },
            'video7b_fast_enhanced': {
                'size': '480*832',
                'sample_steps': 25,  # Fast: Half of default 50 steps
                'sample_guide_scale': 5.0,  # Verified working default
                'frame_num': 17,
                'enhance_prompt': True,
                'expected_time': 49,  # 35s + 14s enhancement
                'content_type': 'video'
            },
            'video7b_high_enhanced': {
                'size': '480*832',
                'sample_steps': 50,  # High: Full 50 steps (verified working)
                'sample_guide_scale': 5.0,  # Verified working default
                'frame_num': 17,
                'enhance_prompt': True,
                'expected_time': 69,  # 55s + 14s enhancement
                'content_type': 'video'
            }
        }
        
        print("🎬 Enhanced OurVidz WAN Worker initialized")
        print(f"📋 Supporting ALL 8 job types: {list(self.job_configs.keys())}")
        print(f"📁 WAN Model Path: {self.model_path}")
        print(f"🤖 Qwen Model Path: {self.qwen_model_path}")
        print(f"💾 HF Cache: {self.hf_cache_path}")
        print(f"⏰ Enhancement Timeout: {self.enhancement_timeout}s")
        print("✨ Enhanced jobs include Qwen 7B prompt enhancement")
        print("🔧 FIXED: Upstash Redis REST API compatibility (RPOP instead of BRPOP)")
        print("🔧 FIXED: Enhancement timeout and upload validation")
        print("🔧 FIXED: Callback format for Supabase edge function compatibility")
        self.log_gpu_memory()

    def log_gpu_memory(self):
        """Monitor RTX 6000 ADA 48GB VRAM usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🔥 GPU Memory - Used: {memory_allocated:.2f}GB / {total_memory:.0f}GB")

    def setup_environment(self):
        """Configure environment variables for WAN and Qwen - VERIFIED PATHS"""
        env = os.environ.copy()
        
        # CRITICAL: Add persistent dependencies to Python path
        python_deps_path = '/workspace/python_deps/lib/python3.11/site-packages'
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            new_pythonpath = f"{python_deps_path}:{current_pythonpath}"
        else:
            new_pythonpath = python_deps_path
        
        env.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'TORCH_USE_CUDA_DSA': '1',
            'PYTHONUNBUFFERED': '1',
            'PYTHONPATH': new_pythonpath,  # VERIFIED: Dependencies exist here
            'HF_HOME': self.hf_cache_path,  # VERIFIED: /workspace/models/huggingface_cache
            'TRANSFORMERS_CACHE': self.hf_cache_path,
            'HUGGINGFACE_HUB_CACHE': f"{self.hf_cache_path}/hub"  # VERIFIED: /hub/ subdirectory
        })
        return env

    def load_qwen_model(self):
        """Load Qwen 2.5-7B model for prompt enhancement with timeout protection"""
        if self.qwen_model is None:
            print("🤖 Loading Qwen 2.5-7B for prompt enhancement...")
            enhancement_start = time.time()
            
            try:
                # Set timeout for model loading
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)  # 2 minute timeout for model loading
                
                # CRITICAL FIX: Always use model name, not local path
                # The local cached model has config issues, load from HF directly
                model_name = "Qwen/Qwen2.5-7B-Instruct"
                print(f"🔄 Loading from HuggingFace: {model_name}")
                print(f"💾 Using cache directory: {self.hf_cache_path}")
                
                # Load tokenizer first
                print("📝 Loading tokenizer...")
                self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.hf_cache_path,
                    trust_remote_code=True,
                    revision="main"  # Use main branch explicitly
                )
                
                # Load model with enhanced error handling
                print("🧠 Loading model...")
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=self.hf_cache_path,
                    trust_remote_code=True,
                    revision="main",  # Use main branch explicitly
                    low_cpu_mem_usage=True,  # Optimize memory usage
                    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
                )
                
                # Clear timeout
                signal.alarm(0)
                
                load_time = time.time() - enhancement_start
                print(f"✅ Qwen 2.5-7B loaded successfully in {load_time:.1f}s")
                self.log_gpu_memory()
                
            except TimeoutException:
                signal.alarm(0)
                print(f"❌ Qwen model loading timed out after 120s")
                print("💡 Try: Increase timeout or use smaller model")
                self.qwen_model = None
                self.qwen_tokenizer = None
            except Exception as e:
                signal.alarm(0)
                error_msg = str(e)
                print(f"❌ Failed to load Qwen model: {error_msg}")
                
                # Enhanced error diagnosis
                if "model_type" in error_msg:
                    print("💡 Model type error - trying alternative loading method...")
                    # Try loading with AutoModel instead
                    try:
                        from transformers import AutoModel
                        print("🔄 Attempting AutoModel fallback...")
                        self.qwen_model = AutoModel.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            cache_dir=self.hf_cache_path,
                            trust_remote_code=True
                        )
                        print("✅ AutoModel fallback successful")
                    except Exception as fallback_error:
                        print(f"❌ AutoModel fallback failed: {fallback_error}")
                        self.qwen_model = None
                        self.qwen_tokenizer = None
                elif "revision" in error_msg:
                    print("💡 Revision error - model may not exist or be accessible")
                elif "cache" in error_msg:
                    print("💡 Cache error - try clearing HuggingFace cache")
                
                if self.qwen_model is None:
                    print("⚠️ All loading attempts failed - enhancement will be disabled")
                    print("💡 Consider using standard jobs without enhancement")
                    # Fall back to no enhancement
                    self.qwen_model = None
                    self.qwen_tokenizer = None

    def unload_qwen_model(self):
        """Free Qwen memory for WAN generation"""
        if self.qwen_model is not None:
            print("🗑️ Unloading Qwen 2.5-7B...")
            del self.qwen_model
            del self.qwen_tokenizer
            self.qwen_model = None
            self.qwen_tokenizer = None
            torch.cuda.empty_cache()
            print("✅ Qwen 2.5-7B unloaded")
            self.log_gpu_memory()

    def enhance_prompt_with_timeout(self, original_prompt):
        """Enhanced prompt generation with strict timeout control"""
        enhancement_start = time.time()
        print(f"🤖 Enhancing prompt with {self.enhancement_timeout}s timeout: {original_prompt[:50]}...")
        
        try:
            # Set timeout signal
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.enhancement_timeout)
            
            # Load model if not already loaded
            self.load_qwen_model()
            
            if self.qwen_model is None:
                signal.alarm(0)
                print("⚠️ Qwen model not available, using original prompt")
                return original_prompt
            
            system_prompt = """你是一个专业的视频制作提示词专家。请将用户的简单描述转换为详细的视频生成提示词。

要求：
1. 保持原始含义和主要元素
2. 添加具体的视觉细节（外观、服装、环境）
3. 包含镜头运动和拍摄角度
4. 添加光影效果和氛围描述
5. 确保描述适合5秒视频生成
6. 使用中文回复，内容要专业且具有电影感

请将以下用户输入转换为专业的视频生成提示词："""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_prompt}
            ]
            
            # Apply chat template
            text = self.qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize and generate with timeout protection
            model_inputs = self.qwen_tokenizer([text], return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                generated_ids = self.qwen_model.generate(
                    **model_inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.qwen_tokenizer.eos_token_id,
                    # Additional timeout protection
                    max_time=self.enhancement_timeout - 10  # Leave 10s buffer
                )
            
            # Extract enhanced prompt
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            enhanced_prompt = self.qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clear timeout
            signal.alarm(0)
            
            enhancement_time = time.time() - enhancement_start
            print(f"✅ Prompt enhanced in {enhancement_time:.1f}s")
            print(f"📝 Enhanced: {enhanced_prompt[:100]}...")
            return enhanced_prompt.strip()
            
        except TimeoutException:
            signal.alarm(0)
            print(f"⚠️ Enhancement timed out after {self.enhancement_timeout}s, using original prompt")
            return original_prompt
        except Exception as e:
            signal.alarm(0)
            print(f"❌ Prompt enhancement failed: {e}")
            return original_prompt
        finally:
            # Always unload model to free memory
            self.unload_qwen_model()

    def enhance_prompt(self, original_prompt):
        """Enhanced prompt with retry logic and graceful fallback"""
        print(f"🤖 Starting enhancement for: {original_prompt[:50]}...")
        
        for attempt in range(self.max_enhancement_attempts):
            try:
                print(f"🔄 Enhancement attempt {attempt + 1}/{self.max_enhancement_attempts}")
                enhanced = self.enhance_prompt_with_timeout(original_prompt)
                
                # Validate enhancement worked (be more lenient with validation)
                if enhanced and enhanced.strip() != original_prompt.strip():
                    print(f"✅ Enhancement successful on attempt {attempt + 1}")
                    print(f"📊 Original: {len(original_prompt)} chars → Enhanced: {len(enhanced)} chars")
                    return enhanced
                else:
                    print(f"⚠️ Enhancement attempt {attempt + 1} returned original prompt")
                    if attempt < self.max_enhancement_attempts - 1:
                        print("⏳ Waiting 5s before retry...")
                        time.sleep(5)  # Wait before retry
                    
            except Exception as e:
                print(f"❌ Enhancement attempt {attempt + 1} failed: {e}")
                if attempt < self.max_enhancement_attempts - 1:
                    print("⏳ Waiting 5s before retry...")
                    time.sleep(5)  # Wait before retry
        
        print("⚠️ All enhancement attempts failed, using original prompt")
        print("💡 Continuing with standard WAN generation...")
        return original_prompt

    def validate_output_file(self, file_path, expected_content_type):
        """Validate that output file is correct type before upload"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ Output file does not exist: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                print(f"❌ Output file is empty: {file_path}")
                return False
            
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            print(f"🔍 File validation: {file_path}")
            print(f"📁 Size: {file_size / 1024**2:.2f}MB")
            print(f"📄 MIME type: {mime_type}")
            
            # CRITICAL FIX: Handle cases where mimetypes can't determine type
            if not mime_type:
                # Try to determine from file content/extension
                file_ext = os.path.splitext(file_path)[1].lower()
                print(f"🔍 File extension: {file_ext}")
                
                if file_ext in ['.mp4', '.avi', '.mov', '.webm']:
                    mime_type = 'video/mp4'  # Default to mp4
                    print(f"📄 Corrected MIME type based on extension: {mime_type}")
                elif file_ext in ['.png', '.jpg', '.jpeg']:
                    mime_type = 'image/png'  # Default to png
                    print(f"📄 Corrected MIME type based on extension: {mime_type}")
                else:
                    # Check file content using 'file' command
                    try:
                        import subprocess
                        result = subprocess.run(['file', '--mime-type', file_path], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            detected_mime = result.stdout.strip().split(':')[-1].strip()
                            print(f"📄 Detected MIME type via file command: {detected_mime}")
                            mime_type = detected_mime
                    except:
                        print("⚠️ Could not detect MIME type via file command")
            
            # Additional check: Read file header to verify it's not text
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(16)
                    # Check for common video/image headers
                    if header.startswith(b'\x00\x00\x00\x20ftypmp4') or header.startswith(b'\x00\x00\x00\x18ftypmp4'):
                        print("✅ File header indicates MP4 video")
                        if not mime_type or mime_type == 'text/plain':
                            mime_type = 'video/mp4'
                            print("🔧 Corrected MIME type to video/mp4 based on header")
                    elif header.startswith(b'\x89PNG'):
                        print("✅ File header indicates PNG image")
                        if not mime_type or mime_type == 'text/plain':
                            mime_type = 'image/png'
                            print("🔧 Corrected MIME type to image/png based on header")
                    elif header.startswith(b'\xFF\xD8\xFF'):
                        print("✅ File header indicates JPEG image")
                        if not mime_type or mime_type == 'text/plain':
                            mime_type = 'image/jpeg'
                            print("🔧 Corrected MIME type to image/jpeg based on header")
                    else:
                        print(f"⚠️ Unknown file header: {header[:8].hex()}")
                        # Check if it's clearly text
                        try:
                            header_text = header.decode('utf-8')
                            if any(c in header_text for c in ['\n', '\r', ' ']):
                                print("❌ File appears to be text based on header content")
                                return False
                        except:
                            pass  # Not text, continue validation
            except Exception as e:
                print(f"⚠️ Could not read file header: {e}")
            
            # Validate based on expected content type
            if expected_content_type == 'video':
                if mime_type not in ['video/mp4', 'video/webm', 'video/avi', 'video/quicktime']:
                    print(f"❌ Invalid video MIME type: {mime_type}")
                    # But if we detected video headers, allow it anyway
                    with open(file_path, 'rb') as f:
                        header = f.read(16)
                        if not (header.startswith(b'\x00\x00\x00\x20ftypmp4') or 
                               header.startswith(b'\x00\x00\x00\x18ftypmp4')):
                            return False
                        else:
                            print("✅ Video header detected, overriding MIME type check")
                            
                if file_size < 50000:  # Less than 50KB is suspicious for video
                    print(f"❌ Video file too small: {file_size} bytes")
                    return False
            elif expected_content_type == 'image':
                if mime_type not in ['image/png', 'image/jpeg', 'image/jpg']:
                    print(f"❌ Invalid image MIME type: {mime_type}")
                    # Check for image headers
                    with open(file_path, 'rb') as f:
                        header = f.read(16)
                        if not (header.startswith(b'\x89PNG') or header.startswith(b'\xFF\xD8\xFF')):
                            return False
                        else:
                            print("✅ Image header detected, overriding MIME type check")
                            
                if file_size < 5000:  # Less than 5KB is suspicious for image
                    print(f"❌ Image file too small: {file_size} bytes")
                    return False
            
            print(f"✅ File validation passed")
            return True
            
        except Exception as e:
            print(f"❌ File validation error: {e}")
            return False

    def generate_content(self, prompt, job_type):
        """Generate image or video content using WAN 2.1 with predictable file paths"""
        if job_type not in self.job_configs:
            raise Exception(f"Unsupported job type: {job_type}")
            
        config = self.job_configs[job_type]
        
        # CRITICAL FIX: Use simple, predictable file paths like manual testing
        file_ext = 'png' if config['content_type'] == 'image' else 'mp4'
        timestamp = int(time.time())
        simple_filename = f"wan_output_{timestamp}.{file_ext}"
        temp_output_path = f"/tmp/{simple_filename}"
        
        print(f"🎯 Using simple output path: {temp_output_path}")
        
        try:
            # Change to WAN code directory
            original_cwd = os.getcwd()
            os.chdir(self.wan_code_path)
            
            # Build WAN generation command (VERIFIED WORKING CONFIGURATION)
            cmd = [
                "python", "generate.py",
                "--task", "t2v-1.3B",
                "--ckpt_dir", self.model_path,
                "--offload_model", "True",  # ✅ VERIFIED WORKING (WAN default)
                "--size", config['size'],  # ✅ VERIFIED: 480*832
                "--sample_steps", str(config['sample_steps']),  # ✅ VERIFIED: 25/50 steps
                "--sample_guide_scale", str(config['sample_guide_scale']),  # ✅ VERIFIED: 5.0
                "--frame_num", str(config['frame_num']),  # ✅ VERIFIED: 1 for images, 17 for videos
                "--prompt", prompt,
                "--save_file", temp_output_path  # ✅ CRITICAL: Include file extension
            ]
            
            # Configure environment
            env = self.setup_environment()
            
            print(f"🎬 Starting WAN generation: {job_type}")
            print(f"📝 Final prompt: {prompt[:100]}...")
            print(f"🔧 Config: {config['sample_steps']} steps, {config['frame_num']} frames, {config['size']}")
            print(f"💾 Output path: {temp_output_path}")
            print(f"📁 Working directory: {self.wan_code_path}")
            print(f"🔧 Command: {' '.join(cmd)}")
            
            # ENHANCED: Add environment validation before execution
            print("🔍 Validating environment before WAN execution...")
            print(f"   PYTHONPATH: {env.get('PYTHONPATH', 'NOT SET')}")
            print(f"   HF_HOME: {env.get('HF_HOME', 'NOT SET')}")
            print(f"   CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
            
            # Test that we can write to the output path
            test_write_path = f"/tmp/test_write_{timestamp}.txt"
            try:
                with open(test_write_path, 'w') as f:
                    f.write("test")
                os.remove(test_write_path)
                print(f"✅ Output directory writable: /tmp/")
            except Exception as e:
                print(f"❌ Cannot write to /tmp/: {e}")
                raise Exception(f"Output directory not writable: {e}")
            
            # ENHANCED: Execute WAN generation with REAL-TIME OUTPUT and 350s timeout
            generation_start = time.time()
            print(f"⏰ Starting WAN subprocess with 350s timeout at {time.strftime('%H:%M:%S')}")
            
            # Use Popen for real-time output instead of subprocess.run
            process = subprocess.Popen(
                cmd,
                cwd=self.wan_code_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Real-time output monitoring with timeout
            output_lines = []
            max_output_lines = 100  # Limit stored lines to prevent memory issues
            start_time = time.time()
            timeout = 350  # 350 seconds as requested
            last_progress_time = start_time
            
            # Enhanced WAN error patterns
            wan_error_patterns = [
                'error', 'failed', 'exception', 'traceback', 'cuda error',
                'out of memory', 'no module named', 'import error', 
                'file not found', 'permission denied', 'cuda out of memory',
                'runtime error', 'assertion error', 'key error', 'value error'
            ]
            
            wan_progress_patterns = [
                'generated', 'saved', 'loading', 'processing', 'step', 
                'progress', 'completed', 'finished', 'done', 'success',
                'creating', 'writing', 'encoding', 'frame'
            ]
            
            print("📺 WAN Real-time Output:")
            print("=" * 50)
            
            while True:
                # Check if process has finished
                if process.poll() is not None:
                    break
                    
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print(f"⏰ TIMEOUT: WAN generation exceeded {timeout}s, killing process...")
                    process.terminate()
                    try:
                        process.wait(timeout=10)  # Give it 10s to terminate gracefully
                    except subprocess.TimeoutExpired:
                        print("🔥 Force killing WAN process...")
                        process.kill()
                        process.wait()
                    raise subprocess.TimeoutExpired(cmd, timeout, None)
                
                # Cross-platform output reading - FIXED VERSION
                try:
                    line = None
                    
                    # Check if we're on Unix-like system and fcntl is available
                    if hasattr(fcntl, 'F_GETFL') and hasattr(os, 'O_NONBLOCK'):
                        try:
                            # Make stdout non-blocking (Unix only)
                            fd = process.stdout.fileno()
                            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                            line = process.stdout.readline()
                        except (IOError, OSError):
                            # No data available or fcntl failed, continue
                            pass
                    else:
                        # Fallback for Windows or systems without fcntl/O_NONBLOCK
                        try:
                            line = process.stdout.readline()
                        except:
                            pass
                    
                    if line and line.strip():
                        line = line.strip()
                        
                        # Buffer management - keep only last N lines
                        output_lines.append(line)
                        if len(output_lines) > max_output_lines:
                            output_lines.pop(0)  # Remove oldest line
                        
                        print(f"[WAN] {line}")
                        
                        # Enhanced error pattern detection
                        line_lower = line.lower()
                        for error_pattern in wan_error_patterns:
                            if error_pattern in line_lower:
                                print(f"🚨 WAN ERROR DETECTED: {line}")
                                break
                        
                        # Enhanced progress pattern detection
                        for progress_pattern in wan_progress_patterns:
                            if progress_pattern in line_lower:
                                print(f"✅ WAN PROGRESS: {line}")
                                break
                    else:
                        # No output, small delay to prevent busy waiting
                        time.sleep(0.1)
                            
                except Exception as e:
                    print(f"⚠️ Error reading WAN output: {e}")
                    time.sleep(1)  # Small delay before retry
                    continue
                
                # Progress indicator every 30 seconds
                if elapsed - last_progress_time >= 30:
                    print(f"⏱️ WAN generation progress: {elapsed:.0f}s / {timeout}s")
                    last_progress_time = elapsed
            
            # Get final return code
            return_code = process.returncode
            generation_time = time.time() - generation_start
            
            print("=" * 50)
            print(f"✅ WAN subprocess completed in {generation_time:.1f}s")
            print(f"📄 WAN return code: {return_code}")
            print(f"📝 Total output lines: {len(output_lines)}")
            
            # Restore original directory
            os.chdir(original_cwd)
            
            if return_code == 0:
                # Check if output file exists and has content
                if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                    file_size = os.path.getsize(temp_output_path) / 1024**2  # MB
                    print(f"✅ WAN generation successful: {file_size:.1f}MB file created")
                    print(f"📁 Output file: {temp_output_path}")
                    
                    # Enhanced file validation
                    if self.validate_output_file(temp_output_path, config['content_type']):
                        print(f"✅ Output file validation passed")
                        return temp_output_path
                    else:
                        print(f"❌ Output file validation failed")
                        raise Exception("Generated file failed validation")
                else:
                    print(f"❌ WAN completed but no output file created at: {temp_output_path}")
                    # Try to find files that WAN might have created
                    import glob
                    possible_files = glob.glob(f"/tmp/wan_output_{timestamp}*")
                    if possible_files:
                        print(f"📁 Found possible output files: {possible_files}")
                        actual_file = possible_files[0]
                        if os.path.getsize(actual_file) > 0:
                            print(f"✅ Using found file: {actual_file}")
                            return actual_file
                    raise Exception("WAN generation completed but no valid output file found")
            else:
                # Process failed
                error_output = '\n'.join(output_lines[-10:]) if output_lines else "No output captured"
                print(f"❌ WAN generation failed with return code {return_code}")
                print(f"📄 Last 10 lines of output:")
                for line in output_lines[-10:]:
                    print(f"   {line}")
                raise Exception(f"WAN generation failed with return code {return_code}: {error_output}")
                
        except subprocess.TimeoutExpired:
            os.chdir(original_cwd)
            print(f"❌ WAN generation timed out after {timeout}s")
            print(f"📄 Output captured before timeout:")
            for line in output_lines[-5:] if 'output_lines' in locals() else []:
                print(f"   {line}")
            # Cleanup any partial files
            import glob
            for partial_file in glob.glob(f"/tmp/wan_output_{timestamp}*"):
                try:
                    os.unlink(partial_file)
                    print(f"🗑️ Cleaned up partial file: {partial_file}")
                except:
                    pass
            raise Exception(f"WAN generation timed out after {timeout} seconds")
        except Exception as e:
            os.chdir(original_cwd)  # Ensure we restore directory
            print(f"❌ WAN generation error: {e}")
            print(f"📄 Output captured before error:")
            for line in output_lines[-5:] if 'output_lines' in locals() else []:
                print(f"   {line}")
            # Cleanup any partial files
            import glob
            for partial_file in glob.glob(f"/tmp/wan_output_{timestamp}*"):
                try:
                    os.unlink(partial_file)
                except:
                    pass
            raise

    def upload_to_supabase(self, file_path, storage_path):
        """Upload file to Supabase storage with enhanced validation"""
        try:
            # Double-check file before upload
            if not os.path.exists(file_path):
                raise Exception(f"File does not exist: {file_path}")
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise Exception(f"File is empty: {file_path}")
            
            # Check MIME type for additional safety
            mime_type, _ = mimetypes.guess_type(file_path)
            print(f"📤 Uploading {mime_type} file ({file_size / 1024**2:.2f}MB)")
            
            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': file},
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                    },
                    timeout=120  # 2 minute upload timeout
                )
            
            if response.status_code == 200:
                # Return only relative path within bucket (avoid double-prefixing)
                path_parts = storage_path.split('/', 1)
                relative_path = path_parts[1] if len(path_parts) == 2 else storage_path
                print(f"✅ Upload successful: {relative_path}")
                return relative_path
            else:
                error_text = response.text[:500]  # First 500 chars of error
                print(f"❌ Upload failed: {response.status_code}")
                print(f"📄 Error response: {error_text}")
                raise Exception(f"Upload failed: {response.status_code} - {error_text}")
                
        except Exception as e:
            print(f"❌ Supabase upload error: {e}")
            raise

    def notify_completion(self, job_id, status, output_url=None, error_message=None):
        """Notify Supabase of job completion with FIXED callback format"""
        try:
            # CRITICAL FIX: Use the correct callback format expected by edge function
            callback_data = {
                'jobId': job_id,
                'status': status,
                'outputUrl': output_url,
                'errorMessage': error_message
            }
            
            print(f"📞 Sending callback for job {job_id}:")
            print(f"   Status: {status}")
            print(f"   Output URL: {output_url}")
            print(f"   Error: {error_message}")
            
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
                print(f"✅ Job {job_id} callback sent successfully")
            else:
                print(f"❌ Callback failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def test_wan_dependencies(self):
        """Test if WAN dependencies are accessible before job processing"""
        print("🔍 Testing WAN dependencies accessibility...")
        try:
            import sys
            print(f"📍 Current Python path: {sys.path}")
            test_imports = [
                ('easydict', 'easydict'),
                ('omegaconf', 'omegaconf'),
                ('einops', 'einops'),
                ('diffusers', 'diffusers'),
                ('transformers', 'transformers'),
                ('flash_attn', 'flash_attn'),
                ('wan', 'wan')
            ]
            for module_name, import_name in test_imports:
                try:
                    __import__(import_name)
                    print(f"✅ {module_name}: Available")
                except ImportError as e:
                    print(f"❌ {module_name}: MISSING - {e}")
            wan_generate_path = os.path.join(self.wan_code_path, 'generate.py')
            if os.path.exists(wan_generate_path):
                print(f"✅ WAN generate.py found: {wan_generate_path}")
            else:
                print(f"❌ WAN generate.py NOT FOUND: {wan_generate_path}")
            if os.path.exists(self.model_path):
                model_files = os.listdir(self.model_path)
                print(f"✅ WAN model directory accessible: {len(model_files)} files")
                print(f"📁 Model files: {model_files[:5]}...")
            else:
                print(f"❌ WAN model directory NOT FOUND: {self.model_path}")
        except Exception as e:
            print(f"❌ Dependency test failed: {e}")

    def test_wan_basic_execution(self):
        """Test basic WAN execution before processing jobs"""
        print("🧪 Testing basic WAN execution...")
        try:
            original_cwd = os.getcwd()
            os.chdir(self.wan_code_path)
            env = self.setup_environment()
            test_cmd = ["python", "generate.py", "--help"]
            print(f"🔧 Testing command: {' '.join(test_cmd)}")
            result = subprocess.run(
                test_cmd,
                cwd=self.wan_code_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=30
            )
            os.chdir(original_cwd)
            if result.returncode == 0:
                print("✅ WAN help command successful")
                print(f"📄 Output preview: {result.stdout[:200]}...")
            else:
                print(f"❌ WAN help command failed: {result.returncode}")
                print(f"📄 stderr: {result.stderr}")
                print(f"📄 stdout: {result.stdout}")
        except Exception as e:
            os.chdir(original_cwd)
            print(f"❌ WAN basic execution test failed: {e}")

    def enhanced_environment_setup(self):
        """Enhanced environment setup with validation"""
        env = self.setup_environment()
        print("🔍 ENHANCED Environment Validation:")
        print(f"   Working Directory: {os.getcwd()}")
        print(f"   WAN Code Path: {self.wan_code_path}")
        print(f"   Model Path: {self.model_path}")
        critical_env_vars = [
            'PYTHONPATH',
            'HF_HOME',
            'HUGGINGFACE_HUB_CACHE',
            'CUDA_VISIBLE_DEVICES'
        ]
        for var in critical_env_vars:
            value = env.get(var, 'NOT SET')
            print(f"   {var}: {value}")
            if var in ['HF_HOME', 'HUGGINGFACE_HUB_CACHE'] and value != 'NOT SET':
                exists = os.path.exists(value)
                print(f"     -> Path exists: {exists}")
            elif var == 'PYTHONPATH' and value != 'NOT SET':
                paths = value.split(':')
                for path in paths:
                    exists = os.path.exists(path)
                    print(f"     -> {path}: {exists}")
        return env

    def process_job_with_enhanced_diagnostics(self, job_data):
        """Enhanced process_job with additional diagnostics"""
        job_id = job_data['jobId']
        job_type = job_data['jobType']
        original_prompt = job_data['prompt']
        video_id = job_data['videoId']
        print(f"🔄 Processing job {job_id} ({job_type}) with enhanced diagnostics")
        print(f"📝 Original prompt: {original_prompt}")
        print(f"🎯 Video ID: {video_id}")
        print("\n🔍 PRE-JOB DIAGNOSTICS:")
        self.test_wan_dependencies()
        print("\n🧪 WAN EXECUTION TEST:")
        self.test_wan_basic_execution()
        print("\n" + "="*60)
        job_start_time = time.time()
        try:
            if job_type not in self.job_configs:
                available_types = list(self.job_configs.keys())
                raise Exception(f"Unknown job type: {job_type}. Available: {available_types}")
            config = self.job_configs[job_type]
            print(f"✅ Job type validated: {job_type} (enhance: {config['enhance_prompt']})")
            if config['enhance_prompt']:
                print("🤖 Starting prompt enhancement with timeout protection...")
                enhanced_prompt = self.enhance_prompt(original_prompt)
                actual_prompt = enhanced_prompt
                if enhanced_prompt != original_prompt:
                    print(f"✅ Prompt successfully enhanced")
                    print(f"📝 Length: {len(original_prompt)} → {len(enhanced_prompt)} chars")
                else:
                    print(f"⚠️ Using original prompt (enhancement failed or timed out)")
            else:
                print("📝 Using original prompt (no enhancement)")
                actual_prompt = original_prompt
            print("🎬 Starting WAN generation with enhanced diagnostics...")
            print(f"🔍 About to call generate_content with:")
            print(f"   Prompt: {actual_prompt[:100]}...")
            print(f"   Job type: {job_type}")
            print(f"   Config: {config}")
            print("\n🔍 FINAL ENVIRONMENT CHECK BEFORE WAN:")
            test_env = self.enhanced_environment_setup()
            output_file = self.generate_content(actual_prompt, job_type)
            if not output_file:
                raise Exception("Content generation failed or produced invalid output")
            file_extension = 'png' if config['content_type'] == 'image' else 'mp4'
            storage_path = f"{job_type}/{video_id}.{file_extension}"
            print(f"📤 Uploading validated file to: {storage_path}")
            relative_path = self.upload_to_supabase(output_file, storage_path)
            os.unlink(output_file)
            self.notify_completion(job_id, 'completed', relative_path)
            total_time = time.time() - job_start_time
            print(f"🎉 Job {job_id} completed successfully in {total_time:.1f}s")
            print(f"📁 Output: {relative_path}")
        except Exception as e:
            error_msg = str(e)
            total_time = time.time() - job_start_time
            print(f"❌ Job {job_id} failed after {total_time:.1f}s: {error_msg}")
            if "timeout" in error_msg.lower():
                print("💡 Timeout detected - WAN subprocess hanging")
            elif "mime" in error_msg.lower() or "validation" in error_msg.lower():
                print("💡 File validation failed - WAN generation produced invalid output")
            elif "upload" in error_msg.lower():
                print("💡 Upload failed - check storage bucket configuration")
            elif "import" in error_msg.lower() or "module" in error_msg.lower():
                print("💡 Import error - WAN dependencies not accessible")
            self.notify_completion(job_id, 'failed', error_message=error_msg)

    def run_with_enhanced_diagnostics(self):
        """Main worker loop with startup diagnostics"""
        print("🎬 Enhanced OurVidz WAN Worker with ENHANCED DIAGNOSTICS started!")
        print("\n🔍 STARTUP DIAGNOSTICS:")
        print("="*60)
        self.test_wan_dependencies()
        print("\n🧪 STARTUP WAN EXECUTION TEST:")
        self.test_wan_basic_execution()
        print("="*60)
        print("🔧 UPSTASH COMPATIBLE: Using non-blocking RPOP for Redis polling")
        print("🔧 ENHANCED FEATURES: Timeout protection, upload validation, graceful fallback")
        print("🔧 CALLBACK FORMAT: Fixed for Supabase edge function compatibility")
        print("📋 Supported job types:")
        for job_type, config in self.job_configs.items():
            enhancement = "✨ Enhanced" if config['enhance_prompt'] else "📝 Standard"
            content = "🖼️ Image" if config['content_type'] == 'image' else "🎬 Video"
            print(f"  • {job_type}: {content} ({config['expected_time']}s) {enhancement}")
        print("⏳ Waiting for jobs...")
        job_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        while True:
            try:
                job_data = self.poll_queue()
                if job_data:
                    job_count += 1
                    consecutive_errors = 0
                    print(f"\n📬 WAN Job #{job_count} received")
                    self.process_job_with_enhanced_diagnostics(job_data)
                    print("=" * 60)
                else:
                    time.sleep(5)
            except KeyboardInterrupt:
                print("🛑 Worker stopped by user")
                break
            except Exception as e:
                consecutive_errors += 1
                print(f"❌ Worker error #{consecutive_errors}: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    print(f"❌ Too many consecutive errors ({consecutive_errors}), shutting down worker")
                    break
                sleep_time = min(30, 5 * consecutive_errors)
                print(f"⏳ Waiting {sleep_time}s before retry...")
                time.sleep(sleep_time)

    def poll_queue(self):
        """Poll Redis queue for new jobs with non-blocking RPOP (Upstash REST API compatible)"""
        try:
            # CRITICAL FIX: Use non-blocking RPOP instead of BRPOP 
            # Upstash Redis REST API doesn't support blocking commands like BRPOP
            response = requests.get(
                f"{self.redis_url}/rpop/wan_queue",  # Changed from brpop to rpop
                headers={
                    'Authorization': f"Bearer {self.redis_token}"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('result'):
                    # RPOP returns job data directly (not array like BRPOP)
                    job_json = result['result']
                    job_data = json.loads(job_json)
                    return job_data
            
            return None
            
        except requests.exceptions.Timeout:
            # Normal timeout, not an error
            return None
        except Exception as e:
            print(f"❌ Queue polling error: {e}")
            return None

if __name__ == "__main__":
    # Environment variable validation
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_SERVICE_KEY', 
        'UPSTASH_REDIS_REST_URL',
        'UPSTASH_REDIS_REST_TOKEN'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    # Verify critical paths
    model_path = "/workspace/models/wan2.1-t2v-1.3b"
    qwen_path = "/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-14B-Instruct"  # Updated to 14B
    wan_code_path = "/workspace/Wan2.1"
    
    if not os.path.exists(model_path):
        print(f"❌ WAN model not found: {model_path}")
        exit(1)
        
    if not os.path.exists(qwen_path):
        print(f"⚠️ Qwen model not found: {qwen_path} (enhancement will be disabled)")
        
    if not os.path.exists(wan_code_path):
        print(f"❌ WAN code not found: {wan_code_path}")
        exit(1)
    
    print("✅ All paths validated, starting worker...")
    
    try:
        worker = EnhancedWanWorker()
        worker.run_with_enhanced_diagnostics()
    except Exception as e:
        print(f"❌ Worker startup failed: {e}")
        exit(1)
    finally:
        print("👋 Enhanced WAN Worker shutdown complete")