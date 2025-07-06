# wan_worker.py - CRITICAL FIX for WAN Video Generation
# FIXES: WAN generating text files instead of videos, MIME type errors, command formatting
# Date: July 6, 2025

import os
import json
import time
import torch
import requests
import subprocess
import tempfile
import signal
import mimetypes
import fcntl
import glob
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
        self.enhancement_timeout = 60
        self.max_enhancement_attempts = 2
        
        # CRITICAL FIX: Updated job configurations based on manual testing results
        # Manual test showed: 480*832 size, proper file extensions (.mp4/.png), guidance 5.0
        self.job_configs = {
            # Standard job types (no enhancement)
            'image_fast': {
                'size': '480*832',           # ✅ VERIFIED working size
                'sample_steps': 25,          # Fast: 25 steps
                'sample_guide_scale': 5.0,   # ✅ VERIFIED working guidance
                'frame_num': 1,              # Single frame for images
                'enhance_prompt': False,
                'expected_time': 25,         # Estimated time
                'content_type': 'image',
                'file_extension': 'png'      # ✅ CRITICAL: Explicit extension
            },
            'image_high': {
                'size': '480*832',
                'sample_steps': 50,          # High quality: 50 steps
                'sample_guide_scale': 5.0,
                'frame_num': 1,
                'enhance_prompt': False,
                'expected_time': 40,
                'content_type': 'image',
                'file_extension': 'png'
            },
            'video_fast': {
                'size': '480*832',           # ✅ VERIFIED working size
                'sample_steps': 25,          # Fast: 25 steps
                'sample_guide_scale': 5.0,   # ✅ VERIFIED working guidance
                'frame_num': 17,             # ✅ VERIFIED: 17 frames for videos
                'enhance_prompt': False,
                'expected_time': 35,
                'content_type': 'video',
                'file_extension': 'mp4'      # ✅ CRITICAL: Explicit extension
            },
            'video_high': {
                'size': '480*832',
                'sample_steps': 50,          # High quality: 50 steps
                'sample_guide_scale': 5.0,
                'frame_num': 17,
                'enhance_prompt': False,
                'expected_time': 55,
                'content_type': 'video',
                'file_extension': 'mp4'
            },
            
            # Enhanced job types (with Qwen 7B enhancement)
            'image7b_fast_enhanced': {
                'size': '480*832',
                'sample_steps': 25,
                'sample_guide_scale': 5.0,
                'frame_num': 1,
                'enhance_prompt': True,
                'expected_time': 85,         # 25s + 60s enhancement
                'content_type': 'image',
                'file_extension': 'png'
            },
            'image7b_high_enhanced': {
                'size': '480*832',
                'sample_steps': 50,
                'sample_guide_scale': 5.0,
                'frame_num': 1,
                'enhance_prompt': True,
                'expected_time': 100,        # 40s + 60s enhancement
                'content_type': 'image',
                'file_extension': 'png'
            },
            'video7b_fast_enhanced': {
                'size': '480*832',
                'sample_steps': 25,
                'sample_guide_scale': 5.0,
                'frame_num': 17,
                'enhance_prompt': True,
                'expected_time': 95,         # 35s + 60s enhancement
                'content_type': 'video',
                'file_extension': 'mp4'
            },
            'video7b_high_enhanced': {
                'size': '480*832',
                'sample_steps': 50,
                'sample_guide_scale': 5.0,
                'frame_num': 17,
                'enhance_prompt': True,
                'expected_time': 115,        # 55s + 60s enhancement
                'content_type': 'video',
                'file_extension': 'mp4'
            }
        }
        
        print("🎬 Enhanced OurVidz WAN Worker initialized")
        print(f"📋 Supporting ALL 8 job types: {list(self.job_configs.keys())}")
        print(f"📁 WAN Model Path: {self.model_path}")
        print(f"🤖 Qwen Model Path: {self.qwen_model_path}")
        print("🔧 CRITICAL FIX: Proper file extensions and WAN command formatting")
        print("🔧 CRITICAL FIX: Enhanced output file validation")
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
            'PYTHONPATH': new_pythonpath,
            'HF_HOME': self.hf_cache_path,
            'TRANSFORMERS_CACHE': self.hf_cache_path,
            'HUGGINGFACE_HUB_CACHE': f"{self.hf_cache_path}/hub"
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
                
                model_name = "Qwen/Qwen2.5-7B-Instruct"
                print(f"🔄 Loading from HuggingFace: {model_name}")
                
                # Load tokenizer first
                print("📝 Loading tokenizer...")
                self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.hf_cache_path,
                    trust_remote_code=True,
                    revision="main"
                )
                
                # Load model
                print("🧠 Loading model...")
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=self.hf_cache_path,
                    trust_remote_code=True,
                    revision="main",
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
                )
                
                signal.alarm(0)
                
                load_time = time.time() - enhancement_start
                print(f"✅ Qwen 2.5-7B loaded successfully in {load_time:.1f}s")
                self.log_gpu_memory()
                
            except TimeoutException:
                signal.alarm(0)
                print(f"❌ Qwen model loading timed out after 120s")
                self.qwen_model = None
                self.qwen_tokenizer = None
            except Exception as e:
                signal.alarm(0)
                print(f"❌ Failed to load Qwen model: {e}")
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
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.enhancement_timeout)
            
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
            
            text = self.qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.qwen_tokenizer([text], return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                generated_ids = self.qwen_model.generate(
                    **model_inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.qwen_tokenizer.eos_token_id,
                    max_time=self.enhancement_timeout - 10
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            enhanced_prompt = self.qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            signal.alarm(0)
            
            enhancement_time = time.time() - enhancement_start
            print(f"✅ Prompt enhanced in {enhancement_time:.1f}s")
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
            self.unload_qwen_model()

    def enhance_prompt(self, original_prompt):
        """Enhanced prompt with retry logic and graceful fallback"""
        print(f"🤖 Starting enhancement for: {original_prompt[:50]}...")
        
        for attempt in range(self.max_enhancement_attempts):
            try:
                print(f"🔄 Enhancement attempt {attempt + 1}/{self.max_enhancement_attempts}")
                enhanced = self.enhance_prompt_with_timeout(original_prompt)
                
                if enhanced and enhanced.strip() != original_prompt.strip():
                    print(f"✅ Enhancement successful on attempt {attempt + 1}")
                    return enhanced
                else:
                    print(f"⚠️ Enhancement attempt {attempt + 1} returned original prompt")
                    if attempt < self.max_enhancement_attempts - 1:
                        time.sleep(5)
                    
            except Exception as e:
                print(f"❌ Enhancement attempt {attempt + 1} failed: {e}")
                if attempt < self.max_enhancement_attempts - 1:
                    time.sleep(5)
        
        print("⚠️ All enhancement attempts failed, using original prompt")
        return original_prompt

    def validate_output_file(self, file_path, expected_content_type):
        """Enhanced file validation with MIME type checking"""
        try:
            print(f"🔍 ENHANCED FILE VALIDATION:")
            print(f"   File path: {file_path}")
            print(f"   Expected type: {expected_content_type}")
            
            # Check 1: File exists
            if not os.path.exists(file_path):
                print(f"❌ File does not exist: {file_path}")
                return False, "File does not exist"
            
            # Check 2: File size
            file_size = os.path.getsize(file_path)
            print(f"📁 File size: {file_size / 1024**2:.2f}MB ({file_size} bytes)")
            
            if file_size == 0:
                print(f"❌ File is empty")
                return False, "File is empty"
            
            # Check 3: MIME type detection
            mime_type, _ = mimetypes.guess_type(file_path)
            print(f"🔍 Detected MIME type: {mime_type}")
            
            # Check 4: Read file header for validation
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(16)
                    print(f"🔍 File header (first 16 bytes): {header.hex()}")
                    
                    # Check if it's actually a text file (common WAN error)
                    if header.startswith(b'Traceback') or header.startswith(b'Error') or header.startswith(b'usage:'):
                        print(f"❌ File contains error/help text, not {expected_content_type}")
                        return False, f"File contains text data, not {expected_content_type}"
                    
                    # Check for proper file format headers
                    if expected_content_type == 'video':
                        # MP4 file should start with ftyp box
                        if not (b'ftyp' in header or b'mdat' in header):
                            print(f"❌ File doesn't have MP4 header signature")
                            return False, "File is not a valid MP4 video"
                    elif expected_content_type == 'image':
                        # PNG should start with PNG signature
                        png_signature = b'\x89PNG\r\n\x1a\n'
                        if not header.startswith(png_signature):
                            print(f"❌ File doesn't have PNG header signature")
                            return False, "File is not a valid PNG image"
            except Exception as e:
                print(f"⚠️ Could not read file header: {e}")
            
            # Check 5: Minimum size requirements
            min_size = 50000 if expected_content_type == 'video' else 5000  # 50KB for video, 5KB for image
            if file_size < min_size:
                print(f"❌ File too small for {expected_content_type}: {file_size} bytes < {min_size} bytes")
                return False, f"File too small for {expected_content_type}"
            
            # Check 6: MIME type validation
            expected_mime = 'video/mp4' if expected_content_type == 'video' else 'image/png'
            if mime_type and mime_type != expected_mime:
                print(f"⚠️ MIME type mismatch: expected {expected_mime}, got {mime_type}")
                # Don't fail on MIME type alone, as it might be detected incorrectly
            
            print(f"✅ ENHANCED VALIDATION PASSED")
            return True, "File validation successful"
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False, f"Validation error: {e}"

    def generate_content(self, prompt, job_type):
        """CRITICAL FIX: Generate content with proper WAN command formatting"""
        if job_type not in self.job_configs:
            raise Exception(f"Unsupported job type: {job_type}")
            
        config = self.job_configs[job_type]
        
        # CRITICAL FIX: Create output path with proper extension
        timestamp = int(time.time())
        file_extension = config['file_extension']  # Use explicit extension from config
        output_filename = f"wan_output_{timestamp}.{file_extension}"
        temp_output_path = f"/tmp/{output_filename}"
        
        print(f"🎯 FIXED: Output path with proper extension: {temp_output_path}")
        print(f"📄 Expected file type: {config['content_type']} (.{file_extension})")
        
        try:
            # Change to WAN code directory
            original_cwd = os.getcwd()
            os.chdir(self.wan_code_path)
            
            # CRITICAL FIX: Build WAN command with proper argument formatting
            # Based on manual testing: proper size format, guidance scale, etc.
            cmd = [
                "python", "generate.py",
                "--task", "t2v-1.3B",                           # ✅ VERIFIED working task
                "--ckpt_dir", self.model_path,                  # ✅ Model path
                "--offload_model", "True",                      # ✅ VERIFIED: Memory management
                "--size", config['size'],                       # ✅ VERIFIED: 480*832
                "--sample_steps", str(config['sample_steps']),  # ✅ Steps: 25 or 50
                "--sample_guide_scale", str(config['sample_guide_scale']),  # ✅ VERIFIED: 5.0
                "--frame_num", str(config['frame_num']),        # ✅ VERIFIED: 1 or 17
                "--prompt", prompt,                             # User prompt
                "--save_file", temp_output_path                 # ✅ CRITICAL: Full path with extension
            ]
            
            # Configure environment
            env = self.setup_environment()
            
            print(f"🎬 FIXED WAN generation: {job_type}")
            print(f"📝 Prompt: {prompt[:100]}...")
            print(f"🔧 Config: {config['sample_steps']} steps, {config['frame_num']} frames, {config['size']}")
            print(f"💾 Output: {temp_output_path}")
            print(f"📁 Working dir: {self.wan_code_path}")
            print(f"🔧 FIXED Command: {' '.join(cmd)}")
            
            # Environment validation
            print("🔍 Environment validation:")
            print(f"   PYTHONPATH: {env.get('PYTHONPATH', 'NOT SET')}")
            print(f"   CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
            print(f"   Output dir writable: {os.access('/tmp/', os.W_OK)}")
            
            # Execute WAN generation with enhanced monitoring
            generation_start = time.time()
            timeout_seconds = 400  # Extended timeout for video generation
            
            print(f"⏰ Starting WAN subprocess with {timeout_seconds}s timeout")
            print(f"🚀 Generation started at {time.strftime('%H:%M:%S')}")

            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.wan_code_path,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds
                )
                
                generation_time = time.time() - generation_start
                os.chdir(original_cwd)  # Restore directory
                
                print(f"✅ WAN subprocess completed in {generation_time:.1f}s")
                print(f"📄 Return code: {result.returncode}")
                
                # Enhanced output analysis
                if result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')
                    print(f"📄 STDOUT ({len(stdout_lines)} lines):")
                    for line in stdout_lines[-10:]:  # Last 10 lines
                        print(f"   [OUT] {line}")
                
                if result.stderr:
                    stderr_lines = result.stderr.strip().split('\n')
                    print(f"📄 STDERR ({len(stderr_lines)} lines):")
                    for line in stderr_lines[-10:]:  # Last 10 lines
                        print(f"   [ERR] {line}")
                
                # CRITICAL: Enhanced success validation
                if result.returncode == 0:
                    print(f"🔍 Checking output file: {temp_output_path}")
                    
                    # Check if exact file exists
                    if os.path.exists(temp_output_path):
                        file_size = os.path.getsize(temp_output_path)
                        print(f"✅ Output file found: {file_size / 1024**2:.2f}MB")
                        
                        # Enhanced file validation
                        is_valid, validation_msg = self.validate_output_file(temp_output_path, config['content_type'])
                        if is_valid:
                            print(f"✅ File validation passed: {validation_msg}")
                            return temp_output_path
                        else:
                            print(f"❌ File validation failed: {validation_msg}")
                            
                            # Show file content for debugging if it's small (likely error text)
                            if file_size < 10000:  # Less than 10KB
                                try:
                                    with open(temp_output_path, 'r', errors='ignore') as f:
                                        content = f.read(500)  # First 500 chars
                                        print(f"📄 File content preview: {content}")
                                except:
                                    pass
                            
                            raise Exception(f"Generated file validation failed: {validation_msg}")
                    else:
                        print(f"❌ Output file not found: {temp_output_path}")
                        
                        # Look for any files created in /tmp/
                        tmp_files = glob.glob("/tmp/wan_output_*")
                        print(f"📁 Files in /tmp/: {tmp_files}")
                        
                        if tmp_files:
                            # Try to use the most recent file
                            latest_file = max(tmp_files, key=os.path.getctime)
                            print(f"🔄 Trying latest file: {latest_file}")
                            
                            if os.path.getsize(latest_file) > 0:
                                is_valid, validation_msg = self.validate_output_file(latest_file, config['content_type'])
                                if is_valid:
                                    print(f"✅ Using alternative file: {latest_file}")
                                    return latest_file
                        
                        # Include stdout in error for debugging
                        error_context = f"No valid output file. STDOUT: {result.stdout[-300:] if result.stdout else 'None'}"
                        raise Exception(error_context)
                        
                else:
                    print(f"❌ WAN failed with return code: {result.returncode}")
                    
                    # Enhanced error analysis
                    error_details = []
                    if result.stderr:
                        error_details.append(f"STDERR: {result.stderr[-300:]}")
                    if result.stdout:
                        error_details.append(f"STDOUT: {result.stdout[-300:]}")
                    
                    error_message = " | ".join(error_details) if error_details else "No error output captured"
                    raise Exception(f"WAN generation failed (code {result.returncode}): {error_message}")
                    
            except subprocess.TimeoutExpired:
                os.chdir(original_cwd)
                print(f"❌ WAN generation timed out after {timeout_seconds}s")
                
                # Cleanup partial files
                for partial_file in glob.glob("/tmp/wan_output_*"):
                    try:
                        size = os.path.getsize(partial_file)
                        print(f"🗑️ Cleaning partial file: {partial_file} ({size} bytes)")
                        os.unlink(partial_file)
                    except:
                        pass
                
                raise Exception(f"WAN generation timed out after {timeout_seconds} seconds")
                
            except Exception as e:
                os.chdir(original_cwd)
                print(f"❌ WAN subprocess error: {e}")
                
                # Cleanup partial files
                for partial_file in glob.glob("/tmp/wan_output_*"):
                    try:
                        os.unlink(partial_file)
                    except:
                        pass
                raise
                
        except Exception as e:
            if os.getcwd() != original_cwd:
                os.chdir(original_cwd)
            print(f"❌ WAN generation error: {e}")
            
            # Final cleanup
            for partial_file in glob.glob("/tmp/wan_output_*"):
                try:
                    os.unlink(partial_file)
                except:
                    pass
            raise

    def upload_to_supabase(self, file_path, storage_path):
        """Upload file to Supabase storage with enhanced validation"""
        try:
            # Pre-upload validation
            if not os.path.exists(file_path):
                raise Exception(f"File does not exist: {file_path}")
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise Exception(f"File is empty: {file_path}")
            
            # Enhanced MIME type checking
            mime_type, _ = mimetypes.guess_type(file_path)
            print(f"📤 Uploading file:")
            print(f"   Path: {file_path}")
            print(f"   Size: {file_size / 1024**2:.2f}MB")
            print(f"   MIME: {mime_type}")
            print(f"   Storage path: {storage_path}")
            
            # Double-check MIME type by reading file header
            with open(file_path, 'rb') as f:
                header = f.read(16)
                print(f"   Header: {header.hex()}")
                
                # Ensure it's not a text file
                if header.startswith(b'Traceback') or header.startswith(b'usage:') or header.startswith(b'Error'):
                    raise Exception(f"File appears to be error text, not binary content")
            
            # Determine MIME type based on file extension
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.mp4':
                mime_type = 'video/mp4'
            elif file_extension == '.png':
                mime_type = 'image/png'
            else:
                # Fallback to detected MIME type
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    mime_type = 'application/octet-stream'
            
            print(f"📤 Uploading with explicit MIME type: {mime_type}")
            
            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': (os.path.basename(file_path), file, mime_type)},
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                    },
                    timeout=180  # 3 minute upload timeout
                )
            
            if response.status_code == 200:
                # Return only relative path within bucket
                path_parts = storage_path.split('/', 1)
                relative_path = path_parts[1] if len(path_parts) == 2 else storage_path
                print(f"✅ Upload successful: {relative_path}")
                return relative_path
            else:
                error_text = response.text[:500]
                print(f"❌ Upload failed: {response.status_code}")
                print(f"📄 Error response: {error_text}")
                raise Exception(f"Upload failed: {response.status_code} - {error_text}")
                
        except Exception as e:
            print(f"❌ Supabase upload error: {e}")
            raise

    def notify_completion(self, job_id, status, output_url=None, error_message=None):
        """Notify Supabase of job completion with FIXED callback format"""
        try:
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
        """CRITICAL FIX: Enhanced process_job with better error handling"""
        job_id = job_data['jobId']
        job_type = job_data['jobType']
        original_prompt = job_data['prompt']
        video_id = job_data['videoId']
        
        print(f"🔄 Processing job {job_id} ({job_type}) with CRITICAL FIXES")
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
            
            # Handle prompt enhancement
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
            
            print("🎬 Starting WAN generation with CRITICAL FIXES...")
            print(f"🔍 About to call generate_content with:")
            print(f"   Prompt: {actual_prompt[:100]}...")
            print(f"   Job type: {job_type}")
            print(f"   Config: {config}")
            print(f"   Expected output: {config['content_type']} (.{config['file_extension']})")
            
            print("\n🔍 FINAL ENVIRONMENT CHECK BEFORE WAN:")
            test_env = self.enhanced_environment_setup()
            
            # CRITICAL: Generate content with enhanced error handling
            output_file = self.generate_content(actual_prompt, job_type)
            
            if not output_file:
                raise Exception("Content generation failed or produced no output")
            
            # Final file validation before upload
            print(f"🔍 Final validation before upload:")
            is_valid, validation_msg = self.validate_output_file(output_file, config['content_type'])
            if not is_valid:
                raise Exception(f"Generated file failed final validation: {validation_msg}")
            
            # Upload with proper storage path
            file_extension = config['file_extension']
            storage_path = f"{job_type}/{video_id}.{file_extension}"
            
            print(f"📤 Uploading validated {config['content_type']} file to: {storage_path}")
            relative_path = self.upload_to_supabase(output_file, storage_path)
            
            # Cleanup temp file
            try:
                os.unlink(output_file)
                print(f"🗑️ Cleaned up temp file: {output_file}")
            except:
                pass
            
            # Success callback
            self.notify_completion(job_id, 'completed', relative_path)
            
            total_time = time.time() - job_start_time
            print(f"🎉 Job {job_id} completed successfully in {total_time:.1f}s")
            print(f"📁 Output: {relative_path}")
            print(f"✅ File type: {config['content_type']} (.{file_extension})")
            
        except Exception as e:
            error_msg = str(e)
            total_time = time.time() - job_start_time
            print(f"❌ Job {job_id} failed after {total_time:.1f}s: {error_msg}")
            
            # Enhanced error categorization
            if "timeout" in error_msg.lower():
                print("💡 TIMEOUT: WAN subprocess exceeded time limit")
            elif "mime" in error_msg.lower() or "text/plain" in error_msg.lower():
                print("💡 MIME ERROR: WAN generated text instead of binary file")
                print("💡 SOLUTION: Check WAN command format and file extensions")
            elif "validation" in error_msg.lower():
                print("💡 VALIDATION ERROR: Generated file doesn't match expected format")
            elif "upload" in error_msg.lower():
                print("💡 UPLOAD ERROR: File upload to Supabase failed")
            elif "import" in error_msg.lower() or "module" in error_msg.lower():
                print("💡 DEPENDENCY ERROR: WAN dependencies not accessible")
            elif "command" in error_msg.lower() or "returncode" in error_msg.lower():
                print("💡 COMMAND ERROR: WAN subprocess failed to execute")
            
            # Cleanup any temp files
            try:
                for temp_file in glob.glob("/tmp/wan_output_*"):
                    os.unlink(temp_file)
            except:
                pass
            
            self.notify_completion(job_id, 'failed', error_message=error_msg)

    def run_with_enhanced_diagnostics(self):
        """Main worker loop with startup diagnostics"""
        print("🎬 Enhanced OurVidz WAN Worker with CRITICAL FIXES started!")
        print("🔧 CRITICAL FIXES APPLIED:")
        print("   • Proper file extensions (.mp4/.png)")
        print("   • Enhanced WAN command formatting")
        print("   • MIME type validation and error detection")
        print("   • File header validation to catch text/error output")
        print("   • Extended timeouts for video generation")
        print("   • Enhanced error categorization and debugging")
        
        print("\n🔍 STARTUP DIAGNOSTICS:")
        print("="*60)
        self.test_wan_dependencies()
        print("\n🧪 STARTUP WAN EXECUTION TEST:")
        self.test_wan_basic_execution()
        print("="*60)
        
        print("🔧 UPSTASH COMPATIBLE: Using non-blocking RPOP for Redis polling")
        print("📋 Supported job types:")
        for job_type, config in self.job_configs.items():
            enhancement = "✨ Enhanced" if config['enhance_prompt'] else "📝 Standard"
            content = "🖼️ Image" if config['content_type'] == 'image' else "🎬 Video"
            print(f"  • {job_type}: {content} (.{config['file_extension']}) ({config['expected_time']}s) {enhancement}")
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
            response = requests.get(
                f"{self.redis_url}/rpop/wan_queue",
                headers={
                    'Authorization': f"Bearer {self.redis_token}"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('result'):
                    job_json = result['result']
                    job_data = json.loads(job_json)
                    return job_data
            
            return None
            
        except requests.exceptions.Timeout:
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
    qwen_path = "/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct"
    wan_code_path = "/workspace/Wan2.1"
    
    if not os.path.exists(model_path):
        print(f"❌ WAN model not found: {model_path}")
        exit(1)
        
    if not os.path.exists(qwen_path):
        print(f"⚠️ Qwen model not found: {qwen_path} (enhancement will be disabled)")
        
    if not os.path.exists(wan_code_path):
        print(f"❌ WAN code not found: {wan_code_path}")
        exit(1)
    
    print("✅ All paths validated, starting worker with CRITICAL FIXES...")
    
    try:
        worker = EnhancedWanWorker()
        worker.run_with_enhanced_diagnostics()
    except Exception as e:
        print(f"❌ Worker startup failed: {e}")
        exit(1)
    finally:
        print("👋 Enhanced WAN Worker shutdown complete")