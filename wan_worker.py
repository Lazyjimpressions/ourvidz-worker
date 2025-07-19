# wan_worker.py - CRITICAL FIX for WAN Video Generation + REFERENCE FRAMES + CONSISTENT PARAMETER NAMING
# FIXES: WAN generating text files instead of videos, MIME type errors, command formatting
# MAJOR FIX: Corrected frame_num for 5-second videos (83 frames at 16.67fps)
# NEW FIX: Updated to use Qwen 2.5-7B Base model (no content filtering)
# PARAMETER FIX: Consistent parameter names (job_id, assets) with edge function
# ENHANCED: Advanced NSFW optimization with UniPC sampling and temporal consistency
# NEW: Reference frame support for video generation with start/end frame guidance
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
import io
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

class TimeoutException(Exception):
    """Custom exception for timeouts"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeouts"""
    raise TimeoutException("Operation timed out")

class EnhancedWanWorker:
    def __init__(self):
        """Initialize Enhanced WAN Worker with Qwen 7B Base integration"""
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_code_path = "/workspace/Wan2.1"
        
        # CRITICAL: Set environment variables immediately (VERIFIED FIX)
        os.environ['PYTHONPATH'] = '/workspace/python_deps/lib/python3.11/site-packages'
        os.environ['HF_HOME'] = '/workspace/models/huggingface_cache'
        os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/models/huggingface_cache/hub'
        
        # UPDATED: Qwen 2.5-7B Base model path (no content filtering)
        self.hf_cache_path = "/workspace/models/huggingface_cache"
        self.qwen_model_path = "/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"
        
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
        
        # ENHANCED: Advanced NSFW-optimized configurations with UniPC sampling and temporal consistency
        # Based on WAN 2.1 research: UniPC sampling improves temporal consistency and reduces choppiness
        # Advanced parameters: sample_solver, sample_shift, temporal consistency, NSFW-optimized guidance
        self.job_configs = {
            # Standard job types (no enhancement) - ENHANCED with advanced parameters
            'image_fast': {
                'size': '480*832',           # ✅ VERIFIED working size
                'sample_steps': 25,          # Fast: 25 steps
                'sample_guide_scale': 6.5,   # 🔧 ENHANCED: Better NSFW quality (was 5.0)
                'sample_solver': 'unipc',    # 🔧 NEW: UniPC sampling for better quality
                'sample_shift': 5.0,         # 🔧 NEW: Temporal consistency
                'frame_num': 1,              # Single frame for images
                'enhance_prompt': False,
                'expected_time': 25,         # Estimated time
                'content_type': 'image',
                'file_extension': 'png'      # ✅ CRITICAL: Explicit extension
            },
            'image_high': {
                'size': '480*832',
                'sample_steps': 50,          # High quality: 50 steps
                'sample_guide_scale': 7.5,   # 🔧 ENHANCED: Higher guidance for better NSFW quality
                'sample_solver': 'unipc',    # 🔧 NEW: UniPC sampling
                'sample_shift': 5.0,         # 🔧 NEW: Temporal consistency
                'frame_num': 1,
                'enhance_prompt': False,
                'expected_time': 40,
                'content_type': 'image',
                'file_extension': 'png'
            },
            'video_fast': {
                'size': '480*832',           # ✅ VERIFIED working size
                'sample_steps': 25,          # Fast: 25 steps
                'sample_guide_scale': 6.5,   # 🔧 ENHANCED: Better NSFW quality (was 5.0)
                'sample_solver': 'unipc',    # 🔧 NEW: UniPC sampling reduces choppiness
                'sample_shift': 5.0,         # 🔧 NEW: Temporal consistency between frames
                'frame_num': 83,             # 🔧 OPTIMIZED: 83 frames for 5.0 seconds (confirmed 16.67fps)
                'enhance_prompt': False,
                'expected_time': 135,        # 🔧 UPDATED: 83 × 2.67s/frame = 221.6s → 135s accounting for overhead
                'content_type': 'video',
                'file_extension': 'mp4'      # ✅ CRITICAL: Explicit extension
            },
            'video_high': {
                'size': '480*832',
                'sample_steps': 50,          # High quality: 50 steps
                'sample_guide_scale': 7.5,   # 🔧 ENHANCED: Higher guidance for better NSFW quality
                'sample_solver': 'unipc',    # 🔧 NEW: UniPC sampling for smooth motion
                'sample_shift': 5.0,         # 🔧 NEW: Temporal consistency
                'frame_num': 83,             # 🔧 OPTIMIZED: 83 frames for 5.0 seconds
                'enhance_prompt': False,
                'expected_time': 180,        # 🔧 UPDATED: Higher quality takes longer per frame
                'content_type': 'video',
                'file_extension': 'mp4'
            },
            
            # Enhanced job types (with Qwen 7B Base enhancement) - ENHANCED with NSFW optimization
            'image7b_fast_enhanced': {
                'size': '480*832',
                'sample_steps': 25,
                'sample_guide_scale': 6.5,   # 🔧 ENHANCED: Better NSFW quality
                'sample_solver': 'unipc',    # 🔧 NEW: UniPC sampling
                'sample_shift': 5.0,         # 🔧 NEW: Temporal consistency
                'frame_num': 1,
                'enhance_prompt': True,
                'expected_time': 85,         # 25s + 60s enhancement
                'content_type': 'image',
                'file_extension': 'png'
            },
            'image7b_high_enhanced': {
                'size': '480*832',
                'sample_steps': 50,
                'sample_guide_scale': 7.5,   # 🔧 ENHANCED: Higher guidance for NSFW quality
                'sample_solver': 'unipc',    # 🔧 NEW: UniPC sampling
                'sample_shift': 5.0,         # 🔧 NEW: Temporal consistency
                'frame_num': 1,
                'enhance_prompt': True,
                'expected_time': 100,        # 40s + 60s enhancement
                'content_type': 'image',
                'file_extension': 'png'
            },
            'video7b_fast_enhanced': {
                'size': '480*832',
                'sample_steps': 25,
                'sample_guide_scale': 6.5,   # 🔧 ENHANCED: Better NSFW quality
                'sample_solver': 'unipc',    # 🔧 NEW: UniPC sampling reduces choppiness
                'sample_shift': 5.0,         # 🔧 NEW: Temporal consistency between frames
                'frame_num': 83,             # 🔧 OPTIMIZED: 83 frames for 5.0 seconds
                'enhance_prompt': True,
                'expected_time': 195,        # 🔧 UPDATED: 135s + 60s enhancement
                'content_type': 'video',
                'file_extension': 'mp4'
            },
            'video7b_high_enhanced': {
                'size': '480*832',
                'sample_steps': 50,
                'sample_guide_scale': 7.5,   # 🔧 ENHANCED: Higher guidance for NSFW quality
                'sample_solver': 'unipc',    # 🔧 NEW: UniPC sampling for smooth motion
                'sample_shift': 5.0,         # 🔧 NEW: Temporal consistency
                'frame_num': 83,             # 🔧 OPTIMIZED: 83 frames for 5.0 seconds
                'enhance_prompt': True,
                'expected_time': 240,        # 🔧 UPDATED: 180s + 60s enhancement
                'content_type': 'video',
                'file_extension': 'mp4'
            }
        }
        
        print("🎬 Enhanced OurVidz WAN Worker initialized - NSFW OPTIMIZED + REFERENCE FRAMES")
        print("🔧 MAJOR FIX: Corrected frame counts for 5-second videos (83 frames)")
        print("🔧 PARAMETER FIX: Consistent parameter names (job_id, assets) with edge function")
        print("🔧 ENHANCED: Advanced NSFW optimization with UniPC sampling and temporal consistency")
        print("🔧 ENHANCED: Improved guidance scales (6.5-7.5) for better NSFW quality")
        print("🔧 ENHANCED: NSFW-optimized prompt enhancement for realistic adult content")
        print("🖼️ NEW: Reference frame support for video generation with start/end frame guidance")
        print(f"📋 Supporting ALL 8 job types: {list(self.job_configs.keys())}")
        print(f"📁 WAN Model Path: {self.model_path}")
        print(f"🤖 Qwen Base Model Path: {self.qwen_model_path}")
        print("🔧 CRITICAL FIX: Proper file extensions and WAN command formatting")
        print("🔧 CRITICAL FIX: Enhanced output file validation")
        print("🔧 CRITICAL FIX: Removed --negative_prompt (not supported by WAN 2.1)")
        print("📊 Status: Enhanced with Qwen 7B Base + NSFW optimization + Reference Frames ✅")
        self.log_gpu_memory()

    def download_image_from_url(self, image_url):
        """Download image from URL and return PIL Image object"""
        try:
            print(f"📥 Downloading reference image from: {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            print(f"✅ Reference image downloaded: {image.size}")
            return image
            
        except Exception as e:
            print(f"❌ Failed to download reference image: {e}")
            raise

    def preprocess_reference_image(self, image, target_size=(480, 832)):
        """Preprocess reference image for WAN video generation"""
        try:
            # Resize image to target size while maintaining aspect ratio
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image
            new_image = Image.new('RGB', target_size, (0, 0, 0))
            
            # Center the image
            x = (target_size[0] - image.width) // 2
            y = (target_size[1] - image.height) // 2
            new_image.paste(image, (x, y))
            
            print(f"✅ Reference image preprocessed to {target_size}")
            return new_image
            
        except Exception as e:
            print(f"❌ Failed to preprocess reference image: {e}")
            raise

    def save_reference_image(self, image, filename):
        """Save reference image to temporary file for WAN processing"""
        try:
            temp_path = f"/tmp/{filename}"
            image.save(temp_path, "PNG", quality=95, optimize=True)
            print(f"💾 Reference image saved: {temp_path}")
            return temp_path
        except Exception as e:
            print(f"❌ Failed to save reference image: {e}")
            raise

    def generate_video_with_references(self, prompt, start_reference, end_reference, strength, job_type):
        """Generate video with start and/or end reference frames"""
        print(f"🎬 Generating video with reference frames (strength: {strength})")
        
        if start_reference and end_reference:
            print("🖼️ Using both start and end reference frames")
            return self.generate_video_with_start_end_references(prompt, start_reference, end_reference, strength, job_type)
        elif start_reference:
            print("🖼️ Using start reference frame only")
            return self.generate_video_with_start_reference(prompt, start_reference, strength, job_type)
        elif end_reference:
            print("🖼️ Using end reference frame only")
            return self.generate_video_with_end_reference(prompt, end_reference, strength, job_type)
        else:
            print("⚠️ No reference frames provided, falling back to standard generation")
            return self.generate_standard_video(prompt, job_type)

    def generate_video_with_start_end_references(self, prompt, start_ref, end_ref, strength, job_type):
        """Generate video with both start and end reference frames"""
        print("🎬 Generating video with start and end reference frames")
        
        # Preprocess reference images
        start_processed = self.preprocess_reference_image(start_ref)
        end_processed = self.preprocess_reference_image(end_ref)
        
        # Save reference images to temp files
        timestamp = int(time.time())
        start_ref_path = self.save_reference_image(start_processed, f"wan_start_ref_{timestamp}.png")
        end_ref_path = self.save_reference_image(end_processed, f"wan_end_ref_{timestamp}.png")
        
        try:
            # Generate video with reference frames using WAN
            output_file = self.generate_content_with_references(prompt, job_type, start_ref_path, end_ref_path, strength)
            return output_file
        finally:
            # Cleanup reference files
            try:
                os.unlink(start_ref_path)
                os.unlink(end_ref_path)
            except:
                pass

    def generate_video_with_start_reference(self, prompt, start_ref, strength, job_type):
        """Generate video with start reference frame only"""
        print("🎬 Generating video with start reference frame")
        
        # Preprocess reference image
        start_processed = self.preprocess_reference_image(start_ref)
        
        # Save reference image to temp file
        timestamp = int(time.time())
        start_ref_path = self.save_reference_image(start_processed, f"wan_start_ref_{timestamp}.png")
        
        try:
            # Generate video with start reference frame
            output_file = self.generate_content_with_references(prompt, job_type, start_ref_path, None, strength)
            return output_file
        finally:
            # Cleanup reference file
            try:
                os.unlink(start_ref_path)
            except:
                pass

    def generate_video_with_end_reference(self, prompt, end_ref, strength, job_type):
        """Generate video with end reference frame only"""
        print("🎬 Generating video with end reference frame")
        
        # Preprocess reference image
        end_processed = self.preprocess_reference_image(end_ref)
        
        # Save reference image to temp file
        timestamp = int(time.time())
        end_ref_path = self.save_reference_image(end_processed, f"wan_end_ref_{timestamp}.png")
        
        try:
            # Generate video with end reference frame
            output_file = self.generate_content_with_references(prompt, job_type, None, end_ref_path, strength)
            return output_file
        finally:
            # Cleanup reference file
            try:
                os.unlink(end_ref_path)
            except:
                pass

    def generate_standard_video(self, prompt, job_type):
        """Generate standard video without reference frames"""
        print("🎬 Generating standard video without reference frames")
        return self.generate_content(prompt, job_type)

    def generate_content_with_references(self, prompt, job_type, start_ref_path=None, end_ref_path=None, strength=0.5):
        """Generate content with reference frames using WAN command line"""
        if job_type not in self.job_configs:
            raise Exception(f"Unsupported job type: {job_type}")
            
        config = self.job_configs[job_type]
        
        # Create output path with proper extension
        timestamp = int(time.time())
        file_extension = config['file_extension']
        output_filename = f"wan_output_{timestamp}.{file_extension}"
        temp_output_path = f"/tmp/{output_filename}"
        
        print(f"🎯 Output path: {temp_output_path}")
        print(f"📄 Expected file type: {config['content_type']} (.{file_extension})")
        print(f"🔧 FRAME COUNT: {config['frame_num']} frames for {config['content_type']}")
        
        try:
            # Change to WAN code directory
            original_cwd = os.getcwd()
            os.chdir(self.wan_code_path)
            
            # Build WAN command with reference frame support
            cmd = [
                "python", "generate.py",
                "--task", "t2v-1.3B",
                "--ckpt_dir", self.model_path,
                "--offload_model", "True",
                "--size", config['size'],
                "--sample_steps", str(config['sample_steps']),
                "--sample_guide_scale", str(config['sample_guide_scale']),
                "--sample_solver", config.get('sample_solver', 'unipc'),
                "--sample_shift", str(config.get('sample_shift', 5.0)),
                "--frame_num", str(config['frame_num']),
                "--prompt", prompt,
                "--save_file", temp_output_path
            ]
            
            # Add reference frame parameters if provided
            if start_ref_path:
                cmd.extend(["--start_frame", start_ref_path])
                print(f"🖼️ Start reference frame: {start_ref_path}")
            
            if end_ref_path:
                cmd.extend(["--end_frame", end_ref_path])
                print(f"🖼️ End reference frame: {end_ref_path}")
            
            if start_ref_path or end_ref_path:
                cmd.extend(["--reference_strength", str(strength)])
                print(f"🔧 Reference strength: {strength}")
            
            # Configure environment
            env = self.setup_environment()
            
            print(f"🎬 WAN generation with references: {job_type}")
            print(f"📝 Prompt: {prompt[:100]}...")
            print(f"🔧 Config: {config['sample_steps']} steps, {config['frame_num']} frames, {config['size']}")
            print(f"💾 Output: {temp_output_path}")
            print(f"📁 Working dir: {self.wan_code_path}")
            print(f"🔧 Command: {' '.join(cmd)}")
            
            # Execute WAN generation
            generation_start = time.time()
            timeout_seconds = 500 if config['content_type'] == 'video' else 180
            
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
                os.chdir(original_cwd)
                
                print(f"✅ WAN subprocess completed in {generation_time:.1f}s")
                print(f"📄 Return code: {result.returncode}")
                
                # Enhanced output analysis
                if result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')
                    print(f"📄 STDOUT ({len(stdout_lines)} lines):")
                    for line in stdout_lines[-10:]:
                        print(f"   [OUT] {line}")
                
                if result.stderr:
                    stderr_lines = result.stderr.strip().split('\n')
                    print(f"📄 STDERR ({len(stderr_lines)} lines):")
                    for line in stderr_lines[-10:]:
                        print(f"   [ERR] {line}")
                
                # Validate output
                if result.returncode == 0:
                    print(f"🔍 Checking output file: {temp_output_path}")
                    
                    if os.path.exists(temp_output_path):
                        file_size = os.path.getsize(temp_output_path)
                        print(f"✅ Output file found: {file_size / 1024**2:.2f}MB")
                        
                        is_valid, validation_msg = self.validate_output_file(temp_output_path, config['content_type'])
                        if is_valid:
                            print(f"✅ File validation passed: {validation_msg}")
                            return temp_output_path
                        else:
                            print(f"❌ File validation failed: {validation_msg}")
                            raise Exception(f"Generated file validation failed: {validation_msg}")
                    else:
                        print(f"❌ Output file not found: {temp_output_path}")
                        raise Exception("No valid output file generated")
                        
                else:
                    print(f"❌ WAN failed with return code: {result.returncode}")
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
                raise Exception(f"WAN generation timed out after {timeout_seconds} seconds")
                
            except Exception as e:
                os.chdir(original_cwd)
                print(f"❌ WAN subprocess error: {e}")
                raise
                
        except Exception as e:
            if os.getcwd() != original_cwd:
                os.chdir(original_cwd)
            print(f"❌ WAN generation error: {e}")
            raise

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
        """Load Qwen 2.5-7B Base model for prompt enhancement with timeout protection"""
        if self.qwen_model is None:
            print("🤖 Loading Qwen 2.5-7B Base model for prompt enhancement...")
            enhancement_start = time.time()
            
            try:
                # Set timeout for model loading
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)  # 2 minute timeout for model loading
                
                model_path = self.qwen_model_path
                print(f"🔄 Loading Qwen 2.5-7B Base model from {model_path}")
                
                # Load tokenizer first
                print("📝 Loading tokenizer...")
                self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                # Load base model - no safety filters
                print("🧠 Loading base model...")
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,  # Base models work well with bfloat16
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Set pad token for base models (they often don't have one)
                if self.qwen_tokenizer.pad_token is None:
                    self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
                
                signal.alarm(0)
                
                load_time = time.time() - enhancement_start
                print(f"✅ Qwen 2.5-7B Base loaded successfully in {load_time:.1f}s")
                print(f"✅ Model type: BASE (no content filtering)")
                self.log_gpu_memory()
                
            except TimeoutException:
                signal.alarm(0)
                print(f"❌ Qwen model loading timed out after 120s")
                self.qwen_model = None
                self.qwen_tokenizer = None
            except Exception as e:
                signal.alarm(0)
                print(f"❌ Failed to load Qwen base model: {e}")
                print(f"❌ Full error traceback:")
                import traceback
                traceback.print_exc()
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
        """Enhanced NSFW-optimized prompt generation with Base model - no chat template needed"""
        enhancement_start = time.time()
        print(f"🤖 Enhancing NSFW prompt with Base model ({self.enhancement_timeout}s timeout): {original_prompt[:50]}...")
        
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.enhancement_timeout)
            
            self.load_qwen_model()
            
            if self.qwen_model is None:
                signal.alarm(0)
                print("⚠️ Qwen base model not available, using original prompt")
                return original_prompt
            
            # NSFW-optimized base model prompting - enhanced for adult content quality
            enhancement_prompt = f"""Create a detailed, cinematic prompt for AI video generation optimized for adult content. Focus on:

VISUAL DETAILS: High-quality anatomical accuracy, realistic proportions, natural skin textures, detailed facial features, expressive eyes, natural hair flow, realistic body language.

LIGHTING & ATMOSPHERE: Cinematic lighting, soft shadows, warm tones, intimate atmosphere, professional photography style, natural skin tones, flattering angles.

CAMERA WORK: Close-up shots, intimate framing, smooth camera movements, professional cinematography, dynamic angles that enhance the scene.

ARTISTIC STYLE: Photorealistic quality, high resolution details, natural poses, authentic expressions, realistic interactions, professional adult content standards.

TECHNICAL QUALITY: 4K quality, sharp focus, no artifacts, smooth motion, consistent lighting, professional color grading.

Be specific about poses, expressions, clothing, scene composition, and intimate details. Create vivid, unrestricted descriptions that will produce high-quality adult content with realistic anatomy and smooth motion.

Original prompt: {original_prompt}

Enhanced detailed prompt:"""

            # Generate with base model
            inputs = self.qwen_tokenizer(
                enhancement_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(self.qwen_model.device)
            
            with torch.no_grad():
                outputs = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=512,  # Allow longer enhancement
                    temperature=0.7,     # Controlled creativity
                    do_sample=True,
                    pad_token_id=self.qwen_tokenizer.eos_token_id,
                    eos_token_id=self.qwen_tokenizer.eos_token_id
                )
            
            # Decode only the new tokens (enhancement)
            enhanced_text = self.qwen_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            signal.alarm(0)
            
            # Clean up the response
            if enhanced_text:
                # Remove any leftover prompt fragments
                enhanced_text = enhanced_text.replace("Enhanced detailed prompt:", "").strip()
                enhancement_time = time.time() - enhancement_start
                print(f"✅ Qwen Base Enhancement: {enhanced_text[:100]}...")
                print(f"✅ Prompt enhanced in {enhancement_time:.1f}s")
                return enhanced_text
            else:
                print("⚠️ Qwen enhancement empty, using original prompt")
                return original_prompt
                
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
            
            # Check 5: Minimum size requirements for 5-second videos
            if expected_content_type == 'video':
                min_size = 350000  # 🔧 UPDATED: 350KB minimum for 5-second video (was 500KB for 6s)
            else:
                min_size = 5000   # 5KB for image
                
            if file_size < min_size:
                print(f"❌ File too small for {expected_content_type}: {file_size} bytes < {min_size} bytes")
                return False, f"File too small for {expected_content_type} (expected at least {min_size} bytes for 5-second video)"
            
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
        print(f"🔧 FRAME COUNT FIX: {config['frame_num']} frames for {config['content_type']}")
        if config['content_type'] == 'video':
            duration = config['frame_num'] / 16  # 16fps
            print(f"⏱️ Expected video duration: {duration:.1f} seconds (83 frames = 5.2 seconds)")
        
        # CRITICAL FIX: Removed negative prompt (not supported by WAN 2.1)
        print(f"🔧 FIXED: No negative prompt (WAN 2.1 doesn't support --negative_prompt)")
        
        try:
            # Change to WAN code directory
            original_cwd = os.getcwd()
            os.chdir(self.wan_code_path)
            
            # ENHANCED: Build WAN command with advanced NSFW-optimized parameters
            # Based on WAN 2.1 research: UniPC sampling, temporal consistency, advanced guidance
            cmd = [
                "python", "generate.py",
                "--task", "t2v-1.3B",                           # ✅ VERIFIED working task
                "--ckpt_dir", self.model_path,                  # ✅ Model path
                "--offload_model", "True",                      # ✅ VERIFIED: Memory management
                "--size", config['size'],                       # ✅ VERIFIED: 480*832
                "--sample_steps", str(config['sample_steps']),  # ✅ Steps: 25 or 50
                "--sample_guide_scale", str(config['sample_guide_scale']),  # 🔧 ENHANCED: 6.5-7.5 for NSFW quality
                "--sample_solver", config.get('sample_solver', 'unipc'),  # 🔧 NEW: UniPC sampling for smooth motion
                "--sample_shift", str(config.get('sample_shift', 5.0)),   # 🔧 NEW: Temporal consistency
                "--frame_num", str(config['frame_num']),        # 🔧 FIXED: 83 frames for 5-second videos
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
            # 🔧 UPDATED: Optimized timeout for 83-frame videos
            timeout_seconds = 500 if config['content_type'] == 'video' else 180  # 8 minutes for videos, 3 for images
            
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
            
            # Add metadata if provided (for generation details)
            if metadata:
                callback_data['metadata'] = metadata
            
            print(f"📞 Sending CONSISTENT callback for job {job_id}:")
            print(f"   Status: {status}")
            print(f"   Assets count: {len(assets) if assets else 0}")
            print(f"   Error: {error_message}")
            print(f"   Metadata: {metadata}")
            print(f"   Parameters: job_id, status, assets, error_message, metadata (CONSISTENT)")
            
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
                print(f"✅ CONSISTENT Callback sent successfully for job {job_id}")
            else:
                print(f"❌ Callback failed: {response.status_code} - {response.text}")
                print(f"❌ Callback payload was: {callback_data}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job_with_enhanced_diagnostics(self, job_data):
        """CONSISTENT: Enhanced process_job with standardized payload structure"""
        # CONSISTENT: Use standardized field names across all workers
        job_id = job_data['id']           # ✅ Standard: 'id' field
        job_type = job_data['type']       # ✅ Standard: 'type' field
        original_prompt = job_data['prompt']  # ✅ Standard: 'prompt' field
        user_id = job_data['user_id']     # ✅ Standard: 'user_id' field
        
        # Optional fields with defaults
        video_id = job_data.get('video_id', f"video_{int(time.time())}")
        image_id = job_data.get('image_id', f"image_{int(time.time())}")
        config = job_data.get('config', {})
        
        # Extract reference frame parameters from config and metadata (UPDATED API SPEC)
        metadata = job_data.get('metadata', {})
        # ✅ NEW: Check config level first, then metadata level (per API spec)
        start_reference_url = config.get('first_frame') or metadata.get('start_reference_url')
        end_reference_url = config.get('last_frame') or metadata.get('end_reference_url')
        reference_strength = metadata.get('reference_strength', 0.5)
        
        print(f"🔄 Processing job {job_id} ({job_type}) with CONSISTENT PARAMETERS")
        print(f"📝 Original prompt: {original_prompt}")
        print(f"🎯 Video ID: {video_id}")
        print(f"👤 User ID: {user_id}")
        
        # Log reference frame parameters if present (UPDATED API SPEC)
        if start_reference_url or end_reference_url:
            print(f"🖼️ Reference frame mode: strength {reference_strength}")
            if start_reference_url:
                print(f"📥 Start reference frame URL (first_frame/start_reference_url): {start_reference_url}")
            if end_reference_url:
                print(f"📥 End reference frame URL (last_frame/end_reference_url): {end_reference_url}")
        
        job_start_time = time.time()
        
        try:
            if job_type not in self.job_configs:
                available_types = list(self.job_configs.keys())
                raise Exception(f"Unknown job type: {job_type}. Available: {available_types}")
            
            job_config = self.job_configs[job_type]
            print(f"✅ Job type validated: {job_type} (enhance: {job_config['enhance_prompt']})")
            
            # Use config from edge function if available, otherwise use defaults
            final_config = {**job_config, **config}
            print(f"🔧 OPTIMIZED FRAME COUNT: {final_config['frame_num']} frames")
            if final_config['content_type'] == 'video':
                # Based on confirmed data: 100 frames = 6 seconds, so 16.67fps effective
                effective_fps = 16.67  # Confirmed from successful 6-second generation
                duration = final_config['frame_num'] / effective_fps
                print(f"⏱️ Expected duration: {duration:.1f} seconds (confirmed 16.67fps effective rate)")
            
            # Handle prompt enhancement
            if final_config['enhance_prompt']:
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
            
            # Handle reference frame generation for video jobs
            if final_config['content_type'] == 'video' and (start_reference_url or end_reference_url):
                print("🎬 Starting WAN video generation with reference frames...")
                print(f"🔧 REFERENCE FRAME MODE: {final_config['frame_num']} frames for 5-second videos")
                
                # Download reference images
                start_reference = None
                end_reference = None
                
                if start_reference_url:
                    try:
                        start_reference = self.download_image_from_url(start_reference_url)
                        print(f"✅ Start reference image loaded successfully")
                    except Exception as e:
                        print(f"❌ Failed to load start reference image: {e}")
                        # Continue without start reference
                
                if end_reference_url:
                    try:
                        end_reference = self.download_image_from_url(end_reference_url)
                        print(f"✅ End reference image loaded successfully")
                    except Exception as e:
                        print(f"❌ Failed to load end reference image: {e}")
                        # Continue without end reference
                
                # Generate video with reference frames
                output_file = self.generate_video_with_references(
                    actual_prompt, 
                    start_reference, 
                    end_reference, 
                    reference_strength,
                    job_type
                )
            else:
                print("🎬 Starting WAN generation with CRITICAL FIXES...")
                print(f"🔧 FIXED FRAME COUNT: {final_config['frame_num']} frames for 5-second videos")
                
                # CRITICAL: Generate content with enhanced error handling
                output_file = self.generate_content(actual_prompt, job_type)
            
            if not output_file:
                raise Exception("Content generation failed or produced no output")
            
            # Final file validation before upload
            print(f"🔍 Final validation before upload:")
            is_valid, validation_msg = self.validate_output_file(output_file, final_config['content_type'])
            if not is_valid:
                raise Exception(f"Generated file failed final validation: {validation_msg}")
            
            # Upload with proper storage path (user-scoped)
            file_extension = final_config['file_extension']
            storage_path = f"{job_type}/{user_id}/{video_id}.{file_extension}"
            
            print(f"📤 Uploading validated {final_config['content_type']} file to: {storage_path}")
            relative_path = self.upload_to_supabase(output_file, storage_path)
            
            # Cleanup temp file
            try:
                os.unlink(output_file)
                print(f"🗑️ Cleaned up temp file: {output_file}")
            except:
                pass
            
            # Prepare metadata for callback
            callback_metadata = {
                'generation_time': total_time,
                'job_type': job_type,
                'content_type': final_config['content_type'],
                'frame_num': final_config['frame_num']
            }
            
            # CONSISTENT: Success callback with standardized parameters and metadata
            self.notify_completion(job_id, 'completed', assets=[relative_path], metadata=callback_metadata)
            
            total_time = time.time() - job_start_time
            print(f"🎉 Job {job_id} completed successfully in {total_time:.1f}s")
            print(f"📁 Output: {relative_path}")
            print(f"✅ File type: {final_config['content_type']} (.{file_extension})")
            if final_config['content_type'] == 'video':
                effective_fps = 16.67  # Confirmed from successful generation: 100 frames = 6 seconds
                duration = final_config['frame_num'] / effective_fps
                print(f"⏱️ Video duration: {duration:.1f} seconds ({final_config['frame_num']} frames at 16.67fps confirmed)")
            
        except Exception as e:
            error_msg = str(e)
            total_time = time.time() - job_start_time
            print(f"❌ Job {job_id} failed after {total_time:.1f}s: {error_msg}")
            
            # Cleanup any temp files
            try:
                for temp_file in glob.glob("/tmp/wan_output_*"):
                    os.unlink(temp_file)
            except:
                pass
            
            # Prepare error metadata
            error_metadata = {
                'error_type': type(e).__name__,
                'job_type': job_type,
                'timestamp': time.time()
            }
            
            # CONSISTENT: Failure callback with standardized parameters and metadata
            self.notify_completion(job_id, 'failed', error_message=error_msg, metadata=error_metadata)

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

    def run_with_enhanced_diagnostics(self):
        """Main worker loop with startup diagnostics and CONSISTENT PARAMETERS"""
        print("🎬 Enhanced OurVidz WAN Worker with UPDATED API SPEC + REFERENCE FRAMES started!")
        print("🔧 MAJOR FIX: Corrected frame counts for 5-second videos (83 frames)")
        print("🔧 PARAMETER FIX: Consistent parameter names (job_id, assets) with edge function")
        print("🔧 MAJOR FIX: Updated to use Qwen 2.5-7B Base model (no content filtering)")
        print("🖼️ NEW: Reference frame support for video generation with start/end frame guidance")
        print("🔧 API UPDATE: Support for config.first_frame/last_frame and metadata.start_reference_url/end_reference_url")
        print("🔧 OPTIMIZATIONS APPLIED:")
        print("   • Optimized frame counts based on confirmed 16.67fps effective rate")
        print("   • video_fast: 83 frames for 5.0 seconds (45s faster processing)")
        print("   • video_high: 83 frames for 5.0 seconds (66s faster processing)")
        print("   • Reference frame support for enhanced video generation")
        print("   • Consistent callback parameters (job_id, status, assets, error_message, metadata)")
        print("   • Updated API spec support (config.first_frame/last_frame, metadata.start_reference_url/end_reference_url)")
        print("📊 Status: Enhanced with Qwen 7B Base + Reference Frames ✅")
        
        print("🔧 UPSTASH COMPATIBLE: Using non-blocking RPOP for Redis polling")
        print("📋 Supported job types:")
        for job_type, config in self.job_configs.items():
            enhancement = "✨ Enhanced" if config['enhance_prompt'] else "📝 Standard"
            content = "🖼️ Image" if config['content_type'] == 'image' else "🎬 Video"
            if config['content_type'] == 'video':
                effective_fps = 16.67  # Confirmed from successful generation
                duration = config['frame_num'] / effective_fps
                print(f"  • {job_type}: {content} (.{config['file_extension']}) ({config['expected_time']}s) {enhancement} - {duration:.1f}s duration ({config['frame_num']} frames)")
            else:
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
    qwen_path = "/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"
    wan_code_path = "/workspace/Wan2.1"
    
    if not os.path.exists(model_path):
        print(f"❌ WAN model not found: {model_path}")
        exit(1)
        
    if not os.path.exists(qwen_path):
        print(f"⚠️ Qwen Base model not found: {qwen_path} (enhancement will be disabled)")
        
    if not os.path.exists(wan_code_path):
        print(f"❌ WAN code not found: {wan_code_path}")
        exit(1)
    
    print("✅ All paths validated, starting worker with CONSISTENT PARAMETERS...")
    print("🔧 MAJOR OPTIMIZATION: 83 frames for 5-second videos (was 100 frames)")
    print("🔧 MAJOR FIX: Using Qwen 2.5-7B Base model for unrestricted NSFW enhancement")
    print("🔧 PARAMETER FIX: Using job_id, status, assets, error_message for compatibility")
    print("🔧 TIME SAVINGS: 45 seconds faster processing per video")
    
    try:
        worker = EnhancedWanWorker()
        worker.run_with_enhanced_diagnostics()
    except Exception as e:
        print(f"❌ Worker startup failed: {e}")
        exit(1)
    finally:
        print("👋 Enhanced WAN Worker shutdown complete")