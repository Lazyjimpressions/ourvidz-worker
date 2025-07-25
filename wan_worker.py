# wan_worker.py - CRITICAL FIX for WAN 1.3B Model + REFERENCE FRAMES + REFERENCE STRENGTH
# FIXES: Correct task names for 1.3B model, proper I2V support for reference frames
# MAJOR FIX: Use correct 1.3B tasks (t2v-1.3B, i2v not flf2v)
# PARAMETER FIX: Consistent parameter names (job_id, assets) with edge function
# REFERENCE STRENGTH FIX: Adjust sample_guide_scale based on reference strength
# Date: July 19, 2025

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
import threading
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# Flask imports for frontend enhancement API
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    print("⚠️ Flask not available - frontend enhancement API will be disabled")
    FLASK_AVAILABLE = False

class TimeoutException(Exception):
    """Custom exception for timeouts"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeouts"""
    raise TimeoutException("Operation timed out")

class EnhancedWanWorker:
    def __init__(self):
        """Initialize Enhanced WAN Worker for 1.3B Model with Reference Support"""
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
        
        # FIXED: 1.3B Model Configurations with correct task names
        self.job_configs = {
            # Standard job types (no enhancement) - FIXED for 1.3B model
            'image_fast': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B for single frame (image)
                'size': '480*832',
                'sample_steps': 25,
                'sample_guide_scale': 6.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 1,                # Single frame for images
                'enhance_prompt': False,
                'expected_time': 25,
                'content_type': 'image',
                'file_extension': 'png'
            },
            'image_high': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B for single frame
                'size': '480*832',
                'sample_steps': 50,
                'sample_guide_scale': 7.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 1,
                'enhance_prompt': False,
                'expected_time': 40,
                'content_type': 'image',
                'file_extension': 'png'
            },
            'video_fast': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B for video
                'size': '480*832',
                'sample_steps': 25,
                'sample_guide_scale': 6.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 83,               # 83 frames for 5-second videos
                'enhance_prompt': False,
                'expected_time': 135,
                'content_type': 'video',
                'file_extension': 'mp4'
            },
            'video_high': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B for video
                'size': '480*832',
                'sample_steps': 50,
                'sample_guide_scale': 7.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 83,
                'enhance_prompt': False,
                'expected_time': 180,
                'content_type': 'video',
                'file_extension': 'mp4'
            },
            
            # Enhanced job types (with Qwen 7B Base enhancement) - FIXED for 1.3B
            'image7b_fast_enhanced': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B
                'size': '480*832',
                'sample_steps': 25,
                'sample_guide_scale': 6.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 1,
                'enhance_prompt': True,
                'expected_time': 85,
                'content_type': 'image',
                'file_extension': 'png'
            },
            'image7b_high_enhanced': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B
                'size': '480*832',
                'sample_steps': 50,
                'sample_guide_scale': 7.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 1,
                'enhance_prompt': True,
                'expected_time': 100,
                'content_type': 'image',
                'file_extension': 'png'
            },
            'video7b_fast_enhanced': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B
                'size': '480*832',
                'sample_steps': 25,
                'sample_guide_scale': 6.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 83,
                'enhance_prompt': True,
                'expected_time': 195,
                'content_type': 'video',
                'file_extension': 'mp4'
            },
            'video7b_high_enhanced': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B
                'size': '480*832',
                'sample_steps': 50,
                'sample_guide_scale': 7.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 83,
                'enhance_prompt': True,
                'expected_time': 240,
                'content_type': 'video',
                'file_extension': 'mp4'
            }
        }
        
        print("🎬 Enhanced OurVidz WAN Worker - 1.3B MODEL + REFERENCE FRAMES + REFERENCE STRENGTH")
        print("🔧 CRITICAL FIX: Using correct t2v-1.3B task for WAN 1.3B model")
        print("🔧 REFERENCE SUPPORT: All 5 reference modes (none, single, start, end, both)")
        print("🔧 REFERENCE STRENGTH: Adjust guidance scale based on reference strength (0.1-1.0)")
        print("🔧 PARAMETER FIX: Consistent parameter names (job_id, assets) with edge function")
        print(f"📋 Supporting ALL 8 job types with 1.3B tasks: {list(self.job_configs.keys())}")
        print(f"📁 WAN 1.3B Model Path: {self.model_path}")
        print(f"🤖 Qwen Base Model Path: {self.qwen_model_path}")
        print("📊 Status: Fixed for 1.3B model + Reference Frames + Reference Strength ✅")
        self.log_gpu_memory()

    def adjust_guidance_for_reference_strength(self, base_guide_scale, reference_strength):
        """
        Adjust sample_guide_scale based on reference strength to control reference influence
        
        Args:
            base_guide_scale (float): Base guidance scale from job config
            reference_strength (float): Reference strength (0.1-1.0)
            
        Returns:
            float: Adjusted guidance scale
        """
        if reference_strength is None:
            return base_guide_scale
            
        # Reference strength affects how much the reference frame influences generation
        # Higher reference strength = higher guidance scale = stronger reference influence
        # Base range: 6.5-7.5, adjusted range: 5.0-9.0
        
        # Calculate adjustment factor
        # 0.1 strength = minimal influence (5.0 guidance)
        # 1.0 strength = maximum influence (9.0 guidance)
        min_guidance = 5.0
        max_guidance = 9.0
        
        # Linear interpolation between min and max guidance
        adjusted_guidance = min_guidance + (max_guidance - min_guidance) * reference_strength
        
        print(f"🎯 Reference strength adjustment:")
        print(f"   Base guidance scale: {base_guide_scale}")
        print(f"   Reference strength: {reference_strength}")
        print(f"   Adjusted guidance scale: {adjusted_guidance:.2f}")
        
        return adjusted_guidance

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
        """Generate video with start and/or end reference frames using FLF2V task"""
        print(f"🎬 Generating video with reference frames using FLF2V task")
        
        if start_reference and end_reference:
            print("🖼️ Using both start and end reference frames with FLF2V-14B")
            return self.generate_flf2v_video(prompt, start_reference, end_reference, job_type)
        elif start_reference:
            print("🖼️ Using start reference frame only with FLF2V-14B")
            return self.generate_flf2v_video(prompt, start_reference, None, job_type)
        elif end_reference:
            print("🖼️ Using end reference frame only with FLF2V-14B")
            return self.generate_flf2v_video(prompt, None, end_reference, job_type)
        else:
            print("⚠️ No reference frames provided, falling back to T2V task")
            return self.generate_t2v_video(prompt, job_type)

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

    def generate_video_with_reference_frame(self, prompt, reference_image, job_type, reference_strength=0.85):
        """Generate video with single reference frame using WAN 1.3B (I2V-style)"""
        print(f"🎬 Generating video with single reference frame using WAN 1.3B")
        
        if job_type not in self.job_configs:
            raise Exception(f"Unsupported job type: {job_type}")
            
        config = self.job_configs[job_type].copy()
        
        # Adjust guidance scale based on reference strength
        base_guide_scale = config['sample_guide_scale']
        adjusted_guide_scale = self.adjust_guidance_for_reference_strength(base_guide_scale, reference_strength)
        config['sample_guide_scale'] = adjusted_guide_scale
        
        # Create output path with proper extension
        timestamp = int(time.time())
        file_extension = config['file_extension']
        output_filename = f"wan_single_ref_output_{timestamp}.{file_extension}"
        temp_output_path = f"/tmp/{output_filename}"
        
        # Preprocess reference image
        processed_image = self.preprocess_reference_image(reference_image)
        
        # Save reference image to temp file
        ref_filename = f"wan_single_ref_{timestamp}.png"
        ref_path = self.save_reference_image(processed_image, ref_filename)
        
        print(f"🎯 Single Reference Output path: {temp_output_path}")
        print(f"📄 Expected file type: {config['content_type']} (.{file_extension})")
        print(f"🔧 FRAME COUNT: {config['frame_num']} frames for {config['content_type']}")
        print(f"🖼️ Single reference image: {ref_path}")
        
        try:
            # Change to WAN code directory
            original_cwd = os.getcwd()
            os.chdir(self.wan_code_path)
            
            # Build WAN command for single reference frame generation
            wan_generate_path = "/workspace/ourvidz-worker/wan_generate.py"
            
            # Use t2v-1.3B task with --image parameter for single reference frame
            cmd = [
                "python", wan_generate_path,
                "--task", "t2v-1.3B",                        # ✅ CORRECT: Use t2v-1.3B with --image for single reference
                "--ckpt_dir", self.model_path,
                "--offload_model", "True",
                "--size", config['size'],
                "--sample_steps", str(config['sample_steps']),
                "--sample_guide_scale", str(config['sample_guide_scale']),
                "--sample_solver", config.get('sample_solver', 'unipc'),
                "--sample_shift", str(config.get('sample_shift', 5.0)),
                "--frame_num", str(config['frame_num']),
                "--prompt", prompt,
                "--image", ref_path,                         # ✅ Single reference image for t2v-1.3B generation
                "--save_file", temp_output_path
            ]
            
            print(f"🎬 WAN 1.3B T2V command with single reference (t2v-1.3B + --image): {' '.join(cmd)}")
            
            # Configure environment
            env = self.setup_environment()
            
            print(f"🎬 WAN 1.3B T2V generation with single reference (t2v-1.3B + --image): {job_type}")
            print(f"📝 Prompt: {prompt[:100]}...")
            print(f"🔧 Config: {config['sample_steps']} steps, {config['frame_num']} frames, {config['size']}")
            print(f"💾 Output: {temp_output_path}")
            print(f"📁 Working dir: {self.wan_code_path}")
            
            # Execute single reference generation
            generation_start = time.time()
            timeout_seconds = 500 if config['content_type'] == 'video' else 180
            
            print(f"⏰ Starting WAN 1.3B T2V subprocess (t2v-1.3B + --image) with {timeout_seconds}s timeout")
            print(f"🚀 T2V generation with single reference started at {time.strftime('%H:%M:%S')}")

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
                
                print(f"✅ WAN 1.3B T2V subprocess (t2v-1.3B + --image) completed in {generation_time:.1f}s")
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
                    print(f"🔍 Checking T2V output file (t2v-1.3B + --image): {temp_output_path}")
                    
                    if os.path.exists(temp_output_path):
                        file_size = os.path.getsize(temp_output_path)
                        print(f"✅ T2V output file found (t2v-1.3B + --image): {file_size / 1024**2:.2f}MB")
                        
                        is_valid, validation_msg = self.validate_output_file(temp_output_path, config['content_type'])
                        if is_valid:
                            print(f"✅ T2V file validation passed (t2v-1.3B + --image): {validation_msg}")
                            return temp_output_path
                        else:
                            print(f"❌ T2V file validation failed (t2v-1.3B + --image): {validation_msg}")
                            raise Exception(f"T2V generated file validation failed: {validation_msg}")
                    else:
                        print(f"❌ T2V output file not found (t2v-1.3B + --image): {temp_output_path}")
                        raise Exception("No valid T2V output file generated")
                        
                else:
                    print(f"❌ T2V failed with return code (t2v-1.3B + --image): {result.returncode}")
                    error_details = []
                    if result.stderr:
                        error_details.append(f"STDERR: {result.stderr[-300:]}")
                    if result.stdout:
                        error_details.append(f"STDOUT: {result.stdout[-300:]}")
                    
                    error_message = " | ".join(error_details) if error_details else "No error output captured"
                    raise Exception(f"T2V generation failed (t2v-1.3B + --image, code {result.returncode}): {error_message}")
                    
            except subprocess.TimeoutExpired:
                os.chdir(original_cwd)
                print(f"❌ T2V generation timed out after {timeout_seconds}s (t2v-1.3B + --image)")
                raise Exception(f"T2V generation timed out after {timeout_seconds} seconds (t2v-1.3B + --image)")
                
            except Exception as e:
                os.chdir(original_cwd)
                print(f"❌ T2V subprocess error (t2v-1.3B + --image): {e}")
                raise
                
        except Exception as e:
            if os.getcwd() != original_cwd:
                os.chdir(original_cwd)
            print(f"❌ T2V generation error (t2v-1.3B + --image): {e}")
            raise
        finally:
            # Cleanup reference file
            try:
                os.unlink(ref_path)
            except:
                pass

    def generate_video_with_start_frame(self, prompt, start_reference_image, job_type, reference_strength=0.85):
        """Generate video with start frame reference using WAN 1.3B"""
        print(f"🎬 Generating video with start frame reference using WAN 1.3B")
        
        if job_type not in self.job_configs:
            raise Exception(f"Unsupported job type: {job_type}")
            
        config = self.job_configs[job_type].copy()
        
        # Adjust guidance scale based on reference strength
        base_guide_scale = config['sample_guide_scale']
        adjusted_guide_scale = self.adjust_guidance_for_reference_strength(base_guide_scale, reference_strength)
        config['sample_guide_scale'] = adjusted_guide_scale
        
        # Create output path with proper extension
        timestamp = int(time.time())
        file_extension = config['file_extension']
        output_filename = f"wan_start_frame_output_{timestamp}.{file_extension}"
        temp_output_path = f"/tmp/{output_filename}"
        
        # Preprocess reference image
        processed_image = self.preprocess_reference_image(start_reference_image)
        
        # Save reference image to temp file
        ref_filename = f"wan_start_frame_{timestamp}.png"
        ref_path = self.save_reference_image(processed_image, ref_filename)
        
        print(f"🎯 Start Frame Output path: {temp_output_path}")
        print(f"📄 Expected file type: {config['content_type']} (.{file_extension})")
        print(f"🔧 FRAME COUNT: {config['frame_num']} frames for {config['content_type']}")
        print(f"🖼️ Start frame reference: {ref_path}")
        
        try:
            # Change to WAN code directory
            original_cwd = os.getcwd()
            os.chdir(self.wan_code_path)
            
            # Build WAN command for start frame reference generation
            wan_generate_path = "/workspace/ourvidz-worker/wan_generate.py"
            
            # Use t2v-1.3B task with --first_frame parameter for start frame reference
            cmd = [
                "python", wan_generate_path,
                "--task", "t2v-1.3B",                        # ✅ CORRECT: Use t2v-1.3B with --first_frame for start frame
                "--ckpt_dir", self.model_path,
                "--offload_model", "True",
                "--size", config['size'],
                "--sample_steps", str(config['sample_steps']),
                "--sample_guide_scale", str(config['sample_guide_scale']),
                "--sample_solver", config.get('sample_solver', 'unipc'),
                "--sample_shift", str(config.get('sample_shift', 5.0)),
                "--frame_num", str(config['frame_num']),
                "--prompt", prompt,
                "--first_frame", ref_path,                   # ✅ Start frame reference for t2v-1.3B generation
                "--save_file", temp_output_path
            ]
            
            print(f"🎬 WAN 1.3B T2V command with start frame (t2v-1.3B + --first_frame): {' '.join(cmd)}")
            
            # Configure environment
            env = self.setup_environment()
            
            print(f"🎬 WAN 1.3B T2V generation with start frame (t2v-1.3B + --first_frame): {job_type}")
            print(f"📝 Prompt: {prompt[:100]}...")
            print(f"🔧 Config: {config['sample_steps']} steps, {config['frame_num']} frames, {config['size']}")
            print(f"💾 Output: {temp_output_path}")
            print(f"📁 Working dir: {self.wan_code_path}")
            
            # Execute start frame generation
            generation_start = time.time()
            timeout_seconds = 500 if config['content_type'] == 'video' else 180
            
            print(f"⏰ Starting WAN 1.3B T2V subprocess (t2v-1.3B + --first_frame) with {timeout_seconds}s timeout")
            print(f"🚀 T2V generation with start frame started at {time.strftime('%H:%M:%S')}")

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
                
                print(f"✅ WAN 1.3B T2V subprocess (t2v-1.3B + --first_frame) completed in {generation_time:.1f}s")
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
                    print(f"🔍 Checking T2V output file (t2v-1.3B + --first_frame): {temp_output_path}")
                    
                    if os.path.exists(temp_output_path):
                        file_size = os.path.getsize(temp_output_path)
                        print(f"✅ T2V output file found (t2v-1.3B + --first_frame): {file_size / 1024**2:.2f}MB")
                        
                        is_valid, validation_msg = self.validate_output_file(temp_output_path, config['content_type'])
                        if is_valid:
                            print(f"✅ T2V file validation passed (t2v-1.3B + --first_frame): {validation_msg}")
                            return temp_output_path
                        else:
                            print(f"❌ T2V file validation failed (t2v-1.3B + --first_frame): {validation_msg}")
                            raise Exception(f"T2V generated file validation failed: {validation_msg}")
                    else:
                        print(f"❌ T2V output file not found (t2v-1.3B + --first_frame): {temp_output_path}")
                        raise Exception("No valid T2V output file generated")
                        
                else:
                    print(f"❌ T2V failed with return code (t2v-1.3B + --first_frame): {result.returncode}")
                    error_details = []
                    if result.stderr:
                        error_details.append(f"STDERR: {result.stderr[-300:]}")
                    if result.stdout:
                        error_details.append(f"STDOUT: {result.stdout[-300:]}")
                    
                    error_message = " | ".join(error_details) if error_details else "No error output captured"
                    raise Exception(f"T2V generation failed (t2v-1.3B + --first_frame, code {result.returncode}): {error_message}")
                    
            except subprocess.TimeoutExpired:
                os.chdir(original_cwd)
                print(f"❌ T2V generation timed out after {timeout_seconds}s (t2v-1.3B + --first_frame)")
                raise Exception(f"T2V generation timed out after {timeout_seconds} seconds (t2v-1.3B + --first_frame)")
                
            except Exception as e:
                os.chdir(original_cwd)
                print(f"❌ T2V subprocess error (t2v-1.3B + --first_frame): {e}")
                raise
                
        except Exception as e:
            if os.getcwd() != original_cwd:
                os.chdir(original_cwd)
            print(f"❌ T2V generation error (t2v-1.3B + --first_frame): {e}")
            raise
        finally:
            # Cleanup reference file
            try:
                os.unlink(ref_path)
            except:
                pass

    def generate_video_with_end_frame(self, prompt, end_reference_image, job_type, reference_strength=0.85):
        """Generate video with end frame reference using WAN 1.3B"""
        print(f"🎬 Generating video with end frame reference using WAN 1.3B")
        
        if job_type not in self.job_configs:
            raise Exception(f"Unsupported job type: {job_type}")
            
        config = self.job_configs[job_type].copy()
        
        # Adjust guidance scale based on reference strength
        base_guide_scale = config['sample_guide_scale']
        adjusted_guide_scale = self.adjust_guidance_for_reference_strength(base_guide_scale, reference_strength)
        config['sample_guide_scale'] = adjusted_guide_scale
        
        # Create output path with proper extension
        timestamp = int(time.time())
        file_extension = config['file_extension']
        output_filename = f"wan_end_frame_output_{timestamp}.{file_extension}"
        temp_output_path = f"/tmp/{output_filename}"
        
        # Preprocess reference image
        processed_image = self.preprocess_reference_image(end_reference_image)
        
        # Save reference image to temp file
        ref_filename = f"wan_end_frame_{timestamp}.png"
        ref_path = self.save_reference_image(processed_image, ref_filename)
        
        print(f"🎯 End Frame Output path: {temp_output_path}")
        print(f"📄 Expected file type: {config['content_type']} (.{file_extension})")
        print(f"🔧 FRAME COUNT: {config['frame_num']} frames for {config['content_type']}")
        print(f"🖼️ End frame reference: {ref_path}")
        
        try:
            # Change to WAN code directory
            original_cwd = os.getcwd()
            os.chdir(self.wan_code_path)
            
            # Build WAN command for end frame reference generation
            wan_generate_path = "/workspace/ourvidz-worker/wan_generate.py"
            
            # Use t2v-1.3B task with --last_frame parameter for end frame reference
            cmd = [
                "python", wan_generate_path,
                "--task", "t2v-1.3B",                        # ✅ CORRECT: Use t2v-1.3B with --last_frame for end frame
                "--ckpt_dir", self.model_path,
                "--offload_model", "True",
                "--size", config['size'],
                "--sample_steps", str(config['sample_steps']),
                "--sample_guide_scale", str(config['sample_guide_scale']),
                "--sample_solver", config.get('sample_solver', 'unipc'),
                "--sample_shift", str(config.get('sample_shift', 5.0)),
                "--frame_num", str(config['frame_num']),
                "--prompt", prompt,
                "--last_frame", ref_path,                    # ✅ End frame reference for t2v-1.3B generation
                "--save_file", temp_output_path
            ]
            
            print(f"🎬 WAN 1.3B T2V command with end frame (t2v-1.3B + --last_frame): {' '.join(cmd)}")
            
            # Configure environment
            env = self.setup_environment()
            
            print(f"🎬 WAN 1.3B T2V generation with end frame (t2v-1.3B + --last_frame): {job_type}")
            print(f"📝 Prompt: {prompt[:100]}...")
            print(f"🔧 Config: {config['sample_steps']} steps, {config['frame_num']} frames, {config['size']}")
            print(f"💾 Output: {temp_output_path}")
            print(f"📁 Working dir: {self.wan_code_path}")
            
            # Execute end frame generation
            generation_start = time.time()
            timeout_seconds = 500 if config['content_type'] == 'video' else 180
            
            print(f"⏰ Starting WAN 1.3B T2V subprocess (t2v-1.3B + --last_frame) with {timeout_seconds}s timeout")
            print(f"🚀 T2V generation with end frame started at {time.strftime('%H:%M:%S')}")

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
                
                print(f"✅ WAN 1.3B T2V subprocess (t2v-1.3B + --last_frame) completed in {generation_time:.1f}s")
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
                    print(f"🔍 Checking T2V output file (t2v-1.3B + --last_frame): {temp_output_path}")
                    
                    if os.path.exists(temp_output_path):
                        file_size = os.path.getsize(temp_output_path)
                        print(f"✅ T2V output file found (t2v-1.3B + --last_frame): {file_size / 1024**2:.2f}MB")
                        
                        is_valid, validation_msg = self.validate_output_file(temp_output_path, config['content_type'])
                        if is_valid:
                            print(f"✅ T2V file validation passed (t2v-1.3B + --last_frame): {validation_msg}")
                            return temp_output_path
                        else:
                            print(f"❌ T2V file validation failed (t2v-1.3B + --last_frame): {validation_msg}")
                            raise Exception(f"T2V generated file validation failed: {validation_msg}")
                    else:
                        print(f"❌ T2V output file not found (t2v-1.3B + --last_frame): {temp_output_path}")
                        raise Exception("No valid T2V output file generated")
                        
                else:
                    print(f"❌ T2V failed with return code (t2v-1.3B + --last_frame): {result.returncode}")
                    error_details = []
                    if result.stderr:
                        error_details.append(f"STDERR: {result.stderr[-300:]}")
                    if result.stdout:
                        error_details.append(f"STDOUT: {result.stdout[-300:]}")
                    
                    error_message = " | ".join(error_details) if error_details else "No error output captured"
                    raise Exception(f"T2V generation failed (t2v-1.3B + --last_frame, code {result.returncode}): {error_message}")
                    
            except subprocess.TimeoutExpired:
                os.chdir(original_cwd)
                print(f"❌ T2V generation timed out after {timeout_seconds}s (t2v-1.3B + --last_frame)")
                raise Exception(f"T2V generation timed out after {timeout_seconds} seconds (t2v-1.3B + --last_frame)")
                
            except Exception as e:
                os.chdir(original_cwd)
                print(f"❌ T2V subprocess error (t2v-1.3B + --last_frame): {e}")
                raise
                
        except Exception as e:
            if os.getcwd() != original_cwd:
                os.chdir(original_cwd)
            print(f"❌ T2V generation error (t2v-1.3B + --last_frame): {e}")
            raise
        finally:
            # Cleanup reference file
            try:
                os.unlink(ref_path)
            except:
                pass

    def generate_video_with_both_frames(self, prompt, start_reference_image, end_reference_image, job_type, reference_strength=0.85):
        """Generate video with both start and end frame references using WAN 1.3B"""
        print(f"🎬 Generating video with both start and end frame references using WAN 1.3B")
        
        if job_type not in self.job_configs:
            raise Exception(f"Unsupported job type: {job_type}")
            
        config = self.job_configs[job_type].copy()
        
        # Adjust guidance scale based on reference strength
        base_guide_scale = config['sample_guide_scale']
        adjusted_guide_scale = self.adjust_guidance_for_reference_strength(base_guide_scale, reference_strength)
        config['sample_guide_scale'] = adjusted_guide_scale
        
        # Create output path with proper extension
        timestamp = int(time.time())
        file_extension = config['file_extension']
        output_filename = f"wan_both_frames_output_{timestamp}.{file_extension}"
        temp_output_path = f"/tmp/{output_filename}"
        
        # Preprocess reference images
        processed_start = self.preprocess_reference_image(start_reference_image)
        processed_end = self.preprocess_reference_image(end_reference_image)
        
        # Save reference images to temp files
        start_ref_filename = f"wan_start_frame_{timestamp}.png"
        end_ref_filename = f"wan_end_frame_{timestamp}.png"
        start_ref_path = self.save_reference_image(processed_start, start_ref_filename)
        end_ref_path = self.save_reference_image(processed_end, end_ref_filename)
        
        print(f"🎯 Both Frames Output path: {temp_output_path}")
        print(f"📄 Expected file type: {config['content_type']} (.{file_extension})")
        print(f"🔧 FRAME COUNT: {config['frame_num']} frames for {config['content_type']}")
        print(f"🖼️ Start frame reference: {start_ref_path}")
        print(f"🖼️ End frame reference: {end_ref_path}")
        
        try:
            # Change to WAN code directory
            original_cwd = os.getcwd()
            os.chdir(self.wan_code_path)
            
            # Build WAN command for both frames reference generation
            wan_generate_path = "/workspace/ourvidz-worker/wan_generate.py"
            
            # Use t2v-1.3B task with --first_frame and --last_frame parameters for both frames
            cmd = [
                "python", wan_generate_path,
                "--task", "t2v-1.3B",                        # ✅ CORRECT: Use t2v-1.3B with both frame parameters
                "--ckpt_dir", self.model_path,
                "--offload_model", "True",
                "--size", config['size'],
                "--sample_steps", str(config['sample_steps']),
                "--sample_guide_scale", str(config['sample_guide_scale']),
                "--sample_solver", config.get('sample_solver', 'unipc'),
                "--sample_shift", str(config.get('sample_shift', 5.0)),
                "--frame_num", str(config['frame_num']),
                "--prompt", prompt,
                "--first_frame", start_ref_path,             # ✅ Start frame reference for t2v-1.3B generation
                "--last_frame", end_ref_path,                # ✅ End frame reference for t2v-1.3B generation
                "--save_file", temp_output_path
            ]
            
            print(f"🎬 WAN 1.3B T2V command with both frames (t2v-1.3B + --first_frame + --last_frame): {' '.join(cmd)}")
            
            # Configure environment
            env = self.setup_environment()
            
            print(f"🎬 WAN 1.3B T2V generation with both frames (t2v-1.3B + --first_frame + --last_frame): {job_type}")
            print(f"📝 Prompt: {prompt[:100]}...")
            print(f"🔧 Config: {config['sample_steps']} steps, {config['frame_num']} frames, {config['size']}")
            print(f"💾 Output: {temp_output_path}")
            print(f"📁 Working dir: {self.wan_code_path}")
            
            # Execute both frames generation
            generation_start = time.time()
            timeout_seconds = 500 if config['content_type'] == 'video' else 180
            
            print(f"⏰ Starting WAN 1.3B T2V subprocess (t2v-1.3B + --first_frame + --last_frame) with {timeout_seconds}s timeout")
            print(f"🚀 T2V generation with both frames started at {time.strftime('%H:%M:%S')}")

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
                
                print(f"✅ WAN 1.3B T2V subprocess (t2v-1.3B + --first_frame + --last_frame) completed in {generation_time:.1f}s")
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
                    print(f"🔍 Checking T2V output file (t2v-1.3B + --first_frame + --last_frame): {temp_output_path}")
                    
                    if os.path.exists(temp_output_path):
                        file_size = os.path.getsize(temp_output_path)
                        print(f"✅ T2V output file found (t2v-1.3B + --first_frame + --last_frame): {file_size / 1024**2:.2f}MB")
                        
                        is_valid, validation_msg = self.validate_output_file(temp_output_path, config['content_type'])
                        if is_valid:
                            print(f"✅ T2V file validation passed (t2v-1.3B + --first_frame + --last_frame): {validation_msg}")
                            return temp_output_path
                        else:
                            print(f"❌ T2V file validation failed (t2v-1.3B + --first_frame + --last_frame): {validation_msg}")
                            raise Exception(f"T2V generated file validation failed: {validation_msg}")
                    else:
                        print(f"❌ T2V output file not found (t2v-1.3B + --first_frame + --last_frame): {temp_output_path}")
                        raise Exception("No valid T2V output file generated")
                        
                else:
                    print(f"❌ T2V failed with return code (t2v-1.3B + --first_frame + --last_frame): {result.returncode}")
                    error_details = []
                    if result.stderr:
                        error_details.append(f"STDERR: {result.stderr[-300:]}")
                    if result.stdout:
                        error_details.append(f"STDOUT: {result.stdout[-300:]}")
                    
                    error_message = " | ".join(error_details) if error_details else "No error output captured"
                    raise Exception(f"T2V generation failed (t2v-1.3B + --first_frame + --last_frame, code {result.returncode}): {error_message}")
                    
            except subprocess.TimeoutExpired:
                os.chdir(original_cwd)
                print(f"❌ T2V generation timed out after {timeout_seconds}s (t2v-1.3B + --first_frame + --last_frame)")
                raise Exception(f"T2V generation timed out after {timeout_seconds} seconds (t2v-1.3B + --first_frame + --last_frame)")
                
            except Exception as e:
                os.chdir(original_cwd)
                print(f"❌ T2V subprocess error (t2v-1.3B + --first_frame + --last_frame): {e}")
                raise
                
        except Exception as e:
            if os.getcwd() != original_cwd:
                os.chdir(original_cwd)
            print(f"❌ T2V generation error (t2v-1.3B + --first_frame + --last_frame): {e}")
            raise
        finally:
            # Cleanup reference files
            try:
                os.unlink(start_ref_path)
                os.unlink(end_ref_path)
            except:
                pass

    def generate_standard_content(self, prompt, job_type):
        """Generate standard content without reference frames using WAN 1.3B"""
        print("🎬 Generating standard content using WAN 1.3B T2V")
        return self.generate_content(prompt, job_type)
        
        if job_type not in self.job_configs:
            raise Exception(f"Unsupported job type: {job_type}")
            
        config = self.job_configs[job_type]
        
        # Create output path with proper extension
        timestamp = int(time.time())
        file_extension = config['file_extension']
        output_filename = f"wan_t2v_output_{timestamp}.{file_extension}"
        temp_output_path = f"/tmp/{output_filename}"
        
        print(f"🎯 T2V Output path: {temp_output_path}")
        print(f"📄 Expected file type: {config['content_type']} (.{file_extension})")
        print(f"🔧 FRAME COUNT: {config['frame_num']} frames for {config['content_type']}")
        
        try:
            # Change to WAN code directory
            original_cwd = os.getcwd()
            os.chdir(self.wan_code_path)
            
            # Build WAN command for T2V task
            # CRITICAL: Use correct path to wan_generate.py in worker repository
            wan_generate_path = "/workspace/ourvidz-worker/wan_generate.py"
            
            cmd = [
                "python", wan_generate_path,  # ✅ UPDATED: Use full path to wan_generate.py
                "--task", "t2v-14B",  # ✅ UPDATED: Use T2V-14B task for standard generation
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
            
            print(f"🎬 T2V command: {' '.join(cmd)}")
            
            # Configure environment
            env = self.setup_environment()
            
            print(f"🎬 T2V generation: {job_type}")
            print(f"📝 Prompt: {prompt[:100]}...")
            print(f"🔧 Config: {config['sample_steps']} steps, {config['frame_num']} frames, {config['size']}")
            print(f"💾 Output: {temp_output_path}")
            print(f"📁 Working dir: {self.wan_code_path}")
            
            # Execute T2V generation
            generation_start = time.time()
            timeout_seconds = 500 if config['content_type'] == 'video' else 180
            
            print(f"⏰ Starting T2V subprocess with {timeout_seconds}s timeout")
            print(f"🚀 T2V generation started at {time.strftime('%H:%M:%S')}")

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
                
                print(f"✅ T2V subprocess completed in {generation_time:.1f}s")
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
                    print(f"🔍 Checking T2V output file: {temp_output_path}")
                    
                    if os.path.exists(temp_output_path):
                        file_size = os.path.getsize(temp_output_path)
                        print(f"✅ T2V output file found: {file_size / 1024**2:.2f}MB")
                        
                        is_valid, validation_msg = self.validate_output_file(temp_output_path, config['content_type'])
                        if is_valid:
                            print(f"✅ T2V file validation passed: {validation_msg}")
                            return temp_output_path
                        else:
                            print(f"❌ T2V file validation failed: {validation_msg}")
                            raise Exception(f"T2V generated file validation failed: {validation_msg}")
                    else:
                        print(f"❌ T2V output file not found: {temp_output_path}")
                        raise Exception("No valid T2V output file generated")
                        
                else:
                    print(f"❌ T2V failed with return code: {result.returncode}")
                    error_details = []
                    if result.stderr:
                        error_details.append(f"STDERR: {result.stderr[-300:]}")
                    if result.stdout:
                        error_details.append(f"STDOUT: {result.stdout[-300:]}")
                    
                    error_message = " | ".join(error_details) if error_details else "No error output captured"
                    raise Exception(f"T2V generation failed (code {result.returncode}): {error_message}")
                    
            except subprocess.TimeoutExpired:
                os.chdir(original_cwd)
                print(f"❌ T2V generation timed out after {timeout_seconds}s")
                raise Exception(f"T2V generation timed out after {timeout_seconds} seconds")
                
            except Exception as e:
                os.chdir(original_cwd)
                print(f"❌ T2V subprocess error: {e}")
                raise
                
        except Exception as e:
            if os.getcwd() != original_cwd:
                os.chdir(original_cwd)
            print(f"❌ T2V generation error: {e}")
            raise

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
            
            # Build WAN command with FLF2V task for reference frames
            # CRITICAL: Use correct path to wan_generate.py in worker repository
            wan_generate_path = "/workspace/ourvidz-worker/wan_generate.py"
            
            cmd = [
                "python", wan_generate_path,  # ✅ UPDATED: Use full path to wan_generate.py
                "--task", "flf2v-14B",  # ✅ UPDATED: Use FLF2V task for reference frames
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
            
            # Add reference frame parameters for FLF2V task
            if start_ref_path:
                cmd.extend(["--first_frame", start_ref_path])  # ✅ UPDATED: Use --first_frame instead of --start_frame
                print(f"🖼️ Start reference frame: {start_ref_path}")
            
            if end_ref_path:
                cmd.extend(["--last_frame", end_ref_path])  # ✅ UPDATED: Use --last_frame instead of --end_frame
                print(f"🖼️ End reference frame: {end_ref_path}")
            
            # ✅ REMOVED: --reference_strength parameter (not needed for FLF2V)
            print(f"🎬 FLF2V command: {' '.join(cmd)}")
            
            # Configure environment
            env = self.setup_environment()
            
            print(f"🎬 FLF2V generation: {job_type}")
            print(f"📝 Prompt: {prompt[:100]}...")
            print(f"🔧 Config: {config['sample_steps']} steps, {config['frame_num']} frames, {config['size']}")
            print(f"💾 Output: {temp_output_path}")
            print(f"📁 Working dir: {self.wan_code_path}")
            
            # Execute FLF2V generation
            generation_start = time.time()
            timeout_seconds = 500 if config['content_type'] == 'video' else 180
            
            print(f"⏰ Starting FLF2V subprocess with {timeout_seconds}s timeout")
            print(f"🚀 FLF2V generation started at {time.strftime('%H:%M:%S')}")

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
                
                print(f"✅ FLF2V subprocess completed in {generation_time:.1f}s")
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
                    print(f"❌ FLF2V failed with return code: {result.returncode}")
                    error_details = []
                    if result.stderr:
                        error_details.append(f"STDERR: {result.stderr[-300:]}")
                    if result.stdout:
                        error_details.append(f"STDOUT: {result.stdout[-300:]}")
                    
                    error_message = " | ".join(error_details) if error_details else "No error output captured"
                    raise Exception(f"FLF2V generation failed (code {result.returncode}): {error_message}")
                    
            except subprocess.TimeoutExpired:
                os.chdir(original_cwd)
                print(f"❌ FLF2V generation timed out after {timeout_seconds}s")
                raise Exception(f"FLF2V generation timed out after {timeout_seconds} seconds")
                
            except Exception as e:
                os.chdir(original_cwd)
                print(f"❌ FLF2V subprocess error: {e}")
                raise
                
        except Exception as e:
            if os.getcwd() != original_cwd:
                os.chdir(original_cwd)
            print(f"❌ FLF2V generation error: {e}")
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
        wan_code_path = '/workspace/Wan2.1'  # Add WAN code directory to Python path
        
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            new_pythonpath = f"{wan_code_path}:{python_deps_path}:{current_pythonpath}"
        else:
            new_pythonpath = f"{wan_code_path}:{python_deps_path}"
        
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

    def load_qwen_instruct_model(self):
        """Load Qwen 2.5-7B Instruct model for chat/conversational enhancement"""
        if hasattr(self, 'qwen_instruct_model') and self.qwen_instruct_model is not None:
            return True
            
        # Use the verified Instruct model path from workspace analysis
        instruct_model_path = "/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct"
        
        if not os.path.exists(instruct_model_path):
            print(f"⚠️ Qwen Instruct model not found at {instruct_model_path}")
            return False
            
        try:
            print("💬 Loading Qwen 2.5-7B Instruct model for chat enhancement...")
            enhancement_start = time.time()
            
            # Set timeout for model loading
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)  # 2 minute timeout for model loading
            
            print(f"🔄 Loading Qwen Instruct model from {instruct_model_path}")
            
            # Load chat model components
            self.qwen_instruct_tokenizer = AutoTokenizer.from_pretrained(
                instruct_model_path,
                trust_remote_code=True,
                local_files_only=True  # Use local model only
            )
            
            self.qwen_instruct_model = AutoModelForCausalLM.from_pretrained(
                instruct_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True  # Use local model only
            )
            
            signal.alarm(0)
            
            load_time = time.time() - enhancement_start
            print(f"✅ Qwen Instruct model loaded successfully in {load_time:.1f}s")
            print(f"✅ Model type: INSTRUCT (conversational)")
            self.log_gpu_memory()
            return True
            
        except TimeoutException:
            signal.alarm(0)
            print(f"❌ Qwen Instruct model loading timed out after 120s")
            return False
        except Exception as e:
            signal.alarm(0)
            print(f"❌ Failed to load Qwen Instruct model: {e}")
            return False

    def unload_qwen_instruct_model(self):
        """Free Qwen Instruct memory"""
        if hasattr(self, 'qwen_instruct_model') and self.qwen_instruct_model is not None:
            print("🗑️ Unloading Qwen Instruct model...")
            del self.qwen_instruct_model
            del self.qwen_instruct_tokenizer
            self.qwen_instruct_model = None
            self.qwen_instruct_tokenizer = None
            torch.cuda.empty_cache()
            print("✅ Qwen Instruct model unloaded")
            self.log_gpu_memory()

    def enhance_prompt_with_chat(self, original_prompt, session_id=None, conversation_context=None):
        """Enhanced prompt generation using Instruct model with conversation memory"""
        enhancement_start = time.time()
        print(f"💬 Enhancing prompt with Instruct model: {original_prompt[:50]}...")
        
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.enhancement_timeout)
            
            if not self.load_qwen_instruct_model():
                print("⚠️ Instruct model not available, falling back to Base model")
                return self.enhance_prompt_with_timeout(original_prompt)
            
            # Build conversation for instruct model
            system_prompt = """You are an expert AI prompt engineer specializing in cinematic and adult content generation.

Your role is to transform simple prompts into detailed, cinematic descriptions while maintaining anatomical accuracy and realism for adult content.

Focus on:
- High-quality visual details and realistic proportions
- Cinematic lighting and professional photography style  
- Specific poses, expressions, and scene composition
- Technical quality like 4K resolution and smooth motion

Always respond with enhanced prompts that are detailed, specific, and optimized for AI generation."""

            # Format conversation for Instruct model
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please enhance this prompt for AI video generation: {original_prompt}"}
            ]
            
            if conversation_context:
                messages.insert(1, {"role": "user", "content": f"Context: {conversation_context}"})
            
            # Apply chat template for Instruct model
            formatted_prompt = self.qwen_instruct_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.qwen_instruct_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.qwen_instruct_model.device)
            
            with torch.no_grad():
                outputs = self.qwen_instruct_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.qwen_instruct_tokenizer.eos_token_id,
                    eos_token_id=self.qwen_instruct_tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            enhanced_text = self.qwen_instruct_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            signal.alarm(0)
            
            if enhanced_text:
                enhancement_time = time.time() - enhancement_start
                print(f"✅ Instruct Enhancement: {enhanced_text[:100]}...")
                print(f"✅ Prompt enhanced with Instruct model in {enhancement_time:.1f}s")
                return enhanced_text
            else:
                print("⚠️ Instruct enhancement empty, falling back to Base")
                return self.enhance_prompt_with_timeout(original_prompt)
                
        except TimeoutException:
            signal.alarm(0)
            print(f"⚠️ Instruct enhancement timed out, falling back to Base")
            return self.enhance_prompt_with_timeout(original_prompt)
        except Exception as e:
            signal.alarm(0)
            print(f"❌ Instruct enhancement failed: {e}, falling back to Base")
            return self.enhance_prompt_with_timeout(original_prompt)
        finally:
            self.unload_qwen_instruct_model()

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

    def enhance_prompt(self, original_prompt, enhancement_type="instruct", session_id=None, conversation_context=None):
        """Enhanced prompt with retry logic and model selection"""
        print(f"🤖 Starting enhancement for: {original_prompt[:50]}... (type: {enhancement_type})")
        
        for attempt in range(self.max_enhancement_attempts):
            try:
                print(f"🔄 Enhancement attempt {attempt + 1}/{self.max_enhancement_attempts}")
                
                # Choose enhancement method based on type
                if enhancement_type == "chat" or enhancement_type == "instruct_chat":
                    enhanced = self.enhance_prompt_with_chat(original_prompt, session_id, conversation_context)
                else:
                    # Use existing Base model enhancement (preserves current functionality)
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
            # CRITICAL: Use correct path to wan_generate.py in worker repository
            wan_generate_path = "/workspace/ourvidz-worker/wan_generate.py"
            
            cmd = [
                "python", wan_generate_path,
                "--task", config['task'],                       # ✅ FIXED: Use task from config (t2v-1.3B)
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
        # Support all reference frame modes: single, start, end, both
        single_reference_url = config.get('image') or metadata.get('reference_image_url')
        start_reference_url = config.get('first_frame') or metadata.get('start_reference_url')
        end_reference_url = config.get('last_frame') or metadata.get('end_reference_url')
        reference_strength = metadata.get('reference_strength', 0.5)
        
        print(f"🔄 Processing job {job_id} ({job_type}) with CONSISTENT PARAMETERS")
        print(f"📝 Original prompt: {original_prompt}")
        print(f"🎯 Video ID: {video_id}")
        print(f"👤 User ID: {user_id}")
        
        # Log reference frame parameters if present (UPDATED API SPEC)
        if single_reference_url or start_reference_url or end_reference_url:
            print(f"🖼️ Reference frame mode: strength {reference_strength}")
            if single_reference_url:
                print(f"📥 Single reference frame URL (image/reference_image_url): {single_reference_url}")
            if start_reference_url:
                print(f"📥 Start reference frame URL (first_frame/start_reference_url): {start_reference_url}")
            if end_reference_url:
                print(f"📥 End reference frame URL (last_frame/end_reference_url): {end_reference_url}")
        
        job_start_time = time.time()
        total_time = 0  # Initialize total_time variable
        
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
            
            # Handle prompt enhancement with chat support
            if final_config['enhance_prompt']:
                print("🤖 Starting prompt enhancement with chat support...")
                
                # Check if chat enhancement was requested in metadata
                enhancement_type = metadata.get('enhancement_type', 'base')
                session_id = metadata.get('session_id')
                conversation_context = metadata.get('conversation_context')
                
                enhanced_prompt = self.enhance_prompt(
                    original_prompt, 
                    enhancement_type=enhancement_type,
                    session_id=session_id,
                    conversation_context=conversation_context
                )
                actual_prompt = enhanced_prompt
                
                if enhanced_prompt != original_prompt:
                    print(f"✅ Prompt successfully enhanced ({enhancement_type})")
                    print(f"📝 Length: {len(original_prompt)} → {len(enhanced_prompt)} chars")
                else:
                    print(f"⚠️ Using original prompt (enhancement failed or timed out)")
            else:
                print("📝 Using original prompt (no enhancement)")
                actual_prompt = original_prompt
            
            # Handle video generation with comprehensive reference frame support for WAN 1.3B
            if final_config['content_type'] == 'video':
                # Determine reference frame mode and route to appropriate generation function
                if single_reference_url and not start_reference_url and not end_reference_url:
                    # Single reference frame mode (I2V-style)
                    print("🎬 Starting single reference video generation (t2v-1.3B + --image)...")
                    print(f"🔧 SINGLE REFERENCE TASK: {final_config['frame_num']} frames for 5-second videos")
                    
                    try:
                        reference_image = self.download_image_from_url(single_reference_url)
                        print(f"✅ Single reference image loaded successfully")
                        
                        # Generate video with single reference frame
                        output_file = self.generate_video_with_reference_frame(
                            actual_prompt, 
                            reference_image, 
                            job_type,
                            reference_strength
                        )
                    except Exception as e:
                        print(f"❌ Failed to load single reference image: {e}")
                        print(f"🔄 Falling back to standard generation")
                        output_file = self.generate_standard_content(actual_prompt, job_type)
                        
                elif start_reference_url and end_reference_url:
                    # Both frames mode (start + end)
                    print("🎬 Starting both frames video generation (t2v-1.3B + --first_frame + --last_frame)...")
                    print(f"🔧 BOTH FRAMES TASK: {final_config['frame_num']} frames for 5-second videos")
                    
                    try:
                        start_reference_image = self.download_image_from_url(start_reference_url)
                        end_reference_image = self.download_image_from_url(end_reference_url)
                        print(f"✅ Both reference images loaded successfully")
                        
                        # Generate video with both start and end frames
                        output_file = self.generate_video_with_both_frames(
                            actual_prompt, 
                            start_reference_image, 
                            end_reference_image, 
                            job_type,
                            reference_strength
                        )
                    except Exception as e:
                        print(f"❌ Failed to load both reference images: {e}")
                        print(f"🔄 Falling back to standard generation")
                        output_file = self.generate_standard_content(actual_prompt, job_type)
                        
                elif start_reference_url and not end_reference_url:
                    # Start frame only mode
                    print("🎬 Starting start frame video generation (t2v-1.3B + --first_frame)...")
                    print(f"🔧 START FRAME TASK: {final_config['frame_num']} frames for 5-second videos")
                    
                    try:
                        start_reference_image = self.download_image_from_url(start_reference_url)
                        print(f"✅ Start reference image loaded successfully")
                        
                        # Generate video with start frame only
                        output_file = self.generate_video_with_start_frame(
                            actual_prompt, 
                            start_reference_image, 
                            job_type,
                            reference_strength
                        )
                    except Exception as e:
                        print(f"❌ Failed to load start reference image: {e}")
                        print(f"🔄 Falling back to standard generation")
                        output_file = self.generate_standard_content(actual_prompt, job_type)
                        
                elif end_reference_url and not start_reference_url:
                    # End frame only mode
                    print("🎬 Starting end frame video generation (t2v-1.3B + --last_frame)...")
                    print(f"🔧 END FRAME TASK: {final_config['frame_num']} frames for 5-second videos")
                    
                    try:
                        end_reference_image = self.download_image_from_url(end_reference_url)
                        print(f"✅ End reference image loaded successfully")
                        
                        # Generate video with end frame only
                        output_file = self.generate_video_with_end_frame(
                            actual_prompt, 
                            end_reference_image, 
                            job_type,
                            reference_strength
                        )
                    except Exception as e:
                        print(f"❌ Failed to load end reference image: {e}")
                        print(f"🔄 Falling back to standard generation")
                        output_file = self.generate_standard_content(actual_prompt, job_type)
                        
                else:
                    # Standard generation (no reference frames)
                    print("🎬 Starting T2V video generation (standard video)...")
                    print(f"🔧 T2V TASK: {final_config['frame_num']} frames for 5-second videos")
                    
                    # Generate video with T2V task (standard generation)
                    output_file = self.generate_standard_content(actual_prompt, job_type)
            else:
                print("🎬 Starting WAN image generation...")
                print(f"🔧 IMAGE GENERATION: {final_config['frame_num']} frames")
                
                # Generate image content
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
            
            # Calculate total time
            total_time = time.time() - job_start_time
            
            # Prepare metadata for callback
            callback_metadata = {
                'generation_time': total_time,
                'job_type': job_type,
                'content_type': final_config['content_type'],
                'frame_num': final_config['frame_num'],
                'wan_task': 't2v-1.3B',  # Always t2v-1.3B for WAN 1.3B model
                'reference_mode': self._determine_reference_mode(single_reference_url, start_reference_url, end_reference_url)
            }
            
            # CONSISTENT: Success callback with standardized parameters and metadata
            self.notify_completion(job_id, 'completed', assets=[relative_path], metadata=callback_metadata)
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
                'wan_task': 't2v-1.3B',  # Always t2v-1.3B for WAN 1.3B model
                'timestamp': time.time()
            }
            
            # CONSISTENT: Failure callback with standardized parameters and metadata
            self.notify_completion(job_id, 'failed', error_message=error_msg, metadata=error_metadata)

    def _determine_reference_mode(self, single_reference_url, start_reference_url, end_reference_url):
        """Determine the reference frame mode based on provided URLs"""
        if single_reference_url and not start_reference_url and not end_reference_url:
            return 'single'
        elif start_reference_url and end_reference_url:
            return 'both'
        elif start_reference_url and not end_reference_url:
            return 'start'
        elif end_reference_url and not start_reference_url:
            return 'end'
        else:
            return 'none'

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
        """Main worker loop for WAN 1.3B model"""
        print("🎬 Enhanced OurVidz WAN Worker - 1.3B MODEL + REFERENCE FRAMES")
        print("🔧 CRITICAL FIX: Using correct t2v-1.3B task for WAN 1.3B model")
        print("🔧 REFERENCE SUPPORT: All 5 reference modes (none, single, start, end, both)")
        print("🔧 PARAMETER FIX: Consistent callback parameters (job_id, status, assets)")
        print("📊 Status: Fixed for WAN 1.3B + Reference Frame Support ✅")
        
        print("🔧 UPSTASH COMPATIBLE: Using non-blocking RPOP for Redis polling")
        print("📋 Supported job types with 1.3B tasks:")
        for job_type, config in self.job_configs.items():
            enhancement = "✨ Enhanced" if config['enhance_prompt'] else "📝 Standard"
            content = "🖼️ Image" if config['content_type'] == 'image' else "🎬 Video"
            print(f"  • {job_type}: {content} ({config['task']}) {enhancement}")
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
                    print(f"\n📬 WAN 1.3B Job #{job_count} received")
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

# Flask server for frontend enhancement API
if FLASK_AVAILABLE:
    # Initialize Flask app
    app = Flask(__name__)

    @app.route('/enhance', methods=['POST'])
    def enhance_endpoint():
        """Frontend enhancement endpoint using real Qwen Base model"""
        try:
            # API Key Authentication
            auth_header = request.headers.get('Authorization')
            expected_key = os.environ.get('WAN_WORKER_API_KEY', 'default_key_123')
            
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({
                    'success': False,
                    'error': 'Missing or invalid Authorization header'
                }), 401
            
            provided_key = auth_header.replace('Bearer ', '')
            if provided_key != expected_key:
                return jsonify({
                    'success': False,
                    'error': 'Invalid API key'
                }), 401
            
            # Validate request
            data = request.json
            if not data or 'prompt' not in data:
                return jsonify({
                    'success': False,
                    'error': 'prompt is required'
                }), 400
            
            original_prompt = data.get('prompt', '')
            model = data.get('model', 'qwen_base')
            enhance_type = data.get('enhance_type', 'natural_language')
            
            if not original_prompt.strip():
                return jsonify({
                    'success': False,
                    'error': 'prompt cannot be empty'
                }), 400
            
            print(f"🎯 Frontend enhancement request: {original_prompt[:50]}...")
            start_time = time.time()
            
            # Use existing Qwen Base enhancement with frontend timeout
            if model == 'qwen_base':
                # Get worker instance (assuming it's available globally)
                worker = globals().get('worker_instance')
                if not worker:
                    return jsonify({
                        'success': False,
                        'error': 'Worker not initialized'
                    }), 500
                
                # Temporarily adjust timeout for frontend responsiveness
                original_timeout = worker.enhancement_timeout
                worker.enhancement_timeout = 25  # Frontend-optimized timeout
                
                try:
                    enhanced_prompt = worker.enhance_prompt_with_timeout(original_prompt)
                    processing_time = time.time() - start_time
                    
                    print(f"✅ Frontend enhancement completed in {processing_time:.1f}s")
                    
                    return jsonify({
                        'success': True,
                        'enhanced_prompt': enhanced_prompt,
                        'original_prompt': original_prompt,
                        'enhancement_source': 'qwen_base',
                        'processing_time': processing_time,
                        'model': model
                    })
                    
                except Exception as e:
                    print(f"❌ Qwen enhancement failed: {e}")
                    return jsonify({
                        'success': False,
                        'error': f'Enhancement failed: {str(e)}',
                        'enhanced_prompt': original_prompt  # Fallback
                    }), 500
                finally:
                    # Restore original timeout
                    worker.enhancement_timeout = original_timeout
            else:
                return jsonify({
                    'success': False,
                    'error': f'Unsupported model: {model}'
                }), 400
                
        except Exception as e:
            print(f"❌ Frontend enhancement endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'enhanced_prompt': data.get('prompt', '') if data else ''
            }), 500

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        worker = globals().get('worker_instance')
        return jsonify({
            'status': 'healthy',
            'qwen_loaded': worker and hasattr(worker, 'qwen_model') and worker.qwen_model is not None,
            'timestamp': time.time(),
            'worker_ready': worker is not None
        })

    def run_flask_server():
        """Run Flask server in a separate thread"""
        try:
            print("🌐 Starting Flask server for frontend enhancement on port 7860...")
            print("🌐 Public endpoint: https://ghy077o4okmjzi-7860.proxy.runpod.net/")
            app.run(host='0.0.0.0', port=7860, debug=False, threaded=True, use_reloader=False)
        except Exception as e:
            print(f"❌ Flask server failed to start: {e}")
else:
    # Placeholder functions when Flask is not available
    def run_flask_server():
        print("⚠️ Flask server not started - Flask not available")
    
    def enhance_endpoint():
        pass
    
    def health_check():
        pass

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
    
    print("✅ All paths validated for 1.3B model")
    print("🔧 FIXED: Using t2v-1.3B task for WAN 1.3B model")
    print("🖼️ REFERENCE: All 5 reference modes (none, single, start, end, both)")
    
    try:
        # Initialize worker
        worker = EnhancedWanWorker()
        
        # Make worker available globally for Flask endpoint
        globals()['worker_instance'] = worker
        
        # Start Flask server in background thread if available
        if FLASK_AVAILABLE:
            flask_thread = threading.Thread(target=run_flask_server, daemon=True)
            flask_thread.start()
            print("✅ Flask server started on port 7860")
            
            # Give Flask a moment to start
            time.sleep(2)
        else:
            print("⚠️ Flask server not started - Flask not available")
        
        # Start main worker loop
        print("🎬 Starting WAN worker main loop...")
        worker.run_with_enhanced_diagnostics()
        
    except KeyboardInterrupt:
        print("🛑 Worker stopped by user")
    except Exception as e:
        print(f"❌ Worker startup failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        print("👋 Enhanced WAN 1.3B Worker shutdown complete")