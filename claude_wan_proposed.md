# wan_worker.py - ENHANCED INTEGRATION for WAN 1.3B Model + REFERENCE FRAMES
# MAJOR ENHANCEMENT: Streamlined reference frame handling with guidance scale adjustment
# INTEGRATION FIX: Unified parameter handling for all reference modes
# PARAMETER CONSISTENCY: Standardized callback format with edge function
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
        """Initialize Enhanced WAN Worker for 1.3B Model with Integrated Reference Support"""
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_code_path = "/workspace/Wan2.1"
        
        # CRITICAL: Set environment variables immediately (VERIFIED FIX)
        os.environ['PYTHONPATH'] = '/workspace/python_deps/lib/python3.11/site-packages'
        os.environ['HF_HOME'] = '/workspace/models/huggingface_cache'
        os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/models/huggingface_cache/hub'
        
        # Qwen 2.5-7B Base model path (no content filtering)
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
        
        # ENHANCED: 1.3B Model Configurations with integrated reference support
        self.job_configs = {
            # Standard job types (no enhancement) - OPTIMIZED for 1.3B model
            'image_fast': {
                'task': 't2v-1.3B',
                'size': '480*832',
                'sample_steps': 25,
                'sample_guide_scale': 6.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 1,
                'enhance_prompt': False,
                'expected_time': 25,
                'content_type': 'image',
                'file_extension': 'png'
            },
            'image_high': {
                'task': 't2v-1.3B',
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
                'task': 't2v-1.3B',
                'size': '480*832',
                'sample_steps': 25,
                'sample_guide_scale': 6.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 83,
                'enhance_prompt': False,
                'expected_time': 135,
                'content_type': 'video',
                'file_extension': 'mp4'
            },
            'video_high': {
                'task': 't2v-1.3B',
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
            
            # Enhanced job types (with Qwen 7B enhancement)
            'image7b_fast_enhanced': {
                'task': 't2v-1.3B',
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
                'task': 't2v-1.3B',
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
                'task': 't2v-1.3B',
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
                'task': 't2v-1.3B',
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
        
        print("🎬 Enhanced OurVidz WAN Worker - INTEGRATED REFERENCE FRAMES")
        print("🔧 ENHANCEMENT: Streamlined reference frame handling with guidance adjustment")
        print("🔧 INTEGRATION: Unified parameter processing for all reference modes")
        print("🔧 CONSISTENCY: Standardized callback format alignment with edge function")
        print(f"📋 Supporting ALL 8 job types with integrated t2v-1.3B tasks: {list(self.job_configs.keys())}")
        print(f"📁 WAN 1.3B Model Path: {self.model_path}")
        print(f"🤖 Qwen Base Model Path: {self.qwen_model_path}")
        print("📊 Status: Enhanced Integration Complete ✅")
        self.log_gpu_memory()

    def adjust_guidance_for_reference_strength(self, base_guide_scale, reference_strength):
        """
        ENHANCED: Adjust sample_guide_scale based on reference strength to control reference influence
        
        Args:
            base_guide_scale (float): Base guidance scale from job config
            reference_strength (float): Reference strength (0.1-1.0)
            
        Returns:
            float: Adjusted guidance scale optimized for WAN 1.3B
        """
        if reference_strength is None:
            return base_guide_scale
            
        # ENHANCED: Reference strength affects how much the reference frame influences generation
        # WAN 1.3B optimized range: 5.0-9.0 for best reference control
        
        # Calculate adjustment factor with non-linear scaling for better control
        min_guidance = 5.0
        max_guidance = 9.0
        
        # Enhanced non-linear interpolation for better reference control
        # Lower values have more dramatic effect
        adjustment_factor = reference_strength ** 0.7  # Slight curve for better low-end control
        adjusted_guidance = min_guidance + (max_guidance - min_guidance) * adjustment_factor
        
        print(f"🎯 ENHANCED Reference strength adjustment:")
        print(f"   Base guidance scale: {base_guide_scale}")
        print(f"   Reference strength: {reference_strength}")
        print(f"   Adjusted guidance scale: {adjusted_guidance:.2f}")
        print(f"   WAN 1.3B optimized range: {min_guidance}-{max_guidance}")
        
        return adjusted_guidance

    def download_and_process_reference_image(self, image_url, filename_prefix):
        """
        ENHANCED: Download and process reference image with optimized preprocessing for WAN 1.3B
        
        Args:
            image_url (str): URL of the reference image
            filename_prefix (str): Prefix for temporary filename
            
        Returns:
            str: Path to processed reference image
        """
        try:
            print(f"📥 Downloading reference image from: {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Load and process image
            image = Image.open(io.BytesIO(response.content))
            print(f"✅ Reference image downloaded: {image.size}")
            
            # Convert to RGB if necessary (WAN 1.3B requirement)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"🔧 Converted image to RGB mode")
            
            # ENHANCED: Optimized resize for WAN 1.3B model (480x832)
            target_size = (480, 832)
            
            # Smart resize that maintains aspect ratio and centers
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create new image with exact target size and paste centered
            processed_image = Image.new('RGB', target_size, (0, 0, 0))
            x = (target_size[0] - image.width) // 2
            y = (target_size[1] - image.height) // 2
            processed_image.paste(image, (x, y))
            
            print(f"✅ Reference image optimized for WAN 1.3B: {target_size}")
            
            # Save to temporary file with optimized settings
            temp_path = f"/tmp/{filename_prefix}.png"
            processed_image.save(temp_path, "PNG", quality=95, optimize=True)
            print(f"💾 Reference image saved: {temp_path}")
            
            return temp_path
            
        except Exception as e:
            print(f"❌ Error processing reference image: {str(e)}")
            raise

    def process_job(self, job_data):
        """
        ENHANCED: Integrated job processing with streamlined reference frame support
        
        Args:
            job_data (dict): Job data with standardized parameters
            
        Returns:
            str: Path to generated output file
        """
        try:
            # CONSISTENT: Extract standardized parameters
            job_id = job_data.get('id')
            job_type = job_data.get('type')
            prompt = job_data.get('prompt', '')
            config = job_data.get('config', {})
            metadata = job_data.get('metadata', {})
            user_id = job_data.get('user_id')
            
            print(f"📬 WAN 1.3B Job #{job_id} received - ENHANCED INTEGRATION")
            print(f"🎯 Job type: {job_type}")
            print(f"📝 Prompt: {prompt}")
            print(f"👤 User ID: {user_id}")
            
            # Validate job type
            if job_type not in self.job_configs:
                raise ValueError(f"Unsupported job type: {job_type}")
            
            # Get base configuration
            base_config = self.job_configs[job_type].copy()
            
            # ENHANCED: Extract reference parameters with unified handling
            reference_strength = metadata.get('reference_strength', 0.85)
            
            # Support all reference parameter naming conventions
            reference_image_url = (config.get('image') or 
                                 metadata.get('reference_image_url') or
                                 metadata.get('image'))
            
            start_reference_url = (config.get('first_frame') or 
                                 metadata.get('start_reference_url') or
                                 metadata.get('start_reference_image_url'))
            
            end_reference_url = (config.get('last_frame') or 
                               metadata.get('end_reference_url') or
                               metadata.get('end_reference_image_url'))
            
            print(f"🖼️ ENHANCED Reference processing: strength {reference_strength}")
            if reference_image_url:
                print(f"📥 Single reference image URL: {reference_image_url}")
            if start_reference_url:
                print(f"📥 Start reference frame URL: {start_reference_url}")
            if end_reference_url:
                print(f"📥 End reference frame URL: {end_reference_url}")
            
            # ENHANCED: Apply reference strength adjustment to guidance scale
            if reference_image_url or start_reference_url or end_reference_url:
                base_guide_scale = base_config['sample_guide_scale']
                adjusted_guide_scale = self.adjust_guidance_for_reference_strength(
                    base_guide_scale, reference_strength
                )
                base_config['sample_guide_scale'] = adjusted_guide_scale
                print(f"🎯 Applied reference strength adjustment: {base_guide_scale} → {adjusted_guide_scale:.2f}")
            
            print(f"✅ Job type validated: {job_type} (enhance: {base_config['enhance_prompt']})")
            
            # Optimize frame count for video jobs
            if base_config['content_type'] == 'video':
                frame_num = base_config['frame_num']
                effective_fps = 16.67  # WAN 1.3B effective frame rate
                duration = frame_num / effective_fps
                print(f"🔧 OPTIMIZED FRAME COUNT: {frame_num} frames")
                print(f"⏱️ Expected duration: {duration:.1f} seconds (WAN 1.3B @ {effective_fps}fps)")
            
            # Apply prompt enhancement if enabled
            if base_config['enhance_prompt']:
                print(f"🚀 Prompt enhancement enabled - using Qwen 7B")
                enhanced_prompt = self.enhance_prompt_with_qwen(prompt)
                if enhanced_prompt:
                    prompt = enhanced_prompt
                    print(f"✨ Enhanced prompt: {prompt[:100]}...")
                else:
                    print(f"📝 Using original prompt (enhancement failed)")
            else:
                print(f"📝 Using original prompt (no enhancement)")
            
            # ENHANCED: Unified generation routing with streamlined logic
            if base_config['content_type'] == 'video':
                result = self.generate_video_content(
                    prompt, base_config, job_id,
                    reference_image_url, start_reference_url, end_reference_url
                )
            else:
                result = self.generate_image_content(
                    prompt, base_config, job_id,
                    reference_image_url
                )
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing job: {str(e)}")
            raise

    def generate_video_content(self, prompt, config, job_id, reference_image_url=None, 
                             start_reference_url=None, end_reference_url=None):
        """
        ENHANCED: Unified video generation with streamlined reference frame handling
        
        Args:
            prompt (str): Generation prompt
            config (dict): Job configuration
            job_id (str): Job identifier
            reference_image_url (str, optional): Single reference image URL
            start_reference_url (str, optional): Start frame reference URL
            end_reference_url (str, optional): End frame reference URL
            
        Returns:
            str: Path to generated video file
        """
        try:
            print(f"🎬 ENHANCED video generation with integrated reference handling")
            
            # Prepare output path
            output_path = f"/tmp/wan_video_output_{job_id}.mp4"
            
            # ENHANCED: Streamlined reference frame processing
            reference_paths = {}
            
            if reference_image_url:
                print(f"🖼️ Processing single reference image")
                reference_paths['image'] = self.download_and_process_reference_image(
                    reference_image_url, f"wan_single_ref_{job_id}"
                )
                
            if start_reference_url:
                print(f"🖼️ Processing start frame reference")
                reference_paths['first_frame'] = self.download_and_process_reference_image(
                    start_reference_url, f"wan_start_frame_{job_id}"
                )
                
            if end_reference_url:
                print(f"🖼️ Processing end frame reference")
                reference_paths['last_frame'] = self.download_and_process_reference_image(
                    end_reference_url, f"wan_end_frame_{job_id}"
                )
            
            # Build and execute WAN command
            cmd = self.build_wan_command(
                task_type="t2v-1.3B",
                prompt=prompt,
                config=config,
                output_path=output_path,
                reference_paths=reference_paths
            )
            
            print(f"🎬 WAN 1.3B enhanced video command: {' '.join(cmd)}")
            
            # Execute generation
            result = self.execute_wan_generation(cmd, output_path, "enhanced video")
            
            # Cleanup reference files
            for ref_path in reference_paths.values():
                try:
                    os.unlink(ref_path)
                except:
                    pass
            
            return result
            
        except Exception as e:
            print(f"❌ Error in enhanced video generation: {str(e)}")
            raise

    def generate_image_content(self, prompt, config, job_id, reference_image_url=None):
        """
        ENHANCED: Unified image generation with optional reference support
        
        Args:
            prompt (str): Generation prompt
            config (dict): Job configuration
            job_id (str): Job identifier
            reference_image_url (str, optional): Reference image URL
            
        Returns:
            str: Path to generated image file
        """
        try:
            print(f"🖼️ ENHANCED image generation with integrated reference handling")
            
            # Prepare output path
            output_path = f"/tmp/wan_image_output_{job_id}.png"
            
            # ENHANCED: Optional reference image processing
            reference_paths = {}
            
            if reference_image_url:
                print(f"🖼️ Processing reference image for I2V-style generation")
                reference_paths['image'] = self.download_and_process_reference_image(
                    reference_image_url, f"wan_image_ref_{job_id}"
                )
            
            # Build and execute WAN command
            cmd = self.build_wan_command(
                task_type="t2v-1.3B",
                prompt=prompt,
                config=config,
                output_path=output_path,
                reference_paths=reference_paths
            )
            
            print(f"🖼️ WAN 1.3B enhanced image command: {' '.join(cmd)}")
            
            # Execute generation
            result = self.execute_wan_generation(cmd, output_path, "enhanced image")
            
            # Cleanup reference files
            for ref_path in reference_paths.values():
                try:
                    os.unlink(ref_path)
                except:
                    pass
            
            return result
            
        except Exception as e:
            print(f"❌ Error in enhanced image generation: {str(e)}")
            raise

    def build_wan_command(self, task_type, prompt, config, output_path, reference_paths=None):
        """
        ENHANCED: Build WAN command with integrated reference parameter handling
        
        Args:
            task_type (str): WAN task type
            prompt (str): Generation prompt
            config (dict): Job configuration
            output_path (str): Output file path
            reference_paths (dict, optional): Dictionary of reference image paths
            
        Returns:
            list: Complete WAN command
        """
        # Use correct path to wan_generate.py in worker repository
        wan_generate_path = "/workspace/ourvidz-worker/wan_generate.py"
        
        cmd = [
            "python", wan_generate_path,
            "--task", task_type,
            "--ckpt_dir", self.model_path,
            "--offload_model", "True",
            "--size", config['size'],
            "--sample_steps", str(config['sample_steps']),
            "--sample_guide_scale", str(config['sample_guide_scale']),
            "--sample_solver", config.get('sample_solver', 'unipc'),
            "--sample_shift", str(config.get('sample_shift', 5.0)),
            "--frame_num", str(config['frame_num']),
            "--prompt", prompt,
            "--save_file", output_path
        ]
        
        # ENHANCED: Add reference parameters if provided
        if reference_paths:
            for param_name, ref_path in reference_paths.items():
                cmd.extend([f"--{param_name}", ref_path])
                print(f"🖼️ Added reference parameter: --{param_name} {ref_path}")
        
        return cmd

    def execute_wan_generation(self, cmd, output_path, generation_type):
        """
        ENHANCED: Execute WAN generation with improved error handling and validation
        
        Args:
            cmd (list): WAN command to execute
            output_path (str): Expected output file path
            generation_type (str): Type of generation for logging
            
        Returns:
            str: Path to generated file
        """
        try:
            print(f"🎯 {generation_type.title()} output path: {output_path}")
            print(f"📄 Expected file type: {generation_type}")
            
            # Set timeout for generation
            timeout_seconds = 500  # 8+ minutes for video generation
            
            print(f"⏰ Starting WAN 1.3B subprocess ({generation_type}) with {timeout_seconds}s timeout")
            
            # Execute subprocess with timeout
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=self.wan_code_path,
                env=self.setup_environment()
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            print(f"✅ WAN 1.3B subprocess ({generation_type}) completed in {execution_time:.1f}s")
            print(f"📄 Return code: {result.returncode}")
            
            # Enhanced output logging
            if result.stdout:
                print(f"📄 STDOUT ({len(result.stdout.splitlines())} lines):")
                for line in result.stdout.splitlines()[-10:]:  # Last 10 lines
                    print(f"[OUT] {line}")
            
            if result.stderr:
                print(f"📄 STDERR ({len(result.stderr.splitlines())} lines):")
                for line in result.stderr.splitlines()[-10:]:  # Last 10 lines
                    print(f"[ERR] {line}")
            
            if result.returncode != 0:
                raise Exception(f"WAN generation failed with return code {result.returncode}")
            
            # Verify and validate output file
            if not os.path.exists(output_path):
                raise Exception(f"Output file not found: {output_path}")
            
            # ENHANCED: Comprehensive file validation
            is_valid, validation_msg = self.validate_output_file(
                output_path, 
                'video' if 'video' in generation_type else 'image'
            )
            
            if not is_valid:
                raise Exception(f"Generated file validation failed: {validation_msg}")
            
            print(f"✅ {generation_type.title()} generation completed successfully")
            return output_path
            
        except subprocess.TimeoutExpired:
            print(f"❌ WAN generation timed out after {timeout_seconds}s")
            raise Exception(f"Generation timed out")
        except Exception as e:
            print(f"❌ Error in WAN generation: {str(e)}")
            raise

    def enhance_prompt_with_qwen(self, prompt):
        """
        ENHANCED: Prompt enhancement using Qwen 2.5-7B with optimized NSFW handling
        
        Args:
            prompt (str): Original prompt
            
        Returns:
            str: Enhanced prompt or original if enhancement fails
        """
        try:
            # Load Qwen model if not already loaded
            if self.qwen_model is None:
                print(f"🤖 Loading Qwen 2.5-7B model for prompt enhancement...")
                self.qwen_tokenizer = AutoTokenizer.from_pretrained(self.qwen_model_path)
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    self.qwen_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                print(f"✅ Qwen 2.5-7B model loaded successfully")
            
            # ENHANCED: NSFW-optimized enhancement prompt
            enhancement_prompt = f"""Create a detailed, cinematic prompt for AI video generation optimized for adult content:

VISUAL DETAILS: Anatomical accuracy, realistic proportions, natural skin textures, detailed facial features, expressive eyes, natural hair flow, realistic body language.

LIGHTING & ATMOSPHERE: Cinematic lighting, soft shadows, warm tones, intimate atmosphere, professional photography style, natural skin tones.

CAMERA WORK: Close-up shots, intimate framing, smooth camera movements, professional cinematography, dynamic angles.

TECHNICAL QUALITY: 4K quality, sharp focus, no artifacts, smooth motion, consistent lighting, professional color grading.

Original: {prompt}

Enhanced:"""
            
            # Generate enhanced prompt
            inputs = self.qwen_tokenizer(enhancement_prompt, return_tensors="pt")
            
            # Set timeout for generation
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.enhancement_timeout)
            
            try:
                with torch.no_grad():
                    outputs = self.qwen_model.generate(
                        inputs.input_ids,
                        max_new_tokens=300,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.qwen_tokenizer.eos_token_id
                    )
                
                enhanced_text = self.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract enhanced part
                if "Enhanced:" in enhanced_text:
                    enhanced_prompt = enhanced_text.split("Enhanced:")[-1].strip()
                    print(f"✨ Prompt enhanced successfully: {len(enhanced_prompt)} characters")
                    return enhanced_prompt
                else:
                    print(f"⚠️ Enhancement failed to extract enhanced prompt")
                    return prompt
                    
            finally:
                signal.alarm(0)  # Cancel timeout
                
        except TimeoutException:
            print(f"⏰ Qwen enhancement timed out after {self.enhancement_timeout}s")
            return prompt
        except Exception as e:
            print(f"❌ Error in Qwen enhancement: {str(e)}")
            return prompt

    def validate_output_file(self, file_path, expected_content_type):
        """
        ENHANCED: Comprehensive file validation with improved content type checking
        
        Args:
            file_path (str): Path to file to validate
            expected_content_type (str): Expected content type ('image' or 'video')
            
        Returns:
            tuple: (is_valid, validation_message)
        """
        try:
            print(f"🔍 ENHANCED FILE VALIDATION:")
            print(f"   File path: {file_path}")
            print(f"   Expected type: {expected_content_type}")
            
            # Check file exists
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            # Check file size
            file_size = os.path.getsize(file_path)
            print(f"📁 File size: {file_size / 1024**2:.2f}MB ({file_size} bytes)")
            
            if file_size == 0:
                return False, "File is empty"
            
            # Check file header for validation
            with open(file_path, 'rb') as f:
                header = f.read(16)
                print(f"🔍 File header: {header.hex()}")
                
                # Check if it's error text
                if header.startswith(b'Traceback') or header.startswith(b'Error') or header.startswith(b'usage:'):
                    return False, f"File contains error text, not {expected_content_type}"
                
                # Check for proper file format headers
                if expected_content_type == 'video':
                    # MP4 file validation
                    if not (b'ftyp' in header or b'mdat' in header):
                        return False, "File is not a valid MP4 video"
                    min_size = 350000  # 350KB minimum for 5-second video
                elif expected_content_type == 'image':
                    # PNG file validation
                    png_signature = b'\x89PNG\r\n\x1a\n'
                    if not header.startswith(png_signature):
                        return False, "File is not a valid PNG image"
                    min_size = 5000   # 5KB minimum for image
                
                # Check minimum size
                if file_size < min_size:
                    return False, f"File too small for {expected_content_type} (expected at least {min_size} bytes)"
            
            print(f"✅ ENHANCED VALIDATION PASSED")
            return True, "File validation successful"
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False, f"Validation error: {e}"

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
        
        # Add persistent dependencies to Python path
        python_deps_path = '/workspace/python_deps/lib/python3.11/site-packages'
        wan_code_path = '/workspace/Wan2.1'
        
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

    def upload_to_supabase(self, file_path, storage_path):
        """
        ENHANCED: Upload file to Supabase storage with comprehensive validation
        
        Args:
            file_path (str): Local file path
            storage_path (str): Storage path in Supabase
            
        Returns:
            str: Relative path in storage bucket
        """
        try:
            # Pre-upload validation
            if not os.path.exists(file_path):
                raise Exception(f"File does not exist: {file_path}")
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise Exception(f"File is empty: {file_path}")
            
            # Determine MIME type
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.mp4':
                mime_type = 'video/mp4'
            elif file_extension == '.png':
                mime_type = 'image/png'
            else:
                mime_type = 'application/octet-stream'
            
            print(f"📤 ENHANCED Upload:")
            print(f"   Path: {file_path}")
            print(f"   Size: {file_size / 1024**2:.2f}MB")
            print(f"   MIME: {mime_type}")
            print(f"   Storage path: {storage_path}")
            
            # Validate file content
            with open(file_path, 'rb') as f:
                header = f.read(16)
                if header.startswith(b'Traceback') or header.startswith(b'usage:') or header.startswith(b'Error'):
                    raise Exception(f"File appears to be error text, not binary content")
            
            # Upload to Supabase
            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': (os.path.basename(file_path), file, mime_type)},
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                    },
                    timeout=180
                )
            
            if response.status_code == 200:
                # Return relative path within bucket
                path_parts = storage_path.split('/', 1)
                relative_path = path_parts[1] if len(path_parts) == 2 else storage_path
                print(f"✅ Enhanced upload successful: {relative_path}")
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
        """
        ENHANCED: Notify Supabase with standardized callback parameters and enhanced metadata
        
        Args:
            job_id (str): Job identifier
            status (str): Job status
            assets (list, optional): List of asset URLs
            error_message (str, optional): Error message if failed
            metadata (dict, optional): Additional metadata
        """
        try:
            # Standardized callback format
            callback_data = {
                'job_id': job_id,
                'status': status,
                'assets': assets if assets else [],
                'error_message': error_message
            }
            
            # Add enhanced metadata if provided
            if metadata:
                callback_data['metadata'] = metadata
            
            print(f"📞 Sending ENHANCED callback for job {job_id}:")
            print(f"   Status: {status}")
            print(f"   Assets count: {len(assets) if assets else 0}")
            print(f"   Error: {error_message}")
            print(f"   Enhanced metadata: {metadata}")
            
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
                print(f"✅ ENHANCED Callback sent successfully for job {job_id}")
            else:
                print(f"❌ Callback failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job_with_enhanced_integration(self, job_data):
        """
        ENHANCED: Complete job processing with integrated reference frame support
        
        Args:
            job_data (dict): Job data with standardized parameters
        """
        # Extract standardized parameters
        job_id = job_data['id']
        job_type = job_data['type']
        original_prompt = job_data['prompt']
        user_id = job_data['user_id']
        
        # Optional fields
        video_id = job_data.get('video_id', f"video_{int(time.time())}")
        image_id = job_data.get('image_id', f"image_{int(time.time())}")
        config = job_data.get('config', {})
        metadata = job_data.get('metadata', {})
        
        print(f"🔄 Processing job {job_id} ({job_type}) with ENHANCED INTEGRATION")
        print(f"📝 Original prompt: {original_prompt}")
        print(f"🎯 Video ID: {video_id}")
        print(f"👤 User ID: {user_id}")
        
        job_start_time = time.time()
        
        try:
            if job_type not in self.job_configs:
                available_types = list(self.job_configs.keys())
                raise Exception(f"Unknown job type: {job_type}. Available: {available_types}")
            
            job_config = self.job_configs[job_type]
            print(f"✅ Job type validated: {job_type} (enhance: {job_config['enhance_prompt']})")
            
            # Use enhanced processing
            output_file = self.process_job(job_data)
            
            if not output_file:
                raise Exception("Content generation failed or produced no output")
            
            # Final validation
            print(f"🔍 Final validation before upload:")
            is_valid, validation_msg = self.validate_output_file(
                output_file, job_config['content_type']
            )
            if not is_valid:
                raise Exception(f"Generated file failed final validation: {validation_msg}")
            
            # Upload with proper storage path
            file_extension = job_config['file_extension']
            storage_path = f"{job_type}/{user_id}/{video_id}.{file_extension}"
            
            print(f"📤 Uploading validated {job_config['content_type']} file to: {storage_path}")
            relative_path = self.upload_to_supabase(output_file, storage_path)
            
            # Cleanup temp file
            try:
                os.unlink(output_file)
                print(f"🗑️ Cleaned up temp file: {output_file}")
            except:
                pass
            
            # Calculate total time
            total_time = time.time() - job_start_time
            
            # Prepare enhanced metadata
            callback_metadata = {
                'generation_time': total_time,
                'job_type': job_type,
                'content_type': job_config['content_type'],
                'frame_num': job_config['frame_num'],
                'wan_task': 't2v-1.3B',
                'enhanced_integration': True,
                'reference_mode': self._determine_reference_mode(job_data),
                'guidance_adjusted': self._has_reference_frames(job_data)
            }
            
            # Success callback
            self.notify_completion(
                job_id, 'completed', 
                assets=[relative_path], 
                metadata=callback_metadata
            )
            
            print(f"🎉 Job {job_id} completed successfully in {total_time:.1f}s")
            print(f"📁 Output: {relative_path}")
            print(f"✅ Enhanced integration complete")
            
        except Exception as e:
            error_msg = str(e)
            total_time = time.time() - job_start_time
            print(f"❌ Job {job_id} failed after {total_time:.1f}s: {error_msg}")
            
            # Cleanup temp files
            try:
                for temp_file in glob.glob("/tmp/wan_*output_*"):
                    os.unlink(temp_file)
            except:
                pass
            
            # Error metadata
            error_metadata = {
                'error_type': type(e).__name__,
                'job_type': job_type,
                'wan_task': 't2v-1.3B',
                'enhanced_integration': True,
                'timestamp': time.time()
            }
            
            # Failure callback
            self.notify_completion(
                job_id, 'failed', 
                error_message=error_msg, 
                metadata=error_metadata
            )

    def _determine_reference_mode(self, job_data):
        """Determine reference mode from job data"""
        config = job_data.get('config', {})
        metadata = job_data.get('metadata', {})
        
        single_ref = config.get('image') or metadata.get('reference_image_url')
        start_ref = config.get('first_frame') or metadata.get('start_reference_url')
        end_ref = config.get('last_frame') or metadata.get('end_reference_url')
        
        if single_ref and not start_ref and not end_ref:
            return 'single'
        elif start_ref and end_ref:
            return 'both'
        elif start_ref and not end_ref:
            return 'start'
        elif end_ref and not start_ref:
            return 'end'
        else:
            return 'none'

    def _has_reference_frames(self, job_data):
        """Check if job has any reference frames"""
        return self._determine_reference_mode(job_data) != 'none'

    def poll_queue(self):
        """Poll Redis queue for new jobs with Upstash REST API compatibility"""
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

    def run_with_enhanced_integration(self):
        """Main worker loop with enhanced integration"""
        print("🎬 Enhanced OurVidz WAN Worker - INTEGRATED REFERENCE FRAMES")
        print("🔧 ENHANCEMENT: Streamlined reference frame handling with guidance adjustment")
        print("🔧 INTEGRATION: Unified parameter processing for all reference modes")
        print("🔧 CONSISTENCY: Standardized callback format alignment with edge function")
        print("📊 Status: Enhanced Integration Complete ✅")
        
        print("🔧 UPSTASH COMPATIBLE: Using non-blocking RPOP for Redis polling")
        print("📋 Supported job types with enhanced integration:")
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
                    print(f"\n📬 Enhanced WAN Job #{job_count} received")
                    self.process_job_with_enhanced_integration(job_data)
                    print("=" * 60)
                else:
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                print("🛑 Enhanced worker stopped by user")
                break
            except Exception as e:
                consecutive_errors += 1
                print(f"❌ Enhanced worker error #{consecutive_errors}: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    print(f"❌ Too many consecutive errors ({consecutive_errors}), shutting down")
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
    
    print("✅ All paths validated for enhanced integration")
    print("🔧 ENHANCED: Streamlined reference frame handling")
    print("🖼️ INTEGRATION: Unified parameter processing for all modes")
    
    try:
        worker = EnhancedWanWorker()
        worker.run_with_enhanced_integration()
    except Exception as e:
        print(f"❌ Enhanced worker startup failed: {e}")
        exit(1)
    finally:
        print("👋 Enhanced WAN Worker shutdown complete")