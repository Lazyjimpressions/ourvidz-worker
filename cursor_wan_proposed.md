# wan_worker.py - CRITICAL FIX for WAN 1.3B Model + REFERENCE FRAMES
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
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B for single frame (image)
                'size': '480*832',
                'sample_steps': 50,
                'sample_guide_scale': 7.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 1,                # Single frame for images
                'enhance_prompt': False,
                'expected_time': 40,
                'content_type': 'image',
                'file_extension': 'png'
            },
            'video_fast': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B for video generation
                'size': '480*832',
                'sample_steps': 25,
                'sample_guide_scale': 6.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 83,               # 5 seconds at 16.67fps
                'enhance_prompt': False,
                'expected_time': 135,
                'content_type': 'video',
                'file_extension': 'mp4'
            },
            'video_high': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B for video generation
                'size': '480*832',
                'sample_steps': 50,
                'sample_guide_scale': 7.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 83,               # 5 seconds at 16.67fps
                'enhance_prompt': False,
                'expected_time': 180,
                'content_type': 'video',
                'file_extension': 'mp4'
            },
            # Enhanced job types (with Qwen 7B enhancement)
            'image7b_fast_enhanced': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B for single frame (image)
                'size': '480*832',
                'sample_steps': 25,
                'sample_guide_scale': 6.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 1,                # Single frame for images
                'enhance_prompt': True,
                'expected_time': 85,
                'content_type': 'image',
                'file_extension': 'png'
            },
            'image7b_high_enhanced': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B for single frame (image)
                'size': '480*832',
                'sample_steps': 50,
                'sample_guide_scale': 7.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 1,                # Single frame for images
                'enhance_prompt': True,
                'expected_time': 100,
                'content_type': 'image',
                'file_extension': 'png'
            },
            'video7b_fast_enhanced': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B for video generation
                'size': '480*832',
                'sample_steps': 25,
                'sample_guide_scale': 6.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 83,               # 5 seconds at 16.67fps
                'enhance_prompt': True,
                'expected_time': 195,
                'content_type': 'video',
                'file_extension': 'mp4'
            },
            'video7b_high_enhanced': {
                'task': 't2v-1.3B',            # ✅ FIXED: Use t2v-1.3B for video generation
                'size': '480*832',
                'sample_steps': 50,
                'sample_guide_scale': 7.5,
                'sample_solver': 'unipc',
                'sample_shift': 5.0,
                'frame_num': 83,               # 5 seconds at 16.67fps
                'enhance_prompt': True,
                'expected_time': 240,
                'content_type': 'video',
                'file_extension': 'mp4'
            }
        }
        
        print("✅ Enhanced WAN Worker initialized with 1.3B model support")
        print(f"📁 Model path: {self.model_path}")
        print(f"📁 WAN code path: {self.wan_code_path}")
        print(f"📁 Qwen model path: {self.qwen_model_path}")

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

    def process_job(self, job_data):
        """Process a job with enhanced reference frame support"""
        try:
            job_id = job_data.get('id')
            job_type = job_data.get('type')
            prompt = job_data.get('prompt', '')
            config = job_data.get('config', {})
            metadata = job_data.get('metadata', {})
            user_id = job_data.get('user_id')
            
            print(f"📬 WAN 1.3B Job #{job_id} received")
            print(f"🎯 Job type: {job_type}")
            print(f"📝 Prompt: {prompt}")
            print(f"👤 User ID: {user_id}")
            
            # Validate job type
            if job_type not in self.job_configs:
                raise ValueError(f"Unsupported job type: {job_type}")
            
            # Get base configuration
            base_config = self.job_configs[job_type].copy()
            
            # Extract reference parameters
            reference_strength = metadata.get('reference_strength', 0.85)
            start_reference_url = config.get('first_frame') or metadata.get('start_reference_url')
            end_reference_url = config.get('last_frame') or metadata.get('end_reference_url')
            
            print(f"🖼️ Reference frame mode: strength {reference_strength}")
            if start_reference_url:
                print(f"📥 Start reference frame URL (first_frame/start_reference_url): {start_reference_url}")
            if end_reference_url:
                print(f"📥 End reference frame URL (last_frame/end_reference_url): {end_reference_url}")
            
            # Validate job configuration
            print(f"✅ Job type validated: {job_type} (enhance: {base_config['enhance_prompt']})")
            
            # Optimize frame count for video jobs
            if base_config['content_type'] == 'video':
                frame_num = base_config['frame_num']
                effective_fps = 16.67  # WAN 1.3B effective frame rate
                duration = frame_num / effective_fps
                print(f"🔧 OPTIMIZED FRAME COUNT: {frame_num} frames")
                print(f"⏱️ Expected duration: {duration:.1f} seconds (confirmed {effective_fps}fps effective rate)")
            
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
            
            # Handle video generation with reference frames
            if base_config['content_type'] == 'video':
                if start_reference_url or end_reference_url:
                    print(f"🎬 Starting start frame video generation (t2v-1.3B + --first_frame)...")
                    
                    # Adjust guidance scale based on reference strength
                    base_guide_scale = base_config['sample_guide_scale']
                    adjusted_guide_scale = self.adjust_guidance_for_reference_strength(base_guide_scale, reference_strength)
                    
                    # Update config with adjusted guidance scale
                    base_config['sample_guide_scale'] = adjusted_guide_scale
                    
                    result = self.generate_video_with_reference_frames(
                        prompt, base_config, start_reference_url, end_reference_url, job_id
                    )
                else:
                    print(f"🎬 Starting standard video generation (t2v-1.3B)...")
                    result = self.generate_standard_video(prompt, base_config, job_id)
            else:
                # Image generation
                print(f"🖼️ Starting image generation (t2v-1.3B)...")
                result = self.generate_image(prompt, base_config, job_id)
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing job: {str(e)}")
            raise

    def generate_video_with_reference_frames(self, prompt, config, start_reference_url, end_reference_url, job_id):
        """Generate video with reference frames using T2V task"""
        try:
            # Download and process reference images
            start_ref_path = None
            end_ref_path = None
            
            if start_reference_url:
                print(f"📥 Downloading reference image from: {start_reference_url}")
                start_ref_path = self.download_and_process_reference_image(start_reference_url, f"wan_start_frame_{job_id}")
                print(f"✅ Start reference image loaded successfully")
            
            if end_reference_url:
                print(f"📥 Downloading reference image from: {end_reference_url}")
                end_ref_path = self.download_and_process_reference_image(end_reference_url, f"wan_end_frame_{job_id}")
                print(f"✅ End reference image loaded successfully")
            
            # Build WAN command with reference frames
            print(f"🎬 Generating video with start frame reference using WAN 1.3B")
            
            # Prepare output path
            output_path = f"/tmp/wan_start_frame_output_{job_id}.mp4"
            
            # Build command with T2V task and reference frames
            cmd = self.build_wan_command(
                task_type="t2v-1.3B",
                prompt=prompt,
                config=config,
                output_path=output_path,
                start_ref_path=start_ref_path,
                end_ref_path=end_ref_path
            )
            
            print(f"🎬 WAN 1.3B T2V command with start frame (t2v-1.3B + --first_frame): {' '.join(cmd)}")
            
            # Execute generation
            result = self.execute_wan_generation(cmd, output_path, "start frame")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in video generation with reference frames: {str(e)}")
            raise

    def generate_standard_video(self, prompt, config, job_id):
        """Generate standard video without reference frames"""
        try:
            print(f"🎬 Starting standard video generation (t2v-1.3B)...")
            
            # Prepare output path
            output_path = f"/tmp/wan_standard_output_{job_id}.mp4"
            
            # Build command without reference frames
            cmd = self.build_wan_command(
                task_type="t2v-1.3B",
                prompt=prompt,
                config=config,
                output_path=output_path
            )
            
            print(f"🎬 WAN 1.3B T2V command (standard): {' '.join(cmd)}")
            
            # Execute generation
            result = self.execute_wan_generation(cmd, output_path, "standard")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in standard video generation: {str(e)}")
            raise

    def generate_image(self, prompt, config, job_id):
        """Generate image using T2V task with single frame"""
        try:
            print(f"🖼️ Starting image generation (t2v-1.3B)...")
            
            # Prepare output path
            output_path = f"/tmp/wan_image_output_{job_id}.png"
            
            # Build command for single frame generation
            cmd = self.build_wan_command(
                task_type="t2v-1.3B",
                prompt=prompt,
                config=config,
                output_path=output_path
            )
            
            print(f"🖼️ WAN 1.3B T2V command (image): {' '.join(cmd)}")
            
            # Execute generation
            result = self.execute_wan_generation(cmd, output_path, "image")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in image generation: {str(e)}")
            raise

    def build_wan_command(self, task_type, prompt, config, output_path, start_ref_path=None, end_ref_path=None):
        """Build WAN command with proper parameters"""
        # CRITICAL: Use correct path to wan_generate.py in worker repository
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
        
        # Add reference frame parameters for T2V task (1.3B Model)
        if start_ref_path:
            cmd.extend(["--first_frame", start_ref_path])
            print(f"🖼️ Start frame reference: {start_ref_path}")
        
        if end_ref_path:
            cmd.extend(["--last_frame", end_ref_path])
            print(f"🖼️ End frame reference: {end_ref_path}")
        
        return cmd

    def download_and_process_reference_image(self, image_url, filename):
        """Download and process reference image for WAN model"""
        try:
            # Download image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Load and process image
            image = Image.open(io.BytesIO(response.content))
            print(f"✅ Reference image downloaded: {image.size}")
            
            # Resize to WAN model input size (480x832)
            target_size = (480, 832)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            print(f"✅ Reference image preprocessed to {target_size}")
            
            # Save to temporary file
            temp_path = f"/tmp/{filename}.png"
            image.save(temp_path, "PNG")
            print(f"💾 Reference image saved: {temp_path}")
            
            return temp_path
            
        except Exception as e:
            print(f"❌ Error processing reference image: {str(e)}")
            raise

    def execute_wan_generation(self, cmd, output_path, generation_type):
        """Execute WAN generation with timeout and error handling"""
        try:
            print(f"🎯 {generation_type.title()} Output path: {output_path}")
            print(f"📄 Expected file type: {generation_type} ({'.mp4' if 'video' in generation_type else '.png'})")
            print(f"🔧 FRAME COUNT: {cmd[cmd.index('--frame_num') + 1]} frames for {generation_type}")
            
            # Set timeout for generation
            timeout_seconds = 500  # 8+ minutes for video generation
            
            print(f"⏰ Starting WAN 1.3B T2V subprocess ({generation_type}) with {timeout_seconds}s timeout")
            
            # Execute subprocess with timeout
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=self.wan_code_path
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            print(f"✅ WAN 1.3B T2V subprocess ({generation_type}) completed in {execution_time:.1f}s")
            print(f"📄 Return code: {result.returncode}")
            
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
            
            # Verify output file exists
            if not os.path.exists(output_path):
                raise Exception(f"Output file not found: {output_path}")
            
            print(f"✅ {generation_type.title()} generation completed successfully")
            return output_path
            
        except subprocess.TimeoutExpired:
            print(f"❌ WAN generation timed out after {timeout_seconds}s")
            raise Exception(f"Generation timed out")
        except Exception as e:
            print(f"❌ Error in WAN generation: {str(e)}")
            raise

    def enhance_prompt_with_qwen(self, prompt):
        """Enhance prompt using Qwen 2.5-7B model"""
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
            
            # Create enhancement prompt
            enhancement_prompt = f"""You are an expert video prompt engineer. Enhance this prompt for high-quality video generation:

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
                        max_new_tokens=200,
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
                    return None
                    
            finally:
                signal.alarm(0)  # Cancel timeout
                
        except TimeoutException:
            print(f"⏰ Qwen enhancement timed out after {self.enhancement_timeout}s")
            return None
        except Exception as e:
            print(f"❌ Error in Qwen enhancement: {str(e)}")
            return None

# Main execution
if __name__ == "__main__":
    worker = EnhancedWanWorker()
    
    # Example job processing
    example_job = {
        "id": "test-123",
        "type": "video_fast",
        "prompt": "beautiful woman walking in garden",
        "config": {
            "first_frame": "https://example.com/start_frame.jpg"
        },
        "metadata": {
            "reference_strength": 0.9,
            "start_reference_url": "https://example.com/start_frame.jpg"
        },
        "user_id": "user-123"
    }
    
    try:
        result = worker.process_job(example_job)
        print(f"✅ Job completed successfully: {result}")
    except Exception as e:
        print(f"❌ Job failed: {str(e)}") 