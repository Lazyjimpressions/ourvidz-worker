# wan_worker.py - ENHANCED WITH QWEN 7B INTEGRATION
# NEW: Supports 4 enhanced job types with Qwen 2.5-7B prompt enhancement
# Performance: Standard jobs + 14s enhancement time

import os
import json
import time
import requests
import subprocess
import uuid
import shutil
import gc
from pathlib import Path
from PIL import Image
import cv2
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Clean environment first
for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
    if key in os.environ:
        del os.environ[key]

# Import torch after cleaning environment
import torch
import torch.nn as nn
import numpy as np

class EnhancedWanWorker:
    def __init__(self):
        logger.info("🚀 ENHANCED WAN WORKER - WITH QWEN 7B INTEGRATION")
        logger.info("✨ NEW: 4 enhanced job types with AI prompt enhancement")
        logger.info("⚡ Performance: Standard generation + 14s enhancement")
        logger.info("🎯 Qwen 2.5-7B: Superior quality, 9x faster than 14B")
        
        # Model paths
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_path = "/workspace/Wan2.1"
        
        # Enhancement configuration
        self.enhancement_config = {
            'model_name': 'Qwen/Qwen2.5-7B-Instruct',  # ✅ Tested and working
            'hf_home': '/workspace/models/huggingface_cache',
            'expected_time': 14,  # seconds (measured performance)
            'pythonpath': '/workspace/python_deps/lib/python3.11/site-packages'
        }
        
        # ENHANCED JOB CONFIGURATIONS - Complete settings for all job types
        self.job_type_mapping = {
            # ===== STANDARD JOB TYPES (Existing) =====
            'image_fast': {
                'content_type': 'image',
                'file_extension': 'png',
                'sample_steps': 12,
                'sample_guide_scale': 6.0,
                'size': '832*480',
                'frame_num': 1,
                'storage_bucket': 'image_fast',
                'expected_time_per_image': 73,
                'supports_batch': True,
                'enhancement': False
            },
            'image_high': {
                'content_type': 'image', 
                'file_extension': 'png',
                'sample_steps': 25,
                'sample_guide_scale': 7.5,
                'size': '832*480',
                'frame_num': 1,
                'storage_bucket': 'image_high',
                'expected_time_per_image': 90,
                'supports_batch': True,
                'enhancement': False
            },
            'video_fast': {
                'content_type': 'video',
                'file_extension': 'mp4',
                'sample_steps': 15,
                'sample_guide_scale': 6.5,
                'size': '480*832',
                'frame_num': 65,
                'storage_bucket': 'video_fast',
                'expected_time': 180,
                'supports_batch': False,
                'enhancement': False
            },
            'video_high': {
                'content_type': 'video',
                'file_extension': 'mp4', 
                'sample_steps': 25,
                'sample_guide_scale': 8.0,
                'size': '832*480',
                'frame_num': 81,
                'storage_bucket': 'video_high',
                'expected_time': 280,
                'supports_batch': False,
                'enhancement': False
            },
            
            # ===== ENHANCED JOB TYPES (New with Qwen 7B) =====
            'image7b_fast_enhanced': {
                'content_type': 'image',
                'file_extension': 'png',
                'sample_steps': 12,                    # Same as image_fast
                'sample_guide_scale': 6.0,             # Same as image_fast
                'size': '832*480',                     # Same as image_fast
                'frame_num': 1,                        # Same as image_fast
                'storage_bucket': 'image7b_fast_enhanced',  # Matches job type exactly
                'expected_time_per_image': 87,         # 73s + 14s enhancement
                'supports_batch': True,
                'enhancement': True,
                'enhancement_model': 'Qwen/Qwen2.5-7B-Instruct'
            },
            'image7b_high_enhanced': {
                'content_type': 'image',
                'file_extension': 'png',
                'sample_steps': 25,                    # Same as image_high
                'sample_guide_scale': 7.5,             # Same as image_high
                'size': '832*480',                     # Same as image_high
                'frame_num': 1,                        # Same as image_high
                'storage_bucket': 'image7b_high_enhanced', # Matches job type exactly
                'expected_time_per_image': 104,        # 90s + 14s enhancement
                'supports_batch': True,
                'enhancement': True,
                'enhancement_model': 'Qwen/Qwen2.5-7B-Instruct'
            },
            'video7b_fast_enhanced': {
                'content_type': 'video',
                'file_extension': 'mp4',
                'sample_steps': 15,                    # Same as video_fast
                'sample_guide_scale': 6.5,             # Same as video_fast
                'size': '480*832',                     # Same as video_fast
                'frame_num': 65,                       # Same as video_fast
                'storage_bucket': 'video7b_fast_enhanced', # Matches job type exactly
                'expected_time': 194,                  # 180s + 14s enhancement
                'supports_batch': False,
                'enhancement': True,
                'enhancement_model': 'Qwen/Qwen2.5-7B-Instruct'
            },
            'video7b_high_enhanced': {
                'content_type': 'video',
                'file_extension': 'mp4',
                'sample_steps': 25,                    # Same as video_high
                'sample_guide_scale': 8.0,             # Same as video_high
                'size': '832*480',                     # Same as video_high
                'frame_num': 81,                       # Same as video_high
                'storage_bucket': 'video7b_high_enhanced', # Matches job type exactly
                'expected_time': 294,                  # 280s + 14s enhancement
                'supports_batch': False,
                'enhancement': True,
                'enhancement_model': 'Qwen/Qwen2.5-7B-Instruct'
            }
        }
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        # Validate environment
        self.validate_environment()
        
        logger.info("🎯 Enhanced WAN Worker ready - 8 job types supported")
        logger.info("📋 Standard: image_fast, image_high, video_fast, video_high")
        logger.info("✨ Enhanced: image7b_fast_enhanced, image7b_high_enhanced, video7b_fast_enhanced, video7b_high_enhanced")

    def validate_environment(self):
        """Validate all required components including Qwen 7B"""
        logger.info("🔍 VALIDATING ENHANCED WAN ENVIRONMENT")
        logger.info("-" * 50)
        
        # Check PyTorch GPU
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ GPU: {device_name} ({total_memory:.1f}GB)")
        else:
            logger.error("❌ CUDA not available")
            
        # Check WAN models
        if Path(self.model_path).exists():
            logger.info(f"✅ WAN 2.1 models: {self.model_path}")
        else:
            logger.error(f"❌ WAN models missing: {self.model_path}")
            
        # Check WAN installation
        if Path(self.wan_path).exists():
            generate_script = Path(self.wan_path) / "generate.py"
            if generate_script.exists():
                logger.info(f"✅ WAN generate script: {generate_script}")
            else:
                logger.error(f"❌ Generate script missing: {generate_script}")
        else:
            logger.error(f"❌ WAN 2.1 missing: {self.wan_path}")
            
        # Check Qwen 7B model
        qwen_path = Path(self.enhancement_config['hf_home']) / "models--Qwen--Qwen2.5-7B-Instruct"
        if qwen_path.exists():
            logger.info(f"✅ Qwen 2.5-7B model: {qwen_path}")
        else:
            logger.error(f"❌ Qwen 7B model missing: {qwen_path}")
            
        # Check Python dependencies
        deps_path = Path(self.enhancement_config['pythonpath'])
        if deps_path.exists():
            logger.info(f"✅ Python dependencies: {deps_path}")
        else:
            logger.error(f"❌ Python deps missing: {deps_path}")
            
        # Check environment variables
        required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            logger.error(f"❌ Missing env vars: {missing}")
        else:
            logger.info("✅ All environment variables configured")

    def enhance_prompt_with_qwen7b(self, original_prompt):
        """Enhance prompt using Qwen 2.5-7B model - TESTED AND WORKING"""
        logger.info(f"✨ Enhancing prompt with Qwen 7B: '{original_prompt}'")
        start_time = time.time()
        
        # Set environment for Qwen
        env = os.environ.copy()
        env.update({
            'HF_HOME': self.enhancement_config['hf_home'],
            'PYTHONPATH': self.enhancement_config['pythonpath'],
            'CUDA_VISIBLE_DEVICES': '0'
        })
        
        # Create enhancement command - EXACTLY AS TESTED
        cmd = [
            "python", "generate.py",
            "--task", "t2v-1.3B",
            "--ckpt_dir", self.model_path,
            "--use_prompt_extend",
            "--prompt_extend_method", "local_qwen", 
            "--prompt_extend_model", self.enhancement_config['model_name'],
            "--size", "832*480",  # Dummy size (not used for enhancement only)
            "--frame_num", "1",   # Dummy frame (not used for enhancement only)
            "--prompt", original_prompt,
            "--save_file", "/tmp/dummy_enhancement.mp4"  # Won't be used
        ]
        
        original_cwd = os.getcwd()
        try:
            os.chdir(self.wan_path)
            
            # Run enhancement - capture both stdout and stderr
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout for enhancement
            )
            
            enhancement_time = time.time() - start_time
            
            if result.returncode == 0:
                # Parse enhanced prompt from output
                output = result.stdout + result.stderr
                
                # Look for "Extended prompt:" in the output
                for line in output.split('\n'):
                    if 'Extended prompt:' in line:
                        enhanced_prompt = line.split('Extended prompt:', 1)[1].strip()
                        if enhanced_prompt:
                            logger.info(f"✅ Prompt enhanced in {enhancement_time:.1f}s")
                            logger.info(f"🎨 Enhanced: {enhanced_prompt[:100]}...")
                            return enhanced_prompt
                
                # Fallback: return original if parsing failed
                logger.warning(f"⚠️ Enhancement parsing failed, using original prompt")
                return original_prompt
            else:
                logger.error(f"❌ Enhancement failed: {result.stderr}")
                return original_prompt
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Enhancement timed out")
            return original_prompt
        except Exception as e:
            logger.error(f"❌ Enhancement error: {e}")
            return original_prompt
        finally:
            os.chdir(original_cwd)
            # Cleanup dummy file if created
            Path("/tmp/dummy_enhancement.mp4").unlink(missing_ok=True)

    def generate_with_wan21(self, prompt, job_type, image_index=None):
        """Generate single image/video with WAN 2.1 - Enhanced with Qwen support"""
        
        if job_type not in self.job_type_mapping:
            raise ValueError(f"Unknown job type: {job_type}")
            
        config = self.job_type_mapping[job_type]
        job_id = str(uuid.uuid4())[:8]
        
        # Apply enhancement if needed
        if config.get('enhancement', False):
            logger.info(f"✨ ENHANCED JOB: Applying Qwen 7B enhancement")
            enhanced_prompt = self.enhance_prompt_with_qwen7b(prompt)
            actual_prompt = enhanced_prompt
        else:
            logger.info(f"📝 STANDARD JOB: Using original prompt")
            actual_prompt = prompt
        
        if image_index is not None:
            logger.info(f"🎬 Starting {job_type} generation {image_index}: {actual_prompt[:50]}...")
        else:
            logger.info(f"🎬 Starting {job_type} generation: {actual_prompt[:50]}...")
            
        logger.info(f"📋 Config: {config['size']}, {config['frame_num']} frames, {config['sample_steps']} steps")
        
        # Create temp directories
        temp_base = Path("/tmp/ourvidz")
        temp_base.mkdir(exist_ok=True)
        temp_processing = temp_base / "processing"
        temp_processing.mkdir(exist_ok=True)
        
        temp_video_path = temp_processing / f"wan21_{job_id}.mp4"
        
        # WAN generation command - SAME AS BEFORE
        cmd = [
            "python", "generate.py",
            "--task", "t2v-1.3B",
            "--ckpt_dir", str(self.model_path),
            "--offload_model", "False",
            "--size", config['size'],
            "--sample_steps", str(config['sample_steps']),
            "--sample_guide_scale", str(config['sample_guide_scale']),
            "--frame_num", str(config['frame_num']),
            "--prompt", actual_prompt,  # ✅ Use enhanced prompt
            "--save_file", str(temp_video_path.absolute())
        ]
        
        # Environment for WAN generation
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'TORCH_USE_CUDA_DSA': '1',
            'PYTHONUNBUFFERED': '1',
            'HF_HOME': self.enhancement_config['hf_home'],
            'PYTHONPATH': self.enhancement_config['pythonpath']
        })
        
        # Execute WAN generation
        original_cwd = os.getcwd()
        try:
            os.chdir(self.wan_path)
            start_time = time.time()
            
            logger.info(f"🎬 STARTING WAN 2.1 GENERATION")
            logger.info(f"📝 Using prompt: '{actual_prompt[:100]}...'")
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            generation_time = time.time() - start_time
            
            if result.returncode == 0 and temp_video_path.exists():
                file_size = temp_video_path.stat().st_size / 1024
                logger.info(f"✅ Generation successful in {generation_time:.1f}s ({file_size:.0f}KB)")
                return str(temp_video_path)
            else:
                logger.error(f"❌ WAN generation failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Generation timed out")
            return None
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return None
        finally:
            os.chdir(original_cwd)

    def extract_image_from_video(self, video_path, output_path):
        """Extract first frame from video for image jobs"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image.save(output_path, "PNG", quality=95, optimize=True)
                return Path(output_path).exists()
            return False
                
        except Exception as e:
            logger.error(f"❌ Frame extraction failed: {e}")
            return False

    def upload_to_supabase(self, file_path, storage_path):
        """Upload file to Supabase storage"""
        try:
            if not Path(file_path).exists():
                return None
                
            # Determine content type
            content_type = 'image/png' if storage_path.endswith('.png') else 'video/mp4'
            
            with open(file_path, 'rb') as file:
                file_data = file.read()
            
            response = requests.post(
                f"{self.supabase_url}/storage/v1/object/{storage_path}",
                data=file_data,
                headers={
                    'Authorization': f"Bearer {self.supabase_service_key}",
                    'Content-Type': content_type,
                    'x-upsert': 'true'
                },
                timeout=120
            )
            
            if response.status_code in [200, 201]:
                path_parts = storage_path.split('/', 1)
                return path_parts[1] if len(path_parts) == 2 else storage_path
            return None
                
        except Exception as e:
            logger.error(f"❌ Upload error: {e}")
            return None

    def process_job(self, job_data):
        """Process enhanced or standard job"""
        job_id = job_data['jobId']
        job_type = job_data['jobType']
        prompt = job_data['prompt']
        user_id = job_data['userId']
        video_id = job_data.get('videoId')
        image_id = job_data.get('imageId')
        
        logger.info(f"🚀 === PROCESSING WAN JOB {job_id} ===")
        logger.info(f"📋 Job Type: {job_type}")
        logger.info(f"📝 Original Prompt: '{prompt}'")
        
        config = self.job_type_mapping.get(job_type)
        if not config:
            error_msg = f"Unknown job type: {job_type}"
            logger.error(f"❌ {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)
            return
        
        # Log enhancement status
        if config.get('enhancement', False):
            logger.info(f"✨ ENHANCED JOB with {config['enhancement_model']}")
        else:
            logger.info(f"📝 STANDARD JOB")
        
        try:
            start_time = time.time()
            
            # Handle batch vs single generation
            is_image_job = config['content_type'] == 'image'
            num_images = job_data.get('metadata', {}).get('num_images', 6 if is_image_job else 1)
            
            if is_image_job and num_images > 1:
                # Batch image generation
                upload_urls = []
                
                for i in range(num_images):
                    video_path = self.generate_with_wan21(prompt, job_type, image_index=i+1)
                    
                    if video_path:
                        timestamp = int(time.time())
                        filename = f"wan_{job_id}_{timestamp}_{i+1}.png"
                        image_path = Path(f"/tmp/{filename}")
                        
                        if self.extract_image_from_video(video_path, image_path):
                            storage_path = f"{config['storage_bucket']}/{user_id}/{filename}"
                            upload_path = self.upload_to_supabase(image_path, storage_path)
                            
                            if upload_path:
                                upload_urls.append(upload_path)
                            
                            image_path.unlink(missing_ok=True)
                        
                        Path(video_path).unlink(missing_ok=True)
                
                if upload_urls:
                    total_time = time.time() - start_time
                    logger.info(f"✅ Enhanced batch job completed in {total_time:.1f}s")
                    self.notify_completion(job_id, 'completed', image_urls=upload_urls)
                else:
                    raise Exception("All batch generations failed")
                    
            else:
                # Single generation
                output_path = self.generate_with_wan21(prompt, job_type)
                
                if not output_path:
                    raise Exception("Generation failed")
                
                timestamp = int(time.time())
                
                if is_image_job:
                    # Single image
                    filename = f"wan_{job_id}_{timestamp}.png"
                    image_path = Path(f"/tmp/{filename}")
                    
                    if self.extract_image_from_video(output_path, image_path):
                        storage_path = f"{config['storage_bucket']}/{user_id}/{filename}"
                        upload_path = self.upload_to_supabase(image_path, storage_path)
                        
                        image_path.unlink(missing_ok=True)
                        Path(output_path).unlink(missing_ok=True)
                        
                        if upload_path:
                            total_time = time.time() - start_time
                            logger.info(f"✅ Enhanced single image completed in {total_time:.1f}s")
                            self.notify_completion(job_id, 'completed', image_urls=[upload_path])
                        else:
                            raise Exception("Image upload failed")
                    else:
                        raise Exception("Frame extraction failed")
                        
                else:
                    # Video
                    filename = f"wan_{job_id}_{timestamp}.mp4"
                    storage_path = f"{config['storage_bucket']}/{user_id}/{filename}"
                    upload_path = self.upload_to_supabase(output_path, storage_path)
                    
                    Path(output_path).unlink(missing_ok=True)
                    
                    if upload_path:
                        total_time = time.time() - start_time
                        logger.info(f"✅ Enhanced video completed in {total_time:.1f}s")
                        self.notify_completion(job_id, 'completed', file_path=upload_path)
                    else:
                        raise Exception("Video upload failed")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Enhanced job {job_id} failed: {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def notify_completion(self, job_id, status, file_path=None, image_urls=None, error_message=None):
        """Notify Supabase of job completion"""
        try:
            callback_data = {
                'jobId': job_id,
                'status': status,
                'errorMessage': error_message
            }
            
            if image_urls:
                callback_data['imageUrls'] = image_urls
            elif file_path:
                callback_data['filePath'] = file_path
            
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
                logger.info(f"✅ Callback sent for job {job_id}")
            else:
                logger.error(f"⚠️ Callback failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Callback error: {e}")

    def poll_queue(self):
        """Poll Redis WAN queue for new jobs"""
        try:
            response = requests.post(
                f"{self.redis_url}/rpop/wan_queue",
                headers={
                    'Authorization': f"Bearer {self.redis_token}",
                    'Content-Type': 'application/json'
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('result'):
                    return json.loads(result['result'])
            return None
                
        except Exception as e:
            if "timeout" not in str(e).lower():
                logger.warning(f"⚠️ Queue poll error: {e}")
            return None

    def run(self):
        """Main enhanced WAN worker loop"""
        logger.info("🎬 ENHANCED WAN WORKER READY!")
        logger.info("✨ AI Enhancement: Qwen 2.5-7B integration active")
        logger.info("⚡ Performance: Standard times + 14s enhancement")
        logger.info("📬 Polling wan_queue for 8 job types:")
        logger.info("  📝 Standard: image_fast, image_high, video_fast, video_high")
        logger.info("  ✨ Enhanced: image7b_fast_enhanced, image7b_high_enhanced, video7b_fast_enhanced, video7b_high_enhanced")
        
        job_count = 0
        
        try:
            while True:
                try:
                    job = self.poll_queue()
                    if job:
                        job_count += 1
                        job_type = job.get('jobType', 'unknown')
                        is_enhanced = 'enhanced' in job_type
                        
                        logger.info(f"📬 WAN Job #{job_count} received")
                        logger.info(f"🎯 Type: {job_type} {'✨ (Enhanced)' if is_enhanced else '📝 (Standard)'}")
                        
                        self.process_job(job)
                        logger.info("=" * 70)
                    else:
                        time.sleep(5)
                        
                except Exception as e:
                    logger.error(f"❌ Job processing error: {e}")
                    import traceback
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    time.sleep(15)
                    
        except KeyboardInterrupt:
            logger.info("👋 Enhanced WAN Worker shutting down...")
        finally:
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("✅ Enhanced WAN Worker cleanup complete")

if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced WAN Worker - Qwen 7B Integration")
    logger.info("✨ Supports both standard and AI-enhanced video generation")
    
    required_vars = [
        'SUPABASE_URL', 
        'SUPABASE_SERVICE_KEY', 
        'UPSTASH_REDIS_REST_URL', 
        'UPSTASH_REDIS_REST_TOKEN'
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    try:
        worker = EnhancedWanWorker()
        worker.run()
    except Exception as e:
        logger.error(f"❌ Enhanced WAN Worker startup failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        exit(1)
