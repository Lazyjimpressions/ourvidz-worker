# wan_worker.py - CORRECTED FILE HANDLING VERSION
# CRITICAL FIXES: Model path, file handling, callback format alignment with SDXL
# Performance: 67-90s per image, ~8-9 minutes for 6 images

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

class OptimizedWanWorker:
    def __init__(self):
        logger.info("🚀 OPTIMIZED WAN WORKER - CORRECTED FILE HANDLING")
        logger.info("✅ Performance: 67-90s per image, supports 6-image batches")
        logger.info("🔄 Queue: wan_queue (dual worker mode)")
        logger.info("🔧 CRITICAL FIX: File handling aligned with SDXL worker")
        
        # CORRECTED PATHS - aligned with working setup
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"  # Same as before
        self.wan_path = "/workspace/Wan2.1"  # Same as before
        
        # Job configurations with corrected storage buckets
        self.job_type_mapping = {
            'image_fast': {
                'content_type': 'image',
                'file_extension': 'png',
                'sample_steps': 12,
                'sample_guide_scale': 6.0,
                'size': '832*480',
                'frame_num': 1,
                'storage_bucket': 'image_fast',  # MATCHES SUPABASE BUCKET
                'expected_time_per_image': 73,
                'supports_batch': True
            },
            'image_high': {
                'content_type': 'image', 
                'file_extension': 'png',
                'sample_steps': 25,
                'sample_guide_scale': 7.5,
                'size': '832*480',
                'frame_num': 1,
                'storage_bucket': 'image_high',  # MATCHES SUPABASE BUCKET
                'expected_time_per_image': 90,
                'supports_batch': True
            },
            'video_fast': {
                'content_type': 'video',
                'file_extension': 'mp4',
                'sample_steps': 15,
                'sample_guide_scale': 6.5,
                'size': '480*832',
                'frame_num': 65,
                'storage_bucket': 'video_fast',  # MATCHES SUPABASE BUCKET
                'expected_time': 180,
                'supports_batch': False
            },
            'video_high': {
                'content_type': 'video',
                'file_extension': 'mp4', 
                'sample_steps': 25,
                'sample_guide_scale': 8.0,
                'size': '832*480',
                'frame_num': 81,
                'storage_bucket': 'video_high',  # MATCHES SUPABASE BUCKET
                'expected_time': 280,
                'supports_batch': False
            }
        }
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        # Validate environment
        self.validate_environment()
        
        logger.info("🔥 WAN GPU worker ready - CORRECTED VERSION")

    def validate_environment(self):
        """Validate all required components"""
        logger.info("🔍 VALIDATING WAN ENVIRONMENT")
        logger.info("-" * 40)
        
        # Check PyTorch GPU
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ GPU: {device_name} ({total_memory:.1f}GB)")
        else:
            logger.error("❌ CUDA not available")
            
        # Check models
        if Path(self.model_path).exists():
            logger.info(f"✅ Wan 2.1 models: {self.model_path}")
        else:
            logger.error(f"❌ Models missing: {self.model_path}")
            
        # Check Wan 2.1 installation
        if Path(self.wan_path).exists():
            logger.info(f"✅ Wan 2.1 code: {self.wan_path}")
            
            # CRITICAL: Check if generate.py exists
            generate_script = Path(self.wan_path) / "generate.py"
            if generate_script.exists():
                logger.info(f"✅ Generate script: {generate_script}")
            else:
                logger.error(f"❌ Generate script missing: {generate_script}")
        else:
            logger.error(f"❌ Wan 2.1 missing: {self.wan_path}")
            
        # Check environment variables
        required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            logger.error(f"❌ Missing env vars: {missing}")
        else:
            logger.info("✅ All environment variables configured")

    def log_gpu_memory(self):
        """Monitor GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"🔥 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.0f}GB")

    def generate_with_wan21(self, prompt, job_type, image_index=None):
        """Generate single image/video with Wan 2.1 - CORRECTED VERSION"""
        
        if job_type not in self.job_type_mapping:
            raise ValueError(f"Unknown job type: {job_type}")
            
        config = self.job_type_mapping[job_type]
        job_id = str(uuid.uuid4())[:8]
        
        if image_index is not None:
            logger.info(f"🎬 Starting {job_type} generation {image_index}: {prompt[:50]}...")
        else:
            logger.info(f"🎬 Starting {job_type} generation: {prompt[:50]}...")
            
        logger.info(f"📋 Config: {config['size']}, {config['frame_num']} frames, {config['sample_steps']} steps")
        
        # Log GPU memory before
        self.log_gpu_memory()
        
        # Create temp directories - ENSURE THEY EXIST
        temp_base = Path("/tmp/ourvidz")
        temp_base.mkdir(exist_ok=True)
        temp_processing = temp_base / "processing"
        temp_processing.mkdir(exist_ok=True)
        
        temp_video_path = temp_processing / f"wan21_{job_id}.mp4"
        
        # GPU-OPTIMIZED COMMAND - EXACTLY AS BEFORE (WORKING)
        cmd = [
            "python", "generate.py",
            "--task", "t2v-1.3B",
            "--ckpt_dir", str(self.model_path),  # ENSURE STRING
            "--offload_model", "False",
            "--size", config['size'],
            "--sample_steps", str(config['sample_steps']),
            "--sample_guide_scale", str(config['sample_guide_scale']),
            "--frame_num", str(config['frame_num']),
            "--prompt", prompt,
            "--save_file", str(temp_video_path.absolute())
        ]
        
        # GPU-forcing environment - EXACTLY AS BEFORE
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'TORCH_USE_CUDA_DSA': '1',
            'PYTHONUNBUFFERED': '1'
        })
        
        # Execute with proper working directory
        original_cwd = os.getcwd()
        try:
            # CRITICAL: Ensure we're in the right directory
            if not os.path.exists(self.wan_path):
                raise Exception(f"WAN path does not exist: {self.wan_path}")
                
            os.chdir(self.wan_path)
            start_time = time.time()
            
            logger.info(f"🎬 STARTING WAN 2.1 GENERATION")
            logger.info(f"📁 Working directory: {os.getcwd()}")
            logger.info(f"📝 Full prompt: '{prompt}'")
            logger.info(f"🔧 Command: {' '.join(cmd)}")
            logger.info(f"⚙️ Model path: {self.model_path}")
            
            # Verify generate.py exists before running
            generate_script = Path("generate.py")
            if not generate_script.exists():
                raise Exception(f"generate.py not found in {os.getcwd()}")
                
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            generation_time = time.time() - start_time
            
            logger.info(f"🏁 GENERATION COMPLETED")
            logger.info(f"⏱️ Total time: {generation_time:.1f}s")
            logger.info(f"🔢 Return code: {result.returncode}")
            logger.info(f"📁 Expected output: {temp_video_path}")
            logger.info(f"📁 File exists: {temp_video_path.exists()}")
            
            if temp_video_path.exists():
                file_size = temp_video_path.stat().st_size
                logger.info(f"📊 File size: {file_size} bytes ({file_size/1024:.1f}KB)")
            
            # Always log stdout/stderr for debugging
            if result.stdout:
                logger.info(f"📤 STDOUT ({len(result.stdout)} chars):")
                logger.info("=" * 50)
                logger.info(result.stdout[-1000:])  # Last 1000 chars
                logger.info("=" * 50)
            
            if result.stderr:
                logger.info(f"📥 STDERR ({len(result.stderr)} chars):")
                logger.info("=" * 50)
                logger.info(result.stderr[-1000:])  # Last 1000 chars
                logger.info("=" * 50)
            
            # Log GPU memory after
            self.log_gpu_memory()
            
            if result.returncode == 0:
                if image_index is not None:
                    logger.info(f"✅ Generation {image_index} successful in {generation_time:.1f}s")
                else:
                    logger.info(f"✅ Generation successful in {generation_time:.1f}s")
                
                # Verify output file exists
                if temp_video_path.exists():
                    file_size = temp_video_path.stat().st_size / 1024
                    logger.info(f"📁 Output file: {file_size:.0f}KB")
                    return str(temp_video_path)
                else:
                    logger.error("❌ Output file not found despite return code 0")
                    logger.error(f"📁 Checked path: {temp_video_path}")
                    logger.error(f"📁 Directory contents: {list(temp_processing.glob('*'))}")
                    return None
            else:
                logger.error(f"❌ WAN 2.1 GENERATION FAILED")
                logger.error(f"🔢 Exit code: {result.returncode}")
                logger.error(f"📝 Original prompt: '{prompt}'")
                logger.error(f"🔧 Full command: {' '.join(cmd)}")
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
        """Extract first frame from video for image jobs - SAME AS SDXL LOGIC"""
        try:
            logger.info(f"🖼️ Extracting frame from {video_path} to {output_path}")
            
            # Use OpenCV to extract first frame
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB and save as PNG
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image.save(output_path, "PNG", quality=95, optimize=True)
                
                # Verify file was created
                if Path(output_path).exists():
                    file_size = Path(output_path).stat().st_size
                    logger.info(f"✅ Frame extracted: {file_size} bytes")
                    return True
                else:
                    logger.error("❌ PNG file not created")
                    return False
            else:
                logger.error("❌ Failed to extract frame from video")
                return False
                
        except Exception as e:
            logger.error(f"❌ Frame extraction failed: {e}")
            return False

    def upload_to_supabase(self, file_path, storage_path):
        """Upload file to Supabase storage - ALIGNED WITH SDXL WORKER"""
        try:
            # Verify file exists before upload
            if not Path(file_path).exists():
                logger.error(f"❌ File does not exist: {file_path}")
                return None
                
            # Get file size for verification
            file_size = Path(file_path).stat().st_size
            logger.info(f"📁 Uploading file: {file_size} bytes to {storage_path}")
            
            # Determine content type based on file extension
            if storage_path.endswith('.png'):
                content_type = 'image/png'
            elif storage_path.endswith('.mp4'):
                content_type = 'video/mp4'
            else:
                content_type = 'application/octet-stream'
            
            # Use proper binary upload with explicit Content-Type - SAME AS SDXL
            with open(file_path, 'rb') as file:
                file_data = file.read()
            
            headers = {
                'Authorization': f"Bearer {self.supabase_service_key}",
                'Content-Type': content_type,  # ✅ Explicit content type
                'x-upsert': 'true'
            }
            
            response = requests.post(
                f"{self.supabase_url}/storage/v1/object/{storage_path}",
                data=file_data,  # ✅ Raw binary data (same as SDXL)
                headers=headers,
                timeout=120
            )
            
            logger.info(f"📤 Upload response: {response.status_code}")
            
            if response.status_code in [200, 201]:
                # Return relative path within bucket - SAME AS SDXL
                path_parts = storage_path.split('/', 1)
                relative_path = path_parts[1] if len(path_parts) == 2 else storage_path
                logger.info(f"✅ Upload successful: {relative_path}")
                return relative_path
            else:
                logger.error(f"❌ Upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Upload error: {e}")
            return None

    def process_job(self, job_data):
        """Process a single job - ALIGNED WITH SDXL WORKER CALLBACK FORMAT"""
        job_id = job_data['jobId']
        job_type = job_data['jobType']
        prompt = job_data['prompt']
        user_id = job_data['userId']
        video_id = job_data.get('videoId')
        image_id = job_data.get('imageId')
        
        # Extract num_images from metadata (default to 1 for videos, 6 for images)
        is_image_job = 'image' in job_type
        default_num = 6 if is_image_job else 1
        num_images = job_data.get('metadata', {}).get('num_images', default_num)
        
        logger.info(f"🚀 === PROCESSING WAN JOB {job_id} ===")
        logger.info(f"📋 Job Type: {job_type}")
        logger.info(f"📝 Prompt: '{prompt}'")
        logger.info(f"👤 User ID: {user_id}")
        logger.info(f"🎬 Video ID: {video_id}")
        logger.info(f"🖼️ Image ID: {image_id}")
        if is_image_job:
            logger.info(f"🔢 Number of Images: {num_images}")
        
        try:
            config = self.job_type_mapping[job_type]
            start_time = time.time()
            
            logger.info(f"⚙️ Storage bucket: {config['storage_bucket']}")
            logger.info(f"⚙️ Content type: {config['content_type']}")
            
            if config['content_type'] == 'image' and num_images > 1:
                logger.info(f"🎨 BATCH IMAGE GENERATION MODE")
                # Batch image generation - multiple WAN calls
                upload_urls = []
                
                for i in range(num_images):
                    try:
                        logger.info(f"🔄 Generating image {i+1}/{num_images}")
                        
                        # Generate single video with WAN 2.1
                        video_path = self.generate_with_wan21(prompt, job_type, image_index=i+1)
                        
                        if video_path:
                            # Extract frame and upload
                            timestamp = int(time.time())
                            filename = f"wan_{job_id}_{timestamp}_{i+1}.png"
                            image_path = Path(f"/tmp/{filename}")
                            
                            if self.extract_image_from_video(video_path, image_path):
                                storage_path = f"{config['storage_bucket']}/{user_id}/{filename}"
                                upload_path = self.upload_to_supabase(image_path, storage_path)
                                
                                if upload_path:
                                    upload_urls.append(upload_path)
                                    logger.info(f"✅ Image {i+1} uploaded: {upload_path}")
                                
                                # Cleanup
                                image_path.unlink(missing_ok=True)
                            
                            # Cleanup video
                            Path(video_path).unlink(missing_ok=True)
                        
                        # Brief pause between generations
                        if i < num_images - 1:
                            time.sleep(2)
                            
                    except Exception as e:
                        logger.error(f"❌ Image {i+1} generation error: {e}")
                
                if not upload_urls:
                    raise Exception("All image generations failed")
                
                total_time = time.time() - start_time
                logger.info(f"✅ WAN Job {job_id} completed in {total_time:.1f}s")
                logger.info(f"📁 Generated {len(upload_urls)} images")
                
                # CRITICAL: Use imageUrls format (same as SDXL)
                self.notify_completion(job_id, 'completed', image_urls=upload_urls)
                
            else:
                logger.info(f"🎬 SINGLE GENERATION MODE")
                
                # Single generation (video or single image)
                output_path = self.generate_with_wan21(prompt, job_type)
                
                if not output_path:
                    raise Exception("WAN 2.1 generation failed - no output file produced")
                
                upload_path = None
                timestamp = int(time.time())
                
                if config['content_type'] == 'image':
                    logger.info(f"🖼️ PROCESSING AS SINGLE IMAGE")
                    # Single image - extract frame
                    filename = f"wan_{job_id}_{timestamp}.png"
                    image_path = Path(f"/tmp/{filename}")
                    
                    if self.extract_image_from_video(output_path, image_path):
                        storage_path = f"{config['storage_bucket']}/{user_id}/{filename}"
                        upload_path = self.upload_to_supabase(image_path, storage_path)
                        
                        # Cleanup
                        image_path.unlink(missing_ok=True)
                    
                    # Cleanup video
                    Path(output_path).unlink(missing_ok=True)
                    
                    if not upload_path:
                        raise Exception("Image upload failed")
                    
                    # CRITICAL: Use imageUrls format for single image (array with one item)
                    self.notify_completion(job_id, 'completed', image_urls=[upload_path])
                        
                else:  # video
                    logger.info(f"📹 PROCESSING AS VIDEO")
                    # Single video upload
                    filename = f"wan_{job_id}_{timestamp}.mp4"
                    storage_path = f"{config['storage_bucket']}/{user_id}/{filename}"
                    upload_path = self.upload_to_supabase(output_path, storage_path)
                    
                    # Cleanup video
                    Path(output_path).unlink(missing_ok=True)
                    
                    if not upload_path:
                        raise Exception("Video upload failed")
                    
                    # CRITICAL: Use filePath format for video (different from SDXL)
                    self.notify_completion(job_id, 'completed', file_path=upload_path)
                
                total_time = time.time() - start_time
                logger.info(f"✅ WAN Job {job_id} completed in {total_time:.1f}s")
                logger.info(f"📁 File: {upload_path}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ WAN Job {job_id} FAILED")
            logger.error(f"💥 Error: {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)
        finally:
            # Cleanup GPU memory and temp files
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"🧹 Cleanup completed for job {job_id}")

    def notify_completion(self, job_id, status, file_path=None, image_urls=None, error_message=None):
        """Notify Supabase of job completion - ALIGNED WITH SDXL WORKER"""
        try:
            callback_data = {
                'jobId': job_id,
                'status': status,
                'errorMessage': error_message
            }
            
            # Add appropriate response data - SAME LOGIC AS SDXL
            if image_urls:
                callback_data['imageUrls'] = image_urls  # ✅ Array format for images
            elif file_path:
                callback_data['filePath'] = file_path    # ✅ String format for videos
            
            logger.info(f"📤 Sending callback: {callback_data}")
            
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
                logger.info(f"✅ Callback sent for WAN job {job_id}")
                if image_urls:
                    logger.info(f"📊 Sent {len(image_urls)} image URLs")
                elif file_path:
                    logger.info(f"📋 Sent file path: {file_path}")
            else:
                logger.error(f"⚠️ Callback failed: {response.status_code} - {response.text}")
                
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
                logger.warning(f"⚠️ WAN queue poll error: {e}")
            return None

    def run(self):
        """Main WAN worker loop"""
        logger.info("🎬 WAN WORKER READY - CORRECTED FILE HANDLING VERSION!")
        logger.info("🔧 CRITICAL FIX: File handling aligned with SDXL worker")
        logger.info("⚡ Performance: 67-90s per image, ~8-9min for 6-image batch")
        logger.info("📬 Polling wan_queue for image_fast, image_high, video_fast, video_high")
        logger.info("🖼️ BATCH: 6-image batch generation for image jobs")
        logger.info("📤 CALLBACK: Proper format alignment with SDXL worker")
        
        job_count = 0
        
        try:
            while True:
                try:
                    # Poll for jobs from wan_queue
                    job = self.poll_queue()
                    if job:
                        job_count += 1
                        logger.info(f"📬 WAN Job #{job_count} received")
                        logger.info(f"🎯 Processing job: {job.get('jobType', 'unknown')}")
                        
                        # Process the job
                        self.process_job(job)
                        
                        logger.info("=" * 60)
                    else:
                        # No job available, wait briefly
                        time.sleep(5)
                        
                except Exception as e:
                    logger.error(f"❌ WAN job processing error: {e}")
                    # Print full traceback for debugging
                    import traceback
                    logger.error(f"❌ Full traceback: {traceback.format_exc()}")
                    time.sleep(15)
                    
        except KeyboardInterrupt:
            logger.info("👋 WAN Worker shutting down...")
        finally:
            # Cleanup on shutdown
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("✅ WAN Worker cleanup complete")

if __name__ == "__main__":
    logger.info("🚀 Starting WAN 2.1 Worker - CORRECTED FILE HANDLING VERSION")
    logger.info("🔧 CRITICAL FIX: File handling, paths, and callbacks aligned with SDXL")
    
    # Environment validation
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
        worker = OptimizedWanWorker()
        worker.run()
    except Exception as e:
        logger.error(f"❌ WAN Worker startup failed: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        exit(1)
