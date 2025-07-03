# wan_worker.py - CLEAN VERSION - SYNTAX ERROR FIXED
# NEW: Supports 6-image batch generation (6 separate Wan2.1 calls)
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
        print("🚀 OPTIMIZED WAN WORKER - BATCH GENERATION VERSION")
        print("✅ Performance: 67-90s per image, supports 6-image batches")
        print("🔄 Queue: wan_queue (dual worker mode)")
        print("🔧 NEW: 6-image batch generation for improved UX")
        
        # Paths
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_path = "/workspace/Wan2.1"
        
        # Job configurations with batch support
        self.job_type_mapping = {
            'image_fast': {
                'content_type': 'image',
                'file_extension': 'png',
                'sample_steps': 12,
                'sample_guide_scale': 6.0,
                'size': '832*480',
                'frame_num': 1,
                'storage_bucket': 'image_fast',
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
                'storage_bucket': 'image_high',
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
                'storage_bucket': 'video_fast',
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
                'storage_bucket': 'video_high',
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
        
        print("🔥 WAN GPU worker ready - batch generation enabled")

    def validate_environment(self):
        """Validate all required components"""
        print("\n🔍 VALIDATING WAN ENVIRONMENT")
        print("-" * 40)
        
        # Check PyTorch GPU
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU: {device_name} ({total_memory:.1f}GB)")
        else:
            print("❌ CUDA not available")
            
        # Check models
        if Path(self.model_path).exists():
            print(f"✅ Wan 2.1 models: {self.model_path}")
        else:
            print(f"❌ Models missing: {self.model_path}")
            
        # Check Wan 2.1 installation
        if Path(self.wan_path).exists():
            print(f"✅ Wan 2.1 code: {self.wan_path}")
        else:
            print(f"❌ Wan 2.1 missing: {self.wan_path}")
            
        # Check environment variables
        required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            print(f"❌ Missing env vars: {missing}")
        else:
            print("✅ All environment variables configured")

    def log_gpu_memory(self):
        """Monitor GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"🔥 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.0f}GB")

    def generate_with_wan21(self, prompt, job_type, image_index=None):
        """Generate single image/video with Wan 2.1"""
        
        if job_type not in self.job_type_mapping:
            raise ValueError(f"Unknown job type: {job_type}")
            
        config = self.job_type_mapping[job_type]
        job_id = str(uuid.uuid4())[:8]
        
        if image_index is not None:
            print(f"🎬 Starting {job_type} generation {image_index}: {prompt[:50]}...")
        else:
            print(f"🎬 Starting {job_type} generation: {prompt[:50]}...")
            
        print(f"📋 Config: {config['size']}, {config['frame_num']} frames, {config['sample_steps']} steps")
        
        # Log GPU memory before
        self.log_gpu_memory()
        
        # Create temp directories
        temp_base = Path("/tmp/ourvidz")
        temp_base.mkdir(exist_ok=True)
        temp_processing = temp_base / "processing"
        temp_processing.mkdir(exist_ok=True)
        
        temp_video_path = temp_processing / f"wan21_{job_id}.mp4"
        
        # GPU-OPTIMIZED COMMAND
        cmd = [
            "python", "generate.py",
            "--task", "t2v-1.3B",
            "--ckpt_dir", self.model_path,
            "--offload_model", "False",
            "--size", config['size'],
            "--sample_steps", str(config['sample_steps']),
            "--sample_guide_scale", str(config['sample_guide_scale']),
            "--frame_num", str(config['frame_num']),
            "--prompt", prompt,
            "--save_file", str(temp_video_path.absolute())
        ]
        
        # GPU-forcing environment
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'TORCH_USE_CUDA_DSA': '1',
            'PYTHONUNBUFFERED': '1'
        })
        
        # Execute with proper working directory
        original_cwd = os.getcwd()
        try:
            os.chdir(self.wan_path)
            start_time = time.time()
            
            print(f"🎬 STARTING WAN 2.1 GENERATION")
            print(f"📁 Working directory: {os.getcwd()}")
            print(f"📝 Full prompt: '{prompt}'")
            print(f"🔧 Command: {' '.join(cmd)}")
            print(f"⚙️ Environment vars: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'not set')}")
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            generation_time = time.time() - start_time
            
            print(f"🏁 GENERATION COMPLETED")
            print(f"⏱️ Total time: {generation_time:.1f}s")
            print(f"🔢 Return code: {result.returncode}")
            print(f"📁 Expected output: {temp_video_path}")
            print(f"📁 File exists: {temp_video_path.exists()}")
            
            if temp_video_path.exists():
                file_size = temp_video_path.stat().st_size
                print(f"📊 File size: {file_size} bytes ({file_size/1024:.1f}KB)")
            
            # Always log stdout/stderr for debugging
            if result.stdout:
                print(f"📤 STDOUT ({len(result.stdout)} chars):")
                print("=" * 50)
                print(result.stdout[-2000:])  # Last 2000 chars
                print("=" * 50)
            
            if result.stderr:
                print(f"📥 STDERR ({len(result.stderr)} chars):")
                print("=" * 50)
                print(result.stderr[-2000:])  # Last 2000 chars
                print("=" * 50)
            
            # Log GPU memory after
            self.log_gpu_memory()
            
            if result.returncode == 0:
                if image_index is not None:
                    print(f"✅ Generation {image_index} successful in {generation_time:.1f}s")
                else:
                    print(f"✅ Generation successful in {generation_time:.1f}s")
                
                # Verify output file exists
                if temp_video_path.exists():
                    file_size = temp_video_path.stat().st_size / 1024
                    print(f"📁 Output file: {file_size:.0f}KB")
                    return str(temp_video_path)
                else:
                    print("❌ Output file not found despite return code 0")
                    print(f"📁 Checked path: {temp_video_path}")
                    print(f"📁 Directory contents: {list(temp_processing.glob('*'))}")
                    return None
            else:
                print(f"❌ WAN 2.1 GENERATION FAILED")
                print(f"🔢 Exit code: {result.returncode}")
                print(f"📝 Original prompt: '{prompt}'")
                print(f"🔧 Full command: {' '.join(cmd)}")
                
                # Check for common error patterns
                full_output = (result.stdout or '') + (result.stderr or '')
                if 'content' in full_output.lower():
                    print("🚨 POSSIBLE CONTENT FILTERING DETECTED")
                if 'safe' in full_output.lower():
                    print("🚨 POSSIBLE SAFETY FILTER DETECTED")
                if 'cuda' in full_output.lower():
                    print("🚨 POSSIBLE CUDA/GPU ISSUE DETECTED")
                if 'memory' in full_output.lower():
                    print("🚨 POSSIBLE MEMORY ISSUE DETECTED")
                
                return None
                
        except subprocess.TimeoutExpired:
            print("❌ Generation timed out")
            return None
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return None
        finally:
            os.chdir(original_cwd)

    def generate_images_batch(self, prompt, job_type, num_images=6):
        """Generate multiple images by calling Wan2.1 multiple times"""
        config = self.job_type_mapping[job_type]
        
        if config['content_type'] != 'image':
            raise ValueError("Batch generation only supported for image jobs")
        
        logger.info(f"🎨 Starting batch generation: {num_images} images for {job_type}")
        logger.info(f"📝 Prompt: {prompt}")
        
        video_paths = []
        batch_start_time = time.time()
        
        for i in range(num_images):
            try:
                logger.info(f"🔄 Generating image {i+1}/{num_images}")
                video_path = self.generate_with_wan21(prompt, job_type, image_index=i+1)
                
                if video_path:
                    video_paths.append(video_path)
                    logger.info(f"✅ Image {i+1} generated successfully")
                else:
                    logger.warning(f"⚠️ Image {i+1} generation failed")
                    video_paths.append(None)
                
                # Brief pause between generations
                if i < num_images - 1:
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"❌ Image {i+1} generation error: {e}")
                video_paths.append(None)
        
        batch_time = time.time() - batch_start_time
        successful_count = len([p for p in video_paths if p is not None])
        
        logger.info(f"🎉 Batch generation complete: {successful_count}/{num_images} successful")
        logger.info(f"⏱️ Total batch time: {batch_time:.1f}s, avg per image: {batch_time/num_images:.1f}s")
        
        return video_paths

    def extract_images_from_videos_batch(self, video_paths, job_id, user_id, config):
        """Extract first frame from multiple videos and upload"""
        upload_urls = []
        timestamp = int(time.time())
        
        logger.info(f"🖼️ Extracting and uploading {len(video_paths)} images...")
        
        for i, video_path in enumerate(video_paths):
            if video_path is None:
                logger.warning(f"⚠️ Skipping image {i+1} - generation failed")
                upload_urls.append(None)
                continue
                
            try:
                # Create unique filename for each image
                filename = f"wan_{job_id}_{timestamp}_{i+1}.png"
                image_path = Path(f"/tmp/{filename}")
                
                # Extract frame from video
                if self.extract_image_from_video(video_path, image_path):
                    # Upload image
                    storage_path = f"{config['storage_bucket']}/{user_id}/{filename}"
                    upload_path = self.upload_to_supabase(image_path, storage_path)
                    
                    if upload_path:
                        upload_urls.append(upload_path)
                        logger.info(f"✅ Image {i+1} uploaded: {upload_path}")
                    else:
                        logger.error(f"❌ Image {i+1} upload failed")
                        upload_urls.append(None)
                    
                    # Cleanup temp files
                    image_path.unlink(missing_ok=True)
                else:
                    logger.error(f"❌ Image {i+1} frame extraction failed")
                    upload_urls.append(None)
                
                # Cleanup video file
                Path(video_path).unlink(missing_ok=True)
                
            except Exception as e:
                logger.error(f"❌ Image {i+1} processing failed: {e}")
                upload_urls.append(None)
        
        # Filter out failed uploads
        successful_uploads = [url for url in upload_urls if url is not None]
        logger.info(f"📊 Upload summary: {len(successful_uploads)}/{len(video_paths)} images successful")
        
        return successful_uploads

    def extract_image_from_video(self, video_path, output_path):
        """Extract first frame from video for image jobs"""
        try:
            # Use OpenCV to extract first frame
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB and save as PNG
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image.save(output_path, "PNG", quality=95, optimize=True)
                return True
            else:
                print("❌ Failed to extract frame from video")
                return False
                
        except Exception as e:
            print(f"❌ Frame extraction failed: {e}")
            return False

    def upload_to_supabase(self, file_path, storage_path):
        """Upload file to Supabase storage with proper Content-Type"""
        try:
            # Verify file exists before upload
            if not Path(file_path).exists():
                logger.error(f"❌ File does not exist: {file_path}")
                return None
                
            # Get file size for verification
            file_size = Path(file_path).stat().st_size
            
            # Determine content type based on file extension
            if storage_path.endswith('.png'):
                content_type = 'image/png'
            elif storage_path.endswith('.mp4'):
                content_type = 'video/mp4'
            else:
                content_type = 'application/octet-stream'
            
            # Use proper binary upload with explicit Content-Type
            with open(file_path, 'rb') as file:
                file_data = file.read()
            
            headers = {
                'Authorization': f"Bearer {self.supabase_service_key}",
                'Content-Type': content_type,
                'x-upsert': 'true'
            }
            
            response = requests.post(
                f"{self.supabase_url}/storage/v1/object/{storage_path}",
                data=file_data,
                headers=headers,
                timeout=120
            )
            
            if response.status_code in [200, 201]:
                # Extract relative path within bucket
                path_parts = storage_path.split('/', 1)
                if len(path_parts) == 2:
                    relative_path = path_parts[1]
                    return relative_path
                else:
                    logger.warning(f"⚠️ Unexpected storage path format: {storage_path}")
                    return storage_path
            else:
                logger.error(f"❌ Upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Upload error: {e}")
            return None

    def process_job(self, job_data):
        """Process a single job with enhanced debugging"""
        job_id = job_data['jobId']
        job_type = job_data['jobType']
        prompt = job_data['prompt']
        user_id = job_data['userId']
        video_id = job_data.get('videoId')
        image_id = job_data.get('imageId')
        
        # Extract num_images from metadata
        num_images = job_data.get('metadata', {}).get('num_images', 6 if 'image' in job_type else 1)
        
        print(f"\n🚀 === PROCESSING WAN JOB {job_id} ===")
        print(f"📋 Job Type: {job_type}")
        print(f"📝 Prompt: '{prompt}'")
        print(f"👤 User ID: {user_id}")
        print(f"🎬 Video ID: {video_id}")
        print(f"🖼️ Image ID: {image_id}")
        if 'image' in job_type:
            print(f"🔢 Number of Images: {num_images}")
        print(f"📦 Full Job Data: {json.dumps(job_data, indent=2)}")
        
        try:
            config = self.job_type_mapping[job_type]
            start_time = time.time()
            
            print(f"⚙️ Job Configuration: {json.dumps(config, indent=2)}")
            
            if config['content_type'] == 'image' and num_images > 1:
                print(f"🎨 BATCH IMAGE GENERATION MODE")
                # Batch image generation
                video_paths = self.generate_images_batch(prompt, job_type, num_images)
                
                if not any(video_paths):
                    raise Exception("All image generations failed")
                
                # Extract and upload all images
                upload_urls = self.extract_images_from_videos_batch(video_paths, job_id, user_id, config)
                
                if not upload_urls:
                    raise Exception("All image uploads failed")
                
                total_time = time.time() - start_time
                print(f"✅ WAN Job {job_id} completed in {total_time:.1f}s")
                print(f"📁 Generated {len(upload_urls)} images")
                print(f"🪣 Bucket: {config['storage_bucket']}")
                
                # Notify completion with image URLs array
                self.notify_completion(job_id, 'completed', image_urls=upload_urls)
                
            else:
                print(f"🎬 SINGLE GENERATION MODE")
                print(f"🔧 Content Type: {config['content_type']}")
                
                # Single generation (video or single image)
                output_path = self.generate_with_wan21(prompt, job_type)
                
                print(f"🎯 Generation Result: {output_path}")
                
                if not output_path:
                    error_msg = f"WAN 2.1 generation failed - no output file produced"
                    print(f"❌ {error_msg}")
                    raise Exception(error_msg)
                
                upload_path = None
                
                if config['content_type'] == 'image':
                    print(f"🖼️ PROCESSING AS IMAGE")
                    # Single image - extract frame
                    image_path = Path(output_path).with_suffix('.png')
                    if self.extract_image_from_video(output_path, image_path):
                        timestamp = int(time.time())
                        filename = f"wan_{job_id}_{timestamp}.png"
                        storage_path = f"{config['storage_bucket']}/{user_id}/{filename}"
                        upload_path = self.upload_to_supabase(image_path, storage_path)
                        
                        # Cleanup temp files
                        Path(output_path).unlink(missing_ok=True)
                        image_path.unlink(missing_ok=True)
                    else:
                        raise Exception("Frame extraction failed")
                        
                else:  # video
                    print(f"📹 PROCESSING AS VIDEO")
                    # Single video upload
                    timestamp = int(time.time())
                    filename = f"wan_{job_id}_{timestamp}.mp4"
                    storage_path = f"{config['storage_bucket']}/{user_id}/{filename}"
                    print(f"📁 Upload path: {storage_path}")
                    upload_path = self.upload_to_supabase(output_path, storage_path)
                    
                    # Cleanup temp file
                    Path(output_path).unlink(missing_ok=True)
                
                if not upload_path:
                    error_msg = f"File upload to Supabase failed"
                    print(f"❌ {error_msg}")
                    raise Exception(error_msg)
                
                total_time = time.time() - start_time
                print(f"✅ WAN Job {job_id} completed in {total_time:.1f}s")
                print(f"📁 File: {upload_path}")
                print(f"🪣 Bucket: {config['storage_bucket']}")
                
                # Notify completion with single file
                self.notify_completion(job_id, 'completed', file_path=upload_path)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ WAN Job {job_id} FAILED")
            print(f"💥 Error: {error_msg}")
            print(f"📋 Job Type: {job_type}")
            print(f"📝 Prompt: '{prompt}'")
            print(f"🕒 Timestamp: {time.time()}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)
        finally:
            # Cleanup GPU memory and temp files
            torch.cuda.empty_cache()
            gc.collect()
            print(f"🧹 Cleanup completed for job {job_id}")
            print(f"=" * 80)

    def notify_completion(self, job_id, status, file_path=None, image_urls=None, error_message=None):
        """Notify Supabase of job completion with batch support"""
        try:
            callback_data = {
                'jobId': job_id,
                'status': status,
                'errorMessage': error_message
            }
            
            # Add appropriate response data
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
                print(f"✅ Callback sent for WAN job {job_id}")
                if image_urls:
                    print(f"📊 Sent {len(image_urls)} image URLs")
                elif file_path:
                    print(f"📋 Sent file path: {file_path}")
            else:
                print(f"⚠️ Callback failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def poll_queue(self):
        """Poll Redis WAN queue for new jobs - FIXED UPSTASH API"""
        try:
            # FIXED: Use proper Upstash Redis REST API format
            response = requests.post(
                f"{self.redis_url}/rpop/wan_queue",  # POST not GET
                headers={
                    'Authorization': f"Bearer {self.redis_token}",
                    'Content-Type': 'application/json'
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('result'):
                    logger.info(f"📬 Job received from wan_queue: {result['result'][:100]}...")
                    return json.loads(result['result'])
                else:
                    # No job in queue (normal)
                    return None
            else:
                logger.warning(f"⚠️ Redis polling error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            if "timeout" not in str(e).lower():
                logger.warning(f"⚠️ WAN queue poll error: {e}")
            return None

    def run(self):
        """Main WAN worker loop"""
        print("🎬 WAN WORKER READY!")
        print("⚡ Performance: 67-90s per image, ~8-9min for 6-image batch")
        print("📬 Polling wan_queue for image_fast, image_high, video_fast, video_high")
        print("🖼️ NEW: 6-image batch generation for image jobs")
        print("🔧 DEBUG: Enhanced error logging for video generation")
        
        job_count = 0
        
        try:
            while True:
                try:
                    job = self.poll_queue()
                    if job:
                        job_count += 1
                        print(f"📬 WAN Job #{job_count} received")
                        self.process_job(job)
                        print("=" * 60)
                    else:
                        # No job available, wait
                        time.sleep(5)
                        
                except Exception as e:
                    logger.error(f"❌ WAN job processing error: {e}")
                    time.sleep(15)
                    
        except KeyboardInterrupt:
            print("👋 WAN Worker shutting down...")
        finally:
            # Cleanup on shutdown
            torch.cuda.empty_cache()
            gc.collect()
            print("✅ WAN Worker cleanup complete")

if __name__ == "__main__":
    print("🚀 Starting WAN 2.1 Worker - CLEAN SYNTAX VERSION")
    
    # Environment validation
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
    
    try:
        worker = OptimizedWanWorker()
        worker.run()
    except Exception as e:
        print(f"❌ WAN Worker startup failed: {e}")
        exit(1)
