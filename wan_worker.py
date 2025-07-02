# wan_worker.py - BATCH GENERATION VERSION
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
                'supports_batch': False  # Videos remain single generation
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
                'supports_batch': False  # Videos remain single generation
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
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            generation_time = time.time() - start_time
            
            # Log GPU memory after
            self.log_gpu_memory()
            
            if result.returncode == 0:
                if image_index is not None:
                    print(f"✅ Generation {image_index} successful in {generation_time:.1f}s")
                else:
                    print(f"✅ Generation successful in {generation_time:.1f}s")
                
                # Verify output file exists
                if temp_video_path.exists():
                    file_size = temp_video_path.stat().st_size / 1024  # KB
                    print(f"📁 Output file: {file_size:.0f}KB")
                    return str(temp_video_path)
                else:
                    print("❌ Output file not found")
                    return None
            else:
                print(f"❌ Generation failed (code {result.returncode})")
                if result.stderr:
                    print(f"Error: {result.stderr[-500:]}")
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
            raise ValueError(f"Batch generation only supporte
