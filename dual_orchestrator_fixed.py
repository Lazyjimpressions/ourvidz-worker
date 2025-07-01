# dual_orchestrator_fixed.py - Production Dual Worker Orchestrator
# Manages concurrent LUSTIFY SDXL + Wan 2.1 operation with proper error handling

import os
import sys
import subprocess
import threading
import time
import signal
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/orchestrator.log')
    ]
)
logger = logging.getLogger(__name__)

class DualWorkerOrchestrator:
    def __init__(self):
        """Initialize dual worker orchestrator for RTX 6000 ADA"""
        print("🚀 OURVIDZ DUAL WORKER ORCHESTRATOR v2.0")
        print("🔥 RTX 6000 ADA (48GB VRAM) - CONCURRENT OPERATION")
        print("🎨 LUSTIFY SDXL: 10.5GB peak (3-8s generation)")
        print("🎬 Wan 2.1: 15-20GB peak (70-300s generation)")
        print("⚡ Total capacity: ~32GB headroom available")
        
        self.workers = {}
        self.worker_threads = {}
        self.running = True
        self.worker_restart_count = {}
        
        # Worker configurations
        self.worker_configs = {
            'sdxl_worker': {
                'script': '/workspace/ourvidz-worker/sdxl_worker_fixed.py',
                'name': 'LUSTIFY SDXL Worker',
                'queue': 'sdxl_queue',
                'expected_vram': '10.5GB',
                'restart_delay': 30,  # seconds
                'max_restarts': 5,
                'job_types': ['sdxl_image_fast', 'sdxl_image_high', 'sdxl_image_premium', 'sdxl_img2img']
            },
            'wan_worker': {
                'script': '/workspace/ourvidz-worker/worker.py',
                'name': 'Wan 2.1 Worker', 
                'queue': 'wan_queue',
                'expected_vram': '15-20GB',
                'restart_delay': 60,  # seconds
                'max_restarts': 3,
                'job_types': ['video_fast', 'video_high', 'image_fast', 'image_high']
            }
        }
        
        # Initialize restart counters
        for worker_id in self.worker_configs:
            self.worker_restart_count[worker_id] = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("🎯 Dual Worker Orchestrator initialized")

    def validate_environment(self):
        """Comprehensive environment validation"""
        logger.info("🔍 Validating dual worker environment...")
        
        # Check Python version
        python_version = sys.version_info
        logger.info(f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check GPU capacity
        try:
            import torch
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✅ GPU: {gpu_name} ({total_vram:.1f}GB VRAM)")
                logger.info(f"✅ PyTorch: {torch.__version__}")
                
                # Test GPU allocation
                test_tensor = torch.randn(1000, 1000, device='cuda')
                allocated = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"✅ GPU test: {allocated:.3f}GB allocated")
                del test_tensor
                torch.cuda.empty_cache()
                
                if total_vram >= 40:
                    logger.info("✅ RTX 6000 ADA detected - concurrent operation enabled")
                    return True
                else:
                    logger.warning(f"⚠️ GPU has {total_vram:.1f}GB - may need sequential operation")
                    return True  # Still try to run
            else:
                logger.error("❌ CUDA not available")
                return False
        except Exception as e:
            logger.error(f"❌ GPU validation failed: {e}")
            return False

    def check_dependencies(self):
        """Verify critical dependencies are available"""
        logger.info("📦 Checking dependencies...")
        
        dependencies = {
            'torch': 'PyTorch',
            'diffusers': 'Diffusers',
            'transformers': 'Transformers',
            'requests': 'Requests',
            'PIL': 'Pillow'
        }
        
        missing = []
        for module, name in dependencies.items():
            try:
                __import__(module)
                logger.debug(f"✅ {name} available")
            except ImportError:
                logger.
