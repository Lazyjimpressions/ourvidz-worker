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
                logger.error(f"❌ {name} missing")
                missing.append(name)
        
        if missing:
            logger.error(f"❌ Missing dependencies: {missing}")
            return False
        
        logger.info("✅ All dependencies available")
        return True

    def check_worker_scripts(self):
        """Verify all worker scripts exist and are executable"""
        logger.info("📂 Checking worker scripts...")
        
        for worker_id, config in self.worker_configs.items():
            script_path = Path(config['script'])
            if script_path.exists():
                if script_path.is_file():
                    logger.info(f"✅ {config['name']}: {script_path}")
                else:
                    logger.error(f"❌ {config['name']}: Path exists but not a file")
                    return False
            else:
                logger.error(f"❌ Missing script: {script_path}")
                return False
        
        return True

    def check_models(self):
        """Verify required models are available"""
        logger.info("🎨 Checking AI models...")
        
        models = {
            'wan_2_1': '/workspace/models/wan2.1-t2v-1.3b',
            'lustify_sdxl': '/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors'
        }
        
        missing = []
        for model_name, model_path in models.items():
            if Path(model_path).exists():
                size = Path(model_path).stat().st_size / (1024**3)
                logger.info(f"✅ {model_name}: {size:.1f}GB")
            else:
                logger.warning(f"⚠️ {model_name}: Missing at {model_path}")
                missing.append(model_name)
        
        if missing:
            logger.warning(f"⚠️ Missing models: {missing} (workers may fail)")
        
        return True  # Don't block startup for missing models

    def start_worker(self, worker_id):
        """Start a specific worker process with proper error handling"""
        config = self.worker_configs[worker_id]
        
        # Check restart limits
        if self.worker_restart_count[worker_id] >= config['max_restarts']:
            logger.error(f"❌ {config['name']}: Max restarts ({config['max_restarts']}) exceeded")
            return False
        
        logger.info(f"🚀 Starting {config['name']} (attempt {self.worker_restart_count[worker_id] + 1})...")
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update({
                'PYTHONUNBUFFERED': '1',
                'CUDA_VISIBLE_DEVICES': '0',
                'TORCH_USE_CUDA_DSA': '1'
            })
            
            # Start worker process
            process = subprocess.Popen(
                [sys.executable, '-u', config['script']],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd='/workspace/ourvidz-worker'
            )
            
            self.workers[worker_id] = process
            self.worker_restart_count[worker_id] += 1
            
            logger.info(f"✅ {config['name']} started (PID: {process.pid})")
            
            # Start output monitoring thread
            monitor_thread = threading.Thread(
                target=self.monitor_worker_output,
                args=(worker_id, process),
                daemon=True,
                name=f"Monitor-{worker_id}"
            )
            monitor_thread.start()
            self.worker_threads[worker_id] = monitor_thread
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start {config['name']}: {e}")
            return False

    def monitor_worker_output(self, worker_id, process):
        """Monitor worker output and log important messages"""
        config = self.worker_configs[worker_id]
        logger.info(f"📊 Starting output monitor for {config['name']}")
        
        while self.running and process.poll() is None:
            try:
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        
                        # Filter and log important messages
                        if any(keyword in line for keyword in [
                            '✅', '❌', '🚀', '⚡', '🎉', '⚠️',
                            'ERROR', 'WARN', 'CRITICAL', 'Exception',
                            'completed', 'failed', 'started', 'ready'
                        ]):
                            logger.info(f"[{config['name']}] {line}")
                        elif 'Job #' in line or 'Processing' in line:
                            logger.info(f"[{config['name']}] {line}")
                        else:
                            # Debug level for routine messages
                            logger.debug(f"[{config['name']}] {line}")
                
                time.sleep(0.1)
                
            except Exception as e:
                if self.running:  # Only log if we're not shutting down
                    logger.error(f"❌ Error monitoring {worker_id}: {e}")
                break
        
        logger.info(f"📊 Output monitor for {config['name']} stopped")

    def check_worker_health(self):
        """Check health of all workers and restart if needed"""
        healthy_workers = 0
        
        for worker_id, process in self.workers.items():
            config = self.worker_configs[worker_id]
            
            if process.poll() is None:  # Still running
                healthy_workers += 1
                logger.debug(f"✅ {config['name']} healthy (PID: {process.pid})")
            else:
                exit_code = process.returncode
                logger.warning(f"⚠️ {config['name']} stopped (exit code: {exit_code})")
                
                # Check if we should restart
                if self.worker_restart_count[worker_id] < config['max_restarts']:
                    logger.info(f"🔄 Scheduling restart for {config['name']} in {config['restart_delay']}s...")
                    
                    # Schedule restart in a separate thread
                    restart_thread = threading.Thread(
                        target=self.delayed_restart,
                        args=(worker_id, config['restart_delay']),
                        daemon=True,
                        name=f"Restart-{worker_id}"
                    )
                    restart_thread.start()
                else:
                    logger.error(f"❌ {config['name']}: Max restarts exceeded, not restarting")
                
        return healthy_workers

    def delayed_restart(self, worker_id, delay):
        """Restart a worker after a delay"""
        if not self.running:
            return
            
        logger.info(f"⏳ Waiting {delay}s before restarting {worker_id}...")
        time.sleep(delay)
        
        if self.running:  # Check if we're still running
            logger.info(f"🔄 Attempting restart of {worker_id}...")
            if self.start_worker(worker_id):
                logger.info(f"✅ {worker_id} restarted successfully")
            else:
                logger.error(f"❌ Failed to restart {worker_id}")

    def get_system_status(self):
        """Get current system status"""
        status = {
            'workers_running': 0,
            'workers_total': len(self.worker_configs),
            'restart_counts': self.worker_restart_count.copy(),
            'vram_info': None
        }
        
        # Count running workers
        for process in self.workers.values():
            if process.poll() is None:
                status['workers_running'] += 1
        
        # Get VRAM info if available
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                status['vram_info'] = {
                    'allocated': allocated,
                    'reserved': reserved,
                    'total': total,
                    'free': total - reserved
                }
        except:
            pass
        
        return status

    def log_status_update(self):
        """Log periodic status update"""
        status = self.get_system_status()
        
        logger.info(f"📊 STATUS: {status['workers_running']}/{status['workers_total']} workers running")
        
        if status['vram_info']:
            vram = status['vram_info']
            logger.info(f"🔥 VRAM: {vram['allocated']:.1f}GB allocated, "
                       f"{vram['free']:.1f}GB free ({vram['total']:.1f}GB total)")
        
        # Log restart counts if any
        restarts = [f"{k}:{v}" for k, v in status['restart_counts'].items() if v > 0]
        if restarts:
            logger.info(f"🔄 Restarts: {', '.join(restarts)}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📶 Received signal {signum}, shutting down...")
        self.shutdown()

    def shutdown(self):
        """Graceful shutdown of all workers"""
        logger.info("🛑 Shutting down dual worker orchestrator...")
        self.running = False
        
        # Terminate all workers
        for worker_id, process in self.workers.items():
            config = self.worker_configs[worker_id]
            if process.poll() is None:
                logger.info(f"🛑 Stopping {config['name']}...")
                
                try:
                    # Send SIGTERM first
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=10)
                        logger.info(f"✅ {config['name']} stopped gracefully")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"⚠️ Force killing {config['name']}...")
                        process.kill()
                        process.wait(timeout=5)
                        logger.info(f"✅ {config['name']} force killed")
                        
                except Exception as e:
                    logger.error(f"❌ Error stopping {config['name']}: {e}")
        
        # Wait for monitor threads to finish
        for worker_id, thread in self.worker_threads.items():
            if thread.is_alive():
                thread.join(timeout=5)
        
        logger.info("👋 Dual worker orchestrator shutdown complete")

    def run(self):
        """Main orchestrator loop"""
        logger.info("🎯 Starting dual worker orchestrator...")
        
        # Comprehensive validation
        if not self.validate_environment():
            logger.error("❌ Environment validation failed")
            return False
        
        if not self.check_dependencies():
            logger.error("❌ Dependency validation failed")
            return False
        
        if not self.check_worker_scripts():
            logger.error("❌ Worker script validation failed")
            return False
        
        if not self.check_models():
            logger.warning("⚠️ Some models missing but continuing...")
        
        # Start all workers
        logger.info("🚀 Starting all workers...")
        failed_workers = []
        
        for worker_id in self.worker_configs.keys():
            if not self.start_worker(worker_id):
                failed_workers.append(worker_id)
        
        if len(failed_workers) == len(self.worker_configs):
            logger.error("❌ All workers failed to start")
            return False
        elif failed_workers:
            logger.warning(f"⚠️ Some workers failed to start: {failed_workers}")
        
        logger.info("🎉 Orchestrator startup complete!")
        logger.info("🔥 RTX 6000 ADA dual worker system operational")
        
        # Main monitoring loop
        last_status_log = time.time()
        health_check_interval = 30  # seconds
        status_log_interval = 300   # 5 minutes
        
        try:
            while self.running:
                # Health check
                healthy_count = self.check_worker_health()
                
                if healthy_count == 0:
                    logger.error("❌ No healthy workers remaining")
                    # Don't exit immediately - workers might restart
                    time.sleep(60)  # Wait a minute for potential restarts
                    continue
                
                # Periodic status logging
                if time.time() - last_status_log > status_log_interval:
                    self.log_status_update()
                    last_status_log = time.time()
                
                time.sleep(health_check_interval)
                
        except KeyboardInterrupt:
            logger.info("👋 Received keyboard interrupt")
        except Exception as e:
            logger.error(f"❌ Orchestrator error: {e}")
        finally:
            self.shutdown()
        
        return True

if __name__ == "__main__":
    print("🚀 OurVidz Dual Worker Orchestrator v2.0")
    print("🔥 RTX 6000 ADA (48GB VRAM) Concurrent Operation")
    print("=" * 80)
    
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
        orchestrator = DualWorkerOrchestrator()
        success = orchestrator.run()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Orchestrator startup failed: {e}")
        exit(1)
