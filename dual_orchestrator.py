# dual_orchestrator.py - RTX 6000 ADA Dual Worker Orchestrator
# Manages concurrent LUSTIFY SDXL + Wan 2.1 operation
import os
import sys
import subprocess
import threading
import time
import signal
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DualWorkerOrchestrator:
    def __init__(self):
        """Initialize dual worker orchestrator for RTX 6000 ADA"""
        print("🚀 OURVIDZ DUAL WORKER ORCHESTRATOR")
        print("🔥 RTX 6000 ADA (48GB VRAM) - CONCURRENT OPERATION")
        print("🎨 LUSTIFY SDXL: 10.5GB peak (3.2s generation)")
        print("🎬 Wan 2.1: 15-20GB peak (3-6min generation)")
        print("⚡ Total capacity: 32GB headroom available")
        
        self.workers = {}
        self.worker_threads = {}
        self.running = True
        
        # Worker configurations
        self.worker_configs = {
            'sdxl_worker': {
                'script': '/workspace/models/sdxl-lustify/sdxl_worker.py',
                'name': 'LUSTIFY SDXL Worker',
                'queue': 'sdxl_queue',
                'expected_vram': '10.5GB',
                'job_types': ['sdxl_image_fast', 'sdxl_image_high', 'sdxl_image_premium', 'sdxl_img2img']
            },
            'wan_worker': {
                'script': '/workspace/ourvidz-worker/worker.py',
                'name': 'Wan 2.1 Worker', 
                'queue': 'wan_queue',
                'expected_vram': '15-20GB',
                'job_types': ['video_fast', 'video_high', 'image_fast', 'image_high']
            }
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("🎯 Dual Worker Orchestrator initialized")

    def validate_environment(self):
        """Validate system requirements for dual worker operation"""
        logger.info("🔍 Validating dual worker environment...")
        
        # Check GPU capacity
        try:
            import torch
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✅ GPU: {gpu_name} ({total_vram:.1f}GB VRAM)")
                
                if total_vram >= 40:
                    logger.info("✅ RTX 6000 ADA detected - concurrent operation enabled")
                    return True
                else:
                    logger.warning(f"⚠️ GPU has {total_vram:.1f}GB - may need sequential operation")
                    return False
            else:
                logger.error("❌ CUDA not available")
                return False
        except Exception as e:
            logger.error(f"❌ GPU validation failed: {e}")
            return False

    def check_worker_scripts(self):
        """Verify all worker scripts exist"""
        logger.info("📂 Checking worker scripts...")
        
        for worker_id, config in self.worker_configs.items():
            script_path = Path(config['script'])
            if script_path.exists():
                logger.info(f"✅ {config['name']}: {script_path}")
            else:
                logger.error(f"❌ Missing script: {script_path}")
                return False
        
        return True

    def start_worker(self, worker_id):
        """Start a specific worker process"""
        config = self.worker_configs[worker_id]
        logger.info(f"🚀 Starting {config['name']}...")
        
        try:
            # Start worker process
            process = subprocess.Popen(
                [sys.executable, config['script']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.workers[worker_id] = process
            logger.info(f"✅ {config['name']} started (PID: {process.pid})")
            
            # Start output monitoring thread
            monitor_thread = threading.Thread(
                target=self.monitor_worker_output,
                args=(worker_id, process),
                daemon=True
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
        
        while self.running and process.poll() is None:
            try:
                # Read stdout
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        # Filter important messages
                        if any(keyword in line for keyword in ['✅', '❌', '🚀', '⚡', '🎉', 'ERROR', 'WARN']):
                            logger.info(f"[{config['name']}] {line.strip()}")
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error monitoring {worker_id}: {e}")
                break

    def check_worker_health(self):
        """Check health of all workers"""
        healthy_workers = 0
        
        for worker_id, process in self.workers.items():
            config = self.worker_configs[worker_id]
            
            if process.poll() is None:  # Still running
                healthy_workers += 1
                logger.debug(f"✅ {config['name']} healthy (PID: {process.pid})")
            else:
                logger.warning(f"⚠️ {config['name']} stopped (exit code: {process.returncode})")
                
                # Attempt to restart
                logger.info(f"🔄 Restarting {config['name']}...")
                if self.start_worker(worker_id):
                    healthy_workers += 1
                
        return healthy_workers

    def get_system_status(self):
        """Get current system status"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                return {
                    'vram_allocated': allocated,
                    'vram_reserved': reserved,
                    'vram_total': total,
                    'vram_free': total - reserved,
                    'workers_running': len([p for p in self.workers.values() if p.poll() is None])
                }
        except:
            pass
        
        return None

    def log_status_update(self):
        """Log periodic status update"""
        status = self.get_system_status()
        if status:
            logger.info(f"🔥 VRAM: {status['vram_allocated']:.1f}GB allocated, "
                       f"{status['vram_free']:.1f}GB free, "
                       f"{status['workers_running']} workers active")

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
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    logger.info(f"✅ {config['name']} stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ Force killing {config['name']}...")
                    process.kill()
        
        logger.info("👋 Dual worker orchestrator shutdown complete")

    def run(self):
        """Main orchestrator loop"""
        logger.info("🎯 Starting dual worker orchestrator...")
        
        # Validate environment
        if not self.validate_environment():
            logger.error("❌ Environment validation failed")
            return False
        
        # Check worker scripts
        if not self.check_worker_scripts():
            logger.error("❌ Worker script validation failed")
            return False
        
        # Start all workers
        logger.info("🚀 Starting all workers...")
        failed_workers = []
        
        for worker_id in self.worker_configs.keys():
            if not self.start_worker(worker_id):
                failed_workers.append(worker_id)
        
        if failed_workers:
            logger.error(f"❌ Failed to start workers: {failed_workers}")
            return False
        
        logger.info("🎉 All workers started successfully!")
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
                    break
                
                # Periodic status logging
                if time.time() - last_status_log > status_log_interval:
                    self.log_status_update()
                    last_status_log = time.time()
                
                time.sleep(health_check_interval)
                
        except KeyboardInterrupt:
            logger.info("👋 Received shutdown signal")
        except Exception as e:
            logger.error(f"❌ Orchestrator error: {e}")
        finally:
            self.shutdown()
        
        return True

if __name__ == "__main__":
    print("🚀 OurVidz Dual Worker Orchestrator")
    print("🔥 RTX 6000 ADA (48GB VRAM) Concurrent Operation")
    print("=" * 60)
    
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
