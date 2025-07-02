# dual_orchestrator.py - Production Dual Worker Manager
# Manages both LUSTIFY SDXL and WAN 2.1 workers concurrently
# Optimized for RTX 6000 ADA 48GB VRAM capacity

import os
import sys
import time
import signal
import subprocess
import threading
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DualWorkerOrchestrator:
    def __init__(self):
        """Initialize dual worker orchestrator"""
        self.processes = {}
        self.shutdown_event = threading.Event()
        
        # Worker configurations
        self.workers = {
            'sdxl': {
                'script': 'sdxl_worker.py',
                'name': 'LUSTIFY SDXL Worker',
                'queue': 'sdxl_queue',
                'expected_vram': '10-15GB',
                'restart_delay': 10
            },
            'wan': {
                'script': 'wan_worker.py', 
                'name': 'WAN 2.1 Worker',
                'queue': 'wan_queue',
                'expected_vram': '15-30GB',
                'restart_delay': 15
            }
        }
        
        logger.info("🎭 Dual Worker Orchestrator initialized")
        logger.info("🎨 SDXL: Fast image generation (3-8s)")
        logger.info("🎬 WAN: Video + backup images (67-354s)")

    def validate_environment(self):
        """Validate environment for dual worker operation"""
        logger.info("🔍 Validating dual worker environment...")
        
        # CRITICAL: Check PyTorch version first (prevent cascade failures)
        try:
            import torch
            current_version = torch.__version__
            current_cuda = torch.version.cuda
            
            logger.info(f"🔧 PyTorch: {current_version}")
            logger.info(f"🔧 CUDA: {current_cuda}")
            
            # Verify we have the stable working versions
            if not current_version.startswith('2.4.1'):
                logger.error(f"❌ WRONG PyTorch version: {current_version} (need 2.4.1+cu124)")
                logger.error("❌ DO NOT PROCEED - version cascade detected!")
                return False
                
            if current_cuda != '12.4':
                logger.error(f"❌ WRONG CUDA version: {current_cuda} (need 12.4)")
                logger.error("❌ DO NOT PROCEED - CUDA version mismatch!")
                return False
                
            logger.info("✅ PyTorch/CUDA versions confirmed stable")
            
        except ImportError:
            logger.error("❌ PyTorch not available")
            return False
        
        # Check Python files exist
        missing_files = []
        for worker_id, config in self.workers.items():
            script_path = Path(config['script'])
            if not script_path.exists():
                missing_files.append(config['script'])
        
        if missing_files:
            logger.error(f"❌ Missing worker scripts: {missing_files}")
            return False
            
        # Check GPU
        try:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"✅ GPU: {device_name} ({total_vram:.1f}GB)")
                
                if total_vram < 40:
                    logger.warning(f"⚠️ GPU has {total_vram:.1f}GB, dual workers need 45GB+ for concurrent operation")
                    
            else:
                logger.error("❌ CUDA not available")
                return False
                
        except Exception as e:
            logger.error(f"❌ GPU check failed: {e}")
            return False
            
        # Check SDXL imports (should already work from previous session)
        try:
            from diffusers import StableDiffusionXLPipeline
            logger.info("✅ SDXL imports confirmed working")
        except ImportError as e:
            logger.error(f"❌ SDXL imports failed: {e}")
            logger.error("❌ DO NOT INSTALL PACKAGES - this will break PyTorch!")
            return False
            
        # Check environment variables
        required_vars = [
            'SUPABASE_URL', 
            'SUPABASE_SERVICE_KEY', 
            'UPSTASH_REDIS_REST_URL', 
            'UPSTASH_REDIS_REST_TOKEN'
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"❌ Missing environment variables: {missing_vars}")
            return False
            
        logger.info("✅ Environment validation passed")
        return True

    def start_worker(self, worker_id):
        """Start a specific worker process"""
        config = self.workers[worker_id]
        
        logger.info(f"🚀 Starting {config['name']}...")
        
        try:
            # Start worker process
            process = subprocess.Popen(
                [sys.executable, config['script']],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes[worker_id] = {
                'process': process,
                'config': config,
                'start_time': time.time(),
                'restart_count': 0
            }
            
            logger.info(f"✅ {config['name']} started (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start {config['name']}: {e}")
            return False

    def monitor_worker(self, worker_id):
        """Monitor a worker process and handle output"""
        worker_info = self.processes[worker_id]
        process = worker_info['process']
        config = worker_info['config']
        
        logger.info(f"👁️ Monitoring {config['name']}...")
        
        try:
            # Read output line by line
            for line in iter(process.stdout.readline, ''):
                if self.shutdown_event.is_set():
                    break
                    
                # Log worker output with prefix
                logger.info(f"[{worker_id.upper()}] {line.strip()}")
                
            # Process ended
            process.wait()
            return_code = process.returncode
            
            if return_code != 0 and not self.shutdown_event.is_set():
                logger.warning(f"⚠️ {config['name']} exited with code {return_code}")
                self.handle_worker_restart(worker_id)
            else:
                logger.info(f"✅ {config['name']} exited normally")
                
        except Exception as e:
            logger.error(f"❌ Error monitoring {config['name']}: {e}")
            self.handle_worker_restart(worker_id)

    def handle_worker_restart(self, worker_id):
        """Handle worker restart logic"""
        if self.shutdown_event.is_set():
            return
            
        worker_info = self.processes[worker_id]
        config = worker_info['config']
        
        worker_info['restart_count'] += 1
        restart_delay = config['restart_delay'] * worker_info['restart_count']
        
        logger.warning(f"🔄 Restarting {config['name']} in {restart_delay}s (attempt #{worker_info['restart_count']})")
        
        if worker_info['restart_count'] > 5:
            logger.error(f"❌ {config['name']} failed too many times, giving up")
            return
            
        # Wait before restart
        time.sleep(restart_delay)
        
        if not self.shutdown_event.is_set():
            self.start_worker(worker_id)
            # Start new monitoring thread
            monitor_thread = threading.Thread(
                target=self.monitor_worker, 
                args=(worker_id,), 
                daemon=True
            )
            monitor_thread.start()

    def start_all_workers(self):
        """Start all workers with monitoring"""
        logger.info("🎬 Starting all workers...")
        
        for worker_id in self.workers.keys():
            if self.start_worker(worker_id):
                # Start monitoring thread for each worker
                monitor_thread = threading.Thread(
                    target=self.monitor_worker, 
                    args=(worker_id,), 
                    daemon=True
                )
                monitor_thread.start()
                
                # Stagger startup to avoid resource conflicts
                time.sleep(5)
            else:
                logger.error(f"❌ Failed to start {worker_id} worker")
                return False
                
        return True

    def stop_all_workers(self):
        """Gracefully stop all workers"""
        logger.info("🛑 Stopping all workers...")
        self.shutdown_event.set()
        
        for worker_id, worker_info in self.processes.items():
            config = worker_info['config']
            process = worker_info['process']
            
            logger.info(f"🛑 Stopping {config['name']}...")
            
            try:
                # Send SIGTERM for graceful shutdown
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    logger.info(f"✅ {config['name']} stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    logger.warning(f"⚠️ Force killing {config['name']}")
                    process.kill()
                    process.wait()
                    
            except Exception as e:
                logger.error(f"❌ Error stopping {config['name']}: {e}")

    def status_monitor(self):
        """Background thread to monitor system status"""
        logger.info("📊 Starting status monitor...")
        
        while not self.shutdown_event.is_set():
            try:
                # Check worker processes
                active_workers = []
                for worker_id, worker_info in self.processes.items():
                    if worker_info['process'].poll() is None:
                        uptime = time.time() - worker_info['start_time']
                        active_workers.append(f"{worker_id}({uptime:.0f}s)")
                
                if active_workers:
                    logger.info(f"💚 Active workers: {', '.join(active_workers)}")
                else:
                    logger.warning("⚠️ No active workers")
                
                # Check GPU memory
                try:
                    import torch
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        logger.info(f"🔥 GPU Memory: {allocated:.1f}GB / {total:.0f}GB")
                except:
                    pass
                    
                # Wait before next check
                time.sleep(60)  # Status check every minute
                
            except Exception as e:
                logger.error(f"❌ Status monitor error: {e}")
                time.sleep(30)

    def run(self):
        """Main orchestrator run loop"""
        logger.info("🎭 DUAL WORKER ORCHESTRATOR STARTING")
        logger.info("=" * 60)
        
        # Validate environment
        if not self.validate_environment():
            logger.error("❌ Environment validation failed")
            return False
            
        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, shutting down...")
            self.stop_all_workers()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start status monitoring
        status_thread = threading.Thread(target=self.status_monitor, daemon=True)
        status_thread.start()
        
        # Start all workers
        if not self.start_all_workers():
            logger.error("❌ Failed to start workers")
            return False
            
        logger.info("🎉 DUAL WORKER SYSTEM READY!")
        logger.info("🎨 SDXL: sdxl_queue → sdxl_image_fast, sdxl_image_high")
        logger.info("🎬 WAN: wan_queue → image_fast, image_high, video_fast, video_high")
        logger.info("💡 Both workers monitoring their respective queues")
        logger.info("=" * 60)
        
        # Main loop - keep orchestrator alive
        try:
            while not self.shutdown_event.is_set():
                time.sleep(10)
                
                # Check if any critical workers died
                dead_workers = []
                for worker_id, worker_info in self.processes.items():
                    if worker_info['process'].poll() is not None:
                        dead_workers.append(worker_id)
                
                if dead_workers and not self.shutdown_event.is_set():
                    logger.warning(f"⚠️ Dead workers detected: {dead_workers}")
                    
        except KeyboardInterrupt:
            logger.info("👋 Orchestrator interrupted by user")
        finally:
            self.stop_all_workers()
            
        logger.info("✅ Dual Worker Orchestrator shutdown complete")
        return True

if __name__ == "__main__":
    logger.info("🚀 Starting OurVidz Dual Worker System")
    
    try:
        orchestrator = DualWorkerOrchestrator()
        success = orchestrator.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Orchestrator startup failed: {e}")
        sys.exit(1)
