# dual_orchestrator.py - UPDATED FOR GRACEFUL VALIDATION
# Manages both LUSTIFY SDXL and WAN 2.1 workers concurrently
# Critical Fix: Graceful SDXL validation + Enhanced WAN worker support
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
                'job_types': ['sdxl_image_fast', 'sdxl_image_high'],
                'expected_vram': '10-15GB',
                'restart_delay': 10,
                'generation_time': '3-8s',
                'status': 'Working ✅'
            },
            'wan': {
                'script': 'wan_worker.py', 
                'name': 'Enhanced WAN Worker',
                'queue': 'wan_queue',
                'job_types': ['image_fast', 'image_high', 'video_fast', 'video_high',
                             'image7b_fast_enhanced', 'image7b_high_enhanced', 
                             'video7b_fast_enhanced', 'video7b_high_enhanced'],
                'expected_vram': '15-30GB',
                'restart_delay': 15,
                'generation_time': '67-294s',
                'status': 'Enhanced with Qwen 7B ✅'
            }
        }
        
        logger.info("🎭 Dual Worker Orchestrator initialized")
        logger.info("🎨 SDXL: Fast image generation (3-8s)")
        logger.info("🎬 Enhanced WAN: Video + AI enhancement (67-294s)")
        logger.info("🔧 FIXED: Graceful validation for production stability")

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
                logger.error(f"❌ Missing worker script: {config['script']}")
        
        if missing_files:
            logger.error(f"❌ Missing worker scripts: {missing_files}")
            return False
        else:
            logger.info("✅ All worker scripts found")
            
        # Check GPU
        try:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"✅ GPU: {device_name} ({total_vram:.1f}GB)")
                
                if total_vram < 40:
                    logger.warning(f"⚠️ GPU has {total_vram:.1f}GB, dual workers need 45GB+ for concurrent operation")
                else:
                    logger.info(f"✅ GPU capacity sufficient for dual workers")
                    
            else:
                logger.error("❌ CUDA not available")
                return False
                
        except Exception as e:
            logger.error(f"❌ GPU check failed: {e}")
            return False
            
        # Check SDXL imports (graceful handling - let workers manage their own imports)
        try:
            from diffusers import StableDiffusionXLPipeline
            logger.info("✅ SDXL imports confirmed working")
        except ImportError as e:
            logger.warning(f"⚠️ SDXL imports failed in orchestrator: {e}")
            logger.info("📝 Will let SDXL worker handle its own imports")
            # Don't fail here - let workers handle their own dependencies
            
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
        else:
            logger.info("✅ All environment variables configured")
            
        # Validate parameter fix implementation
        logger.info("🔧 Validating parameter fix implementation...")
        wan_script_path = Path('wan_worker.py')
        if wan_script_path.exists():
            with open(wan_script_path, 'r') as f:
                content = f.read()
                if "'filePath': file_path" in content:
                    logger.info("✅ WAN worker parameter fix confirmed")
                elif "'outputUrl': file_path" in content:
                    logger.error("❌ WAN worker still uses old parameter name")
                    return False
                else:
                    logger.warning("⚠️ Could not verify WAN worker parameter fix")
            
        logger.info("✅ Environment validation passed")
        return True

    def start_worker(self, worker_id):
        """Start a specific worker process"""
        config = self.workers[worker_id]
        
        logger.info(f"🚀 Starting {config['name']}...")
        logger.info(f"📋 Job Types: {', '.join(config['job_types'])}")
        logger.info(f"⚡ Performance: {config['generation_time']}")
        
        try:
            # Set up environment with persistent Python path
            env = os.environ.copy()
            persistent_deps = "/workspace/python_deps/lib/python3.11/site-packages"
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{persistent_deps}:{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = persistent_deps
            
            logger.info(f"🔧 Setting PYTHONPATH: {env['PYTHONPATH']}")
            
            # Start worker process with proper environment
            process = subprocess.Popen(
                [sys.executable, config['script']],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env  # ✅ Pass environment with PYTHONPATH
            )
            
            self.processes[worker_id] = {
                'process': process,
                'config': config,
                'start_time': time.time(),
                'restart_count': 0,
                'job_count': 0
            }
            
            logger.info(f"✅ {config['name']} started (PID: {process.pid})")
            logger.info(f"📊 Status: {config['status']}")
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
                    
                # Track job completions
                if "Job #" in line and "received" in line:
                    worker_info['job_count'] += 1
                    
                # Log worker output with prefix
                logger.info(f"[{worker_id.upper()}] {line.strip()}")
                
                # Look for parameter fix confirmations
                if "filePath =" in line and worker_id == 'wan':
                    logger.info(f"✅ WAN parameter fix confirmed in operation")
                
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
        logger.info(f"📊 Worker processed {worker_info['job_count']} jobs before restart")
        
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
            logger.info(f"📊 Final job count: {worker_info['job_count']}")
            
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
                total_jobs = 0
                
                for worker_id, worker_info in self.processes.items():
                    if worker_info['process'].poll() is None:
                        uptime = time.time() - worker_info['start_time']
                        job_count = worker_info['job_count']
                        total_jobs += job_count
                        active_workers.append(f"{worker_id}({uptime:.0f}s/{job_count}j)")
                
                if active_workers:
                    logger.info(f"💚 Active workers: {', '.join(active_workers)} | Total jobs: {total_jobs}")
                else:
                    logger.warning("⚠️ No active workers")
                
                # Check GPU memory
                try:
                    import torch
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        utilization = (allocated / total) * 100
                        logger.info(f"🔥 GPU Memory: {allocated:.1f}GB / {total:.0f}GB ({utilization:.1f}% used)")
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
        logger.info("🔧 GRACEFUL VALIDATION VERSION - Production Ready")
        logger.info("=" * 70)
        
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
        logger.info("=" * 70)
        logger.info("🎨 SDXL Worker: sdxl_queue → sdxl_image_fast, sdxl_image_high")
        logger.info("  ⚡ Performance: 3-8s generation")
        logger.info("")
        logger.info("🎬 Enhanced WAN Worker: wan_queue → 8 job types")
        logger.info("  📝 Standard: image_fast, image_high, video_fast, video_high")
        logger.info("  ✨ Enhanced: image7b_fast_enhanced, image7b_high_enhanced, video7b_fast_enhanced, video7b_high_enhanced")
        logger.info("  ⚡ Performance: 67-294s generation (includes Qwen 7B enhancement)")
        logger.info("")
        logger.info("💡 Both workers monitoring their respective queues")
        logger.info("🔧 Fixed: Graceful SDXL validation, Enhanced WAN with Qwen 7B")
        logger.info("=" * 70)
        
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
    logger.info("🚀 Starting OurVidz Dual Worker System - GRACEFUL VALIDATION VERSION")
    
    try:
        orchestrator = DualWorkerOrchestrator()
        success = orchestrator.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Orchestrator startup failed: {e}")
        sys.exit(1)
