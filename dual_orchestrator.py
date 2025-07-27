# dual_orchestrator.py - UPDATED FOR TRIPLE WORKER SYSTEM (SDXL + WAN + CHAT)
# Manages LUSTIFY SDXL, WAN 2.1, and Chat workers concurrently
# Critical Fix: Graceful SDXL validation + Enhanced WAN worker support + Chat worker integration
# Optimized for RTX 6000 ADA 48GB VRAM capacity

import os
import sys
import time
import signal
import subprocess
import threading
import logging
from pathlib import Path

# CRITICAL: Set persistent PYTHONPATH for RunPod environment
PYTHON_DEPS_PATH = '/workspace/python_deps/lib/python3.11/site-packages'
if PYTHON_DEPS_PATH not in sys.path:
    sys.path.insert(0, PYTHON_DEPS_PATH)

# Set environment variables for worker processes
os.environ['PYTHONPATH'] = f"{PYTHON_DEPS_PATH}:{os.environ.get('PYTHONPATH', '')}"
os.environ['HF_HOME'] = '/workspace/models/huggingface_cache'

print(f"✅ PYTHONPATH set: {os.environ['PYTHONPATH']}")
print(f"✅ HF_HOME set: {os.environ['HF_HOME']}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class DualWorkerOrchestrator:
    def __init__(self):
        """Initialize triple worker orchestrator"""
        self.processes = {}
        self.shutdown_event = threading.Event()
        
        # Worker configurations - UPDATED FOR TRIPLE WORKER SYSTEM
        self.workers = {
            'sdxl': {
                'script': 'sdxl_worker.py',
                'name': 'LUSTIFY SDXL Worker',
                'queue': 'sdxl_queue',
                'job_types': ['sdxl_image_fast', 'sdxl_image_high'],
                'expected_vram': '10-15GB',
                'restart_delay': 10,
                'generation_time': '3-8s',
                'status': 'Working ✅',
                'port': 7860,  # ✅ ADDED: Port for Flask API
                'priority': 1   # ✅ ADDED: Startup priority (1 = highest)
            },
            'chat': {
                'script': 'chat_worker.py',
                'name': 'Chat Worker (Qwen Instruct)',
                'queue': 'chat_queue',
                'job_types': ['chat_enhance', 'chat_conversation', 'admin_utilities'],
                'expected_vram': '15-20GB',
                'restart_delay': 12,
                'generation_time': '5-15s',
                'status': 'Qwen 2.5-7B Instruct Service ✅',
                'port': 7861,  # ✅ ADDED: Port for Flask API
                'priority': 2   # ✅ ADDED: Startup priority (2 = medium)
            },
            'wan': {
                'script': 'wan_worker.py', 
                'name': 'Enhanced WAN Worker (Qwen 7B + FLF2V/T2V)',
                'queue': 'wan_queue',
                'job_types': ['image_fast', 'image_high', 'video_fast', 'video_high',
                             'image7b_fast_enhanced', 'image7b_high_enhanced', 
                             'video7b_fast_enhanced', 'video7b_high_enhanced'],
                'expected_vram': '15-30GB',
                'restart_delay': 15,
                'generation_time': '67-294s',
                'status': 'Qwen 7B Enhancement + FLF2V/T2V Tasks ✅',
                'port': 7860,  # ✅ ADDED: Port for Flask API (same as SDXL)
                'priority': 3   # ✅ ADDED: Startup priority (3 = lowest)
            }
        }
        
        logger.info("🎭 Triple Worker Orchestrator initialized")
        logger.info("🎨 SDXL: Fast image generation (3-8s) - Port 7860")
        logger.info("💬 Chat: Qwen Instruct service (5-15s) - Port 7861")
        logger.info("🎬 Enhanced WAN: Video + Qwen 7B enhancement + FLF2V/T2V tasks (67-294s) - Port 7860")
        logger.info("🔧 FIXED: Graceful validation + consistent parameter naming + FLF2V/T2V support + Chat integration")

    def get_worker_startup_command(self, worker_id):
        """Generate startup command and environment for a worker"""
        config = self.workers[worker_id]
        
        # Base command
        cmd = [sys.executable, config['script']]
        
        # Environment setup
        env = os.environ.copy()
        persistent_deps = "/workspace/python_deps/lib/python3.11/site-packages"
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{persistent_deps}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = persistent_deps
        
        # Worker-specific environment variables
        env['WORKER_TYPE'] = worker_id
        env['WORKER_PORT'] = str(config['port'])
        
        return cmd, env

    def validate_environment(self):
        """Validate environment for triple worker operation"""
        logger.info("🔍 Validating triple worker environment...")
        
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
                    logger.warning(f"⚠️ GPU has {total_vram:.1f}GB, triple workers need 45GB+ for concurrent operation")
                else:
                    logger.info(f"✅ GPU capacity sufficient for triple workers")
                    
            else:
                logger.error("❌ CUDA not available")
                return False
                
        except Exception as e:
            logger.error(f"❌ GPU check failed: {e}")
            return False
            
        # Check imports for all workers
        try:
            # SDXL imports
            from diffusers import StableDiffusionXLPipeline
            logger.info("✅ SDXL imports confirmed working")
        except ImportError as e:
            logger.warning(f"⚠️ SDXL imports failed in orchestrator: {e}")
            logger.info("📝 Will let SDXL worker handle its own imports")
            
        try:
            # Flask imports (for WAN and Chat workers)
            from flask import Flask
            logger.info("✅ Flask imports confirmed working")
        except ImportError as e:
            logger.warning(f"⚠️ Flask imports failed in orchestrator: {e}")
            logger.info("📝 Will let workers handle their own Flask imports")
            
        try:
            # Transformers imports (for WAN and Chat workers)
            from transformers import AutoTokenizer, AutoModelForCausalLM
            logger.info("✅ Transformers imports confirmed working")
        except ImportError as e:
            logger.warning(f"⚠️ Transformers imports failed in orchestrator: {e}")
            logger.info("📝 Will let workers handle their own transformers imports")
            
        # Check environment variables
        required_vars = [
            'SUPABASE_URL', 
            'SUPABASE_SERVICE_KEY', 
            'UPSTASH_REDIS_REST_URL', 
            'UPSTASH_REDIS_REST_TOKEN',
            'WAN_WORKER_API_KEY'  # ✅ ADDED: Required for /enhance endpoint
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"❌ Missing environment variables: {missing_vars}")
            return False
        else:
            logger.info("✅ All environment variables configured")
            
        # Check model paths for all workers
        model_paths = {
            'SDXL': '/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors',
            'WAN': '/workspace/models/wan2.1-t2v-1.3b',
            'Qwen Base': '/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796',
            'Qwen Instruct': '/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct'
        }
        
        missing_models = []
        for model_name, model_path in model_paths.items():
            if not os.path.exists(model_path):
                missing_models.append(f"{model_name}: {model_path}")
                logger.warning(f"⚠️ Model not found: {model_name} at {model_path}")
            else:
                logger.info(f"✅ Model found: {model_name}")
        
        if missing_models:
            logger.warning(f"⚠️ Missing models: {missing_models}")
            logger.info("📝 Some workers may not function properly without their models")
            
        # Validate parameter consistency in worker files
        logger.info("🔧 Validating parameter consistency across workers...")
        wan_script_path = Path('wan_worker.py')
        sdxl_script_path = Path('sdxl_worker.py')
        chat_script_path = Path('chat_worker.py')
        
        consistency_issues = []
        
        if wan_script_path.exists():
            with open(wan_script_path, 'r') as f:
                wan_content = f.read()
                # Check for consistent parameter naming
                if "'job_id':" in wan_content and "'assets':" in wan_content:
                    logger.info("✅ WAN worker uses consistent parameter naming (job_id, assets)")
                else:
                    consistency_issues.append("WAN worker parameter naming inconsistent")
                
                # Check for FLF2V/T2V task support
                if "flf2v-14B" in wan_content and "t2v-14B" in wan_content:
                    logger.info("✅ WAN worker supports FLF2V/T2V tasks")
                else:
                    consistency_issues.append("WAN worker missing FLF2V/T2V task support")
                
                # Check for correct parameter names
                if "--first_frame" in wan_content and "--last_frame" in wan_content:
                    logger.info("✅ WAN worker uses correct FLF2V parameter names (--first_frame, --last_frame)")
                else:
                    consistency_issues.append("WAN worker missing correct FLF2V parameter names")
        
        if sdxl_script_path.exists():
            with open(sdxl_script_path, 'r') as f:
                sdxl_content = f.read()
                # Check for consistent parameter naming
                if "'job_id':" in sdxl_content and "'assets':" in sdxl_content:
                    logger.info("✅ SDXL worker uses consistent parameter naming (job_id, assets)")
                else:
                    consistency_issues.append("SDXL worker parameter naming inconsistent")
        
        if chat_script_path.exists():
            with open(chat_script_path, 'r') as f:
                chat_content = f.read()
                # Check for Flask setup
                if "Flask" in chat_content and "port" in chat_content:
                    logger.info("✅ Chat worker has Flask setup")
                else:
                    consistency_issues.append("Chat worker missing Flask setup")
                
                # Check for Qwen Instruct model loading
                if "Qwen2.5-7B-Instruct" in chat_content:
                    logger.info("✅ Chat worker configured for Qwen Instruct model")
                else:
                    consistency_issues.append("Chat worker missing Qwen Instruct configuration")
        
        if consistency_issues:
            logger.error(f"❌ Parameter consistency issues: {consistency_issues}")
            return False
        else:
            logger.info("✅ Parameter naming consistency validated")
            
        logger.info("✅ Environment validation passed")
        return True

    def start_worker(self, worker_id):
        """Start a specific worker process"""
        config = self.workers[worker_id]
        
        logger.info(f"🚀 Starting {config['name']}...")
        logger.info(f"📋 Job Types: {', '.join(config['job_types'])}")
        logger.info(f"⚡ Performance: {config['generation_time']}")
        logger.info(f"🌐 Port: {config['port']}")
        
        try:
            # Get startup command and environment
            cmd, env = self.get_worker_startup_command(worker_id)
            
            logger.info(f"🔧 Setting PYTHONPATH: {env['PYTHONPATH']}")
            logger.info(f"🔧 Worker environment: WORKER_TYPE={worker_id}, WORKER_PORT={config['port']}")
            
            # Start worker process with proper environment
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env
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
                
                # Look for parameter consistency confirmations
                if "job_id" in line and "assets" in line and worker_id in ['wan', 'sdxl']:
                    logger.info(f"✅ {worker_id.upper()} parameter consistency confirmed in operation")
                
                # Look for FLF2V/T2V task confirmations
                if "FLF2V" in line or "T2V" in line and worker_id == 'wan':
                    logger.info(f"✅ {worker_id.upper()} FLF2V/T2V task support confirmed in operation")
                
                # Look for Chat worker confirmations
                if "Chat Worker" in line and worker_id == 'chat':
                    logger.info(f"✅ {worker_id.upper()} Chat service confirmed in operation")
                
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
        """Start all workers with monitoring in priority order"""
        logger.info("🎬 Starting all workers in priority order...")
        
        # Sort workers by priority (1 = highest, 3 = lowest)
        sorted_workers = sorted(self.workers.items(), key=lambda x: x[1]['priority'])
        
        for worker_id, config in sorted_workers:
            logger.info(f"🚀 Starting {config['name']} (Priority: {config['priority']})...")
            
            if self.start_worker(worker_id):
                # Start monitoring thread for each worker
                monitor_thread = threading.Thread(
                    target=self.monitor_worker, 
                    args=(worker_id,), 
                    daemon=True
                )
                monitor_thread.start()
                
                # Stagger startup to avoid resource conflicts
                time.sleep(3)
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
                        port = worker_info['config']['port']
                        active_workers.append(f"{worker_id}({uptime:.0f}s/{job_count}j/p{port})")
                
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
                        available = total - allocated
                        utilization = (allocated / total) * 100
                        logger.info(f"🔥 GPU Memory: {allocated:.1f}GB / {total:.0f}GB ({utilization:.1f}% used, {available:.1f}GB available)")
                        
                        # Warn if memory is getting low
                        if available < 5:
                            logger.warning(f"⚠️ Low VRAM available: {available:.1f}GB")
                except:
                    pass
                    
                # Wait before next check
                time.sleep(60)  # Status check every minute
                
            except Exception as e:
                logger.error(f"❌ Status monitor error: {e}")
                time.sleep(30)

    def run(self):
        """Main orchestrator run loop"""
        logger.info("🎭 TRIPLE WORKER ORCHESTRATOR STARTING")
        logger.info("🔧 GRACEFUL VALIDATION + CONSISTENT PARAMETERS + QWEN 7B + FLF2V/T2V + CHAT INTEGRATION - Production Ready")
        logger.info("=" * 80)
        
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
            
        logger.info("🎉 TRIPLE WORKER SYSTEM READY!")
        logger.info("=" * 80)
        logger.info("🎨 SDXL Worker: sdxl_queue → sdxl_image_fast, sdxl_image_high")
        logger.info("  ⚡ Performance: 3-8s generation")
        logger.info("  📋 Parameters: job_id, assets (consistent)")
        logger.info("  🌐 Port: 7860")
        logger.info("")
        logger.info("💬 Chat Worker: chat_queue → chat_enhance, chat_conversation, admin_utilities")
        logger.info("  🤖 Model: Qwen 2.5-7B Instruct")
        logger.info("  ⚡ Performance: 5-15s generation")
        logger.info("  🌐 Port: 7861")
        logger.info("")
        logger.info("🎬 Enhanced WAN Worker: wan_queue → 8 job types")
        logger.info("  📝 Standard: image_fast, image_high, video_fast, video_high")
        logger.info("  ✨ Qwen 7B Enhanced: image7b_fast_enhanced, image7b_high_enhanced, video7b_fast_enhanced, video7b_high_enhanced")
        logger.info("  ⚡ Performance: 67-294s generation (includes Qwen 7B prompt enhancement)")
        logger.info("  🎬 FLF2V/T2V Tasks: Automatic task selection for video with reference frames")
        logger.info("  📋 Parameters: job_id, assets (consistent)")
        logger.info("  🌐 Port: 7860")
        logger.info("")
        logger.info("💡 All workers monitoring their respective queues")
        logger.info("🔧 Fixed: Graceful SDXL validation, Enhanced WAN with Qwen 7B prompt enhancement + FLF2V/T2V tasks, Chat integration")
        logger.info("=" * 80)
        
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
                    
                    # Auto-restart high-priority workers
                    for dead_worker in dead_workers:
                        config = self.workers[dead_worker]
                        if config['priority'] <= 2:  # High priority workers (SDXL, Chat)
                            logger.info(f"🔄 Auto-restarting high-priority worker: {dead_worker}")
                            self.handle_worker_restart(dead_worker)
                    
        except KeyboardInterrupt:
            logger.info("👋 Orchestrator interrupted by user")
        finally:
            self.stop_all_workers()
            
        logger.info("✅ Triple Worker Orchestrator shutdown complete")
        return True

if __name__ == "__main__":
    logger.info("🚀 Starting OurVidz Triple Worker System - CONSISTENT PARAMETERS + QWEN 7B + FLF2V/T2V + CHAT VERSION")
    
    try:
        # ✅ AUTO-REGISTER WORKER URL
        logger.info("🌐 Worker URL auto-registration will be handled by WAN worker")
        logger.info("🔧 Starting worker processes...")
        orchestrator = DualWorkerOrchestrator()
        success = orchestrator.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Orchestrator startup failed: {e}")
        sys.exit(1)