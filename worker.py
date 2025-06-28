# worker.py - Complete Enhanced GPU Performance Worker with Fixes
import os
import json
import time
import requests
import subprocess
import uuid
import shutil
import tempfile
import threading
import signal
from pathlib import Path
from PIL import Image
import cv2
import torch

class VideoWorker:
    def __init__(self):
        print("🚀 OurVidz Worker initialized (COMPLETE GPU DIAGNOSTICS FIX)")
        print("⚡ Fixed GPU monitoring with realistic performance expectations")
        
        # Create dedicated temp directories for better organization
        self.temp_base = Path("/tmp/ourvidz")
        self.temp_base.mkdir(exist_ok=True)
        
        self.temp_models = self.temp_base / "models"
        self.temp_outputs = self.temp_base / "outputs" 
        self.temp_processing = self.temp_base / "processing"
        
        for temp_dir in [self.temp_models, self.temp_outputs, self.temp_processing]:
            temp_dir.mkdir(exist_ok=True)
            print(f"📁 Created temp dir: {temp_dir}")

        self.ffmpeg_available = shutil.which('ffmpeg') is not None
        print(f"🔧 FFmpeg Available: {self.ffmpeg_available}")
        
        # Enhanced GPU monitoring state
        self.gpu_monitoring_active = False
        self.gpu_monitor_thread = None
        self.generation_active = False
        self.last_gpu_stats = {}
        
        # Run simple diagnostic first to understand our environment
        self.simple_gpu_diagnostic()
        
        # CRITICAL: Force GPU activation before anything else
        self.detect_gpu()
        self.force_gpu_activation()
        self.init_hardware_optimizations()
        self.start_enhanced_gpu_management()

        # Use temp storage for models - much faster I/O
        self.model_path = str(self.temp_models / 'wan2.1-t2v-1.3b')
        self.model_loaded = False

        # Job type mapping with REALISTIC timing expectations based on actual performance
        self.job_type_mapping = {
            'image_fast': {
                'content_type': 'image',
                'resolution': 'small',            
                'quality': 'fast',                
                'storage_bucket': 'image_fast',
                'expected_time': 90,              # REALISTIC: Based on actual 88s performance
                'description': 'Small resolution, fastest available speed'
            },
            'image_high': {
                'content_type': 'image',
                'resolution': 'standard',         
                'quality': 'balanced',            
                'storage_bucket': 'image_high',
                'expected_time': 100,             # REALISTIC: Slightly slower for higher quality
                'description': 'Standard resolution, balanced quality'
            },
            'video_fast': {
                'content_type': 'video',
                'resolution': 'small',            
                'quality': 'fast',                
                'storage_bucket': 'video_fast',
                'expected_time': 95,              # REALISTIC: Video slightly slower than image
                'description': 'Small resolution, fast video'
            },
            'video_high': {
                'content_type': 'video', 
                'resolution': 'standard',         
                'quality': 'balanced',            
                'storage_bucket': 'video_high',
                'expected_time': 110,             # REALISTIC: Higher quality video
                'description': 'Standard resolution, quality video'
            }
        }
        
        # Resolution configurations using ONLY supported Wan 2.1 sizes
        self.resolution_configs = {
            'small': {
                'size': '480*832',              
                'multiplier': 0.7,              
                'description': 'Small (480×832) - Fastest supported'
            },
            'standard': {
                'size': '832*480',              
                'multiplier': 1.0,              
                'description': 'Standard (832×480) - Current working'
            }
        }
        
        # Quality configurations optimized for speed
        self.quality_configs = {
            'fast': {
                'sample_steps': 8,              
                'sample_guide_scale': 5.5,      
                'description': 'Fast - Speed optimized'
            },
            'balanced': {
                'sample_steps': 10,             
                'sample_guide_scale': 6.0,      
                'description': 'Balanced - Speed/quality balance'
            }
        }

        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')

        print("🎬 Complete GPU Diagnostics Worker ready")
        print("🔥 Fixed GPU monitoring with realistic 90s expectations")

    def simple_gpu_diagnostic(self):
        """Simple diagnostic to understand GPU environment"""
        print("🔍 GPU Environment Diagnostic:")
        
        try:
            # Test basic nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,clocks.current.graphics', '--format=csv,noheader'], 
                                   capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                print(f"✅ GPU: {gpu_info[0]}")
                print(f"✅ Current Graphics Clock: {gpu_info[1]}")
            else:
                print(f"❌ nvidia-smi basic query failed: {result.stderr}")
                
            # Test the problematic query that was failing
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print("✅ Full GPU stats query working")
                stats = result.stdout.strip().split(', ')
                print(f"📊 Stats: {stats[0]}MHz graphics, {stats[1]}MHz memory, {stats[2]}W power")
            else:
                print("⚠️ Full GPU stats query failed - will use fallback")
                
        except Exception as e:
            print(f"❌ GPU diagnostic error: {e}")
        
        # Test PyTorch CUDA
        try:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"✅ PyTorch CUDA: {device_name}")
                print(f"✅ Total VRAM: {memory_total:.1f}GB")
            else:
                print("❌ PyTorch CUDA not available")
        except Exception as e:
            print(f"❌ PyTorch diagnostic error: {e}")
            
        print("🔍 Diagnostic complete\n")

    def detect_gpu(self):
        """Enhanced GPU detection with comprehensive performance analysis"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,memory.total,memory.free,memory.used,clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                print(f"🔥 GPU: {gpu_info[0]} ({gpu_info[1]}GB total)")
                print(f"💾 VRAM: {gpu_info[3]}MB used, {gpu_info[2]}MB free")
                print(f"⚡ Graphics Clock: {gpu_info[4]}MHz")
                print(f"💽 Memory Clock: {gpu_info[5]}MHz") 
                print(f"🔌 Power Draw: {gpu_info[6]}W")
                print(f"🌡️ Temperature: {gpu_info[7]}°C")
                print(f"📊 Utilization: {gpu_info[8]}%")
                
                # Store baseline stats
                self.last_gpu_stats = {
                    'graphics_clock': int(gpu_info[4]) if gpu_info[4].isdigit() else 0,
                    'memory_clock': int(gpu_info[5]) if gpu_info[5].isdigit() else 0,
                    'power_draw': float(gpu_info[6]) if gpu_info[6].replace('.', '').isdigit() else 0,
                    'temperature': float(gpu_info[7]) if gpu_info[7].replace('.', '').isdigit() else 0,
                    'utilization': int(gpu_info[8]) if gpu_info[8].isdigit() else 0,
                }
                
                # Alert if clocks are low
                graphics_clock = self.last_gpu_stats['graphics_clock']
                if graphics_clock < 1000:
                    print(f"⚠️ WARNING: Graphics clock {graphics_clock}MHz is very low!")
                    print("🔧 Attempting GPU activation...")
                    
            else:
                print("⚠️ Full GPU detection failed, trying basic detection...")
                # Fallback to basic detection
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,clocks.current.graphics', '--format=csv,noheader'], 
                                       capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split(', ')
                    print(f"🔥 GPU: {gpu_info[0]}")
                    print(f"⚡ Graphics Clock: {gpu_info[1]}")
                else:
                    print("❌ Even basic GPU detection failed")
                    
        except Exception as e:
            print(f"⚠️ GPU detection failed: {e}")

    def get_current_gpu_stats(self):
        """FIXED: GPU statistics with proper fallback handling"""
        try:
            # Try comprehensive query first (without performance.state that was failing)
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                stats = result.stdout.strip().split(', ')
                if len(stats) >= 5:
                    current_stats = {
                        'graphics_clock': int(stats[0]) if stats[0].isdigit() else 0,
                        'memory_clock': int(stats[1]) if stats[1].isdigit() else 0,
                        'power_draw': float(stats[2]) if stats[2].replace('.', '').isdigit() else 0,
                        'temperature': float(stats[3]) if stats[3].replace('.', '').isdigit() else 0,
                        'utilization': int(stats[4]) if stats[4].isdigit() else 0,
                    }
                    return current_stats
            
            print("⚠️ Full GPU query failed, trying basic clocks...")
            # Fallback: Try basic clocks only
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=clocks.current.graphics,clocks.current.memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                stats = result.stdout.strip().split(', ')
                if len(stats) >= 2:
                    basic_stats = {
                        'graphics_clock': int(stats[0]) if stats[0].isdigit() else 0,
                        'memory_clock': int(stats[1]) if stats[1].isdigit() else 0,
                        'power_draw': 0,
                        'temperature': 0,
                        'utilization': 0,
                    }
                    return basic_stats
                    
        except Exception as e:
            print(f"⚠️ GPU stats error: {e}")
        
        return None

    def check_gpu_performance_enhanced(self):
        """FIXED: GPU performance check with proper error handling"""
        stats = self.get_current_gpu_stats()
        if not stats:
            print("❌ Cannot get GPU stats - assuming medium performance")
            return "medium"
        
        graphics_mhz = stats['graphics_clock']
        utilization = stats['utilization']
        power_draw = stats['power_draw']
        
        print(f"📊 GPU Status: {graphics_mhz}MHz graphics, {stats['memory_clock']}MHz memory")
        if power_draw > 0:
            print(f"🔌 Power: {power_draw}W, Temp: {stats['temperature']}°C, Util: {utilization}%")
        
        # Performance classification (more lenient since we know 88s is normal)
        if graphics_mhz > 2000:
            return "excellent"
        elif graphics_mhz > 1500:
            return "high"
        elif graphics_mhz > 800:
            return "medium"
        elif graphics_mhz > 0:  # Any valid reading is acceptable
            return "working"
        else:
            return "unknown"

    def force_gpu_activation(self):
        """Enhanced GPU activation with multiple strategies"""
        print("🔥 Enhanced GPU activation - multiple strategies...")
        
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                print(f"✅ CUDA available with {device_count} device(s)")
                
                # Strategy 1: Force large memory allocation
                print("🔥 Strategy 1: Large memory allocation...")
                activation_tensor = torch.zeros((3000, 3000), dtype=torch.float16, device='cuda')
                
                # Strategy 2: Intensive computation with different operations
                print("🔥 Strategy 2: Mixed intensive computations...")
                for i in range(20):
                    # Matrix operations
                    result1 = torch.matmul(activation_tensor, activation_tensor)
                    # Trigonometric operations
                    result2 = torch.sin(result1) + torch.cos(result1)
                    # Exponential operations
                    result3 = torch.exp(result2 * 0.001)  # Small multiplier to prevent overflow
                    # Random operations
                    result4 = torch.randn_like(result3) * result3
                    torch.cuda.synchronize()
                    
                    if i % 5 == 0:
                        print(f"🔥 Activation iteration {i+1}/20...")
                        # Check clocks during activation
                        stats = self.get_current_gpu_stats()
                        if stats:
                            print(f"📊 Current clocks: {stats['graphics_clock']}MHz")
                
                # Strategy 3: Persistent warmup tensor
                print("🔥 Strategy 3: Persistent warmup tensor...")
                self.warmup_tensor = torch.ones((1000, 1000), dtype=torch.float16, device='cuda')
                
                # Check final memory usage
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"✅ GPU activation complete - {memory_allocated:.1f}GB allocated")
                
                # Clean up large activation tensors but keep warmup tensor
                del activation_tensor, result1, result2, result3, result4
                torch.cuda.empty_cache()
                
                # Final GPU check
                time.sleep(2)
                final_stats = self.get_current_gpu_stats()
                if final_stats and final_stats.get('graphics_clock', 0) > 1000:
                    print(f"✅ GPU activation successful: {final_stats['graphics_clock']}MHz")
                else:
                    print("⚠️ GPU activation completed (clocks may be normal for idle state)")
                
            else:
                print("❌ CUDA not available!")
                
        except Exception as e:
            print(f"⚠️ Enhanced GPU activation failed: {e}")

    def start_enhanced_gpu_management(self):
        """Start enhanced GPU management with real-time monitoring"""
        print("🔥 Starting enhanced GPU management system...")
        
        def gpu_management_thread():
            try:
                iteration = 0
                
                while True:
                    try:
                        # Continuous warmup operations
                        if hasattr(self, 'warmup_tensor'):
                            result = torch.matmul(self.warmup_tensor, self.warmup_tensor)
                            torch.cuda.synchronize()
                        
                        iteration += 1
                        
                        # Detailed monitoring every 30 seconds during generation
                        if self.generation_active and iteration % 6 == 0:
                            stats = self.get_current_gpu_stats()
                            if stats:
                                print(f"🔥 [GENERATION] GPU: {stats['graphics_clock']}MHz, "
                                     f"Util: {stats['utilization']}%, Power: {stats['power_draw']}W")
                                
                                # Alert if performance drops during generation
                                if stats['graphics_clock'] > 0 and stats['graphics_clock'] < 800:
                                    print("⚠️ [GENERATION] GPU clocks dropped! Attempting reactivation...")
                                    self.force_gpu_activation()
                        
                        # Regular monitoring every 60 seconds when idle
                        elif not self.generation_active and iteration % 12 == 0:
                            stats = self.get_current_gpu_stats()
                            if stats:
                                print(f"🔥 [IDLE] GPU: {stats['graphics_clock']}MHz, "
                                     f"Temp: {stats['temperature']}°C")
                        
                        time.sleep(5)  # Check every 5 seconds
                        
                    except Exception as e:
                        print(f"⚠️ GPU management iteration error: {e}")
                        time.sleep(10)
                        
            except Exception as e:
                print(f"❌ GPU management thread failed: {e}")
        
        # Start management thread
        management_thread = threading.Thread(target=gpu_management_thread, daemon=True)
        management_thread.start()
        print("✅ Enhanced GPU management thread started")

    def start_real_time_gpu_monitor(self):
        """Start real-time GPU monitoring during generation"""
        if self.gpu_monitoring_active:
            return
            
        print("📊 Starting real-time GPU monitoring for generation...")
        self.gpu_monitoring_active = True
        
        def monitor_gpu():
            try:
                while self.gpu_monitoring_active:
                    if self.generation_active:
                        stats = self.get_current_gpu_stats()
                        if stats:
                            print(f"📊 [REALTIME] GPU: {stats['graphics_clock']}MHz | "
                                 f"Util: {stats['utilization']}% | "
                                 f"Power: {stats['power_draw']}W | "
                                 f"Temp: {stats['temperature']}°C")
                    time.sleep(3)  # Monitor every 3 seconds during generation
            except Exception as e:
                print(f"⚠️ Real-time monitor error: {e}")
            finally:
                self.gpu_monitoring_active = False
        
        self.gpu_monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
        self.gpu_monitor_thread.start()

    def stop_real_time_gpu_monitor(self):
        """Stop real-time GPU monitoring"""
        if self.gpu_monitoring_active:
            print("📊 Stopping real-time GPU monitoring...")
            self.gpu_monitoring_active = False

    def init_hardware_optimizations(self):
        """Enhanced hardware optimizations with additional performance settings"""
        print("🔧 Initializing enhanced hardware optimizations...")
        
        try:
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.version.cuda}")
                
                # Maximum performance environment variables
                os.environ.update({
                    'CUDA_LAUNCH_BLOCKING': '0',
                    'CUDA_CACHE_DISABLE': '0',
                    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,roundup_power2_divisions:16',
                    'TORCH_USE_CUDA_DSA': '1',
                    'CUDA_VISIBLE_DEVICES': '0',
                    'NVIDIA_DRIVER_CAPABILITIES': 'all',
                    'CUDA_DEVICE_MAX_CONNECTIONS': '1',
                })
                
                # PyTorch maximum performance settings
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True  
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.enabled = True
                
                # Additional performance optimizations
                if hasattr(torch, 'set_float32_matmul_precision'):
                    torch.set_float32_matmul_precision('medium')
                
                if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                    torch.backends.cuda.enable_flash_sdp(True)
                
                print("✅ Enhanced hardware optimizations applied")
                
            else:
                print("⚠️ CUDA not available")
                
        except Exception as e:
            print(f"⚠️ Enhanced hardware optimization failed: {e}")

    def ensure_model_ready(self):
        """Ensure model is ready with enhanced GPU performance check"""
        if os.path.exists(self.model_path):
            print("✅ Model already in temp storage")
            
            # Enhanced GPU performance check before proceeding
            perf_level = self.check_gpu_performance_enhanced()
            if perf_level in ["unknown"]:
                print(f"⚠️ GPU performance {perf_level}, forcing reactivation...")
                self.force_gpu_activation()
            
            return True
            
        # Standard model copying logic
        network_model_path = "/workspace/models/wan2.1-t2v-1.3b"
        if os.path.exists(network_model_path):
            print("📦 Copying model from network volume to temp storage...")
            try:
                start_time = time.time()
                shutil.copytree(network_model_path, self.model_path)
                copy_time = time.time() - start_time
                print(f"✅ Model copied to temp storage in {copy_time:.1f}s")
                return True
            except Exception as e:
                print(f"❌ Model copy failed: {e}")
                self.model_path = network_model_path
                return True
        
        print("❌ Model not found")
        return False

    def get_job_config(self, job_type):
        """UPDATED: Get configuration with realistic timing expectations"""
        job_mapping = self.job_type_mapping.get(job_type)
        if not job_mapping:
            return {
                'size': '832*480',
                'frame_num': 1,
                'sample_steps': 10,
                'sample_guide_scale': 6.0,
                'expected_time': 90,  # REALISTIC default
                'storage_bucket': 'image_fast',
                'content_type': 'image'
            }
        
        resolution_config = self.resolution_configs[job_mapping['resolution']]
        quality_config = self.quality_configs[job_mapping['quality']]
        
        frame_num = 1 if job_mapping['content_type'] == 'image' else 17
            
        return {
            'size': resolution_config['size'],
            'frame_num': frame_num,
            'sample_steps': quality_config['sample_steps'],
            'sample_guide_scale': quality_config['sample_guide_scale'],
            'expected_time': job_mapping['expected_time'],
            'storage_bucket': job_mapping['storage_bucket'],
            'content_type': job_mapping['content_type'],
            'resolution_desc': resolution_config['description'],
            'quality_desc': quality_config['description']
        }

    def get_expected_time(self, job_type):
        """Get expected generation time"""
        job_mapping = self.job_type_mapping.get(job_type, {})
        return f"{job_mapping.get('expected_time', 90)}s"

    def generate_with_gpu_monitoring(self, prompt, job_type):
        """Enhanced generation with comprehensive GPU monitoring"""
        config = self.get_job_config(job_type)

        # Ensure model and GPU are ready
        if not self.ensure_model_ready():
            return None

        # Start real-time monitoring
        self.start_real_time_gpu_monitor()
        self.generation_active = True

        # Pre-generation GPU check and activation
        perf_level = self.check_gpu_performance_enhanced()
        print(f"🔥 Pre-generation GPU Performance: {perf_level}")
        
        if perf_level in ["unknown"]:
            print("🔥 Performing pre-generation GPU activation...")
            self.force_gpu_activation()
            time.sleep(2)
            perf_level = self.check_gpu_performance_enhanced()
            print(f"🔥 Post-activation GPU Performance: {perf_level}")

        job_id = str(uuid.uuid4())[:8]
        expected_time = config['expected_time']

        print(f"⚡ {job_type.upper()} generation (GPU: {perf_level})")
        print(f"📝 Prompt: {prompt}")
        print(f"📐 Resolution: {config.get('resolution_desc', 'unknown')}")
        print(f"⚙️ Quality: {config.get('quality_desc', 'unknown')}")
        print(f"🔧 Config: {config['sample_steps']} steps, {config['sample_guide_scale']} guidance, {config['size']}")
        print(f"🎯 Expected: {expected_time}s (realistic expectation based on performance data)")

        # Use temp processing directory
        output_filename = f"{job_type}_{job_id}.mp4"
        temp_output_path = self.temp_processing / output_filename
        
        cmd = [
            "python", "generate.py",
            "--task", "t2v-1.3B",
            "--size", config['size'],
            "--ckpt_dir", self.model_path,
            "--prompt", prompt,
            "--save_file", str(temp_output_path),
            "--sample_steps", str(config['sample_steps']),
            "--sample_guide_scale", str(config['sample_guide_scale']),
            "--frame_num", str(config['frame_num'])
        ]

        original_cwd = os.getcwd()
        os.chdir("/workspace/Wan2.1")
        
        try:
            start_time = time.time()
            
            # Enhanced environment for maximum performance
            env = os.environ.copy()
            env.update({
                'CUDA_LAUNCH_BLOCKING': '0',
                'TORCH_USE_CUDA_DSA': '1',
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
                'CUDA_CACHE_DISABLE': '0',
                'OMP_NUM_THREADS': '8',
                'MKL_NUM_THREADS': '8',
            })
            
            # Final GPU activation before generation
            print("🔥 Final GPU activation before generation...")
            if hasattr(self, 'warmup_tensor'):
                final_warmup = torch.matmul(self.warmup_tensor, self.warmup_tensor)
                torch.cuda.synchronize()
                del final_warmup
            
            # Log GPU state right before generation
            pre_gen_stats = self.get_current_gpu_stats()
            if pre_gen_stats:
                print(f"🔥 Starting generation with GPU at {pre_gen_stats['graphics_clock']}MHz")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
            generation_time = time.time() - start_time
            
            # Log GPU state right after generation
            post_gen_stats = self.get_current_gpu_stats()
            if post_gen_stats:
                print(f"🔥 Generation completed with GPU at {post_gen_stats['graphics_clock']}MHz")
            
            if result.returncode != 0:
                print(f"❌ Generation failed: {result.stderr}")
                return None
                
            print(f"⚡ Generation completed in {generation_time:.1f}s (expected {expected_time}s)")
            
            # More realistic performance analysis
            performance_ratio = generation_time / expected_time
            if performance_ratio > 1.3:  # More than 30% slower than realistic expectation
                print(f"⚠️ Generation took {performance_ratio:.1f}x longer than expected")
                if pre_gen_stats and post_gen_stats:
                    clock_drop = pre_gen_stats['graphics_clock'] - post_gen_stats['graphics_clock']
                    if clock_drop > 500:
                        print(f"⚠️ GPU clocks dropped {clock_drop}MHz during generation")
            elif performance_ratio < 0.9:  # Faster than expected
                print(f"🚀 Generation completed {performance_ratio:.1f}x faster than expected!")
                
            if not temp_output_path.exists():
                fallback_path = Path(output_filename)
                if fallback_path.exists():
                    shutil.move(str(fallback_path), str(temp_output_path))
                else:
                    print("❌ Output file not found")
                    return None
            
            print(f"✅ Generation completed: {temp_output_path}")
            
            if config['content_type'] == 'image':
                return self.extract_frame_from_video(str(temp_output_path), job_id, job_type)
            
            return str(temp_output_path)
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return None
        finally:
            self.generation_active = False
            self.stop_real_time_gpu_monitor()
            os.chdir(original_cwd)

    def extract_frame_from_video(self, video_path, job_id, job_type):
        """Enhanced frame extraction with job-type-aware optimization"""
        image_path = self.temp_processing / f"{job_type}_{job_id}.png"
        
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Optimize compression based on job type
                if 'fast' in job_type:
                    img.save(str(image_path), "PNG", optimize=True, compress_level=9)
                else:
                    img.save(str(image_path), "PNG", optimize=True, compress_level=6)
                
                # Get file size for logging
                file_size = os.path.getsize(image_path) / 1024  # KB
                config = self.get_job_config(job_type)
                size_desc = config.get('size', 'unknown')
                print(f"📊 Output: {size_desc} resolution, {file_size:.0f}KB")
                
                # Clean up video file immediately
                try:
                    os.remove(video_path)
                except:
                    pass
                    
                return str(image_path)
        except Exception as e:
            print(f"❌ Frame extraction error: {e}")
        return None

    def optimize_file_for_upload(self, file_path, job_type):
        """Job-type-aware file optimization"""
        config = self.get_job_config(job_type)
        content_type = config['content_type']
        
        if content_type == 'image':
            return file_path
            
        if content_type == 'video' and self.ffmpeg_available:
            optimized_path = str(Path(file_path).with_suffix('.optimized.mp4'))
            
            size = config['size']
            width, height = size.split('*')
            
            if 'fast' in job_type:
                preset = 'veryfast'
                crf = '26'
            else:
                preset = 'fast'
                crf = '23'
            
            cmd = [
                'ffmpeg', '-i', file_path,
                '-c:v', 'libx264',
                '-preset', preset,
                '-crf', crf,
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-vf', f'scale={width}:{height}',
                '-y', optimized_path
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0 and os.path.exists(optimized_path):
                    orig_size = os.path.getsize(file_path)
                    opt_size = os.path.getsize(optimized_path)
                    
                    if opt_size < orig_size:
                        print(f"📉 Optimized: {orig_size//1024}KB → {opt_size//1024}KB")
                        os.remove(file_path)
                        return optimized_path
                    else:
                        os.remove(optimized_path)
                        
            except Exception as e:
                print(f"⚠️ Optimization failed: {e}")
                
        return file_path

    def upload_to_supabase(self, file_path, job_type, user_id, job_id):
        """Upload to the correct storage bucket based on job type"""
        if not os.path.exists(file_path):
            return None
            
        optimized_path = self.optimize_file_for_upload(file_path, job_type)
        config = self.get_job_config(job_type)
        
        storage_bucket = config['storage_bucket']
        content_type = config['content_type']
        
        filename = f"job_{job_id}_{int(time.time())}_{job_type}.{'png' if content_type == 'image' else 'mp4'}"
        full_path = f"{storage_bucket}/{user_id}/{filename}"
        mime_type = 'image/png' if content_type == 'image' else 'video/mp4'
        
        print(f"📤 Uploading to bucket: {storage_bucket}")
        
        try:
            with open(optimized_path, 'rb') as f:
                file_data = f.read()
                file_size = len(file_data) / 1024
                print(f"📊 File size: {file_size:.0f}KB")
                
                for attempt in range(3):
                    try:
                        print(f"🔄 Upload attempt {attempt + 1}/3...")
                        
                        r = requests.post(
                            f"{self.supabase_url}/storage/v1/object/{full_path}",
                            data=file_data,
                            headers={
                                'Authorization': f"Bearer {self.supabase_service_key}",
                                'Content-Type': mime_type,
                                'x-upsert': 'true'
                            },
                            timeout=120
                        )
                        
                        print(f"📡 Response: {r.status_code}")
                        
                        if r.status_code in [200, 201]:
                            print(f"✅ Upload successful: {full_path}")
                            return f"{user_id}/{filename}"
                        else:
                            print(f"⚠️ Upload attempt {attempt + 1} failed: {r.status_code} - {r.text}")
                            
                            if r.status_code in [401, 403, 404]:
                                break
                                
                    except requests.RequestException as e:
                        print(f"⚠️ Upload attempt {attempt + 1} error: {e}")
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                            
        except Exception as e:
            print(f"❌ Upload preparation failed: {e}")
        finally:
            self.cleanup_temp_files([file_path, optimized_path])
            
        print("❌ All upload attempts failed")
        return None

    def cleanup_temp_files(self, file_paths):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass

    def cleanup_old_temp_files(self):
        """Cleanup old temp files"""
        try:
            current_time = time.time()
            cleaned_count = 0
            
            for temp_dir in [self.temp_outputs, self.temp_processing]:
                for file_path in temp_dir.glob("*"):
                    if file_path.is_file():
                        if (current_time - file_path.stat().st_mtime) > 1200:  # 20 minutes
                            try:
                                file_path.unlink()
                                cleaned_count += 1
                            except:
                                pass
                                
            if cleaned_count > 0:
                print(f"🧹 Cleaned up {cleaned_count} old temp files")
                
        except Exception as e:
            print(f"⚠️ Temp cleanup error: {e}")

    def notify_completion(self, job_id, status, file_path=None, error_message=None):
        """Enhanced callback with performance metrics"""
        data = {
            'jobId': job_id, 
            'status': status, 
            'filePath': file_path, 
            'errorMessage': error_message
        }
        
        print(f"📞 Calling job-callback for job {job_id}: {status}")
        
        try:
            r = requests.post(
                f"{self.supabase_url}/functions/v1/job-callback", 
                json=data,
                headers={
                    'Authorization': f"Bearer {self.supabase_service_key}", 
                    'Content-Type': 'application/json'
                },
                timeout=30
            )
            
            if r.status_code == 200:
                print("✅ Callback sent successfully")
            else:
                print(f"❌ Callback failed: {r.status_code} - {r.text}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job(self, job_data):
        """Enhanced job processing with comprehensive GPU monitoring"""
        job_id = job_data.get('jobId')
        job_type = job_data.get('jobType')
        prompt = job_data.get('prompt')
        user_id = job_data.get('userId')
        
        print(f"📋 Received job data keys: {list(job_data.keys())}")
        print(f"📋 Job details: ID={job_id}, Type={job_type}, User={user_id}")
        
        if not all([job_id, job_type, user_id, prompt]):
            missing_fields = []
            if not job_id: missing_fields.append('jobId')
            if not job_type: missing_fields.append('jobType') 
            if not user_id: missing_fields.append('userId')
            if not prompt: missing_fields.append('prompt')
            
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            print(f"❌ {error_msg}")
            self.notify_completion(job_id or 'unknown', 'failed', error_message=error_msg)
            return

        print(f"📝 Prompt: {prompt}")
        print(f"📥 Processing job: {job_id} ({job_type})")
        
        expected_time = self.get_expected_time(job_type)
        print(f"⏱️ Expected completion: {expected_time}")
        
        # Pre-job GPU analysis
        pre_job_stats = self.get_current_gpu_stats()
        if pre_job_stats:
            print(f"🔥 Pre-job GPU state: {pre_job_stats['graphics_clock']}MHz, "
                 f"{pre_job_stats['utilization']}% util, {pre_job_stats['power_draw']}W")
        
        start_time = time.time()
        
        try:
            # Use enhanced generation method
            output_path = self.generate_with_gpu_monitoring(prompt, job_type)
            if output_path:
                supa_path = self.upload_to_supabase(output_path, job_type, user_id, job_id)
                if supa_path:
                    duration = time.time() - start_time
                    
                    # Post-job GPU analysis
                    post_job_stats = self.get_current_gpu_stats()
                    if post_job_stats:
                        print(f"🔥 Post-job GPU state: {post_job_stats['graphics_clock']}MHz, "
                             f"{post_job_stats['utilization']}% util, {post_job_stats['power_draw']}W")
                    
                    print(f"🎉 Job completed successfully in {duration:.1f}s")
                    
                    # Realistic performance analysis
                    config = self.get_job_config(job_type)
                    expected_duration = config['expected_time']
                    performance_ratio = duration / expected_duration
                    
                    if performance_ratio > 1.3:  # More than 30% slower than realistic expectation
                        print(f"⚠️ Performance: {performance_ratio:.1f}x slower than expected")
                        if pre_job_stats and post_job_stats:
                            clock_change = post_job_stats['graphics_clock'] - pre_job_stats['graphics_clock']
                            print(f"📊 GPU clock change during job: {clock_change:+d}MHz")
                    elif performance_ratio < 0.8:  # Faster than expected
                        print(f"🚀 Performance: {performance_ratio:.1f}x faster than expected!")
                    else:
                        print(f"✅ Performance: {performance_ratio:.1f}x expected time (good performance)")
                    
                    self.notify_completion(job_id, 'completed', supa_path)
                    return
                    
            self.notify_completion(job_id, 'failed', error_message="Generation or upload failed")
            
        except Exception as e:
            print(f"❌ Job processing error: {e}")
            self.notify_completion(job_id, 'failed', error_message=str(e))

    def poll_queue(self):
        """Reliable queue polling with enhanced error handling"""
        try:
            r = requests.get(
                f"{self.redis_url}/rpop/job_queue",
                headers={'Authorization': f"Bearer {self.redis_token}"}, 
                timeout=10
            )
            if r.status_code == 200 and r.json().get('result'):
                return json.loads(r.json()['result'])
        except Exception as e:
            print(f"❌ Poll error: {e}")
        return None

    def run_comprehensive_diagnostic(self):
        """Run comprehensive diagnostic for troubleshooting"""
        print("\n🔍 COMPREHENSIVE GPU DIAGNOSTIC")
        print("=" * 50)
        
        # Test 1: nvidia-smi availability and queries
        print("1. Testing nvidia-smi queries...")
        queries_to_test = [
            'name',
            'clocks.current.graphics',
            'clocks.current.memory',
            'power.draw',
            'temperature.gpu',
            'utilization.gpu',
        ]
        
        working_queries = []
        for query in queries_to_test:
            try:
                result = subprocess.run([
                    'nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    value = result.stdout.strip()
                    print(f"   ✅ {query}: {value}")
                    working_queries.append(query)
                else:
                    print(f"   ❌ {query}: FAILED")
            except Exception as e:
                print(f"   ❌ {query}: ERROR - {e}")
        
        # Test 2: Combined query (what we actually use)
        print("\n2. Testing combined GPU stats query...")
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                stats = result.stdout.strip().split(', ')
                print(f"   ✅ Combined query successful: {stats}")
            else:
                print(f"   ❌ Combined query failed: {result.stderr}")
        except Exception as e:
            print(f"   ❌ Combined query error: {e}")
        
        # Test 3: PyTorch performance test
        print("\n3. Testing PyTorch CUDA performance...")
        try:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                
                print(f"   ✅ Device: {device_name}")
                print(f"   ✅ VRAM: {memory_allocated:.1f}GB / {memory_total:.1f}GB")
                
                # Quick performance test
                print("   🔥 Running performance test...")
                start_time = time.time()
                test_tensor = torch.randn(1000, 1000, device='cuda')
                for i in range(10):
                    result = torch.matmul(test_tensor, test_tensor)
                    torch.cuda.synchronize()
                test_time = time.time() - start_time
                print(f"   📊 PyTorch test: {test_time:.2f}s for 10 matrix ops")
                
                del test_tensor, result
                torch.cuda.empty_cache()
            else:
                print("   ❌ PyTorch CUDA not available")
        except Exception as e:
            print(f"   ❌ PyTorch test error: {e}")
        
        # Test 4: Check if model exists and is accessible
        print("\n4. Testing model availability...")
        model_paths = [
            "/workspace/models/wan2.1-t2v-1.3b",
            str(self.temp_models / 'wan2.1-t2v-1.3b'),
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    model_files = list(Path(model_path).glob("**/*"))
                    total_size = sum(f.stat().st_size for f in model_files if f.is_file())
                    print(f"   ✅ Model found: {model_path}")
                    print(f"   📊 {len(model_files)} files, {total_size/1024**3:.1f}GB total")
                except Exception as e:
                    print(f"   ⚠️ Model path exists but error accessing: {e}")
            else:
                print(f"   ❌ Model not found: {model_path}")
        
        print("\n" + "=" * 50)
        print("🔍 Diagnostic complete")

    def run(self):
        """Enhanced main loop with comprehensive GPU performance monitoring"""
        print("⏳ Waiting for jobs...")
        print("🎯 UPDATED Realistic Performance Targets:")
        print("   • image_fast: 90s (based on actual measured performance)")
        print("   • image_high: 100s (realistic expectation)")  
        print("   • video_fast: 95s (video processing)")
        print("   • video_high: 110s (higher quality)")
        print("📊 Previous unrealistic target was 35s - now using data-driven expectations")
        print("🔥 Enhanced GPU management active:")
        print("   • Fixed GPU monitoring (no more 'unknown' status)")
        print("   • Real-time monitoring during generation")
        print("   • Persistent activation threads")
        print("   • Realistic performance analysis")
        
        # Run comprehensive diagnostic on startup
        self.run_comprehensive_diagnostic()
        
        last_cleanup = time.time()
        last_gpu_check = time.time()
        last_performance_report = time.time()
        job_count = 0
        
        while True:
            # Enhanced GPU monitoring every 2 minutes
            if time.time() - last_gpu_check > 120:
                perf_level = self.check_gpu_performance_enhanced()
                print(f"🔥 Periodic GPU check: {perf_level} performance")
                
                if perf_level in ["unknown"]:
                    print("⚠️ GPU performance unknown, performing recovery activation...")
                    self.force_gpu_activation()
                    time.sleep(2)
                    new_perf = self.check_gpu_performance_enhanced()
                    print(f"🔥 Recovery result: {new_perf} performance")
                    
                last_gpu_check = time.time()
            
            # Performance report every 10 minutes
            if time.time() - last_performance_report > 600:
                stats = self.get_current_gpu_stats()
                if stats:
                    print(f"📊 Performance Report - Jobs processed: {job_count}")
                    print(f"📊 Current GPU: {stats['graphics_clock']}MHz graphics, "
                         f"{stats['memory_clock']}MHz memory")
                    if stats['temperature'] > 0:
                        print(f"📊 Thermal: {stats['temperature']}°C, "
                             f"Power: {stats['power_draw']}W")
                    print(f"📊 Utilization: {stats['utilization']}%")
                else:
                    print("📊 Performance Report - GPU stats unavailable")
                last_performance_report = time.time()
            
            # Cleanup every 10 minutes
            if time.time() - last_cleanup > 600:
                self.cleanup_old_temp_files()
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    print(f"🧹 Memory cleanup - {memory_allocated:.1f}GB allocated")
                
                last_cleanup = time.time()
                
            # Poll for jobs
            job = self.poll_queue()
            if job:
                job_count += 1
                print(f"🎯 Processing job #{job_count}")
                print(f"🔥 Pre-job GPU check...")
                
                # Check GPU state before each job
                current_perf = self.check_gpu_performance_enhanced()
                if current_perf in ["unknown"]:
                    print(f"🔥 Current performance is {current_perf}, attempting activation...")
                    self.force_gpu_activation()
                
                self.process_job(job)
                print("⏳ Job complete, checking GPU state...")
                
                # Check GPU state after job
                post_job_perf = self.check_gpu_performance_enhanced()
                print(f"🔥 Post-job GPU performance: {post_job_perf}")
                
            else:
                time.sleep(5)

if __name__ == "__main__":
    print("🚀 Starting OurVidz Complete GPU Diagnostics Worker")
    print("🔥 Fixed GPU monitoring with realistic performance expectations")
    print("📊 Real-time generation monitoring and comprehensive diagnostics")
    print("🎯 Updated timing: 90s realistic target (vs previous unrealistic 35s)")
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Graceful shutdown requested...")
        if 'worker' in locals():
            worker.stop_real_time_gpu_monitor()
        print("✅ Worker shutdown complete")
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        worker = VideoWorker()
        worker.run()
    except Exception as e:
        print(f"❌ Worker failed to start: {e}")
        exit(1)
