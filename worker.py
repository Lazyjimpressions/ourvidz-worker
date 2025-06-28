# worker.py - GPU Activation Fix for RunPod Performance Issues
import os
import json
import time
import requests
import subprocess
import uuid
import shutil
import tempfile
import threading
from pathlib import Path
from PIL import Image
import cv2
import torch

class VideoWorker:
    def __init__(self):
        print("🚀 OurVidz Worker initialized (GPU ACTIVATION FIX)")
        print("⚡ Implementing RunPod RTX 4090 performance workarounds")
        
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
        
        # CRITICAL: Force GPU activation before anything else
        self.detect_gpu()
        self.force_gpu_activation()
        self.init_hardware_optimizations()
        self.start_gpu_keepalive()

        # Use temp storage for models - much faster I/O
        self.model_path = str(self.temp_models / 'wan2.1-t2v-1.3b')
        self.model_loaded = False

        # Job type mapping using supported resolutions
        self.job_type_mapping = {
            'image_fast': {
                'content_type': 'image',
                'resolution': 'small',            # 480×832 (fastest supported)
                'quality': 'fast',                # 8 steps, 5.5 guidance
                'storage_bucket': 'image_fast',
                'expected_time': 35,              # Optimistic with GPU fix
                'description': 'Small resolution, maximum speed'
            },
            'image_high': {
                'content_type': 'image',
                'resolution': 'standard',         # 832×480 (current working)
                'quality': 'balanced',            # 10 steps, 6.0 guidance  
                'storage_bucket': 'image_high',
                'expected_time': 50,              # Optimistic with GPU fix
                'description': 'Standard resolution, balanced quality'
            },
            'video_fast': {
                'content_type': 'video',
                'resolution': 'small',            # 480×832 (fastest supported)
                'quality': 'fast',                # 8 steps, 5.5 guidance
                'storage_bucket': 'video_fast',
                'expected_time': 45,              # Optimistic with GPU fix
                'description': 'Small resolution, fast video'
            },
            'video_high': {
                'content_type': 'video', 
                'resolution': 'standard',         # 832×480 (current working)
                'quality': 'balanced',            # 10 steps, 6.0 guidance
                'storage_bucket': 'video_high',
                'expected_time': 65,              # Optimistic with GPU fix
                'description': 'Standard resolution, quality video'
            }
        }
        
        # Resolution configurations using ONLY supported Wan 2.1 sizes
        self.resolution_configs = {
            'small': {
                'size': '480*832',              # ✅ SUPPORTED - Portrait, smaller
                'multiplier': 0.7,              # Speed improvement from smaller size
                'description': 'Small (480×832) - Fastest supported'
            },
            'standard': {
                'size': '832*480',              # ✅ SUPPORTED - Current working
                'multiplier': 1.0,              # Baseline
                'description': 'Standard (832×480) - Current working'
            }
        }
        
        # Quality configurations optimized for speed
        self.quality_configs = {
            'fast': {
                'sample_steps': 8,              # Minimum reasonable steps
                'sample_guide_scale': 5.5,      # Lower guidance for speed
                'description': 'Fast - Speed optimized'
            },
            'balanced': {
                'sample_steps': 10,             # Balanced steps
                'sample_guide_scale': 6.0,      # Moderate guidance
                'description': 'Balanced - Speed/quality balance'
            }
        }

        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')

        print("🎬 GPU Activation Worker ready")
        print("🔥 GPU performance workarounds applied")

    def detect_gpu(self):
        """Enhanced GPU detection with performance analysis"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used,clocks.current.graphics,clocks.current.memory,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                print(f"🔥 GPU: {gpu_info[0]} ({gpu_info[1]}GB total)")
                print(f"💾 VRAM: {gpu_info[3]}MB used, {gpu_info[2]}MB free")
                print(f"⚡ Graphics Clock: {gpu_info[4]}MHz")
                print(f"💽 Memory Clock: {gpu_info[5]}MHz") 
                print(f"🔌 Power Draw: {gpu_info[6]}W")
                
                # Alert if clocks are low
                graphics_clock = int(gpu_info[4])
                if graphics_clock < 1000:
                    print(f"⚠️ WARNING: Graphics clock {graphics_clock}MHz is very low!")
                    print("🔧 Attempting GPU activation...")
                    
        except Exception as e:
            print(f"⚠️ GPU detection failed: {e}")

    def force_gpu_activation(self):
        """Force GPU out of P8 idle state into active performance state"""
        print("🔥 Forcing GPU activation to escape P8 idle state...")
        
        try:
            # Initialize CUDA if not already done
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                print(f"✅ CUDA available with {device_count} device(s)")
                
                # Force GPU memory allocation and computation
                print("🔥 Allocating GPU memory to force active state...")
                activation_tensor = torch.zeros((2000, 2000), dtype=torch.float16, device='cuda')
                
                print("🔥 Running computation to activate performance clocks...")
                for i in range(15):
                    # Force intensive computation to wake up GPU
                    result = torch.matmul(activation_tensor, activation_tensor)
                    result = torch.sin(result)
                    result = torch.exp(result)
                    torch.cuda.synchronize()  # Force completion
                    
                    if i % 5 == 0:
                        print(f"🔥 Activation iteration {i+1}/15...")
                
                # Check memory usage to confirm activation
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"✅ GPU activation complete - {memory_allocated:.1f}GB allocated")
                
                # Clean up activation tensors
                del activation_tensor, result
                torch.cuda.empty_cache()
                
                # Check if clocks improved
                time.sleep(2)
                subprocess.run(['nvidia-smi', '--query-gpu=clocks.current.graphics,clocks.current.memory,power.draw', '--format=csv,noheader,nounits'], capture_output=False)
                
            else:
                print("❌ CUDA not available!")
                
        except Exception as e:
            print(f"⚠️ GPU activation failed: {e}")

    def start_gpu_keepalive(self):
        """Start background thread to keep GPU in active state"""
        print("🔥 Starting GPU keepalive thread...")
        
        def gpu_keepalive():
            try:
                # Keep a small tensor active to prevent P8 idle state
                keepalive_tensor = torch.ones((200, 200), dtype=torch.float16, device='cuda')
                iteration = 0
                
                while True:
                    try:
                        # Small computation every 10 seconds
                        result = torch.matmul(keepalive_tensor, keepalive_tensor)
                        torch.cuda.synchronize()
                        
                        iteration += 1
                        if iteration % 30 == 0:  # Log every 5 minutes
                            print(f"🔥 GPU keepalive active (iteration {iteration})")
                            
                        time.sleep(10)  # Wait 10 seconds between computations
                        
                    except Exception as e:
                        print(f"⚠️ Keepalive error: {e}")
                        time.sleep(30)  # Wait longer if there's an error
                        
            except Exception as e:
                print(f"❌ Keepalive thread failed: {e}")
        
        # Start background thread
        keepalive_thread = threading.Thread(target=gpu_keepalive, daemon=True)
        keepalive_thread.start()
        print("✅ GPU keepalive thread started")

    def init_hardware_optimizations(self):
        """Initialize hardware optimizations with RunPod workarounds"""
        print("🔧 Initializing hardware optimizations...")
        
        try:
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.version.cuda}")
                
                # Aggressive memory and performance settings
                os.environ.update({
                    'CUDA_LAUNCH_BLOCKING': '0',
                    'CUDA_CACHE_DISABLE': '0',
                    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,roundup_power2_divisions:16',
                    'TORCH_USE_CUDA_DSA': '1',
                })
                
                # PyTorch performance settings
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True  
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Force PyTorch to use maximum performance
                if hasattr(torch, 'set_float32_matmul_precision'):
                    torch.set_float32_matmul_precision('medium')  # Use TensorFloat-32
                
                print("✅ Hardware optimizations applied")
                
            else:
                print("⚠️ CUDA not available")
                
        except Exception as e:
            print(f"⚠️ Hardware optimization failed: {e}")

    def check_gpu_performance(self):
        """Check if GPU is in active performance state"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=clocks.current.graphics,clocks.current.memory,power.draw,performance.state', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                graphics, memory, power, perf_state = result.stdout.strip().split(', ')
                
                print(f"📊 GPU Status: {graphics}MHz graphics, {memory}MHz memory, {power}W, {perf_state}")
                
                # Return performance level assessment
                graphics_mhz = int(graphics)
                if graphics_mhz > 1500:
                    return "high"
                elif graphics_mhz > 800:
                    return "medium"  
                else:
                    return "low"
        except:
            pass
        return "unknown"

    def ensure_model_ready(self):
        """Ensure model is ready with GPU performance check"""
        if os.path.exists(self.model_path):
            print("✅ Model already in temp storage")
            
            # Check GPU performance before proceeding
            perf_level = self.check_gpu_performance()
            if perf_level == "low":
                print("⚠️ GPU performance low, forcing reactivation...")
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
        """Get configuration using only supported resolutions"""
        job_mapping = self.job_type_mapping.get(job_type)
        if not job_mapping:
            return {
                'size': '832*480',
                'frame_num': 1,
                'sample_steps': 10,
                'sample_guide_scale': 6.0,
                'expected_time': 50,
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
        return f"{job_mapping.get('expected_time', 50)}s"

    def generate(self, prompt, job_type):
        """Enhanced generation with GPU performance monitoring"""
        config = self.get_job_config(job_type)

        # Ensure model and GPU are ready
        if not self.ensure_model_ready():
            return None

        # Check GPU performance before generation
        perf_level = self.check_gpu_performance()
        print(f"🔥 GPU Performance Level: {perf_level}")

        job_id = str(uuid.uuid4())[:8]
        expected_time = config['expected_time']

        print(f"⚡ {job_type.upper()} generation (GPU: {perf_level})")
        print(f"📝 Prompt: {prompt}")
        print(f"📐 Resolution: {config.get('resolution_desc', 'unknown')}")
        print(f"⚙️ Quality: {config.get('quality_desc', 'unknown')}")
        print(f"🔧 Config: {config['sample_steps']} steps, {config['sample_guide_scale']} guidance, {config['size']}")
        print(f"🎯 Expected: {expected_time}s (with GPU activation)")

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
            
            # Enhanced environment for performance
            env = os.environ.copy()
            env.update({
                'CUDA_LAUNCH_BLOCKING': '0',
                'TORCH_USE_CUDA_DSA': '1',
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
                'CUDA_CACHE_DISABLE': '0',
            })
            
            # Force one more GPU activation right before generation
            if perf_level == "low":
                print("🔥 GPU performance low, activating before generation...")
                temp_tensor = torch.randn((500, 500), device='cuda')
                torch.matmul(temp_tensor, temp_tensor)
                torch.cuda.synchronize()
                del temp_tensor
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
            generation_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"❌ Generation failed: {result.stderr}")
                return None
                
            # Check final GPU performance
            final_perf = self.check_gpu_performance()
            print(f"⚡ Generation completed in {generation_time:.1f}s (expected {expected_time}s)")
            print(f"🔥 Final GPU performance: {final_perf}")
                
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
            os.chdir(original_cwd)

    # [Include all other methods from the previous worker.py...]
    # [extract_frame_from_video, optimize_file_for_upload, upload_to_supabase, etc.]
    # [For brevity, I'm showing just the key GPU optimization parts]

    def run(self):
        """Enhanced main loop with GPU performance monitoring"""
        print("⏳ Waiting for jobs...")
        print("🎯 GPU Activation Targets (with RTX 4090 fixes):")
        print("   • image_fast: 35s (if GPU activation works)")
        print("   • image_high: 50s (if GPU activation works)")  
        print("   • Previous: 93s (with GPU in P8 idle mode)")
        print("🔥 GPU keepalive thread running to prevent P8 idle")
        
        last_cleanup = time.time()
        last_gpu_check = time.time()
        job_count = 0
        
        while True:
            # Monitor GPU performance every 5 minutes
            if time.time() - last_gpu_check > 300:
                perf_level = self.check_gpu_performance()
                print(f"🔥 Periodic GPU check: {perf_level} performance")
                if perf_level == "low":
                    print("⚠️ GPU performance degraded, reactivating...")
                    self.force_gpu_activation()
                last_gpu_check = time.time()
            
            # Cleanup every 10 minutes
            if time.time() - last_cleanup > 600:
                # [cleanup logic]
                last_cleanup = time.time()
                
            job = self.poll_queue()
            if job:
                job_count += 1
                print(f"🎯 Processing job #{job_count}")
                self.process_job(job)
                print("⏳ Job complete, checking queue...")
            else:
                time.sleep(5)

    # [Include remaining methods: extract_frame_from_video, upload_to_supabase, etc.]

if __name__ == "__main__":
    print("🚀 Starting OurVidz Worker (GPU ACTIVATION FIX)")
    print("🔥 Implementing RTX 4090 performance workarounds for RunPod")
    worker = VideoWorker()
    worker.run()
