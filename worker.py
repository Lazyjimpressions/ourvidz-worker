# worker.py - COMPREHENSIVE CUDA DIAGNOSTIC MODE
# Systematic diagnosis of GPU utilization issues following established checklist
import os
import json
import time
import requests
import subprocess
import uuid
import shutil
from pathlib import Path
from PIL import Image
import cv2
import sys

# Clean environment first
for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
    if key in os.environ:
        del os.environ[key]

# Import torch after cleaning environment
import torch
import torch.nn as nn
import numpy as np

class CUDADiagnosticWorker:
    def __init__(self):
        print("🔍 CUDA DIAGNOSTIC WORKER - Comprehensive GPU Analysis")
        print("📋 Following systematic checklist to identify CPU fallback issues")
        
        # Run comprehensive CUDA diagnostics
        self.run_cuda_diagnostics()
        
        # Paths
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_path = "/workspace/Wan2.1"
        
        # Job configurations
        self.job_type_mapping = {
            'image_fast': {
                'content_type': 'image',
                'file_extension': 'png',
                'sample_steps': 4,
                'sample_guide_scale': 3.0,
                'size': '480*832',
                'frame_num': 1,
                'storage_bucket': 'image_fast'
            }
        }
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        print("🔧 Diagnostic worker ready for systematic analysis")

    def run_cuda_diagnostics(self):
        """Comprehensive CUDA diagnostics following checklist"""
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE CUDA DIAGNOSTICS")
        print("="*80)
        
        # 1. Confirm Torch CUDA Availability
        self.check_torch_cuda_availability()
        
        # 2. Match Torch/CUDA combo with GPU
        self.check_torch_cuda_compatibility()
        
        # 3. Monitor System RAM
        self.check_system_resources()
        
        # 4. Check GPU Architecture Support
        self.check_gpu_architecture_support()
        
        # 5. Test GPU Tensor Operations
        self.test_gpu_tensor_operations()
        
        # 6. Check CUDA Runtime Environment
        self.check_cuda_runtime_environment()
        
        # 7. Test Real GPU Memory Allocation
        self.test_gpu_memory_allocation()
        
        # 8. Check for Multi-GPU Conflicts
        self.check_multi_gpu_conflicts()
        
        # 9. Test Flash Attention 2 Properly
        self.test_flash_attention_properly()
        
        print("="*80)
        print("🔍 CUDA DIAGNOSTICS COMPLETE")
        print("="*80 + "\n")

    def check_torch_cuda_availability(self):
        """Step 1: Confirm Torch CUDA Availability"""
        print("\n📋 STEP 1: Torch CUDA Availability")
        print("-" * 40)
        
        # Basic availability
        cuda_available = torch.cuda.is_available()
        print(f"   torch.cuda.is_available(): {cuda_available}")
        
        if not cuda_available:
            print("   ❌ CRITICAL: PyTorch cannot see CUDA")
            print("   💡 This likely means CPU-only PyTorch installation")
            return False
        
        # Device count
        device_count = torch.cuda.device_count()
        print(f"   torch.cuda.device_count(): {device_count}")
        
        # Current device
        try:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"   Current device: {current_device}")
            print(f"   Device name: {device_name}")
            
            # Memory info
            props = torch.cuda.get_device_properties(current_device)
            total_memory = props.total_memory / (1024**3)
            print(f"   Total memory: {total_memory:.1f}GB")
            print(f"   Compute capability: {props.major}.{props.minor}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error getting device info: {e}")
            return False

    def check_torch_cuda_compatibility(self):
        """Step 2: Match Torch/CUDA combo with GPU"""
        print("\n📋 STEP 2: Torch/CUDA Compatibility")
        print("-" * 40)
        
        # PyTorch version
        torch_version = torch.__version__
        print(f"   PyTorch version: {torch_version}")
        
        # CUDA version that PyTorch was compiled with
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"   PyTorch CUDA version: {cuda_version}")
            
            # Check if this is a CUDA build
            if '+cu' in torch_version:
                cuda_build = torch_version.split('+cu')[1]
                print(f"   PyTorch CUDA build: cu{cuda_build}")
            else:
                print("   ⚠️ WARNING: PyTorch version doesn't indicate CUDA build")
        
        # System CUDA version
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                nvcc_output = result.stdout
                for line in nvcc_output.split('\n'):
                    if 'release' in line.lower():
                        print(f"   System CUDA (nvcc): {line.strip()}")
        except:
            print("   ⚠️ nvcc not found")
        
        # Driver version
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                driver_version = result.stdout.strip()
                print(f"   NVIDIA driver: {driver_version}")
        except:
            print("   ⚠️ nvidia-smi not found")

    def check_system_resources(self):
        """Step 3: Monitor System RAM"""
        print("\n📋 STEP 3: System Resources")
        print("-" * 40)
        
        # Memory usage
        try:
            result = subprocess.run(['free', '-h'], capture_output=True, text=True)
            if result.returncode == 0:
                print("   System memory:")
                for line in result.stdout.split('\n')[1:3]:  # Mem and Swap lines
                    if line.strip():
                        print(f"     {line}")
        except:
            print("   ⚠️ Cannot check system memory")
        
        # CPU info
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
                cpu_count = cpu_info.count('processor')
                print(f"   CPU cores: {cpu_count}")
        except:
            print("   ⚠️ Cannot read CPU info")

    def check_gpu_architecture_support(self):
        """Step 4: Check GPU Architecture Support"""
        print("\n📋 STEP 4: GPU Architecture Support")
        print("-" * 40)
        
        if not torch.cuda.is_available():
            print("   ❌ Cannot check - CUDA not available")
            return
        
        # Get compute capability
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        major, minor = props.major, props.minor
        compute_cap = f"{major}.{minor}"
        
        print(f"   GPU compute capability: {compute_cap}")
        
        # Map to architecture name
        arch_map = {
            (8, 9): "Ada Lovelace (RTX 40 series)",
            (8, 6): "Ampere (RTX 30 series)",
            (7, 5): "Turing (RTX 20 series)",
            (6, 1): "Pascal (GTX 10 series)"
        }
        
        arch_name = arch_map.get((major, minor), f"Unknown architecture {compute_cap}")
        print(f"   Architecture: {arch_name}")
        
        # Check if PyTorch supports this architecture
        if major >= 8:
            print("   ✅ Modern GPU architecture")
            if minor == 9:
                print("   📝 Ada Lovelace requires PyTorch 1.13+ with CUDA 11.8+")
        else:
            print("   ⚠️ Older GPU architecture")

    def test_gpu_tensor_operations(self):
        """Step 5: Test GPU Tensor Operations"""
        print("\n📋 STEP 5: GPU Tensor Operations Test")
        print("-" * 40)
        
        if not torch.cuda.is_available():
            print("   ❌ Cannot test - CUDA not available")
            return
        
        try:
            # Create tensors on GPU
            print("   Creating test tensors on GPU...")
            device = torch.device('cuda')
            
            # Test tensor creation
            a = torch.randn(1000, 1000, device=device, dtype=torch.float32)
            b = torch.randn(1000, 1000, device=device, dtype=torch.float32)
            
            # Check memory usage
            allocated_before = torch.cuda.memory_allocated() / (1024**3)
            print(f"   GPU memory after tensor creation: {allocated_before:.2f}GB")
            
            if allocated_before < 0.001:
                print("   ❌ CRITICAL: No GPU memory allocated - tensors may be on CPU!")
                return False
            
            # Test computation
            start_time = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()  # Wait for GPU computation
            gpu_time = time.time() - start_time
            
            print(f"   GPU matrix multiplication time: {gpu_time:.4f}s")
            
            # Test CPU for comparison
            a_cpu = a.cpu()
            b_cpu = b.cpu()
            start_time = time.time()
            c_cpu = torch.matmul(a_cpu, b_cpu)
            cpu_time = time.time() - start_time
            
            print(f"   CPU matrix multiplication time: {cpu_time:.4f}s")
            print(f"   GPU speedup: {cpu_time/gpu_time:.1f}x")
            
            # Cleanup
            del a, b, c, a_cpu, b_cpu, c_cpu
            torch.cuda.empty_cache()
            
            if gpu_time < cpu_time:
                print("   ✅ GPU computation working correctly")
                return True
            else:
                print("   ❌ GPU not faster than CPU - possible issue")
                return False
                
        except Exception as e:
            print(f"   ❌ GPU tensor test failed: {e}")
            return False

    def check_cuda_runtime_environment(self):
        """Step 6: Check CUDA Runtime Environment"""
        print("\n📋 STEP 6: CUDA Runtime Environment")
        print("-" * 40)
        
        # Environment variables
        cuda_env_vars = [
            'CUDA_VISIBLE_DEVICES',
            'CUDA_DEVICE_ORDER',
            'TORCH_USE_CUDA_DSA',
            'PYTORCH_CUDA_ALLOC_CONF'
        ]
        
        for var in cuda_env_vars:
            value = os.environ.get(var)
            if value:
                print(f"   {var}: {value}")
            else:
                print(f"   {var}: (not set)")
        
        # CUDA runtime version
        if torch.cuda.is_available():
            try:
                runtime_version = torch.version.cuda
                print(f"   CUDA runtime version: {runtime_version}")
            except:
                print("   ⚠️ Cannot get CUDA runtime version")

    def test_gpu_memory_allocation(self):
        """Step 7: Test Real GPU Memory Allocation"""
        print("\n📋 STEP 7: GPU Memory Allocation Test")
        print("-" * 40)
        
        if not torch.cuda.is_available():
            print("   ❌ Cannot test - CUDA not available")
            return
        
        try:
            device = torch.device('cuda')
            
            # Test progressively larger allocations
            sizes = [100, 500, 1000, 2000]  # MB
            
            for size_mb in sizes:
                size_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
                
                print(f"   Allocating {size_mb}MB tensor...")
                
                try:
                    tensor = torch.randn(size_elements, device=device, dtype=torch.float32)
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    print(f"     ✅ Success - GPU memory: {allocated:.2f}GB")
                    del tensor
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"     ❌ Failed: {e}")
                    break
                    
        except Exception as e:
            print(f"   ❌ Memory allocation test failed: {e}")

    def test_flash_attention_properly(self):
        """Step 9: Test Flash Attention 2 Properly"""
        print("\n📋 STEP 9: Flash Attention 2 Comprehensive Test")
        print("-" * 40)
        
        # Import test
        try:
            import flash_attn
            from flash_attn import flash_attn_func
            print("   ✅ Flash Attention 2 import successful")
            print(f"   Flash Attention version: {getattr(flash_attn, '__version__', 'unknown')}")
        except ImportError as e:
            print(f"   ❌ Flash Attention 2 import failed: {e}")
            return False
        
        if not torch.cuda.is_available():
            print("   ❌ Cannot test Flash Attention - CUDA not available")
            return False
        
        # Proper Flash Attention test with correct dimensions
        try:
            device = torch.device('cuda')
            
            # Flash Attention expects: (batch_size, seq_len, num_heads, head_dim)
            # Common transformer dimensions
            batch_size = 1
            seq_len = 512  # Sequence length
            num_heads = 8  # Number of attention heads
            head_dim = 64  # Dimension per head
            
            print(f"   Testing with dimensions: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}")
            
            # Create properly shaped tensors
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
            
            print("   Created test tensors with correct Flash Attention dimensions")
            
            # Test Flash Attention function
            start_time = time.time()
            try:
                # Flash Attention 2 function call
                out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
                flash_time = time.time() - start_time
                
                print(f"   ✅ Flash Attention 2 functional test PASSED")
                print(f"   Flash Attention computation time: {flash_time:.4f}s")
                print(f"   Output shape: {out.shape}")
                
                # Verify GPU memory was used
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                print(f"   GPU memory during Flash Attention: {gpu_memory:.2f}GB")
                
                if gpu_memory < 0.001:
                    print("   ⚠️ WARNING: No GPU memory usage detected during Flash Attention")
                
                # Cleanup
                del q, k, v, out
                torch.cuda.empty_cache()
                
                return True
                
            except Exception as e:
                flash_error = str(e)
                print(f"   ❌ Flash Attention 2 functional test FAILED: {flash_error}")
                
                # Analyze the error
                if "dimension out of range" in flash_error.lower():
                    print("   💡 Dimension error suggests Flash Attention version/compilation issue")
                elif "cuda" in flash_error.lower():
                    print("   💡 CUDA error suggests GPU/driver compatibility issue")
                elif "unsupported" in flash_error.lower():
                    print("   💡 Unsupported operation suggests architecture mismatch")
                
                # Test standard PyTorch attention as fallback
                print("   🔄 Testing standard PyTorch attention as comparison...")
                self.test_standard_attention(q, k, v)
                
                return False
                
        except Exception as e:
            print(f"   ❌ Flash Attention test setup failed: {e}")
            return False

    def test_standard_attention(self, q, k, v):
        """Test standard PyTorch attention for comparison"""
        try:
            # Reshape for standard attention: (batch, heads, seq_len, head_dim)
            q_std = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
            k_std = k.transpose(1, 2)
            v_std = v.transpose(1, 2)
            
            start_time = time.time()
            # Standard scaled dot-product attention
            out_std = torch.nn.functional.scaled_dot_product_attention(
                q_std, k_std, v_std, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            std_time = time.time() - start_time
            
            print(f"   ✅ Standard PyTorch attention works: {std_time:.4f}s")
            print(f"   Standard attention output shape: {out_std.shape}")
            
            # Check GPU usage
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"   GPU memory during standard attention: {gpu_memory:.2f}GB")
            
            del q_std, k_std, v_std, out_std
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"   ❌ Standard attention also failed: {e}")
            return False

    def check_flash_attention_build_info(self):
        """Check Flash Attention build information"""
        print("\n🔍 Flash Attention Build Information")
        print("-" * 40)
        
        try:
            import flash_attn
            
            # Version info
            if hasattr(flash_attn, '__version__'):
                print(f"   Version: {flash_attn.__version__}")
            
            # Check if it was compiled for current architecture
            device = torch.device('cuda')
            props = torch.cuda.get_device_properties(device)
            arch = f"sm_{props.major}{props.minor}"
            
            print(f"   GPU architecture: {arch}")
            
            # Try to get build info (if available)
            try:
                # Some versions expose build info
                if hasattr(flash_attn, '__cuda_version__'):
                    print(f"   Flash Attention CUDA version: {flash_attn.__cuda_version__}")
            except:
                pass
            
        except Exception as e:
            print(f"   Error getting Flash Attention build info: {e}")
        """Step 8: Check for Multi-GPU Conflicts"""
        print("\n📋 STEP 8: Multi-GPU Conflicts")
        print("-" * 40)
        
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"   Available GPU devices: {device_count}")
        
        if device_count > 1:
            print("   ⚠️ Multiple GPUs detected")
            print("   💡 Ensure CUDA_VISIBLE_DEVICES=0 to use only first GPU")
        else:
            print("   ✅ Single GPU configuration")
        
        # Check for distributed training environment variables
        dist_vars = ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']
        dist_set = [var for var in dist_vars if os.environ.get(var)]
        
        if dist_set:
            print(f"   ⚠️ Distributed training vars set: {dist_set}")
            print("   💡 Consider removing for single GPU setup")
    def check_multi_gpu_conflicts(self):
        """Step 8: Check for Multi-GPU Conflicts"""
        print("\n📋 STEP 8: Multi-GPU Conflicts")
        print("-" * 40)
        
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"   Available GPU devices: {device_count}")
        
        if device_count > 1:
            print("   ⚠️ Multiple GPUs detected")
            print("   💡 Ensure CUDA_VISIBLE_DEVICES=0 to use only first GPU")
        else:
            print("   ✅ Single GPU configuration")
        
        # Check for distributed training environment variables
        dist_vars = ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']
        dist_set = [var for var in dist_vars if os.environ.get(var)]
        
        if dist_set:
            print(f"   ⚠️ Distributed training vars set: {dist_set}")
            print("   💡 Consider removing for single GPU setup")
        else:
            print("   ✅ No distributed training variables set")

    def diagnose_wan_gpu_usage(self, prompt):
        """Diagnose GPU usage during Wan 2.1 generation"""
        print("\n🔍 WAN 2.1 GPU USAGE DIAGNOSIS")
        print("-" * 50)
        
        config = self.job_type_mapping['image_fast']
        job_id = str(uuid.uuid4())[:8]
        
        # Create temp directories
        temp_base = Path("/tmp/ourvidz")
        temp_base.mkdir(exist_ok=True)
        temp_processing = temp_base / "processing"
        temp_processing.mkdir(exist_ok=True)
        
        temp_video_path = temp_processing / f"diagnostic_{job_id}.mp4"
        
        # Build command with GPU monitoring
        cmd = [
            "python", "generate.py",
            "--task", "t2v-1.3B",
            "--ckpt_dir", self.model_path,
            "--offload_model", "False",  # Keep on GPU
            "--size", config['size'],
            "--sample_steps", str(config['sample_steps']),
            "--sample_guide_scale", str(config['sample_guide_scale']),
            "--frame_num", str(config['frame_num']),
            "--prompt", prompt,
            "--save_file", str(temp_video_path.absolute())
        ]
        
        print(f"📋 Command: {' '.join(cmd)}")
        
        # Environment with GPU forced
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'TORCH_USE_CUDA_DSA': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
            'PYTHONUNBUFFERED': '1'
        })
        
        original_cwd = os.getcwd()
        
        try:
            os.chdir(self.wan_path)
            
            print("🚀 Starting generation with intensive GPU monitoring...")
            
            # Start subprocess
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Monitor GPU usage during generation
            gpu_peaks = []
            start_time = time.time()
            
            while process.poll() is None:
                # Check GPU memory every 2 seconds
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    
                    if allocated > 0.1:  # More than 100MB
                        gpu_peaks.append({
                            'time': time.time() - start_time,
                            'allocated': allocated,
                            'reserved': reserved
                        })
                        print(f"   🔥 GPU usage detected: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
                time.sleep(2)
            
            # Get final output
            stdout, stderr = process.communicate()
            generation_time = time.time() - start_time
            
            print(f"\n📊 GENERATION RESULTS:")
            print(f"   Total time: {generation_time:.1f}s")
            print(f"   Return code: {process.returncode}")
            print(f"   GPU peaks detected: {len(gpu_peaks)}")
            
            if gpu_peaks:
                max_usage = max(peak['allocated'] for peak in gpu_peaks)
                print(f"   Peak GPU usage: {max_usage:.2f}GB")
                print("   ✅ GPU was actively used during generation")
            else:
                print("   ❌ NO GPU usage detected during generation")
                print("   💡 This confirms CPU fallback is occurring")
            
            # Print last few lines of output
            if stderr:
                print(f"\n📋 Generation stderr (last 5 lines):")
                for line in stderr.split('\n')[-6:-1]:
                    if line.strip():
                        print(f"   {line}")
            
            return {
                'success': process.returncode == 0,
                'gpu_used': len(gpu_peaks) > 0,
                'peak_usage': max((peak['allocated'] for peak in gpu_peaks), default=0),
                'generation_time': generation_time
            }
            
        except Exception as e:
            print(f"❌ Diagnostic generation failed: {e}")
            return {'success': False, 'gpu_used': False, 'error': str(e)}
        finally:
            os.chdir(original_cwd)

    def generate_fix_recommendations(self):
        """Generate specific fix recommendations based on diagnostics"""
        print("\n💡 FIX RECOMMENDATIONS")
        print("="*50)
        
        if not torch.cuda.is_available():
            print("🔧 CRITICAL: Install CUDA-enabled PyTorch")
            print("   pip uninstall torch torchvision")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            return
        
        # Check if we have modern GPU with potential architecture issues
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if props.major >= 8 and props.minor >= 9:  # Ada Lovelace
                print("🔧 Ada Lovelace GPU detected - ensuring latest PyTorch")
                print("   pip install torch==2.4.1+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124")
            
            # Check memory allocation capability
            try:
                test_tensor = torch.randn(1000, 1000, device='cuda')
                allocated = torch.cuda.memory_allocated()
                del test_tensor
                torch.cuda.empty_cache()
                
                if allocated == 0:
                    print("🔧 GPU memory allocation issue detected")
                    print("   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
                    print("   export CUDA_LAUNCH_BLOCKING=1")
            except Exception as e:
                print(f"🔧 GPU allocation test failed: {e}")
                print("   Check CUDA driver compatibility")

    def poll_queue(self):
        """Poll Redis queue for diagnostic jobs"""
        try:
            response = requests.get(
                f"{self.redis_url}/rpop/job_queue",
                headers={'Authorization': f"Bearer {self.redis_token}"},
                timeout=5
            )
            if response.status_code == 200 and response.json().get('result'):
                return json.loads(response.json()['result'])
        except Exception as e:
            print(f"❌ Poll error: {e}")
        return None

    def run(self):
        """Main diagnostic loop"""
        print("\n🎬 CUDA DIAGNOSTIC WORKER READY!")
        print("🔍 Will perform comprehensive diagnosis on first job")
        print("⏳ Waiting for jobs...\n")
        
        job_count = 0
        
        while True:
            job = self.poll_queue()
            if job:
                job_count += 1
                print(f"🚀 === DIAGNOSTIC JOB #{job_count} ===")
                
                # Run Wan 2.1 diagnostic on first job
                if job_count == 1:
                    prompt = job.get('prompt', 'diagnostic test')
                    result = self.diagnose_wan_gpu_usage(prompt)
                    
                    print(f"\n🎯 DIAGNOSTIC SUMMARY:")
                    print(f"   Generation successful: {result.get('success', False)}")
                    print(f"   GPU utilized: {result.get('gpu_used', False)}")
                    print(f"   Peak GPU usage: {result.get('peak_usage', 0):.2f}GB")
                    print(f"   Generation time: {result.get('generation_time', 0):.1f}s")
                    
                    if not result.get('gpu_used', False):
                        print("\n❌ CONFIRMED: CPU FALLBACK OCCURRING")
                        self.generate_fix_recommendations()
                    else:
                        print("\n✅ GPU utilization confirmed")
                
                print("=" * 80)
            else:
                time.sleep(5)

if __name__ == "__main__":
    print("🔍 Starting CUDA Diagnostic Worker")
    
    # Verify environment
    required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        exit(1)
    
    try:
        worker = CUDADiagnosticWorker()
        worker.run()
    except Exception as e:
        print(f"❌ Diagnostic worker failed: {e}")
        exit(1)
