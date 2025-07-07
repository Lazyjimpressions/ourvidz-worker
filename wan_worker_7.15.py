# wan_worker.py - Enhanced WAN Worker with Qwen 7B Integration
# Fixed: Polling issue resolved - proper 5-second intervals
# Date: July 5, 2025

import os
import json
import time
import torch
import requests
import subprocess
import tempfile
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

class EnhancedWanWorker:
    def __init__(self):
        """Initialize Enhanced WAN Worker with Qwen 7B integration"""
        self.model_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.wan_code_path = "/workspace/Wan2.1"
        
        # CRITICAL: Set environment variables immediately (VERIFIED FIX)
        os.environ['PYTHONPATH'] = '/workspace/python_deps/lib/python3.11/site-packages'
        os.environ['HF_HOME'] = '/workspace/models/huggingface_cache'
        os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/models/huggingface_cache/hub'
        
        # Updated HuggingFace cache configuration
        self.hf_cache_path = "/workspace/models/huggingface_cache"
        self.qwen_model_path = f"{self.hf_cache_path}/hub/models--Qwen--Qwen2.5-7B-Instruct"
        
        # Environment configuration
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        # Model instances (loaded on demand)
        self.qwen_model = None
        self.qwen_tokenizer = None
        
        # Job type configurations
        self.job_configs = {
            # Standard job types (no enhancement)
            'image_fast': {
                'size': '480*832',
                'sample_steps': 4,
                'sample_guide_scale': 3.0,
                'frame_num': 1,
                'enhance_prompt': False,
                'expected_time': 73
            },
            'image_high': {
                'size': '480*832',
                'sample_steps': 6,
                'sample_guide_scale': 4.0,
                'frame_num': 1,
                'enhance_prompt': False,
                'expected_time': 90
            },
            'video_fast': {
                'size': '480*832',
                'sample_steps': 4,
                'sample_guide_scale': 3.0,
                'frame_num': 17,
                'enhance_prompt': False,
                'expected_time': 180
            },
            'video_high': {
                'size': '480*832',
                'sample_steps': 6,
                'sample_guide_scale': 4.0,
                'frame_num': 17,
                'enhance_prompt': False,
                'expected_time': 280
            },
            
            # Enhanced job types (with Qwen 7B enhancement)
            'image7b_fast_enhanced': {
                'size': '480*832',
                'sample_steps': 4,
                'sample_guide_scale': 3.0,
                'frame_num': 1,
                'enhance_prompt': True,
                'expected_time': 87  # 73s + 14s enhancement
            },
            'image7b_high_enhanced': {
                'size': '480*832',
                'sample_steps': 6,
                'sample_guide_scale': 4.0,
                'frame_num': 1,
                'enhance_prompt': True,
                'expected_time': 104  # 90s + 14s enhancement
            },
            'video7b_fast_enhanced': {
                'size': '480*832',
                'sample_steps': 4,
                'sample_guide_scale': 3.0,
                'frame_num': 17,
                'enhance_prompt': True,
                'expected_time': 194  # 180s + 14s enhancement
            },
            'video7b_high_enhanced': {
                'size': '480*832',
                'sample_steps': 6,
                'sample_guide_scale': 4.0,
                'frame_num': 17,
                'enhance_prompt': True,
                'expected_time': 294  # 280s + 14s enhancement
            }
        }
        
        print("🎬 Enhanced OurVidz WAN Worker initialized")
        print(f"📁 WAN Model Path: {self.model_path}")
        print(f"🤖 Qwen Model Path: {self.qwen_model_path}")
        print(f"💾 HF Cache: {self.hf_cache_path}")
        self.log_gpu_memory()

    def log_gpu_memory(self):
        """Monitor RTX 6000 ADA 48GB VRAM usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🔥 GPU Memory - Used: {memory_allocated:.2f}GB / {total_memory:.0f}GB")

    def setup_environment(self):
        """Configure environment variables for WAN and Qwen - VERIFIED PATHS"""
        env = os.environ.copy()
        
        # CRITICAL: Add persistent dependencies to Python path
        python_deps_path = '/workspace/python_deps/lib/python3.11/site-packages'
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            new_pythonpath = f"{python_deps_path}:{current_pythonpath}"
        else:
            new_pythonpath = python_deps_path
        
        env.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'TORCH_USE_CUDA_DSA': '1',
            'PYTHONUNBUFFERED': '1',
            'PYTHONPATH': new_pythonpath,  # VERIFIED: Dependencies exist here
            'HF_HOME': self.hf_cache_path,  # VERIFIED: /workspace/models/huggingface_cache
            'TRANSFORMERS_CACHE': self.hf_cache_path,
            'HUGGINGFACE_HUB_CACHE': f"{self.hf_cache_path}/hub"  # VERIFIED: /hub/ subdirectory
        })
        return env

    def load_qwen_model(self):
        """Load Qwen 2.5-7B model for prompt enhancement"""
        if self.qwen_model is None:
            print("🤖 Loading Qwen 2.5-7B for prompt enhancement...")
            
            try:
                # Verify model path exists
                if not os.path.exists(self.qwen_model_path):
                    raise FileNotFoundError(f"Qwen model not found at {self.qwen_model_path}")
                
                # Load tokenizer and model from local cache
                self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                    self.qwen_model_path,
                    cache_dir=self.hf_cache_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    self.qwen_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=self.hf_cache_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                print("✅ Qwen 2.5-7B loaded successfully")
                self.log_gpu_memory()
                
            except Exception as e:
                print(f"❌ Failed to load Qwen model: {e}")
                # Fall back to no enhancement
                self.qwen_model = None
                self.qwen_tokenizer = None

    def unload_qwen_model(self):
        """Free Qwen memory for WAN generation"""
        if self.qwen_model is not None:
            print("🗑️ Unloading Qwen 2.5-7B...")
            del self.qwen_model
            del self.qwen_tokenizer
            self.qwen_model = None
            self.qwen_tokenizer = None
            torch.cuda.empty_cache()
            print("✅ Qwen 2.5-7B unloaded")
            self.log_gpu_memory()

    def enhance_prompt(self, original_prompt):
        """Use Qwen 2.5-7B to enhance user prompt"""
        self.load_qwen_model()
        
        if self.qwen_model is None:
            print("⚠️ Qwen model not available, using original prompt")
            return original_prompt
        
        try:
            system_prompt = """你是一个专业的视频制作提示词专家。请将用户的简单描述转换为详细的视频生成提示词。

要求：
1. 保持原始含义和主要元素
2. 添加具体的视觉细节（外观、服装、环境）
3. 包含镜头运动和拍摄角度
4. 添加光影效果和氛围描述
5. 确保描述适合5秒视频生成
6. 使用中文回复，内容要专业且具有电影感

请将以下用户输入转换为专业的视频生成提示词："""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_prompt}
            ]
            
            # Apply chat template
            text = self.qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize and generate
            model_inputs = self.qwen_tokenizer([text], return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                generated_ids = self.qwen_model.generate(
                    **model_inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.qwen_tokenizer.eos_token_id
                )
            
            # Extract enhanced prompt
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            enhanced_prompt = self.qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            self.unload_qwen_model()
            
            print(f"✅ Prompt enhanced: {original_prompt[:50]}... → {enhanced_prompt[:50]}...")
            return enhanced_prompt.strip()
            
        except Exception as e:
            print(f"❌ Prompt enhancement failed: {e}")
            self.unload_qwen_model()
            return original_prompt

    def generate_content(self, prompt, job_type):
        """Generate image or video content using WAN 2.1"""
        config = self.job_configs.get(job_type, self.job_configs['image_fast'])
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_video_path = Path(temp_file.name)
        
        try:
            # Change to WAN code directory
            original_cwd = os.getcwd()
            os.chdir(self.wan_code_path)
            
            # Build WAN generation command (VERIFIED OPTIMAL CONFIGURATION)
            cmd = [
                "python", "generate.py",
                "--task", "t2v-1.3B",
                "--ckpt_dir", self.model_path,
                "--offload_model", "False",  # ✅ CORRECT - prevents model offloading
                "--size", config['size'],
                "--sample_steps", str(config['sample_steps']),
                "--sample_guide_scale", str(config['sample_guide_scale']),
                "--frame_num", str(config['frame_num']),
                "--prompt", prompt,
                "--save_file", str(temp_video_path.absolute())
            ]
            
            # Configure environment
            env = self.setup_environment()
            
            print(f"🎬 Starting WAN generation: {job_type}")
            print(f"📝 Prompt: {prompt[:100]}...")
            
            # Execute WAN generation
            result = subprocess.run(
                cmd,
                cwd=self.wan_code_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            # Restore original directory
            os.chdir(original_cwd)
            
            if result.returncode == 0:
                if temp_video_path.exists() and temp_video_path.stat().st_size > 0:
                    print(f"✅ WAN generation successful: {temp_video_path.stat().st_size} bytes")
                    return str(temp_video_path)
                else:
                    raise Exception("Output file not created or empty")
            else:
                raise Exception(f"WAN generation failed: {result.stderr}")
                
        except Exception as e:
            os.chdir(original_cwd)  # Ensure we restore directory
            print(f"❌ WAN generation error: {e}")
            if temp_video_path.exists():
                temp_video_path.unlink()
            return None

    def upload_to_supabase(self, file_path, storage_path):
        """Upload file to Supabase storage"""
        try:
            with open(file_path, 'rb') as file:
                response = requests.post(
                    f"{self.supabase_url}/storage/v1/object/{storage_path}",
                    files={'file': file},
                    headers={
                        'Authorization': f"Bearer {self.supabase_service_key}",
                    }
                )
            
            if response.status_code == 200:
                # Return only relative path within bucket (avoid double-prefixing)
                path_parts = storage_path.split('/', 1)
                relative_path = path_parts[1] if len(path_parts) == 2 else storage_path
                print(f"✅ Upload successful: {relative_path}")
                return relative_path
            else:
                raise Exception(f"Upload failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Supabase upload error: {e}")
            raise

    def notify_completion(self, job_id, status, output_url=None, error_message=None):
        """Notify Supabase of job completion"""
        try:
            callback_data = {
                'jobId': job_id,
                'status': status,
                'outputUrl': output_url,
                'errorMessage': error_message
            }
            
            response = requests.post(
                f"{self.supabase_url}/functions/v1/job-callback",
                json=callback_data,
                headers={
                    'Authorization': f"Bearer {self.supabase_service_key}",
                    'Content-Type': 'application/json'
                }
            )
            
            if response.status_code == 200:
                print(f"✅ Job {job_id} callback sent successfully")
            else:
                print(f"❌ Callback failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Callback error: {e}")

    def process_job(self, job_data):
        """Process a single job"""
        job_id = job_data['jobId']
        job_type = job_data['jobType']
        original_prompt = job_data['prompt']
        video_id = job_data['videoId']
        
        print(f"🔄 Processing job {job_id} ({job_type})")
        
        try:
            # Validate job type
            if job_type not in self.job_configs:
                raise Exception(f"Unknown job type: {job_type}")
            
            config = self.job_configs[job_type]
            
            # Step 1: Enhance prompt if required
            if config['enhance_prompt']:
                print("🤖 Starting prompt enhancement...")
                enhanced_prompt = self.enhance_prompt(original_prompt)
                actual_prompt = enhanced_prompt
            else:
                actual_prompt = original_prompt
            
            # Step 2: Generate content with WAN 2.1
            print("🎬 Starting WAN generation...")
            output_file = self.generate_content(actual_prompt, job_type)
            
            if not output_file:
                raise Exception("Content generation failed")
            
            # Step 3: Upload to Supabase (user-scoped)
            file_extension = 'png' if config['frame_num'] == 1 else 'mp4'
            user_id = job_data.get('user_id', job_data.get('userId', 'unknown'))
            storage_path = f"{job_type}/{user_id}/{video_id}.{file_extension}"
            
            relative_path = self.upload_to_supabase(output_file, storage_path)
            
            # Step 4: Cleanup local file
            os.unlink(output_file)
            
            # Step 5: Notify completion
            self.notify_completion(job_id, 'completed', relative_path)
            
            print(f"🎉 Job {job_id} completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Job {job_id} failed: {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)

    def poll_queue(self):
        """Poll Redis queue for new jobs"""
        try:
            response = requests.get(
                f"{self.redis_url}/brpop/wan_queue/5",
                headers={
                    'Authorization': f"Bearer {self.redis_token}"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('result'):
                    queue_name, job_json = result['result']
                    job_data = json.loads(job_json)
                    return job_data
            
            return None
            
        except Exception as e:
            print(f"❌ Queue polling error: {e}")
            return None

    def run(self):
        """Main worker loop - FIXED: Proper 5-second polling"""
        print("🎬 Enhanced OurVidz WAN Worker with Qwen 7B started!")
        print("⏳ Waiting for jobs...")
        
        while True:
            try:
                job_data = self.poll_queue()
                
                if job_data:
                    self.process_job(job_data)
                else:
                    time.sleep(5)  # ✅ FIXED: Proper 5-second delay, no log spam
                    
            except KeyboardInterrupt:
                print("🛑 Worker stopped by user")
                break
            except Exception as e:
                print(f"❌ Worker error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    # Environment variable validation
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_SERVICE_KEY', 
        'UPSTASH_REDIS_REST_URL',
        'UPSTASH_REDIS_REST_TOKEN'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    # Verify critical paths
    model_path = "/workspace/models/wan2.1-t2v-1.3b"
    qwen_path = "/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B-Instruct"
    wan_code_path = "/workspace/Wan2.1"
    
    if not os.path.exists(model_path):
        print(f"❌ WAN model not found: {model_path}")
        exit(1)
        
    if not os.path.exists(qwen_path):
        print(f"⚠️ Qwen model not found: {qwen_path} (enhancement will be disabled)")
        
    if not os.path.exists(wan_code_path):
        print(f"❌ WAN code not found: {wan_code_path}")
        exit(1)
    
    print("✅ All paths validated, starting worker...")
    worker = EnhancedWanWorker()
    worker.run()
