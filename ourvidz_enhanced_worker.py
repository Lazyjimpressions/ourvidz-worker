# ourvidz_enhanced_worker.py - Multi-Model Integration
import os
import json
import time
import torch
import requests
import subprocess
from typing import Dict, Optional, List
from pathlib import Path

class ModelConfig:
    """Functional model configuration based on quality/speed tiers"""
    
    MODELS = {
        "image_low_res": {
            "repo": "Wan-AI/Wan2.1-Lightning-T5",
            "path": "/workspace/models/wan_lightning",
            "media_type": "image",
            "quality": "low_res",
            "task": "lightning-t5",
            "size": "720*720",
            "steps": 5,
            "time_estimate": "<1s",
            "vram_gb": 8,
            "use_cases": ["preview", "quick_concept", "thumbnail"]
        },
        "image_high_res": {
            "repo": "Wan-AI/Wan2.1-Base-Diffusion", 
            "path": "/workspace/models/wan_base",
            "media_type": "image",
            "quality": "high_res",
            "task": "base-diffusion",
            "size": "1024*1024", 
            "steps": 25,
            "time_estimate": "10-15s",
            "vram_gb": 12,
            "use_cases": ["character_reference", "final_image", "consistency_check"]
        },
        "video_fast": {
            "repo": "Wan-AI/Wan2.1-T2V-1.3B",
            "path": "/workspace/models/wan2.1-t2v-1.3b",
            "media_type": "video",
            "quality": "fast",
            "task": "t2v-1.3B",
            "size": "832*480",
            "time_estimate": "4-6min",
            "vram_gb": 18,
            "use_cases": ["quick_video", "preview_video", "standard_content"]
        },
        "video_premium": {
            "repo": "Wan-AI/Wan2.1-T2V-14B", 
            "path": "/workspace/models/wan_14b",
            "media_type": "video",
            "quality": "premium",
            "task": "t2v-14B",
            "size": "1280*720",
            "time_estimate": "6-8min", 
            "vram_gb": 22,
            "use_cases": ["high_quality_video", "final_production", "professional_content"]
        }
    }
    
    @classmethod
    def get_model_for_job(cls, media_type: str, quality: str = "fast") -> Optional[Dict]:
        """Map functional requests to model configurations"""
        
        # Functional mapping based on media type and quality
        functional_key = f"{media_type}_{quality}"
        
        # Handle legacy/alternative naming
        legacy_mapping = {
            # Image alternatives
            "image_low": "image_low_res",
            "image_quick": "image_low_res", 
            "image_preview": "image_low_res",
            "image_high": "image_high_res",
            "image_detailed": "image_high_res",
            "image_character": "image_high_res",
            
            # Video alternatives  
            "video_low": "video_fast",
            "video_standard": "video_fast",
            "video_high": "video_premium",
            "video_hd": "video_premium",
            
            # Legacy support
            "preview": "image_low_res",
            "character": "image_high_res",
            "video": "video_fast"
        }
        
        # Try direct match first, then legacy mapping
        model_key = functional_key if functional_key in cls.MODELS else legacy_mapping.get(functional_key)
        
        if not model_key:
            return None
            
        return cls.MODELS.get(model_key)
    
    @classmethod
    def list_available_options(cls) -> Dict[str, List[str]]:
        """Return available media types and quality levels for frontend"""
        options = {
            "image": [],
            "video": []
        }
        
        for model_key, config in cls.MODELS.items():
            media_type = config["media_type"]
            quality = config["quality"]
            
            if quality not in options[media_type]:
                options[media_type].append(quality)
        
        return options

class EnhancedVideoWorker:
    """Multi-model video generation worker"""
    
    def __init__(self):
        self.model_config = ModelConfig()
        self.current_model = None
        self.current_model_type = None
        
        # Environment validation
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        print("🚀 Enhanced OurVidz Worker with Multi-Model Support")
        self.validate_models()
        self.log_gpu_memory()

    def validate_models(self):
        """Check which models are available"""
        print("\n📋 Model Availability Check:")
        
        for model_name, config in self.model_config.MODELS.items():
            model_path = Path(config["path"])
            status = "✅ Available" if model_path.exists() else "❌ Missing"
            print(f"  {model_name:15} -> {status} ({config['purpose']})")
        
        # Check if core Wan 2.1 repo is available
        wan_repo = Path("/workspace/Wan2.1")
        if wan_repo.exists():
            print("  📁 Wan2.1 repo   -> ✅ Available")
        else:
            print("  📁 Wan2.1 repo   -> ❌ Missing (will clone)")

    def log_gpu_memory(self):
        """Monitor RTX 4090 VRAM usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🔥 GPU Memory: {memory_allocated:.1f}GB allocated / {total_memory:.0f}GB total")
        else:
            print("❌ No GPU detected")

    def ensure_model_downloaded(self, model_name: str) -> bool:
        """Download model if not present"""
        config = self.model_config.MODELS.get(model_name)
        if not config:
            return False
            
        model_path = Path(config["path"])
        
        if model_path.exists():
            print(f"✅ Model {model_name} already available")
            return True
            
        print(f"📥 Downloading {model_name} from {config['repo']}...")
        
        try:
            # Use huggingface_hub to download if available
            from huggingface_hub import snapshot_download
            
            snapshot_download(
                repo_id=config["repo"],
                local_dir=config["path"],
                token=os.getenv('HF_TOKEN')  # Optional HF token
            )
            
            print(f"✅ {model_name} downloaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            return False

    def load_model(self, model_name: str):
        """Load specific model with memory management"""
        if self.current_model_type == model_name:
            print(f"♻️ Model {model_name} already loaded")
            return True
            
        # Unload current model
        self.unload_current_model()
        
        config = self.model_config.MODELS.get(model_name)
        if not config:
            raise ValueError(f"Unknown model: {model_name}")
            
        print(f"🔄 Loading {model_name} ({config['purpose']})...")
        
        # Ensure model is downloaded
        if not self.ensure_model_downloaded(model_name):
            raise Exception(f"Failed to ensure model {model_name} is available")
        
        # For now, we'll use the existing Wan 2.1 CLI approach
        # Future: Load models directly into memory for better performance
        self.current_model_type = model_name
        
        print(f"✅ {model_name} loaded (VRAM: ~{config['vram_gb']}GB)")
        self.log_gpu_memory()
        return True

    def unload_current_model(self):
        """Free current model memory"""
        if self.current_model:
            print(f"🗑️ Unloading {self.current_model_type}...")
            del self.current_model
            self.current_model = None
            
        self.current_model_type = None
        torch.cuda.empty_cache()
        print("✅ Model memory freed")

    def generate_content(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate content with specified model"""
        
        # Load the appropriate model
        self.load_model(model_name)
        
        config = self.model_config.MODELS[model_name]
        
        # Build generation command
        cmd = self._build_generation_command(config, prompt, **kwargs)
        
        print(f"⚡ Generating with {model_name} (ETA: {config['time_estimate']})")
        start_time = time.time()
        
        try:
            # Execute generation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd="/workspace/Wan2.1"
            )
            
            generation_time = time.time() - start_time
            print(f"✅ Generation completed in {generation_time:.1f}s")
            
            # Find output file (implementation depends on Wan 2.1 output structure)
            output_path = self._find_output_file(config['purpose'])
            
            if output_path and Path(output_path).exists():
                return output_path
            else:
                raise Exception("Generated file not found")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Generation failed: {e.stderr}")
            raise Exception(f"Generation failed: {e.stderr}")

    def _build_generation_command(self, config: Dict, prompt: str, **kwargs) -> List[str]:
        """Build command for Wan 2.1 generation"""
        
        # Base command structure (adjust based on actual Wan 2.1 CLI)
        cmd = [
            "python", "generate.py",
            "--task", config["task"],
            "--size", config["size"], 
            "--ckpt_dir", config["path"],
            "--prompt", prompt,
            "--steps", str(config.get("steps", 25))
        ]
        
        # Add model-specific parameters
        if config["task"] == "lightning-t5":
            cmd.extend(["--fast_mode", "true"])
        elif config["task"] == "t2v-1.3B":
            cmd.extend(["--num_frames", "80"])  # 5 seconds at 16fps
        elif config["task"] == "t2v-14B":
            cmd.extend(["--num_frames", "80", "--quality", "high"])
            
        return cmd

    def _find_output_file(self, purpose: str) -> Optional[str]:
        """Find generated output file based on purpose"""
        output_dir = Path("/workspace/Wan2.1/outputs")
        
        # Look for most recent file matching the purpose
        patterns = {
            "instant_preview": "*.png",
            "character_image": "*.png", 
            "fast_video": "*.mp4",
            "premium_video": "*.mp4"
        }
        
        pattern = patterns.get(purpose, "*")
        files = list(output_dir.glob(pattern))
        
        if files:
            # Return most recent file
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            return str(latest_file)
            
        return None

    def process_job(self, job_data: Dict):
        """Enhanced job processing with functional model selection"""
        job_id = job_data['jobId']
        
        # Extract functional parameters
        media_type = job_data.get('mediaType', 'video')  # 'image' or 'video'
        quality = job_data.get('quality', 'fast')        # 'low_res', 'high_res', 'fast', 'premium'
        prompt = job_data['prompt']
        
        print(f"\n🔄 Processing job {job_id}")
        print(f"📝 Request: {media_type}_{quality}")
        print(f"✏️ Prompt: {prompt[:100]}...")
        
        try:
            # Get model configuration for functional request
            model_config = self.model_config.get_model_for_job(media_type, quality)
            
            if not model_config:
                raise ValueError(f"No model configured for {media_type}_{quality}")
            
            # Find model name from config
            model_name = None
            for name, config in self.model_config.MODELS.items():
                if config == model_config:
                    model_name = name
                    break
            
            print(f"🎯 Using model: {model_name} ({model_config['time_estimate']})")
            
            # Generate content
            output_path = self.generate_content(model_name, prompt)
            
            # Upload to appropriate bucket based on media type
            bucket_mapping = {
                "image": "scene-previews",  # All images go here regardless of quality
                "video": "videos-final"     # All videos go here regardless of quality
            }
            
            bucket = bucket_mapping.get(media_type, "videos-final")
            file_ext = "png" if media_type == "image" else "mp4"
            filename = f"{job_data['videoId']}_{media_type}_{quality}.{file_ext}"
            user_id = job_data.get('user_id', job_data.get('userId', 'unknown'))
            
            upload_url = self.upload_to_supabase(output_path, f"{bucket}/{user_id}/{filename}")
            
            print(f"✅ Job {job_id} completed successfully")
            print(f"📤 Uploaded: {media_type}_{quality} -> {upload_url}")
            
            # Notify completion
            self.notify_completion(job_id, 'completed', upload_url)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Job {job_id} failed: {error_msg}")
            self.notify_completion(job_id, 'failed', error_message=error_msg)

    def upload_to_supabase(self, file_path: str, storage_path: str) -> str:
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
                return f"{self.supabase_url}/storage/v1/object/public/{storage_path}"
            else:
                raise Exception(f"Upload failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Supabase upload error: {e}")
            raise

    def notify_completion(self, job_id: str, status: str, output_url: str = None, error_message: str = None):
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

    def run(self):
        """Main worker loop with enhanced model support"""
        print("\n🎬 Enhanced OurVidz GPU Worker Started!")
        print("⏳ Waiting for jobs...")
        
        while True:
            try:
                # Poll Redis queue (existing logic)
                response = requests.get(
                    f"{os.getenv('UPSTASH_REDIS_REST_URL')}/brpop/job-queue/5",
                    headers={
                        'Authorization': f"Bearer {os.getenv('UPSTASH_REDIS_REST_TOKEN')}"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('result'):
                        queue_name, job_json = result['result']
                        job_data = json.loads(job_json)
                        self.process_job(job_data)
                    else:
                        print("💤 No jobs, waiting...")
                else:
                    print(f"⚠️ Redis connection issue: {response.status_code}")
                    time.sleep(30)
                    
            except Exception as e:
                print(f"❌ Worker error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    # Environment variable validation
    required_vars = [
        'UPSTASH_REDIS_REST_URL',
        'UPSTASH_REDIS_REST_TOKEN', 
        'SUPABASE_URL',
        'SUPABASE_SERVICE_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    worker = EnhancedVideoWorker()
    worker.run()
