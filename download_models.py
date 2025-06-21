#!/usr/bin/env python3
# download_models.py - Download Wan 2.1 14B models to network volume

import os
import torch
from pathlib import Path

def check_and_download_models():
    """Download models to network volume if not already present"""
    model_path = Path("/workspace/models")
    model_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Checking models in: {model_path}")
    
    # Check if models already exist
    wan_t2v_path = model_path / "wan_t2v"
    wan_i2v_path = model_path / "wan_i2v"
    mistral_path = model_path / "mistral"
    
    models_exist = (
        wan_t2v_path.exists() and 
        wan_i2v_path.exists() and 
        mistral_path.exists() and
        len(list(wan_t2v_path.glob("*"))) > 5 and
        len(list(mistral_path.glob("*"))) > 5
    )
    
    if models_exist:
        print("✅ Models already downloaded to network volume!")
        return
    
    print("📥 Downloading models to network volume (this will take ~45 minutes)...")
    
    try:
        # Import after checking if models exist to save time
        from diffusers import DiffusionPipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Download Wan 2.1 14B Text-to-Video
        print("🎥 Downloading Wan 2.1 14B Text-to-Video...")
        try:
            wan_t2v_pipeline = DiffusionPipeline.from_pretrained(
                "Wan-AI/Wan2.1-T2V-14B",
                torch_dtype=torch.float16,
                cache_dir=str(wan_t2v_path),
                local_files_only=False
            )
            print("✅ Wan 2.1 T2V downloaded")
            del wan_t2v_pipeline
        except Exception as e:
            print(f"⚠️ Wan 2.1 T2V download failed: {e}")
            print("📝 Will use fallback model during generation")
        
        # Download Wan 2.1 14B Image-to-Video (Phase 2)
        print("🖼️ Downloading Wan 2.1 14B Image-to-Video...")
        try:
            wan_i2v_pipeline = DiffusionPipeline.from_pretrained(
                "Wan-AI/Wan2.1-I2V-14B-720P", 
                torch_dtype=torch.float16,
                cache_dir=str(wan_i2v_path),
                local_files_only=False
            )
            print("✅ Wan 2.1 I2V downloaded")
            del wan_i2v_pipeline
        except Exception as e:
            print(f"⚠️ Wan 2.1 I2V download failed: {e}")
        
        # Download Mistral 7B for prompt enhancement
        print("📝 Downloading Mistral 7B...")
        try:
            mistral_tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                cache_dir=str(mistral_path)
            )
            mistral_model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                torch_dtype=torch.float16,
                cache_dir=str(mistral_path)
            )
            print("✅ Mistral 7B downloaded")
            del mistral_model, mistral_tokenizer
        except Exception as e:
            print(f"⚠️ Mistral 7B download failed: {e}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🎉 Model download completed!")
        
        # Create completion marker
        (model_path / "download_complete.txt").write_text(
            f"Models downloaded successfully at {os.path.basename(__file__)}"
        )
        
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        raise

if __name__ == "__main__":
    check_and_download_models()
