# download_models.py - Fixed model names and ungated alternatives
import os
import torch
from diffusers import WanVideoPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_models():
    """Download and cache working models"""
    model_path = "/workspace/models"
    os.makedirs(model_path, exist_ok=True)
    
    print("🎥 Downloading Wan 2.1 14B Text-to-Video...")
    try:
        # Updated model name (correct spelling)
        wan_t2v_pipeline = WanVideoPipeline.from_pretrained(
            "Wan-AI/Wan2.1-T2V-14B",
            torch_dtype=torch.float16,
            cache_dir=f"{model_path}/wan_t2v"
        )
        print("✅ Wan 2.1 T2V downloaded successfully")
    except Exception as e:
        print(f"⚠️ Wan 2.1 T2V download failed: {e}")
        print("📝 Will use fallback approach during generation")
    
    # Try the Image-to-Video model for Phase 2
    print("🖼️ Downloading Wan 2.1 14B Image-to-Video...")
    try:
        wan_i2v_pipeline = WanVideoPipeline.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-720P",
            torch_dtype=torch.float16,
            cache_dir=f"{model_path}/wan_i2v"
        )
        print("✅ Wan 2.1 I2V downloaded successfully")
    except Exception as e:
        print(f"⚠️ Wan 2.1 I2V download failed: {e}")
        print("📝 Phase 2 feature, will implement later")
    
    # Use ungated Mistral model
    print("📝 Downloading Mistral 7B (ungated version)...")
    try:
        mistral_tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1",  # Ungated base model
            cache_dir=f"{model_path}/mistral"
        )
        mistral_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",  # Ungated base model
            torch_dtype=torch.float16,
            cache_dir=f"{model_path}/mistral"
        )
        print("✅ Mistral 7B downloaded successfully")
    except Exception as e:
        print(f"⚠️ Mistral 7B download failed: {e}")
        
        # Try alternative uncensored model
        print("📝 Trying alternative uncensored model...")
        try:
            mistral_tokenizer = AutoTokenizer.from_pretrained(
                "ehartford/dolphin-2.0-mistral-7b",
                cache_dir=f"{model_path}/mistral"
            )
            mistral_model = AutoModelForCausalLM.from_pretrained(
                "ehartford/dolphin-2.0-mistral-7b",
                torch_dtype=torch.float16,
                cache_dir=f"{model_path}/mistral"
            )
            print("✅ Dolphin Mistral 7B downloaded successfully")
        except Exception as e2:
            print(f"❌ All Mistral models failed: {e2}")
    
    print("🎉 Model download process completed!")
    print(f"📁 Models stored in: {model_path}")

if __name__ == "__main__":
    download_models()
