# download_models.py - Fixed to handle missing WanVideoPipeline
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import video generation models
try:
    from diffusers import WanVideoPipeline
    HAS_WAN = True
    print("✅ WanVideoPipeline available")
except ImportError:
    print("⚠️ WanVideoPipeline not available in current diffusers version")
    HAS_WAN = False

def download_models():
    """Download and cache working models"""
    model_path = "/workspace/models"
    os.makedirs(model_path, exist_ok=True)
    
    # Only try Wan models if available
    if HAS_WAN:
        print("🎥 Downloading Wan 2.1 14B Text-to-Video...")
        try:
            wan_t2v_pipeline = WanVideoPipeline.from_pretrained(
                "Wan-AI/Wan2.1-T2V-14B",
                torch_dtype=torch.float16,
                cache_dir=f"{model_path}/wan_t2v"
            )
            print("✅ Wan 2.1 T2V downloaded successfully")
            del wan_t2v_pipeline
        except Exception as e:
            print(f"⚠️ Wan 2.1 T2V download failed: {e}")
    else:
        print("⏭️ Skipping Wan model downloads (not available)")
        print("📝 Will use placeholder generation for now")

    # Download Mistral models (ungated versions)
    print("📝 Downloading Mistral 7B...")
    mistral_success = False
    
    # Try ungated Mistral 7B v0.1 first
    try:
        print("📥 Downloading Mistral 7B v0.1 (ungated)...")
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            cache_dir=f"{model_path}/mistral"
        )
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            torch_dtype=torch.float16,
            cache_dir=f"{model_path}/mistral"
        )
        print("✅ Mistral 7B v0.1 (ungated) downloaded successfully")
        mistral_success = True
        del tokenizer, model
    except Exception as e:
        print(f"⚠️ Mistral 7B v0.1 download failed: {e}")
        
        # Try uncensored alternative
        try:
            print("📥 Downloading Dolphin Mistral 7B (uncensored)...")
            tokenizer = AutoTokenizer.from_pretrained(
                "ehartford/dolphin-2.0-mistral-7b",
                cache_dir=f"{model_path}/mistral_alt"
            )
            model = AutoModelForCausalLM.from_pretrained(
                "ehartford/dolphin-2.0-mistral-7b",
                torch_dtype=torch.float16,
                cache_dir=f"{model_path}/mistral_alt"
            )
            print("✅ Dolphin Mistral 7B (uncensored) downloaded successfully")
            mistral_success = True
            del tokenizer, model
        except Exception as e2:
            print(f"⚠️ Dolphin Mistral download failed: {e2}")

    if mistral_success:
        print("✅ At least one Mistral model downloaded successfully")
    else:
        print("❌ All Mistral model downloads failed")
    
    # Create completion marker
    with open(f"{model_path}/download_complete.txt", "w") as f:
        f.write(f"Download completed at: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"HAS_WAN: {HAS_WAN}\n")
        f.write(f"Mistral Success: {mistral_success}\n")
        
    print("🎉 Model download process completed!")
    print(f"📁 Models stored in: {model_path}")
    
    if not HAS_WAN:
        print("💡 Note: Video generation will use placeholder mode")
        print("💡 To get Wan 2.1 working, we need to install from source")

if __name__ == "__main__":
    download_models()
