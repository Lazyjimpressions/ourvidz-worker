#!/bin/bash
# setup.sh - Conservative Version Setup for Stability
set -e

echo "🚀 OURVIDZ DUAL WORKER SETUP"
echo "🔥 RTX 6000 ADA - Conservative Version Strategy"

# Check current PyTorch version
CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
echo "Current PyTorch: $CURRENT_TORCH"

# Clean conflicting packages if wrong version
if [[ "$CURRENT_TORCH" != "2.4.1+cu124" && "$CURRENT_TORCH" != "none" ]]; then
    echo "🧹 Cleaning conflicting versions..."
    pip uninstall -y torch torchvision torchaudio diffusers xformers transformers accelerate flash-attn || true
    pip cache purge || true
fi

# Install base PyTorch first (this is the foundation)
if [[ "$CURRENT_TORCH" == "2.4.1+cu124" ]]; then
    echo "✅ PyTorch 2.4.1+cu124 already installed, skipping"
else
    echo "📦 Installing PyTorch 2.4.1+cu124..."
    pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 \
        --index-url https://download.pytorch.org/whl/cu124 --no-deps
fi

echo "📚 Installing core support libraries first..."
pip install numpy==1.26.4 pillow==10.4.0 requests==2.32.3 --no-deps

# Install diffusers ecosystem with known stable versions
echo "🎨 Installing diffusers ecosystem (stable versions)..."
pip install diffusers==0.29.2 transformers==4.42.4 \
    safetensors==0.4.3 huggingface-hub==0.23.4 --no-deps

# Install accelerate separately
echo "⚡ Installing accelerate..."
pip install accelerate==0.32.1 --no-deps

# Try xformers (if it fails, continue without it)
echo "🔧 Installing xformers (may fail, will continue)..."
pip install xformers==0.0.27.post2 --no-deps || echo "⚠️ xformers failed, continuing without it"

# Skip flash attention for now to avoid conflicts
echo "⚠️ Skipping flash attention to avoid conflicts"

echo "📚 Installing remaining support libraries..."
pip install opencv-python==4.10.0.84 scipy==1.13.1 --no-deps

pip install tokenizers==0.19.1 sentencepiece==0.2.0 tqdm==4.66.5 \
    pydantic==2.8.2 einops==0.8.0 psutil==6.0.0 --no-deps

pip install imageio==2.34.2 imageio-ffmpeg==0.5.1 \
    click==8.1.7 omegaconf==2.3.0 --no-deps

# Setup Wan 2.1
echo "🎥 Setting up Wan 2.1..."
cd /workspace
if [ ! -d "Wan2.1" ]; then
    git clone https://github.com/Wan-Video/Wan2.1.git
fi
cd Wan2.1
pip install -e . --no-deps --no-build-isolation

# Download SDXL model if needed
echo "🎨 Checking SDXL model..."
SDXL_MODEL="/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors"
if [ ! -f "$SDXL_MODEL" ]; then
    echo "📥 Downloading LUSTIFY SDXL (6.9GB)..."
    mkdir -p /workspace/models/sdxl-lustify
    cd /workspace/models/sdxl-lustify
    wget -c -O lustifySDXLNSFWSFW_v20.safetensors \
        "https://huggingface.co/John6666/lustify-sdxl-nsfw-checkpoint-olt-one-last-time-sdxl/resolve/main/lustifySDXLNSFWSFW_v20.safetensors"
fi

# Validate environment
echo "🔧 Validating setup..."
cd /workspace/ourvidz-worker
python -c "
import torch; print(f'PyTorch: {torch.__version__}')
import diffusers; print(f'Diffusers: {diffusers.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f}GB')

try:
    import xformers
    print(f'xformers: {xformers.__version__}')
except:
    print('xformers: Not available')

try:
    import accelerate
    print(f'accelerate: {accelerate.__version__}')
except:
    print('accelerate: Not available')
"

echo "🚀 Starting dual orchestrator..."
exec python -u dual_orchestrator.py
