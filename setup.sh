#!/bin/bash
# setup.sh - Version-Locked Dual Worker Setup
set -e

echo "🚀 OURVIDZ DUAL WORKER SETUP"
echo "🔥 RTX 6000 ADA - Version Locked Dependencies"

# Check current PyTorch version
CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
echo "Current PyTorch: $CURRENT_TORCH"

# Clean conflicting packages if wrong version
if [[ "$CURRENT_TORCH" != "2.4.1+cu124" && "$CURRENT_TORCH" != "none" ]]; then
    echo "🧹 Cleaning conflicting versions..."
    pip uninstall -y torch torchvision torchaudio diffusers xformers transformers accelerate flash-attn || true
    pip cache purge || true
fi

# Install in locked order
echo "📦 Installing PyTorch 2.4.1+cu124..."
pip install --no-cache-dir torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 --no-deps

echo "⚡ Installing Flash Attention..."
pip install flash-attn==2.8.0.post2 --no-build-isolation --no-deps

echo "🔧 Installing xformers..."
pip install xformers==0.0.28.post2 --no-deps

echo "🎨 Installing diffusers ecosystem..."
pip install diffusers==0.31.0 transformers==4.45.2 accelerate==0.35.0 \
    safetensors==0.4.5 huggingface-hub==0.26.2 --no-deps

echo "📚 Installing support libraries..."
pip install numpy==1.26.4 pillow==10.4.0 opencv-python==4.10.0.84 \
    requests==2.32.3 scipy==1.14.1 --no-deps

pip install tokenizers==0.20.3 sentencepiece==0.2.0 tqdm==4.66.5 \
    pydantic==2.9.2 einops==0.8.0 psutil==6.0.0 --no-deps

pip install imageio==2.36.0 imageio-ffmpeg==0.5.1 \
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
"

echo "🚀 Starting dual orchestrator..."
exec python -u dual_orchestrator.py
