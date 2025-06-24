#!/bin/bash
# Enhanced RunPod Startup Command for Multi-Model Setup

set -e
cd /workspace

echo "🚀 Starting OurVidz Enhanced Multi-Model Setup"
echo "=============================================="

# Clean up any existing worker
rm -rf ourvidz-worker

# Clone latest worker code
echo "📥 Cloning worker repository..."
git clone https://github.com/Lazyjimpressions/ourvidz-worker.git
cd ourvidz-worker

# Install/upgrade Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ pip install completed"

# Install/upgrade Wan 2.1 framework
echo "🎥 Setting up Wan 2.1 framework..."
cd /workspace

if [ ! -d "Wan2.1" ]; then
    echo "📥 Cloning Wan 2.1 repository..."
    git clone https://github.com/Wan-Video/Wan2.1.git
    cd Wan2.1
    pip install -e .
    echo "✅ Wan 2.1 installed"
else
    echo "🔄 Updating existing Wan 2.1..."
    cd Wan2.1
    git pull origin main
    pip install -e . --upgrade
    echo "✅ Wan 2.1 updated"
fi

# Download additional models (Lightning-T5 and Base-Diffusion)
echo "📥 Setting up enhanced model suite..."
cd /workspace/ourvidz-worker

# Check if we need to download new models
python -c "
import os
from pathlib import Path

models_to_check = [
    '/workspace/models/wan_lightning',
    '/workspace/models/wan_base'
]

need_download = False
for model_path in models_to_check:
    if not Path(model_path).exists():
        need_download = True
        break

if need_download:
    print('🎯 New models needed - running download script')
    exit(1)
else:
    print('✅ All models already available')
    exit(0)
"

# Download models if needed
if [ $? -eq 1 ]; then
    echo "📦 Downloading Lightning-T5 and Base-Diffusion models..."
    python download_all_models.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Model download completed"
    else
        echo "⚠️ Model download had issues, but continuing..."
    fi
else
    echo "✅ All models already available"
fi

# Verify system status
echo "🔧 System verification..."
echo "📊 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

echo "📊 Disk Usage:"
df -h /workspace

echo "📋 Model Inventory:" 
ls -la /workspace/models/ || echo "Models directory not found"

# Start the enhanced worker
echo "🎬 Starting enhanced video generation worker..."
cd /workspace/ourvidz-worker
exec python -u ourvidz_enhanced_worker.py
