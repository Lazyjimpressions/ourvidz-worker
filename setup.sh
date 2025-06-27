#!/bin/bash
# setup.sh - Comprehensive OurVidz Worker Setup
# Place this file in your GitHub repository: Lazyjimpressions/ourvidz-worker

set -e
echo "🚀 OurVidz Worker Detailed Setup..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt || {
    echo "⚠️ Some dependencies failed, installing critical ones individually..."
    pip install opencv-python pillow requests
}

# Setup Wan2.1 if needed
cd /workspace
if [ ! -d 'Wan2.1' ]; then
    echo "📥 Installing Wan2.1..."
    git clone https://github.com/Wan-Video/Wan2.1.git
    cd Wan2.1 && pip install -e .
elif [ ! -f 'Wan2.1/setup.py' ] && [ ! -f 'Wan2.1/pyproject.toml' ]; then
    echo "🔧 Reinstalling Wan2.1 package..."
    cd Wan2.1 && pip install -e .
fi

# Create optimized temp directories
echo "📁 Creating temp directories..."
mkdir -p /tmp/ourvidz/{models,outputs,processing}

# Quick dependency check
echo "🔍 Quick verification:"
python -c "import cv2, PIL, requests; print('✅ Core packages OK')" || echo "⚠️ Some packages missing"
ffmpeg -version | head -1 || echo "⚠️ FFmpeg not available"

# Return to worker directory
cd /workspace/ourvidz-worker
echo "✅ Setup complete!"
