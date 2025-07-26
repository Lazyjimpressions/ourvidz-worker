#!/bin/bash
set -e
cd /workspace/ourvidz-worker

echo "=== SAFETY: Verifying stable environment ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}')
if not torch.__version__.startswith('2.4.1') or torch.version.cuda != '12.4':
    print('❌ Version mismatch - ABORT!')
    exit(1)
print('✅ Versions confirmed stable')
"

echo "=== CHAT: Checking dual model status ==="
base_ready=false
instruct_ready=false
[ -d "/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B" ] && base_ready=true && echo "✅ Base model ready"
[ -d "/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct" ] && instruct_ready=true && echo "✅ Instruct model ready"

if [ "$base_ready" = true ] && [ "$instruct_ready" = true ]; then
    ln -sf models--Qwen--Qwen2.5-7B-Instruct /workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Chat 2>/dev/null || true
    echo "🎯 Chat Integration: ✅ FULLY READY"
else
    echo "🎯 Chat Integration: ⚠️ PARTIAL"
fi

echo "=== Setting environment ==="
export PYTHONPATH=/workspace/python_deps/lib/python3.11/site-packages
export HF_HOME=/workspace/models/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/workspace/models/huggingface_cache/hub
export TRANSFORMERS_CACHE=/workspace/models/huggingface_cache/hub

echo "=== Verifying dependencies ==="
python << 'EOF'
import sys
sys.path.insert(0, "/workspace/python_deps/lib/python3.11/site-packages")
try:
    import transformers, torch, diffusers, compel
    print("Dependencies OK: transformers " + transformers.__version__)
    
    import os
    base_ok = os.path.exists("/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B")
    inst_ok = os.path.exists("/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct")
    status = "READY" if base_ok and inst_ok else "PARTIAL"
    print("Chat: " + status)
    
except ImportError as e:
    print("Missing dependency: " + str(e))
    exit(1)
EOF

echo "=== Starting dual workers (auto-registration handled by WAN worker) ==="
exec python -u dual_orchestrator.py