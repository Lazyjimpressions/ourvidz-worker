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

echo "=== MODELS: Checking triple worker model status ==="
sdxl_ready=false
wan_ready=false
qwen_base_ready=false
qwen_instruct_ready=false

[ -f "/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors" ] && sdxl_ready=true && echo "✅ SDXL model ready"
[ -f "/workspace/models/wan2.1-t2v-1.3b/diffusion_pytorch_model.safetensors" ] && wan_ready=true && echo "✅ WAN model ready"
[ -d "/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B" ] && qwen_base_ready=true && echo "✅ Qwen Base model ready"
[ -d "/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct" ] && qwen_instruct_ready=true && echo "✅ Qwen Instruct model ready"

# Triple Worker Status Assessment
echo ""
echo "=== TRIPLE WORKER STATUS ==="
if [ "$sdxl_ready" = true ]; then
    echo "🎨 SDXL Worker: ✅ READY (Port 7860, Priority 1)"
else
    echo "🎨 SDXL Worker: ❌ NOT READY (missing model)"
fi

if [ "$qwen_instruct_ready" = true ]; then
    echo "💬 Chat Worker: ✅ READY (Port 7861, Priority 2)"
else
    echo "💬 Chat Worker: ❌ NOT READY (missing Qwen Instruct model)"
fi

if [ "$wan_ready" = true ] && [ "$qwen_base_ready" = true ]; then
    echo "🎬 WAN Worker: ✅ READY (Port 7860, Priority 3)"
else
    echo "🎬 WAN Worker: ⚠️ PARTIAL (missing WAN or Qwen Base model)"
fi

# Overall System Status
ready_workers=0
[ "$sdxl_ready" = true ] && ready_workers=$((ready_workers + 1))
[ "$qwen_instruct_ready" = true ] && ready_workers=$((ready_workers + 1))
[ "$wan_ready" = true ] && [ "$qwen_base_ready" = true ] && ready_workers=$((ready_workers + 1))

echo ""
if [ $ready_workers -eq 3 ]; then
    echo "🎉 TRIPLE WORKER SYSTEM: ✅ FULLY READY"
elif [ $ready_workers -eq 2 ]; then
    echo "🎯 TRIPLE WORKER SYSTEM: ⚠️ PARTIALLY READY ($ready_workers/3 workers)"
elif [ $ready_workers -eq 1 ]; then
    echo "⚠️ TRIPLE WORKER SYSTEM: LIMITED ($ready_workers/3 workers)"
else
    echo "❌ TRIPLE WORKER SYSTEM: NOT READY (0/3 workers)"
fi

echo "=== AUTO-REGISTRATION: Checking RunPod environment ==="
if [ -n "$RUNPOD_POD_ID" ]; then
    detected_url="https://${RUNPOD_POD_ID}-7860.proxy.runpod.net"
    chat_url="https://${RUNPOD_POD_ID}-7861.proxy.runpod.net"
    echo "✅ RunPod Pod ID detected: $RUNPOD_POD_ID"
    echo "🌐 WAN/SDXL Worker URL: $detected_url"
    echo "🌐 Chat Worker URL: $chat_url"
    echo "🔧 Auto-registration will be handled by WAN worker after Flask startup"
else
    echo "⚠️ RUNPOD_POD_ID not found - will try hostname fallback"
    hostname=$(hostname)
    echo "🔍 Hostname: $hostname"
    if [[ "$hostname" =~ ^[a-z0-9]+-[a-z0-9]+ ]]; then
        pod_id=$(echo "$hostname" | cut -d'-' -f1)
        detected_url="https://${pod_id}-7860.proxy.runpod.net"
        chat_url="https://${pod_id}-7861.proxy.runpod.net"
        echo "✅ Pod ID from hostname: $pod_id"
        echo "🌐 WAN/SDXL Worker URL: $detected_url"
        echo "🌐 Chat Worker URL: $chat_url"
    else
        echo "⚠️ Could not detect Pod ID from hostname"
        echo "🔧 Auto-registration may not work - manual registration may be needed"
    fi
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
    # Check all model paths
    sdxl_ok = os.path.exists("/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors")
    wan_ok = os.path.exists("/workspace/models/wan2.1-t2v-1.3b/diffusion_pytorch_model.safetensors")
    base_ok = os.path.exists("/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B")
    inst_ok = os.path.exists("/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct")
    
    print(f"SDXL: {'READY' if sdxl_ok else 'MISSING'}")
    print(f"WAN: {'READY' if wan_ok else 'MISSING'}")
    print(f"Qwen Base: {'READY' if base_ok else 'MISSING'}")
    print(f"Qwen Instruct: {'READY' if inst_ok else 'MISSING'}")
    
    ready_count = sum([sdxl_ok, wan_ok and base_ok, inst_ok])
    print(f"Triple Worker Status: {ready_count}/3 workers ready")
    
except ImportError as e:
    print("Missing dependency: " + str(e))
    exit(1)
EOF

echo "=== Starting triple workers (auto-registration handled by WAN worker) ==="
echo "🌐 Auto-registration sequence:"
echo "  1. Orchestrator starts all three workers in priority order"
echo "  2. SDXL worker starts (Priority 1, Port 7860)"
echo "  3. Chat worker starts (Priority 2, Port 7861)"
echo "  4. WAN worker starts (Priority 3, Port 7860)"
echo "  5. WAN worker starts Flask server on port 7860"
echo "  6. WAN worker detects RunPod URL using RUNPOD_POD_ID"
echo "  7. WAN worker validates URL health"
echo "  8. WAN worker registers URL with Supabase"
echo "  9. WAN worker starts periodic health monitoring"
echo "  10. Edge functions can now find worker automatically"
echo ""
echo "🎭 Triple Worker System Architecture:"
echo "  🎨 SDXL Worker: Fast image generation (3-8s) - Port 7860"
echo "  💬 Chat Worker: Qwen Instruct service (5-15s) - Port 7861"
echo "  🎬 WAN Worker: Video + Qwen enhancement (67-294s) - Port 7860"
echo ""

exec python -u dual_orchestrator.py