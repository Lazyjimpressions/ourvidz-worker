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
qwen_instruct_ready=$(python3 -c "
import os
paths = [
    '/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct',
    '/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B-Instruct'
]
found = any(os.path.exists(p) for p in paths)
print('true' if found else 'false')
")

if [ "$qwen_instruct_ready" = "true" ]; then
    echo "✅ Qwen Instruct model ready"
else
    echo "⚠️ Qwen Instruct model missing"
fi

# Chat Integration Status
if [ "$qwen_instruct_ready" = true ]; then
    echo "🤖 Chat Worker: ✅ READY (Qwen Instruct available)"
else
    echo "🤖 Chat Worker: ⚠️ PARTIAL (Qwen Instruct missing)"
fi

# Enhanced Job Status  
if [ "$qwen_base_ready" = true ]; then
    echo "✨ Enhanced Jobs: ✅ READY (Qwen Base available)"
else
    echo "✨ Enhanced Jobs: ⚠️ PARTIAL (Qwen Base missing)"
fi

echo "=== AUTO-REGISTRATION: Checking RunPod environment ==="
if [ -n "$RUNPOD_POD_ID" ]; then
    # Detect all worker URLs
    sdxl_url="https://${RUNPOD_POD_ID}-7859.proxy.runpod.net"
    wan_url="https://${RUNPOD_POD_ID}-7860.proxy.runpod.net"
    chat_url="https://${RUNPOD_POD_ID}-7861.proxy.runpod.net"
    
    echo "✅ RunPod Pod ID detected: $RUNPOD_POD_ID"
    echo "🌐 Worker URLs will be:"
    echo "  🎨 SDXL Worker: $sdxl_url"
    echo "  🎬 WAN Worker: $wan_url"
    echo "  🤖 Chat Worker: $chat_url"
    echo "🔧 Auto-registration will be handled by workers after startup"
else
    echo "⚠️ RUNPOD_POD_ID not found - will try hostname fallback"
    hostname=$(hostname)
    echo "🔍 Hostname: $hostname"
    if [[ "$hostname" =~ ^[a-z0-9]+-[a-z0-9]+ ]]; then
        pod_id=$(echo "$hostname" | cut -d'-' -f1)
        sdxl_url="https://${pod_id}-7859.proxy.runpod.net"
        wan_url="https://${pod_id}-7860.proxy.runpod.net"
        chat_url="https://${pod_id}-7861.proxy.runpod.net"
        
        echo "✅ Pod ID from hostname: $pod_id"
        echo "🌐 Worker URLs will be:"
        echo "  🎨 SDXL Worker: $sdxl_url"
        echo "  🎬 WAN Worker: $wan_url"
        echo "  🤖 Chat Worker: $chat_url"
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

# Memory management environment variables - CRITICAL: Set before any PyTorch imports
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
echo "🧠 Memory management: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "🧠 Memory management: CUDA_VISIBLE_DEVICES=0"

echo "=== Verifying dependencies ==="
python << 'EOF'
import sys
sys.path.insert(0, "/workspace/python_deps/lib/python3.11/site-packages")
try:
    import transformers, torch, diffusers, compel, flask
    print("Dependencies OK:")
    print("  transformers: " + transformers.__version__)
    print("  torch: " + torch.__version__)
    print("  flask: " + flask.__version__)
    
    # Check GPU memory for triple worker system
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  CUDA: {total_vram:.0f}GB VRAM available")
        
        if total_vram >= 45:
            print("  Memory: ✅ Sufficient for all workers")
        else:
            print("  Memory: ⚠️ May need smart loading")
    
    import os
    # Check model availability
    sdxl_ok = os.path.exists("/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors")
    wan_ok = os.path.exists("/workspace/models/wan2.1-t2v-1.3b/diffusion_pytorch_model.safetensors")
    base_ok = os.path.exists("/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B")
    inst_ok = os.path.exists("/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct")
    
    workers_ready = []
    if sdxl_ok:
        workers_ready.append("SDXL")
    if wan_ok and base_ok:
        workers_ready.append("WAN")
    if inst_ok:
        workers_ready.append("Chat")
    
    print(f"  Workers Ready: {', '.join(workers_ready) if workers_ready else 'None'}")
    
except ImportError as e:
    print("Missing dependency: " + str(e))
    exit(1)
EOF

echo "=== Starting triple workers (auto-registration handled by each worker) ==="
echo "🌐 Triple Worker Auto-registration sequence:"
echo "  1. Orchestrator starts all three workers in priority order"
echo "  2. SDXL worker starts first (highest priority)"
echo "  3. Chat worker starts second (real-time responses)"
echo "  4. WAN worker starts third (batch processing)"
echo "  5. Each worker detects RunPod URL using RUNPOD_POD_ID"
echo "  6. Each worker validates URL health on their respective ports"
echo "  7. Workers register URLs with Supabase (chat worker registers for enhancement)"
echo "  8. Edge functions can now route to appropriate workers automatically"
echo "  9. Smart memory management handles model loading conflicts"
echo ""
echo "🎯 Worker Ports:"
echo "  🎨 SDXL Worker: 7859 (always loaded)"
echo "  🎬 WAN Worker: 7860 (load on demand)"  
echo "  🤖 Chat Worker: 7861 (load when possible)"
echo ""

echo "🔧 Final validation before startup..."
python << 'EOF'
import sys
sys.path.insert(0, "/workspace/python_deps/lib/python3.11/site-packages")
try:
    import torch
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🔥 GPU: RTX 6000 ADA with {total_vram:.0f}GB VRAM")
        
        # Memory allocation validation
        sdxl_mem = 10
        chat_mem = 15  
        wan_mem = 30
        total_needed = sdxl_mem + chat_mem + wan_mem  # 55GB max scenario
        
        if total_vram >= total_needed:
            print(f"✅ Memory validation: {total_needed}GB max needed, {total_vram:.0f}GB available")
        else:
            print(f"⚠️ Memory warning: {total_needed}GB max needed, {total_vram:.0f}GB available - smart loading required")
        
        # Worker memory allocation
        print(f"📊 Worker Memory Allocation:")
        print(f"  🎨 SDXL Worker: {sdxl_mem}GB (always loaded)")
        print(f"  🤖 Chat Worker: {chat_mem}GB (load when possible)")
        print(f"  🎬 WAN Worker: {wan_mem}GB (load on demand)")
        print(f"  🔄 Smart Management: Unload chat for WAN jobs if needed")
        
    # Check all worker scripts exist
    import os
    worker_scripts = [
        'dual_orchestrator.py',
        'sdxl_worker.py', 
        'wan_worker.py',
        'chat_worker.py'
    ]
    
    missing_scripts = []
    for script in worker_scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"❌ Missing worker scripts: {missing_scripts}")
        exit(1)
    else:
        print(f"✅ All worker scripts present")
        
    print(f"✅ Triple worker system validation complete")
    
except Exception as e:
    print(f"❌ Validation failed: {e}")
    exit(1)
EOF

echo ""
echo "📊 Temporary storage analysis:"
df -h /tmp
echo "💾 Available for caching: $(df -h /tmp | awk 'NR==2{print $4}')"

# Smart cache allocation based on available temp space
echo "🔧 Smart cache allocation strategy:"
TEMP_AVAILABLE=$(df /tmp --output=avail | tail -1)
TEMP_AVAILABLE_GB=$((TEMP_AVAILABLE / 1024 / 1024))

if [ "$TEMP_AVAILABLE" -gt 20000000 ]; then  # >20GB
    CACHE_SIZE="8GB"
    CACHE_STRATEGY="aggressive"
    echo "✅ High temp storage available: ${TEMP_AVAILABLE_GB}GB"
    echo "📈 Cache strategy: ${CACHE_STRATEGY} (${CACHE_SIZE})"
else
    CACHE_SIZE="4GB"
    CACHE_STRATEGY="conservative"
    echo "⚠️ Limited temp storage: ${TEMP_AVAILABLE_GB}GB"
    echo "📉 Cache strategy: ${CACHE_STRATEGY} (${CACHE_SIZE})"
fi

# Set cache environment variables for workers
export TEMP_CACHE_SIZE="$CACHE_SIZE"
export TEMP_CACHE_STRATEGY="$CACHE_STRATEGY"
export TEMP_AVAILABLE_GB="$TEMP_AVAILABLE_GB"

echo "🔧 Cache configuration exported:"
echo "  TEMP_CACHE_SIZE=$TEMP_CACHE_SIZE"
echo "  TEMP_CACHE_STRATEGY=$TEMP_CACHE_STRATEGY"
echo "  TEMP_AVAILABLE_GB=${TEMP_AVAILABLE_GB}GB"

echo ""
echo "🚀 LAUNCHING TRIPLE WORKER SYSTEM"
echo "=================================="
echo "⚡ Expected startup sequence:"
echo "1. Orchestrator validates environment and starts workers"
echo "2. SDXL Worker (7859) - Loads immediately, always ready"
echo "3. Chat Worker (7861) - Loads Qwen Instruct when memory allows"  
echo "4. WAN Worker (7860) - Loads models on-demand for jobs"
echo "5. All workers register URLs and start heartbeat monitoring"
echo "6. Edge functions route to appropriate workers automatically"
echo ""
echo "🎯 Performance Targets:"
echo "  🎨 SDXL: 3-8 seconds per image"
echo "  🤖 Chat: <3 seconds per enhancement"
echo "  🎬 WAN: 67-294 seconds per video"
echo ""
echo "🧠 Memory Management:"
echo "  📊 Smart allocation based on priority and availability"
echo "  🔄 Dynamic loading/unloading as needed"
echo "  ⚠️ Chat worker may temporarily unload for WAN jobs"
echo ""
echo "💾 Cache Strategy:"
echo "  📈 Strategy: $CACHE_STRATEGY"
echo "  💾 Cache Size: $CACHE_SIZE"
echo "  📊 Temp Available: ${TEMP_AVAILABLE_GB}GB"
echo "  🔄 Workers will use cache for model loading and temporary files"
echo ""

exec python -u dual_orchestrator.py