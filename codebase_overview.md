# OurVidz Worker Codebase Overview

## 🎯 **System Overview**

Your **OurVidz Worker** is a sophisticated **GPU-accelerated AI content generation system** designed for RunPod deployment. It's a multi-model system that supports both image and video generation with different quality tiers and performance characteristics.

## 🚀 **ACTIVE PRODUCTION ARCHITECTURE**

### **Main Components**

| Component | File | Status | Purpose |
|-----------|------|--------|---------|
| 🎭 **Dual Worker Orchestrator** | `dual_orchestrator.py` | ✅ **ACTIVE** | Main controller managing both SDXL and WAN workers |
| 🎨 **SDXL Worker** | `sdxl_worker.py` | ✅ **ACTIVE** | Fast image generation using LUSTIFY SDXL model |
| 🎬 **Enhanced WAN Worker** | `wan_worker.py` | ✅ **ACTIVE** | Video/image generation with AI prompt enhancement |

### **Key Features**

- **🔄 Concurrent Processing**: SDXL (3-8s) + WAN (67-294s) workers running simultaneously
- **📊 Multiple Quality Tiers**: Fast, high, and enhanced variants
- **🤖 AI Enhancement**: Qwen 7B integration for prompt improvement
- **🖼️ Batch Generation**: 6-image batches for better UX
- **💾 Memory Optimization**: GPU memory management and model unloading
- **🛡️ Production Ready**: Comprehensive error handling and automatic restart

## 📊 **Performance Characteristics**

### **SDXL Jobs**
| Job Type | Quality | Steps | Time | Resolution | Use Case |
|----------|---------|-------|------|------------|----------|
| `sdxl_image_fast` | Fast | 15 | 3-8s | 1024x1024 | Quick preview |
| `sdxl_image_high` | High | 25 | 3-8s | 1024x1024 | Final quality |

### **WAN Jobs**
| Job Type | Quality | Steps | Frames | Time | Resolution | Enhancement |
|----------|---------|-------|--------|------|------------|-------------|
| `image_fast` | Fast | 4 | 1 | 73s | 480x832 | No |
| `image_high` | High | 6 | 1 | 90s | 480x832 | No |
| `video_fast` | Fast | 4 | 17 | 180s | 480x832 | No |
| `video_high` | High | 6 | 17 | 280s | 480x832 | No |
| `image7b_fast_enhanced` | Fast | 4 | 1 | 87s | 480x832 | Yes |
| `image7b_high_enhanced` | High | 6 | 1 | 104s | 480x832 | Yes |
| `video7b_fast_enhanced` | Fast | 4 | 17 | 194s | 480x832 | Yes |
| `video7b_high_enhanced` | High | 6 | 17 | 294s | 480x832 | Yes |

## 🔧 **Technical Stack**

### **Hardware & Runtime**
- **GPU**: Optimized for RTX 6000 ADA 48GB VRAM
- **PyTorch**: 2.4.1+cu124 with CUDA 12.4
- **Python**: 3.11 with optimized dependencies

### **AI Models**
- **WAN 2.1 T2V 1.3B**: Video generation (primary)
- **LUSTIFY SDXL v2.0**: Image generation (6.9GB model)
- **Qwen 2.5-7B**: AI prompt enhancement

### **Infrastructure**
- **Storage**: Supabase storage with proper Content-Type headers
- **Queues**: Redis queues (Upstash) with 5-second polling
- **Deployment**: RunPod with automated setup scripts

## 📁 **File Organization**

### **✅ Active Production Files**
- `dual_orchestrator.py` - **Main entry point** (production orchestrator)
- `sdxl_worker.py` - **Image generation** (batch processing)
- `wan_worker.py` - **Video generation** (AI enhancement)
- `setup.sh` - **Automated environment setup**
- `requirements.txt` - **Dependency specification**

### **❌ Legacy/Backup Files**
- `worker.py` - Previous optimized worker (replaced by enhanced WAN worker)
- `ourvidz_enhanced_worker.py` - Previous multi-model worker (replaced by dual orchestrator)
- `worker-14b.py` - Previous 14B worker (functionality integrated)
- `worker_Old_Wan_only.py` - Original WAN-only implementation
- `sdxl_worker_old.py` - Previous SDXL implementation
- `dual_orchestrator_old.py` - Previous orchestrator version
- `wan_worker_7.15.py` - Backup version (identical to current)

## 🎯 **Current Production Status**

### **✅ Active Components**
- **Dual Orchestrator**: Main production controller with monitoring
- **SDXL Worker**: Fast image generation with 6-image batch support
- **Enhanced WAN Worker**: Video generation with Qwen 7B AI enhancement

### **❌ Legacy Components** (Not Used)
- All other worker files are legacy/backup versions
- Functionality has been consolidated into the active trio

## 🚀 **Quick Start**

### **Production Deployment**
```bash
# 1. Setup environment
./setup.sh

# 2. Start production system
python dual_orchestrator.py
```

### **Individual Testing** (Development Only)
```bash
# Test SDXL only
python sdxl_worker.py

# Test WAN only  
python wan_worker.py
```

## 📋 **Environment Variables**

```bash
SUPABASE_URL=              # Supabase database URL
SUPABASE_SERVICE_KEY=      # Supabase service key
UPSTASH_REDIS_REST_URL=    # Redis queue URL
UPSTASH_REDIS_REST_TOKEN=  # Redis authentication token
HF_TOKEN=                  # Optional HuggingFace token
```

## 📚 **Additional Documentation**

The comprehensive **`CODEBASE_INDEX.md`** provides detailed documentation of all components, their relationships, configuration options, and usage instructions. This serves as a complete reference for understanding and working with your AI content generation system.

---

**This codebase represents a production-ready AI content generation system** optimized for high-performance GPU environments with comprehensive error handling and monitoring capabilities. The current architecture uses a **dual-worker orchestration pattern** for optimal resource utilization and reliability.