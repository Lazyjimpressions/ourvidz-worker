# OurVidz Worker Codebase Index

## Overview
OurVidz Worker is a GPU-accelerated AI content generation system designed for RunPod deployment. It supports multiple AI models for image and video generation with different quality tiers and performance characteristics.

## 🚀 **ACTIVE PRODUCTION ARCHITECTURE**

### 🎭 Dual Worker Orchestrator (`dual_orchestrator.py`) - **ACTIVE**
**Purpose**: Main orchestrator that manages both SDXL and WAN workers concurrently
- **Key Features**:
  - Manages two worker processes: SDXL (image) and WAN (video+image)
  - Handles graceful validation and error recovery
  - Optimized for RTX 6000 ADA 48GB VRAM
  - Automatic restart and monitoring capabilities
  - Production-ready with comprehensive logging
- **Job Types Managed**:
  - SDXL: `sdxl_image_fast`, `sdxl_image_high`
  - WAN: `image_fast`, `image_high`, `video_fast`, `video_high`, enhanced variants
- **Performance**: 3-8s (SDXL), 67-294s (WAN)
- **Status**: ✅ **ACTIVE - Production System**

### 🎨 SDXL Worker (`sdxl_worker.py`) - **ACTIVE**
**Purpose**: Fast image generation using LUSTIFY SDXL model
- **Model**: `lustifySDXLNSFWSFW_v20.safetensors`
- **Features**:
  - **Batch generation (6 images per request)** - Major UX improvement
  - Two quality tiers: fast (15 steps) and high (25 steps)
  - Optimized for speed: 3.6s per image
  - Memory-efficient with attention slicing and xformers
  - Proper PNG Content-Type headers for uploads
- **Job Types**: `sdxl_image_fast`, `sdxl_image_high`
- **Output**: PNG images, 1024x1024 resolution
- **Status**: ✅ **ACTIVE - Production System**

### 🎬 Enhanced WAN Worker (`wan_worker.py`) - **ACTIVE**
**Purpose**: Video and image generation with AI prompt enhancement
- **Models**: 
  - WAN 2.1 T2V 1.3B for video generation
  - Qwen 2.5-7B for prompt enhancement
- **Features**:
  - **AI-powered prompt enhancement** using Qwen 7B
  - Multiple quality tiers for both images and videos
  - Enhanced variants with automatic prompt improvement
  - Memory management with model unloading
  - Fixed polling intervals (5-second proper delays)
- **Job Types**: 
  - Standard: `image_fast`, `image_high`, `video_fast`, `video_high`
  - Enhanced: `image7b_fast_enhanced`, `image7b_high_enhanced`, `video7b_fast_enhanced`, `video7b_high_enhanced`
- **Status**: ✅ **ACTIVE - Production System**

## 📁 **LEGACY/BACKUP WORKERS** (Not Active)

### 🚀 Optimized Worker (`worker.py`) - **LEGACY**
**Purpose**: Previous GPU-optimized production version with 2.6x performance improvement
- **Status**: ❌ **LEGACY - Replaced by enhanced WAN worker**

### 🎯 Enhanced Multi-Model Worker (`ourvidz_enhanced_worker.py`) - **LEGACY**
**Purpose**: Previous multi-model integration with functional quality tiers
- **Status**: ❌ **LEGACY - Replaced by dual orchestrator architecture**

### 🔧 14B Worker (`worker-14b.py`) - **LEGACY**
**Purpose**: Previous high-quality generation using 14B parameter models
- **Status**: ❌ **LEGACY - Functionality integrated into enhanced WAN worker**

## 📁 **LEGACY/BACKUP WORKERS** (Not Active)

### 📁 Legacy Workers
- `worker_Old_Wan_only.py`: Original WAN-only implementation
- `sdxl_worker_old.py`: Previous SDXL implementation  
- `dual_orchestrator_old.py`: Previous orchestrator version
- `worker.py`: Previous optimized worker (replaced by enhanced WAN worker)
- `ourvidz_enhanced_worker.py`: Previous multi-model worker (replaced by dual orchestrator)
- `worker-14b.py`: Previous 14B worker (functionality integrated into enhanced WAN worker)

## Setup and Configuration

### 🔧 Setup Script (`setup.sh`)
**Purpose**: Automated environment setup and dependency installation
- **Key Features**:
  - PyTorch 2.4.1+cu124 installation
  - Diffusers ecosystem setup
  - WAN 2.1 repository cloning
  - SDXL model download
  - Comprehensive dependency validation
- **Dependencies**: CUDA 12.4, Python 3.11

### 📦 Requirements (`requirements.txt`)
**Purpose**: Python dependency specification
- **Core Dependencies**:
  - PyTorch 2.4.1+cu124 ecosystem
  - Diffusers 0.31.0, Transformers 4.45.2
  - xformers 0.0.28.post2 for memory optimization
  - WAN 2.1 specific: easydict, av, decord, omegaconf, hydra-core

### 📥 Model Download Scripts
- `download_models.py`: Downloads WAN and Mistral models
- `download_all_models.py`: Comprehensive model download utility

## Environment Configuration

### 🔑 Required Environment Variables
```bash
SUPABASE_URL=              # Supabase database URL
SUPABASE_SERVICE_KEY=      # Supabase service key
UPSTASH_REDIS_REST_URL=    # Redis queue URL
UPSTASH_REDIS_REST_TOKEN=  # Redis authentication token
HF_TOKEN=                  # Optional HuggingFace token
```

### 🗂️ Directory Structure
```
/workspace/
├── models/
│   ├── sdxl-lustify/          # SDXL model files
│   ├── wan2.1-t2v-1.3b/       # WAN 1.3B model
│   ├── wan2.1-t2v-14b/        # WAN 14B model
│   └── huggingface_cache/     # HF model cache
├── Wan2.1/                    # WAN 2.1 source code
└── python_deps/               # Persistent Python dependencies
```

## Job Processing Flow

### 📋 Job Types and Configurations

#### SDXL Jobs
| Job Type | Quality | Steps | Time | Resolution | Use Case |
|----------|---------|-------|------|------------|----------|
| `sdxl_image_fast` | Fast | 15 | 3-8s | 1024x1024 | Quick preview |
| `sdxl_image_high` | High | 25 | 3-8s | 1024x1024 | Final quality |

#### WAN Jobs
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

### 🔄 Processing Pipeline
1. **Job Polling**: Workers poll Redis queue for new jobs
2. **Model Loading**: Load appropriate model based on job type
3. **Content Generation**: Execute generation with optimized parameters
4. **File Upload**: Upload generated content to Supabase storage
5. **Completion Notification**: Notify job completion via Redis

## Performance Characteristics

### ⚡ Speed Optimizations
- **SDXL**: 3.6s per image, batch processing for 6 images
- **WAN Fast**: 67s for images, 180s for videos
- **WAN High**: 90s for images, 280s for videos
- **GPU Memory**: Optimized for RTX 6000 ADA 48GB

### 🔧 Memory Management
- Model unloading between jobs
- Attention slicing and xformers optimization
- GPU memory monitoring and cleanup
- Conditional model offloading

## Error Handling and Monitoring

### 🛡️ Robustness Features
- Graceful validation in orchestrator
- Automatic worker restart on failure
- Comprehensive error logging
- Timeout handling (10-15 minute limits)
- GPU memory monitoring

### 📊 Monitoring
- Real-time GPU memory usage
- Generation time tracking
- Worker status monitoring
- Performance metrics logging

## Development Notes

### 🔄 Version History
- Multiple iterations of workers with performance improvements
- Legacy versions preserved for reference
- Continuous optimization for production stability

### 🎯 Key Improvements
- 2.6x performance improvement in optimized worker
- Batch generation for better UX
- AI prompt enhancement integration
- Dual worker orchestration for concurrent processing

### 🔧 Technical Debt
- Multiple worker versions (consolidation opportunity)
- Legacy files present but not actively used
- Environment setup complexity (addressed in setup.sh)

## Usage Instructions

### 🚀 **PRODUCTION STARTUP**
1. Run `setup.sh` for automated environment setup
2. Configure environment variables
3. **Start production system**: `python dual_orchestrator.py`
   - This starts both SDXL and WAN workers concurrently
   - Automatic monitoring and restart capabilities
   - Production-ready with comprehensive logging

### 🔧 **INDIVIDUAL WORKER TESTING** (Development Only)
- `python sdxl_worker.py` - Test SDXL image generation only
- `python wan_worker.py` - Test WAN video/image generation only

### 🔧 Customization
- Modify job configurations in respective worker files
- Adjust model paths and parameters as needed
- Add new job types by extending configuration dictionaries

## 🎯 **CURRENT PRODUCTION STATUS**

### ✅ **Active Components**
- **Dual Orchestrator**: Main production controller
- **SDXL Worker**: Fast image generation with batch support
- **Enhanced WAN Worker**: Video generation with AI enhancement

### ❌ **Legacy Components** (Not Used)
- All other worker files are legacy/backup versions
- Functionality has been consolidated into the active trio

This codebase represents a **production-ready AI content generation system** optimized for high-performance GPU environments with comprehensive error handling and monitoring capabilities. The current architecture uses a **dual-worker orchestration pattern** for optimal resource utilization and reliability. 