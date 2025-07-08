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
| `sdxl_image_fast` | Fast | 15 | 29.9s | 1024x1024 | Quick preview (6 images) |
| `sdxl_image_high` | High | 25 | 42.4s | 1024x1024 | Final quality (6 images) |

#### WAN Jobs
| Job Type | Quality | Steps | Frames | Time | Resolution | Enhancement |
|----------|---------|-------|--------|------|------------|-------------|
| `image_fast` | Fast | 4 | 1 | 73s | 480x832 | No |
| `image_high` | High | 6 | 1 | 90s | 480x832 | No |
| `video_fast` | Fast | 4 | 17 | 241.4s | 480x832 | No |
| `video_high` | High | 6 | 17 | 360s | 480x832 | No |
| `image7b_fast_enhanced` | Fast | 4 | 1 | 233.5s | 480x832 | Yes |
| `image7b_high_enhanced` | High | 6 | 1 | 104s | 480x832 | Yes |
| `video7b_fast_enhanced` | Fast | 4 | 17 | 266.1s | 480x832 | Yes |
| `video7b_high_enhanced` | High | 6 | 17 | 361s | 480x832 | Yes |

### 🔄 Processing Pipeline
1. **Job Polling**: Workers poll Redis queue for new jobs
2. **Model Loading**: Load appropriate model based on job type
3. **Content Generation**: Execute generation with optimized parameters
4. **File Upload**: Upload generated content to Supabase storage
5. **Completion Notification**: Notify job completion via Redis

## Performance Characteristics

### ⚡ Speed Optimizations
- **SDXL**: 29.9-42.4s total (3.1-5.0s per image), batch processing for 6 images
- **WAN Fast**: 73-241s for images/videos
- **WAN High**: 90-360s for images/videos
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
- Legacy versions removed for cleaner codebase
- Continuous optimization for production stability

### 🎯 Key Improvements
- 2.6x performance improvement in optimized worker
- Batch generation for better UX
- AI prompt enhancement integration
- Dual worker orchestration for concurrent processing

### 🔧 Technical Debt
- Legacy files removed for cleaner maintenance
- Environment setup complexity (addressed in setup.sh)
- Performance optimization opportunities identified

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

### 📊 **Testing Status**
- **SDXL Jobs**: ✅ Both job types tested and working
- **WAN Jobs**: ✅ 5/8 job types tested and working
- **Enhanced Jobs**: ✅ Working but quality optimization needed
- **Performance Baselines**: ✅ Real data established for tested jobs

### 🚧 **Pending Testing**
- **WAN Standard**: `image_high`, `video_high`
- **WAN Enhanced**: `image7b_high_enhanced`
- **Performance Optimization**: Model pre-loading implementation

This codebase represents a **production-ready AI content generation system** optimized for high-performance GPU environments with comprehensive error handling and monitoring capabilities. The current architecture uses a **dual-worker orchestration pattern** for optimal resource utilization and reliability. 