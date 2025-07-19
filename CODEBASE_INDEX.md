# OurVidz Worker Codebase Index

## Overview
OurVidz Worker is a GPU-accelerated AI content generation system designed for RunPod deployment. It supports multiple AI models for image and video generation with different quality tiers and performance characteristics.

**📋 For detailed API specifications, see [WORKER_API.md](./WORKER_API.md)**

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
- **Performance**: 30-42s (SDXL), 25-240s (WAN)
- **Status**: ✅ **ACTIVE - Production System**

### 🎨 SDXL Worker (`sdxl_worker.py`) - **ACTIVE**
**Purpose**: Fast image generation using LUSTIFY SDXL model
- **Model**: `lustifySDXLNSFWSFW_v20.safetensors`
- **Features**:
  - **Batch generation (1, 3, or 6 images per request)** - Major UX improvement
  - Two quality tiers: fast (15 steps) and high (25 steps)
  - Optimized for speed: 3-8s per image
  - Memory-efficient with attention slicing and xformers
  - Proper PNG Content-Type headers for uploads
  - **Reference image support** with style, composition, and character modes
- **Job Types**: `sdxl_image_fast`, `sdxl_image_high`
- **Output**: PNG images, 1024x1024 resolution
- **Status**: ✅ **ACTIVE - Production System**

### 🎬 Enhanced WAN Worker (`wan_worker.py`) - **ACTIVE**
**Purpose**: Video and image generation with AI prompt enhancement and comprehensive reference frame support
- **Models**: 
  - WAN 2.1 T2V 1.3B for video generation
  - Qwen 2.5-7B for prompt enhancement
- **Features**:
  - **AI-powered prompt enhancement** using Qwen 7B
  - **Comprehensive reference frame support** - All 5 modes (none, single, start, end, both)
  - Multiple quality tiers for both images and videos
  - Enhanced variants with automatic prompt improvement
  - Memory management with model unloading
  - Fixed polling intervals (5-second proper delays)
  - **Correct WAN 1.3B task usage** - Always uses `t2v-1.3B` with appropriate parameters
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
│   └── huggingface_cache/     # HF model cache
├── Wan2.1/                    # WAN 2.1 source code
├── ourvidz-worker/            # Worker repository
│   ├── wan_generate.py        # WAN generation script
│   ├── sdxl_worker.py         # SDXL worker
│   ├── wan_worker.py          # WAN worker
│   └── dual_orchestrator.py   # Main orchestrator
└── python_deps/               # Persistent Python dependencies
```

## Job Processing Flow

### 📋 Job Types and Configurations

#### SDXL Jobs
| Job Type | Quality | Steps | Time | Resolution | Use Case |
|----------|---------|-------|------|------------|----------|
| `sdxl_image_fast` | Fast | 15 | 30s | 1024x1024 | Quick preview (1,3,6 images) |
| `sdxl_image_high` | High | 25 | 42s | 1024x1024 | Final quality (1,3,6 images) |

#### WAN Jobs
| Job Type | Quality | Steps | Frames | Time | Resolution | Enhancement | Reference Support |
|----------|---------|-------|--------|------|------------|-------------|-------------------|
| `image_fast` | Fast | 25 | 1 | 25-40s | 480x832 | No | ✅ All 5 modes |
| `image_high` | High | 50 | 1 | 40-100s | 480x832 | No | ✅ All 5 modes |
| `video_fast` | Fast | 25 | 83 | 135-180s | 480x832 | No | ✅ All 5 modes |
| `video_high` | High | 50 | 83 | 180-240s | 480x832 | No | ✅ All 5 modes |
| `image7b_fast_enhanced` | Fast Enhanced | 25 | 1 | 85-100s | 480x832 | Yes | ✅ All 5 modes |
| `image7b_high_enhanced` | High Enhanced | 50 | 1 | 100-240s | 480x832 | Yes | ✅ All 5 modes |
| `video7b_fast_enhanced` | Fast Enhanced | 25 | 83 | 195-240s | 480x832 | Yes | ✅ All 5 modes |
| `video7b_high_enhanced` | High Enhanced | 50 | 83 | 240+s | 480x832 | Yes | ✅ All 5 modes |

### 🔄 Processing Pipeline
1. **Job Polling**: Workers poll Redis queue for new jobs
2. **Reference Frame Detection**: WAN worker detects reference frame mode (none, single, start, end, both)
3. **Model Loading**: Load appropriate model based on job type
4. **Content Generation**: Execute generation with optimized parameters and reference frames
5. **File Upload**: Upload generated content to Supabase storage
6. **Completion Notification**: Notify job completion via Redis with comprehensive metadata

## Performance Characteristics

### ⚡ Speed Optimizations
- **SDXL**: 30-42s total (3-8s per image), batch processing for 1, 3, or 6 images
- **WAN Fast**: 25-180s for images/videos
- **WAN High**: 40-240s for images/videos
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
- **Reference frame fallback**: Graceful degradation if reference processing fails

### 📊 Monitoring
- Real-time GPU memory usage
- Generation time tracking
- Worker status monitoring
- Performance metrics logging
- Reference frame mode tracking

## Development Notes

### 🔄 Version History
- Multiple iterations of workers with performance improvements
- Legacy versions removed for cleaner codebase
- Continuous optimization for production stability
- **Latest**: WAN 1.3B model support with comprehensive reference frame capabilities

### 🎯 Key Improvements
- 2.6x performance improvement in optimized worker
- Batch generation for better UX (1, 3, or 6 images)
- AI prompt enhancement integration
- Dual worker orchestration for concurrent processing
- **Comprehensive reference frame support** - All 5 modes implemented
- **Fixed module imports** - Proper PYTHONPATH configuration
- **Correct file paths** - Uses `/workspace/ourvidz-worker/wan_generate.py`

### 🔧 Technical Debt
- Legacy files removed for cleaner maintenance
- Environment setup complexity (addressed in setup.sh)
- Performance optimization opportunities identified
- **Documentation consolidated** - Single source of truth in WORKER_API.md

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
- **Reference frame modes**: Configure via job payload parameters

## 🎯 **CURRENT PRODUCTION STATUS**

### ✅ **Active Components**
- **Dual Orchestrator**: Main production controller
- **SDXL Worker**: Fast image generation with batch support and reference images
- **Enhanced WAN Worker**: Video generation with AI enhancement and comprehensive reference frame support

### 📊 **Testing Status**
- **SDXL Jobs**: ✅ Both job types tested and working
- **WAN Jobs**: ✅ All 8 job types tested and working
- **Reference Frames**: ✅ All 5 reference modes tested and working
- **Performance Baselines**: ✅ Real data established for all jobs

### 🚧 **System Capabilities**
- **✅ 10 Job Types**: All job types operational
- **✅ 5 Reference Modes**: Complete reference frame support (none, single, start, end, both)
- **✅ Batch Processing**: SDXL supports 1, 3, or 6 images
- **✅ AI Enhancement**: WAN enhanced variants with Qwen 7B
- **✅ Error Recovery**: Robust error handling and fallback mechanisms
- **✅ Performance Monitoring**: Comprehensive metrics and logging

### 📋 **Reference Frame Support Matrix**
| **Reference Mode** | **Config Parameter** | **Metadata Fallback** | **WAN Parameters** | **Use Case** |
|-------------------|---------------------|----------------------|-------------------|--------------|
| **None** | No parameters | No parameters | None | Standard T2V |
| **Single** | `config.image` | `metadata.reference_image_url` | `--image ref.png` | I2V-style |
| **Start** | `config.first_frame` | `metadata.start_reference_url` | `--first_frame start.png` | Start frame |
| **End** | `config.last_frame` | `metadata.end_reference_url` | `--last_frame end.png` | End frame |
| **Both** | `config.first_frame` + `config.last_frame` | `metadata.start_reference_url` + `metadata.end_reference_url` | `--first_frame start.png --last_frame end.png` | Transition |

This codebase represents a **production-ready AI content generation system** optimized for high-performance GPU environments with comprehensive error handling, monitoring capabilities, and **complete reference frame support**. The current architecture uses a **dual-worker orchestration pattern** for optimal resource utilization and reliability.

**📋 For complete API specifications and implementation details, see [WORKER_API.md](./WORKER_API.md)** 