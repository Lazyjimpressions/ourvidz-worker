# OurVidz Worker Codebase Index

## Overview
OurVidz Worker is a GPU-accelerated AI content generation system designed for RunPod deployment. It supports multiple AI models for image and video generation with different quality tiers and performance characteristics.

**📋 For detailed API specifications, see [WORKER_API.md](./WORKER_API.md)**

## 🚀 **ACTIVE PRODUCTION ARCHITECTURE**

### 🎭 Triple Worker Orchestrator (`dual_orchestrator.py`) - **ACTIVE**
**Purpose**: Main orchestrator that manages SDXL, Chat, and WAN workers concurrently
- **Key Features**:
  - Manages three worker processes: SDXL (image), Chat (Qwen Instruct), and WAN (video+image)
  - Priority-based startup: SDXL (1) → Chat (2) → WAN (3)
  - Handles graceful validation and error recovery
  - Optimized for RTX 6000 ADA 48GB VRAM
  - Automatic restart and monitoring capabilities
  - Production-ready with comprehensive logging
- **Job Types Managed**:
  - SDXL: `sdxl_image_fast`, `sdxl_image_high`
  - Chat: `chat_enhance`, `chat_conversation`, `admin_utilities`
  - WAN: `image_fast`, `image_high`, `video_fast`, `video_high`, enhanced variants
- **Performance**: 30-42s (SDXL), 5-15s (Chat), 25-240s (WAN)
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
- **Port**: 7860 (shared with WAN worker)
- **Status**: ✅ **ACTIVE - Production System**

### 💬 Enhanced Chat Worker (`chat_worker.py`) - **ACTIVE**
**Purpose**: Advanced Qwen Instruct service with dynamic prompts, unrestricted mode, and NSFW optimization
- **Model**: Qwen 2.5-7B Instruct
- **Features**:
  - **Dynamic system prompts** with custom prompts per conversation
  - **Unrestricted mode detection** for automatic adult content handling
  - **Simple prompt enhancement** using direct Qwen Instruct model
  - **NSFW optimization** with zero content restrictions and anatomical accuracy
  - **Memory management** with smart loading/unloading
  - **PyTorch 2.0 compilation** for performance optimization
  - **Comprehensive OOM error handling** with retry logic
  - **Model validation** and device pinning
  - **Admin utilities** for memory management and enhancement info
  - **Health monitoring** and performance tracking
- **Job Types**: `chat_enhance`, `chat_conversation`, `chat_unrestricted`, `admin_utilities`
- **Output**: Enhanced prompts, chat responses, memory status, enhancement info
- **Port**: 7861 (dedicated)
- **Status**: ✅ **ACTIVE - Production System**

### 🎬 Enhanced WAN Worker (`wan_worker.py`) - **ACTIVE**
**Purpose**: Video and image generation with AI prompt enhancement and comprehensive reference frame support
- **Models**: 
  - WAN 2.1 T2V 1.3B for video generation
  - Qwen 2.5-7B Base for prompt enhancement
- **Features**:
  - **AI-powered prompt enhancement** using Qwen 7B Base
  - **Comprehensive reference frame support** - All 5 modes (none, single, start, end, both)
  - Multiple quality tiers for both images and videos
  - Enhanced variants with automatic prompt improvement
  - Memory management with model unloading
  - Fixed polling intervals (5-second proper delays)
  - **Correct WAN 1.3B task usage** - Always uses `t2v-1.3B` with appropriate parameters
  - **Thread-safe timeouts** using concurrent.futures
  - **Flask API endpoints** for frontend integration
- **Job Types**: 
  - Standard: `image_fast`, `image_high`, `video_fast`, `video_high`
  - Enhanced: `image7b_fast_enhanced`, `image7b_high_enhanced`, `video7b_fast_enhanced`, `video7b_high_enhanced`
- **Port**: 7860 (shared with SDXL worker)
- **Status**: ✅ **ACTIVE - Production System**

### 🧠 Memory Manager (`memory_manager.py`) - **ACTIVE**
**Purpose**: Smart VRAM allocation and coordination for triple worker system
- **Key Features**:
  - **Memory pressure detection** (critical/high/medium/low)
  - **Emergency memory management** with intelligent fallback
  - **Force unload capabilities** for critical situations
  - **Predictive loading** based on usage patterns
  - **Comprehensive emergency status reporting**
  - **Worker coordination** via HTTP APIs
- **Memory Allocation**:
  - SDXL: 10GB (Always loaded)
  - Chat: 15GB (Load when possible)
  - WAN: 30GB (Load on demand)
- **Status**: ✅ **ACTIVE - Production System**

### 🔧 Worker Registration (`worker_registration.py`) - **ACTIVE**
**Purpose**: Automatic worker registration and URL management for RunPod deployment
- **Key Features**:
  - Automatic RunPod URL detection and registration
  - Worker heartbeat monitoring
  - Supabase integration for worker status tracking
  - Graceful shutdown handling
  - URL validation and health checks
- **Status**: ✅ **ACTIVE - Production System**

### 🎬 WAN Generation Script (`wan_generate.py`) - **ACTIVE**
**Purpose**: Core WAN 2.1 generation script used by WAN worker
- **Features**:
  - Direct WAN 2.1 T2V 1.3B model integration
  - Command-line interface for video generation
  - Support for all reference frame modes
  - Quality tier configuration
  - Performance optimization
- **Status**: ✅ **ACTIVE - Production System**

## Setup and Configuration

### 🚀 Startup Script (`startup.sh`)
**Purpose**: Production startup with triple worker system validation
- **Key Features**:
  - **Triple worker status assessment** (SDXL, Chat, WAN)
  - **Model readiness verification** for all workers
  - **RunPod URL detection** and auto-registration
  - **Environment configuration** for all workers
  - **Priority-based startup sequence**
- **Status**: ✅ **ACTIVE - Production System**

### 📦 Requirements (`requirements.txt`)
**Purpose**: Python dependency specification
- **Core Dependencies**:
  - PyTorch 2.4.1+cu124 ecosystem
  - Diffusers 0.31.0, Transformers 4.45.2
  - xformers 0.0.28.post2 for memory optimization
  - WAN 2.1 specific: easydict, av, decord, omegaconf, hydra-core
  - Flask for API endpoints
  - psutil for system monitoring

## Environment Configuration

### 🔑 Required Environment Variables
```bash
SUPABASE_URL=              # Supabase database URL
SUPABASE_SERVICE_KEY=      # Supabase service key
UPSTASH_REDIS_REST_URL=    # Redis queue URL
UPSTASH_REDIS_REST_TOKEN=  # Redis authentication token
WAN_WORKER_API_KEY=        # API key for WAN worker authentication
HF_TOKEN=                  # Optional HuggingFace token
```

### 🗂️ Directory Structure
```
/workspace/
├── models/
│   ├── sdxl-lustify/          # SDXL model files
│   ├── wan2.1-t2v-1.3b/       # WAN 1.3B model
│   └── huggingface_cache/     # HF model cache
│       ├── models--Qwen--Qwen2.5-7B/           # Qwen Base model
│       └── models--Qwen--Qwen2.5-7B-Instruct/  # Qwen Instruct model
├── Wan2.1/                    # WAN 2.1 source code
├── ourvidz-worker/            # Worker repository
│   ├── wan_generate.py        # WAN generation script
│   ├── sdxl_worker.py         # SDXL worker
│   ├── chat_worker.py         # Chat worker
│   ├── wan_worker.py          # WAN worker
│   ├── dual_orchestrator.py   # Main orchestrator
│   ├── memory_manager.py      # Memory management
│   ├── worker_registration.py # Worker registration
│   ├── startup.sh             # Production startup script
│   └── archive/               # Archived documentation and test files
└── python_deps/               # Persistent Python dependencies
```

## Job Processing Flow

### 📋 Job Types and Configurations

#### SDXL Jobs
| Job Type | Quality | Steps | Time | Resolution | Use Case |
|----------|---------|-------|------|------------|----------|
| `sdxl_image_fast` | Fast | 15 | 30s | 1024x1024 | Quick preview (1,3,6 images) |
| `sdxl_image_high` | High | 25 | 42s | 1024x1024 | Final quality (1,3,6 images) |

#### Enhanced Chat Jobs
| Job Type | Purpose | Model | Time | Features |
|----------|---------|-------|------|----------|
| `chat_enhance` | Simple prompt enhancement | Qwen Instruct | 1-3s | Direct Qwen Instruct enhancement, NSFW optimization |
| `chat_conversation` | Dynamic chat interface | Qwen Instruct | 5-15s | Custom system prompts, unrestricted mode detection |
| `chat_unrestricted` | Dedicated NSFW chat | Qwen Instruct | 5-15s | Adult content optimization, anatomical accuracy |
| `admin_utilities` | System management | N/A | <1s | Memory status, enhancement info |

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
2. **Memory Management**: Memory manager coordinates resource allocation
3. **Reference Frame Detection**: WAN worker detects reference frame mode (none, single, start, end, both)
4. **Model Loading**: Load appropriate model based on job type
5. **Content Generation**: Execute generation with optimized parameters and reference frames
6. **File Upload**: Upload generated content to Supabase storage
7. **Completion Notification**: Notify job completion via Redis with comprehensive metadata

## Performance Characteristics

### ⚡ Speed Optimizations
- **SDXL**: 30-42s total (3-8s per image), batch processing for 1, 3, or 6 images
- **Chat**: 5-15s for prompt enhancement and chat responses
- **WAN Fast**: 25-180s for images/videos
- **WAN High**: 40-240s for images/videos
- **GPU Memory**: Optimized for RTX 6000 ADA 48GB

### 🔧 Memory Management
- **Smart allocation**: Priority-based memory management
- **Dynamic loading**: Chat worker loads/unloads based on demand
- **Emergency management**: Force unload capabilities for critical situations
- **Pressure detection**: Real-time memory pressure monitoring
- **Predictive loading**: Smart preloading based on usage patterns

## Error Handling and Monitoring

### 🛡️ Robustness Features
- Graceful validation in orchestrator
- Automatic worker restart on failure
- Comprehensive error logging
- Timeout handling (10-15 minute limits)
- GPU memory monitoring
- **Reference frame fallback**: Graceful degradation if reference processing fails
- **Memory pressure handling**: Emergency memory management
- **Thread-safe operations**: Concurrent.futures for timeouts

### 📊 Monitoring
- Real-time GPU memory usage
- Generation time tracking
- Worker status monitoring
- Performance metrics logging
- Reference frame mode tracking
- Memory pressure levels
- Emergency action tracking

## API Endpoints

### 🎨 SDXL Worker (Port 7860)
- **Health**: `GET /health`
- **Status**: `GET /status`

### 💬 Enhanced Chat Worker (Port 7861)
- **Health**: `GET /health`
- **Chat**: `POST /chat`
- **Unrestricted Chat**: `POST /chat/unrestricted`
- **Enhance**: `POST /enhance`
- **Legacy Enhance**: `POST /enhance/legacy`
- **Enhancement Info**: `GET /enhancement/info`
- **Memory Status**: `GET /memory/status`
- **Model Info**: `GET /model/info`
- **Memory Load**: `POST /memory/load`
- **Memory Unload**: `POST /memory/unload`

### 🎬 WAN Worker (Port 7860)
- **Health**: `GET /health`
- **Debug**: `GET /debug/env`
- **Enhance**: `POST /enhance`

### 🧠 Memory Manager
- **Status**: `GET /memory/status`
- **Emergency Operations**: `POST /emergency/operation`
- **Memory Report**: `GET /memory/report`

## Development Notes

### 🔄 Version History
- Multiple iterations of workers with performance improvements
- Legacy versions removed for cleaner codebase
- Continuous optimization for production stability
- **Latest**: Triple worker system with Chat worker and Memory Manager
- **August 16, 2025**: Codebase cleanup and documentation consolidation

### 🎯 Key Improvements
- 2.6x performance improvement in optimized worker
- Batch generation for better UX (1, 3, or 6 images)
- AI prompt enhancement integration
- Triple worker orchestration for concurrent processing
- **Comprehensive reference frame support** - All 5 modes implemented
- **Enhanced chat worker** - Dynamic prompts, unrestricted mode, NSFW optimization
- **Simplified enhancement system** - Direct Qwen Instruct enhancement
- **Memory management** - OOM handling with retry logic
- **Memory manager** - Smart VRAM allocation and coordination
- **Thread-safe timeouts** - Concurrent.futures implementation
- **Emergency memory management** - Critical situation handling
- **Zero content restrictions** - Full NSFW optimization with anatomical accuracy
- **Codebase cleanup** - Archived unused documentation and test files

### 🔧 Technical Debt
- Legacy files archived for cleaner maintenance
- Environment setup complexity (addressed in startup.sh)
- Performance optimization opportunities identified
- **Documentation consolidated** - Single source of truth in WORKER_API.md

## Usage Instructions

### 🚀 **PRODUCTION STARTUP**
1. Configure environment variables
2. **Start production system**: `./startup.sh`
   - This starts all three workers (SDXL, Chat, WAN) in priority order
   - Automatic monitoring and restart capabilities
   - Production-ready with comprehensive logging
   - Memory management and coordination

### 🔧 **INDIVIDUAL WORKER TESTING** (Development Only)
- `python sdxl_worker.py` - Test SDXL image generation only
- `python chat_worker.py` - Test Chat worker only
- `python wan_worker.py` - Test WAN video/image generation only

### 🔧 Customization
- Modify job configurations in respective worker files
- Adjust model paths and parameters as needed
- Add new job types by extending configuration dictionaries
- **Reference frame modes**: Configure via job payload parameters
- **Memory management**: Configure via memory_manager.py

## 🎯 **CURRENT PRODUCTION STATUS**

### ✅ **Active Components**
- **Triple Orchestrator**: Main production controller
- **SDXL Worker**: Fast image generation with batch support and reference images
- **Chat Worker**: Qwen Instruct service for prompt enhancement and chat
- **Enhanced WAN Worker**: Video generation with AI enhancement and comprehensive reference frame support
- **Memory Manager**: Smart VRAM allocation and coordination
- **Worker Registration**: Automatic RunPod URL management
- **WAN Generation Script**: Core WAN 2.1 integration

### 📊 **Testing Status**
- **SDXL Jobs**: ✅ Both job types tested and working
- **Chat Jobs**: ✅ All endpoints tested and working
- **WAN Jobs**: ✅ All 8 job types tested and working
- **Reference Frames**: ✅ All 5 reference modes tested and working
- **Memory Management**: ✅ All emergency operations tested and working
- **Performance Baselines**: ✅ Real data established for all jobs

### 🚧 **System Capabilities**
- **✅ 14 Job Types**: All job types operational (including new chat_unrestricted)
- **✅ 5 Reference Modes**: Complete reference frame support (none, single, start, end, both)
- **✅ Batch Processing**: SDXL supports 1, 3, or 6 images
- **✅ AI Enhancement**: WAN enhanced variants with Qwen 7B
- **✅ Enhanced Chat Service**: Dynamic prompts, unrestricted mode, NSFW optimization
- **✅ Simple Enhancement**: Direct Qwen Instruct enhancement
- **✅ Memory Management**: Smart VRAM allocation and emergency handling
- **✅ Error Recovery**: Robust error handling and fallback mechanisms
- **✅ Performance Monitoring**: Comprehensive metrics and logging
- **✅ Zero Content Restrictions**: Full NSFW optimization with anatomical accuracy
- **✅ Auto-Registration**: Automatic RunPod URL management

### 📋 **Reference Frame Support Matrix**
| **Reference Mode** | **Config Parameter** | **Metadata Fallback** | **WAN Parameters** | **Use Case** |
|-------------------|---------------------|----------------------|-------------------|--------------|
| **None** | No parameters | No parameters | None | Standard T2V |
| **Single** | `config.image` | `metadata.reference_image_url` | `--image ref.png` | I2V-style |
| **Start** | `config.first_frame` | `metadata.start_reference_url` | `--first_frame start.png` | Start frame |
| **End** | `config.last_frame` | `metadata.end_reference_url` | `--last_frame end.png` | End frame |
| **Both** | `config.first_frame` + `config.last_frame` | `metadata.start_reference_url` + `metadata.end_reference_url` | `--first_frame start.png --last_frame end.png` | Transition |

### 🧠 **Memory Management Features**
| **Feature** | **Description** | **Use Case** |
|-------------|----------------|--------------|
| **Pressure Detection** | Critical/High/Medium/Low levels | Real-time monitoring |
| **Emergency Unload** | Force unload all except target | Critical situations |
| **Predictive Loading** | Smart preloading based on patterns | Performance optimization |
| **Intelligent Fallback** | Selective vs nuclear unloading | Memory pressure handling |
| **Worker Coordination** | HTTP-based memory management | Cross-worker communication |

### 💬 **Enhanced Chat Worker Features**
| **Feature** | **Description** | **Use Case** |
|-------------|----------------|--------------|
| **Dynamic System Prompts** | Custom prompts per conversation | Context-aware responses |
| **Unrestricted Mode** | Automatic adult content detection | NSFW content creation |
| **Simple Enhancement** | Direct Qwen Instruct enhancement | High-quality prompt enhancement |
| **NSFW Optimization** | Zero content restrictions | Unrestricted adult content |
| **Anatomical Accuracy** | Realistic proportions and poses | Professional quality output |
| **Memory Management** | Smart loading/unloading | Resource optimization |
| **Error Handling** | Comprehensive OOM handling | System stability |

## 📁 **ARCHIVE CONTENTS**

The `archive/` directory contains historical documentation and test files that are no longer actively used:

### 📚 **Archived Documentation**
- `ARCHITECTURAL_CLEANUP_SUMMARY.md` - Historical architectural changes
- `CHAT_WORKER_PURE_INFERENCE_OVERHAUL.md` - Chat worker development history
- `COMPREHENSIVE_CHANGES_VERIFICATION.md` - Change verification documentation
- `DOCUMENTATION_UPDATE_SUMMARY.md` - Documentation consolidation history
- `FINAL_UPDATE_SUMMARY.md` - Final update documentation
- `FRONTEND_SYSTEM_CHANGES_SUMMARY.md` - Frontend integration history
- `SDXL_WORKER_RUN_METHOD_FIX.md` - SDXL worker fixes
- `SYSTEM_PROMPT_FIXES_SUMMARY.md` - System prompt optimization history

### 🧪 **Archived Test Files**
- `test_response_extraction.py` - Response extraction testing
- `simple_verification.py` - Simple verification tests
- `verify_fixes.py` - Fix verification tests
- `test_system_prompt_fixes.py` - System prompt testing
- `chat_worker_validator.py` - Chat worker validation (empty)
- `comprehensive_test.sh` - Comprehensive testing script (empty)
- `quick_health_check.sh` - Health check script (empty)
- `README.md` - Testing documentation (empty)

This codebase represents a **production-ready AI content generation system** optimized for high-performance GPU environments with comprehensive error handling, monitoring capabilities, **complete reference frame support**, **enhanced chat service with NSFW optimization**, and **smart memory management**. The current architecture uses a **triple-worker orchestration pattern** for optimal resource utilization and reliability.

**📋 For complete API specifications and implementation details, see [WORKER_API.md](./WORKER_API.md)**

**📅 Last Updated: August 16, 2025** 