# ourvidz-worker

OurVidz GPU Worker for RunPod

---

## 🚀 Overview
This repository contains the **GPU worker system** for [OurVidz.com](https://ourvidz.lovable.app/), designed for high-performance AI image and video generation on RunPod infrastructure. It supports multiple AI models (SDXL, WAN 1.3B, Qwen) and is optimized for RTX 6000 ADA (48GB VRAM).

- **Production-ready**: Triple worker orchestration (SDXL + Chat + WAN 1.3B)
- **Comprehensive reference frame support**: 5 reference modes for video generation
- **I2I Pipeline**: First-class Image-to-Image support with parameter clamping and two modes
- **Thumbnail Generation**: 256px WEBP thumbnails for images and mid-frame thumbnails for videos
- **Batch image & video generation**: 14 job types, NSFW-capable
- **Smart memory management**: Intelligent VRAM allocation and emergency handling
- **Enhanced chat service**: Qwen Instruct with dynamic prompts and unrestricted mode
- **NSFW optimization**: Zero content restrictions with anatomical accuracy focus
- **Backend**: Supabase (PostgreSQL, Auth, Storage, Edge Functions)
- **Queue**: Upstash Redis (REST API)
- **Frontend**: [Lovable](https://ourvidz.lovable.app/) (React/TypeScript)

---

## ⚡ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/ourvidz-worker.git
   cd ourvidz-worker
   ```

2. **Configure environment variables**
   - See the [Environment Configuration](#environment-configuration) section below

3. **Start the production system**
   ```bash
   ./startup.sh
   ```

---

## 🛠️ Development & Testing
- **Test SDXL worker only:**
  ```bash
  python sdxl_worker.py
  ```
- **Test Chat worker only:**
  ```bash
  python chat_worker.py
  ```
- **Test WAN worker only:**
  ```bash
  python wan_worker.py
  ```

---

## 📚 Documentation

### **📋 API Reference**
- **[SYSTEM_SUMMARY.md](./SYSTEM_SUMMARY.md)** - Frontend API reference for current system structure and capabilities
- **[WORKER_API.md](./WORKER_API.md)** - Complete API specifications, job types, and reference frame support
- **[CODEBASE_INDEX.md](./CODEBASE_INDEX.md)** - System architecture and component overview
- **[CHAT_WORKER_CONSOLIDATED.md](./CHAT_WORKER_CONSOLIDATED.md)** - Enhanced chat worker features and NSFW optimization
- **[I2I_THUMBNAIL_IMPLEMENTATION_SUMMARY.md](./I2I_THUMBNAIL_IMPLEMENTATION_SUMMARY.md)** - I2I pipeline and thumbnail generation details

### **🎯 Key Features**

#### **SDXL Worker**
- **Batch generation**: 1, 3, or 6 images per request
- **Two quality tiers**: Fast (15 steps) and High (25 steps)
- **I2I Pipeline**: First-class support using StableDiffusionXLImg2ImgPipeline
- **Two I2I Modes**:
  - **Promptless Exact Copy**: `denoise_strength ≤ 0.05`, `guidance_scale = 1.0`, `steps = 6-10`
  - **Reference Modify**: `denoise_strength = 0.10-0.25`, `guidance_scale = 4-7`, `steps = 15-30`
- **Parameter Clamping**: Worker-side guards ensure consistent behavior
- **Thumbnail Generation**: 256px WEBP thumbnails for each image
- **Reference image support**: Style, composition, and character modes
- **Enhanced error handling**: Comprehensive traceback logging and upload progress tracking
- **Correct callback format**: Uses `url` field for asset paths as expected by edge functions
- **Performance**: 30-42s total (3-8s per image), +2-5s for I2I, +0.5-1s for thumbnails

#### **Enhanced Chat Worker**
- **Dynamic system prompts**: Custom prompts for each conversation
- **Unrestricted mode detection**: Automatic adult content detection
- **Prompt enhancement**: Qwen 2.5-7B Instruct with intelligent fallback
- **NSFW optimization**: Zero content restrictions with anatomical accuracy
- **Memory management**: Smart loading/unloading with PyTorch 2.0 compilation
- **Performance**: 5-15s for prompt enhancement, 1-3s for cached responses

#### **WAN 1.3B Worker**
- **Video generation**: High-quality video with temporal consistency
- **Comprehensive reference frame support**: All 5 modes (none, single, start, end, both)
- **I2I Pipeline**: `denoise_strength` parameter replaces `reference_strength` for consistency
- **Video Thumbnail Generation**: Mid-frame extraction for better representation, 256px WEBP format
- **AI enhancement**: Qwen 7B Base prompt enhancement for improved quality
- **Performance**: 25-240s depending on job type and quality, +1-2s for video thumbnails

#### **Memory Manager**
- **Smart VRAM allocation**: Priority-based memory management
- **Emergency handling**: Force unload capabilities for critical situations
- **Pressure detection**: Real-time memory pressure monitoring
- **Predictive loading**: Smart preloading based on usage patterns

#### **I2I Pipeline Support Matrix**
| **Worker** | **Pipeline** | **Modes** | **Parameter** | **Clamping** | **Thumbnails** |
|------------|--------------|-----------|---------------|--------------|----------------|
| **SDXL** | StableDiffusionXLImg2ImgPipeline | Exact Copy + Reference Modify | `denoise_strength` | Worker-side guards | 256px WEBP |
| **WAN** | WAN 2.1 T2V 1.3B | All reference modes | `denoise_strength` | Worker-side guards | Mid-frame WEBP |

#### **Thumbnail Generation Matrix**
| **Asset Type** | **Thumbnail Source** | **Format** | **Size** | **Quality** | **Storage** |
|----------------|---------------------|------------|----------|-------------|-------------|
| **SDXL Images** | Generated image | WEBP | 256px longest edge | 85 | workspace-temp |
| **WAN Images** | Generated image | WEBP | 256px longest edge | 85 | workspace-temp |
| **WAN Videos** | Mid-frame extraction | WEBP | 256px longest edge | 85 | workspace-temp |

#### **Reference Frame Support Matrix**
| **Reference Mode** | **Config Parameter** | **WAN Parameters** | **Use Case** |
|-------------------|---------------------|-------------------|--------------|
| **None** | No parameters | None | Standard T2V |
| **Single** | `config.image` | `--image ref.png` | I2V-style |
| **Start** | `config.first_frame` | `--first_frame start.png` | Start frame |
| **End** | `config.last_frame` | `--last_frame end.png` | End frame |
| **Both** | `config.first_frame` + `config.last_frame` | `--first_frame start.png --last_frame end.png` | Transition |

### **🔧 System Architecture**
- **Triple Worker Orchestrator**: Manages SDXL, Chat, and WAN workers concurrently
- **Priority-based startup**: SDXL (1) → Chat (2) → WAN (3)
- **Smart Memory Management**: Intelligent VRAM allocation and coordination
- **Job Queue System**: Redis-based job distribution
- **Storage Integration**: Supabase storage for generated content and thumbnails
- **Error Handling**: Comprehensive error recovery and fallback mechanisms
- **Auto-Registration**: Automatic RunPod URL management

### **📊 Job Types**
- **SDXL**: `sdxl_image_fast`, `sdxl_image_high`
- **Chat**: `chat_enhance`, `chat_conversation`, `chat_unrestricted`, `admin_utilities`
- **WAN Standard**: `image_fast`, `image_high`, `video_fast`, `video_high`
- **WAN Enhanced**: `image7b_fast_enhanced`, `image7b_high_enhanced`, `video7b_fast_enhanced`, `video7b_high_enhanced`

### **🧠 Enhanced Chat Worker Features**
| **Feature** | **Description** | **Use Case** |
|-------------|----------------|--------------|
| **Dynamic System Prompts** | Custom prompts per conversation | Context-aware responses |
| **Unrestricted Mode** | Automatic adult content detection | NSFW content creation |
| **Intelligent Enhancement** | Edge function integration with fallback | High-quality prompt enhancement |
| **NSFW Optimization** | Zero content restrictions | Unrestricted adult content |
| **Anatomical Accuracy** | Realistic proportions and poses | Professional quality output |
| **Performance Caching** | Enhancement result caching | Faster repeated requests |

### **🧠 Memory Management Features**
| **Feature** | **Description** | **Use Case** |
|-------------|----------------|--------------|
| **Pressure Detection** | Critical/High/Medium/Low levels | Real-time monitoring |
| **Emergency Unload** | Force unload all except target | Critical situations |
| **Predictive Loading** | Smart preloading based on patterns | Performance optimization |
| **Intelligent Fallback** | Selective vs nuclear unloading | Memory pressure handling |
| **Worker Coordination** | HTTP-based memory management | Cross-worker communication |

---

## 🔑 Environment Configuration

### **Dependencies**
All Python dependencies are pre-installed in persistent `/workspace` storage:
```bash
/workspace/python_deps/lib/python3.11/site-packages/
├── cv2/                    # OpenCV-Python 4.10.0.84
├── torch/                  # PyTorch 2.4.1+cu124
├── transformers/           # Transformers 4.45.2
├── diffusers/             # Diffusers 0.31.0
├── flask/                 # Flask 3.0.2
└── ... (other packages)
```

**Environment Setup**:
- `PYTHONPATH` automatically set in `startup.sh`
- Dependencies persist across pod restarts
- OpenCV-Python available for video processing and thumbnail generation

### **Required Environment Variables**
```bash
SUPABASE_URL=              # Supabase database URL
SUPABASE_SERVICE_KEY=      # Supabase service key
UPSTASH_REDIS_REST_URL=    # Redis queue URL
UPSTASH_REDIS_REST_TOKEN=  # Redis authentication token
WAN_WORKER_API_KEY=        # API key for WAN worker authentication
HF_TOKEN=                  # Optional HuggingFace token
```

### **Directory Structure**
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

---

## 🤝 Contributing & Support
- For issues, feature requests, or contributions, please open a GitHub issue or pull request.
- For business or technical questions, contact the maintainer.

---

**© 2025 OurVidz.com. All rights reserved.**  
**Last Updated:** August 18, 2025
