# ourvidz-worker

OurVidz GPU Worker for RunPod

---

## 🚀 Overview
This repository contains the **GPU worker system** for [OurVidz.com](https://ourvidz.lovable.app/), designed for high-performance AI image and video generation on RunPod infrastructure. It supports multiple AI models (SDXL, WAN 1.3B, Qwen) and is optimized for RTX 6000 ADA (48GB VRAM).

- **Production-ready**: Triple worker orchestration (SDXL + Chat + WAN 1.3B)
- **Comprehensive reference frame support**: 5 reference modes for video generation
- **Batch image & video generation**: 13 job types, NSFW-capable
- **Smart memory management**: Intelligent VRAM allocation and emergency handling
- **Dedicated chat service**: Qwen Instruct for prompt enhancement
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

2. **Set up the environment**
   ```bash
   ./setup.sh
   # or manually install dependencies as per requirements.txt
   ```

3. **Configure environment variables**
   - See `.env.example` or the [Deployment Guide](docs/DEPLOYMENT.md#🔑-environment-configuration)

4. **Start the production system**
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
- **[WORKER_API.md](./WORKER_API.md)** - Complete API specifications, job types, and reference frame support
- **[CODEBASE_INDEX.md](./CODEBASE_INDEX.md)** - System architecture and component overview

### **🎯 Key Features**

#### **SDXL Worker**
- **Batch generation**: 1, 3, or 6 images per request
- **Two quality tiers**: Fast (15 steps) and High (25 steps)
- **Reference image support**: Style, composition, and character modes
- **Performance**: 30-42s total (3-8s per image)

#### **Chat Worker**
- **Prompt enhancement**: Qwen 2.5-7B Instruct for cinematic focus
- **Memory management**: Smart loading/unloading with PyTorch 2.0 compilation
- **Admin utilities**: Memory status, model info, emergency operations
- **Performance**: 5-15s for prompt enhancement

#### **WAN 1.3B Worker**
- **Video generation**: High-quality video with temporal consistency
- **Comprehensive reference frame support**: All 5 modes (none, single, start, end, both)
- **AI enhancement**: Qwen 7B Base prompt enhancement for improved quality
- **Performance**: 25-240s depending on job type and quality

#### **Memory Manager**
- **Smart VRAM allocation**: Priority-based memory management
- **Emergency handling**: Force unload capabilities for critical situations
- **Pressure detection**: Real-time memory pressure monitoring
- **Predictive loading**: Smart preloading based on usage patterns

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
- **Storage Integration**: Supabase storage for generated content
- **Error Handling**: Comprehensive error recovery and fallback mechanisms

### **📊 Job Types**
- **SDXL**: `sdxl_image_fast`, `sdxl_image_high`
- **Chat**: `chat_enhance`, `chat_conversation`, `admin_utilities`
- **WAN Standard**: `image_fast`, `image_high`, `video_fast`, `video_high`
- **WAN Enhanced**: `image7b_fast_enhanced`, `image7b_high_enhanced`, `video7b_fast_enhanced`, `video7b_high_enhanced`

### **🧠 Memory Management Features**
| **Feature** | **Description** | **Use Case** |
|-------------|----------------|--------------|
| **Pressure Detection** | Critical/High/Medium/Low levels | Real-time monitoring |
| **Emergency Unload** | Force unload all except target | Critical situations |
| **Predictive Loading** | Smart preloading based on patterns | Performance optimization |
| **Intelligent Fallback** | Selective vs nuclear unloading | Memory pressure handling |
| **Worker Coordination** | HTTP-based memory management | Cross-worker communication |

---

## 🤝 Contributing & Support
- For issues, feature requests, or contributions, please open a GitHub issue or pull request.
- For business or technical questions, contact the maintainer.

---

**© 2025 OurVidz.com. All rights reserved.**
