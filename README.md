# ourvidz-worker

OurVidz GPU Worker for RunPod

---

## 🚀 Overview
This repository contains the **GPU worker system** for [OurVidz.com](https://ourvidz.lovable.app/), designed for high-performance AI image and video generation on RunPod infrastructure. It supports multiple AI models (SDXL, WAN 1.3B, Qwen) and is optimized for RTX 6000 ADA (48GB VRAM).

- **Production-ready**: Dual worker orchestration (SDXL + WAN 1.3B)
- **Comprehensive reference frame support**: 5 reference modes for video generation
- **Batch image & video generation**: 10 job types, NSFW-capable
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
   python dual_orchestrator.py
   ```

---

## 🛠️ Development & Testing
- **Test SDXL worker only:**
  ```bash
  python sdxl_worker.py
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

#### **WAN 1.3B Worker**
- **Video generation**: High-quality video with temporal consistency
- **Comprehensive reference frame support**: All 5 modes (none, single, start, end, both)
- **AI enhancement**: Qwen 7B prompt enhancement for improved quality
- **Performance**: 25-240s depending on job type and quality

#### **Reference Frame Support Matrix**
| **Reference Mode** | **Config Parameter** | **WAN Parameters** | **Use Case** |
|-------------------|---------------------|-------------------|--------------|
| **None** | No parameters | None | Standard T2V |
| **Single** | `config.image` | `--image ref.png` | I2V-style |
| **Start** | `config.first_frame` | `--first_frame start.png` | Start frame |
| **End** | `config.last_frame` | `--last_frame end.png` | End frame |
| **Both** | `config.first_frame` + `config.last_frame` | `--first_frame start.png --last_frame end.png` | Transition |

### **🔧 System Architecture**
- **Dual Worker Orchestrator**: Manages SDXL and WAN workers concurrently
- **Job Queue System**: Redis-based job distribution
- **Storage Integration**: Supabase storage for generated content
- **Error Handling**: Comprehensive error recovery and fallback mechanisms

### **📊 Job Types**
- **SDXL**: `sdxl_image_fast`, `sdxl_image_high`
- **WAN Standard**: `image_fast`, `image_high`, `video_fast`, `video_high`
- **WAN Enhanced**: `image7b_fast_enhanced`, `image7b_high_enhanced`, `video7b_fast_enhanced`, `video7b_high_enhanced`

---

## 🤝 Contributing & Support
- For issues, feature requests, or contributions, please open a GitHub issue or pull request.
- For business or technical questions, contact the maintainer.

---

**© 2025 OurVidz.com. All rights reserved.**
