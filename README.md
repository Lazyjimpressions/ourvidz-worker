# ourvidz-worker

OurVidz GPU Worker for RunPod

---

## 🚀 Overview
This repository contains the **GPU worker system** for [OurVidz.com](https://ourvidz.lovable.app/), designed for high-performance AI image and video generation on RunPod infrastructure. It supports multiple AI models (SDXL, WAN, Qwen) and is optimized for RTX 6000 ADA (48GB VRAM).

- **Production-ready**: Dual worker orchestration (SDXL + WAN)
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

Full technical and business documentation is available in [`docs/README.md`](docs/README.md):
- System architecture
- Worker deployment & configuration
- Edge function/API reference
- Business context & job types
- Changelog & lessons learned

---

## 🤝 Contributing & Support
- For issues, feature requests, or contributions, please open a GitHub issue or pull request.
- For business or technical questions, contact the maintainer.

---

**© 2025 OurVidz.com. All rights reserved.**
