# OurVidz Worker System Summary

**Last Updated:** August 16, 2025  
**Purpose:** Frontend API reference for current system structure and capabilities

---

## 🎯 **System Overview**

OurVidz Worker is a **pure inference triple-worker AI content generation system** optimized for RTX 6000 ADA (48GB VRAM). Workers execute exactly what's provided by edge functions - all intelligence lives in the edge function layer.

### **Architecture Philosophy**
**"Workers are dumb execution engines. All intelligence lives in the edge function."**

### **Core Capabilities**
- **14 Job Types** across 3 pure inference workers
- **Batch Image Generation** (1, 3, or 6 images per request)
- **5 Reference Frame Modes** for video generation
- **Pure Enhancement Inference** with Qwen 2.5-7B models
- **Zero Content Restrictions** with anatomical accuracy
- **Smart Memory Management** with emergency handling

---

## 🚀 **Active Workers & Capabilities**

### **🎨 SDXL Worker** (`sdxl_worker.py`)
**Port:** 7860 (shared with WAN)  
**Model:** LUSTIFY SDXL (`lustifySDXLNSFWSFW_v20.safetensors`)

**Job Types:**
- `sdxl_image_fast` - 15 steps, 30s total (3-8s per image)
- `sdxl_image_high` - 25 steps, 42s total (5-10s per image)

**Key Features:**
- ✅ **Batch Processing:** 1, 3, or 6 images per request
- ✅ **Reference Image Support:** Style, composition, character modes
- ✅ **NSFW Optimization:** Zero content restrictions
- ✅ **Memory Efficient:** Attention slicing + xformers

**API Endpoints:**
- `GET /health` - Worker health check
- `GET /status` - Model and batch support info

---

### **💬 Chat Worker** (`chat_worker.py`)
**Port:** 7861 (dedicated)  
**Models:** Qwen 2.5-7B Instruct + Base

**Job Types:**
- `chat_enhance` - Pure enhancement inference (1-3s)
- `chat_conversation` - Pure chat inference (5-15s)
- `chat_unrestricted` - Pure NSFW inference (5-15s)
- `admin_utilities` - System management (<1s)

**Key Features:**
- ✅ **Pure Inference:** Executes exactly what's provided
- ✅ **No Hardcoded Prompts:** All prompts from edge function
- ✅ **Dual Model Support:** Instruct and Base models
- ✅ **Memory Management:** Smart loading/unloading
- ✅ **PyTorch 2.0 Compilation:** Performance optimization

**API Endpoints:**
- `POST /chat` - Standard chat conversation
- `POST /chat/unrestricted` - NSFW chat
- `POST /enhance` - Prompt enhancement
- `GET /enhancement/info` - Enhancement system info
- `GET /memory/status` - Model memory status
- `POST /memory/load` - Load model
- `POST /memory/unload` - Unload model

---

### **🎬 WAN Worker** (`wan_worker.py`)
**Port:** 7860 (shared with SDXL)  
**Models:** WAN 2.1 T2V 1.3B + Qwen 2.5-7B Base

**Job Types:**
- **Standard:** `image_fast` (25-40s), `image_high` (40-100s), `video_fast` (135-180s), `video_high` (180-240s)
- **Enhanced:** `image7b_fast_enhanced` (85-100s), `image7b_high_enhanced` (100-240s), `video7b_fast_enhanced` (195-240s), `video7b_high_enhanced` (240+s)

**Key Features:**
- ✅ **Pure Inference:** Executes exactly what's provided
- ✅ **5 Reference Frame Modes:** none, single, start, end, both
- ✅ **Internal Auto-Enhancement:** Qwen Base for enhanced job types
- ✅ **Thread-Safe Timeouts:** Concurrent.futures implementation
- ✅ **Memory Management:** Model unloading capabilities

**API Endpoints:**
- `GET /health` - Worker health check
- `GET /debug/env` - Environment debug info
- `POST /enhance` - Prompt enhancement

---

## 🧠 **System Management**

### **🎭 Triple Orchestrator** (`dual_orchestrator.py`)
**Purpose:** Central job distribution and worker coordination

**Key Features:**
- **Priority-based startup:** SDXL (1) → Chat (2) → WAN (3)
- **Graceful validation:** Environment and model readiness checks
- **Automatic restart:** Worker failure recovery
- **Production logging:** Comprehensive monitoring

### **🧠 Memory Manager** (`memory_manager.py`)
**Purpose:** Intelligent VRAM allocation and coordination

**Key Features:**
- **Pressure Detection:** Critical/High/Medium/Low levels
- **Emergency Operations:** Force unload capabilities
- **Predictive Loading:** Smart preloading based on patterns
- **Worker Coordination:** HTTP-based memory management

**Memory Allocation:**
- **SDXL:** 10GB (Always loaded)
- **Chat:** 15GB (Load when possible)
- **WAN:** 30GB (Load on demand)

**API Endpoints:**
- `GET /memory/status` - Current memory status
- `POST /emergency/operation` - Emergency memory operations
- `GET /memory/report` - Comprehensive memory report

### **🔧 Worker Registration** (`worker_registration.py`)
**Purpose:** Automatic RunPod URL management

**Key Features:**
- **Auto-Detection:** RunPod environment detection
- **URL Registration:** Dynamic worker URL management
- **Heartbeat Monitoring:** Worker status tracking
- **Graceful Shutdown:** Proper cleanup on termination

---

## 🛠️ **Infrastructure Files**

### **🚀 Startup Script** (`startup.sh`)
**Purpose:** Production system initialization

**Key Features:**
- **Triple worker validation:** Model readiness checks
- **Environment configuration:** Path and dependency setup
- **Memory assessment:** VRAM availability analysis
- **Cache allocation:** Smart temp storage management

### **🎬 WAN Generation** (`wan_generate.py`)
**Purpose:** Core WAN 2.1 generation script

**Key Features:**
- **Command-line interface:** Direct WAN model integration
- **Reference frame support:** All 5 modes implementation
- **Quality configuration:** Fast/high tier settings
- **Performance optimization:** Thread-safe operations

### **📦 Dependencies** (`requirements.txt`)
**Core Dependencies:**
- PyTorch 2.4.1+cu124 ecosystem
- Diffusers 0.31.0, Transformers 4.45.2
- xformers 0.0.28.post2 (memory optimization)
- Flask (API endpoints)
- psutil (system monitoring)

---

## 📊 **Job Type Matrix**

| **Worker** | **Job Type** | **Quality** | **Time** | **Features** |
|------------|--------------|-------------|----------|--------------|
| **SDXL** | `sdxl_image_fast` | Fast (15 steps) | 30s | Batch: 1,3,6 images |
| **SDXL** | `sdxl_image_high` | High (25 steps) | 42s | Batch: 1,3,6 images |
| **Chat** | `chat_enhance` | Standard | 1-3s | Direct Qwen enhancement |
| **Chat** | `chat_conversation` | Standard | 5-15s | Dynamic prompts |
| **Chat** | `chat_unrestricted` | NSFW | 5-15s | Adult content optimized |
| **Chat** | `admin_utilities` | System | <1s | Memory management |
| **WAN** | `image_fast` | Fast | 25-40s | Reference frames |
| **WAN** | `image_high` | High | 40-100s | Reference frames |
| **WAN** | `video_fast` | Fast | 135-180s | Reference frames |
| **WAN** | `video_high` | High | 180-240s | Reference frames |
| **WAN** | `image7b_fast_enhanced` | Fast Enhanced | 85-100s | AI enhancement |
| **WAN** | `image7b_high_enhanced` | High Enhanced | 100-240s | AI enhancement |
| **WAN** | `video7b_fast_enhanced` | Fast Enhanced | 195-240s | AI enhancement |
| **WAN** | `video7b_high_enhanced` | High Enhanced | 240+s | AI enhancement |

---

## 🔄 **Reference Frame Support**

| **Mode** | **Config Parameter** | **Metadata Fallback** | **Use Case** |
|----------|---------------------|----------------------|--------------|
| **None** | No parameters | No parameters | Standard T2V |
| **Single** | `config.image` | `metadata.reference_image_url` | I2V-style |
| **Start** | `config.first_frame` | `metadata.start_reference_url` | Start frame |
| **End** | `config.last_frame` | `metadata.end_reference_url` | End frame |
| **Both** | `config.first_frame` + `config.last_frame` | `metadata.start_reference_url` + `metadata.end_reference_url` | Transition |

---

## 🔑 **Environment Configuration**

### **Required Variables**
```bash
SUPABASE_URL=              # Database URL
SUPABASE_SERVICE_KEY=      # Service key
UPSTASH_REDIS_REST_URL=    # Redis queue URL
UPSTASH_REDIS_REST_TOKEN=  # Redis token
WAN_WORKER_API_KEY=        # WAN worker API key
HF_TOKEN=                  # HuggingFace token (optional)
```

### **Port Configuration**
- **SDXL Worker:** 7860 (shared with WAN)
- **Chat Worker:** 7861 (dedicated)
- **WAN Worker:** 7860 (shared with SDXL)

---

## 🛡️ **Error Handling & Recovery**

### **Error Types**
- `OOM_ERROR` - Out of memory (retryable)
- `MODEL_LOAD_ERROR` - Model loading failed
- `INVALID_PROMPT` - Prompt validation failed
- `WORKER_UNAVAILABLE` - Worker not loaded
- `TIMEOUT_ERROR` - Request timeout
- `REFERENCE_FRAME_ERROR` - Reference processing failed

### **Recovery Mechanisms**
- **OOM Errors:** Automatic retry with memory cleanup
- **Network Errors:** 3 retries with exponential backoff
- **Model Errors:** Single retry with model reload
- **Reference Frame Errors:** Graceful fallback to standard generation

---

## 🚀 **Production Startup**

### **Startup Sequence**
1. **Environment Validation:** PyTorch/CUDA version checks
2. **Model Readiness:** SDXL, WAN, Qwen model verification
3. **Memory Assessment:** VRAM availability analysis
4. **Worker Launch:** Priority-based startup (SDXL → Chat → WAN)
5. **Auto-Registration:** RunPod URL detection and registration
6. **Health Monitoring:** Continuous worker status tracking

### **Startup Command**
```bash
./startup.sh
```

---

## 📋 **Integration Points**

### **Frontend Integration**
- **Job Submission:** Send to appropriate worker endpoints
- **Status Monitoring:** Poll callback endpoint for job status
- **Asset Retrieval:** Download from callback URLs
- **Memory Management:** Monitor memory status for optimization
- **Error Handling:** Implement retry logic for transient errors

### **Callback Format**
```json
{
  "job_id": "job_123",
  "worker_id": "worker_001",
  "status": "completed|failed|processing",
  "assets": [{"type": "image|video|text", "url": "..."}],
  "metadata": {
    "enhancement_source": "qwen_instruct",
    "unrestricted_mode": false,
    "processing_time": 15.2,
    "reference_mode": "none",
    "batch_size": 1
  }
}
```

---

## 📚 **Documentation Structure**

- **SYSTEM_SUMMARY.md** - This file (Frontend API reference)
- **WORKER_API.md** - Detailed API specifications and examples
- **CODEBASE_INDEX.md** - Comprehensive system architecture
- **Configuration/CONFIGURATION_APPROACH.md** - Configuration philosophy and edge function requirements
- **CLEANUP_SUMMARY.md** - Codebase organization and cleanup details

---

**🎯 This summary provides the frontend API with complete context of the current OurVidz Worker system structure, capabilities, and integration patterns.**
