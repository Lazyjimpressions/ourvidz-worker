# OurVidz Worker System Summary

**Last Updated:** August 31, 2025  
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
- **I2I Pipeline** with parameter clamping and two modes
- **Thumbnail Generation** for images and videos
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
- ✅ **I2I Pipeline:** First-class support using StableDiffusionXLImg2ImgPipeline
- ✅ **Two I2I Modes:**
  - **Promptless Exact Copy:** `denoise_strength ≤ 0.05`, `guidance_scale = 1.0`, `steps = 6-10`
  - **Reference Modify:** `denoise_strength = 0.10-0.25`, `guidance_scale = 4-7`, `steps = 15-30`
- ✅ **Parameter Clamping:** Worker-side guards ensure consistent behavior
- ✅ **Thumbnail Generation:** 256px WEBP thumbnails for each image
- ✅ **Reference Image Support:** Style, composition, character modes
- ✅ **NSFW Optimization:** Zero content restrictions
- ✅ **Memory Efficient:** Attention slicing + xformers
- ✅ **Enhanced Error Handling:** Comprehensive traceback logging
- ✅ **Correct Callback Format:** Uses `url` field for asset paths

**Performance:**
- **Standard Generation:** 30-42s total (3-8s per image)
- **I2I Processing:** +2-5s for reference image processing
- **Thumbnail Generation:** +0.5-1s per image

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
- ✅ **Pure Inference Engine:** Executes exactly what's provided by edge functions
- ✅ **No Hardcoded Prompts:** All system prompts come from edge function layer
- ✅ **Dual Model Support:** Qwen 2.5-7B Instruct (primary) + Base (enhancement)
- ✅ **Memory Management:** Smart loading/unloading with 15GB VRAM requirement
- ✅ **PyTorch 2.0 Compilation:** Performance optimization when available
- ✅ **Auto-Registration:** Detects RunPod URL and registers with Supabase
- ✅ **Health Monitoring:** Comprehensive status endpoints

**Model Architecture:**
- **Qwen 2.5-7B Instruct:** Primary model for chat and enhancement (safety-tuned)
- **Qwen 2.5-7B Base:** Secondary model for enhanced jobs (no extra safety)
- **Model Paths:** `/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct/`
- **Device Management:** Automatic CUDA device allocation and pinning

**API Endpoints:**
- `POST /chat` - Standard chat conversation with messages array
- `POST /enhance` - Prompt enhancement inference
- `POST /generate` - Generic inference endpoint
- `GET /health` - Health check with uptime and stats
- `GET /worker/info` - Worker capabilities and model status
- `GET /debug/model` - Model loading and device information
- `GET /memory/status` - VRAM usage and model status
- `POST /memory/load` - Force load specific model
- `POST /memory/unload` - Force unload models

**Pure Inference Payload Format:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "System prompt from edge function"
    },
    {
      "role": "user",
      "content": "User input"
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "model": "qwen_instruct|qwen_base",
  "sfw_mode": false
}
```

**Auto-Registration Process:**
- Detects `RUNPOD_POD_ID` environment variable
- Constructs URL: `https://{pod_id}-7861.proxy.runpod.net`
- Registers with Supabase via `register-chat-worker` edge function
- Includes capabilities: `pure_inference: true`, `hardcoded_prompts: false`

**Performance:**
- **Chat Enhancement:** 1-3 seconds (direct inference)
- **Chat Conversation:** 5-15 seconds (dynamic prompts)
- **Model Loading:** 15GB VRAM required for Qwen Instruct
- **Memory Management:** Automatic cleanup and validation

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
- ✅ **I2I Pipeline:** `denoise_strength` parameter replaces `reference_strength` for consistency
- ✅ **Video Thumbnail Generation:** Mid-frame extraction for better representation, 256px WEBP format
- ✅ **Internal Auto-Enhancement:** Qwen Base for enhanced job types
- ✅ **Thread-Safe Timeouts:** Concurrent.futures implementation
- ✅ **Memory Management:** Model unloading capabilities

**Performance:**
- **Standard Generation:** 25-240s depending on job type and quality
- **Video Thumbnail Generation:** +1-2s per video (mid-frame extraction)

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
**Purpose**: Python dependencies with specific versions for stable operation
- **Key Dependencies**:
  - PyTorch 2.4.1+cu124 (stable version)
  - Diffusers 0.31.0 (SDXL support)
  - Transformers 4.45.2 (Qwen models)
  - Flask 3.0.2 (HTTP APIs)
  - **OpenCV-Python 4.10.0.84** (video processing and thumbnail generation)
  - Pillow 10.4.0 (image processing and thumbnail generation)
  - WAN 2.1 dependencies (easydict, av, decord, etc.)

**Dependency Management**:
- **Path**: `/workspace/python_deps/lib/python3.11/site-packages/`
- **Environment**: `PYTHONPATH` set in `startup.sh`
- **Persistence**: Pre-installed in `/workspace` volume, survives pod restarts
- **OpenCV-Python**: Available at `/workspace/python_deps/lib/python3.11/site-packages/cv2/`

---

## 📊 **Job Type Matrix**

| **Worker** | **Job Type** | **Quality** | **Time** | **Features** |
|------------|--------------|-------------|----------|--------------|
| **SDXL** | `sdxl_image_fast` | Fast (15 steps) | 30s | Batch: 1,3,6 images, I2I, Thumbnails |
| **SDXL** | `sdxl_image_high` | High (25 steps) | 42s | Batch: 1,3,6 images, I2I, Thumbnails |
| **Chat** | `chat_enhance` | Standard | 1-3s | Direct Qwen enhancement |
| **Chat** | `chat_conversation` | Standard | 5-15s | Dynamic prompts |
| **Chat** | `chat_unrestricted` | NSFW | 5-15s | Adult content optimized |
| **Chat** | `admin_utilities` | System | <1s | Memory management |
| **WAN** | `image_fast` | Fast | 25-40s | Reference frames, I2I, Thumbnails |
| **WAN** | `image_high` | High | 40-100s | Reference frames, I2I, Thumbnails |
| **WAN** | `video_fast` | Fast | 135-180s | Reference frames, I2I, Thumbnails |
| **WAN** | `video_high` | High | 180-240s | Reference frames, I2I, Thumbnails |
| **WAN** | `image7b_fast_enhanced` | Fast Enhanced | 85-100s | AI enhancement, I2I, Thumbnails |
| **WAN** | `image7b_high_enhanced` | High Enhanced | 100-240s | AI enhancement, I2I, Thumbnails |
| **WAN** | `video7b_fast_enhanced` | Fast Enhanced | 195-240s | AI enhancement, I2I, Thumbnails |
| **WAN** | `video7b_high_enhanced` | High Enhanced | 240+s | AI enhancement, I2I, Thumbnails |

---

## 🔄 **I2I Pipeline Support**

### **SDXL I2I Pipeline**
- **Pipeline:** StableDiffusionXLImg2ImgPipeline
- **Parameter:** `denoise_strength` (0.0-1.0)
- **Backward Compatibility:** `reference_strength` automatically converted to `denoise_strength = 1 - reference_strength`

### **I2I Generation Modes**

#### **Promptless Exact Copy Mode**
- **Trigger:** `exact_copy_mode: true` with empty prompt
- **Parameters:**
  - `denoise_strength`: Clamped to ≤ 0.05
  - `guidance_scale`: Fixed at 1.0
  - `steps`: 6-10 (based on denoise_strength)
  - `negative_prompt`: Omitted
- **Use Case:** Upload reference image for exact copy with minimal modification

#### **Reference Modify Mode**
- **Trigger:** `exact_copy_mode: false` or not specified
- **Parameters:**
  - `denoise_strength`: As provided by edge function (NO CLAMPING)
  - `guidance_scale`: As provided by edge function (NO CLAMPING)
  - `steps`: As provided by edge function (NO CLAMPING)
  - `negative_prompt`: Standard quality prompts
- **Use Case:** Modify reference image with provided prompt
- **Worker Contract:** Workers respect edge function parameters completely

### **WAN I2I Pipeline**
- **Pipeline:** WAN 2.1 T2V 1.3B with reference frame support
- **Parameter:** `denoise_strength` (0.0-1.0)
- **Backward Compatibility:** `reference_strength` automatically converted to `denoise_strength = 1 - reference_strength`

---

## 🖼️ **Thumbnail Generation**

### **Thumbnail Generation Matrix**
| **Asset Type** | **Thumbnail Source** | **Format** | **Size** | **Quality** | **Storage** |
|----------------|---------------------|------------|----------|-------------|-------------|
| **SDXL Images** | Generated image | WEBP | 256px longest edge | 85 | workspace-temp |
| **WAN Images** | Generated image | WEBP | 256px longest edge | 85 | workspace-temp |
| **WAN Videos** | Mid-frame extraction | WEBP | 256px longest edge | 85 | workspace-temp |

### **Thumbnail Features**
- **256px WEBP Format:** Longest edge 256px, preserve aspect ratio, quality 85
- **Video Mid-Frame:** Extract middle frame for better representation
- **Storage:** Both original and thumbnail uploaded to `workspace-temp`
- **Callback Format:** Includes both `url` and `thumbnail_url` for each asset

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

### **RunPod Deployment**
- **Chat Worker URL:** `https://{RUNPOD_POD_ID}-7861.proxy.runpod.net`
- **Auto-Registration:** Detects `RUNPOD_POD_ID` and registers with Supabase
- **Health Monitoring:** Continuous status tracking via `/health` endpoints

---

## 🛡️ **Error Handling & Recovery**

### **Error Types**
- `OOM_ERROR` - Out of memory (retryable)
- `MODEL_LOAD_ERROR` - Model loading failed
- `INVALID_PROMPT` - Prompt validation failed
- `WORKER_UNAVAILABLE` - Worker not loaded
- `TIMEOUT_ERROR` - Request timeout
- `REFERENCE_FRAME_ERROR` - Reference processing failed
- `I2I_PIPELINE_ERROR` - Image-to-image pipeline error
- `THUMBNAIL_GENERATION_ERROR` - Thumbnail generation failed

### **Recovery Mechanisms**
- **OOM Errors:** Automatic retry with memory cleanup
- **Network Errors:** 3 retries with exponential backoff
- **Model Errors:** Single retry with model reload
- **Reference Frame Errors:** Graceful fallback to standard generation
- **I2I Errors:** Fallback to text-to-image generation
- **Thumbnail Errors:** Continue without thumbnail (non-critical)

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
- **Thumbnail Display:** Use `thumbnail_url` for grid views and previews
- **Memory Management:** Monitor memory status for optimization
- **Error Handling:** Implement retry logic for transient errors

### **Enhanced Callback Format**
```json
{
  "job_id": "job_123",
  "worker_id": "worker_001",
  "status": "completed|failed|processing",
  "assets": [
    {
      "type": "image|video|text",
      "url": "workspace-temp/user123/job123/0.png",
      "thumbnail_url": "workspace-temp/user123/job123/0.thumb.webp",
      "metadata": {
        "width": 1024,
        "height": 1024,
        "format": "png",
        "batch_size": 1,
        "steps": 25,
        "guidance_scale": 7.5,
        "seed": 12345,
        "file_size_bytes": 2048576,
        "asset_index": 0,
        "denoise_strength": 0.15,
        "pipeline": "img2img",
        "resize_policy": "center_crop",
        "negative_prompt_used": true
      }
    }
  ],
  "metadata": {
    "enhancement_source": "qwen_instruct",
    "unrestricted_mode": false,
    "processing_time": 15.2,
    "reference_mode": "single",
    "batch_size": 1,
    "exact_copy_mode": false
  }
}
```

### **Enhanced Asset Metadata Fields**

#### **For I2I Jobs (SDXL and WAN)**
- `denoise_strength`: Float 0.0-1.0 (I2I strength parameter)
- `pipeline`: String "img2img" (indicates I2I generation)
- `resize_policy`: String "center_crop" (reference image processing)
- `negative_prompt_used`: Boolean (whether negative prompts were applied)

#### **For All Assets**
- `thumbnail_url`: String path to 256px WEBP thumbnail
- `file_size_bytes`: Integer file size in bytes
- `asset_index`: Integer 0-based asset index in batch

---

## 📚 **Documentation Structure**

- **SYSTEM_SUMMARY.md** - This file (Frontend API reference)
- **WORKER_API.md** - Detailed API specifications and examples
- **CODEBASE_INDEX.md** - Comprehensive system architecture
- **Configuration/CONFIGURATION_APPROACH.md** - Configuration philosophy and edge function requirements
- **I2I_THUMBNAIL_IMPLEMENTATION_SUMMARY.md** - I2I pipeline and thumbnail generation details
- **CLEANUP_SUMMARY.md** - Codebase organization and cleanup details

---

**🎯 This summary provides the frontend API with complete context of the current OurVidz Worker system structure, capabilities, and integration patterns including I2I pipeline, thumbnail generation, and pure inference chat worker architecture as of August 31, 2025.**
