# Worker API Documentation

**Last Updated:** August 18, 2025

## System Overview

The `ourvidz-worker` repository manages a **pure inference AI content generation system** with three specialized workers that execute exactly what's provided by edge functions. This document provides the frontend AI with complete context of all active workers, Python files, APIs, and system architecture.

### Architecture Philosophy
**"Workers are dumb execution engines. All intelligence lives in the edge function."**

### Architecture Components

#### Core Workers (Pure Inference)
1. **SDXL Worker** (`sdxl_worker.py`) - Pure image generation using Stable Diffusion XL with batch processing and I2I pipeline
2. **Chat Worker** (`chat_worker.py`) - Pure inference for chat and enhancement using Qwen models
3. **WAN Worker** (`wan_worker.py`) - Pure video generation using WAN 2.1 with reference frame support and I2I pipeline

#### System Management
- **Triple Orchestrator** (`dual_orchestrator.py`) - Central job distribution and worker coordination
- **Memory Manager** (`memory_manager.py`) - Intelligent VRAM management and worker coordination
- **Worker Registration** (`worker_registration.py`) - Dynamic worker discovery and registration

#### Infrastructure
- **Redis Job Queues** - `sdxl_queue`, `chat_queue`, `wan_queue` for job distribution
- **Flask HTTP APIs** - RESTful endpoints for each worker and system management
- **Callback System** - Standardized job status reporting via `POST /functions/v1/job-callback`

### File Structure

```
ourvidz-worker/
├── Core Workers/
│   ├── sdxl_worker.py          # Pure SDXL image generation
│   ├── chat_worker.py          # Pure chat and enhancement inference
│   └── wan_worker.py           # Pure WAN video generation
├── Configuration/
│   ├── worker_configs.py       # Worker configuration templates
│   ├── validation_schemas.py   # Request validation schemas
│   └── CONFIGURATION_APPROACH.md # Configuration philosophy
├── System Management/
│   ├── dual_orchestrator.py    # Central job orchestrator
│   ├── memory_manager.py       # VRAM and worker management
│   └── worker_registration.py  # Dynamic worker registration
├── Infrastructure/
│   ├── startup.sh              # System startup script
│   ├── wan_generate.py         # WAN generation utilities
│   └── requirements.txt        # Python dependencies
├── Documentation/
│   ├── README.md
│   ├── WORKER_API.md           # This file
│   ├── CODEBASE_INDEX.md
│   ├── CHAT_WORKER_CONSOLIDATED.md
│   └── CLEANUP_SUMMARY.md
└── Archive/
    └── [Historical documentation and test files]
```

## Environment and Dependencies

### **📦 Dependency Management**
**Path Structure**: Dependencies are pre-installed in persistent `/workspace` storage
```bash
/workspace/python_deps/
└── lib/
    └── python3.11/
        └── site-packages/
            ├── cv2/                    # OpenCV-Python module
            ├── torch/                   # PyTorch
            ├── transformers/            # Transformers
            ├── diffusers/              # Diffusers
            ├── flask/                  # Flask
            ├── PIL/                    # Pillow
            ├── numpy/                  # NumPy
            ├── requests/               # Requests
            └── ... (other packages)
```

**Environment Configuration**:
- `PYTHONPATH=/workspace/python_deps/lib/python3.11/site-packages`
- Set in `startup.sh` for all workers
- Persistent across pod restarts on RunPod

**Key Dependencies**:
- **OpenCV-Python 4.10.0.84**: Available for video processing and thumbnail generation
- **PyTorch 2.4.1+cu124**: Stable version for all ML operations
- **Flask 3.0.2**: HTTP APIs for worker communication
- **Pillow 10.4.0**: Image processing and thumbnail generation

**Usage in Workers**:
```python
import sys
sys.path.insert(0, "/workspace/python_deps/lib/python3.11/site-packages")
import cv2  # Available for video processing
```

## Core Workers (Pure Inference)

### Overview
The Chat Worker provides **pure inference** for AI conversation and prompt enhancement using Qwen 2.5-7B models. All system prompts and enhancement logic are provided by the edge function - the worker executes exactly what's requested.

### API Endpoints

#### Primary Chat Endpoints

**POST /chat** - Chat Conversation
```json
{
  "prompt": "User message",
  "config": {
    "system_prompt": "Optional custom system prompt",
    "job_type": "chat_conversation",
    "quality": "high"
  },
  "metadata": {
    "unrestricted_mode": false,
    "nsfw_optimization": true
  }
}
```

**POST /chat/unrestricted** - Unrestricted Mode Chat
```json
{
  "prompt": "Adult content request",
  "config": {
    "system_prompt": "NSFW-optimized system prompt",
    "job_type": "chat_unrestricted",
    "quality": "high"
  },
  "metadata": {
    "unrestricted_mode": true,
    "nsfw_optimization": true
  }
}
```

#### Enhancement Endpoints

**POST /enhance** - Pure Enhancement Inference
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Enhancement system prompt from edge function"
    },
    {
      "role": "user", 
      "content": "Original prompt to enhance"
    }
  ],
  "max_tokens": 200,
  "temperature": 0.7,
  "model": "qwen_instruct|qwen_base"
}
```

**GET /enhancement/info** - Pure Inference Info
```json
{
  "worker_type": "pure_inference_engine",
  "models_available": ["qwen_instruct", "qwen_base"],
  "capabilities": {
    "chat_conversation": true,
    "prompt_enhancement": true,
    "no_hardcoded_prompts": true,
    "pure_inference": true
  },
  "model_info": {
    "instruct_model": "Qwen2.5-7B-Instruct",
    "base_model": "Qwen2.5-7B-Base",
    "enhancement_method": "Pure inference with edge function prompts"
  }
}
```

**GET /memory/status** - Memory Status
```json
{
  "model_loaded": true,
  "memory_usage": "15GB",
  "device": "cuda:0",
  "compilation_status": "compiled"
}
```

**POST /memory/load** - Load Model
```json
{
  "force": false
}
```

**POST /memory/unload** - Unload Model
```json
{
  "force": false
}
```

### Chat Worker Pure Inference Payload
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

## SDXL Worker (Pure Inference)

### Overview
Pure image generation using Stable Diffusion XL with batch processing support (1, 3, or 6 images per request) and comprehensive Image-to-Image (I2I) pipeline. Receives complete parameters from edge function and executes exactly what's provided.

### I2I Pipeline Features
- **StableDiffusionXLImg2ImgPipeline**: First-class I2I support using dedicated pipeline
- **Two Explicit Modes**:
  - **Promptless Exact Copy**: `denoise_strength ≤ 0.05`, `guidance_scale = 1.0`, `steps = 6-10`, `negative_prompt = ''`
  - **Reference Modify**: `denoise_strength = 0.10-0.25`, `guidance_scale = 4-7`, `steps = 15-30`
- **Parameter Clamping**: Worker-side guards ensure consistent behavior
- **Backward Compatibility**: `reference_strength` automatically converted to `denoise_strength = 1 - reference_strength`

### Thumbnail Generation
- **256px WEBP Thumbnails**: Generated for each image (longest edge 256px, preserve aspect ratio)
- **Storage**: Both original and thumbnail uploaded to `workspace-temp`
- **Callback Format**: Includes both `url` and `thumbnail_url` for each asset

### API Endpoints

**GET /health** - Health Check
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0"
}
```

**GET /status** - Worker Status
```json
{
  "worker_type": "sdxl",
  "model": "lustifySDXLNSFWSFW_v20.safetensors",
  "batch_support": [1, 3, 6],
  "quality_tiers": ["fast", "high"],
  "i2i_pipeline": "StableDiffusionXLImg2ImgPipeline",
  "thumbnail_generation": true
}
```

### SDXL Pure Inference Payload
```json
{
  "id": "sdxl_job_123",
  "type": "sdxl_image_fast|sdxl_image_high",
  "prompt": "Complete prompt from edge function",
  "user_id": "user_123",
  "config": {
    "num_images": 1|3|6,
    "steps": 10-50,
    "guidance_scale": 1.0-20.0,
    "resolution": "1024x1024",
    "seed": 0-2147483647,
    "negative_prompt": "Optional negative prompt"
  },
  "metadata": {
    "reference_image_url": "Optional reference image URL",
    "denoise_strength": 0.0-1.0,
    "exact_copy_mode": false,
    "reference_type": "style|composition|character"
  },
  "compel_enabled": boolean,
  "compel_weights": "Optional Compel weights"
}
```

### I2I Generation Modes

#### Promptless Exact Copy Mode
- **Trigger**: `exact_copy_mode: true` with empty prompt
- **Parameters**:
  - `denoise_strength`: Clamped to ≤ 0.05
  - `guidance_scale`: Fixed at 1.0
  - `steps`: 6-10 (based on denoise_strength)
  - `negative_prompt`: Omitted
- **Use Case**: Upload reference image for exact copy with minimal modification

#### Reference Modify Mode
- **Trigger**: `exact_copy_mode: false` or not specified
- **Parameters**:
  - `denoise_strength`: As provided by edge function (NO CLAMPING)
  - `guidance_scale`: As provided by edge function (NO CLAMPING)
  - `steps`: As provided by edge function (NO CLAMPING)
  - `negative_prompt`: Standard quality prompts
- **Use Case**: Modify reference image with provided prompt
- **Worker Contract**: Workers respect edge function parameters completely

## WAN Worker (Pure Inference)

### Overview
Pure video generation using WAN 2.1 with comprehensive reference frame support (5 modes) and I2I pipeline. Receives complete parameters from edge function and executes exactly what's provided. Includes internal auto-enhancement for enhanced job types.

### I2I Pipeline Features
- **denoise_strength Parameter**: Replaces `reference_strength` for consistency
- **Backward Compatibility**: `reference_strength` automatically converted to `denoise_strength = 1 - reference_strength`
- **Parameter Handling**: Workers respect edge function parameters (clamping only in exact copy mode)

### Video Thumbnail Generation
- **Mid-Frame Thumbnails**: Extract middle frame of video for better representation
- **256px WEBP Format**: Longest edge 256px, preserve aspect ratio, quality 85
- **Storage**: Both original video and thumbnail uploaded to `workspace-temp`
- **Callback Format**: Includes both `url` and `thumbnail_url` for each asset

### API Endpoints

**GET /health** - Health Check
```json
{
  "status": "healthy",
  "wan_model_loaded": true,
  "qwen_model_loaded": true,
  "device": "cuda:0"
}
```

**GET /debug/env** - Environment Debug
```json
{
  "wan_generate_path": "/workspace/ourvidz-worker/wan_generate.py",
  "model_paths": {
    "wan": "/workspace/models/wan2.1-t2v-1.3b",
    "qwen": "/workspace/models/huggingface_cache"
  }
}
```

**POST /enhance** - Internal Enhancement (WAN Auto-Enhancement)
```json
{
  "prompt": "Original prompt",
  "config": {
    "job_type": "image_fast|image_high|video_fast|video_high"
  }
}
```

### WAN Pure Inference Payload
```json
{
  "id": "wan_job_123",
  "type": "image_fast|image_high|video_fast|video_high|image7b_fast_enhanced|image7b_high_enhanced|video7b_fast_enhanced|video7b_high_enhanced",
  "prompt": "Complete prompt from edge function",
  "user_id": "user_123",
  "config": {
    "width": 480,
    "height": 832,
    "frames": 1-83,
    "fps": 8-24,
    "reference_mode": "none|single|start|end|both",
    "image": "Optional single reference",
    "first_frame": "Optional start reference",
    "last_frame": "Optional end reference"
  },
  "metadata": {
    "reference_image_url": "Fallback reference URL",
    "start_reference_url": "Fallback start URL",
    "end_reference_url": "Fallback end URL",
    "denoise_strength": 0.0-1.0
  }
}
```

### Reference Frame Support
| **Reference Mode** | **Config Parameter** | **Metadata Fallback** | **WAN Parameters** | **Use Case** |
|-------------------|---------------------|----------------------|-------------------|--------------|
| **None** | No parameters | No parameters | None | Standard T2V |
| **Single** | `config.image` | `metadata.reference_image_url` | `--image ref.png` | I2V-style |
| **Start** | `config.first_frame` | `metadata.start_reference_url` | `--first_frame start.png` | Start frame |
| **End** | `config.last_frame` | `metadata.end_reference_url` | `--last_frame end.png` | End frame |
| **Both** | `config.first_frame` + `config.last_frame` | `metadata.start_reference_url` + `metadata.end_reference_url` | `--first_frame start.png --last_frame end.png` | Transition |

## Memory Manager

### Overview
Intelligent VRAM management with pressure detection, emergency unloading, and predictive loading for the triple worker system.

### API Endpoints

**GET /memory/status** - Memory Status
```json
{
  "vram_usage": {
    "total": 24576,
    "used": 18432,
    "free": 6144,
    "pressure": "medium"
  },
  "workers": {
    "sdxl_worker": "loaded",
    "chat_worker": "loaded",
    "wan_worker": "unloaded"
  },
  "memory_allocation": {
    "sdxl": "10GB (always loaded)",
    "chat": "15GB (load when possible)",
    "wan": "30GB (load on demand)"
  }
}
```

**POST /emergency/operation** - Emergency Memory Operations
```json
{
  "operation": "force_unload|predictive_load|pressure_check",
  "target_worker": "chat_worker|wan_worker",
  "force": false
}
```

**GET /memory/report** - Comprehensive Memory Report
```json
{
  "pressure_level": "medium",
  "available_vram": 6144,
  "worker_status": {
    "sdxl": {"loaded": true, "memory": 10240},
    "chat": {"loaded": true, "memory": 15360},
    "wan": {"loaded": false, "memory": 0}
  },
  "recommendations": ["unload_chat_for_wan", "monitor_pressure"]
}
```

## Triple Orchestrator

### Overview
Central job distribution system that routes jobs to appropriate workers based on job type and current system load, managing SDXL, Chat, and WAN workers concurrently.

### Job Types

| Job Type | Worker | Description | Performance |
|----------|--------|-------------|-------------|
| `sdxl_image_fast` | SDXL | Fast image generation (15 steps) | 30s total |
| `sdxl_image_high` | SDXL | High-quality image generation (25 steps) | 42s total |
| `chat_enhance` | Chat | Simple prompt enhancement | 1-3s |
| `chat_conversation` | Chat | AI conversation | 5-15s |
| `chat_unrestricted` | Chat | NSFW chat | 5-15s |
| `admin_utilities` | Chat | System management | <1s |
| `image_fast` | WAN | Fast image generation | 25-40s |
| `image_high` | WAN | High-quality image generation | 40-100s |
| `video_fast` | WAN | Fast video generation | 135-180s |
| `video_high` | WAN | High-quality video generation | 180-240s |
| `image7b_fast_enhanced` | WAN | Fast enhanced image | 85-100s |
| `image7b_high_enhanced` | WAN | High enhanced image | 100-240s |
| `video7b_fast_enhanced` | WAN | Fast enhanced video | 195-240s |
| `video7b_high_enhanced` | WAN | High enhanced video | 240+s |

## Callback Payload Format

All workers use a standardized callback format for job status updates with enhanced metadata for I2I and thumbnail support:

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
    "system_prompt_used": "Custom system prompt",
    "nsfw_optimization": true,
    "processing_time": 15.2,
    "vram_used": 8192,
    "reference_mode": "single",
    "batch_size": 1,
    "exact_copy_mode": false
  },
  "error": {
    "code": "OOM_ERROR",
    "message": "Out of memory",
    "retryable": true
  }
}
```

### Enhanced Asset Metadata Fields

#### For I2I Jobs (SDXL and WAN)
- `denoise_strength`: Float 0.0-1.0 (I2I strength parameter)
- `pipeline`: String "img2img" (indicates I2I generation)
- `resize_policy`: String "center_crop" (reference image processing)
- `negative_prompt_used`: Boolean (whether negative prompts were applied)

#### For All Assets
- `thumbnail_url`: String path to 256px WEBP thumbnail
- `file_size_bytes`: Integer file size in bytes
- `asset_index`: Integer 0-based asset index in batch

## Performance Metrics

### Chat Enhancement
- **Direct Enhancement:** 1-3 seconds
- **New Requests:** 5-15 seconds
- **Memory Management:** Smart loading/unloading

### Chat Conversation
- **Standard Mode:** 2-5 seconds
- **Unrestricted Mode:** 3-7 seconds
- **Model Compilation:** PyTorch 2.0 optimization

### SDXL Generation
- **Fast (15 steps):** 30s total (3-8s per image)
- **High (25 steps):** 42s total (5-10s per image)
- **Batch Support:** 1, 3, or 6 images per request
- **I2I Pipeline:** +2-5s for reference image processing
- **Thumbnail Generation:** +0.5-1s per image

### WAN Generation
- **Fast Images:** 25-40s
- **High Images:** 40-100s
- **Fast Videos:** 135-180s
- **High Videos:** 180-240s
- **Enhanced Variants:** +60-120s for AI enhancement
- **Video Thumbnail Generation:** +1-2s per video (mid-frame extraction)

## System Configuration

### Environment Variables
```bash
SUPABASE_URL=              # Supabase database URL
SUPABASE_SERVICE_KEY=      # Supabase service key
UPSTASH_REDIS_REST_URL=    # Redis queue URL
UPSTASH_REDIS_REST_TOKEN=  # Redis authentication token
WAN_WORKER_API_KEY=        # API key for WAN worker authentication
HF_TOKEN=                  # Optional HuggingFace token
```

### Worker Configuration
```json
{
  "sdxl_worker": {
    "model_path": "/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors",
    "max_batch_size": 6,
    "enable_xformers": true,
    "attention_slicing": "auto",
    "port": 7860
  },
  "chat_worker": {
    "primary_model": "Qwen2.5-7B-Instruct",
    "max_tokens": 2048,
    "port": 7861,
    "compilation": true
  },
  "wan_worker": {
    "model_path": "/workspace/models/wan2.1-t2v-1.3b",
    "max_frames": 83,
    "reference_modes": ["none", "single", "start", "end", "both"],
    "port": 7860
  }
}
```

## Error Handling

### Common Error Codes
- `OOM_ERROR` - Out of memory, retryable
- `MODEL_LOAD_ERROR` - Model loading failed
- `INVALID_PROMPT` - Prompt validation failed
- `WORKER_UNAVAILABLE` - Worker not loaded
- `TIMEOUT_ERROR` - Request timeout
- `REFERENCE_FRAME_ERROR` - Reference frame processing failed
- `I2I_PIPELINE_ERROR` - Image-to-image pipeline error
- `THUMBNAIL_GENERATION_ERROR` - Thumbnail generation failed

### Retry Logic
- **OOM Errors:** Automatic retry with memory cleanup
- **Network Errors:** 3 retries with exponential backoff
- **Model Errors:** Single retry with model reload
- **Reference Frame Errors:** Graceful fallback to standard generation
- **I2I Errors:** Fallback to text-to-image generation
- **Thumbnail Errors:** Continue without thumbnail (non-critical)

## Security and NSFW Features

### NSFW Optimization
- **Zero Content Restrictions:** No filtering or censorship
- **Anatomical Accuracy:** Professional quality standards
- **Adult Content Support:** Explicit support across all workers
- **Unrestricted Mode:** Dedicated endpoints for adult content
- **Dynamic System Prompts:** Context-aware NSFW handling

### Security Features
- **Input Validation:** Comprehensive prompt validation
- **Rate Limiting:** Per-user and per-endpoint limits
- **Authentication:** Optional API key support
- **Logging:** Comprehensive audit trails
- **Memory Safety:** Emergency memory management

## Integration Guide

### Frontend Integration
1. **Job Submission:** Send jobs to appropriate worker endpoints
2. **Status Monitoring:** Poll callback endpoint for job status
3. **Asset Retrieval:** Download generated assets from callback URLs
4. **Error Handling:** Implement retry logic for transient errors
5. **Memory Management:** Monitor memory status for optimal performance

### API Client Example
```python
import requests

# Submit chat job
response = requests.post("http://worker:7861/chat", json={
    "prompt": "User message",
    "config": {"job_type": "chat_conversation"},
    "metadata": {"user_id": "user_123"}
})

# Monitor job status
job_id = response.json()["job_id"]
status = requests.get(f"http://orchestrator:8002/job/{job_id}")

# Check memory status
memory = requests.get("http://memory-manager:8001/memory/status")
```

## Python Files Overview

### Core Worker Files
- **`sdxl_worker.py`**: SDXL image generation with batch processing, I2I pipeline, and thumbnail generation
- **`chat_worker.py`**: Enhanced chat and prompt enhancement system with Qwen Instruct
- **`wan_worker.py`**: WAN video and image processing with comprehensive reference frame support, I2I pipeline, and video thumbnail generation

### System Management Files
- **`dual_orchestrator.py`**: Central job distribution and triple worker coordination
- **`memory_manager.py`**: Intelligent VRAM management and worker coordination
- **`worker_registration.py`**: Dynamic worker discovery and RunPod URL registration

### Infrastructure Files
- **`startup.sh`**: System startup and initialization script
- **`requirements.txt`**: Python dependencies and versions
- **`wan_generate.py`**: WAN generation utilities and command-line interface

### Documentation Files
- **`README.md`**: Project overview and quick start guide
- **`CODEBASE_INDEX.md`**: Comprehensive system architecture and component overview
- **`CHAT_WORKER_CONSOLIDATED.md`**: Enhanced chat worker features and NSFW optimization
- **`CLEANUP_SUMMARY.md`**: Codebase cleanup and organization summary

## Edge Function Requirements Summary

### **Pure Inference Architecture**
All workers are designed as pure inference engines. The edge function must provide:

#### **SDXL Worker Requirements**
- **Complete Parameters:** All generation parameters (steps, guidance_scale, batch_size, etc.)
- **Enhanced Prompts:** Pre-enhanced prompts from Chat Worker
- **Reference Images:** Downloaded and processed reference images
- **I2I Parameters:** `denoise_strength` (0.0-1.0) and `exact_copy_mode` (boolean)
- **User Validation:** Permissions and content restrictions
- **Parameter Conversion:** Frontend presets → worker parameters

#### **Chat Worker Requirements**
- **System Prompts:** Complete system prompts for all contexts
- **Message Arrays:** Properly formatted message arrays
- **Model Selection:** Base vs Instruct model choice
- **User Validation:** Chat permissions and restrictions
- **Enhancement Context:** Target model and enhancement type

#### **WAN Worker Requirements**
- **Complete Parameters:** All video generation parameters
- **Enhanced Prompts:** Pre-enhanced prompts from Chat Worker
- **Reference Frames:** Downloaded and processed reference images
- **I2I Parameters:** `denoise_strength` (0.0-1.0) for reference frame processing
- **User Validation:** Video generation permissions
- **Parameter Conversion:** Frontend presets → worker parameters

### **Frontend Integration**
- **Parameter Validation:** Validate all parameters before sending to edge function
- **Callback Processing:** Handle different asset types and thumbnail URLs appropriately
- **Error Handling:** Implement retry logic for transient failures
- **Progress Tracking:** Display processing time and queue position
- **Thumbnail Display:** Use `thumbnail_url` for grid views and previews

### **Configuration Files**
- **Configuration/CONFIGURATION_APPROACH.md** - Complete edge function requirements and validation rules
- **Configuration/worker_configs.py** - Worker configuration templates
- **Configuration/validation_schemas.py** - Request validation schemas
- **I2I_THUMBNAIL_IMPLEMENTATION_SUMMARY.md** - I2I pipeline and thumbnail generation details

This documentation provides the frontend AI with complete context of the ourvidz-worker system architecture, all active workers, Python files, APIs, and integration patterns after the August 18, 2025 I2I pipeline and thumbnail generation implementation. 