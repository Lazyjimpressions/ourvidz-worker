# Worker API Documentation

**Last Updated:** August 16, 2025

## System Overview

The `ourvidz-worker` repository manages a comprehensive AI content generation system with three specialized workers orchestrated by a central memory management system. This document provides the frontend AI with complete context of all active workers, Python files, APIs, and system architecture.

### Architecture Components

#### Core Workers
1. **SDXL Worker** (`sdxl_worker.py`) - High-quality image generation using Stable Diffusion XL with batch processing
2. **Enhanced Chat Worker** (`chat_worker.py`) - AI conversation and intelligent prompt enhancement using Qwen Instruct
3. **WAN Worker** (`wan_worker.py`) - Video generation and enhanced image processing using WAN 2.1 with comprehensive reference frame support

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
│   ├── sdxl_worker.py          # SDXL image generation worker
│   ├── chat_worker.py          # Enhanced chat and prompt enhancement
│   └── wan_worker.py           # WAN video and image processing
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

## Enhanced Chat Worker

### Overview
The Enhanced Chat Worker provides AI conversation capabilities and intelligent prompt enhancement with NSFW optimization, dynamic system prompts, and performance optimization using Qwen 2.5-7B Instruct.

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

**POST /enhance** - Simple Prompt Enhancement
```json
{
  "prompt": "Original prompt",
  "config": {
    "job_type": "sdxl_image_fast|sdxl_image_high|image_fast|image_high|video_fast|video_high",
    "quality": "high|medium|low"
  },
  "metadata": {
    "nsfw_optimization": true,
    "anatomical_accuracy": true
  }
}
```

**GET /enhancement/info** - Enhancement System Info
```json
{
  "enhancement_system": "Direct Qwen Instruct Enhancement",
  "supported_job_types": ["sdxl_image_fast", "sdxl_image_high", "image_fast", "image_high", "video_fast", "video_high"],
  "model_info": {
    "model_name": "Qwen2.5-7B-Instruct",
    "model_loaded": true,
    "enhancement_method": "Direct Qwen Instruct with dynamic prompts"
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

### Enhanced Chat Job Payload
```json
{
  "worker_id": "chat_worker_001",
  "job_id": "chat_job_123",
  "prompt": "User input",
  "config": {
    "system_prompt": "Custom system prompt",
    "job_type": "chat_conversation|chat_enhance|chat_unrestricted|admin_utilities",
    "quality": "high|medium|low"
  },
  "metadata": {
    "unrestricted_mode": false,
    "nsfw_optimization": true,
    "user_id": "user_123",
    "session_id": "session_456"
  }
}
```

## SDXL Worker

### Overview
High-quality image generation using Stable Diffusion XL with NSFW optimization, anatomical accuracy focus, and batch processing support (1, 3, or 6 images per request).

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
  "quality_tiers": ["fast", "high"]
}
```

### SDXL Job Payload
```json
{
  "worker_id": "sdxl_worker_001",
  "job_id": "sdxl_job_123",
  "prompt": "Enhanced prompt",
  "config": {
    "job_type": "sdxl_image_fast|sdxl_image_high",
    "width": 1024,
    "height": 1024,
    "steps": 15,
    "guidance_scale": 7.5,
    "seed": 42,
    "negative_prompt": "Optional negative prompt",
    "batch_size": 1
  },
  "metadata": {
    "nsfw_optimization": true,
    "anatomical_accuracy": true,
    "user_id": "user_123",
    "reference_image_url": "Optional reference image"
  }
}
```

## WAN Worker

### Overview
Video generation and enhanced image processing using WAN 2.1 with comprehensive reference frame support (5 modes), AI prompt enhancement, and NSFW optimization.

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

**POST /enhance** - Prompt Enhancement
```json
{
  "prompt": "Original prompt",
  "config": {
    "job_type": "image_fast|image_high|video_fast|video_high"
  }
}
```

### WAN Job Payload
```json
{
  "worker_id": "wan_worker_001",
  "job_id": "wan_job_123",
  "prompt": "Enhanced prompt",
  "config": {
    "job_type": "image_fast|image_high|video_fast|video_high|image7b_fast_enhanced|image7b_high_enhanced|video7b_fast_enhanced|video7b_high_enhanced",
    "width": 480,
    "height": 832,
    "frames": 83,
    "reference_mode": "none|single|start|end|both",
    "reference_image": "base64_encoded_image",
    "fps": 24
  },
  "metadata": {
    "nsfw_optimization": true,
    "anatomical_accuracy": true,
    "user_id": "user_123",
    "reference_image_url": "Optional reference image URL",
    "start_reference_url": "Optional start frame URL",
    "end_reference_url": "Optional end frame URL"
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

All workers use a standardized callback format for job status updates:

```json
{
  "job_id": "job_123",
  "worker_id": "worker_001",
  "status": "completed|failed|processing",
  "assets": [
    {
      "type": "image|video|text",
      "url": "https://cdn.example.com/asset.jpg",
      "metadata": {
        "width": 1024,
        "height": 1024,
        "format": "png",
        "batch_size": 1
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
    "reference_mode": "none",
    "batch_size": 1
  },
  "error": {
    "code": "OOM_ERROR",
    "message": "Out of memory",
    "retryable": true
  }
}
```

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

### WAN Generation
- **Fast Images:** 25-40s
- **High Images:** 40-100s
- **Fast Videos:** 135-180s
- **High Videos:** 180-240s
- **Enhanced Variants:** +60-120s for AI enhancement

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

### Retry Logic
- **OOM Errors:** Automatic retry with memory cleanup
- **Network Errors:** 3 retries with exponential backoff
- **Model Errors:** Single retry with model reload
- **Reference Frame Errors:** Graceful fallback to standard generation

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
- **`sdxl_worker.py`**: SDXL image generation with batch processing and NSFW optimization
- **`chat_worker.py`**: Enhanced chat and prompt enhancement system with Qwen Instruct
- **`wan_worker.py`**: WAN video and image processing with comprehensive reference frame support

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

This documentation provides the frontend AI with complete context of the ourvidz-worker system architecture, all active workers, Python files, APIs, and integration patterns after the August 16, 2025 cleanup. 