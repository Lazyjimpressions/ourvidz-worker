# Worker API Documentation

**Last Updated:** July 30, 2025

## System Overview

The `ourvidz-worker` repository manages a comprehensive AI content generation system with three specialized workers orchestrated by a central memory management system. This document provides the frontend AI with complete context of all active workers, Python files, APIs, and system architecture.

### Architecture Components

#### Core Workers
1. **SDXL Worker** (`sdxl_worker.py`) - High-quality image generation using Stable Diffusion XL
2. **Enhanced Chat Worker** (`chat_worker.py`) - AI conversation and intelligent prompt enhancement using Qwen Instruct
3. **WAN Worker** (`wan_worker.py`) - Video generation and enhanced image processing using WAN 2.1

#### System Management
- **Dual Orchestrator** (`dual_orchestrator.py`) - Central job distribution and worker coordination
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
│   └── requirements.txt        # Python dependencies
├── Testing/
│   ├── chat_worker_validator.py
│   ├── comprehensive_test.sh
│   └── quick_health_check.sh
└── Documentation/
    ├── README.md
    ├── WORKER_API.md           # This file
    ├── CODEBASE_INDEX.md
    └── CHAT_WORKER_CONSOLIDATED.md
```

## Enhanced Chat Worker

### Overview
The Enhanced Chat Worker provides AI conversation capabilities and intelligent prompt enhancement with NSFW optimization, dynamic system prompts, and performance optimization.

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
    "job_type": "sdxl_lustify|wan_2.1_video",
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
  "supported_job_types": ["sdxl_image_fast", "sdxl_image_high", "video_fast", "video_high"],
  "model_info": {
    "model_name": "Qwen2.5-7B-Instruct",
    "model_loaded": true,
    "enhancement_method": "Direct Qwen Instruct with dynamic prompts"
  }
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
    "job_type": "chat_conversation|chat_enhance|chat_unrestricted",
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
High-quality image generation using Stable Diffusion XL with NSFW optimization and anatomical accuracy focus.

### API Endpoints

**POST /sdxl/generate** - Image Generation
```json
{
  "prompt": "Enhanced prompt",
  "config": {
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance_scale": 7.5,
    "seed": 42
  },
  "metadata": {
    "nsfw_optimization": true,
    "anatomical_accuracy": true
  }
}
```

### SDXL Job Payload
```json
{
  "worker_id": "sdxl_worker_001",
  "job_id": "sdxl_job_123",
  "prompt": "Enhanced prompt",
  "config": {
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance_scale": 7.5,
    "seed": 42,
    "negative_prompt": "Optional negative prompt"
  },
  "metadata": {
    "nsfw_optimization": true,
    "anatomical_accuracy": true,
    "user_id": "user_123"
  }
}
```

## WAN Worker

### Overview
Video generation and enhanced image processing using WAN 2.1 with reference frame support and NSFW optimization.

### API Endpoints

**POST /wan/generate** - Video/Image Generation
```json
{
  "prompt": "Enhanced prompt",
  "config": {
    "job_type": "video|image",
    "width": 1024,
    "height": 1024,
    "frames": 24,
    "reference_mode": "none|single|start|end|both",
    "reference_image": "base64_encoded_image"
  },
  "metadata": {
    "nsfw_optimization": true,
    "anatomical_accuracy": true
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
    "job_type": "video|image",
    "width": 1024,
    "height": 1024,
    "frames": 24,
    "reference_mode": "none|single|start|end|both",
    "reference_image": "base64_encoded_image",
    "fps": 24
  },
  "metadata": {
    "nsfw_optimization": true,
    "anatomical_accuracy": true,
    "user_id": "user_123"
  }
}
```

## Memory Manager

### Overview
Intelligent VRAM management with pressure detection, emergency unloading, and predictive loading.

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
  }
}
```

**POST /memory/unload** - Unload Worker
```json
{
  "worker_id": "sdxl_worker_001",
  "force": false
}
```

**POST /memory/load** - Load Worker
```json
{
  "worker_id": "sdxl_worker_001",
  "priority": "high"
}
```

## Dual Orchestrator

### Overview
Central job distribution system that routes jobs to appropriate workers based on job type and current system load.

### Job Types

| Job Type | Worker | Description |
|----------|--------|-------------|
| `sdxl_generate` | SDXL | Image generation |
| `chat_conversation` | Chat | AI conversation |
| `chat_enhance` | Chat | Prompt enhancement |
| `chat_unrestricted` | Chat | NSFW chat |
| `wan_video` | WAN | Video generation |
| `wan_image` | WAN | Enhanced image generation |

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
        "format": "png"
      }
    }
  ],
  "metadata": {
    "enhancement_source": "qwen_instruct",
    "unrestricted_mode": false,
    "system_prompt_used": "Custom system prompt",
    "nsfw_optimization": true,
    "processing_time": 15.2,
    "vram_used": 8192
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

### Chat Conversation
- **Standard Mode:** 2-5 seconds
- **Unrestricted Mode:** 3-7 seconds

### SDXL Generation
- **1024x1024:** 15-30 seconds
- **512x512:** 8-15 seconds

### WAN Generation
- **Video (24 frames):** 60-120 seconds
- **Image:** 20-40 seconds

## System Configuration

### Environment Variables
```bash
REDIS_URL=redis://localhost:6379
WORKER_PORT=8000
MEMORY_MANAGER_PORT=8001
ORCHESTRATOR_PORT=8002
MODEL_CACHE_DIR=/models
ASSET_CACHE_DIR=/assets
```

### Worker Configuration
```json
{
  "sdxl_worker": {
    "model_path": "/models/sdxl",
    "max_batch_size": 1,
    "enable_xformers": true,
    "attention_slicing": "auto"
  },
  "chat_worker": {
    "primary_model": "/models/qwen-instruct",
    "max_tokens": 2048
  },
  "wan_worker": {
    "model_path": "/models/wan-2.1",
    "max_frames": 48,
    "reference_modes": ["none", "single", "start", "end", "both"]
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

### Retry Logic
- **OOM Errors:** Automatic retry with memory cleanup
- **Network Errors:** 3 retries with exponential backoff
- **Model Errors:** Single retry with model reload

## Security and NSFW Features

### NSFW Optimization
- **Zero Content Restrictions:** No filtering or censorship
- **Anatomical Accuracy:** Professional quality standards
- **Adult Content Support:** Explicit support across all workers
- **Unrestricted Mode:** Dedicated endpoints for adult content

### Security Features
- **Input Validation:** Comprehensive prompt validation
- **Rate Limiting:** Per-user and per-endpoint limits
- **Authentication:** Optional API key support
- **Logging:** Comprehensive audit trails

## Integration Guide

### Frontend Integration
1. **Job Submission:** Send jobs to appropriate worker endpoints
2. **Status Monitoring:** Poll callback endpoint for job status
3. **Asset Retrieval:** Download generated assets from callback URLs
4. **Error Handling:** Implement retry logic for transient errors

### API Client Example
```python
import requests

# Submit chat job
response = requests.post("http://worker:8000/chat", json={
    "prompt": "User message",
    "config": {"job_type": "chat_conversation"},
    "metadata": {"user_id": "user_123"}
})

# Monitor job status
job_id = response.json()["job_id"]
status = requests.get(f"http://orchestrator:8002/job/{job_id}")
```

## Python Files Overview

### Core Worker Files
- **`sdxl_worker.py`**: SDXL image generation with NSFW optimization
- **`chat_worker.py`**: Enhanced chat and prompt enhancement system
- **`wan_worker.py`**: WAN video and image processing with reference frames

### System Management Files
- **`dual_orchestrator.py`**: Central job distribution and coordination
- **`memory_manager.py`**: Intelligent VRAM management and worker coordination
- **`worker_registration.py`**: Dynamic worker discovery and registration

### Infrastructure Files
- **`startup.sh`**: System startup and initialization script
- **`requirements.txt`**: Python dependencies and versions
- **`wan_generate.py`**: WAN generation utilities

### Testing Files
- **`testing/chat_worker_validator.py`**: Chat worker validation tests
- **`testing/comprehensive_test.sh`**: Comprehensive system testing
- **`testing/quick_health_check.sh`**: Quick health check script

This documentation provides the frontend AI with complete context of the ourvidz-worker system architecture, all active workers, Python files, APIs, and integration patterns. 