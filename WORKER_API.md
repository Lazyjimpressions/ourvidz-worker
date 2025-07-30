# OurVidz Worker API Reference

**Last Updated:** July 30, 2025 at 2:30 PM CST  
**Status:** ✅ Production Ready - Triple Worker System (SDXL + Chat + WAN) with Smart Memory Management  
**System:** Triple Worker (SDXL + Chat + WAN) on RTX 6000 ADA (48GB VRAM)

---

## **🎯 Worker System Overview**

OurVidz operates with a triple-worker architecture managed by a centralized orchestrator:

1. **SDXL Worker** - High-quality image generation with flexible quantities and Compel integration
2. **Chat Worker** - Dedicated Qwen Instruct service for prompt enhancement and chat interface
3. **WAN Worker** - Video generation and enhanced image processing with Qwen 7B enhancement
4. **Memory Manager** - Smart VRAM allocation and coordination for all workers
5. **Triple Orchestrator** - Centralized management and monitoring of all workers

All workers use standardized callback parameters and comprehensive metadata management.

---

## **📤 Job Queue System**

### **Queue Structure**
- **`sdxl_queue`** - SDXL image generation jobs
- **`chat_queue`** - Chat enhancement and conversation jobs
- **`wan_queue`** - WAN video and enhanced image jobs

### **Job Payload Format (Standardized)**

#### **SDXL Job Payload**
```json
{
  "id": "uuid",
  "type": "sdxl_image_fast" | "sdxl_image_high",
  "prompt": "string",
  "user_id": "uuid",
  "compel_enabled": true | false,
  "compel_weights": "(beautiful:1.3), (woman:1.2), (garden:1.1)",
  "config": {
    "num_images": 1 | 3 | 6,
    "seed": 123456789
  },
  "metadata": {
    "reference_image_url": "string",
    "reference_type": "style" | "composition" | "character",
    "reference_strength": 0.1-1.0
  }
}
```

#### **Enhanced Chat Job Payload**
```json
{
  "id": "uuid",
  "type": "chat_enhance" | "chat_conversation" | "chat_unrestricted" | "admin_utilities",
  "prompt": "string",
  "user_id": "uuid",
  "config": {
    "enhancement_type": "manual" | "cinematic" | "custom",
    "conversation_context": "string",
    "system_prompt": "string",
    "job_type": "sdxl_image_fast" | "sdxl_image_high" | "video_fast" | "video_high",
    "quality": "fast" | "high"
  },
  "metadata": {
    "session_id": "string",
    "memory_management": "auto" | "manual",
    "model_preference": "qwen_instruct",
    "unrestricted_mode": true | false,
    "nsfw_optimization": true
  }
}
```

#### **WAN Job Payload**
```json
{
  "id": "uuid",
  "type": "image_fast" | "image_high" | "video_fast" | "video_high" | "image7b_fast_enhanced" | "image7b_high_enhanced" | "video7b_fast_enhanced" | "video7b_high_enhanced",
  "prompt": "string",
  "user_id": "uuid",
  "config": {
    "first_frame": "string",
    "last_frame": "string"
  },
  "metadata": {
    "start_reference_url": "string",
    "end_reference_url": "string",
    "reference_strength": 0.1-1.0,
    "enhancement_type": "base" | "chat" | "instruct_chat",
    "session_id": "string",
    "conversation_context": "string"
  }
}
```

---

## **📥 Callback System (Standardized)**

### **Callback Endpoint**
```
POST /functions/v1/job-callback
```

### **Callback Payload Format (Standardized)**
```json
{
  "job_id": "uuid",
  "status": "processing" | "completed" | "failed",
  "assets": ["url1", "url2", "url3"],
  "error_message": "string",
  "metadata": {
    "seed": 123456789,
    "generation_time": 15.5,
    "num_images": 3,
    "compel_enabled": true,
    "compel_weights": "(beautiful:1.3), (woman:1.2)",
    "enhancement_strategy": "compel" | "fallback" | "none",
    "enhancement_type": "base" | "chat" | "instruct_chat",
    "enhancement_success": true,
    "enhancement_time": 2.5,
    "original_prompt": "string",
    "enhanced_prompt": "string",
    "enhancement_source": "edge_function" | "worker_fallback" | "emergency_fallback",
    "quality_score": 0.85,
    "worker_optimizations": {
      "caching": true,
      "post_processing": true,
      "fallback_ready": true
    },
    "unrestricted_mode": true | false,
    "system_prompt_used": true | false,
    "nsfw_optimization": true,
    "memory_pressure": "low" | "medium" | "high" | "critical",
    "worker_used": "sdxl" | "chat" | "wan"
  }
}
```

---

## **🌐 API Endpoints**

### **🎨 SDXL Worker (Port 7860)**

#### **Health Check**
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "worker": "sdxl",
  "uptime": 3600,
  "model_loaded": true,
  "memory_usage": "10.2GB"
}
```

#### **Status**
```http
GET /status
```
**Response:**
```json
{
  "worker": "sdxl",
  "status": "active",
  "jobs_processed": 150,
  "current_job": null,
  "model_info": {
    "name": "lustifySDXLNSFWSFW_v20",
    "loaded": true,
    "memory_usage": "10.2GB"
  }
}
```

### **💬 Enhanced Chat Worker (Port 7861)**

#### **Health Check**
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime": 1800,
  "stats": {
    "requests_served": 45,
    "model_loads": 1,
    "model_unloads": 0
  },
  "enhancement_system": "active",
  "nsfw_optimization": "enabled"
}
```

#### **Chat Conversation**
```http
POST /chat
Content-Type: application/json

{
  "message": "Tell me a story about a magical forest",
  "system_prompt": "You are a creative storyteller specializing in fantasy tales.",
  "conversation_id": "story_001"
}
```
**Response:**
```json
{
  "success": true,
  "response": "Once upon a time, in a mystical forest...",
  "generation_time": 2.3,
  "conversation_id": "story_001",
  "context_type": "general",
  "message_id": "msg_1234567890",
  "system_prompt_used": true,
  "unrestricted_mode": false
}
```

#### **Unrestricted Mode Chat**
```http
POST /chat/unrestricted
Content-Type: application/json

{
  "message": "Help me create adult content for my project",
  "conversation_id": "adult_002"
}
```
**Response:**
```json
{
  "success": true,
  "response": "I can help you create high-quality adult content...",
  "generation_time": 3.1,
  "conversation_id": "adult_002",
  "context_type": "unrestricted",
  "message_id": "msg_1234567891",
  "system_prompt_used": true,
  "unrestricted_mode": true
}
```

#### **Intelligent Prompt Enhancement**
```http
POST /enhance
Content-Type: application/json

{
  "prompt": "beautiful woman",
  "job_type": "sdxl_image_fast",
  "quality": "fast",
  "enhancement_type": "manual"
}
```
**Response:**
```json
{
  "success": true,
  "original_prompt": "beautiful woman",
  "enhanced_prompt": "masterpiece, best quality, ultra detailed, beautiful woman, professional photography, detailed, photorealistic, realistic proportions, anatomical accuracy",
  "generation_time": 1.23,
  "enhancement_type": "manual",
  "job_type": "sdxl_image_fast",
  "quality": "fast",
  "enhancement_source": "worker_fallback",
  "worker_optimizations": {
    "caching": true,
    "post_processing": true,
    "fallback_ready": true
  },
  "quality_score": 0.85,
  "sdxl_optimizations": {
    "has_quality_tags": true,
    "has_lighting": true,
    "has_technical_terms": true,
    "has_resolution": true,
    "has_anatomical_accuracy": true,
    "token_count": 75
  }
}
```

#### **Enhancement System Info**
```http
GET /enhancement/info
```
**Response:**
```json
{
  "enhancement_system": "active",
  "supported_job_types": ["sdxl_image_fast", "sdxl_image_high", "video_fast", "video_high"],
  "cache_size": 45,
  "model_status": "loaded",
  "nsfw_optimization": "enabled"
}
```

#### **Clear Enhancement Cache**
```http
POST /enhancement/cache/clear
```
**Response:**
```json
{
  "success": true,
  "cache_cleared": 45,
  "cache_size": 0
}
```

#### **Memory Status**
```http
GET /memory/status
```
**Response:**
```json
{
  "total_vram": 48.0,
  "allocated_vram": 15.2,
  "available_vram": 32.8,
      "model_loaded": true,
  "model_device": "cuda:0",
  "device_type": "cuda"
}
```

#### **Model Information**
```http
GET /model/info
```
**Response:**
```json
{
  "model_loaded": true,
  "model_path": "/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct",
  "model_device": "cuda:0",
  "device_type": "cuda",
  "model_parameters": 7250000000,
  "model_size_gb": 13.5,
  "is_eval_mode": true,
  "torch_version": "2.4.1",
  "cuda_available": true,
  "cuda_version": "12.4",
  "gpu_name": "NVIDIA RTX 6000 Ada Generation"
}
```

#### **Memory Load**
```http
POST /memory/load
```
**Response:**
```json
{
  "success": true,
  "message": "Model loaded successfully"
}
```

#### **Memory Unload**
```http
POST /memory/unload
```
**Response:**
```json
{
  "success": true,
  "message": "Model unloaded"
}
```

### **🎬 WAN Worker (Port 7860)**

#### **Health Check**
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "worker": "wan",
  "uptime": 7200,
  "model_loaded": true,
  "qwen_loaded": true,
  "thread_safe_timeouts": true
}
```

#### **Debug Environment**
```http
GET /debug/env
```
**Response:**
```json
{
  "worker_type": "wan",
  "environment_variables": {
    "WAN_WORKER_API_KEY": "7a23fcd9-05be-4cc0-b03e-0b2197f4d214",
    "SUPABASE_URL": "https://...",
    "UPSTASH_REDIS_REST_URL": "https://...",
    "RUNPOD_POD_ID": "8gt0qj6wuj7jub"
  },
  "model_paths": {
    "wan_model": "/workspace/models/wan2.1-t2v-1.3b",
    "qwen_model": "/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B"
    }
}
```

#### **Prompt Enhancement**
```http
POST /enhance
Content-Type: application/json
Authorization: Bearer 7a23fcd9-05be-4cc0-b03e-0b2197f4d214

{
  "prompt": "a beautiful sunset",
  "model": "qwen_base"
}
```
**Response:**
```json
{
  "success": true,
  "original_prompt": "a beautiful sunset",
  "enhanced_prompt": "a beautiful sunset",
  "enhancement_source": "qwen_base",
  "model": "qwen_base",
  "processing_time": 0.0001
}
```

### **🧠 Memory Manager**

#### **Memory Status**
```http
GET /memory/status
```
**Response:**
```json
{
  "memory_pressure": "medium",
  "total_used_gb": 25.2,
  "available_gb": 21.8,
  "worker_status": {
    "sdxl": true,
    "chat": true,
    "wan": false
  },
  "can_load_wan": true,
  "can_load_chat": true,
  "emergency_actions_available": {
    "force_unload_chat": true,
    "force_unload_sdxl": true,
    "force_unload_all_except_wan": false,
    "force_unload_all_except_chat": false
  }
}
```

#### **Emergency Memory Operation**
```http
POST /emergency/operation
Content-Type: application/json

{
  "operation": "handle_emergency_request",
  "target_worker": "wan",
  "reason": "wan_job_failed"
}
```
**Response:**
```json
{
  "success": true,
  "action_taken": "unload_chat",
  "reason": "high_pressure_wan_job_failed",
  "status": {
    "memory_pressure": "medium",
    "total_used_gb": 10.2,
    "available_gb": 36.8,
    "can_load_wan": true
  }
}
```

#### **Memory Report**
```http
GET /memory/report
```
**Response:**
```json
{
  "total_vram": 49.0,
  "usable_vram": 47.0,
  "safety_buffer": 2.0,
  "current_usage": 25.2,
  "available": 21.8,
  "memory_pressure": "medium",
  "worker_status": {
    "sdxl": true,
    "chat": true,
    "wan": false
  },
  "worker_memory_requirements": {
    "sdxl": 10,
    "chat": 15,
    "wan": 30
  },
  "can_load": {
    "sdxl": true,
    "chat": true,
    "wan": true
  },
  "emergency_actions": {
    "force_unload_all_except_wan": false,
    "force_unload_all_except_chat": false,
    "should_preload_chat": true
  }
}
```

---

## **🎯 Job Types Reference**

### **SDXL Jobs**
| Job Type | Quality | Steps | Time | Resolution | Batch Size | Features |
|----------|---------|-------|------|------------|------------|----------|
| `sdxl_image_fast` | Fast | 15 | 30s | 1024x1024 | 1,3,6 | Quick preview |
| `sdxl_image_high` | High | 25 | 42s | 1024x1024 | 1,3,6 | Final quality |

### **Enhanced Chat Jobs**
| Job Type | Purpose | Model | Time | Features |
|----------|---------|-------|------|----------|
| `chat_enhance` | Intelligent prompt enhancement | Qwen Instruct | 1-3s | Edge function integration, caching, NSFW optimization |
| `chat_conversation` | Dynamic chat interface | Qwen Instruct | 5-15s | Custom system prompts, unrestricted mode detection |
| `chat_unrestricted` | Dedicated NSFW chat | Qwen Instruct | 5-15s | Adult content optimization, anatomical accuracy |
| `admin_utilities` | System management | N/A | <1s | Memory status, enhancement info |

### **WAN Jobs**
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

---

## **🖼️ Reference Frame Support**

### **Reference Frame Modes**
| **Mode** | **Config Parameter** | **Metadata Fallback** | **WAN Parameters** | **Use Case** |
|----------|---------------------|----------------------|-------------------|--------------|
| **None** | No parameters | No parameters | None | Standard T2V |
| **Single** | `config.image` | `metadata.reference_image_url` | `--image ref.png` | I2V-style |
| **Start** | `config.first_frame` | `metadata.start_reference_url` | `--first_frame start.png` | Start frame |
| **End** | `config.last_frame` | `metadata.end_reference_url` | `--last_frame end.png` | End frame |
| **Both** | `config.first_frame` + `config.last_frame` | `metadata.start_reference_url` + `metadata.end_reference_url` | `--first_frame start.png --last_frame end.png` | Transition |

### **Reference Frame Example**
```json
{
  "id": "uuid",
  "type": "video7b_high_enhanced",
  "prompt": "a woman walking through a garden",
  "config": {
    "first_frame": "https://example.com/start.jpg",
    "last_frame": "https://example.com/end.jpg"
  },
    "metadata": {
    "start_reference_url": "https://example.com/start.jpg",
    "end_reference_url": "https://example.com/end.jpg",
    "reference_strength": 0.8
  }
}
```

---

## **🧠 Memory Management**

### **Memory Pressure Levels**
| **Level** | **Available VRAM** | **Action** | **Description** |
|-----------|-------------------|------------|-----------------|
| **Critical** | <5GB | Nuclear unload | Emergency measures required |
| **High** | 5-10GB | Selective unloading | Try chat unload first |
| **Medium** | 10-15GB | Normal operations | Monitor closely |
| **Low** | >15GB | Preloading opportunities | Optimize for performance |

### **Emergency Operations**
| **Operation** | **Description** | **Use Case** |
|---------------|----------------|--------------|
| `force_unload_all_except` | Nuclear option | Critical memory pressure |
| `handle_emergency_request` | Intelligent fallback | Smart memory management |
| `get_emergency_memory_status` | Status check | Emergency monitoring |

### **Memory Management Example**
```json
{
  "operation": "handle_emergency_request",
  "target_worker": "wan",
  "reason": "wan_job_failed"
}
```

---

## **🔧 Error Handling**

### **Common Error Responses**
```json
{
  "success": false,
  "error": "Model not available",
  "enhanced_prompt": "original_prompt"
}
```

### **Memory Pressure Response**
```json
{
  "success": false,
  "error": "Insufficient VRAM for model loading",
  "memory_pressure": "critical",
  "available_vram": 3.2
}
```

### **Timeout Response**
```json
{
  "success": false,
  "error": "Operation timed out after 25 seconds",
  "timeout_seconds": 25
}
```

---

## **📊 Performance Metrics**

### **Response Time Benchmarks**
| **Operation** | **Typical Time** | **Peak Time** | **Notes** |
|---------------|------------------|---------------|-----------|
| SDXL Fast (1 image) | 30s | 42s | 15 steps |
| SDXL High (1 image) | 42s | 60s | 25 steps |
| Chat Enhancement (cached) | 1-3s | 5s | Intelligent enhancement |
| Chat Enhancement (new) | 5-15s | 20s | Qwen Instruct |
| Chat Conversation | 5-15s | 20s | Dynamic prompts |
| Chat Unrestricted | 5-15s | 20s | NSFW optimization |
| WAN Fast Image | 25-40s | 60s | No enhancement |
| WAN High Image | 40-100s | 120s | No enhancement |
| WAN Fast Video | 135-180s | 240s | 83 frames |
| WAN High Video | 180-240s | 300s | 83 frames |
| WAN Enhanced | +60s | +120s | Qwen enhancement |

### **Memory Usage**
| **Worker** | **Base Memory** | **Peak Memory** | **Notes** |
|------------|-----------------|-----------------|-----------|
| SDXL | 10GB | 12GB | Always loaded |
| Chat | 15GB | 18GB | Load when possible |
| WAN | 30GB | 35GB | Load on demand |

---

## **🚀 Production Deployment**

### **Environment Variables**
```bash
# Required
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key
UPSTASH_REDIS_REST_URL=your_redis_url
UPSTASH_REDIS_REST_TOKEN=your_redis_token
WAN_WORKER_API_KEY=your_api_key

# Optional
HF_TOKEN=your_huggingface_token
RUNPOD_POD_ID=auto_detected
```

### **Startup Sequence**
1. **Setup**: `./setup.sh` - Environment preparation
2. **Startup**: `./startup.sh` - Production startup
3. **Priority Order**: SDXL (1) → Chat (2) → WAN (3)
4. **Auto-registration**: WAN worker registers URLs
5. **Monitoring**: Continuous health checks

### **Health Monitoring**
- **SDXL**: Port 7860 `/health`
- **Chat**: Port 7861 `/health`
- **WAN**: Port 7860 `/health`
- **Memory**: `/memory/status` endpoints

---

## **📋 Integration Examples**

### **Frontend Integration**
```javascript
// Chat Worker Enhancement
const enhancePrompt = async (prompt) => {
  const response = await fetch('https://worker-url:7861/enhance', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, enhancement_type: 'manual' })
  });
  return response.json();
};

// WAN Worker Enhancement
const enhancePromptWAN = async (prompt) => {
  const response = await fetch('https://worker-url:7860/enhance', {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      'Authorization': 'Bearer your_api_key'
    },
    body: JSON.stringify({ prompt, model: 'qwen_base' })
  });
  return response.json();
};
```

### **Memory Management Integration**
```javascript
// Check memory status
const checkMemory = async () => {
  const response = await fetch('https://worker-url:7861/memory/status');
  return response.json();
};

// Emergency memory operation
const emergencyMemory = async (targetWorker, reason) => {
  const response = await fetch('/emergency/operation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      operation: 'handle_emergency_request',
      target_worker: targetWorker,
      reason: reason
    })
  });
  return response.json();
};
```

---

This API reference provides comprehensive documentation for the OurVidz triple worker system, including all endpoints, job types, memory management, and integration examples. The system is designed for high-performance AI content generation with smart resource management and comprehensive error handling.