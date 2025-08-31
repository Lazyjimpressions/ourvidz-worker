# Chat Worker Documentation

**Last Updated:** August 31, 2025  
**Status:** ✅ ACTIVE - Pure Inference Engine Architecture

## **🎯 Overview**

The Chat Worker is a **pure inference engine** for AI conversation and prompt enhancement using Qwen 2.5-7B models. It's part of the triple worker system and executes exactly what's provided by edge functions - no hardcoded prompts or business logic.

### **Key Capabilities**
- **Pure Inference Engine**: Executes exactly what edge functions provide
- **Dual Model Support**: Qwen 2.5-7B Instruct (primary) + Base (enhancement)
- **Auto-Registration**: Detects RunPod URL and registers with Supabase
- **Memory Management**: Smart loading/unloading with 15GB VRAM requirement
- **Health Monitoring**: Comprehensive status endpoints
- **Zero Content Restrictions**: No filtering or censorship

---

## **🔧 Technical Setup**

### **Model Configuration**
```python
# Qwen 2.5-7B Models Setup
MODEL_PATHS = {
    "instruct": "/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct/",
    "base": "/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B/"
}
MODEL_NAMES = {
    "instruct": "Qwen2.5-7B-Instruct",
    "base": "Qwen2.5-7B"
}
VRAM_USAGE = "15GB"  # Peak usage during generation
MAX_CONCURRENT_JOBS = 8
```

### **Hardware Requirements**
- **GPU**: NVIDIA RTX 6000 ADA (48GB VRAM)
- **Memory**: 15GB VRAM peak usage
- **Storage**: Model files (~30GB total for both models)
- **Performance**: 1-15 seconds for responses
- **Port**: 7861 (dedicated)

### **Worker Configuration**
```python
WORKER_CONFIG = {
    "chat": {
        "primary_model": "Qwen2.5-7B-Instruct",
        "secondary_model": "Qwen2.5-7B",
        "max_concurrent_jobs": 8,
        "memory_limit": 15,  # GB
        "polling_interval": 3,
        "job_types": ["chat_enhance", "chat_conversation", "chat_unrestricted", "admin_utilities"],
        "port": 7861,
        "compilation": true,
        "auto_registration": true
    }
}
```

---

## **🏗️ Pure Inference Architecture**

### **Architecture Philosophy**
**"Workers are dumb execution engines. All intelligence lives in the edge function."**

### **Key Changes (August 2025)**
- **Removed hardcoded prompts** - worker no longer contains any prompt logic
- **Template override risk eliminated** - workers cannot override database templates
- **New pure inference endpoints** (`/chat`, `/enhance`, `/generate`, `/worker/info`)
- **Enhanced logging** throughout all worker interactions
- **Memory optimization** and improved OOM handling
- **Edge function control** over all prompt construction

### **Security Benefits**
- Workers execute exactly what edge functions provide
- No possibility of prompt manipulation or overrides
- Complete audit trail of all AI interactions
- Database-driven template system with edge function control

---

## **💬 API Endpoints**

### **POST /chat** - Chat Conversation
```json
{
  "messages": [
    {
      "role": "system",
      "content": "System prompt from edge function"
    },
    {
      "role": "user",
      "content": "User message"
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "model": "qwen_instruct|qwen_base",
  "sfw_mode": false
}
```

### **POST /enhance** - Pure Enhancement Inference
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

### **POST /generate** - Generic Inference
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

### **GET /health** - Health Check
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime": 3600.5,
  "stats": {
    "requests_served": 150,
    "model_loads": 2,
    "sfw_requests": 25,
    "base_model_uses": 10,
    "instruct_model_uses": 140
  },
  "worker_type": "pure_inference_engine",
  "no_hardcoded_prompts": true
}
```

### **GET /worker/info** - Worker Information
```json
{
  "worker_type": "pure_inference_engine",
  "model": "Qwen2.5-7B-Instruct",
  "capabilities": {
    "chat": true,
    "enhancement": true,
    "generation": true,
    "hardcoded_prompts": false,
    "prompt_modification": false,
    "pure_inference": true
  },
  "models_loaded": {
    "instruct_loaded": true,
    "base_loaded": false,
    "active_model_type": "instruct"
  },
  "model_paths": {
    "instruct": "/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    "base": "/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"
  },
  "endpoints": {
    "/chat": "POST - Chat inference with messages array",
    "/enhance": "POST - Enhancement inference with messages array",
    "/generate": "POST - Generic inference with messages array",
    "/health": "GET - Health check",
    "/worker/info": "GET - This information",
    "/debug/model": "GET - Current model/debug status"
  }
}
```

### **GET /debug/model** - Model Debug Information
```json
{
  "active_model_type": "instruct",
  "instruct_loaded": true,
  "base_loaded": false,
  "paths": {
    "instruct": "/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    "base": "/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"
  },
  "devices": {
    "instruct": "cuda:0",
    "base": null
  },
  "stats": {
    "requests_served": 150,
    "model_loads": 2,
    "sfw_requests": 25,
    "base_model_uses": 10,
    "instruct_model_uses": 140
  }
}
```

### **GET /memory/status** - Memory Status
```json
{
  "total_vram": 48.0,
  "allocated_vram": 15.2,
  "available_vram": 32.8,
  "model_loaded": true,
  "instruct_loaded": true,
  "base_loaded": false,
  "instruct_device": "cuda:0"
}
```

### **POST /memory/load** - Load Model
```json
{
  "which": "instruct|base|all",
  "force": false
}
```

### **POST /memory/unload** - Unload Model
```json
{
  "which": "instruct|base|all"
}
```

---

## **🔗 Auto-Registration Process**

The Chat Worker automatically registers itself with Supabase on startup:

1. **URL Detection:** Detects `RUNPOD_POD_ID` environment variable
2. **URL Construction:** Creates `https://{pod_id}-7861.proxy.runpod.net`
3. **Registration:** Calls `register-chat-worker` edge function with:
```json
{
  "worker_url": "https://{pod_id}-7861.proxy.runpod.net",
  "auto_registered": true,
  "registration_method": "pure_inference_chat_worker",
  "worker_type": "pure_inference_engine",
  "capabilities": {
    "hardcoded_prompts": false,
    "prompt_modification": false,
    "pure_inference": true
  },
  "timestamp": "2025-08-31T10:00:00Z"
}
```

---

## **📊 Performance Characteristics**

### **Response Times**
- **Chat Enhancement:** 1-3 seconds (direct inference)
- **Chat Conversation:** 5-15 seconds (dynamic prompts)
- **Model Loading:** 15GB VRAM required for Qwen Instruct
- **Memory Management:** Automatic cleanup and validation
- **PyTorch 2.0 Compilation:** Performance optimization when available

### **Job Types**
- `chat_enhance` - Pure enhancement inference (1-3s)
- `chat_conversation` - Pure chat inference (5-15s)
- `chat_unrestricted` - Pure NSFW inference (5-15s)
- `admin_utilities` - System management (<1s)

---

## **🧠 Memory Management**

### **Model Loading Strategy**
```python
class ChatWorker:
    def __init__(self):
        self.instruct_model = None
        self.base_model = None
        self.instruct_loaded = False
        self.base_loaded = False
    
    def ensure_model_loaded(self, model_type="instruct"):
        """Ensure specified model is loaded"""
        if model_type == "instruct" and not self.instruct_loaded:
            self.instruct_model = load_qwen_instruct_model()
            self.instruct_loaded = True
        elif model_type == "base" and not self.base_loaded:
            self.base_model = load_qwen_base_model()
            self.base_loaded = True
```

### **Memory Optimization**
```python
def optimize_chat_memory():
    """Optimize memory usage for chat worker"""
    
    # Clear GPU cache between conversations
    torch.cuda.empty_cache()
    
    # Monitor VRAM usage
    vram_usage = torch.cuda.memory_allocated() / 1024**3
    if vram_usage > 12:  # GB
        gc.collect()
        torch.cuda.empty_cache()
    
    # Smart model unloading
    if not actively_using_base_model():
        unload_base_model()
```

---

## **📈 Monitoring and Logging**

### **Enhanced Logging**
```python
# Request logging
logger.info(f"🎯 Pure inference request received: {len(messages)} messages")
logger.info(f"💬 System prompt length: {len(system_prompt)} characters")
logger.info(f"👤 User prompt length: {len(user_prompt)} characters")

# Execution logging
logger.info(f"🚀 Executing pure inference with {max_tokens} max tokens")
logger.info(f"⏱️ Generation started at {start_time}")

# Response logging
logger.info(f"✅ Pure inference completed in {generation_time:.2f}s")
logger.info(f"📊 Generated {tokens_generated} tokens")
logger.info(f"🎯 Pure inference mode: no template overrides detected")

# Error logging
logger.error(f"❌ Pure inference failed: {error}")
logger.warning(f"⚠️ Fallback to original prompt due to error")
```

### **Performance Metrics**
```python
def log_chat_metrics(job_data, response_time, response_length):
    """Log chat worker performance metrics"""
    
    metrics = {
        'worker': 'pure_inference_chat',
        'job_type': job_data.get('job_type'),
        'response_time': response_time,
        'response_length': response_length,
        'conversation_length': len(job_data.get('messages', [])),
        'temperature': job_data.get('temperature'),
        'max_tokens': job_data.get('max_tokens'),
        'model_used': job_data.get('model', 'qwen_instruct'),
        'vram_usage': torch.cuda.memory_allocated() / 1024**3,
        'pure_inference': True,
        'timestamp': datetime.now().isoformat()
    }
    
    log_metrics(metrics)
```

---

## **🎯 Chat Output Specifications**

### **Response Format**
```yaml
Format: Plain text with markdown support
Max Length: 2048 tokens (configurable)
Temperature: 0.7 default (0.0-2.0 range)
Top-p: 0.9 default (0.0-1.0 range)
Response Time: 1-15 seconds typical
Model: qwen_instruct (default) or qwen_base
```

### **Pure Inference Settings**
```python
PURE_INFERENCE_SETTINGS = {
    'max_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.9,
    'model': 'qwen_instruct',
    'sfw_mode': False,
    'no_hardcoded_prompts': True,
    'edge_function_control': True
}
```

---

## **🚀 Future Enhancements**

### **Planned Improvements**
1. **Multi-character Roleplay**: Support for multiple characters in conversation
2. **Voice Integration**: Text-to-speech for roleplay responses
3. **Emotion Detection**: Detect and respond to user emotions
4. **Story Continuity**: Maintain story consistency across sessions

### **Integration Opportunities**
1. **Storyboard System**: Generate storyboards from chat conversations
2. **Character Creation**: Create character profiles from chat interactions
3. **Content Generation**: Generate images/videos based on chat scenarios

---

## **🔑 Environment Configuration**

### **Required Environment Variables**
```bash
SUPABASE_URL=              # Supabase database URL
SUPABASE_SERVICE_ROLE_KEY= # Supabase service key
UPSTASH_REDIS_REST_URL=    # Redis queue URL
UPSTASH_REDIS_REST_TOKEN=  # Redis authentication token
WAN_WORKER_API_KEY=        # API key for WAN worker authentication
HF_TOKEN=                  # Optional HuggingFace token
RUNPOD_POD_ID=             # RunPod pod ID for auto-registration
```

### **RunPod Deployment**
- **Chat Worker URL:** `https://{RUNPOD_POD_ID}-7861.proxy.runpod.net`
- **Auto-Registration:** Detects `RUNPOD_POD_ID` and registers with Supabase
- **Health Monitoring:** Continuous status tracking via `/health` endpoints

---

**Note**: This worker provides the foundation for natural conversation and roleplay capabilities with a pure inference engine architecture. The Qwen 2.5-7B Instruct model ensures high-quality, contextually appropriate responses while maintaining complete edge function control over all prompts and logic.
