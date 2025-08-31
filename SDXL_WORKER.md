# SDXL Worker Documentation

**Last Updated:** August 31, 2025  
**Status:** ✅ ACTIVE - Pure Inference Engine with I2I Pipeline and Compel Integration

## **🎯 Overview**

The SDXL Worker is a **pure inference engine** for image generation using the LUSTIFY SDXL model with flexible batch processing (1, 3, or 6 images per request), comprehensive Image-to-Image (I2I) pipeline, and SDXL-specific Compel library integration. It receives complete parameters from edge functions and executes exactly what's provided.

### **Key Capabilities**
- **Pure Inference Engine**: Executes exactly what edge functions provide
- **Flexible Batch Processing**: User-selected quantities (1, 3, or 6 images)
- **I2I Pipeline**: First-class support using StableDiffusionXLImg2ImgPipeline
- **Compel Integration**: SDXL-specific prompt weighting with prompt_embeds and pooled_prompt_embeds
- **Thumbnail Generation**: 256px WEBP thumbnails for each image
- **Reference Image Support**: Style, composition, character modes with exact copy mode
- **NSFW Optimization**: Zero content restrictions
- **Memory Efficient**: Attention slicing + xformers
- **Seed Control**: Reproducible generation for character consistency

---

## **🔧 Technical Setup**

### **Model Configuration**
```python
# SDXL Lustify Model Setup
MODEL_PATH = "/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors"
MODEL_NAME = "lustifySDXLNSFWSFW_v20.safetensors"
VRAM_USAGE = "10GB"  # Always loaded
MAX_CONCURRENT_JOBS = 2
BATCH_SUPPORT = [1, 3, 6]  # User-selected quantities
```

### **Hardware Requirements**
- **GPU**: NVIDIA RTX 6000 ADA (48GB VRAM)
- **Memory**: 10GB VRAM dedicated to SDXL
- **Storage**: Model files (~6.5GB)
- **Performance**: 1 image: 3-8s, 3 images: 9-24s, 6 images: 18-48s
- **Port**: 7860 (shared with WAN)

### **Worker Configuration**
```python
WORKER_CONFIG = {
    "sdxl": {
        "model_path": "/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors",
        "max_batch_size": 6,
        "enable_xformers": true,
        "attention_slicing": "auto",
        "port": 7860,
        "max_concurrent_jobs": 2,
        "memory_limit": 10,  # GB
        "polling_interval": 2,
        "job_types": ["sdxl_image_fast", "sdxl_image_high"],
        "supports_flexible_quantities": True
    }
}
```

---

## **🎨 I2I Pipeline Features**

### **I2I Pipeline Implementation**
- **StableDiffusionXLImg2ImgPipeline**: First-class I2I support using dedicated pipeline
- **Two Explicit Modes**:
  - **Promptless Exact Copy**: `denoise_strength ≤ 0.05`, `guidance_scale = 1.0`, `steps = 6-10`, `negative_prompt = ''`
  - **Reference Modify**: `denoise_strength = 0.10-0.25`, `guidance_scale = 4-7`, `steps = 15-30`
- **Parameter Clamping**: Worker-side guards ensure consistent behavior
- **Backward Compatibility**: `reference_strength` automatically converted to `denoise_strength = 1 - reference_strength`
- **Reference Image Processing**: Automatic download, preprocessing, and center-crop resizing

### **I2I Generation Modes**

#### **Promptless Exact Copy Mode**
- **Trigger**: `exact_copy_mode: true` with empty prompt
- **Parameters**:
  - `denoise_strength`: Clamped to ≤ 0.05
  - `guidance_scale`: Fixed at 1.0
  - `steps`: 6-10 (based on denoise_strength)
  - `negative_prompt`: Omitted
- **Use Case**: Upload reference image for exact copy with minimal modification

#### **Reference Modify Mode**
- **Trigger**: `exact_copy_mode: false` or not specified
- **Parameters**:
  - `denoise_strength`: As provided by edge function (NO CLAMPING)
  - `guidance_scale`: As provided by edge function (NO CLAMPING)
  - `steps`: As provided by edge function (NO CLAMPING)
  - `negative_prompt`: Standard quality prompts
- **Use Case**: Modify reference image with provided prompt
- **Worker Contract**: Workers respect edge function parameters completely

---

## **🌱 Compel Integration**

### **SDXL-Specific Compel Library**
- **Library**: `compel` with SDXL-specific integration
- **Tensors**: `prompt_embeds`, `pooled_prompt_embeds`, `negative_prompt_embeds`, `negative_pooled_prompt_embeds`
- **Usage**: Simple string concatenation with weight syntax
- **Fallback**: Automatic fallback to string prompts when Compel not enabled

### **Compel Usage Examples**
```python
# Example 1: Basic Compel enhancement
job_data = {
    "id": "job-123",
    "type": "sdxl_image_high",
    "prompt": "beautiful woman in garden",
    "user_id": "user-123",
    "compel_enabled": True,
    "compel_weights": "(beautiful:1.3), (woman:1.2), (garden:1.1)",
    "config": {
        "num_images": 1
    }
}

# Example 2: Compel with image-to-image
job_data = {
    "id": "job-124", 
    "type": "sdxl_image_fast",
    "prompt": "portrait of a person",
    "user_id": "user-123",
    "compel_enabled": True,
    "compel_weights": "(portrait:1.4), (person:1.1)",
    "config": {
        "num_images": 3
    },
    "metadata": {
        "reference_image_url": "https://example.com/reference.jpg",
        "denoise_strength": 0.7,
        "exact_copy_mode": false
    }
}

# Example 3: No Compel (standard generation)
job_data = {
    "id": "job-125",
    "type": "sdxl_image_high", 
    "prompt": "landscape painting",
    "user_id": "user-123",
    "compel_enabled": False,
    "config": {
        "num_images": 6
    }
}
```

---

## **🖼️ Thumbnail Generation**

### **Thumbnail Features**
- **256px WEBP Thumbnails**: Generated for each image (longest edge 256px, preserve aspect ratio)
- **Storage**: Both original and thumbnail uploaded to `workspace-temp`
- **Callback Format**: Includes both `url` and `thumbnail_url` for each asset
- **Quality**: WEBP format with quality 85 for optimal file size

---

## **💬 API Endpoints**

### **GET /health** - Health Check
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0"
}
```

### **GET /status** - Worker Status
```json
{
  "worker_type": "sdxl",
  "model": "lustifySDXLNSFWSFW_v20.safetensors",
  "batch_support": [1, 3, 6],
  "quality_tiers": ["fast", "high"],
  "i2i_pipeline": "StableDiffusionXLImg2ImgPipeline",
  "thumbnail_generation": true,
  "compel_integration": true
}
```

---

## **📋 Pure Inference Payload**

### **SDXL Job Payload**
```json
{
  "id": "sdxl_job_123",
  "type": "sdxl_image_fast|sdxl_image_high",
  "prompt": "Complete prompt from edge function",
  "user_id": "user_123",
  "config": {
    "num_images": 1|3|6,
    "num_inference_steps": 10-50,
    "guidance_scale": 1.0-20.0,
    "width": 1024,
    "height": 1024,
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
  "compel_weights": "Optional Compel weights string"
}
```

### **Enhanced Callback Format**
```json
{
  "job_id": "sdxl_job_123",
  "worker_id": "sdxl_worker_001",
  "status": "completed|failed|processing",
  "assets": [
    {
      "type": "image",
      "url": "workspace-temp/user123/job123/0.png",
      "thumbnail_url": "workspace-temp/user123/job123/0.thumb.webp",
      "metadata": {
        "width": 1024,
        "height": 1024,
        "format": "png",
        "batch_size": 3,
        "num_inference_steps": 25,
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
    "enhancement_source": "qwen_instruct|none",
    "compel_enhancement": true|false,
    "reference_mode": "none|style|composition|character",
    "processing_time": 15.2,
    "vram_used": 8192,
    "batch_size": 3,
    "exact_copy_mode": false
  }
}
```

---

## **📊 Performance Characteristics**

### **Generation Times**
- **Fast (15 steps)**: 1 image: 3-8s, 3 images: 9-24s, 6 images: 18-48s
- **High (25 steps)**: 1 image: 5-10s, 3 images: 15-30s, 6 images: 30-60s
- **Batch Support**: User-selected quantities (1, 3, or 6 images)
- **I2I Processing**: +2-5s for reference image processing
- **Thumbnail Generation**: +0.5-1s per image

### **Job Types**
- `sdxl_image_fast` - 15 steps, guidance_scale 6.0, expected_time_per_image 4s
- `sdxl_image_high` - 25 steps, guidance_scale 7.5, expected_time_per_image 8s

---

## **🔧 Edge Function Requirements**

### **Required Parameters**
```json
{
  "id": "string",                    // Job ID (required)
  "type": "sdxl_image_fast|sdxl_image_high", // Job type (required)
  "prompt": "string",                // Complete prompt (required)
  "user_id": "string",               // User ID (required)
  "config": {
    "num_images": 1|3|6,            // User-selected quantity (required)
    "num_inference_steps": 10-50,   // Generation steps (optional, default: from job type)
    "guidance_scale": 1.0-20.0,     // CFG scale (optional, default: from job type)
    "width": 1024,                  // Image width (optional, default: 1024)
    "height": 1024,                 // Image height (optional, default: 1024)
    "seed": 0-2147483647,           // Random seed (optional)
    "negative_prompt": "string"     // Negative prompt (optional)
  },
  "metadata": {
    "reference_image_url": "string", // Reference image URL (optional)
    "denoise_strength": 0.0-1.0,    // I2I strength (optional, default: 0.5)
    "exact_copy_mode": false,       // Exact copy mode (optional)
    "reference_type": "style|composition|character" // Reference type (optional)
  },
  "compel_enabled": boolean,        // Compel enhancement (optional, default: false)
  "compel_weights": "string"        // Compel weights (optional)
}
```

### **Edge Function Processing**
1. **Validate User Permissions**: Check if user can request NSFW content
2. **Enhance Prompt**: Call Chat Worker for prompt enhancement if requested
3. **Convert Presets**: Transform frontend presets to worker parameters
4. **Validate Parameters**: Check against SDXL validation rules
5. **Route to Worker**: Send complete job data to SDXL worker

---

## **🧠 Memory Management**

### **Memory Allocation**
```python
def setup_sdxl_memory():
    """Configure SDXL memory allocation"""
    if torch.cuda.is_available():
        # SDXL uses 10GB of 48GB VRAM
        torch.cuda.set_per_process_memory_fraction(0.21)  # 10GB / 48GB
        
        # Enable memory optimizations
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
```

### **Batch Processing**
```python
def process_batch_generation(prompts, batch_size):
    """Process batch image generation"""
    
    # Validate batch size
    if batch_size not in [1, 3, 6]:
        raise ValueError("Batch size must be 1, 3, or 6")
    
    # Generate images in batch
    images = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_images = generate_batch(batch_prompts)
        images.extend(batch_images)
    
    return images
```

---

## **📈 Monitoring and Logging**

### **Performance Metrics**
```python
def log_sdxl_metrics(job_data, response_time, batch_size):
    """Log SDXL worker performance metrics"""
    
    metrics = {
        'worker': 'sdxl',
        'job_type': job_data.get('job_type'),
        'response_time': response_time,
        'batch_size': batch_size,
        'num_inference_steps': job_data.get('config', {}).get('num_inference_steps'),
        'guidance_scale': job_data.get('config', {}).get('guidance_scale'),
        'resolution': f"{job_data.get('config', {}).get('width')}x{job_data.get('config', {}).get('height')}",
        'i2i_mode': job_data.get('metadata', {}).get('exact_copy_mode', False),
        'denoise_strength': job_data.get('metadata', {}).get('denoise_strength'),
        'compel_enabled': job_data.get('compel_enabled', False),
        'vram_usage': torch.cuda.memory_allocated() / 1024**3,
        'timestamp': datetime.now().isoformat()
    }
    
    log_metrics(metrics)
```

### **I2I Settings Logging**
```python
def log_i2i_settings(job_id, job_type, denoise_strength, exact_copy_mode, config, reference_image_url, prompt):
    """Log comprehensive i2i settings for debugging and monitoring"""
    logger.info("🖼️ IMAGE-TO-IMAGE SETTINGS LOG")
    logger.info(f"🆔 Job ID: {job_id}")
    logger.info(f"📋 Job Type: {job_type}")
    logger.info(f"🎯 Mode: {'EXACT COPY' if exact_copy_mode else 'REFERENCE MODIFY'}")
    logger.info(f"🔧 Denoise Strength: {denoise_strength:.3f}")
    logger.info(f"📥 Reference Image URL: {reference_image_url}")
    logger.info(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
```

---

## **🔑 Environment Configuration**

### **Required Environment Variables**
```bash
SUPABASE_URL=              # Supabase database URL
SUPABASE_SERVICE_KEY=      # Supabase service key
UPSTASH_REDIS_REST_URL=    # Redis queue URL
UPSTASH_REDIS_REST_TOKEN=  # Redis authentication token
HF_TOKEN=                  # Optional HuggingFace token
```

### **RunPod Deployment**
- **SDXL Worker URL**: `https://{RUNPOD_POD_ID}-7860.proxy.runpod.net`
- **Port**: 7860 (shared with WAN worker)
- **Health Monitoring**: Continuous status tracking via `/health` endpoints

---

## **🚀 Future Enhancements**

### **Planned Improvements**
1. **Advanced I2I Modes**: Additional reference processing options
2. **Custom Models**: Support for custom fine-tuned models
3. **Batch Optimization**: Improved batch processing performance
4. **Real-time Generation**: Streaming generation progress

### **Integration Opportunities**
1. **Character Consistency**: Integration with character reference system
2. **Style Transfer**: Advanced style transfer capabilities
3. **Quality Enhancement**: Post-processing quality improvements

---

**Note**: This worker provides high-quality image generation with comprehensive I2I support, flexible batch processing, and SDXL-specific Compel integration. The LUSTIFY SDXL model ensures professional-quality results with zero content restrictions.
