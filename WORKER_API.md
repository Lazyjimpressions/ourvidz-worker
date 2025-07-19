# OurVidz Worker API Reference

**Last Updated:** July 16, 2025  
**Status:** ✅ Production Ready - All Job Types Operational + Comprehensive Reference Frame Support  
**System:** Dual Worker (SDXL + WAN) on RTX 6000 ADA (48GB VRAM)

---

## **🎯 Worker System Overview**

OurVidz operates with a dual-worker architecture:

1. **SDXL Worker** - High-quality image generation with flexible quantities (1, 3, or 6 images)
2. **WAN Worker** - Video generation and enhanced image processing with comprehensive reference frame support

Both workers use standardized callback parameters and comprehensive metadata management.

---

## **📤 Job Queue System**

### **Queue Structure**
- **`sdxl_queue`** - SDXL image generation jobs
- **`wan_queue`** - WAN video and enhanced image jobs

### **Job Payload Format (Standardized)**

#### **SDXL Job Payload**
```json
{
  "id": "uuid",
  "type": "sdxl_image_fast" | "sdxl_image_high",
  "prompt": "string",
  "config": {
    "size": "1024*1024",
    "sample_steps": 15 | 25,
    "sample_guide_scale": 6.0 | 7.5,
    "sample_solver": "unipc",
    "sample_shift": 5.0,
    "frame_num": 1,
    "enhance_prompt": false,
    "seed": 123456789,
    "expected_time": 30 | 42,
    "content_type": "image",
    "file_extension": "png",
    "num_images": 1 | 3 | 6
  },
  "user_id": "uuid",
  "created_at": "2025-07-16T...",
  "negative_prompt": "string",
  "video_id": null,
  "image_id": "uuid",
  "character_id": "uuid",
  "model_variant": "lustify_sdxl",
  "bucket": "sdxl_image_fast" | "sdxl_image_high",
  "metadata": {
    "model_variant": "lustify_sdxl",
    "queue": "sdxl_queue",
    "negative_prompt": "string",
    "seed": 123456789,
    "num_images": 1 | 3 | 6,
    "reference_image_url": "string",
    "reference_type": "style" | "composition" | "character",
    "reference_strength": 0.1-1.0,
    "expected_generation_time": 30 | 42,
    "dual_worker_routing": true,
    "negative_prompt_supported": true,
    "edge_function_version": "2.1.0"
  }
}
```

#### **WAN Job Payload**
```json
{
  "id": "uuid",
  "type": "image_fast" | "image_high" | "video_fast" | "video_high" | "image7b_fast_enhanced" | "image7b_high_enhanced" | "video7b_fast_enhanced" | "video7b_high_enhanced",
  "prompt": "string",
  "config": {
    "size": "480*832",
    "sample_steps": 25 | 50,
    "sample_guide_scale": 6.5 | 7.5,
    "sample_solver": "unipc",
    "sample_shift": 5.0,
    "frame_num": 1 | 83,
    "enhance_prompt": true | false,
    "seed": 123456789,
    "expected_time": 25-240,
    "content_type": "image" | "video",
    "file_extension": "png" | "mp4",
    "num_images": 1,
    "task": "t2v-1.3B",
    "image": "string",           // ✅ Single reference frame URL (I2V-style)
    "first_frame": "string",     // ✅ Start reference frame URL
    "last_frame": "string"       // ✅ End reference frame URL
  },
  "user_id": "uuid",
  "created_at": "2025-07-16T...",
  "video_id": "uuid",
  "image_id": "uuid",
  "character_id": "uuid",
  "model_variant": "wan_2_1_1_3b",
  "bucket": "image_fast" | "image_high" | "video_fast" | "video_high" | "image7b_fast_enhanced" | "image7b_high_enhanced" | "video7b_fast_enhanced" | "video7b_high_enhanced",
  "metadata": {
    "model_variant": "wan_2_1_1_3b",
    "queue": "wan_queue",
    "seed": 123456789,
    "num_images": 1,
    "reference_image_url": "string",  // ✅ Single reference frame URL (fallback)
    "start_reference_url": "string",  // ✅ Start reference frame URL (fallback)
    "end_reference_url": "string",    // ✅ End reference frame URL (fallback)
    "reference_strength": 0.1-1.0,
    "expected_generation_time": 25-240,
    "dual_worker_routing": true,
    "negative_prompt_supported": false,
    "edge_function_version": "2.1.0"
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
  "enhancedPrompt": "string",
  "metadata": {
    "seed": 123456789,
    "generation_time": 15.5,
    "num_images": 3,
    "job_type": "sdxl_image_fast",
    "content_type": "image",
    "frame_num": 1,
    "wan_task": "t2v-1.3B",
    "reference_mode": "none" | "single" | "start" | "end" | "both"
  }
}
```

### **Callback Response**
```json
{
  "success": true,
  "message": "Job callback processed successfully with standardized parameters",
  "debug": {
    "job_id": "uuid",
    "jobStatus": "completed",
    "jobType": "sdxl_image_fast",
    "format": "image",
    "quality": "fast",
    "isSDXL": true,
    "isEnhanced": false,
    "assetsProcessed": 3,
    "processingTimestamp": "2025-07-16T..."
  }
}
```

---

## **🎨 SDXL Worker Specifications**

### **Model Configuration**
- **Model Path**: `/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors`
- **Pipeline**: StableDiffusionXLPipeline
- **VRAM**: 48GB RTX 6000 ADA

### **Job Types Supported**
| Job Type | Quality | Steps | Guidance | Time | Quantity |
|----------|---------|-------|----------|------|----------|
| `sdxl_image_fast` | Fast | 15 | 6.0 | 30s | 1,3,6 |
| `sdxl_image_high` | High | 25 | 7.5 | 42s | 1,3,6 |

### **Performance Metrics**
- **1 Image**: 3-8 seconds
- **3 Images**: 9-24 seconds
- **6 Images**: 18-48 seconds

### **Key Features**
- **Flexible Quantities**: User-selectable 1, 3, or 6 images per batch
- **Image-to-Image**: Support for style, composition, and character references
- **Seed Control**: Reproducible generation with user-controlled seeds
- **Enhanced Negative Prompts**: Intelligent generation with multi-party scene detection
- **Batch Processing**: Efficient multi-image generation

### **Reference Image Support**
```python
# Reference image parameters
reference_image_url = "https://storage.example.com/reference.jpg"
reference_type = "style" | "composition" | "character"
reference_strength = 0.1-1.0

# Image-to-image generation
if reference_image_url:
    # Load and process reference image
    reference_image = load_reference_image(reference_image_url)
    # Apply reference influence based on type and strength
    result = generate_with_reference(prompt, reference_image, reference_type, reference_strength)
```

---

## **🎬 WAN Worker Specifications**

### **Model Configuration**
- **Model**: WAN 2.1 T2V 1.3B
- **Model Path**: `/workspace/models/wan2.1-t2v-1.3b`
- **Pipeline**: Video generation and enhanced image processing
- **VRAM**: 48GB RTX 6000 ADA
- **Script Path**: `/workspace/ourvidz-worker/wan_generate.py`

### **WAN 1.3B Definitive Capabilities**

| **Reference Mode** | **Task** | **Parameters** | **Use Case** |
|-------------------|----------|----------------|--------------|
| **Standard Generation** | `t2v-1.3B` | None | Text-to-video only |
| **Single Reference** | `t2v-1.3B` | `--image ref.png` | Image-to-video style |
| **Start Frame Only** | `t2v-1.3B` | `--first_frame start.png` | Video begins with reference |
| **End Frame Only** | `t2v-1.3B` | `--last_frame end.png` | Video ends with reference |
| **Both Frames** | `t2v-1.3B` | `--first_frame start.png --last_frame end.png` | Transition video |

### **Job Types Supported**
| Job Type | Quality | Steps | Guidance | Time | Quantity | Enhancement |
|----------|---------|-------|----------|------|----------|-------------|
| `image_fast` | Fast | 25 | 6.5 | 25-40s | 1 | No |
| `image_high` | High | 50 | 7.5 | 40-100s | 1 | No |
| `video_fast` | Fast | 25 | 6.5 | 135-180s | 1 | No |
| `video_high` | High | 50 | 7.5 | 180-240s | 1 | No |
| `image7b_fast_enhanced` | Fast Enhanced | 25 | 6.5 | 85-100s | 1 | Yes |
| `image7b_high_enhanced` | High Enhanced | 50 | 7.5 | 100-240s | 1 | Yes |
| `video7b_fast_enhanced` | Fast Enhanced | 25 | 6.5 | 195-240s | 1 | Yes |
| `video7b_high_enhanced` | High Enhanced | 50 | 7.5 | 240+s | 1 | Yes |

### **Key Features**
- **Video Generation**: High-quality video output with temporal consistency
- **Enhanced Processing**: 7B model variants for improved quality with AI prompt enhancement
- **Comprehensive Reference Support**: All 5 reference frame modes supported
- **Seed Control**: Reproducible generation (no negative prompts)
- **Path Consistency**: Fixed video path handling with correct script paths

### **Reference Frame Processing**

#### **Reference Mode Detection**
```python
def determine_reference_mode(single_reference_url, start_reference_url, end_reference_url):
    if single_reference_url and not start_reference_url and not end_reference_url:
        return 'single'
    elif start_reference_url and end_reference_url:
        return 'both'
    elif start_reference_url and not end_reference_url:
        return 'start'
    elif end_reference_url and not start_reference_url:
        return 'end'
    else:
        return 'none'
```

#### **Command Building Logic**
```python
def build_wan_command(prompt, job_type, config):
    cmd = [
        "python", "/workspace/ourvidz-worker/wan_generate.py",
        "--task", "t2v-1.3B",  # Always t2v-1.3B
        "--ckpt_dir", "/workspace/models/wan2.1-t2v-1.3b",
        # ... other standard parameters ...
        "--prompt", prompt
    ]
    
    # Add reference parameters based on config
    if 'image' in config:
        cmd.extend(["--image", config['image']])
    
    if 'first_frame' in config:
        cmd.extend(["--first_frame", config['first_frame']])
        
    if 'last_frame' in config:
        cmd.extend(["--last_frame", config['last_frame']])
    
    return cmd
```

#### **Reference Frame Generation Examples**

**Standard Generation (no reference):**
```bash
python /workspace/ourvidz-worker/wan_generate.py --task t2v-1.3B --prompt "..." --save_file output.mp4
```

**Single Reference:**
```bash
python /workspace/ourvidz-worker/wan_generate.py --task t2v-1.3B --prompt "..." --image reference.png --save_file output.mp4
```

**Start Frame Only:**
```bash
python /workspace/ourvidz-worker/wan_generate.py --task t2v-1.3B --prompt "..." --first_frame start.png --save_file output.mp4
```

**End Frame Only:**
```bash
python /workspace/ourvidz-worker/wan_generate.py --task t2v-1.3B --prompt "..." --last_frame end.png --save_file output.mp4
```

**Both Frames:**
```bash
python /workspace/ourvidz-worker/wan_generate.py --task t2v-1.3B --prompt "..." --first_frame start.png --last_frame end.png --save_file output.mp4
```

### **Environment Setup**
```python
def setup_environment(self):
    """Configure environment variables for WAN and Qwen"""
    env = os.environ.copy()
    
    # CRITICAL: Add WAN code directory to Python path for module resolution
    python_deps_path = '/workspace/python_deps/lib/python3.11/site-packages'
    wan_code_path = '/workspace/Wan2.1'  # WAN code directory
    
    current_pythonpath = env.get('PYTHONPATH', '')
    if current_pythonpath:
        new_pythonpath = f"{wan_code_path}:{python_deps_path}:{current_pythonpath}"
    else:
        new_pythonpath = f"{wan_code_path}:{python_deps_path}"
    
    env.update({
        'CUDA_VISIBLE_DEVICES': '0',
        'TORCH_USE_CUDA_DSA': '1',
        'PYTHONUNBUFFERED': '1',
        'PYTHONPATH': new_pythonpath,
        'HF_HOME': self.hf_cache_path,
        'TRANSFORMERS_CACHE': self.hf_cache_path,
        'HUGGINGFACE_HUB_CACHE': f"{self.hf_cache_path}/hub"
    })
    return env
```

---

## **🔄 Job Processing Flow**

### **1. Job Retrieval**
```python
# Worker retrieves job from queue
job = redis_client.rpop(queue_name)
job_data = json.loads(job)

# Extract job parameters
job_id = job_data["id"]
job_type = job_data["type"]
prompt = job_data["prompt"]
config = job_data["config"]
user_id = job_data["user_id"]
```

### **2. Reference Frame Detection (WAN Worker)**
```python
# Extract reference frame parameters from config and metadata
metadata = job_data.get('metadata', {})
single_reference_url = config.get('image') or metadata.get('reference_image_url')
start_reference_url = config.get('first_frame') or metadata.get('start_reference_url')
end_reference_url = config.get('last_frame') or metadata.get('end_reference_url')

# Determine reference mode
reference_mode = determine_reference_mode(single_reference_url, start_reference_url, end_reference_url)
```

### **3. Model Loading**
```python
# Load appropriate model based on job type
if job_type.startswith("sdxl_"):
    model = load_sdxl_model()
elif job_type.startswith("video"):
    model = load_wan_video_model()
else:
    model = load_wan_image_model()
```

### **4. Generation Execution**

#### **SDXL Generation**
```python
if job_type.startswith("sdxl_"):
    # SDXL generation with flexible quantities
    num_images = config.get("num_images", 1)
    results = []
    
    for i in range(num_images):
        # Generate with seed for consistency
        seed = config.get("seed", random.randint(1, 999999999))
        result = generate_sdxl_image(prompt, config, seed)
        results.append(result)
    
    assets = results
```

#### **WAN Generation**
```python
else:
    # WAN generation with comprehensive reference frame support
    if config.get('content_type') == 'video':
        # Determine reference frame mode and route to appropriate generation function
        if single_reference_url and not start_reference_url and not end_reference_url:
            # Single reference frame mode (I2V-style)
            output_file = generate_video_with_reference_frame(prompt, single_reference_image, job_type)
        elif start_reference_url and end_reference_url:
            # Both frames mode (start + end)
            output_file = generate_video_with_both_frames(prompt, start_reference_image, end_reference_image, job_type)
        elif start_reference_url and not end_reference_url:
            # Start frame only mode
            output_file = generate_video_with_start_frame(prompt, start_reference_image, job_type)
        elif end_reference_url and not start_reference_url:
            # End frame only mode
            output_file = generate_video_with_end_frame(prompt, end_reference_image, job_type)
        else:
            # Standard generation (no reference frames)
            output_file = generate_standard_content(prompt, job_type)
    else:
        # Standard image generation
        output_file = generate_content(prompt, job_type)
    
    assets = [output_file]
```

### **5. Asset Upload**
```python
# Upload generated assets to storage
uploaded_assets = []
for asset in assets:
    # Upload to appropriate bucket
    bucket = determine_bucket(job_type)
    asset_url = upload_to_storage(asset, bucket, user_id, job_id)
    uploaded_assets.append(asset_url)
```

### **6. Callback Execution**
```python
# Prepare metadata for callback
callback_metadata = {
    'generation_time': total_time,
    'job_type': job_type,
    'content_type': final_config['content_type'],
    'frame_num': final_config['frame_num'],
    'wan_task': 't2v-1.3B',  # Always t2v-1.3B for WAN 1.3B model
    'reference_mode': reference_mode
}

# Send standardized callback
callback_data = {
    "job_id": job_id,
    "status": "completed",
    "assets": uploaded_assets,
    "metadata": callback_metadata
}

response = requests.post(callback_url, json=callback_data)
```

---

## **📊 Performance Monitoring**

### **Generation Time Tracking**
```python
# Track actual generation time
start_time = time.time()
result = generate_content(prompt, config)
generation_time = time.time() - start_time

# Include in callback metadata
callback_metadata = {
    "generation_time": generation_time,
    "expected_time": config.get("expected_time"),
    "performance_ratio": generation_time / config.get("expected_time")
}
```

### **Resource Utilization**
```python
# Monitor VRAM usage
import torch
vram_used = torch.cuda.memory_allocated() / 1024**3  # GB
vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

# Include in metadata
metadata["vram_used_gb"] = vram_used
metadata["vram_total_gb"] = vram_total
metadata["vram_utilization"] = vram_used / vram_total
```

---

## **🔧 Error Handling**

### **Standardized Error Responses**
```python
# Error callback format
error_callback = {
    "job_id": job_id,
    "status": "failed",
    "error_message": str(error),
    "metadata": {
        "error_type": type(error).__name__,
        "error_timestamp": datetime.now().isoformat(),
        "worker_version": "2.1.0",
        "wan_task": "t2v-1.3B",
        "reference_mode": reference_mode
    }
}
```

### **Common Error Scenarios**
```python
try:
    # Generation attempt
    result = generate_content(prompt, config)
except torch.cuda.OutOfMemoryError:
    # Handle VRAM issues
    error_message = "Insufficient VRAM for generation"
    cleanup_gpu_memory()
except ValueError as e:
    # Handle parameter errors
    error_message = f"Invalid parameters: {str(e)}"
except Exception as e:
    # Handle unexpected errors
    error_message = f"Generation failed: {str(e)}"
    log_error(e)
```

---

## **🛠️ Storage Buckets**

### **SDXL Buckets**
- `sdxl_image_fast` - Fast SDXL image generation
- `sdxl_image_high` - High quality SDXL image generation

### **WAN Buckets**
- `image_fast` - Fast WAN image generation
- `image_high` - High quality WAN image generation
- `video_fast` - Fast WAN video generation
- `video_high` - High quality WAN video generation

### **Enhanced WAN Buckets**
- `image7b_fast_enhanced` - Enhanced fast image generation
- `image7b_high_enhanced` - Enhanced high quality image generation
- `video7b_fast_enhanced` - Enhanced fast video generation
- `video7b_high_enhanced` - Enhanced high quality video generation

### **Reference Image Buckets**
- `reference_images` - User-uploaded reference images
- `workspace_assets` - Workspace reference assets

---

## **📈 Usage Tracking**

### **Worker Metrics**
```python
# Track worker performance
worker_metrics = {
    "worker_id": worker_id,
    "job_type": job_type,
    "generation_time": generation_time,
    "vram_used": vram_used,
    "reference_mode": reference_mode,
    "success": True,
    "timestamp": datetime.now().isoformat()
}

# Send to metrics endpoint
requests.post(metrics_url, json=worker_metrics)
```

### **Job Completion Tracking**
```python
# Track job completion statistics
completion_stats = {
    "job_id": job_id,
    "user_id": user_id,
    "job_type": job_type,
    "reference_mode": reference_mode,
    "assets_generated": len(assets),
    "total_size_mb": sum(get_file_size(asset) for asset in assets),
    "completion_timestamp": datetime.now().isoformat()
}
```

---

## **🚀 Recent Updates (July 16, 2025)**

### **Major Enhancements**
1. **✅ WAN 1.3B Model Support**: Complete support for WAN 2.1 T2V 1.3B model
2. **✅ Comprehensive Reference Frame Support**: All 5 reference modes (none, single, start, end, both)
3. **✅ Correct Task Usage**: Always uses `t2v-1.3B` task with appropriate parameters
4. **✅ Fixed Module Imports**: Proper PYTHONPATH configuration for WAN module resolution
5. **✅ Correct File Paths**: Uses `/workspace/ourvidz-worker/wan_generate.py` consistently
6. **✅ Standardized Callback Parameters**: Consistent `job_id`, `assets` array across all workers
7. **✅ Enhanced Negative Prompts**: Intelligent generation for SDXL with multi-party scene detection
8. **✅ Seed Support**: User-controlled seeds for reproducible generation
9. **✅ Flexible SDXL Quantities**: User-selectable 1, 3, or 6 images per batch
10. **✅ Comprehensive Error Handling**: Enhanced debugging and error tracking
11. **✅ Metadata Consistency**: Improved data flow and storage
12. **✅ Path Consistency Fix**: Fixed video path handling for WAN workers

### **Performance Improvements**
- Optimized batch processing for multi-image SDXL jobs
- Enhanced error recovery and retry mechanisms
- Improved Redis queue management
- Better resource utilization tracking
- Fixed module import issues for reliable WAN execution

### **Developer Experience**
- Enhanced API documentation and examples
- Comprehensive debugging information
- Backward compatibility preservation
- Clear error messages and status codes

### **Backward Compatibility**
- All existing job types remain functional
- Legacy metadata fields are preserved
- Single-reference workflows continue to work
- Non-reference generation unchanged

---

## **📋 WAN 1.3B Reference Frame Support Matrix**

| **Reference Mode** | **Config Parameter** | **Metadata Fallback** | **WAN Parameters** | **Use Case** |
|-------------------|---------------------|----------------------|-------------------|--------------|
| **None** | No parameters | No parameters | None | Standard T2V |
| **Single** | `config.image` | `metadata.reference_image_url` | `--image ref.png` | I2V-style |
| **Start** | `config.first_frame` | `metadata.start_reference_url` | `--first_frame start.png` | Start frame |
| **End** | `config.last_frame` | `metadata.end_reference_url` | `--last_frame end.png` | End frame |
| **Both** | `config.first_frame` + `config.last_frame` | `metadata.start_reference_url` + `metadata.end_reference_url` | `--first_frame start.png --last_frame end.png` | Transition |

---

## **✅ Production Status**

### **Active Components**
- **Dual Orchestrator**: Main production controller managing both workers
- **SDXL Worker**: Fast image generation with batch support (1, 3, or 6 images)
- **Enhanced WAN Worker**: Video generation with comprehensive reference frame support

### **Testing Status**
- **SDXL Jobs**: ✅ Both job types tested and working
- **WAN Jobs**: ✅ All 8 job types tested and working
- **Reference Frames**: ✅ All 5 reference modes tested and working
- **Performance Baselines**: ✅ Real data established for all jobs

### **System Capabilities**
- **✅ 10 Job Types**: All job types operational
- **✅ 5 Reference Modes**: Complete reference frame support
- **✅ Batch Processing**: SDXL supports 1, 3, or 6 images
- **✅ AI Enhancement**: WAN enhanced variants with Qwen 7B
- **✅ Error Recovery**: Robust error handling and fallback mechanisms
- **✅ Performance Monitoring**: Comprehensive metrics and logging

The OurVidz Worker system is **production-ready** with comprehensive reference frame support, robust error handling, and optimized performance for high-quality AI content generation! 🎯 