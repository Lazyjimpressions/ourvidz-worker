# OurVidz Worker API Reference

**Last Updated:** July 16, 2025  
**Status:** ✅ Production Ready - All 10 Job Types Operational + Multi-Reference System Live  
**System:** Dual Worker (SDXL + WAN) on RTX 6000 ADA (48GB VRAM)

---

## **🎯 Worker System Overview**

OurVidz operates with a dual-worker architecture:

1. **SDXL Worker** - High-quality image generation with flexible quantities
2. **WAN Worker** - Video generation and enhanced image processing

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
    "size": "480*832",
    "sample_steps": 15 | 25,
    "sample_guide_scale": 6.0 | 7.5,
    "sample_solver": "unipc",
    "sample_shift": 5.0,
    "frame_num": 1,
    "enhance_prompt": false,
    "seed": 123456789,
    "expected_time": 4 | 8,
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
    "expected_generation_time": 4 | 8,
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
    "first_frame": "string",  // ✅ NEW: Start reference frame URL for video generation
    "last_frame": "string"    // ✅ NEW: End reference frame URL for video generation
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
    "reference_image_url": "string",
    "reference_type": "style" | "composition" | "character",
    "reference_strength": 0.1-1.0,
    "start_reference_url": "string",  // ✅ NEW: Start reference frame URL for video generation
    "end_reference_url": "string",    // ✅ NEW: End reference frame URL for video generation
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
    "num_images": 3
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
| `sdxl_image_fast` | Fast | 15 | 6.0 | 3-8s | 1,3,6 |
| `sdxl_image_high` | High | 25 | 7.5 | 9-24s | 1,3,6 |

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

### **Video Reference Frame Support** ✅ NEW
```python
# Video reference frame parameters
start_reference_url = "https://storage.example.com/start_frame.jpg"
end_reference_url = "https://storage.example.com/end_frame.jpg"
reference_strength = 0.85

# Video generation with reference frames using T2V task (1.3B Model)
if start_reference_url:
    # Use T2V task with --first_frame parameter for start reference
    task_type = "t2v-1.3B"  # ✅ CORRECT: T2V with start frame reference
    
    # Generate video with reference frames
    video = generate_flf2v_video(
        prompt, 
        start_reference_url, 
        end_reference_url, 
        frame_num,
        task_type
    )
else:
    # Use T2V (Text to Video) task for standard generation
    task_type = "t2v-14B"  # or "t2v-1.3B" for smaller model
    video = generate_t2v_video(prompt, frame_num, task_type)
```

### **WAN Command with FLF2V Task for Reference Frames**
```python
# Determine task type based on reference availability (1.3B Model)
if start_ref_path or end_ref_path:
    # Use T2V task with --first_frame and/or --last_frame parameters
    task_type = "t2v-1.3B"  # ✅ CORRECT: T2V with frame references
    print(f"🎬 Using T2V task with frame references (1.3B model)")
else:
    # Use standard T2V task
    task_type = "t2v-1.3B"  # ✅ CORRECT: Text-to-Video standard
    print(f"🎬 Using T2V task for standard video generation")

# Build WAN command with correct task type
# CRITICAL: Use correct path to wan_generate.py in worker repository
wan_generate_path = "/workspace/ourvidz-worker/wan_generate.py"

cmd = [
    "python", wan_generate_path,
    "--task", task_type,
    "--ckpt_dir", model_path,
    "--offload_model", "True",
    "--size", config['size'],
    "--sample_steps", str(config['sample_steps']),
    "--sample_guide_scale", str(config['sample_guide_scale']),
    "--sample_solver", config.get('sample_solver', 'unipc'),
    "--sample_shift", str(config.get('sample_shift', 5.0)),
    "--frame_num", str(config['frame_num']),
    "--prompt", prompt,
    "--save_file", output_path
]

# Add reference frame parameters for T2V task (1.3B Model)
if start_ref_path:
    cmd.extend(["--first_frame", start_ref_path])
    print(f"🖼️ Start reference frame: {start_ref_path}")

if end_ref_path:
    cmd.extend(["--last_frame", end_ref_path])
    print(f"🖼️ End reference frame: {end_ref_path}")

print(f"🎬 T2V command with references: {' '.join(cmd)}")
```

### **Enhanced Negative Prompt Generation**
```python
def generate_negative_prompt_for_sdxl(user_prompt):
    # Priority 1: Critical Quality (Always Included)
    critical_negatives = [
        "bad anatomy", "extra limbs", "deformed", "missing limbs",
        "worst quality", "low quality", "normal quality", "lowres"
    ]
    
    # Priority 2: Anatomical Accuracy (Always Included)
    anatomical_negatives = [
        "deformed hands", "extra fingers", "deformed face", "malformed",
        "bad hands", "bad fingers", "missing fingers", "distorted features"
    ]
    
    # Priority 3: Technical Artifacts (High Priority)
    artifact_negatives = [
        "text", "watermark", "logo", "signature", "contact info",
        "username", "artist name", "title", "caption"
    ]
    
    # Priority 4: Style Prevention (Medium Priority)
    style_negatives = [
        "anime", "cartoon", "graphic", "render", "cgi", "3d",
        "painting", "drawing", "illustration", "sketch"
    ]
    
    # Priority 5: NSFW-Specific (Conditional)
    nsfw_negatives = ["child", "minor"]
    
    # Priority 6: Multi-Party Scene Prevention (Critical for group scenes)
    multi_party_negatives = [
        "three girls", "all girls", "only girls", "no male", "missing male",
        "disembodied penis", "floating penis", "detached penis",
        "penis not attached", "wrong gender ratio", "incorrect participants",
        "wrong number of people"
    ]
    
    # Priority 7: Position and Action Accuracy (Critical for explicit scenes)
    position_negatives = [
        "wrong position", "incorrect pose", "impossible position",
        "unnatural pose", "penis in wrong place", "anatomical mismatch",
        "position confusion", "wrong body parts", "misplaced anatomy",
        "anatomical errors"
    ]
    
    # Priority 8: NSFW Anatomical Improvements (Conditional)
    nsfw_anatomical_negatives = [
        "deformed breasts", "extra breasts", "anatomical errors",
        "wrong anatomy", "distorted bodies", "unnatural poses"
    ]
    
    # Build SDXL negative prompt with priority system
    sdxl_negatives = [
        *critical_negatives,
        *anatomical_negatives,
        *artifact_negatives[:4],
        *style_negatives[:3],
        *nsfw_negatives
    ]
    
    # Enhanced multi-party prevention for group scenes
    prompt_lower = user_prompt.lower()
    has_multiple_people = any(word in prompt_lower for word in ['two', 'both', 'sisters', 'girls'])
    has_females = any(word in prompt_lower for word in ['girl', 'woman', 'sister', 'female'])
    has_males = any(word in prompt_lower for word in ['guy', 'man', 'male', 'boy'])
    
    if has_multiple_people and has_females and has_males:
        sdxl_negatives.extend(multi_party_negatives[:6])
    elif has_multiple_people and has_females and not has_males:
        sdxl_negatives.extend(["three girls", "all girls", "only girls", "wrong number of people"])
    
    # Add position accuracy for explicit scenes
    if any(word in prompt_lower for word in ['sex', 'oral', 'doggy', 'sucking']):
        sdxl_negatives.extend(position_negatives[:5])
    
    # Add NSFW anatomical improvements if applicable
    if any(word in prompt_lower for word in ['naked', 'nude', 'sex', 'topless']):
        sdxl_negatives.extend(nsfw_anatomical_negatives[:4])
    
    return ", ".join(sdxl_negatives)
```

---

## **🎬 WAN Worker Specifications**

### **Model Configuration**
- **Model**: WAN 2.1.1.3B
- **Pipeline**: Video generation and enhanced image processing
- **VRAM**: 48GB RTX 6000 ADA

### **Job Types Supported**
| Job Type | Quality | Steps | Guidance | Time | Quantity |
|----------|---------|-------|----------|------|----------|
| `image_fast` | Fast | 25 | 6.5 | 25-40s | 1 |
| `image_high` | High | 50 | 7.5 | 40-100s | 1 |
| `video_fast` | Fast | 25 | 6.5 | 135-180s | 1 |
| `video_high` | High | 50 | 7.5 | 180-240s | 1 |
| `image7b_fast_enhanced` | Fast Enhanced | 25 | 6.5 | 85-100s | 1 |
| `image7b_high_enhanced` | High Enhanced | 50 | 7.5 | 100-240s | 1 |
| `video7b_fast_enhanced` | Fast Enhanced | 25 | 6.5 | 195-240s | 1 |
| `video7b_high_enhanced` | High Enhanced | 50 | 7.5 | 240+s | 1 |

### **Key Features**
- **Video Generation**: High-quality video output with temporal consistency
- **Enhanced Processing**: 7B model variants for improved quality
- **Reference Support**: Image-to-image for video start/end frames
- **Seed Control**: Reproducible generation (no negative prompts)
- **Path Consistency**: Fixed video path handling

### **Video Generation with Reference Frames (WAN 1.3B Model)**

#### **Reference Strength Control Implementation**
The WAN 1.3B model uses `--first_frame` parameter for reference frames, but doesn't have built-in strength control. The worker implements reference strength by adjusting the `sample_guide_scale` parameter:

```python
def adjust_guidance_for_reference_strength(base_guide_scale, reference_strength):
    """
    Adjust sample_guide_scale based on reference strength to control reference influence
    
    Args:
        base_guide_scale (float): Base guidance scale from job config
        reference_strength (float): Reference strength (0.1-1.0)
        
    Returns:
        float: Adjusted guidance scale
    """
    if reference_strength is None:
        return base_guide_scale
        
    # Reference strength affects how much the reference frame influences generation
    # Higher reference strength = higher guidance scale = stronger reference influence
    # Base range: 6.5-7.5, adjusted range: 5.0-9.0
    
    # Calculate adjustment factor
    # 0.1 strength = minimal influence (5.0 guidance)
    # 1.0 strength = maximum influence (9.0 guidance)
    min_guidance = 5.0
    max_guidance = 9.0
    
    # Linear interpolation between min and max guidance
    adjusted_guidance = min_guidance + (max_guidance - min_guidance) * reference_strength
    
    return adjusted_guidance
```

#### **Reference Strength Mapping**
- **0.1 strength** → **5.0 guidance** (minimal reference influence)
- **0.5 strength** → **7.0 guidance** (moderate reference influence)  
- **0.9 strength** → **8.6 guidance** (strong reference influence)
- **1.0 strength** → **9.0 guidance** (maximum reference influence)

#### **Video Generation Process**
```python
# Video generation parameters
frame_num = 83  # Number of frames
sample_solver = "unipc"  # Temporal consistency
sample_shift = 5.0  # Motion control

# Extract reference frame parameters from job config and metadata
start_reference_url = config.get('first_frame') or metadata.get('start_reference_url')
end_reference_url = config.get('last_frame') or metadata.get('end_reference_url')
reference_strength = metadata.get('reference_strength', 0.85)

# Adjust guidance scale based on reference strength
base_guide_scale = config.get('sample_guide_scale', 6.5)
adjusted_guide_scale = adjust_guidance_for_reference_strength(base_guide_scale, reference_strength)
config['sample_guide_scale'] = adjusted_guide_scale

# Determine task type based on reference availability (1.3B Model)
if start_reference_url:
    # Use T2V task with --first_frame parameter for start reference
    task_type = "t2v-1.3B"  # ✅ CORRECT: T2V with start frame reference
    print(f"🎬 Using T2V task with start frame reference (1.3B model)")
    print(f"🎯 Reference strength: {reference_strength} → Guidance scale: {adjusted_guide_scale}")
else:
    # Use T2V task for standard video generation
    task_type = "t2v-1.3B"  # ✅ CORRECT: Text-to-Video standard
    print(f"🎬 Using T2V task for standard video generation")

# Note: WAN 1.3B model does not support FLF2V (dual reference frames)
# Only start reference frame is used for I2V-style generation
```

#### **Generate video with appropriate task type (1.3B Model)**
```python
if start_reference_url:
    # I2V-style generation with start reference frame
    video = generate_video_with_reference_frame(
        prompt, 
        start_reference_url, 
        frame_num,
        task_type
    )
else:
    # Standard T2V generation
    video = generate_t2v_video(prompt, frame_num, task_type)
```

#### **Optimal Reference Strength Settings**
- **Character Consistency**: 0.8-0.9 (strong character influence)
- **Motion Quality**: 0.7-0.8 (balanced motion and reference)
- **Style Transfer**: 0.6-0.7 (preserve style while allowing changes)
- **Portrait Videos**: 0.8-0.9 (maintain facial features)
- **Action Sequences**: 0.7-0.8 (allow motion flexibility)

### **Reference Frame Processing with FLF2V Task**
```python
def download_image_from_url(image_url):
    """Download image from URL and return PIL Image object"""
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def preprocess_reference_image(image, target_size=(480, 832)):
    """Preprocess reference image for WAN video generation"""
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    new_image = Image.new('RGB', target_size, (0, 0, 0))
    x = (target_size[0] - image.width) // 2
    y = (target_size[1] - image.height) // 2
    new_image.paste(image, (x, y))
    return new_image

def generate_flf2v_video(prompt, start_reference_url, end_reference_url, frame_num, task_type):
    """Generate video using FLF2V task with reference frames"""
    # Download and process reference images
    start_reference = None
    end_reference = None
    
    if start_reference_url:
        start_reference = download_image_from_url(start_reference_url)
        start_reference = preprocess_reference_image(start_reference)
        print(f"🖼️ Start reference frame processed: {start_reference_url}")
    
    if end_reference_url:
        end_reference = download_image_from_url(end_reference_url)
        end_reference = preprocess_reference_image(end_reference)
        print(f"🖼️ End reference frame processed: {end_reference_url}")
    
    # Build WAN command for FLF2V task
    # CRITICAL: Use correct path to wan_generate.py in worker repository
    wan_generate_path = "/workspace/ourvidz-worker/wan_generate.py"
    
    cmd = [
        "python", wan_generate_path,
        "--task", task_type,  # "flf2v-14B" or "flf2v-1.3B"
        "--ckpt_dir", model_path,
        "--offload_model", "True",
        "--size", "480*832",
        "--sample_steps", "25",
        "--sample_guide_scale", "6.5",
        "--sample_solver", "unipc",
        "--sample_shift", "5.0",
        "--frame_num", str(frame_num),
        "--prompt", prompt,
        "--save_file", output_path
    ]
    
    # Add reference frame parameters
    if start_reference_url:
        cmd.extend(["--first_frame", start_reference_url])
    if end_reference_url:
        cmd.extend(["--last_frame", end_reference_url])
    
    print(f"🎬 FLF2V command: {' '.join(cmd)}")
    
    # Execute the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"FLF2V generation failed: {result.stderr}")
    
    return output_path

def generate_t2v_video(prompt, frame_num, task_type):
    """Generate video using T2V task (standard video generation)"""
    # Build WAN command for T2V task
    # CRITICAL: Use correct path to wan_generate.py in worker repository
    wan_generate_path = "/workspace/ourvidz-worker/wan_generate.py"
    
    cmd = [
        "python", wan_generate_path,
        "--task", task_type,  # "t2v-14B" or "t2v-1.3B"
        "--ckpt_dir", model_path,
        "--offload_model", "True",
        "--size", "480*832",
        "--sample_steps", "25",
        "--sample_guide_scale", "6.5",
        "--sample_solver", "unipc",
        "--sample_shift", "5.0",
        "--frame_num", str(frame_num),
        "--prompt", prompt,
        "--save_file", output_path
    ]
    
    print(f"🎬 T2V command: {' '.join(cmd)}")
    
    # Execute the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"T2V generation failed: {result.stderr}")
    
    return output_path
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
    # WAN generation with reference frame support
    if config.get('content_type') == 'video':
        # Check for video reference frames
        start_reference_url = config.get('first_frame') or metadata.get('start_reference_url')
        end_reference_url = config.get('last_frame') or metadata.get('end_reference_url')
        
        if start_reference_url or end_reference_url:
            # Use T2V task with --first_frame and/or --last_frame parameters (1.3B Model)
            task_type = "t2v-1.3B"  # ✅ CORRECT: T2V with frame references
            result = generate_t2v_video_with_references(prompt, start_reference_url, end_reference_url, config.get('frame_num', 83), task_type)
        else:
            # Use T2V task for standard video generation
            task_type = "t2v-1.3B"  # ✅ CORRECT: Text-to-Video standard
            result = generate_t2v_video(prompt, config.get('frame_num', 83), task_type)
    else:
        # Standard image generation
        result = generate_wan_content(prompt, config)
    
    assets = [result]
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
# Send standardized callback
callback_data = {
    "job_id": job_id,
    "status": "completed",
    "assets": uploaded_assets,
    "metadata": {
        "seed": config.get("seed"),
        "generation_time": generation_time,
        "num_images": len(uploaded_assets)
    }
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
        "worker_version": "2.1.0"
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
    "assets_generated": len(assets),
    "total_size_mb": sum(get_file_size(asset) for asset in assets),
    "completion_timestamp": datetime.now().isoformat()
}
```

---

## **🚀 Recent Updates (July 16, 2025)**

### **Major Enhancements**
1. **Standardized Callback Parameters**: Consistent `job_id`, `assets` array across all workers
2. **Enhanced Negative Prompts**: Intelligent generation for SDXL with multi-party scene detection
3. **Seed Support**: User-controlled seeds for reproducible generation
4. **Flexible SDXL Quantities**: User-selectable 1, 3, or 6 images per batch
5. **Reference Image Support**: Optional image-to-image with type and strength control
6. **Video Reference Frame Support**: ✅ NEW: I2V-style generation with start reference frame for WAN 1.3B model
7. **Comprehensive Error Handling**: Enhanced debugging and error tracking
8. **Metadata Consistency**: Improved data flow and storage
9. **Path Consistency Fix**: Fixed video path handling for WAN workers

### **Performance Improvements**
- Optimized batch processing for multi-image SDXL jobs
- Enhanced error recovery and retry mechanisms
- Improved Redis queue management
- Better resource utilization tracking

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