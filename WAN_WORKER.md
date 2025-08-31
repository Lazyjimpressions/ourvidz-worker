# WAN Worker Documentation

**Last Updated:** August 31, 2025  
**Status:** ✅ ACTIVE - Fully implemented for video generation with enhanced features

## **🎯 Overview**

The WAN Worker is responsible for video generation using the WAN 2.1 T2V 1.3B model, with integration to Qwen 2.5-7B Base for prompt enhancement. It's part of the triple worker system and provides high-quality video generation capabilities with comprehensive reference frame support and I2I pipeline.

### **Key Capabilities**
- **Video Generation**: High-quality video generation with WAN 2.1 T2V 1.3B
- **Prompt Enhancement**: Integration with Qwen 2.5-7B Base for enhanced prompts
- **Multiple Formats**: Support for different video lengths and qualities
- **Image-to-Video**: Convert images to video sequences with 5 reference frame modes
- **I2I Pipeline**: First-class Image-to-Image support with parameter clamping
- **Thumbnail Generation**: Mid-frame video thumbnails in 256px WEBP format
- **Auto-Registration**: Automatic RunPod URL detection and Supabase registration
- **Thread-Safe Timeouts**: Concurrent.futures implementation for reliability

---

## **🔧 Technical Setup**

### **Model Configuration**
```python
# WAN 2.1 T2V 1.3B Model Setup
MODEL_PATH = "/workspace/models/wan2.1-t2v-1.3b/"
MODEL_NAME = "Wan2.1-T2V-1.3B"
VRAM_USAGE = "30GB"  # Peak usage during generation
MAX_CONCURRENT_JOBS = 4
PORT = 7860  # Shared with SDXL worker
```

### **Hardware Requirements**
- **GPU**: NVIDIA RTX 6000 ADA (48GB VRAM)
- **Memory**: 30GB VRAM peak usage
- **Storage**: Model files (~8GB)
- **Performance**: 25-240 seconds for 5-second video

### **Worker Configuration**
```python
WORKER_CONFIG = {
    "wan": {
        "model_path": "/workspace/models/wan2.1-t2v-1.3b/",
        "max_concurrent_jobs": 4,
        "memory_limit": 30,  # GB
        "polling_interval": 5,
        "port": 7860,
        "job_types": [
            "image_fast", "image_high", "video_fast", "video_high",
            "image7b_fast_enhanced", "image7b_high_enhanced", 
            "video7b_fast_enhanced", "video7b_high_enhanced"
        ]
    }
}
```

---

## **🎬 Video Generation Implementation**

### **Job Types and Performance**

#### **Standard Job Types (No Enhancement)**
| **Job Type** | **Task** | **Frames** | **Steps** | **Time** | **Content** |
|--------------|----------|------------|-----------|----------|-------------|
| `image_fast` | t2v-1.3B | 1 | 25 | 25s | Image |
| `image_high` | t2v-1.3B | 1 | 50 | 40s | Image |
| `video_fast` | t2v-1.3B | 83 | 25 | 135s | Video |
| `video_high` | t2v-1.3B | 83 | 50 | 180s | Video |

#### **Enhanced Job Types (Qwen Base Enhancement)**
| **Job Type** | **Task** | **Frames** | **Steps** | **Time** | **Content** |
|--------------|----------|------------|-----------|----------|-------------|
| `image7b_fast_enhanced` | t2v-1.3B | 1 | 25 | 85s | Image |
| `image7b_high_enhanced` | t2v-1.3B | 1 | 50 | 100s | Image |
| `video7b_fast_enhanced` | t2v-1.3B | 83 | 25 | 195s | Video |
| `video7b_high_enhanced` | t2v-1.3B | 83 | 50 | 240s | Video |

### **Video Processing Pipeline**

#### **Standard Video Generation**
```python
def generate_video(job_data):
    """Standard video generation pipeline"""
    
    # 1. Load model
    model = load_wan_model()
    
    # 2. Process prompt enhancement with Qwen (if enhanced job type)
    if job_data.get('enhance_prompt'):
        enhanced_prompt = enhance_prompt_with_qwen(job_data['prompt'])
    else:
        enhanced_prompt = job_data['prompt']
    
    # 3. Generate video with WAN 2.1
    video = model.generate(
        task='t2v-1.3B',
        prompt=enhanced_prompt,
        negative_prompt=job_data.get('negative_prompt', ''),
        sample_guide_scale=job_data.get('sample_guide_scale', 6.5),
        sample_steps=job_data.get('sample_steps', 25),
        size=job_data.get('size', '480*832'),
        frame_num=job_data.get('frame_num', 83),  # 83 frames for 5-second video
        sample_solver='unipc',
        sample_shift=5.0
    )
    
    return video
```

#### **Image-to-Video Generation with Reference Frames**
```python
def generate_image_to_video(job_data):
    """Image-to-video generation with reference frame support"""
    
    # 1. Load model
    model = load_wan_model()
    
    # 2. Process reference image based on mode
    reference_mode = job_data.get('reference_mode', 'none')
    
    if reference_mode == 'single':
        reference_image = load_reference_image(job_data['reference_image_url'])
        wan_params = f"--image {reference_image}"
    elif reference_mode == 'start':
        reference_image = load_reference_image(job_data['first_frame'])
        wan_params = f"--first_frame {reference_image}"
    elif reference_mode == 'end':
        reference_image = load_reference_image(job_data['last_frame'])
        wan_params = f"--last_frame {reference_image}"
    elif reference_mode == 'both':
        start_frame = load_reference_image(job_data['first_frame'])
        end_frame = load_reference_image(job_data['last_frame'])
        wan_params = f"--first_frame {start_frame} --last_frame {end_frame}"
    else:
        wan_params = ""
    
    # 3. Process prompt enhancement
    if job_data.get('enhance_prompt'):
        enhanced_prompt = enhance_prompt_with_qwen(job_data['prompt'])
    else:
        enhanced_prompt = job_data['prompt']
    
    # 4. Generate video with reference frames
    video = model.generate(
        task='t2v-1.3B',
        prompt=enhanced_prompt,
        negative_prompt=job_data.get('negative_prompt', ''),
        sample_guide_scale=job_data.get('sample_guide_scale', 6.5),
        sample_steps=job_data.get('sample_steps', 25),
        size=job_data.get('size', '480*832'),
        frame_num=job_data.get('frame_num', 83),
        sample_solver='unipc',
        sample_shift=5.0,
        **wan_params
    )
    
    return video
```

### **Qwen Integration**

#### **Prompt Enhancement with Qwen Base**
```python
def enhance_prompt_with_qwen(prompt, job_type="video_fast", quality="fast"):
    """Enhance prompt using Qwen 2.5-7B Base model"""
    
    # Load Qwen model (if not already loaded)
    qwen_model = load_qwen_model()
    
    # Create contextually-aware enhancement messages
    messages = create_enhanced_messages(prompt, job_type, quality)
    
    # Generate enhanced prompt with thread-safe timeout
    enhanced_prompt = run_with_timeout(
        lambda: qwen_model.generate(messages),
        timeout_seconds=60
    )
    
    return enhanced_prompt.strip()
```

#### **WAN-Specific System Prompts**
```python
def get_wan_system_prompt(job_type="video_fast", quality="fast"):
    """WAN 2.1-specific system prompt for enhancement"""
    
    base_prompt = """You are an expert AI prompt engineer specializing in WAN 2.1 video generation and temporal consistency.

CRITICAL REQUIREMENTS:
- Target Model: WAN 2.1 T2V 1.3B (motion-focused, 5-second videos)
- Content Focus: Temporal consistency, smooth motion, cinematic quality
- Quality Priority: Motion realism, scene coherence, professional cinematography

ENHANCEMENT STRATEGY:
1. MOTION FIRST: Describe natural, fluid movements and transitions
2. TEMPORAL CONSISTENCY: Ensure elements maintain coherence across frames
3. CINEMATOGRAPHY: Add professional camera work (smooth pans, steady shots)
4. SCENE SETTING: Establish clear environment and spatial relationships  
5. TECHNICAL QUALITY: Video-specific quality terms (smooth motion, stable)

WAN-SPECIFIC OPTIMIZATION:
- Motion descriptions: "smooth movement, natural motion, fluid transitions"
- Temporal stability: "consistent lighting, stable composition, coherent scene"
- Cinematography: "professional camera work, smooth pans, steady shots"
- Video quality: "high framerate, smooth motion, temporal consistency"
- Scene coherence: "well-lit environment, clear spatial relationships"

TOKEN STRATEGY: 150-250 tokens optimal for detailed motion description"""
    
    return base_prompt
```

---

## **🖼️ I2I Pipeline and Thumbnail Generation**

### **I2I Pipeline Features**
- **denoise_strength Parameter**: Replaces `reference_strength` for consistency
- **Backward Compatibility**: `reference_strength` automatically converted to `denoise_strength = 1 - reference_strength`
- **Parameter Handling**: Workers respect edge function parameters (clamping only in exact copy mode)

### **Video Thumbnail Generation**
- **Mid-Frame Thumbnails**: Extract middle frame of video for better representation
- **256px WEBP Format**: Longest edge 256px, preserve aspect ratio, quality 85
- **Storage**: Both original video and thumbnail uploaded to `workspace-temp`
- **Callback Format**: Includes both `url` and `thumbnail_url` for each asset

### **Reference Frame Support**
| **Reference Mode** | **Config Parameter** | **Metadata Fallback** | **WAN Parameters** | **Use Case** |
|-------------------|---------------------|----------------------|-------------------|--------------|
| **None** | No parameters | No parameters | None | Standard T2V |
| **Single** | `config.image` | `metadata.reference_image_url` | `--image ref.png` | I2V-style |
| **Start** | `config.first_frame` | `metadata.start_reference_url` | `--first_frame start.png` | Start frame |
| **End** | `config.last_frame` | `metadata.end_reference_url` | `--last_frame end.png` | End frame |
| **Both** | `config.first_frame` + `config.last_frame` | `metadata.start_reference_url` + `metadata.end_reference_url` | `--first_frame start.png --last_frame end.png` | Transition |

---

## **🔗 Frontend Integration**

### **Job Submission**
```typescript
// Frontend job submission to WAN worker
const submitWANJob = async (params: WANJobParams) => {
  const jobData = {
    job_type: 'video7b_high_enhanced',
    prompt: params.prompt,
    negative_prompt: params.negativePrompt,
    width: params.width || 480,
    height: params.height || 832,
    frame_num: params.frameNum || 83,
    fps: params.fps || 16,
    quality: params.quality || 'high',
    
    // Reference frame parameters
    reference_mode: params.referenceMode || 'none',
    reference_image_url: params.referenceImageUrl,
    first_frame: params.firstFrame,
    last_frame: params.lastFrame,
    
    // I2I parameters
    denoise_strength: params.denoiseStrength,
    
    // Worker routing
    target_worker: 'wan'
  };
  
  return await queueJob(jobData);
};
```

### **Progress Tracking**
```typescript
// Frontend progress tracking for video generation
const trackVideoProgress = (jobId: string) => {
  const progressCallback = (progress: VideoProgress) => {
    // Update UI with progress
    updateProgressUI({
      status: progress.status,
      percentage: progress.percentage,
      estimatedTime: progress.estimatedTime,
      currentStep: progress.currentStep,
      referenceMode: progress.referenceMode
    });
  };
  
  return subscribeToJobProgress(jobId, progressCallback);
};
```

---

## **🌐 API Endpoints**

### **Health Check**
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "qwen_loaded": true,
  "timestamp": 1693456789.123,
  "worker_ready": true,
  "thread_safe_timeouts": true
}
```

### **Debug Environment**
```http
GET /debug/env
```
**Response:**
```json
{
  "wan_worker_api_key_set": true,
  "wan_worker_api_key_value": "abc123def...",
  "expected_key": "abc123def...",
  "thread_safe_timeouts": true,
  "all_env_vars": {
    "SUPABASE_URL": "https://...",
    "SUPABASE_SERVICE_KEY": "***",
    "WAN_WORKER_API_KEY": "***"
  }
}
```

### **Prompt Enhancement**
```http
POST /enhance
Authorization: Bearer {WAN_WORKER_API_KEY}
Content-Type: application/json

{
  "prompt": "Original prompt to enhance",
  "model": "qwen_base"
}
```
**Response:**
```json
{
  "success": true,
  "enhanced_prompt": "Enhanced prompt with WAN-specific optimization",
  "original_prompt": "Original prompt to enhance",
  "enhancement_source": "qwen_base",
  "processing_time": 2.5,
  "model": "qwen_base",
  "thread_safe": true,
  "enhancement_applied": true,
  "model_was_loaded": true,
  "worker": "wan",
  "note": "Enhanced using Qwen Base model"
}
```

---

## **📊 Performance Optimization**

### **Memory Management**
```python
def optimize_wan_memory():
    """Optimize memory usage for WAN worker"""
    
    # Clear GPU cache before generation
    torch.cuda.empty_cache()
    
    # Monitor VRAM usage
    vram_usage = torch.cuda.memory_allocated() / 1024**3
    if vram_usage > 25:  # GB
        gc.collect()
        torch.cuda.empty_cache()
    
    # Unload Qwen model if not needed
    if not is_qwen_needed():
        unload_qwen_model()
```

### **Thread-Safe Timeouts**
```python
def run_with_timeout(func, timeout_seconds, *args, **kwargs):
    """Thread-safe timeout wrapper using concurrent.futures"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutException(f"Operation timed out after {timeout_seconds} seconds")
```

### **Model Loading Strategy**
```python
class EnhancedWanWorker:
    def __init__(self):
        self.wan_model = None
        self.qwen_model = None
        self.wan_loaded = False
        self.qwen_loaded = False
    
    def ensure_models_loaded(self):
        """Ensure required models are loaded"""
        if not self.wan_loaded:
            self.wan_model = load_wan_model()
            self.wan_loaded = True
        
        # Load Qwen only when needed for enhanced jobs
        if not self.qwen_loaded and needs_enhancement():
            self.qwen_model = load_qwen_model()
            self.qwen_loaded = True
```

---

## **🔍 Error Handling**

### **Common Issues**
```python
def handle_wan_errors(error, job_data):
    """Handle common WAN worker errors"""
    
    if "CUDA out of memory" in str(error):
        # Clear memory and retry with lower quality
        torch.cuda.empty_cache()
        job_data['quality'] = 'fast'
        return retry_job(job_data)
    
    elif "Model not found" in str(error):
        # Reload model
        reload_wan_model()
        return retry_job(job_data)
    
    elif "Invalid reference image" in str(error):
        # Fallback to text-to-video
        job_data.pop('reference_image_url', None)
        return retry_job(job_data)
    
    else:
        # Log unknown error
        log_error(error, job_data)
        return {
            'error': 'Video generation failed',
            'job_id': job_data.get('job_id')
        }
```

### **Quality Degradation**
```python
def degrade_quality_on_error(job_data):
    """Degrade quality settings on error"""
    
    current_quality = job_data.get('quality', 'high')
    
    if current_quality == 'high':
        job_data['quality'] = 'balanced'
        job_data['sample_guide_scale'] = 6.5
        job_data['sample_steps'] = 20
    elif current_quality == 'balanced':
        job_data['quality'] = 'fast'
        job_data['sample_guide_scale'] = 5.0
        job_data['sample_steps'] = 15
    
    return job_data
```

---

## **📈 Monitoring and Logging**

### **Performance Metrics**
```python
def log_video_metrics(job_data, generation_time, video_info):
    """Log WAN worker performance metrics"""
    
    metrics = {
        'worker': 'wan',
        'job_type': job_data.get('job_type'),
        'generation_time': generation_time,
        'video_duration': video_info.get('duration'),
        'video_size': video_info.get('file_size'),
        'resolution': f"{video_info.get('width')}x{video_info.get('height')}",
        'fps': video_info.get('fps'),
        'quality': job_data.get('quality'),
        'reference_mode': job_data.get('reference_mode'),
        'enhancement_applied': job_data.get('enhance_prompt'),
        'vram_usage': torch.cuda.memory_allocated() / 1024**3,
        'timestamp': datetime.now().isoformat()
    }
    
    log_metrics(metrics)
```

### **Quality Monitoring**
```python
def monitor_video_quality(job_data, result):
    """Monitor video generation quality"""
    
    # Check for common quality issues
    if result.get('error'):
        log_quality_issue('video_generation_error', job_data, result)
    
    # Monitor generation times
    if result.get('generation_time', 0) > 300:  # 5 minutes
        log_quality_issue('slow_video_generation', job_data, result)
    
    # Monitor video file size
    if result.get('file_size', 0) < 1000000:  # 1MB
        log_quality_issue('small_video_file', job_data, result)
```

---

## **🎯 Video Output Specifications**

### **Standard Output**
```yaml
Format: MP4 with H.264 encoding
Resolution: 480x832 (portrait) default, configurable
Frame Rate: 16fps default, 12-24fps range
Duration: 5 seconds (83 frames), extendable
File Size: 15-25MB typical for 5-second video
Quality: High-definition, web-optimized
Thumbnail: 256px WEBP mid-frame extraction
```

### **Quality Settings**
```python
QUALITY_SETTINGS = {
    'fast': {
        'sample_guide_scale': 6.5,
        'sample_steps': 25,
        'fps': 16,
        'estimated_time': '25-135 seconds'
    },
    'balanced': {
        'sample_guide_scale': 7.0,
        'sample_steps': 35,
        'fps': 16,
        'estimated_time': '40-180 seconds'
    },
    'high': {
        'sample_guide_scale': 7.5,
        'sample_steps': 50,
        'fps': 16,
        'estimated_time': '40-180 seconds'
    }
}
```

---

## **🚀 Auto-Registration and Deployment**

### **RunPod URL Detection**
```python
def detect_runpod_url():
    """Detect current RunPod URL using official environment variables"""
    try:
        # Method 1: Official RUNPOD_POD_ID (most reliable)
        pod_id = os.environ.get('RUNPOD_POD_ID')
        if pod_id:
            url = f"https://{pod_id}-7860.proxy.runpod.net"
            return url
        
        # Method 2: Fallback to hostname parsing
        hostname = socket.gethostname()
        if 'runpod' in hostname.lower() or len(hostname) > 8:
            pod_id = hostname.split('-')[0] if '-' in hostname else hostname
            url = f"https://{pod_id}-7860.proxy.runpod.net"
            return url
        
        return None
        
    except Exception as e:
        print(f"❌ URL detection error: {e}")
        return None
```

### **Auto-Registration Process**
```python
def auto_register_worker_url():
    """Auto-register worker URL with Supabase"""
    
    # Detect current URL
    detected_url = detect_runpod_url()
    if not detected_url:
        return False
    
    # Register with Supabase
    registration_data = {
        "worker_url": detected_url,
        "auto_registered": True,
        "registration_method": "wan_worker_self_registration",
        "detection_method": "RUNPOD_POD_ID",
        "timestamp": datetime.now().isoformat()
    }
    
    # Call update-worker-url edge function
    response = requests.post(
        f"{os.environ['SUPABASE_URL']}/functions/v1/update-worker-url",
        headers={
            "Authorization": f"Bearer {os.environ['SUPABASE_SERVICE_KEY']}",
            "Content-Type": "application/json"
        },
        json=registration_data,
        timeout=15
    )
    
    return response.status_code == 200
```

---

## **🔄 Future Enhancements**

### **Planned Improvements**
1. **Extended Video Lengths**: Support for 15s, 30s, 60s videos
2. **Video Stitching**: Combine multiple clips with continuity
3. **Advanced Motion**: Better motion control and consistency
4. **Audio Integration**: Add audio generation capabilities

### **Integration Opportunities**
1. **Storyboard System**: Generate videos from storyboard sequences
2. **Character Consistency**: Maintain character appearance across video clips
3. **Scene Transitions**: Smooth transitions between video segments

---

**Note**: This worker is optimized for the OurVidz platform and provides the foundation for video generation capabilities. The Qwen Base integration ensures high-quality prompt enhancement for better video results. The worker automatically registers itself with Supabase and provides comprehensive health monitoring and error handling.
