# OurVidz Worker API Reference

**Last Updated:** July 20, 2025 at 11:45 PM CST  
**Status:** ✅ Production Ready - All 10 Job Types Operational + Compel Integration + Multi-Reference System Live  
**System:** Dual Worker (SDXL + WAN) on RTX 6000 ADA (48GB VRAM)

---

## **🎯 Worker System Overview**

OurVidz operates with a dual-worker architecture managed by a centralized orchestrator:

1. **SDXL Worker** - High-quality image generation with flexible quantities and Compel integration
2. **WAN Worker** - Video generation and enhanced image processing with Qwen 7B enhancement
3. **Dual Orchestrator** - Centralized management and monitoring of both workers

All workers use standardized callback parameters and comprehensive metadata management.

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
    "reference_strength": 0.1-1.0
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
    "enhancement_strategy": "compel" | "fallback" | "none"
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
- **🎯 Compel Integration**: Prompt enhancement with weighted attention
- **Batch Processing**: Efficient multi-image generation

### **🎯 Compel Integration (CURRENT STATUS)**

#### **✅ Working Implementation**
The SDXL worker successfully receives and processes Compel weights:

```python
# Compel usage examples
job_data = {
    "id": "job-123",
    "type": "sdxl_image_high",
    "prompt": "two teenage lovers making out on the couch",
    "user_id": "user-123",
    "compel_enabled": True,
    "compel_weights": "(highly detailed:1.3), (intricate:1.2), (fine details:1.1), (masterpiece:1.2), (best quality:1.3), (professional:1.1), (perfect anatomy:1.2), (realistic:1.3), (natural proportions:1.1), (professional photography:1.2), (studio lighting:1.1), (cinematic:1.1), (high quality:1.3), (detailed:1.3), (perfect anatomy:1.3), (professional photography:1.2)",
    "config": {
        "num_images": 1
    }
}
```

#### **⚠️ Current Issue: CLIP Token Limit Exceeded**
```
Token indices sequence length is longer than the specified maximum sequence length for this model (132 > 77). Running this sequence through the model will result in indexing errors
The following part of your input was truncated because CLIP can only handle sequences up to 77 tokens: ['), ( professional photography : 1. 2 ), ( studio lighting : 1. 1 ), ( cinematic : 1. 1 ), ( high quality : 1. 3 ), ( detailed : 1. 3 ), ( perfect anatomy : 1. 3 ), ( professional photography : 1. 2 )']
```

#### **Current Compel Processing**
```python
def process_compel_weights(self, prompt, weights_config=None):
    """
    Process prompt with Compel weights (simple string concatenation)
    CURRENT ISSUE: Creates token sequences exceeding CLIP's 77-token limit
    """
    if not weights_config:
        return prompt, None
        
    try:
        # Simple string concatenation approach (CAUSES TOKEN LIMIT ISSUES)
        final_prompt = f"{prompt} {weights_config}"
        logger.info(f"✅ Compel weights applied: {prompt} -> {final_prompt}")
        return final_prompt, prompt  # Return enhanced and original
    except Exception as e:
        logger.error(f"❌ Compel processing failed: {e}")
        return prompt, None  # Fallback to original prompt
```

#### **🔧 Required Fix: SDXL-Specific Compel Library Integration**
The current implementation uses string concatenation, which causes CLIP token limit violations. The fix requires SDXL-specific Compel library integration with both `prompt_embeds` and `pooled_prompt_embeds`:

```python
# REQUIRED: SDXL-specific Compel library integration
import compel
from compel import Compel

def process_compel_weights_sdxl(self, prompt, weights_config=None):
    """
    Process prompt with SDXL-specific Compel library integration
    This generates both prompt_embeds and pooled_prompt_embeds for SDXL
    """
    if not weights_config:
        return prompt, None
        
    try:
        # Initialize Compel with both SDXL text encoders and tokenizers
        compel_processor = Compel(
            tokenizers=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoders=[self.pipe.text_encoder, self.pipe.text_encoder_2]
        )
        
        # Build both conditioning tensors for SDXL
        conditioning, pooled_conditioning = compel_processor.build_conditioning_tensor(
            f"{prompt} {weights_config}"
        )
        
        logger.info(f"✅ Compel weights applied with SDXL library integration")
        logger.info(f"🔧 Generated prompt_embeds: {conditioning.shape}")
        logger.info(f"🔧 Generated pooled_prompt_embeds: {pooled_conditioning.shape}")
        
        # Return both conditioning tensors and original prompt
        return (conditioning, pooled_conditioning), prompt
        
    except Exception as e:
        logger.error(f"❌ Compel processing failed: {e}")
        return prompt, None  # Fallback to original prompt
```

#### **🎯 Frontend Optimization Required**
The frontend should generate more concise Compel weights to stay within token limits:

```typescript
// CURRENT: Too many weights (132 tokens)
"(highly detailed:1.3), (intricate:1.2), (fine details:1.1), (masterpiece:1.2), (best quality:1.3), (professional:1.1), (perfect anatomy:1.2), (realistic:1.3), (natural proportions:1.1), (professional photography:1.2), (studio lighting:1.1), (cinematic:1.1), (high quality:1.3), (detailed:1.3), (perfect anatomy:1.3), (professional photography:1.2)"

// OPTIMIZED: Fewer, more impactful weights (~40 tokens)
"(masterpiece:1.3), (best quality:1.2), (perfect anatomy:1.2), (professional:1.1)"
```

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
- **Model**: WAN 2.1.1.3B
- **Pipeline**: Video generation and enhanced image processing
- **VRAM**: 48GB RTX 6000 ADA
- **Enhancement**: Qwen 7B model for prompt enhancement

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
- **Enhanced Processing**: 7B model variants for improved quality
- **Reference Support**: Image-to-image for video start/end frames
- **Seed Control**: Reproducible generation (no negative prompts)
- **Path Consistency**: Fixed video path handling
- **🤖 Qwen 7B Enhancement**: AI-powered prompt enhancement

### **Video Generation with Reference Frames (WAN 1.3B Model)**

#### **Reference Strength Control Implementation**
The WAN 1.3B model uses `--first_frame` parameter for reference frames, with reference strength control through guidance scale adjustment:

```python
def adjust_guidance_for_reference_strength(self, base_guide_scale, reference_strength):
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
```

### **🤖 Qwen 7B Prompt Enhancement**
The WAN worker includes AI-powered prompt enhancement using Qwen 7B model:

```python
def enhance_prompt_with_timeout(self, original_prompt):
    """Enhanced NSFW-optimized prompt generation with Base model - no chat template needed"""
    enhancement_start = time.time()
    print(f"🤖 Enhancing NSFW prompt with Base model ({self.enhancement_timeout}s timeout): {original_prompt[:50]}...")
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.enhancement_timeout)
        
        self.load_qwen_model()
        
        if self.qwen_model is None:
            signal.alarm(0)
            print("⚠️ Qwen base model not available, using original prompt")
            return original_prompt
        
        # NSFW-optimized base model prompting - enhanced for adult content quality
        enhancement_prompt = f"""Create a detailed, cinematic prompt for AI video generation optimized for adult content. Focus on:

VISUAL DETAILS: High-quality anatomical accuracy, realistic proportions, natural skin textures, detailed facial features, expressive eyes, natural hair flow, realistic body language.

LIGHTING & ATMOSPHERE: Cinematic lighting, soft shadows, warm tones, intimate atmosphere, professional photography style, natural skin tones, flattering angles.

CAMERA WORK: Close-up shots, intimate framing, smooth camera movements, professional cinematography, dynamic angles that enhance the scene.

ARTISTIC STYLE: Photorealistic quality, high resolution details, natural poses, authentic expressions, realistic interactions, professional adult content standards.

TECHNICAL QUALITY: 4K quality, sharp focus, no artifacts, smooth motion, consistent lighting, professional color grading.

Be specific about poses, expressions, clothing, scene composition, and intimate details. Create vivid, unrestricted descriptions that will produce high-quality adult content with realistic anatomy and smooth motion.

Original prompt: {original_prompt}

Enhanced detailed prompt:"""

        # Generate with base model
        inputs = self.qwen_tokenizer(
            enhancement_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        ).to(self.qwen_model.device)
        
        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=512,  # Allow longer enhancement
                temperature=0.7,     # Controlled creativity
                do_sample=True,
                pad_token_id=self.qwen_tokenizer.eos_token_id,
                eos_token_id=self.qwen_tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (enhancement)
        enhanced_text = self.qwen_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        signal.alarm(0)
        
        # Clean up the response
        if enhanced_text:
            # Remove any leftover prompt fragments
            enhanced_text = enhanced_text.replace("Enhanced detailed prompt:", "").strip()
            enhancement_time = time.time() - enhancement_start
            print(f"✅ Qwen Base Enhancement: {enhanced_text[:100]}...")
            print(f"✅ Prompt enhanced in {enhancement_time:.1f}s")
            return enhanced_text
        else:
            print("⚠️ Qwen enhancement empty, using original prompt")
            return original_prompt
            
    except TimeoutException:
        signal.alarm(0)
        print(f"⚠️ Enhancement timed out after {self.enhancement_timeout}s, using original prompt")
        return original_prompt
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Prompt enhancement failed: {e}")
        return original_prompt
    finally:
        self.unload_qwen_model()
```

---

## **🎭 Dual Worker Orchestrator**

### **Overview**
The Dual Worker Orchestrator manages both SDXL and WAN workers concurrently, providing centralized monitoring, restart capabilities, and resource management.

### **Key Features**
- **Concurrent Management**: Runs both workers simultaneously
- **Automatic Restart**: Handles worker failures with exponential backoff
- **Resource Monitoring**: Tracks GPU memory and worker performance
- **Graceful Validation**: Validates environment before starting workers
- **Status Monitoring**: Real-time worker status and job tracking

### **Worker Configurations**
```python
self.workers = {
    'sdxl': {
        'script': 'sdxl_worker.py',
        'name': 'LUSTIFY SDXL Worker',
        'queue': 'sdxl_queue',
        'job_types': ['sdxl_image_fast', 'sdxl_image_high'],
        'expected_vram': '10-15GB',
        'restart_delay': 10,
        'generation_time': '3-8s',
        'status': 'Working ✅'
    },
    'wan': {
        'script': 'wan_worker.py', 
        'name': 'Enhanced WAN Worker (Qwen 7B + FLF2V/T2V)',
        'queue': 'wan_queue',
        'job_types': ['image_fast', 'image_high', 'video_fast', 'video_high',
                     'image7b_fast_enhanced', 'image7b_high_enhanced', 
                     'video7b_fast_enhanced', 'video7b_high_enhanced'],
        'expected_vram': '15-30GB',
        'restart_delay': 15,
        'generation_time': '67-294s',
        'status': 'Qwen 7B Enhancement + FLF2V/T2V Tasks ✅'
    }
}
```

### **Environment Validation**
```python
def validate_environment(self):
    """Validate environment for dual worker operation"""
    logger.info("🔍 Validating dual worker environment...")
    
    # CRITICAL: Check PyTorch version first (prevent cascade failures)
    try:
        import torch
        current_version = torch.__version__
        current_cuda = torch.version.cuda
        
        logger.info(f"🔧 PyTorch: {current_version}")
        logger.info(f"🔧 CUDA: {current_cuda}")
        
        # Verify we have the stable working versions
        if not current_version.startswith('2.4.1'):
            logger.error(f"❌ WRONG PyTorch version: {current_version} (need 2.4.1+cu124)")
            logger.error("❌ DO NOT PROCEED - version cascade detected!")
            return False
            
        if current_cuda != '12.4':
            logger.error(f"❌ WRONG CUDA version: {current_cuda} (need 12.4)")
            logger.error("❌ DO NOT PROCEED - CUDA version mismatch!")
            return False
            
        logger.info("✅ PyTorch/CUDA versions confirmed stable")
        
    except ImportError:
        logger.error("❌ PyTorch not available")
        return False
    
    # Check Python files exist
    missing_files = []
    for worker_id, config in self.workers.items():
        script_path = Path(config['script'])
        if not script_path.exists():
            missing_files.append(config['script'])
            logger.error(f"❌ Missing worker script: {config['script']}")
    
    if missing_files:
        logger.error(f"❌ Missing worker scripts: {missing_files}")
        return False
    else:
        logger.info("✅ All worker scripts found")
        
    # Check GPU
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ GPU: {device_name} ({total_vram:.1f}GB)")
            
            if total_vram < 40:
                logger.warning(f"⚠️ GPU has {total_vram:.1f}GB, dual workers need 45GB+ for concurrent operation")
            else:
                logger.info(f"✅ GPU capacity sufficient for dual workers")
                
        else:
            logger.error("❌ CUDA not available")
            return False
            
    except Exception as e:
        logger.error(f"❌ GPU check failed: {e}")
        return False
        
    # Check SDXL imports (graceful handling - let workers manage their own imports)
    try:
        from diffusers import StableDiffusionXLPipeline
        logger.info("✅ SDXL imports confirmed working")
    except ImportError as e:
        logger.warning(f"⚠️ SDXL imports failed in orchestrator: {e}")
        logger.info("📝 Will let SDXL worker handle its own imports")
        # Don't fail here - let workers handle their own dependencies
        
    # Check environment variables
    required_vars = [
        'SUPABASE_URL', 
        'SUPABASE_SERVICE_KEY', 
        'UPSTASH_REDIS_REST_URL', 
        'UPSTASH_REDIS_REST_TOKEN'
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {missing_vars}")
        return False
    else:
        logger.info("✅ All environment variables configured")
        
    # Validate parameter consistency in worker files
    logger.info("🔧 Validating parameter consistency across workers...")
    wan_script_path = Path('wan_worker.py')
    sdxl_script_path = Path('sdxl_worker.py')
    
    consistency_issues = []
    
    if wan_script_path.exists():
        with open(wan_script_path, 'r') as f:
            wan_content = f.read()
            # Check for consistent parameter naming
            if "'job_id':" in wan_content and "'assets':" in wan_content:
                logger.info("✅ WAN worker uses consistent parameter naming (job_id, assets)")
            else:
                consistency_issues.append("WAN worker parameter naming inconsistent")
            
            # Check for FLF2V/T2V task support
            if "flf2v-14B" in wan_content and "t2v-14B" in wan_content:
                logger.info("✅ WAN worker supports FLF2V/T2V tasks")
            else:
                consistency_issues.append("WAN worker missing FLF2V/T2V task support")
            
            # Check for correct parameter names
            if "--first_frame" in wan_content and "--last_frame" in wan_content:
                logger.info("✅ WAN worker uses correct FLF2V parameter names (--first_frame, --last_frame)")
            else:
                consistency_issues.append("WAN worker missing correct FLF2V parameter names")
    
    if sdxl_script_path.exists():
        with open(sdxl_script_path, 'r') as f:
            sdxl_content = f.read()
            # Check for consistent parameter naming
            if "'job_id':" in sdxl_content and "'assets':" in sdxl_content:
                logger.info("✅ SDXL worker uses consistent parameter naming (job_id, assets)")
            else:
                consistency_issues.append("SDXL worker parameter naming inconsistent")
    
    if consistency_issues:
        logger.error(f"❌ Parameter consistency issues: {consistency_issues}")
        return False
    else:
        logger.info("✅ Parameter naming consistency validated")
        
    logger.info("✅ Environment validation passed")
    return True
```

### **Status Monitoring**
```python
def status_monitor(self):
    """Background thread to monitor system status"""
    logger.info("📊 Starting status monitor...")
    
    while not self.shutdown_event.is_set():
        try:
            # Check worker processes
            active_workers = []
            total_jobs = 0
            
            for worker_id, worker_info in self.processes.items():
                if worker_info['process'].poll() is None:
                    uptime = time.time() - worker_info['start_time']
                    job_count = worker_info['job_count']
                    total_jobs += job_count
                    active_workers.append(f"{worker_id}({uptime:.0f}s/{job_count}j)")
            
            if active_workers:
                logger.info(f"💚 Active workers: {', '.join(active_workers)} | Total jobs: {total_jobs}")
            else:
                logger.warning("⚠️ No active workers")
            
            # Check GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    utilization = (allocated / total) * 100
                    logger.info(f"🔥 GPU Memory: {allocated:.1f}GB / {total:.0f}GB ({utilization:.1f}% used)")
            except:
                pass
                
            # Wait before next check
            time.sleep(60)  # Status check every minute
            
        except Exception as e:
            logger.error(f"❌ Status monitor error: {e}")
            time.sleep(30)
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

#### **SDXL Generation with Compel**
```python
if job_type.startswith("sdxl_"):
    # Extract Compel parameters
    compel_enabled = job_data.get("compel_enabled", False)
    compel_weights = job_data.get("compel_weights", "")
    
    # Process Compel enhancement
    if compel_enabled and compel_weights:
        final_prompt, original_prompt = process_compel_weights(prompt, compel_weights)
    else:
        final_prompt = prompt
    
    # SDXL generation with flexible quantities
    num_images = config.get("num_images", 1)
    results = []
    
    for i in range(num_images):
        # Generate with seed for consistency
        seed = config.get("seed", random.randint(1, 999999999))
        result = generate_sdxl_image(final_prompt, config, seed)
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
        "num_images": len(uploaded_assets),
        "compel_enabled": compel_enabled,
        "compel_weights": compel_weights,
        "enhancement_strategy": enhancement_strategy
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

## **🚀 Recent Updates (July 20, 2025)**

### **Major Enhancements**
1. **🎯 Compel Integration**: SDXL worker now supports Compel prompt enhancement with weighted attention
2. **🤖 Qwen 7B Enhancement**: WAN worker includes AI-powered prompt enhancement
3. **🎭 Dual Orchestrator**: Centralized management of both workers with monitoring and restart capabilities
4. **Standardized Callback Parameters**: Consistent `job_id`, `assets` array across all workers
5. **Enhanced Negative Prompts**: Intelligent generation for SDXL with multi-party scene detection
6. **Seed Support**: User-controlled seeds for reproducible generation
7. **Flexible SDXL Quantities**: User-selectable 1, 3, or 6 images per batch
8. **Reference Image Support**: Optional image-to-image with type and strength control
9. **Video Reference Frame Support**: I2V-style generation with start reference frame for WAN 1.3B model
10. **Comprehensive Error Handling**: Enhanced debugging and error tracking
11. **Metadata Consistency**: Improved data flow and storage
12. **Path Consistency Fix**: Fixed video path handling for WAN workers

### **Performance Improvements**
- Optimized batch processing for multi-image SDXL jobs
- Enhanced error recovery and retry mechanisms
- Improved Redis queue management
- Better resource utilization tracking
- AI-powered prompt enhancement for higher quality output

### **Developer Experience**
- Enhanced API documentation and examples
- Comprehensive debugging information
- Backward compatibility preservation
- Clear error messages and status codes
- Centralized worker management

### **Backward Compatibility**
- All existing job types remain functional
- Legacy metadata fields are preserved
- Single-reference workflows continue to work
- Non-reference generation unchanged
- Compel integration is optional and backward compatible

---

## **⚠️ Current Issues & Required Fixes**

### **🔧 Compel Integration Issues**

#### **1. CLIP Token Limit Exceeded**
**Status**: ⚠️ **CRITICAL** - Compel weights causing token limit violations
**Impact**: Only ~60% of Compel weights are being processed due to truncation
**Solution**: Implement proper Compel library integration instead of string concatenation

#### **2. Frontend Weight Optimization**
**Status**: ⚠️ **HIGH** - Too many Compel weights generated
**Impact**: 132 tokens generated when CLIP limit is 77 tokens
**Solution**: Optimize frontend to generate fewer, more impactful weights

#### **3. SDXL Worker Compel Processing**
**Status**: ⚠️ **MEDIUM** - Using string concatenation instead of proper library
**Impact**: Token limit violations and partial processing
**Solution**: Update SDXL worker to use Compel library's native processing

### **🎯 Immediate Action Plan**

#### **Priority 1: Fix SDXL Worker Compel Integration**
```python
# REQUIRED: Replace string concatenation with SDXL-specific Compel library usage
def process_compel_weights_sdxl(self, prompt, weights_config=None):
    """Process prompt with SDXL-specific Compel library integration"""
    if not weights_config:
        return prompt, None
        
    try:
        # Initialize Compel with both SDXL text encoders and tokenizers
        compel_processor = Compel(
            tokenizers=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoders=[self.pipe.text_encoder, self.pipe.text_encoder_2]
        )
        
        # Build both conditioning tensors for SDXL
        conditioning, pooled_conditioning = compel_processor.build_conditioning_tensor(
            f"{prompt} {weights_config}"
        )
        
        logger.info(f"✅ Compel weights applied with SDXL library integration")
        return (conditioning, pooled_conditioning), prompt
        
    except Exception as e:
        logger.error(f"❌ Compel processing failed: {e}")
        return prompt, None  # Fallback to original prompt
```

#### **Priority 2: Optimize Frontend Compel Weights**
```typescript
// OPTIMIZED: Generate fewer, more impactful weights
const OPTIMIZED_QUICK_BOOSTS = [
  { id: 'masterpiece', label: 'Masterpiece', weight: 1.3 },
  { id: 'best_quality', label: 'Best Quality', weight: 1.2 },
  { id: 'perfect_anatomy', label: 'Perfect Anatomy', weight: 1.2 },
  { id: 'professional', label: 'Professional', weight: 1.1 }
];

// Target: ~40 tokens instead of 132 tokens
```

#### **Priority 3: Add Token Count Validation**
```typescript
// Add token counting to prevent exceeding CLIP limits
function countTokens(text: string): number {
  // Implement token counting logic
  return text.split(' ').length; // Simplified for now
}

function validateCompelWeights(prompt: string, weights: string): boolean {
  const totalTokens = countTokens(`${prompt} ${weights}`);
  return totalTokens <= 77; // CLIP limit
}
```

---

## **🔧 SDXL Worker Repository Changes Required**

### **Overview**
The SDXL worker needs critical updates to fix the Compel integration token limit issues. The current implementation uses string concatenation which causes CLIP token limit violations (132 > 77 tokens).

### **Required Changes**

#### **1. Update Compel Processing Function**

**File**: `sdxl_worker.py`  
**Function**: `process_compel_weights`  
**Current Issue**: String concatenation causing token limit violations

**Replace this function:**
```python
def process_compel_weights(self, prompt, weights_config=None):
    """
    Process prompt with Compel weights (simple string concatenation)
    CURRENT ISSUE: Creates token sequences exceeding CLIP's 77-token limit
    """
    if not weights_config:
        return prompt, None
        
    try:
        # Simple string concatenation approach (CAUSES TOKEN LIMIT ISSUES)
        final_prompt = f"{prompt} {weights_config}"
        logger.info(f"✅ Compel weights applied: {prompt} -> {final_prompt}")
        return final_prompt, prompt  # Return enhanced and original
    except Exception as e:
        logger.error(f"❌ Compel processing failed: {e}")
        return prompt, None  # Fallback to original prompt
```

**With this implementation:**
```python
def process_compel_weights(self, prompt, weights_config=None):
    """
    Process prompt with SDXL-specific Compel library integration
    FIXES: Generate both prompt_embeds and pooled_prompt_embeds for SDXL
    """
    if not weights_config:
        return prompt, None
        
    try:
        # Import Compel at the top of the file
        import compel
        from compel import Compel
        
        # Initialize Compel with both SDXL text encoders and tokenizers
        compel_processor = Compel(
            tokenizers=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoders=[self.pipe.text_encoder, self.pipe.text_encoder_2]
        )
        
        # Build both conditioning tensors for SDXL
        conditioning, pooled_conditioning = compel_processor.build_conditioning_tensor(
            f"{prompt} {weights_config}"
        )
        
        logger.info(f"✅ Compel weights applied with SDXL library integration")
        logger.info(f"📝 Original prompt: {prompt}")
        logger.info(f"🎯 Compel weights: {weights_config}")
        logger.info(f"🔧 Generated prompt_embeds: {conditioning.shape}")
        logger.info(f"🔧 Generated pooled_prompt_embeds: {pooled_conditioning.shape}")
        
        # Return both conditioning tensors and original prompt
        return (conditioning, pooled_conditioning), prompt
        
    except Exception as e:
        logger.error(f"❌ Compel processing failed: {e}")
        logger.info(f"🔄 Falling back to original prompt: {prompt}")
        return prompt, None  # Fallback to original prompt
```

#### **2. Update SDXL Generation Function**

**File**: `sdxl_worker.py`  
**Function**: `generate_sdxl_image` or similar generation function

**Find the generation call and update it to handle Compel conditioning:**

**Current (problematic):**
```python
# Current generation call
result = self.pipe(
    prompt=final_prompt,  # This is a string with concatenated weights
    **config
).images[0]
```

**Replace with:**
```python
# Updated generation call with proper SDXL Compel handling
if isinstance(final_prompt, tuple) and len(final_prompt) == 2:
    # Compel conditioning tensors were returned (prompt_embeds, pooled_prompt_embeds)
    prompt_embeds, pooled_prompt_embeds = final_prompt
    result = self.pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,  # SDXL requires this
        **config
    ).images[0]
    logger.info("✅ Generated with Compel conditioning tensors (SDXL)")
elif isinstance(final_prompt, torch.Tensor):
    # Legacy single conditioning tensor (fallback)
    result = self.pipe(
        prompt_embeds=final_prompt,  # Use single conditioning tensor
        **config
    ).images[0]
    logger.info("✅ Generated with single Compel conditioning tensor (legacy)")
else:
    # Fallback to string prompt (no Compel or Compel failed)
    result = self.pipe(
        prompt=final_prompt,  # Use string prompt
        **config
    ).images[0]
    logger.info("✅ Generated with string prompt (no Compel)")
```

#### **3. Add Compel Import**

**File**: `sdxl_worker.py`  
**Location**: Top of file with other imports

**Add these imports:**
```python
import compel
from compel import Compel
```

**Full import section should look like:**
```python
import os
import json
import time
import requests
import uuid
import torch
import gc
import io
import sys
import compel  # ADD THIS
from compel import Compel  # ADD THIS
sys.path.append('/workspace/python_deps/lib/python3.11/site-packages')
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLPipeline
import logging
```

#### **4. Update Error Handling**

**File**: `sdxl_worker.py`  
**Location**: In the main job processing function

**Add better error handling for Compel processing:**

```python
# In the main job processing function, update the Compel section:
if compel_enabled and compel_weights:
    logger.info(f"🎯 Compel enhancement enabled: {compel_weights}")
    
    try:
        # Apply Compel weights to the prompt (proper library integration)
        final_prompt, original_prompt = self.process_compel_weights(prompt, compel_weights)
        compel_success = True
        logger.info(f"✅ Compel processing successful")
        
    except Exception as e:
        logger.error(f"❌ Compel processing failed: {e}")
        final_prompt = prompt  # Fallback to original prompt
        original_prompt = None
        compel_success = False
        logger.info(f"🔄 Using original prompt due to Compel failure: {prompt}")
    
    # Log the final prompt being used
    if isinstance(final_prompt, torch.Tensor):
        logger.info(f"🎯 Using Compel conditioning tensor for generation")
    else:
        logger.info(f"🎯 Using Compel-enhanced prompt: {final_prompt}")
else:
    final_prompt = prompt
    original_prompt = None
    compel_success = False
    logger.info(f"🎯 Using standard prompt (no Compel): {prompt}")
```

#### **5. Update Metadata in Callback**

**File**: `sdxl_worker.py`  
**Location**: In the callback section

**Update the metadata to include Compel processing status:**

```python
# In the callback metadata section:
callback_metadata = {
    "seed": config.get("seed"),
    "generation_time": generation_time,
    "num_images": len(uploaded_assets),
    "compel_enabled": compel_enabled,
    "compel_weights": compel_weights if compel_enabled else None,
    "compel_success": compel_success if compel_enabled else False,
    "enhancement_strategy": "compel" if compel_success else "fallback" if compel_enabled else "none",
    "original_prompt": original_prompt,
    "final_prompt_type": "conditioning_tensor" if isinstance(final_prompt, torch.Tensor) else "string"
}
```

### **Testing Instructions**

#### **1. Test Compel Integration**
```bash
# Test with a simple prompt and Compel weights
curl -X POST http://localhost:8000/test-compel \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "beautiful woman",
    "compel_weights": "(masterpiece:1.3), (best quality:1.2)"
  }'
```

#### **2. Verify Token Count**
- Check logs for "Token indices sequence length" warnings
- Should see "Using Compel conditioning tensor for generation"
- No more token limit violations

#### **3. Test Fallback Behavior**
- Test with invalid Compel weights
- Should fallback to original prompt
- Should log "Using string prompt (no Compel)"

### **Expected Log Output**

**Successful Compel Processing:**
```
🎯 Compel enhancement enabled: (masterpiece:1.3), (best quality:1.2)
✅ Compel weights applied with SDXL library integration
📝 Original prompt: beautiful woman
🎯 Compel weights: (masterpiece:1.3), (best quality:1.2)
🔧 Generated prompt_embeds: torch.Size([1, 77, 2048])
🔧 Generated pooled_prompt_embeds: torch.Size([1, 1280])
✅ Compel processing successful
🎯 Using Compel conditioning tensors for SDXL generation
✅ Generated with Compel conditioning tensors (SDXL)
```

**Fallback Behavior:**
```
🎯 Compel enhancement enabled: (invalid:weights)
❌ Compel processing failed: [error details]
🔄 Falling back to original prompt: beautiful woman
🔄 Using original prompt due to Compel failure: beautiful woman
🎯 Using standard prompt (no Compel): beautiful woman
✅ Generated with string prompt (no Compel)
```

**Legacy Single Tensor Fallback:**
```
🎯 Compel enhancement enabled: (masterpiece:1.3), (best quality:1.2)
✅ Compel weights applied with library integration (legacy)
📝 Original prompt: beautiful woman
🎯 Compel weights: (masterpiece:1.3), (best quality:1.2)
✅ Compel processing successful
🎯 Using single Compel conditioning tensor for generation
✅ Generated with single Compel conditioning tensor (legacy)
```

### **Dependencies**

**Ensure these are installed in the SDXL worker environment:**
```bash
pip install compel==0.1.8
pip install pyparsing==3.0.9
```

**Verify PYTHONPATH is set:**
```bash
export PYTHONPATH=/workspace/python_deps/lib/python3.11/site-packages:$PYTHONPATH
```

### **Validation Checklist**

- [ ] Compel library imports successfully
- [ ] `process_compel_weights` function updated with SDXL-specific library integration
- [ ] Generation function handles both prompt_embeds and pooled_prompt_embeds for SDXL
- [ ] Error handling includes fallback to original prompt
- [ ] Metadata includes Compel processing status and tensor types
- [ ] No more "Token indices sequence length" warnings in logs
- [ ] Compel weights are properly applied without token limit violations
- [ ] Both conditioning tensors are generated and used correctly
- [ ] Legacy single tensor fallback works for backward compatibility

### **Rollback Plan**

If issues occur, the worker can be rolled back by:
1. Reverting to the string concatenation approach
2. Disabling Compel processing temporarily
3. Using the original prompt without enhancement

The fallback mechanisms ensure the worker continues to function even if Compel processing fails.

---

## **✅ Production Status**

### **Active Components**
- **🎭 Dual Orchestrator**: Main production controller managing both workers
- **🎨 SDXL Worker**: Fast image generation with batch support (1, 3, or 6 images) and Compel integration
- **🎬 Enhanced WAN Worker**: Video generation with Qwen 7B enhancement and comprehensive reference frame support

### **Testing Status**
- **SDXL Jobs**: ✅ Both job types tested and working
- **WAN Jobs**: ✅ All 8 job types tested and working
- **Reference Frames**: ✅ All 5 reference modes tested and working
- **Compel Integration**: ⚠️ **PARTIAL** - Working but with token limit issues
- **Qwen 7B Enhancement**: ✅ Tested and working with timeout protection
- **Performance Baselines**: ✅ Real data established for all jobs

### **System Capabilities**
- **✅ 10 Job Types**: All job types operational
- **✅ 5 Reference Modes**: Complete reference frame support
- **✅ Batch Processing**: SDXL supports 1, 3, or 6 images
- **✅ AI Enhancement**: WAN enhanced variants with Qwen 7B
- **⚠️ Compel Integration**: SDXL prompt enhancement working but needs optimization
- **✅ Error Recovery**: Robust error handling and fallback mechanisms
- **✅ Performance Monitoring**: Comprehensive metrics and logging
- **✅ Centralized Management**: Dual orchestrator with monitoring and restart capabilities

The OurVidz Worker system is **production-ready** with comprehensive reference frame support, AI-powered prompt enhancement, Compel integration (needing optimization), robust error handling, and optimized performance for high-quality AI content generation! 🎯 