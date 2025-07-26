# OurVidz Worker API Reference

**Last Updated:** July 23, 2025 at 7:15 PM CST  
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
    "enhanced_prompt": "string"
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

#### **Current Compel Processing (Compel 2.x+)**
```python
def process_compel_weights(self, prompt, weights_config=None):
    """
    Process prompt with proper Compel library integration for SDXL.
    FIXED: Use requires_pooled=[False, True] for Compel 2.1.1 with SDXL.
    """
    if not weights_config:
        return prompt, None
    try:
        if not self.model_loaded:
            self.load_model()
        logger.info(f"🔧 Initializing Compel 2.1.1 with SDXL encoders - FIXED SYNTAX")
        compel_processor = Compel(
            tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
            text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
            requires_pooled=[False, True]  # ✅ FIXED: List instead of boolean for Compel 2.1.1
        )
        logger.info(f"✅ Compel processor initialized successfully with FIXED SDXL syntax")
        combined_prompt = f"{prompt} {weights_config}"
        logger.info(f"📝 Combined prompt: {combined_prompt}")
        # Always unpack as a tuple
        prompt_embeds, pooled_prompt_embeds = compel_processor(combined_prompt)
        logger.info(f"✅ Compel weights applied with proper SDXL library integration")
        logger.info(f"📝 Original prompt: {prompt}")
        logger.info(f"🎯 Compel weights: {weights_config}")
        logger.info(f"🔧 Generated prompt_embeds: {prompt_embeds.shape}")
        logger.info(f"🔧 Generated pooled_prompt_embeds: {pooled_prompt_embeds.shape}")
        return (prompt_embeds, pooled_prompt_embeds), prompt
    except Exception as e:
        logger.error(f"❌ Compel processing failed: {e}")
        logger.info(f"🔄 Falling back to original prompt: {prompt}")
        return prompt, None  # Fallback to original prompt
```

#### **Critical Note:**
- **For SDXL with Compel 2.x, always unpack the result of `compel_processor(...)` as a tuple:**
  ```python
  prompt_embeds, pooled_prompt_embeds = compel_processor(combined_prompt)
  ```
- **Do NOT check for or handle a single tensor or boolean for SDXL.**
- **Remove any fallback logic for single tensor/boolean for SDXL.**
- If you get `'bool' object is not iterable'`, Compel did not return a tuple—this is a sign of a misconfiguration, version mismatch, or a Compel bug.
- This is required for SDXL with `requires_pooled=True`.

#### **Summary Table: Compel 2.x SDXL API**
| Symptom/Error                  | Cause                                 | Fix                                 |
|--------------------------------|---------------------------------------|-------------------------------------|
| `'bool' object is not iterable'` | Not unpacking Compel output as tuple  | Always unpack: `a, b = compel(...)` |
| `unexpected keyword argument`    | Using keyword args for encoders       | Use positional args                 |

#### **Troubleshooting & Learnings**
- For SDXL, always expect a tuple and unpack it directly.
- Do not try to handle a single tensor or boolean for SDXL.
- If you see `'bool' object is not iterable'`, you are not unpacking the tuple or Compel is misconfigured.
- If you see `unexpected keyword argument`, you are using keyword args for encoders—switch to positional.
- **If you see duplicate or conflicting weights in logs, check that your weights are normalized and deduplicated as above.**
- Always check your Compel version (`import compel; print(compel.__version__)`).
- For SDXL, always use positional arguments for encoders and `requires_pooled=True`.

#### **Validation Checklist**
- [x] Compel library imports successfully
- [x] `process_compel_weights` uses `requires_pooled=[False, True]` for SDXL
- [x] Always unpacks Compel output as a tuple for SDXL
- [x] No fallback logic for single tensor/boolean for SDXL
- [x] Generation function handles both prompt_embeds and pooled_prompt_embeds for SDXL
- [x] Error handling includes fallback to original prompt
- [x] Metadata includes Compel processing status and tensor types
- [x] No more 'bool object is not iterable' or argument errors in logs
- [x] Compel weights are properly applied without token limit violations
- [x] Both conditioning tensors are generated and used correctly
- [x] Legacy single tensor fallback works for backward compatibility (non-SDXL)

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
- **🤖 Qwen 7B Enhancement**: AI-powered prompt enhancement with multiple strategies
- **💬 Chat-Based Enhancement**: Conversational prompt enhancement with context memory
- **🔄 Robust Fallbacks**: Automatic fallback to base model if chat enhancement fails

### **Video Generation with Reference Frames (WAN 1.3B Model)**

#### **Current Implementation (Updated)**
The WAN 1.3B model uses the `t2v-1.3B` task for all video generation, including reference frames:

- **Standard Video**: `t2v-1.3B` task (no reference frames)
- **Single Reference**: `t2v-1.3B` task with `--image` parameter
- **Start Frame**: `t2v-1.3B` task with `--first_frame` parameter  
- **End Frame**: `t2v-1.3B` task with `--last_frame` parameter
- **Both Frames**: `t2v-1.3B` task with `--first_frame` and `--last_frame` parameters

**Note**: The `flf2v-14B` task is not currently used in the implementation.

#### **Reference Strength Control Implementation**
The WAN 1.3B model uses reference frame parameters with reference strength control through guidance scale adjustment:

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

# Determine reference frame mode and task type (1.3B Model)
if single_reference_url and not start_reference_url and not end_reference_url:
    # Single reference frame mode (I2V-style)
    task_type = "t2v-1.3B"  # ✅ CORRECT: T2V with --image parameter
    print(f"🎬 Using T2V task with single reference frame (1.3B model)")
elif start_reference_url and end_reference_url:
    # Both frames mode (start + end)
    task_type = "t2v-1.3B"  # ✅ CORRECT: T2V with --first_frame + --last_frame
    print(f"🎬 Using T2V task with both reference frames (1.3B model)")
elif start_reference_url and not end_reference_url:
    # Start frame only mode
    task_type = "t2v-1.3B"  # ✅ CORRECT: T2V with --first_frame parameter
    print(f"🎬 Using T2V task with start frame reference (1.3B model)")
elif end_reference_url and not start_reference_url:
    # End frame only mode
    task_type = "t2v-1.3B"  # ✅ CORRECT: T2V with --last_frame parameter
    print(f"🎬 Using T2V task with end frame reference (1.3B model)")
else:
    # Standard generation (no reference frames)
    task_type = "t2v-1.3B"  # ✅ CORRECT: Text-to-Video standard
    print(f"🎬 Using T2V task for standard video generation")

print(f"🎯 Reference strength: {reference_strength} → Guidance scale: {adjusted_guide_scale}")
```

### **🤖 Qwen 7B Prompt Enhancement (Enhanced with Chat Support)**
The WAN worker includes AI-powered prompt enhancement using Qwen 7B models with multiple enhancement strategies:

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

#### **🤖 Chat-Based Prompt Enhancement (NEW)**
The WAN worker now supports conversational prompt enhancement using the Qwen 2.5-7B Instruct model:

```python
def enhance_prompt_with_chat(self, original_prompt, session_id=None, conversation_context=None):
    """Enhanced prompt generation using Instruct model with conversation memory"""
    enhancement_start = time.time()
    print(f"💬 Enhancing prompt with Instruct model: {original_prompt[:50]}...")
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.enhancement_timeout)
        
        if not self.load_qwen_instruct_model():
            print("⚠️ Instruct model not available, falling back to Base model")
            return self.enhance_prompt_with_timeout(original_prompt)
        
        # Build conversation for instruct model
        system_prompt = """You are an expert AI prompt engineer specializing in cinematic and adult content generation.

Your role is to transform simple prompts into detailed, cinematic descriptions while maintaining anatomical accuracy and realism for adult content.

Focus on:
- High-quality visual details and realistic proportions
- Cinematic lighting and professional photography style  
- Specific poses, expressions, and scene composition
- Technical quality like 4K resolution and smooth motion

Always respond with enhanced prompts that are detailed, specific, and optimized for AI generation."""

        # Format conversation for Instruct model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please enhance this prompt for AI video generation: {original_prompt}"}
        ]
        
        if conversation_context:
            messages.insert(1, {"role": "user", "content": f"Context: {conversation_context}"})
        
        # Apply chat template for Instruct model
        formatted_prompt = self.qwen_instruct_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.qwen_instruct_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.qwen_instruct_model.device)
        
        with torch.no_grad():
            outputs = self.qwen_instruct_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.qwen_instruct_tokenizer.eos_token_id,
                eos_token_id=self.qwen_instruct_tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        enhanced_text = self.qwen_instruct_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        signal.alarm(0)
        
        if enhanced_text:
            enhancement_time = time.time() - enhancement_start
            print(f"✅ Instruct Enhancement: {enhanced_text[:100]}...")
            print(f"✅ Prompt enhanced with Instruct model in {enhancement_time:.1f}s")
            return enhanced_text
        else:
            print("⚠️ Instruct enhancement empty, falling back to Base")
            return self.enhance_prompt_with_timeout(original_prompt)
            
    except TimeoutException:
        signal.alarm(0)
        print(f"⚠️ Instruct enhancement timed out, falling back to Base")
        return self.enhance_prompt_with_timeout(original_prompt)
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Instruct enhancement failed: {e}, falling back to Base")
        return self.enhance_prompt_with_timeout(original_prompt)
    finally:
        self.unload_qwen_instruct_model()
```

#### **🎯 Enhancement Type Selection**
The WAN worker supports three enhancement strategies controlled via metadata:

```python
def enhance_prompt(self, original_prompt, enhancement_type="instruct", session_id=None, conversation_context=None):
    """Enhanced prompt with retry logic and model selection"""
    print(f"🤖 Starting enhancement for: {original_prompt[:50]}... (type: {enhancement_type})")
    
    for attempt in range(self.max_enhancement_attempts):
        try:
            print(f"🔄 Enhancement attempt {attempt + 1}/{self.max_enhancement_attempts}")
            
            # Choose enhancement method based on type
            if enhancement_type == "chat" or enhancement_type == "instruct_chat":
                enhanced = self.enhance_prompt_with_chat(original_prompt, session_id, conversation_context)
            else:
                # Use existing Base model enhancement (preserves current functionality)
                enhanced = self.enhance_prompt_with_timeout(original_prompt)
            
            if enhanced and enhanced.strip() != original_prompt.strip():
                print(f"✅ Enhancement successful on attempt {attempt + 1}")
                return enhanced
            else:
                print(f"⚠️ Enhancement attempt {attempt + 1} returned original prompt")
                if attempt < self.max_enhancement_attempts - 1:
                    time.sleep(5)
                
        except Exception as e:
            print(f"❌ Enhancement attempt {attempt + 1} failed: {e}")
            if attempt < self.max_enhancement_attempts - 1:
                time.sleep(5)
    
    print("⚠️ All enhancement attempts failed, using original prompt")
    return original_prompt
```

#### **📋 Enhancement Types**
| Enhancement Type | Model Used | Features | Use Case |
|------------------|------------|----------|----------|
| `base` | Qwen 2.5-7B Base | NSFW-optimized, single-shot | Standard enhancement |
| `chat` | Qwen 2.5-7B Instruct | Conversational, context-aware | Interactive sessions |
| `instruct_chat` | Qwen 2.5-7B Instruct | Conversational, context-aware | Interactive sessions (alias) |

#### **💬 Conversation Context Support**
The chat-based enhancement supports conversation memory for coherent multi-turn interactions:

```json
{
  "metadata": {
    "enhancement_type": "chat",
    "session_id": "user_session_123",
    "conversation_context": "Previous prompt: 'beautiful woman in garden' → Enhanced: 'stunning woman with flowing hair in sunlit garden'"
  }
}
```

#### **📋 Chat Enhancement Metadata Parameters**

**Required for Chat Enhancement:**
- `enhancement_type`: `"chat"` or `"instruct_chat"` (enables conversational enhancement)
- `session_id`: Unique identifier for the user session (enables conversation memory)

**Optional for Enhanced Context:**
- `conversation_context`: Previous conversation history or context (improves coherence)

**Example Usage:**
```json
{
  "metadata": {
    "enhancement_type": "chat",
    "session_id": "user_123_session_456",
    "conversation_context": "User requested: 'woman in red dress' → Enhanced to: 'elegant woman in flowing red silk dress'"
  }
}
```

**Fallback Behavior:**
- If `enhancement_type` is not specified: Uses `"base"` enhancement
- If `session_id` is missing: Chat enhancement falls back to base enhancement
- If `conversation_context` is missing: Uses only current prompt for enhancement

#### **🔄 Fallback Strategy**
The enhancement system includes robust fallback mechanisms:
1. **Primary**: Attempt chat-based enhancement with Instruct model
2. **Secondary**: Fallback to base model enhancement if Instruct fails
3. **Final**: Use original prompt if all enhancement attempts fail
4. **Timeout Protection**: 120-second timeout for model loading and generation

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
- **🌐 Automatic URL Registration**: Automatically detects RunPod URL, validates worker health, and registers with Supabase
- **🔄 Periodic Health Monitoring**: Continuous health checks and URL re-registration every 5 minutes

### **Orchestrator API Endpoints**

#### **Worker Status Monitoring**
```http
GET /worker/status
```

**Response:**
```json
{
  "status": "running",
  "workers": {
    "sdxl": {
      "status": "active",
      "uptime": 3600,
      "jobs_processed": 25,
      "last_job_time": "2025-07-23T18:30:00Z",
      "memory_usage": "12.5GB",
      "error_count": 0
    },
    "wan": {
      "status": "active", 
      "uptime": 3600,
      "jobs_processed": 15,
      "last_job_time": "2025-07-23T18:25:00Z",
      "memory_usage": "28.3GB",
      "error_count": 0
    }
  },
  "system": {
    "gpu_memory_total": "48GB",
    "gpu_memory_used": "40.8GB",
    "gpu_utilization": "85%",
    "active_jobs": 3
  }
}
```

#### **Worker Restart**
```http
POST /worker/restart/{worker_id}
```

**Parameters:**
- `worker_id`: `sdxl` | `wan`

**Response:**
```json
{
  "status": "restarting",
  "worker_id": "sdxl",
  "restart_time": "2025-07-23T18:35:00Z",
  "estimated_startup_time": "30s"
}
```

#### **Resource Monitoring**
```http
GET /worker/resources
```

**Response:**
```json
{
  "gpu": {
    "name": "RTX 6000 ADA",
    "memory_total": "48GB",
    "memory_used": "40.8GB",
    "memory_free": "7.2GB",
    "utilization": "85%",
    "temperature": "72°C"
  },
  "workers": {
    "sdxl": {
      "memory_allocated": "12.5GB",
      "model_loaded": true,
      "queue_size": 2
    },
    "wan": {
      "memory_allocated": "28.3GB", 
      "model_loaded": true,
      "queue_size": 1
    }
  },
  "queues": {
    "sdxl_queue": 2,
    "wan_queue": 1
  }
}
```

#### **🌐 Automatic URL Registration**

The Dual Orchestrator includes automatic RunPod URL detection, validation, and Supabase registration functionality that runs at startup and continues monitoring throughout operation.

##### **Startup Registration Process**
1. **URL Detection**: Uses `RUNPOD_POD_ID` environment variable to construct RunPod proxy URL
2. **Health Validation**: Tests the worker's `/health` endpoint to ensure it's operational
3. **Supabase Registration**: Calls the `update-worker-url` edge function to register the URL
4. **Periodic Monitoring**: Starts background thread for continuous health checks and re-registration

##### **URL Detection**
```python
# Automatic RunPod URL construction
pod_id = os.environ.get('RUNPOD_POD_ID')
runpod_url = f"https://{pod_id}-7860.proxy.runpod.net/"
```

##### **Health Validation**
```http
GET {runpod_url}/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "qwen_loaded": true,
  "timestamp": 1732320000.0,
  "worker_ready": true
}
```

##### **Supabase Registration**
```http
POST {SUPABASE_URL}/functions/v1/update-worker-url
```

**Request Payload:**
```json
{
  "worker_url": "https://ghy077o4okmjzi-7860.proxy.runpod.net/",
  "timestamp": "2025-07-23T18:30:00Z",
  "status": "active"
}
```

**Headers:**
```http
Authorization: Bearer {SUPABASE_SERVICE_KEY}
Content-Type: application/json
```

##### **Periodic Health Monitoring**
- **Frequency**: Every 5 minutes
- **Actions**:
  - Detect current RunPod URL
  - Validate worker health endpoint
  - Re-register URL with Supabase if healthy
  - Log health status and registration results

##### **Environment Variables Required**
```bash
# RunPod environment
RUNPOD_POD_ID=ghy077o4okmjzi

# Supabase credentials
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_key_here
```

##### **Error Handling**
- **URL Detection Failure**: Logs warning and continues without registration
- **Health Validation Failure**: Skips registration until worker is healthy
- **Registration Failure**: Logs error and retries in next monitoring cycle
- **Missing Credentials**: Logs error and disables registration functionality

#### **Environment Validation**
```http
GET /worker/validate
```

**Response:**
```json
{
  "status": "valid",
  "checks": {
    "environment_variables": true,
    "model_paths": true,
    "gpu_availability": true,
    "python_dependencies": true,
    "worker_scripts": true
  },
  "details": {
    "pytorch_version": "2.4.1+cu124",
    "cuda_version": "12.4",
    "gpu_memory": "48GB",
    "missing_dependencies": []
  }
}
```

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

## **🌐 Frontend Enhancement API**

### **Overview**
The WAN worker includes a Flask-based API for real-time prompt enhancement using the Qwen 7B model. This allows frontend applications to enhance prompts before submitting them to the main worker queue.

### **Enhancement Endpoint**
```http
POST https://ghy077o4okmjzi-7860.proxy.runpod.net/enhance
```

### **Request Format**
```json
{
  "prompt": "beautiful woman in garden",
  "model": "qwen_base",
  "enhance_type": "natural_language"
}
```

### **Headers**
```http
Authorization: Bearer your_api_key_here
Content-Type: application/json
```

### **Response Format**
```json
{
  "success": true,
  "enhanced_prompt": "stunning woman with flowing hair in sunlit garden, cinematic lighting, 4K quality",
  "original_prompt": "beautiful woman in garden",
  "enhancement_source": "qwen_base",
  "processing_time": 2.5,
  "model": "qwen_base"
}
```

### **Error Response**
```json
{
  "success": false,
  "error": "Enhancement failed: Model not available",
  "enhanced_prompt": "beautiful woman in garden"
}
```

### **Health Check Endpoint**
```http
GET https://ghy077o4okmjzi-7860.proxy.runpod.net/health
```

### **Health Response**
```json
{
  "status": "healthy",
  "qwen_loaded": true,
  "timestamp": 1732320000.0,
  "worker_ready": true
}
```

### **Environment Variables**
```bash
# API Key for frontend enhancement (optional, defaults to 'default_key_123')
WAN_WORKER_API_KEY=your_secure_api_key_here
```

### **Usage Example**
```python
import requests

# Enhance a prompt
response = requests.post(
    'https://ghy077o4okmjzi-7860.proxy.runpod.net/enhance',
    headers={
        'Authorization': 'Bearer your_api_key_here',
        'Content-Type': 'application/json'
    },
    json={
        'prompt': 'beautiful woman in garden',
        'model': 'qwen_base'
    }
)

if response.status_code == 200:
    result = response.json()
    enhanced_prompt = result['enhanced_prompt']
    print(f"Enhanced: {enhanced_prompt}")
else:
    print(f"Error: {response.json()['error']}")
```

---

## **🔧 Environment Requirements**

### **System Requirements**
- **GPU**: NVIDIA RTX 6000 ADA (48GB VRAM) or equivalent
- **RAM**: 64GB+ system memory
- **Storage**: 500GB+ SSD for models and temporary files
- **OS**: Linux (Ubuntu 20.04+ recommended)

### **Required Environment Variables**
```bash
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key

# Redis Configuration (Upstash)
UPSTASH_REDIS_REST_URL=https://your-redis.upstash.io
UPSTASH_REDIS_REST_TOKEN=your-redis-token

# Optional: Custom paths
PYTHONPATH=/workspace/python_deps/lib/python3.11/site-packages
HF_HOME=/workspace/models/huggingface_cache
```

### **Model Paths**
```bash
# WAN 1.3B Model
/workspace/models/wan2.1-t2v-1.3b

# SDXL Model
/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors

# Qwen Models
/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct

# WAN Code
/workspace/Wan2.1
```

### **Python Dependencies**
```bash
# Core Dependencies
torch==2.4.1+cu124
transformers>=4.36.0
diffusers>=0.24.0
accelerate>=0.25.0

# WAN Dependencies
wan>=2.1.0
xfuser>=0.1.0

# SDXL Dependencies
compel==0.1.8
pyparsing==3.0.9

# Utility Dependencies
Pillow>=10.0.0
requests>=2.31.0
redis>=5.0.0
```

### **Environment Validation**
The system automatically validates the environment before starting workers:

```python
# Validation checks performed
- PyTorch version (2.4.1+cu124 required)
- CUDA version (12.4 required)
- GPU availability and memory
- Model file existence
- Python package availability
- Environment variable configuration
- Worker script accessibility
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
        
        # Determine reference frame mode and route to appropriate generation function
        if single_reference_url and not start_reference_url and not end_reference_url:
            # Single reference frame mode (I2V-style)
            task_type = "t2v-1.3B"  # ✅ CORRECT: T2V with --image parameter
            result = generate_video_with_single_reference(prompt, single_reference_url, config.get('frame_num', 83), task_type)
        elif start_reference_url and end_reference_url:
            # Both frames mode (start + end)
            task_type = "t2v-1.3B"  # ✅ CORRECT: T2V with --first_frame + --last_frame
            result = generate_video_with_both_frames(prompt, start_reference_url, end_reference_url, config.get('frame_num', 83), task_type)
        elif start_reference_url and not end_reference_url:
            # Start frame only mode
            task_type = "t2v-1.3B"  # ✅ CORRECT: T2V with --first_frame parameter
            result = generate_video_with_start_frame(prompt, start_reference_url, config.get('frame_num', 83), task_type)
        elif end_reference_url and not start_reference_url:
            # End frame only mode
            task_type = "t2v-1.3B"  # ✅ CORRECT: T2V with --last_frame parameter
            result = generate_video_with_end_frame(prompt, end_reference_url, config.get('frame_num', 83), task_type)
        else:
            # Standard generation (no reference frames)
            task_type = "t2v-1.3B"  # ✅ CORRECT: Text-to-Video standard
            result = generate_standard_video(prompt, config.get('frame_num', 83), task_type)
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
        "compel_weights": compel_weights if compel_enabled else None,
        "compel_success": compel_success if compel_enabled else False,
        "enhancement_strategy": "compel" if compel_success else "fallback" if compel_enabled else "none",
        "original_prompt": original_prompt,
        "final_prompt_type": "conditioning_tensor" if isinstance(final_prompt, torch.Tensor) else "string"
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

## **🚀 Recent Updates (July 23, 2025)**

### **Major Enhancements**
1. **🎯 Compel Integration**: SDXL worker now supports Compel prompt enhancement with weighted attention
2. **🤖 Qwen 7B Enhancement**: WAN worker includes AI-powered prompt enhancement with multiple strategies
3. **💬 Chat-Based Enhancement**: WAN worker now supports conversational prompt enhancement with Qwen 2.5-7B Instruct model
4. **🎭 Dual Orchestrator**: Centralized management of both workers with monitoring and restart capabilities
5. **Standardized Callback Parameters**: Consistent `job_id`, `assets` array across all workers
6. **Enhanced Negative Prompts**: Intelligent generation for SDXL with multi-party scene detection
7. **Seed Support**: User-controlled seeds for reproducible generation
8. **Flexible SDXL Quantities**: User-selectable 1, 3, or 6 images per batch
9. **Reference Image Support**: Optional image-to-image with type and strength control
10. **Video Reference Frame Support**: I2V-style generation with start reference frame for WAN 1.3B model
11. **Comprehensive Error Handling**: Enhanced debugging and error tracking
12. **Metadata Consistency**: Improved data flow and storage
13. **Path Consistency Fix**: Fixed video path handling for WAN workers
14. **🔄 Robust Fallback System**: Automatic fallback from chat enhancement to base enhancement to original prompt

### **Latest Documentation Updates (July 23, 2025)**
1. **🔧 Environment Requirements**: Added comprehensive system requirements, environment variables, and model paths
2. **🎭 Dual Orchestrator API**: Added complete API endpoints for worker monitoring, restart, and resource management
3. **💬 Chat Enhancement Documentation**: Enhanced documentation for chat-based prompt enhancement with session management
4. **🎬 WAN 1.3B Task Clarification**: Updated to reflect current implementation using only `t2v-1.3B` task
5. **📋 Reference Frame Modes**: Clarified all 5 reference frame modes and their parameter usage
6. **🔍 Environment Validation**: Added detailed validation requirements and error handling
7. **📊 Resource Monitoring**: Added comprehensive resource monitoring and status tracking
8. **🛠️ Python Dependencies**: Added complete dependency list with version requirements
9. **🌐 Frontend Enhancement API**: Added Flask-based API for real-time prompt enhancement with Qwen 7B model
10. **🌐 Automatic URL Registration**: Added automatic RunPod URL detection, validation, and Supabase registration functionality

### **Performance Improvements**
- Optimized batch processing for multi-image SDXL jobs
- Enhanced error recovery and retry mechanisms
- Improved Redis queue management
- Better resource utilization tracking
- AI-powered prompt enhancement for higher quality output
- Conversational prompt enhancement with context memory for improved coherence
- Intelligent model loading/unloading for optimal memory management

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
- Chat enhancement is optional and falls back to base enhancement if not specified
- Original prompt enhancement behavior preserved when `enhancement_type` is not provided

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
✅ Compel weights applied with SDXL library integration (list API)
📝 Final combined prompt: (beautiful woman:1.1), (masterpiece:1.3), (best quality:1.2)
🔧 Generated prompt_embeds: torch.Size([1, 77, 2048])
🔧 Generated pooled_prompt_embeds: torch.Size([1, 1280])
🔧 Generated negative_prompt_embeds: torch.Size([1, 77, 2048])
🔧 Generated negative_pooled_prompt_embeds: torch.Size([1, 1280])
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

#### **Deprecated/Incorrect Approaches**
- String concatenation for SDXL+Compel
- More than 6 enhancement weights
- Duplicate weights
- No negative conditioning for SDXL
- Logging or metadata that dumps tensor values

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

- [x] Compel library imports successfully
- [x] `process_compel_weights` uses `requires_pooled=[False, True]` for SDXL
- [x] Generation function handles both prompt_embeds and pooled_prompt_embeds for SDXL
- [x] Error handling includes fallback to original prompt
- [x] Metadata includes Compel processing status and tensor types
- [x] No more 'bool object is not iterable' or argument errors in logs
- [x] Compel weights are properly applied without token limit violations
- [x] Both conditioning tensors are generated and used correctly
- [x] Legacy single tensor fallback works for backward compatibility

### **Rollback Plan**

If issues occur, the worker can be rolled back by:
1. Reverting to the string concatenation approach
2. Disabling Compel processing temporarily
3. Using the original prompt without enhancement

The fallback mechanisms ensure the worker continues to function even if Compel processing fails. 