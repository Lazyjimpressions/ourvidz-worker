# Configuration Approach & Pure Inference Architecture

**Date:** August 16, 2025  
**Purpose:** Define the configuration ethos and edge function requirements for pure inference workers

---

## 🎯 **Configuration Ethos**

### **Pure Inference Philosophy**
**"Workers are dumb execution engines. All intelligence lives in the edge function."**

### **Core Principles**
1. **No Business Logic in Workers:** Workers execute exactly what's provided
2. **Complete Parameter Passing:** Edge function provides all necessary configuration
3. **No Hardcoded Assumptions:** Workers accept any valid input
4. **Centralized Intelligence:** Edge function handles all decision-making
5. **Validation at Edge:** Parameter validation happens before reaching workers

---

## 🏗️ **Architecture Overview**

### **Configuration Hierarchy**
```
Frontend → Edge Function → Worker
    ↓           ↓           ↓
User Input → Business Logic → Pure Execution
```

### **Edge Function Responsibilities**
- **User Validation:** Permissions, quotas, restrictions
- **Parameter Conversion:** Frontend presets → worker parameters
- **Content Restrictions:** User-type based filtering
- **Prompt Enhancement:** AI-powered prompt improvement
- **Job Routing:** Worker selection and priority assignment
- **Error Handling:** Comprehensive validation and fallbacks

### **Worker Responsibilities**
- **Parameter Validation:** Format and range checking only
- **Pure Execution:** Generate exactly what's requested
- **Result Return:** Standardized callback format
- **No Business Logic:** No content decisions or user validation

---

## 🔧 **Worker Configuration Files**

### **Configuration Structure**
```
Configuration/
├── worker_configs.py       # Worker configuration templates
├── validation_schemas.py   # Request validation schemas
└── CONFIGURATION_APPROACH.md # This file
```

### **Configuration Classes**

#### **WorkerConfig**
- **Purpose:** Worker-specific configuration templates
- **Usage:** Edge function validates against these templates
- **Examples:** Model paths, supported parameters, memory requirements

#### **ModelConfig**
- **Purpose:** Model-specific optimization settings
- **Usage:** Worker initialization and optimization
- **Examples:** Compilation settings, attention slicing, dtype

#### **ValidationConfig**
- **Purpose:** Request parameter validation rules
- **Usage:** Worker-side parameter validation
- **Examples:** Required fields, value ranges, format validation

---

## 📋 **Edge Function Requirements by Worker**

### **🎨 SDXL Worker Requirements**

#### **Required Parameters**
```json
{
  "id": "string",                    // Job ID (required)
  "type": "sdxl_image_fast|sdxl_image_high", // Job type (required)
  "prompt": "string",                // Complete prompt (required)
  "user_id": "string",               // User ID (required)
  "config": {
    "num_images": 1|3|6,            // Batch size (required)
    "steps": 10-50,                 // Generation steps (optional, default: 25)
    "guidance_scale": 1.0-20.0,     // CFG scale (optional, default: 7.5)
    "resolution": "1024x1024",      // Image resolution (optional, default: 1024x1024)
    "seed": 0-2147483647,           // Random seed (optional)
    "negative_prompt": "string"     // Negative prompt (optional)
  },
  "metadata": {
    "reference_image_url": "string", // Reference image URL (optional)
    "reference_strength": 0.0-1.0,  // Reference strength (optional, default: 0.5)
    "reference_type": "style|composition|character" // Reference type (optional)
  },
  "compel_enabled": boolean,        // Compel enhancement (optional, default: false)
  "compel_weights": "string"        // Compel weights (optional)
}
```

#### **Edge Function Processing**
1. **Validate User Permissions:** Check if user can request NSFW content
2. **Enhance Prompt:** Call Chat Worker for prompt enhancement if requested
3. **Convert Presets:** Transform frontend presets to worker parameters
4. **Validate Parameters:** Check against SDXL validation rules
5. **Route to Worker:** Send complete job data to SDXL worker

#### **Callback Format**
```json
{
  "job_id": "string",
  "worker_id": "sdxl_worker_001",
  "status": "completed|failed|processing",
  "assets": [
    {
      "type": "image",
      "url": "https://cdn.example.com/image.png",
      "metadata": {
        "width": 1024,
        "height": 1024,
        "format": "png",
        "batch_size": 3,
        "steps": 25,
        "guidance_scale": 7.5,
        "seed": 12345
      }
    }
  ],
  "metadata": {
    "enhancement_source": "qwen_instruct|none",
    "compel_enhancement": true|false,
    "reference_mode": "none|style|composition|character",
    "processing_time": 15.2,
    "vram_used": 8192
  }
}
```

---

### **💬 Chat Worker Requirements**

#### **Required Parameters**
```json
{
  "messages": [                     // Message array (required)
    {
      "role": "system|user|assistant", // Message role (required)
      "content": "string"           // Message content (required)
    }
  ],
  "max_tokens": 100-2048,          // Max tokens (optional, default: 512)
  "temperature": 0.0-2.0,          // Temperature (optional, default: 0.7)
  "top_p": 0.0-1.0,               // Top-p (optional, default: 0.9)
  "model": "qwen_instruct|qwen_base", // Model selection (optional, default: qwen_instruct)
  "sfw_mode": boolean              // SFW mode (optional, default: false)
}
```

#### **Edge Function Processing**
1. **Build System Prompt:** Create appropriate system prompt based on context
2. **Validate User Permissions:** Check chat permissions and restrictions
3. **Select Model:** Choose Base vs Instruct based on enhancement needs
4. **Format Messages:** Ensure proper message array format
5. **Route to Worker:** Send complete message array to Chat worker

#### **Callback Format**
```json
{
  "job_id": "string",
  "worker_id": "chat_worker_001",
  "status": "completed|failed|processing",
  "assets": [
    {
      "type": "text",
      "content": "Generated response text",
      "metadata": {
        "tokens_generated": 150,
        "model_used": "qwen_instruct|qwen_base",
        "generation_time": 2.5
      }
    }
  ],
  "metadata": {
    "enhancement_type": "base|instruct|none",
    "sfw_mode": true|false,
    "system_prompt_used": "Custom system prompt",
    "processing_time": 2.5,
    "vram_used": 15360
  }
}
```

---

### **🎬 WAN Worker Requirements**

#### **Required Parameters**
```json
{
  "id": "string",                    // Job ID (required)
  "type": "image_fast|image_high|video_fast|video_high|image7b_fast_enhanced|image7b_high_enhanced|video7b_fast_enhanced|video7b_high_enhanced", // Job type (required)
  "prompt": "string",                // Complete prompt (required)
  "user_id": "string",               // User ID (required)
  "config": {
    "width": 480,                   // Width (optional, default: 480)
    "height": 832,                  // Height (optional, default: 832)
    "frames": 1-83,                 // Video frames (optional, default: 83)
    "fps": 8-24,                   // FPS (optional, default: 24)
    "reference_mode": "none|single|start|end|both", // Reference mode (optional, default: none)
    "image": "string",              // Single reference (optional)
    "first_frame": "string",        // Start reference (optional)
    "last_frame": "string"          // End reference (optional)
  },
  "metadata": {
    "reference_image_url": "string", // Fallback reference (optional)
    "start_reference_url": "string", // Fallback start (optional)
    "end_reference_url": "string",   // Fallback end (optional)
    "reference_strength": 0.0-1.0   // Reference strength (optional, default: 0.5)
  }
}
```

#### **Edge Function Processing**
1. **Validate User Permissions:** Check video generation permissions
2. **Enhance Prompt:** Call Chat Worker for prompt enhancement if needed
3. **Process References:** Download and validate reference images
4. **Convert Presets:** Transform frontend presets to worker parameters
5. **Route to Worker:** Send complete job data to WAN worker

#### **Callback Format**
```json
{
  "job_id": "string",
  "worker_id": "wan_worker_001",
  "status": "completed|failed|processing",
  "assets": [
    {
      "type": "video|image",
      "url": "https://cdn.example.com/video.mp4",
      "metadata": {
        "width": 480,
        "height": 832,
        "frames": 83,
        "fps": 24,
        "duration": 3.46,
        "format": "mp4"
      }
    }
  ],
  "metadata": {
    "enhancement_source": "qwen_base|none",
    "reference_mode": "none|single|start|end|both",
    "processing_time": 180.5,
    "vram_used": 30720,
    "auto_enhancement": true|false
  }
}
```

---

## 🏷️ **Naming Conventions**

### **Job ID Format**
```
{worker_type}_{job_type}_{timestamp}_{random_suffix}
Examples:
- sdxl_image_high_1734567890_abc123
- wan_video_fast_1734567890_def456
- chat_enhance_1734567890_ghi789
```

### **Asset URL Format**
```
https://cdn.example.com/{bucket}/{user_id}/{job_id}/{asset_type}_{index}.{extension}
Examples:
- https://cdn.example.com/user-library/user123/sdxl_image_high_1734567890_abc123/image_0.png
- https://cdn.example.com/user-library/user123/wan_video_fast_1734567890_def456/video_0.mp4
```

### **Metadata Keys**
- **enhancement_source:** `qwen_instruct|qwen_base|none`
- **reference_mode:** `none|single|start|end|both`
- **processing_time:** Float in seconds
- **vram_used:** Integer in MB
- **batch_size:** Integer (1, 3, or 6 for SDXL)
- **model_used:** `qwen_instruct|qwen_base|lustify_sdxl|wan_2.1`

---

## 🔍 **Frontend Validation Requirements**

### **Parameter Validation**
Frontend should validate all parameters before sending to edge function:

#### **SDXL Validation**
- **num_images:** Must be 1, 3, or 6
- **steps:** Must be 10-50
- **guidance_scale:** Must be 1.0-20.0
- **resolution:** Must match pattern `^\d+x\d+$`
- **prompt:** Must be 1-1000 characters

#### **WAN Validation**
- **frames:** Must be 1-83
- **fps:** Must be 8-24
- **width/height:** Must be valid dimensions
- **reference_mode:** Must be valid mode
- **prompt:** Must be 1-1000 characters

#### **Chat Validation**
- **messages:** Must be array with valid roles
- **max_tokens:** Must be 100-2048
- **temperature:** Must be 0.0-2.0
- **top_p:** Must be 0.0-1.0

### **Callback Processing**
Frontend should process callbacks based on worker type:

#### **SDXL Callbacks**
- **Asset Type:** `image`
- **Batch Processing:** Check `batch_size` in metadata
- **Reference Images:** Check `reference_mode` in metadata

#### **WAN Callbacks**
- **Asset Type:** `video` or `image`
- **Duration:** Check `duration` in metadata for videos
- **Reference Mode:** Check `reference_mode` in metadata

#### **Chat Callbacks**
- **Asset Type:** `text`
- **Model Used:** Check `model_used` in metadata
- **Enhancement:** Check `enhancement_type` in metadata

---

## 🎯 **Configuration Best Practices**

### **Edge Function Guidelines**
1. **Always Validate:** Check all parameters against worker schemas
2. **Provide Defaults:** Set sensible defaults for optional parameters
3. **Enhance Prompts:** Use Chat Worker for prompt enhancement when appropriate
4. **Handle Errors:** Provide meaningful error messages for validation failures
5. **Log Operations:** Track all parameter conversions and routing decisions

### **Worker Guidelines**
1. **Accept Everything:** Workers should accept any valid input
2. **Validate Format:** Check parameter types and ranges
3. **Provide Feedback:** Return detailed error messages for invalid parameters
4. **Standardize Callbacks:** Use consistent callback format across all workers
5. **Log Execution:** Track processing time and resource usage

### **Frontend Guidelines**
1. **Validate Early:** Check parameters before sending to edge function
2. **Handle Callbacks:** Process different asset types appropriately
3. **Show Progress:** Display processing time and queue position
4. **Error Recovery:** Implement retry logic for transient failures
5. **User Feedback:** Provide clear status updates and error messages

---

## 📚 **Configuration Files Summary**

### **worker_configs.py**
- Worker configuration templates
- Model-specific settings
- Memory allocation specifications
- Performance optimization parameters

### **validation_schemas.py**
- Request validation rules
- Parameter type checking
- Value range validation
- Format pattern matching

### **CONFIGURATION_APPROACH.md**
- This document
- Architecture philosophy
- Edge function requirements
- Naming conventions
- Best practices

---

**🎯 This configuration approach ensures pure inference workers with centralized edge function intelligence, providing maximum flexibility and maintainability.**
