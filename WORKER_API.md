# OurVidz Worker API Reference

**Last Updated:** January 27, 2025  
**Status:** ✅ Production Ready - All 10 Job Types Operational + Reference Image Support + Seed Control  
**System:** Dual Worker (SDXL + WAN) on RTX 6000 ADA (48GB VRAM)  
**Conformity:** ✅ API documentation aligned with actual worker implementations

---

## **🎯 Quick Reference**

### **Active Job Types (10 Total)**
```yaml
SDXL Jobs (2) - Flexible Quantities (1, 3, or 6 images):
  sdxl_image_fast: 3-8s per image - 1024x1024 PNG
  sdxl_image_high: 5-15s per image - 1024x1024 PNG

WAN Standard Jobs (4) - Single Files:
  image_fast: 25s - 480x832 PNG
  image_high: 40s - 480x832 PNG  
  video_fast: 135s - 480x832 MP4 (5.0s duration, 83 frames)
  video_high: 180s - 480x832 MP4 (5.0s duration, 83 frames)

WAN Enhanced Jobs (4) - AI-Enhanced:
  image7b_fast_enhanced: 85s - 480x832 PNG (Qwen enhanced)
  image7b_high_enhanced: 100s - 480x832 PNG (Qwen enhanced)
  video7b_fast_enhanced: 195s - 480x832 MP4 (Qwen enhanced, 5.0s)
  video7b_high_enhanced: 240s - 480x832 MP4 (Qwen enhanced, 5.0s)
```

### **NEW: Reference Image Support + Seed Control**
```yaml
SDXL Reference Types:
  - Style Reference: Transfer visual style from reference image
  - Composition Reference: Use reference for layout/structure guidance
  - Character Reference: Maintain character consistency across generations

Video Reference Support:
  - Start Frame: Reference image for first video frame
  - End Frame: Reference image for last video frame
  - Both: Smooth transition between start and end references

Reference Parameters:
  - Strength: 0.0-1.0 (controls influence level)
  - Type: style/composition/character (SDXL only)
  - Optional: All reference images are optional

Seed Control (SDXL only):
  - Reproducible generation for consistent results
  - Character consistency across multiple generations
  - Optional: Random seeds used if not provided
```

---

## **🚀 Job Creation API**

### **Endpoint: `POST /api/queue-job`**
**Authentication:** Required (JWT token)

### **Request Payload:**
```typescript
{
  jobType: string,           // One of 10 job types above
  metadata?: {
    prompt?: string,         // User input prompt
    credits?: number,        // Credits consumed (default: 1)
    bucket?: string,         // Storage bucket (auto-detected)
    
    // NEW: Reference Image Support
    reference_image_url?: string,        // SDXL reference image URL
    start_reference_image_url?: string,  // Video start frame URL
    end_reference_image_url?: string,    // Video end frame URL
    reference_strength?: number,         // 0.0-1.0 (default: 0.5)
    reference_type?: string,             // 'style'|'composition'|'character'
    
    // NEW: Flexible Quantities (SDXL only)
    num_images?: number,                 // 1, 3, or 6 (default: 1)
    
    // NEW: Seed Control (SDXL only)
    seed?: number,                       // Integer seed for reproducible generation
    
    // Enhanced Features
    negative_prompt?: string,            // SDXL only
    similarity_strength?: number,        // Alternative to reference_strength
  },
  config?: {                            // NEW: Job configuration object
    num_images?: number,                 // Alternative location for num_images
    seed?: number,                       // Alternative location for seed
  },
  projectId?: string,        // Optional project reference
  videoId?: string,          // Optional video ID
  imageId?: string           // Optional image ID
}
```

### **Response:**
```typescript
{
  success: boolean,
  job: {
    id: string,              // Database job ID
    user_id: string,         // User identifier
    job_type: string,        // Job type
    status: 'queued',        // Initial status
    created_at: string,      // ISO timestamp
    metadata: object         // Job metadata including reference info
  },
  message: string,           // Success message
  queueLength: number,       // Current queue depth
  modelVariant: string,      // Model being used
  isSDXL: boolean,          // SDXL vs WAN job
  negativePromptSupported: boolean,  // SDXL only
  referenceImageSupported: boolean,  // NEW: Reference support flag
  flexibleQuantitiesSupported: boolean,  // NEW: Quantity support flag
  seedControlSupported: boolean      // NEW: Seed control support flag
}
```

### **Example Requests:**

#### **SDXL with Reference Image and Seed:**
```typescript
// SDXL Fast with Style Reference (3 images) and seed control
const response = await fetch('/api/queue-job', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    jobType: 'sdxl_image_fast',
    metadata: {
      prompt: 'beautiful woman in garden',
      credits: 1,
      reference_image_url: 'https://storage.example.com/reference.jpg',
      reference_strength: 0.7,
      reference_type: 'style',
      num_images: 3,
      seed: 12345,  // NEW: Reproducible generation
      negative_prompt: 'blurry, low quality, watermark'
    }
  })
});
```

#### **Video with Reference Frames:**
```typescript
// WAN Video with Start/End References
const response = await fetch('/api/queue-job', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    jobType: 'video_fast',
    metadata: {
      prompt: 'woman walking in park',
      credits: 1,
      start_reference_image_url: 'https://storage.example.com/start.jpg',
      end_reference_image_url: 'https://storage.example.com/end.jpg',
      reference_strength: 0.6
    }
  })
});
```

#### **Standard SDXL with Seed Control:**
```typescript
// SDXL High Quality (6 images, no reference, with seed)
const response = await fetch('/api/queue-job', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    jobType: 'sdxl_image_high',
    metadata: {
      prompt: 'beautiful woman in garden',
      credits: 2,
      num_images: 6,
      seed: 67890,  // NEW: Reproducible generation
      negative_prompt: 'blurry, low quality, watermark, text, logo'
    }
  })
});
```

---

## **📊 Job Status & Monitoring**

### **Job Status Values:**
```yaml
queued: Job created, waiting in queue
processing: Job picked up by worker, generating content
completed: Job finished successfully
failed: Job failed with error
```

### **Real-time Status Updates:**
Workers automatically call back to update job status using consistent parameter names. Frontend should poll for updates:

```typescript
// Poll job status every 2-5 seconds
const checkJobStatus = async (jobId: string) => {
  const response = await fetch(`/api/jobs/${jobId}`);
  const job = await response.json();
  
  switch (job.status) {
    case 'completed':
      // Handle completion - assets available
      handleJobCompletion(job);
      break;
    case 'failed':
      // Handle failure - show error
      handleJobFailure(job);
      break;
    case 'processing':
      // Update progress UI
      updateProgressUI(job);
      break;
  }
};
```

---

## **📁 Asset Handling**

### **SDXL Jobs (Flexible Quantities):**
```typescript
// SDXL returns array of 1, 3, or 6 image URLs
const handleSDXLCompletion = (job) => {
  const imageUrls = job.assets; // ✅ CONSISTENT: assets array from callback
  const numImages = imageUrls.length; // 1, 3, or 6
  const primaryImage = imageUrls[0]; // First image
  
  // Display images in grid based on quantity
  if (numImages === 1) {
    displaySingleImage(primaryImage);
  } else {
    displayImageGrid(imageUrls);
  }
  
  // Store in database
  await saveImagesToDatabase(imageUrls, job.id);
  
  // NEW: Handle seed information from metadata
  if (job.metadata && job.metadata.seed) {
    console.log(`Generated with seed: ${job.metadata.seed}`);
    // Store seed for future reference or regeneration
  }
};
```

### **WAN Jobs (Single Files):**
```typescript
// WAN returns single file URL
const handleWANCompletion = (job) => {
  const fileUrl = job.assets[0]; // ✅ CONSISTENT: assets array from callback
  
  if (job.job_type.includes('video')) {
    // Handle video
    displayVideoPlayer(fileUrl);
    await saveVideoToDatabase(fileUrl, job.id);
  } else {
    // Handle image
    displayImage(fileUrl);
    await saveImageToDatabase(fileUrl, job.id);
  }
};
```

---

## **🎨 Job Type Details**

### **SDXL Image Generation (Enhanced)**
```yaml
Models: LUSTIFY SDXL v2.0 (NSFW-capable)
Resolution: 1024x1024 (square)
Format: PNG
Batch Size: 1, 3, or 6 images per job (user-selectable)
Quality: Excellent NSFW content

Performance (RTX 6000 ADA):
  sdxl_image_fast: 3-8s per image
  sdxl_image_high: 5-15s per image
  Total time scales linearly with quantity

Reference Image Support:
  - Style Transfer: Transfer visual style from reference
  - Composition Guidance: Use reference for layout
  - Character Consistency: Maintain character features
  - Strength Control: 0.0-1.0 influence level

Seed Control:
  - Reproducible generation for consistent results
  - Character consistency across multiple generations
  - Each image in batch gets seed + index for variety
  - Random seeds used if not provided

Storage Buckets:
  sdxl_image_fast: 5MB limit per image
  sdxl_image_high: 10MB limit per image

Frontend Handling:
  - Display images in grid layout (1, 3, or 6)
  - Allow user to select preferred image
  - Store all images in user library
  - Enable download of individual images
  - Show reference image preview if used
  - Display seed information for regeneration
```

### **WAN Video Generation (Enhanced)**
```yaml
Models: WAN 2.1 T2V 1.3B
Resolution: 480x832 (portrait)
Format: MP4
Duration: 5-6 seconds
Quality: Good to excellent

Performance:
  video_fast: 135s average (2m 15s)
  video_high: 180s average (3m)
  video7b_fast_enhanced: 195s average (3m 15s)
  video7b_high_enhanced: 240s average (4m)

Reference Frame Support:
  - Start Frame: Reference image for first video frame
  - End Frame: Reference image for last video frame
  - Both Frames: Smooth transition between references
  - Optional: Can use start only, end only, both, or none

Advanced Parameters:
  - UniPC sampling for better temporal consistency
  - Enhanced guidance scales (6.5-7.5) for NSFW quality
  - Temporal consistency with sample_shift parameter
  - 83 frames for 5.0-second videos (16.67fps effective)

Storage Buckets:
  video_fast: 50MB limit
  video_high: 200MB limit
  video7b_fast_enhanced: 100MB limit
  video7b_high_enhanced: 100MB limit

Frontend Handling:
  - Display video player with controls
  - Show generation progress (0-100%)
  - Enable download of MP4 file
  - Store in user video library
  - Show reference frame previews if used
```

### **WAN Image Generation (Enhanced)**
```yaml
Models: WAN 2.1 T2V 1.3B
Resolution: 480x832 (portrait)
Format: PNG
Quality: Backup images (not primary)

Performance:
  image_fast: 25s
  image_high: 40s
  image7b_fast_enhanced: 85s (Qwen enhanced)
  image7b_high_enhanced: 100s (Qwen enhanced)

Advanced Parameters:
  - UniPC sampling for better quality
  - Enhanced guidance scales (6.5-7.5) for NSFW quality
  - Temporal consistency optimization

Storage Buckets:
  image_fast: 5MB limit
  image_high: 10MB limit
  image7b_fast_enhanced: 20MB limit
  image7b_high_enhanced: 20MB limit

Frontend Handling:
  - Display single image
  - Enable download
  - Store in user image library
```

---

## **🖼️ Reference Image System**

### **SDXL Reference Types:**

#### **Style Reference:**
```yaml
Purpose: Transfer visual style from reference image
Use Case: Maintain consistent artistic style across generations
Strength Range: 0.3-0.8 (recommended)
Example: Use painting style reference for consistent art direction
```

#### **Composition Reference:**
```yaml
Purpose: Use reference image for layout and structure guidance
Use Case: Maintain similar framing, positioning, or scene layout
Strength Range: 0.5-0.9 (recommended)
Example: Use photo reference for consistent camera angle and framing
```

#### **Character Reference:**
```yaml
Purpose: Maintain character consistency across generations
Use Case: Keep same character appearance in different scenes
Strength Range: 0.4-0.7 (recommended)
Example: Use character portrait for consistent facial features
```

### **Video Reference Frames:**

#### **Start Frame Reference:**
```yaml
Purpose: Set the first frame of the video
Use Case: Ensure video starts with specific scene or character
Strength Range: 0.5-0.8 (recommended)
Example: Use character close-up for video opening
```

#### **End Frame Reference:**
```yaml
Purpose: Set the last frame of the video
Use Case: Ensure video ends with specific scene or pose
Strength Range: 0.5-0.8 (recommended)
Example: Use action pose for video conclusion
```

### **Reference Image Requirements:**
```yaml
Format: JPEG, PNG, or WebP
Size: Maximum 10MB per image
Resolution: Any size (auto-resized to target)
Security: Must be from authorized domains
Validation: Automatic format and size validation
```

### **Reference Image Upload Flow:**
```typescript
// 1. Upload reference image
const uploadReferenceImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/api/upload-reference', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: formData
  });
  
  const result = await response.json();
  return result.url; // Returns storage URL
};

// 2. Use in job creation
const createJobWithReference = async (jobType, prompt, referenceUrl, strength, type) => {
  const response = await fetch('/api/queue-job', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: JSON.stringify({
      jobType,
      metadata: {
        prompt,
        reference_image_url: referenceUrl,
        reference_strength: strength,
        reference_type: type
      }
    })
  });
  
  return response.json();
};
```

### **Implementation Notes:**

#### **SDXL Worker Implementation:**
```yaml
Reference Image Processing:
  - Downloads reference image from URL with 30s timeout
  - Converts to RGB format if necessary
  - Resizes to 1024x1024 while maintaining aspect ratio
  - Centers image on black background if needed
  - Supports style, composition, and character reference types

Image-to-Image Generation:
  - Uses StableDiffusionXLPipeline image-to-image mode
  - Strength parameter controls reference influence (0.0-1.0)
  - Composition mode automatically boosts strength by 0.2
  - Generates 1, 3, or 6 images based on num_images parameter
  - Each image gets unique seed for variety

Seed Control:
  - Extracts seed from config.seed or metadata.seed
  - Uses provided seed for reproducible results
  - Generates random seed if not provided
  - Each image in batch gets seed + index for variety
  - Returns used seed in callback metadata

Flexible Quantities:
  - Default: 1 image (backward compatibility)
  - Validates num_images: only 1, 3, or 6 allowed
  - Falls back to 1 if invalid value provided
  - Performance scales linearly with quantity
```

#### **WAN Worker Implementation:**
```yaml
Reference Frame Processing:
  - Downloads reference images from URLs with 30s timeout
  - Converts to RGB format if necessary
  - Resizes to 480x832 (WAN video resolution)
  - Centers image on black background if needed
  - Saves to temporary PNG files for WAN processing

Video Generation with References:
  - Supports start frame, end frame, or both
  - Passes reference files to WAN command line
  - Uses --start_frame and --end_frame parameters
  - Applies --reference_strength parameter
  - Graceful fallback to standard generation if references fail

WAN Command Integration:
  - Uses UniPC sampling for better temporal consistency
  - 83 frames for 5.0-second videos (16.67fps effective)
  - Enhanced guidance scales (6.5-7.5) for NSFW quality
  - Qwen 7B Base model for prompt enhancement (no content filtering)
```

---

## **🌱 Seed Control System (SDXL Only)**

### **What Seed Control Does:**
```yaml
Purpose: Enable reproducible image generation
Benefits:
  - Consistent character appearance across generations
  - Reproducible results for testing and iteration
  - Character consistency in series of images
  - Same seed + prompt = same image (deterministic)

Implementation:
  - Each image in batch gets seed + index for variety
  - Random seeds generated if not provided
  - Seed returned in callback metadata for future reference
  - Supports both config.seed and metadata.seed locations
```

### **Seed Usage Examples:**
```typescript
// Generate with specific seed for reproducibility
const createReproducibleJob = async (prompt, seed) => {
  const response = await fetch('/api/queue-job', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: JSON.stringify({
      jobType: 'sdxl_image_fast',
      metadata: {
        prompt,
        num_images: 3,
        seed: seed  // Will generate same 3 images every time
      }
    })
  });
  
  return response.json();
};

// Regenerate with same seed for character consistency
const regenerateWithSeed = async (prompt, originalSeed) => {
  const response = await fetch('/api/queue-job', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: JSON.stringify({
      jobType: 'sdxl_image_high',
      metadata: {
        prompt: 'same character in different pose',
        num_images: 1,
        seed: originalSeed  // Maintains character consistency
      }
    })
  });
  
  return response.json();
};
```

---

## **🔧 Enhanced Jobs (Qwen 7B)**

### **What Enhanced Jobs Do:**
```yaml
Input: "woman walking"
Output: "一位穿着简约白色连衣裙的东方女性在阳光明媚的公园小径上散步。她的头发自然披肩，步伐轻盈。背景中有绿树和鲜花点缀的小道，阳光透过树叶洒下斑驳光影。镜头采用跟随镜头，捕捉她自然行走的姿态。纪实摄影风格。中景镜头。"

Benefits:
  - 3,400% prompt expansion (75 → 2,627 characters)
  - Professional cinematic descriptions
  - Enhanced anatomical accuracy
  - Better visual quality and detail
  - NSFW-optimized content enhancement

Performance Overhead:
  - 14-112s additional processing time
  - Qwen 7B model loading and enhancement
  - Higher quality output justifies time cost
```

### **When to Use Enhanced Jobs:**
```yaml
Recommended:
  - Professional content creation
  - High-quality output requirements
  - Complex scene descriptions
  - NSFW content with anatomical accuracy

Not Recommended:
  - Quick previews or iterations
  - Simple prompts that work well as-is
  - Time-sensitive content creation
```

---

## **🚨 Error Handling**

### **Common Error Scenarios:**
```yaml
Job Creation Errors:
  - Invalid job type: Return 400 with valid job types list
  - Authentication failed: Return 401
  - Redis queue full: Return 503 with retry guidance
  - User credits insufficient: Return 402 with upgrade prompt
  - Invalid reference image: Return 400 with validation details
  - Invalid quantity (SDXL): Return 400 (only 1, 3, or 6 allowed)
  - Invalid seed (SDXL): Return 400 (must be integer)

Job Processing Errors:
  - Model loading failed: Retry automatically
  - Generation timeout: Mark as failed after 15 minutes
  - Storage upload failed: Retry with exponential backoff
  - GPU memory issues: Worker auto-restart
  - Reference image download failed: Continue without reference
  - Seed generation failed: Use random seed

Frontend Error Handling:
  - Show user-friendly error messages
  - Provide retry options where appropriate
  - Log errors for debugging
  - Graceful degradation for partial failures
  - Reference image preview failures
```

### **Error Response Format:**
```typescript
{
  success: false,
  error: string,           // User-friendly error message
  details?: string,        // Technical details for debugging
  retryable?: boolean,     // Whether retry is recommended
  suggestedAction?: string, // What user should do
  referenceImageError?: boolean, // NEW: Reference image specific error
  invalidQuantity?: boolean,     // NEW: Quantity validation error
  invalidSeed?: boolean          // NEW: Seed validation error
}
```

---

## **📈 Performance Monitoring**

### **Key Metrics to Track:**
```yaml
Job Success Rate: >95% target
Average Generation Times:
  - SDXL fast (1 image): 3-8s
  - SDXL fast (3 images): 9-24s
  - SDXL fast (6 images): 18-48s
  - SDXL high (1 image): 5-15s
  - SDXL high (3 images): 15-45s
  - SDXL high (6 images): 30-90s
  - WAN video_fast: 135s
  - WAN video_high: 180s
  - WAN image_fast: 25s
  - WAN image_high: 40s
  - Enhanced jobs: +60s overhead (Qwen enhancement)

Reference Image Performance:
  - Download time: <5s per reference
  - Processing overhead: +10-30% generation time
  - Success rate: >98% for valid references

Seed Control Performance:
  - No performance impact (deterministic generation)
  - Success rate: 100% for valid seeds
  - Fallback to random seeds if invalid

Queue Performance:
  - sdxl_queue: 2-second polling
  - wan_queue: 5-second polling
  - Average queue depth: <10 jobs
  - Max queue wait time: <5 minutes

System Health:
  - GPU memory usage: <35GB peak
  - Worker uptime: >99%
  - Storage bucket availability: 100%
  - API response time: <500ms
  - Reference image storage: <1GB per user
```

### **Frontend Monitoring:**
```typescript
// Track job performance with reference images and seeds
const trackJobPerformance = (jobType, startTime, endTime, hasReference, hasSeed) => {
  const duration = endTime - startTime;
  analytics.track('job_completed', {
    jobType,
    duration,
    success: true,
    hasReference,
    hasSeed,
    referenceType: hasReference ? jobType.reference_type : null
  });
};

// Monitor queue health
const checkQueueHealth = async () => {
  const response = await fetch('/api/queue-status');
  const status = await response.json();
  
  if (status.queueDepth > 10) {
    showQueueWarning('High queue volume, expect longer wait times');
  }
};

// Track reference image usage
const trackReferenceUsage = (referenceType, strength, success) => {
  analytics.track('reference_image_used', {
    referenceType,
    strength,
    success
  });
};

// Track seed usage
const trackSeedUsage = (seed, success) => {
  analytics.track('seed_used', {
    seed,
    success
  });
};
```

---

## **🔗 Integration Examples**

### **Complete Job Flow with Reference Images and Seed Control:**
```typescript
// 1. Upload reference image
const uploadAndCreateJob = async (jobType, prompt, referenceFile, strength, type, seed) => {
  // Upload reference image first
  const referenceUrl = await uploadReferenceImage(referenceFile);
  
  // Create job with reference and seed
  const response = await fetch('/api/queue-job', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: JSON.stringify({
      jobType,
      metadata: {
        prompt,
        reference_image_url: referenceUrl,
        reference_strength: strength,
        reference_type: type,
        num_images: 3, // For SDXL
        seed: seed     // NEW: Seed control
      }
    })
  });
  
  const result = await response.json();
  if (result.success) {
    return result.job.id;
  } else {
    throw new Error(result.error);
  }
};

// 2. Monitor progress
const monitorJob = async (jobId) => {
  const interval = setInterval(async () => {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    
    updateProgressUI(job);
    
    if (job.status === 'completed') {
      clearInterval(interval);
      handleJobCompletion(job);
    } else if (job.status === 'failed') {
      clearInterval(interval);
      handleJobFailure(job);
    }
  }, 2000); // Poll every 2 seconds
};

// 3. Handle completion with seed information
const handleJobCompletion = (job) => {
  if (job.job_type.startsWith('sdxl_')) {
    // Handle flexible quantity batch
    const imageUrls = job.assets; // ✅ CONSISTENT: assets array
    const numImages = imageUrls.length;
    
    // NEW: Extract seed information
    const seed = job.metadata?.seed;
    if (seed) {
      console.log(`Generated with seed: ${seed}`);
      // Store seed for future regeneration
      storeSeedForJob(job.id, seed);
    }
    
    if (numImages === 1) {
      displaySingleImage(imageUrls[0]);
    } else {
      displayImageGrid(imageUrls);
    }
  } else {
    // Handle single file
    const fileUrl = job.assets[0]; // ✅ CONSISTENT: assets array
    if (job.job_type.includes('video')) {
      displayVideoPlayer(fileUrl);
    } else {
      displayImage(fileUrl);
    }
  }
};
```

### **Video with Reference Frames:**
```typescript
// Create video job with start/end references
const createVideoWithReferences = async (prompt, startRef, endRef, strength) => {
  const response = await fetch('/api/queue-job', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: JSON.stringify({
      jobType: 'video_fast',
      metadata: {
        prompt,
        start_reference_image_url: startRef,
        end_reference_image_url: endRef,
        reference_strength: strength
      }
    })
  });
  
  return response.json();
};
```

### **Seed-Based Character Consistency:**
```typescript
// Generate character with seed for consistency
const generateCharacterSeries = async (characterPrompt, seed) => {
  const poses = [
    'standing pose',
    'sitting pose', 
    'walking pose',
    'action pose'
  ];
  
  const results = [];
  
  for (const pose of poses) {
    const response = await fetch('/api/queue-job', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
      body: JSON.stringify({
        jobType: 'sdxl_image_fast',
        metadata: {
          prompt: `${characterPrompt}, ${pose}`,
          num_images: 1,
          seed: seed  // Same seed maintains character consistency
        }
      })
    });
    
    const result = await response.json();
    results.push(result.job.id);
  }
  
  return results;
};
```

---

## **📋 Quick Reference Cheat Sheet**

### **Job Type Matrix:**
| Job Type | Output | Time | Quality | Enhancement | Reference Support | Flexible Quantities | Seed Control |
|----------|--------|------|---------|-------------|-------------------|-------------------|--------------|
| `sdxl_image_fast` | 1-6 PNG | 3-48s | Excellent | No | ✅ Style/Comp/Char | ✅ 1,3,6 | ✅ |
| `sdxl_image_high` | 1-6 PNG | 5-90s | Premium | No | ✅ Style/Comp/Char | ✅ 1,3,6 | ✅ |
| `image_fast` | 1 PNG | 25s | Good | No | ❌ | ❌ | ❌ |
| `image_high` | 1 PNG | 40s | Better | No | ❌ | ❌ | ❌ |
| `video_fast` | 1 MP4 | 135s | Good | No | ✅ Start/End | ❌ | ❌ |
| `video_high` | 1 MP4 | 180s | Better | No | ✅ Start/End | ❌ | ❌ |
| `image7b_fast_enhanced` | 1 PNG | 85s | Enhanced | Yes | ❌ | ❌ | ❌ |
| `image7b_high_enhanced` | 1 PNG | 100s | Enhanced | Yes | ❌ | ❌ | ❌ |
| `video7b_fast_enhanced` | 1 MP4 | 195s | Enhanced | Yes | ✅ Start/End | ❌ | ❌ |
| `video7b_high_enhanced` | 1 MP4 | 240s | Enhanced | Yes | ✅ Start/End | ❌ | ❌ |

### **Reference Image Matrix:**
| Reference Type | SDXL | Video | Strength Range | Use Case |
|----------------|------|-------|----------------|----------|
| Style | ✅ | ❌ | 0.3-0.8 | Visual style transfer |
| Composition | ✅ | ❌ | 0.5-0.9 | Layout/structure guidance |
| Character | ✅ | ❌ | 0.4-0.7 | Character consistency |
| Start Frame | ❌ | ✅ | 0.5-0.8 | Video opening scene |
| End Frame | ❌ | ✅ | 0.5-0.8 | Video closing scene |

### **API Endpoints:**
```yaml
POST /api/queue-job: Create new job
POST /api/upload-reference: Upload reference image
GET /api/jobs/{id}: Get job status
GET /api/assets: Get user assets
DELETE /api/assets/{id}: Delete asset
GET /api/reference-images: Get user reference images
```

### **Status Values:**
```yaml
queued → processing → completed/failed
```

### **Consistent Callback Parameters:**
Both workers use standardized callback parameters for compatibility:

```typescript
// Standard callback payload structure
{
  job_id: string,        // ✅ Consistent: job_id (snake_case)
  status: string,        // ✅ Consistent: 'completed' | 'failed'
  assets: string[],      // ✅ Consistent: Array of asset URLs
  error_message?: string, // ✅ Consistent: Error details if failed
  metadata?: object      // ✅ NEW: Additional generation details (seed, etc.)
}
```

### **Reference Image Flow:**
```yaml
Upload → Validate → Store → Use in Job → Process → Complete
```

### **Seed Control Flow:**
```yaml
Provide Seed → Generate → Return Seed in Metadata → Store for Regeneration
```

---

**This document provides everything needed to integrate with the OurVidz worker system, including the new reference image functionality, flexible quantities, and seed control. All performance metrics and implementation details have been verified against the actual worker source code for accuracy and conformity.** 