# OurVidz Worker API Reference

**Last Updated:** July 7, 2025  
**Status:** ✅ Production Ready - All 10 Job Types Operational  
**System:** Dual Worker (SDXL + WAN) on RTX 6000 ADA (48GB VRAM)

---

## **🎯 Quick Reference**

### **Active Job Types (10 Total)**
```yaml
SDXL Jobs (2) - 6-Image Batches:
  sdxl_image_fast: 29.9s (3.1s per image) - 1024x1024 PNG
  sdxl_image_high: 42.4s (5.0s per image) - 1024x1024 PNG

WAN Standard Jobs (4) - Single Files:
  image_fast: 73s - 480x832 PNG
  image_high: 90s - 480x832 PNG  
  video_fast: 241.4s - 480x832 MP4 (5s duration)
  video_high: 360s - 480x832 MP4 (6s duration)

WAN Enhanced Jobs (4) - AI-Enhanced:
  image7b_fast_enhanced: 233.5s - 480x832 PNG (Qwen enhanced)
  image7b_high_enhanced: 104s - 480x832 PNG (Qwen enhanced)
  video7b_fast_enhanced: 266.1s - 480x832 MP4 (Qwen enhanced)
  video7b_high_enhanced: 361s - 480x832 MP4 (Qwen enhanced)
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
    bucket?: string          // Storage bucket (auto-detected)
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
    metadata: object         // Job metadata
  },
  message: string,           // Success message
  queueLength: number,       // Current queue depth
  modelVariant: string,      // Model being used
  isSDXL: boolean,          // SDXL vs WAN job
  negativePromptSupported: boolean  // SDXL only
}
```

### **Example Request:**
```typescript
// SDXL Fast Image Generation (6 images)
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
      credits: 1
    }
  })
});

// WAN Video Generation (single video)
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
      credits: 1
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
Workers automatically call back to update job status. Frontend should poll for updates:

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

### **SDXL Jobs (6-Image Batches):**
```typescript
// SDXL returns array of 6 image URLs
const handleSDXLCompletion = (job) => {
  const imageUrls = job.metadata.all_assets; // Array of 6 URLs
  const primaryImage = job.metadata.primary_asset; // First image
  
  // Display 6 images in grid
  displayImageGrid(imageUrls);
  
  // Store in database
  await saveImagesToDatabase(imageUrls, job.id);
};
```

### **WAN Jobs (Single Files):**
```typescript
// WAN returns single file URL
const handleWANCompletion = (job) => {
  const fileUrl = job.metadata.primary_asset; // Single URL
  
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

### **SDXL Image Generation**
```yaml
Models: LUSTIFY SDXL (NSFW-capable)
Resolution: 1024x1024 (square)
Format: PNG
Batch Size: 6 images per job
Quality: Excellent NSFW content

Performance:
  sdxl_image_fast: 29.9s total (3.1s per image)
  sdxl_image_high: 42.4s total (5.0s per image)

Storage Buckets:
  sdxl_image_fast: 5MB limit per image
  sdxl_image_high: 10MB limit per image

Frontend Handling:
  - Display 6 images in grid layout
  - Allow user to select preferred image
  - Store all 6 images in user library
  - Enable download of individual images
```

### **WAN Video Generation**
```yaml
Models: WAN 2.1 T2V 1.3B
Resolution: 480x832 (portrait)
Format: MP4
Duration: 5-6 seconds
Quality: Good to excellent

Performance:
  video_fast: 241.4s average (4m 1s)
  video_high: 360s average (6m)
  video7b_fast_enhanced: 266.1s average (4m 26s)
  video7b_high_enhanced: 361s average (6m 1s)

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
```

### **WAN Image Generation**
```yaml
Models: WAN 2.1 T2V 1.3B
Resolution: 480x832 (portrait)
Format: PNG
Quality: Backup images (not primary)

Performance:
  image_fast: 73s (estimated)
  image_high: 90s (estimated)
  image7b_fast_enhanced: 233.5s (Qwen enhanced)
  image7b_high_enhanced: 104s (Qwen enhanced)

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

Job Processing Errors:
  - Model loading failed: Retry automatically
  - Generation timeout: Mark as failed after 15 minutes
  - Storage upload failed: Retry with exponential backoff
  - GPU memory issues: Worker auto-restart

Frontend Error Handling:
  - Show user-friendly error messages
  - Provide retry options where appropriate
  - Log errors for debugging
  - Graceful degradation for partial failures
```

### **Error Response Format:**
```typescript
{
  success: false,
  error: string,           // User-friendly error message
  details?: string,        // Technical details for debugging
  retryable?: boolean,     // Whether retry is recommended
  suggestedAction?: string // What user should do
}
```

---

## **📈 Performance Monitoring**

### **Key Metrics to Track:**
```yaml
Job Success Rate: >95% target
Average Generation Times:
  - SDXL fast: 29.9s
  - SDXL high: 42.4s
  - WAN video_fast: 241.4s
  - WAN video_high: 360s
  - Enhanced jobs: +14-112s overhead

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
```

### **Frontend Monitoring:**
```typescript
// Track job performance
const trackJobPerformance = (jobType, startTime, endTime) => {
  const duration = endTime - startTime;
  analytics.track('job_completed', {
    jobType,
    duration,
    success: true
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
```

---

## **🔗 Integration Examples**

### **Complete Job Flow:**
```typescript
// 1. Create job
const createJob = async (jobType, prompt) => {
  const response = await fetch('/api/queue-job', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: JSON.stringify({ jobType, metadata: { prompt } })
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

// 3. Handle completion
const handleJobCompletion = (job) => {
  if (job.job_type.startsWith('sdxl_')) {
    // Handle 6-image batch
    const imageUrls = job.metadata.all_assets;
    displayImageGrid(imageUrls);
  } else {
    // Handle single file
    const fileUrl = job.metadata.primary_asset;
    if (job.job_type.includes('video')) {
      displayVideoPlayer(fileUrl);
    } else {
      displayImage(fileUrl);
    }
  }
};
```

---

## **📋 Quick Reference Cheat Sheet**

### **Job Type Matrix:**
| Job Type | Output | Time | Quality | Enhancement |
|----------|--------|------|---------|-------------|
| `sdxl_image_fast` | 6 PNG | 30s | Excellent | No |
| `sdxl_image_high` | 6 PNG | 42s | Premium | No |
| `image_fast` | 1 PNG | 73s | Good | No |
| `image_high` | 1 PNG | 90s | Better | No |
| `video_fast` | 1 MP4 | 241s | Good | No |
| `video_high` | 1 MP4 | 360s | Better | No |
| `image7b_fast_enhanced` | 1 PNG | 234s | Enhanced | Yes |
| `image7b_high_enhanced` | 1 PNG | 104s | Enhanced | Yes |
| `video7b_fast_enhanced` | 1 MP4 | 266s | Enhanced | Yes |
| `video7b_high_enhanced` | 1 MP4 | 361s | Enhanced | Yes |

### **API Endpoints:**
```yaml
POST /api/queue-job: Create new job
GET /api/jobs/{id}: Get job status
GET /api/assets: Get user assets
DELETE /api/assets/{id}: Delete asset
```

### **Status Values:**
```yaml
queued → processing → completed/failed
```

---

**This document provides everything needed to integrate with the OurVidz worker system. For technical implementation details, see the worker source code in this repository.** 