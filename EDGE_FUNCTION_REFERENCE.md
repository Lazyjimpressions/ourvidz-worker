# Job Callback Edge Function Reference

## Overview
This edge function serves as the central callback handler for OurVidz AI content generation workers. It processes completion notifications from both SDXL and WAN workers, updating job statuses and managing file paths in the Supabase database.

## 🚀 **FUNCTION PURPOSE**
**Primary Role**: Process worker completion callbacks and update database records
- **SDXL Workers**: Handle batch image generation (6 images per job)
- **WAN Workers**: Handle single image/video generation with AI enhancement
- **Status Management**: Update jobs, images, and videos tables
- **File Path Handling**: Normalize and store generated content URLs

## 📋 **SUPPORTED JOB TYPES**

### **SDXL Jobs** (Fast Image Generation)
| Job Type | Format | Quality | Output | Batch Size |
|----------|--------|---------|--------|------------|
| `sdxl_image_fast` | Image | Fast (15 steps) | PNG | 6 images |
| `sdxl_image_high` | Image | High (25 steps) | PNG | 6 images |

### **WAN Jobs** (Video + Image Generation)
| Job Type | Format | Quality | Enhancement | Output |
|----------|--------|---------|-------------|--------|
| `image_fast` | Image | Fast (4 steps) | No | PNG |
| `image_high` | Image | High (6 steps) | No | PNG |
| `video_fast` | Video | Fast (4 steps) | No | MP4 |
| `video_high` | Video | High (6 steps) | No | MP4 |
| `image7b_fast_enhanced` | Image | Fast (4 steps) | Yes | PNG |
| `image7b_high_enhanced` | Image | High (6 steps) | Yes | PNG |
| `video7b_fast_enhanced` | Video | Fast (4 steps) | Yes | MP4 |
| `video7b_high_enhanced` | Video | High (6 steps) | Yes | MP4 |

## 🔧 **CALLBACK PARAMETERS**

### **Required Parameters**
```typescript
{
  jobId: string,           // Database job ID (required)
  status: 'processing' | 'completed' | 'failed',  // Job status
}
```

### **Optional Parameters**
```typescript
{
  filePath?: string,       // SDXL workers: file path in storage
  outputUrl?: string,      // WAN workers: output URL
  errorMessage?: string,   // Error details for failed jobs
  enhancedPrompt?: string, // AI-enhanced prompt (for enhanced jobs)
  imageUrls?: string[],    // Multiple image URLs (SDXL batch)
}
```

### **Parameter Compatibility**
- **SDXL Workers**: Send `filePath` parameter
- **WAN Workers**: Send `outputUrl` parameter
- **Function Logic**: Resolves to `resolvedFilePath = filePath || outputUrl`

## 🗄️ **DATABASE UPDATES**

### **Jobs Table Updates**
```typescript
{
  status: string,                    // 'processing' | 'completed' | 'failed'
  completed_at: string | null,       // ISO timestamp for completed/failed
  error_message: string | null,      // Error details if failed
  metadata: {                        // Enhanced metadata object
    file_path?: string,              // Resolved file path
    enhanced_prompt?: string,        // AI-enhanced prompt
    callback_processed_at: string,   // Processing timestamp
    callback_debug: object,          // Debug information
    model_type: 'sdxl' | 'wan',     // Model identification
    bucket: string,                  // Storage bucket name
    is_sdxl: boolean,               // SDXL flag
    file_path_validation: object,   // Path validation details
    debug_info: object              // Additional debugging
  }
}
```

### **Images Table Updates** (Image Jobs)
```typescript
{
  status: 'completed' | 'failed' | 'generating',
  image_url: string,                 // Primary image URL
  image_urls: string[] | null,       // Multiple URLs (SDXL batch)
  thumbnail_url: string,             // Thumbnail reference
  quality: 'fast' | 'high',          // Generation quality
  metadata: {                        // Enhanced metadata
    model_type: 'sdxl' | 'wan',
    is_sdxl: boolean,
    bucket: string,
    callback_processed_at: string,
    file_path_validation: object,
    debug_info: object
  }
}
```

### **Videos Table Updates** (Video Jobs)
```typescript
{
  status: 'completed' | 'failed' | 'processing',
  video_url: string,                 // Video file URL
  completed_at: string,              // Completion timestamp
  error_message?: string             // Error details if failed
}
```

## 🔍 **JOB TYPE PARSING LOGIC**

### **SDXL Job Parsing**
```typescript
// Input: 'sdxl_image_fast'
// Output: { format: 'image', quality: 'fast', isSDXL: true, isEnhanced: false }
if (job.job_type.startsWith('sdxl_')) {
  isSDXL = true;
  const parts = job.job_type.replace('sdxl_', '').split('_');
  format = parts[0];    // 'image'
  quality = parts[1];   // 'fast' or 'high'
}
```

### **Enhanced WAN Job Parsing**
```typescript
// Input: 'video7b_fast_enhanced'
// Output: { format: 'video', quality: 'fast', isSDXL: false, isEnhanced: true }
if (job.job_type.includes('enhanced')) {
  isEnhanced = true;
  
  if (job.job_type.startsWith('video7b_')) {
    format = 'video';
    quality = job.job_type.includes('_fast_') ? 'fast' : 'high';
  } else if (job.job_type.startsWith('image7b_')) {
    format = 'image';
    quality = job.job_type.includes('_fast_') ? 'fast' : 'high';
  }
}
```

### **Standard WAN Job Parsing**
```typescript
// Input: 'image_fast'
// Output: { format: 'image', quality: 'fast', isSDXL: false, isEnhanced: false }
const parts = job.job_type.split('_');
format = parts[0];    // 'image' or 'video'
quality = parts[1];   // 'fast' or 'high'
```

## 📁 **FILE PATH HANDLING**

### **Path Normalization Function**
```typescript
function normalizeAssetPath(filePath, userId) {
  if (!filePath || !userId) return filePath;
  
  // Check if path already contains user ID prefix
  if (filePath.startsWith(`${userId}/`)) {
    return filePath; // Already user-scoped
  }
  
  // Add user ID prefix for consistency
  return `${userId}/${filePath}`;
}
```

### **SDXL Path Handling**
- **Images**: User-scoped paths with SDXL prefix
- **Pattern**: `${userId}/sdxl_${jobId}_*.png`
- **Batch**: Multiple URLs in `imageUrls` array
- **Primary**: First image in array becomes `image_url`

### **WAN Path Handling**
- **Images**: User-scoped paths without prefix
- **Videos**: Bucket root paths (just filename)
- **Pattern**: `${userId}/${jobId}_*.png` or `${jobId}_*.mp4`

## 🛡️ **ERROR HANDLING & VALIDATION**

### **Critical Validations**
```typescript
// Job ID validation
if (!jobId) {
  throw new Error('jobId is required');
}

// File path validation for completed jobs
if (!resolvedFilePath && status === 'completed') {
  console.error('❌ CRITICAL: No file path provided for completed job');
}

// Job fetch validation
const { data: currentJob, error: fetchError } = await supabase
  .from('jobs')
  .select('metadata, job_type, image_id, video_id, format, quality, model_type, user_id')
  .eq('id', jobId)
  .single();
```

### **File Path Validation**
```typescript
const filePathValidation = {
  hasSlash: primaryImageUrl ? primaryImageUrl.includes('/') : false,
  hasUnderscore: primaryImageUrl ? primaryImageUrl.includes('_') : false,
  hasPngExtension: primaryImageUrl ? primaryImageUrl.endsWith('.png') : false,
  length: primaryImageUrl ? primaryImageUrl.length : 0,
  startsWithUserId: primaryImageUrl ? primaryImageUrl.startsWith(job.user_id) : false,
  expectedPattern: `${job.user_id}/${isSDXL ? 'sdxl_' : ''}${job.id}_*.png`,
  isMultipleImages: !!imageUrlsArray,
  imageCount: imageUrlsArray ? imageUrlsArray.length : 1
};
```

## 🔄 **PROCESSING FLOW**

### **1. Request Validation**
- Handle CORS preflight requests
- Parse and validate callback parameters
- Resolve file path compatibility (SDXL vs WAN)

### **2. Job Retrieval**
- Fetch current job details from database
- Preserve existing metadata
- Validate job exists and has required fields

### **3. Status Processing**
- Update job status and completion timestamp
- Merge metadata instead of overwriting
- Handle enhanced prompts for image_high jobs

### **4. File Path Processing**
- Normalize paths for user-scoped consistency
- Handle multiple images (SDXL batch)
- Validate file path format and structure

### **5. Database Updates**
- Update jobs table with status and metadata
- Update images/videos tables based on job type
- Store debugging information for troubleshooting

### **6. Response Generation**
- Return success/error response with debugging info
- Include processing timestamp and job details

## 📊 **DEBUGGING & MONITORING**

### **Enhanced Logging**
```typescript
console.log('🔍 ENHANCED CALLBACK DEBUGGING - Received request:', {
  jobId,
  status,
  filePath,
  outputUrl,
  resolvedFilePath,
  imageUrls,
  errorMessage,
  enhancedPrompt,
  fullRequestBody: requestBody,
  timestamp: new Date().toISOString()
});
```

### **Debug Information Storage**
```typescript
metadata: {
  callback_processed_at: new Date().toISOString(),
  callback_debug: {
    received_file_path: filePath,
    received_output_url: outputUrl,
    resolved_file_path: resolvedFilePath,
    job_type: currentJob.job_type,
    processing_timestamp: new Date().toISOString()
  },
  file_path_validation: filePathValidation,
  debug_info: {
    original_file_path: filePath,
    image_urls_received: imageUrlsArray,
    job_type: job.job_type,
    processed_at: new Date().toISOString()
  }
}
```

## 🚨 **CRITICAL CONSIDERATIONS**

### **Environment Variables**
```bash
SUPABASE_URL=              # Supabase database URL
SUPABASE_SERVICE_ROLE_KEY= # Supabase service key (admin access)
```

### **CORS Headers**
```typescript
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type'
};
```

### **Error Response Format**
```typescript
{
  error: string,           // Error message
  success: false,          // Success flag
  debug: {                 // Debug information
    timestamp: string,     // ISO timestamp
    errorType: string      // Error constructor name
  }
}
```

### **Success Response Format**
```typescript
{
  success: true,
  message: 'Job callback processed successfully with enhanced debugging',
  debug: {
    jobId: string,
    jobStatus: string,
    jobType: string,
    format: string,
    quality: string,
    isSDXL: boolean,
    isEnhanced: boolean,
    filePath: string,
    processingTimestamp: string
  }
}
```

## 🔧 **INTEGRATION POINTS**

### **Worker Integration**
- **SDXL Workers**: Send `filePath` and `imageUrls` for batch processing
- **WAN Workers**: Send `outputUrl` and `enhancedPrompt` for single processing
- **Error Handling**: Send `errorMessage` for failed jobs

### **Database Integration**
- **Jobs Table**: Status updates and metadata storage
- **Images Table**: Image URL storage and batch handling
- **Videos Table**: Video URL storage and completion tracking

### **Storage Integration**
- **Supabase Storage**: File path normalization and bucket management
- **Asset Service**: Compatible URL formats for frontend access

## 📋 **DEPLOYMENT NOTES**

### **Edge Function Requirements**
- **Runtime**: Deno
- **Permissions**: Supabase service role access
- **CORS**: Configured for cross-origin requests
- **Timeout**: Standard edge function limits

### **Monitoring Considerations**
- **Logging**: Comprehensive debug logging for troubleshooting
- **Error Tracking**: Detailed error information in responses
- **Performance**: Optimized for quick callback processing
- **Reliability**: Graceful handling of missing or invalid data

---

## **🎯 SUMMARY**

This edge function is the **central nervous system** for OurVidz worker completion handling. It:

1. **Processes all worker callbacks** (SDXL and WAN)
2. **Manages database updates** across jobs, images, and videos tables
3. **Handles file path normalization** for consistent storage access
4. **Provides comprehensive debugging** for troubleshooting
5. **Supports batch processing** for SDXL multi-image generation
6. **Manages enhanced prompts** for AI-powered content generation

The function is **production-ready** with robust error handling, detailed logging, and compatibility with all current worker types in the OurVidz system. 