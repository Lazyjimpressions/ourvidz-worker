# OurVidz Edge Functions - Complete Implementation Reference

**Last Updated:** July 6, 2025 at 10:11 AM CST  
**Status:** ✅ Production Ready - All Functions Aligned with Worker Conventions

---

## **Overview**

This document contains the complete implementation of OurVidz edge functions, preserved for reference and development. All functions are **perfectly aligned** with the worker parameter conventions and support all 10 job types.

---

## **Queue-Job Edge Function (`queue-job.ts`)**

**Purpose**: Job creation and queue routing with standardized parameter handling  
**Authentication**: JWT verification required  
**Status**: ✅ Production Ready

```typescript
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type'
};

serve(async (req)=>{
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      headers: corsHeaders
    });
  }
  
  try {
    console.log('🚀 Queue-job function called - STANDARDIZED: Worker callback parameter consistency');
    const supabase = createClient(Deno.env.get('SUPABASE_URL') ?? '', Deno.env.get('SUPABASE_ANON_KEY') ?? '', {
      global: {
        headers: {
          Authorization: req.headers.get('Authorization')
        }
      }
    });
    
    // Get the current user
    const { data: { user }, error: userError } = await supabase.auth.getUser();
    if (userError || !user) {
      console.error('❌ Authentication failed:', userError?.message);
      return new Response(JSON.stringify({
        error: 'Authentication required',
        success: false,
        details: userError?.message
      }), {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json'
        },
        status: 401
      });
    }
    
    console.log('✅ User authenticated:', user.id);
    const { jobType, metadata, projectId, videoId, imageId } = await req.json();
    
    console.log('📋 Creating job with STANDARDIZED worker parameters:', {
      jobType,
      projectId,
      videoId,
      imageId,
      userId: user.id,
      queue: metadata?.queue,
      timestamp: new Date().toISOString()
    });
    
    // FIXED: Negative prompt generation - ONLY for SDXL jobs
    function generateNegativePromptForSDXL(userPrompt = '') {
      console.log('🎨 Generating negative prompt for SDXL job only');
      // SDXL-optimized negative prompts (keep under 77 tokens)
      const criticalNegatives = [
        "bad anatomy",
        "extra limbs",
        "deformed",
        "missing limbs"
      ];
      const qualityNegatives = [
        "low quality",
        "bad quality",
        "worst quality",
        "blurry",
        "pixelated"
      ];
      const anatomicalNegatives = [
        "deformed hands",
        "extra fingers",
        "deformed face",
        "malformed"
      ];
      const artifactNegatives = [
        "text",
        "watermark",
        "logo",
        "signature"
      ];
      // NSFW-specific anatomical improvements for SDXL
      const nsfwNegatives = [
        "deformed breasts",
        "extra breasts",
        "anatomical errors",
        "wrong anatomy",
        "distorted bodies",
        "unnatural poses"
      ];
      // Build SDXL negative prompt (token-efficient)
      const sdxlNegatives = [
        ...criticalNegatives,
        ...qualityNegatives.slice(0, 3),
        ...anatomicalNegatives.slice(0, 4),
        ...artifactNegatives.slice(0, 3),
        "ugly",
        "poorly drawn"
      ];
      // Add NSFW negatives if applicable
      if (userPrompt.toLowerCase().includes('naked') || userPrompt.toLowerCase().includes('nude') || userPrompt.toLowerCase().includes('sex')) {
        sdxlNegatives.push(...nsfwNegatives.slice(0, 4)); // Limit for token efficiency
      }
      const result = sdxlNegatives.join(", ");
      console.log('✅ SDXL negative prompt generated:', result);
      return result;
    }
    
    // Enhanced job type validation
    const validJobTypes = [
      'sdxl_image_fast',
      'sdxl_image_high',
      'image_fast',
      'image_high',
      'video_fast',
      'video_high',
      'image7b_fast_enhanced',
      'image7b_high_enhanced',
      'video7b_fast_enhanced',
      'video7b_high_enhanced'
    ];
    
    if (!validJobTypes.includes(jobType)) {
      console.error('❌ Invalid job type provided:', jobType);
      console.log('✅ Valid job types:', validJobTypes);
      return new Response(JSON.stringify({
        error: `Invalid job type: ${jobType}`,
        success: false,
        validJobTypes: validJobTypes
      }), {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json'
        },
        status: 400
      });
    }
    
    // Robust parsing function for all job type patterns
    function parseJobType(jobType) {
      const isSDXL = jobType.startsWith('sdxl_');
      const isEnhanced = jobType.includes('enhanced');
      let format;
      let quality;
      
      if (isSDXL) {
        // SDXL patterns: sdxl_image_fast, sdxl_image_high
        const parts = jobType.split('_');
        format = parts[1]; // 'image'
        quality = parts[2]; // 'fast' or 'high'
      } else if (isEnhanced) {
        // Enhanced patterns: image7b_fast_enhanced, video7b_high_enhanced
        if (jobType.startsWith('image7b_')) {
          format = 'image';
          quality = jobType.includes('_fast_') ? 'fast' : 'high';
        } else if (jobType.startsWith('video7b_')) {
          format = 'video';
          quality = jobType.includes('_fast_') ? 'fast' : 'high';
        } else {
          // Fallback for unknown enhanced patterns
          format = jobType.includes('video') ? 'video' : 'image';
          quality = jobType.includes('fast') ? 'fast' : 'high';
        }
      } else {
        // Standard patterns: image_fast, image_high, video_fast, video_high
        const parts = jobType.split('_');
        format = parts[0]; // 'image' or 'video'
        quality = parts[1]; // 'fast' or 'high'
      }
      
      return {
        format,
        quality,
        isSDXL,
        isEnhanced
      };
    }
    
    // Extract format and quality from job type
    const { format, quality, isSDXL, isEnhanced } = parseJobType(jobType);
    const modelVariant = isSDXL ? 'lustify_sdxl' : 'wan_2_1_1_3b';
    
    // Determine queue routing - all enhanced jobs use wan_queue
    const queueName = isSDXL ? 'sdxl_queue' : 'wan_queue';
    
    // Enhanced logging with format and quality detection
    console.log('🎯 FIXED job routing determined:', {
      isSDXL,
      isEnhanced,
      queueName,
      modelVariant,
      format,
      quality,
      originalJobType: jobType,
      negativePromptSupported: isSDXL,
      parsedCorrectly: true
    });
    
    // Validate Redis configuration
    const redisUrl = Deno.env.get('UPSTASH_REDIS_REST_URL');
    const redisToken = Deno.env.get('UPSTASH_REDIS_REST_TOKEN');
    if (!redisUrl || !redisToken) {
      console.error('❌ Redis configuration missing:', {
        hasUrl: !!redisUrl,
        hasToken: !!redisToken
      });
      return new Response(JSON.stringify({
        error: 'Redis configuration missing',
        success: false,
        details: 'UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN not configured'
      }), {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json'
        },
        status: 500
      });
    }
    
    // Create job record with enhanced error handling
    const { data: job, error: jobError } = await supabase.from('jobs').insert({
      user_id: user.id,
      job_type: jobType,
      format: format,
      quality: quality,
      model_type: jobType,
      metadata: {
        ...metadata,
        model_variant: modelVariant,
        queue: queueName,
        dual_worker_routing: true,
        negative_prompt_supported: isSDXL,
        created_timestamp: new Date().toISOString()
      },
      project_id: projectId,
      video_id: videoId,
      image_id: imageId,
      status: 'queued'
    }).select().single();
    
    if (jobError) {
      console.error('❌ Error creating job in database:', {
        error: jobError,
        jobType,
        userId: user.id,
        format,
        quality
      });
      return new Response(JSON.stringify({
        error: 'Failed to create job record',
        success: false,
        details: jobError.message,
        jobType: jobType
      }), {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json'
        },
        status: 500
      });
    }
    
    console.log('✅ Job created successfully in database:', job.id);
    
    // Get project details for the prompt (if projectId provided)
    let prompt = '';
    let characterId = null;
    if (projectId) {
      const { data: project, error: projectError } = await supabase.from('projects').select('enhanced_prompt, original_prompt, character_id').eq('id', projectId).single();
      if (!projectError && project) {
        prompt = project.enhanced_prompt || project.original_prompt || '';
        characterId = project.character_id;
        console.log('📄 Project prompt retrieved:', {
          projectId,
          hasPrompt: !!prompt
        });
      } else {
        console.warn('⚠️ Could not retrieve project prompt:', projectError?.message);
      }
    }
    
    // Use prompt from metadata if no project prompt available
    if (!prompt && metadata?.prompt) {
      prompt = metadata.prompt;
      console.log('📝 Using metadata prompt');
    }
    
    // CRITICAL FIX: Only generate negative prompt for SDXL jobs
    let negativePrompt = '';
    if (isSDXL) {
      negativePrompt = generateNegativePromptForSDXL(prompt);
      console.log('🚫 Generated SDXL negative prompt:', negativePrompt);
    } else {
      console.log('🚫 WAN job detected - NO negative prompt (not supported by WAN 2.1)');
    }
    
    // Format job payload for appropriate worker
    const jobPayload = {
      id: job.id,
      type: jobType,
      prompt: prompt,
      config: {
        size: '480*832',
        sample_steps: quality === 'high' ? 50 : 25,
        sample_guide_scale: 5.0,
        frame_num: format === 'video' ? 83 : 1,
        enhance_prompt: isEnhanced,
        expected_time: isEnhanced ? format === 'video' ? quality === 'high' ? 294 : 194 : quality === 'high' ? 104 : 87 : format === 'video' ? quality === 'high' ? 280 : 180 : quality === 'high' ? 90 : 73,
        content_type: format,
        file_extension: format === 'video' ? 'mp4' : 'png'
      },
      user_id: user.id,
      created_at: new Date().toISOString(),
      // CRITICAL FIX: Only include negative_prompt for SDXL jobs
      ...isSDXL && {
        negative_prompt: negativePrompt
      },
      // Additional metadata
      video_id: videoId,
      image_id: imageId,
      character_id: characterId,
      model_variant: modelVariant,
      bucket: metadata?.bucket || (isSDXL ? `sdxl_image_${quality}` : isEnhanced ? `${format}7b_${quality}_enhanced` : `${format}_${quality}`),
      metadata: {
        ...metadata,
        model_variant: modelVariant,
        dual_worker_routing: true,
        negative_prompt_supported: isSDXL,
        // Only include negative_prompt in metadata for SDXL
        ...isSDXL && {
          negative_prompt: negativePrompt
        },
        num_images: isSDXL ? 6 : 1,
        queue_timestamp: new Date().toISOString()
      }
    };
    
    console.log('📤 Pushing FIXED job to Redis queue:', {
      jobId: job.id,
      jobType,
      queueName,
      isSDXL,
      hasPrompt: !!prompt,
      hasNegativePrompt: isSDXL && !!negativePrompt,
      negativePromptSupported: isSDXL,
      payloadSize: JSON.stringify(jobPayload).length
    });
    
    // Use LPUSH to add job to the appropriate queue (worker uses RPOP)
    const redisResponse = await fetch(`${redisUrl}/lpush/${queueName}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${redisToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(jobPayload)
    });
    
    if (!redisResponse.ok) {
      const redisError = await redisResponse.text();
      console.error('❌ Redis push failed:', {
        status: redisResponse.status,
        statusText: redisResponse.statusText,
        error: redisError,
        queueName,
        jobId: job.id
      });
      // Update job status to failed
      await supabase.from('jobs').update({
        status: 'failed',
        error_message: `Redis queue failed: ${redisError}`
      }).eq('id', job.id);
      return new Response(JSON.stringify({
        error: `Failed to queue job in Redis: ${redisError}`,
        success: false,
        details: {
          redisStatus: redisResponse.status,
          queueName,
          jobId: job.id
        }
      }), {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json'
        },
        status: 500
      });
    }
    
    const redisResult = await redisResponse.json();
    console.log('✅ Job queued in Redis successfully:', {
      jobId: job.id,
      queueLength: redisResult.result || 0,
      queueName,
      negativePromptIncluded: isSDXL
    });
    
    // Log usage with enhanced dual worker tracking
    const usageLogResult = await supabase.from('usage_logs').insert({
      user_id: user.id,
      action: jobType,
      format: format,
      quality: quality,
      credits_consumed: metadata.credits || 1,
      metadata: {
        job_id: job.id,
        project_id: projectId,
        image_id: imageId,
        video_id: videoId,
        model_type: jobType,
        model_variant: modelVariant,
        queue: queueName,
        dual_worker_routing: true,
        negative_prompt_supported: isSDXL,
        usage_timestamp: new Date().toISOString()
      }
    });
    
    if (usageLogResult.error) {
      console.warn('⚠️ Usage logging failed:', usageLogResult.error);
    } else {
      console.log('📈 Usage logged successfully');
    }
    
    return new Response(JSON.stringify({
      success: true,
      job,
      message: 'Job queued successfully - FIXED: WAN negative prompt removal',
      queueLength: redisResult.result || 0,
      modelVariant: modelVariant,
      jobType: jobType,
      queue: queueName,
      isSDXL: isSDXL,
      negativePromptSupported: isSDXL,
      fixes_applied: [
        'Removed negative prompt generation for WAN jobs',
        'Simplified job payload structure',
        'Fixed parameter naming consistency',
        'Added proper WAN 2.1 configuration'
      ],
      debug: {
        userId: user.id,
        hasPrompt: !!prompt,
        hasNegativePrompt: isSDXL && !!negativePrompt,
        redisConfigured: true,
        timestamp: new Date().toISOString()
      }
    }), {
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json'
      },
      status: 200
    });
    
  } catch (error) {
    console.error('❌ Unhandled error in queue-job function:', {
      error: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString()
    });
    return new Response(JSON.stringify({
      error: error.message,
      success: false,
      details: 'Unhandled server error',
      timestamp: new Date().toISOString()
    }), {
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json'
      },
      status: 500
    });
  }
});
```

---

## **Job Callback Edge Function (`job-callback.ts`)**

**Purpose**: Central callback handler for OurVidz AI content generation workers  
**Authentication**: JWT verification disabled (called by workers)  
**Status**: ✅ Production Ready

```typescript
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type'
};

serve(async (req)=>{
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      headers: corsHeaders
    });
  }
  
  try {
    const supabase = createClient(Deno.env.get('SUPABASE_URL') ?? '', Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '');
    const requestBody = await req.json();
    const { job_id, status, assets, error_message, enhancedPrompt } = requestBody;
    
    // Standardized parameter handling: all workers now send 'assets' array
    const primaryAsset = assets && assets.length > 0 ? assets[0] : null;
    
    console.log('🔍 STANDARDIZED CALLBACK DEBUGGING - Received request:', {
      job_id,
      status,
      assets,
      assetsCount: assets ? assets.length : 0,
      primaryAsset,
      error_message,
      enhancedPrompt,
      fullRequestBody: requestBody,
      timestamp: new Date().toISOString()
    });
    
    // Validate critical parameters with standardized naming
    if (!job_id) {
      console.error('❌ CRITICAL: No job_id provided in callback');
      throw new Error('job_id is required');
    }
    
    if (!primaryAsset && status === 'completed') {
      console.error('❌ CRITICAL: No assets provided for completed job', {
        job_id,
        status,
        assets,
        assetsCount: assets ? assets.length : 0,
        primaryAsset
      });
    }
    
    // Get current job to preserve existing metadata and check format
    console.log('🔍 Fetching job details for:', job_id);
    const { data: currentJob, error: fetchError } = await supabase.from('jobs').select('metadata, job_type, image_id, video_id, format, quality, model_type, user_id').eq('id', job_id).single();
    
    if (fetchError) {
      console.error('❌ CRITICAL: Error fetching current job:', {
        job_id,
        error: fetchError,
        errorMessage: fetchError.message,
        errorCode: fetchError.code
      });
      throw fetchError;
    }
    
    console.log('✅ Job details fetched successfully:', {
      job_id: currentJob.id,
      jobType: currentJob.job_type,
      imageId: currentJob.image_id,
      videoId: currentJob.video_id,
      userId: currentJob.user_id,
      quality: currentJob.quality,
      modelType: currentJob.model_type,
      existingMetadata: currentJob.metadata
    });
    
    // Prepare update data
    const updateData = {
      status,
      completed_at: status === 'completed' || status === 'failed' ? new Date().toISOString() : null,
      error_message: error_message || null
    };
    
    // Merge metadata instead of overwriting
    let updatedMetadata = currentJob.metadata || {};
    
    // Handle enhanced prompt for image_high jobs (enhancement)
    if (currentJob.job_type === 'image_high' && enhancedPrompt) {
      updatedMetadata.enhanced_prompt = enhancedPrompt;
      console.log('📝 Storing enhanced prompt for image_high job:', enhancedPrompt);
    }
    
    // Add assets for completed jobs with standardized validation
    if (status === 'completed' && primaryAsset) {
      console.log('📁 Processing completed job with standardized assets:', {
        job_id,
        assets,
        assetsCount: assets ? assets.length : 0,
        primaryAsset,
        assetLength: primaryAsset.length,
        assetPattern: primaryAsset.includes('/') ? 'contains slash' : 'no slash'
      });
      updatedMetadata.primary_asset = primaryAsset;
      updatedMetadata.all_assets = assets;
      updatedMetadata.callback_processed_at = new Date().toISOString();
      updatedMetadata.callback_debug = {
        received_assets: assets,
        primary_asset: primaryAsset,
        job_type: currentJob.job_type,
        processing_timestamp: new Date().toISOString()
      };
    } else if (status === 'completed' && !primaryAsset) {
      console.error('❌ CRITICAL: Completed job has no assets!', {
        job_id,
        status,
        assets,
        primaryAsset,
        jobType: currentJob.job_type
      });
      updatedMetadata.callback_error = {
        issue: 'completed_without_assets',
        timestamp: new Date().toISOString(),
        received_status: status,
        received_assets: assets,
        primary_asset: primaryAsset
      };
    }
    
    updateData.metadata = updatedMetadata;
    
    console.log('🔄 Updating job with standardized metadata:', {
      job_id,
      updateData,
      metadataKeys: Object.keys(updatedMetadata)
    });
    
    // Update job status
    const { data: job, error: updateError } = await supabase.from('jobs').update(updateData).eq('id', job_id).select().single();
    
    if (updateError) {
      console.error('❌ CRITICAL: Error updating job:', {
        job_id,
        error: updateError,
        updateData
      });
      throw updateError;
    }
    
    console.log('✅ Job updated successfully with standardized processing:', {
      job_id: job.id,
      status: job.status,
      jobType: job.job_type,
      metadata: job.metadata
    });
    
    // Enhanced job type parsing to handle SDXL jobs AND enhanced WAN jobs
    let format, quality, isSDXL = false, isEnhanced = false;
    
    if (job.job_type.startsWith('sdxl_')) {
      // Handle SDXL jobs: sdxl_image_fast -> image, fast, true
      isSDXL = true;
      const parts = job.job_type.replace('sdxl_', '').split('_');
      format = parts[0]; // 'image'
      quality = parts[1]; // 'fast' or 'high'
    } else if (job.job_type.includes('enhanced')) {
      // Handle enhanced WAN jobs: video7b_fast_enhanced, image7b_high_enhanced
      isEnhanced = true;
      if (job.job_type.startsWith('video7b_')) {
        format = 'video';
        quality = job.job_type.includes('_fast_') ? 'fast' : 'high';
      } else if (job.job_type.startsWith('image7b_')) {
        format = 'image';
        quality = job.job_type.includes('_fast_') ? 'fast' : 'high';
      } else {
        // Fallback for unknown enhanced patterns
        const parts = job.job_type.split('_');
        format = parts[0].replace('7b', ''); // Remove '7b' suffix
        quality = parts[1]; // 'fast' or 'high'
      }
    } else {
      // Handle standard WAN jobs: image_fast, video_high -> image/video, fast/high, false
      const parts = job.job_type.split('_');
      format = parts[0]; // 'image' or 'video'
      quality = parts[1]; // 'fast' or 'high'
    }
    
    console.log('🔧 Enhanced job type parsing with enhanced job support:', {
      originalJobType: job.job_type,
      parsedFormat: format,
      parsedQuality: quality,
      isSDXL,
      isEnhanced,
      expectedBucket: isSDXL ? `sdxl_image_${quality}` : isEnhanced ? `${format}7b_${quality}_enhanced` : `${format}_${quality}`
    });
    
    // Handle different job types based on parsed format with standardized assets
    if (format === 'image' && job.image_id) {
      console.log('🖼️ Processing image job callback...');
      await handleImageJobCallback(supabase, job, status, assets, error_message, quality, isSDXL, isEnhanced);
    } else if (format === 'video' && job.video_id) {
      console.log('📹 Processing video job callback...');
      await handleVideoJobCallback(supabase, job, status, assets, error_message, quality, isEnhanced);
    } else {
      console.error('❌ CRITICAL: Unknown job format or missing ID:', {
        format,
        imageId: job.image_id,
        videoId: job.video_id,
        jobType: job.job_type
      });
    }
    
    console.log('✅ STANDARDIZED CALLBACK PROCESSING COMPLETE:', {
      job_id,
      status,
      format,
      quality,
      isSDXL,
      isEnhanced,
      assets,
      assetsCount: assets ? assets.length : 0,
      processingTimestamp: new Date().toISOString()
    });
    
    return new Response(JSON.stringify({
      success: true,
      message: 'Job callback processed successfully with standardized parameters',
      debug: {
        job_id,
        jobStatus: status,
        jobType: job.job_type,
        format: format,
        quality: quality,
        isSDXL: isSDXL,
        isEnhanced: isEnhanced,
        assetsProcessed: assets ? assets.length : 0,
        processingTimestamp: new Date().toISOString()
      }
    }), {
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json'
      },
      status: 200
    });
    
  } catch (error) {
    console.error('❌ CRITICAL: Error in job callback function:', {
      error: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString()
    });
    return new Response(JSON.stringify({
      error: error.message,
      success: false,
      debug: {
        timestamp: new Date().toISOString(),
        errorType: error.constructor.name
      }
    }), {
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json'
      },
      status: 400
    });
  }
});

// Helper function to normalize asset paths to be user-scoped
function normalizeAssetPath(filePath, userId) {
  if (!filePath || !userId) return filePath;
  
  // Check if path already contains user ID prefix
  if (filePath.startsWith(`${userId}/`)) {
    return filePath; // Already user-scoped
  }
  
  // Add user ID prefix for consistency
  const normalizedPath = `${userId}/${filePath}`;
  console.log('🔧 Path normalization:', {
    originalPath: filePath,
    userId: userId,
    normalizedPath: normalizedPath
  });
  return normalizedPath;
}

async function handleImageJobCallback(supabase, job, status, assets, error_message, quality, isSDXL, isEnhanced) {
  console.log('🖼️ STANDARDIZED IMAGE CALLBACK PROCESSING:', {
    job_id: job.id,
    imageId: job.image_id,
    status,
    assets,
    assetsCount: assets ? assets.length : 0,
    jobType: job.job_type,
    quality,
    isSDXL,
    expectedBucket: isSDXL ? `sdxl_image_${quality}` : isEnhanced ? `image7b_${quality}_enhanced` : `image_${quality}`
  });
  
  if (status === 'completed' && assets && assets.length > 0) {
    console.log('✅ Processing completed image job with standardized assets');
    
    // Normalize paths to ensure user-scoped consistency
    let primaryImageUrl = normalizeAssetPath(assets[0], job.user_id);
    let imageUrlsArray = null;
    
    if (assets.length > 1) {
      console.log('🖼️ Multiple images received:', assets.length);
      // Normalize all image URLs in the array
      imageUrlsArray = assets.map((url)=>normalizeAssetPath(url, job.user_id));
      primaryImageUrl = imageUrlsArray[0]; // Use first image as primary
    } else {
      console.log('🖼️ Single image received:', assets[0]);
    }
    
    // Validate file path format
    const filePathValidation = {
      hasSlash: primaryImageUrl ? primaryImageUrl.includes('/') : false,
      hasUnderscore: primaryImageUrl ? primaryImageUrl.includes('_') : false,
      hasPngExtension: primaryImageUrl ? primaryImageUrl.endsWith('.png') : false,
      length: primaryImageUrl ? primaryImageUrl.length : 0,
      startsWithUserId: primaryImageUrl ? primaryImageUrl.startsWith(job.user_id || 'unknown') : false,
      expectedPattern: `${job.user_id}/${isSDXL ? 'sdxl_' : ''}${job.id}_*.png`,
      isMultipleImages: !!imageUrlsArray,
      imageCount: imageUrlsArray ? imageUrlsArray.length : 1
    };
    
    console.log('🔍 File path validation:', filePathValidation);
    
    // Update image record with model type information and enhanced debugging
    const updateData = {
      status: 'completed',
      image_url: primaryImageUrl,
      image_urls: imageUrlsArray,
      thumbnail_url: primaryImageUrl,
      quality: quality,
      metadata: {
        ...job.metadata || {},
        model_type: isSDXL ? 'sdxl' : 'wan',
        is_sdxl: isSDXL,
        bucket: isSDXL ? `sdxl_image_${quality}` : isEnhanced ? `image7b_${quality}_enhanced` : `image_${quality}`,
        callback_processed_at: new Date().toISOString(),
        file_path_validation: filePathValidation,
        debug_info: {
          original_assets: assets,
          image_urls_received: imageUrlsArray,
          job_type: job.job_type,
          processed_at: new Date().toISOString()
        }
      }
    };
    
    console.log('🔄 Updating image record:', {
      imageId: job.image_id,
      updateData,
      expectedBucket: updateData.metadata.bucket
    });
    
    const { data: updatedImage, error: imageError } = await supabase.from('images').update(updateData).eq('id', job.image_id).select().single();
    
    if (imageError) {
      console.error('❌ CRITICAL: Error updating image record:', {
        imageId: job.image_id,
        error: imageError,
        updateData
      });
    } else {
      console.log('✅ Image record updated successfully:', {
        imageId: updatedImage.id,
        status: updatedImage.status,
        imageUrl: updatedImage.image_url,
        quality: updatedImage.quality,
        bucket: updatedImage.metadata?.bucket,
        isSDXL: updatedImage.metadata?.is_sdxl
      });
      
      // Verify the image can be found by AssetService logic
      console.log('🔍 Verifying image accessibility for AssetService:', {
        imageId: updatedImage.id,
        userId: job.user_id,
        imageUrl: updatedImage.image_url,
        status: updatedImage.status,
        quality: updatedImage.quality,
        metadata: updatedImage.metadata
      });
    }
  } else if (status === 'failed') {
    console.log('❌ Processing failed image job');
    const { error: imageError } = await supabase.from('images').update({
      status: 'failed',
      metadata: {
        ...job.metadata || {},
        error_message: error_message,
        failed_at: new Date().toISOString()
      }
    }).eq('id', job.image_id);
    
    if (imageError) {
      console.error('❌ Error updating image status to failed:', imageError);
    } else {
      console.log('✅ Image job marked as failed');
    }
  } else if (status === 'processing') {
    console.log('🔄 Processing image job in progress');
    const { error: imageError } = await supabase.from('images').update({
      status: 'generating',
      metadata: {
        ...job.metadata || {},
        processing_started_at: new Date().toISOString()
      }
    }).eq('id', job.image_id);
    
    if (imageError) {
      console.error('❌ Error updating image status to generating:', imageError);
    } else {
      console.log('✅ Image job marked as generating');
    }
  }
}

async function handleVideoJobCallback(supabase, job, status, assets, error_message, quality, isEnhanced) {
  console.log('📹 STANDARDIZED VIDEO CALLBACK PROCESSING:', {
    job_id: job.id,
    videoId: job.video_id,
    status,
    assets,
    assetsCount: assets ? assets.length : 0,
    jobType: job.job_type,
    quality,
    isEnhanced
  });
  
  if (status === 'completed' && job.video_id && assets && assets.length > 0) {
    // Normalize ALL video paths to be user-scoped for consistency
    const normalizedVideoPath = normalizeAssetPath(assets[0], job.user_id);
    
    console.log('📹 Video path handling:', {
      originalPath: assets[0],
      normalizedPath: normalizedVideoPath,
      userId: job.user_id,
      jobType: job.job_type
    });
    
    // Store in both video_url and signed_url fields for consistency
    const updateData = {
      status: 'completed',
      video_url: normalizedVideoPath,
      signed_url: normalizedVideoPath,
      signed_url_expires_at: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
      completed_at: new Date().toISOString(),
      metadata: {
        ...job.metadata || {},
        callback_processed_at: new Date().toISOString(),
        normalized_path: normalizedVideoPath,
        bucket: isEnhanced ? `video7b_${quality}_enhanced` : `video_${quality}`
      }
    };
    
    const { error: videoError } = await supabase.from('videos').update(updateData).eq('id', job.video_id);
    
    if (videoError) {
      console.error('❌ Error updating video:', {
        videoId: job.video_id,
        error: videoError,
        updateData
      });
    } else {
      console.log('✅ Video job updated successfully:', {
        videoId: job.video_id,
        path: normalizedVideoPath,
        bucket: updateData.metadata.bucket
      });
    }
  }
  
  if (status === 'failed' && job.video_id) {
    const { error: videoError } = await supabase.from('videos').update({
      status: 'failed',
      error_message: error_message
    }).eq('id', job.video_id);
    
    if (videoError) {
      console.error('❌ Error updating video status to failed:', videoError);
    } else {
      console.log('✅ Video job marked as failed');
    }
  }
  
  if (status === 'processing' && job.video_id) {
    const { error: videoError } = await supabase.from('videos').update({
      status: 'processing'
    }).eq('id', job.video_id);
    
    if (videoError) {
      console.error('❌ Error updating video status to processing:', videoError);
    } else {
      console.log('✅ Video job marked as processing');
    }
  }
}
```

---

## **Parameter Consistency Summary**

### **Perfect Alignment Achieved**

All components now use **identical parameter conventions**:

| Component | Job Payload | Callback Format | Status |
|-----------|-------------|-----------------|---------|
| **Queue-Job** | `id`, `type`, `prompt`, `user_id` | N/A | ✅ Perfect |
| **Workers** | `id`, `type`, `prompt`, `user_id` | `job_id`, `status`, `assets`, `error_message` | ✅ Perfect |
| **Job-Callback** | N/A | `job_id`, `status`, `assets`, `error_message` | ✅ Perfect |

### **Critical Fixes Applied**

1. **Negative Prompt Handling**: Only SDXL jobs generate negative prompts (WAN 2.1 doesn't support `--negative_prompt`)
2. **Parameter Standardization**: All field names use consistent snake_case format
3. **Queue Routing**: Proper routing based on job type (SDXL → sdxl_queue, WAN → wan_queue)
4. **Enhanced Job Support**: All 10 job types properly parsed and configured
5. **File Path Normalization**: Consistent user-scoped path handling
6. **Error Handling**: Comprehensive validation and error reporting

### **Production Ready**

Both edge functions are **production-ready** and perfectly aligned with the worker conventions. The system should operate seamlessly with standardized parameter handling across all components.

**Status: ✅ All Edge Functions Aligned and Production Ready** 