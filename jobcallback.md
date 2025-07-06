import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type'
};

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      headers: corsHeaders
    });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    const requestBody = await req.json();
    
    // FIXED: Support both parameter naming conventions for compatibility
    const {
      jobId,           // Frontend/legacy format
      job_id,          // Worker format
      status,
      filePath,
      outputUrl,
      errorMessage,
      enhancedPrompt,
      imageUrls,
      assets           // New assets array format
    } = requestBody;

    // CRITICAL FIX: Use job_id if available, otherwise use jobId
    const resolvedJobId = job_id || jobId;
    const resolvedFilePath = filePath || outputUrl;
    const resolvedAssets = assets || imageUrls;

    console.log('🔍 FIXED CALLBACK DEBUGGING - Received request:', {
      jobId,
      job_id,
      resolvedJobId,
      status,
      filePath,
      outputUrl,
      resolvedFilePath,
      imageUrls,
      assets,
      resolvedAssets,
      errorMessage,
      enhancedPrompt,
      fullRequestBody: requestBody,
      timestamp: new Date().toISOString()
    });

    // Validate critical parameters
    if (!resolvedJobId) {
      console.error('❌ CRITICAL: No job ID provided in callback', {
        jobId,
        job_id,
        resolvedJobId,
        requestKeys: Object.keys(requestBody)
      });
      throw new Error('Job ID is required (job_id or jobId)');
    }

    if (!resolvedFilePath && !resolvedAssets && status === 'completed') {
      console.error('❌ CRITICAL: No file path or assets provided for completed job', {
        resolvedJobId,
        status,
        hasFilePath: !!filePath,
        hasOutputUrl: !!outputUrl,
        hasImageUrls: !!imageUrls,
        hasAssets: !!assets,
        resolvedFilePath,
        resolvedAssets
      });
    }

    // Get current job to preserve existing metadata and check format
    console.log('🔍 Fetching job details for:', resolvedJobId);
    const { data: currentJob, error: fetchError } = await supabase
      .from('jobs')
      .select('metadata, job_type, image_id, video_id, format, quality, model_type, user_id')
      .eq('id', resolvedJobId)
      .single();

    if (fetchError) {
      console.error('❌ CRITICAL: Error fetching current job:', {
        jobId: resolvedJobId,
        error: fetchError,
        errorMessage: fetchError.message,
        errorCode: fetchError.code
      });
      throw fetchError;
    }

    console.log('✅ Job details fetched successfully:', {
      jobId: currentJob.id,
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
      error_message: errorMessage || null
    };

    // Merge metadata instead of overwriting
    let updatedMetadata = currentJob.metadata || {};

    // Handle enhanced prompt for image_high jobs (enhancement)
    if (currentJob.job_type === 'image_high' && enhancedPrompt) {
      updatedMetadata.enhanced_prompt = enhancedPrompt;
      console.log('📝 Storing enhanced prompt for image_high job:', enhancedPrompt);
    }

    // FIXED: Handle both single file path and multiple assets
    if (status === 'completed') {
      if (resolvedAssets && Array.isArray(resolvedAssets) && resolvedAssets.length > 0) {
        // Multiple assets (SDXL batch generation)
        console.log('📁 Processing completed job with multiple assets:', {
          jobId: resolvedJobId,
          assetCount: resolvedAssets.length,
          assets: resolvedAssets
        });
        
        updatedMetadata.assets = resolvedAssets;
        updatedMetadata.primary_asset = resolvedAssets[0]; // First asset as primary
        updatedMetadata.asset_count = resolvedAssets.length;
        
      } else if (resolvedFilePath) {
        // Single asset (WAN generation)
        console.log('📁 Processing completed job with single file path:', {
          jobId: resolvedJobId,
          originalFilePath: filePath,
          originalOutputUrl: outputUrl,
          resolvedFilePath: resolvedFilePath
        });
        
        updatedMetadata.file_path = resolvedFilePath;
        updatedMetadata.assets = [resolvedFilePath]; // Normalize to array
        updatedMetadata.primary_asset = resolvedFilePath;
        updatedMetadata.asset_count = 1;
      }
      
      updatedMetadata.callback_processed_at = new Date().toISOString();
      updatedMetadata.callback_debug = {
        received_file_path: filePath,
        received_output_url: outputUrl,
        received_assets: resolvedAssets,
        resolved_file_path: resolvedFilePath,
        job_type: currentJob.job_type,
        processing_timestamp: new Date().toISOString()
      };
      
    } else if (status === 'completed' && !resolvedFilePath && !resolvedAssets) {
      console.error('❌ CRITICAL: Completed job has no file path or assets!', {
        jobId: resolvedJobId,
        status,
        filePath,
        outputUrl,
        imageUrls,
        assets,
        resolvedFilePath,
        resolvedAssets,
        jobType: currentJob.job_type
      });
      
      updatedMetadata.callback_error = {
        issue: 'completed_without_assets',
        timestamp: new Date().toISOString(),
        received_status: status,
        received_file_path: filePath,
        received_output_url: outputUrl,
        received_assets: resolvedAssets,
        resolved_file_path: resolvedFilePath
      };
    }

    updateData.metadata = updatedMetadata;

    console.log('🔄 Updating job with enhanced metadata:', {
      jobId: resolvedJobId,
      updateData,
      metadataKeys: Object.keys(updatedMetadata)
    });

    // Update job status
    const { data: job, error: updateError } = await supabase
      .from('jobs')
      .update(updateData)
      .eq('id', resolvedJobId)
      .select()
      .single();

    if (updateError) {
      console.error('❌ CRITICAL: Error updating job:', {
        jobId: resolvedJobId,
        error: updateError,
        updateData
      });
      throw updateError;
    }

    console.log('✅ Job updated successfully with enhanced debugging:', {
      jobId: job.id,
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

    // Handle different job types based on parsed format
    if (format === 'image' && job.image_id) {
      console.log('🖼️ Processing image job callback...');
      await handleImageJobCallback(supabase, job, status, resolvedFilePath, errorMessage, quality, isSDXL, resolvedAssets, isEnhanced);
    } else if (format === 'video' && job.video_id) {
      console.log('📹 Processing video job callback...');
      await handleVideoJobCallback(supabase, job, status, resolvedFilePath, errorMessage, quality, isEnhanced);
    } else {
      console.error('❌ CRITICAL: Unknown job format or missing ID:', {
        format,
        imageId: job.image_id,
        videoId: job.video_id,
        jobType: job.job_type
      });
    }

    console.log('✅ CALLBACK PROCESSING COMPLETE:', {
      jobId: resolvedJobId,
      status,
      format,
      quality,
      isSDXL,
      isEnhanced,
      filePath,
      outputUrl,
      assets: resolvedAssets,
      resolvedFilePath,
      processingTimestamp: new Date().toISOString()
    });

    return new Response(JSON.stringify({
      success: true,
      message: 'Job callback processed successfully with enhanced debugging',
      debug: {
        jobId: resolvedJobId,
        jobStatus: status,
        jobType: job.job_type,
        format: format,
        quality: quality,
        isSDXL: isSDXL,
        isEnhanced: isEnhanced,
        filePath: resolvedFilePath,
        assets: resolvedAssets,
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

async function handleImageJobCallback(supabase, job, status, filePath, errorMessage, quality, isSDXL, imageAssets, isEnhanced) {
  console.log('🖼️ ENHANCED IMAGE CALLBACK DEBUGGING:', {
    jobId: job.id,
    imageId: job.image_id,
    status,
    filePath,
    imageAssets,
    imageAssetsCount: imageAssets ? imageAssets.length : 0,
    jobType: job.job_type,
    quality,
    isSDXL,
    isEnhanced,
    expectedBucket: isSDXL ? `sdxl_image_${quality}` : isEnhanced ? `image7b_${quality}_enhanced` : `image_${quality}`
  });

  if (status === 'completed' && (filePath || imageAssets)) {
    console.log('✅ Processing completed image job with file path or image assets');
    
    // Normalize paths to ensure user-scoped consistency
    let primaryImageUrl = null;
    let imageUrlsArray = null;

    if (imageAssets && Array.isArray(imageAssets) && imageAssets.length > 0) {
      console.log('🖼️ Multiple images received:', imageAssets.length);
      // Normalize all image URLs in the array
      imageUrlsArray = imageAssets.map(url => normalizeAssetPath(url, job.user_id));
      primaryImageUrl = imageUrlsArray[0]; // Use first image as primary
    } else if (filePath) {
      console.log('🖼️ Single image received:', filePath);
      primaryImageUrl = normalizeAssetPath(filePath, job.user_id);
      imageUrlsArray = [primaryImageUrl]; // Normalize to array format
    }

    // Validate file path format
    const filePathValidation = {
      hasSlash: primaryImageUrl ? primaryImageUrl.includes('/') : false,
      hasUnderscore: primaryImageUrl ? primaryImageUrl.includes('_') : false,
      hasPngExtension: primaryImageUrl ? primaryImageUrl.endsWith('.png') : false,
      length: primaryImageUrl ? primaryImageUrl.length : 0,
      startsWithUserId: primaryImageUrl ? primaryImageUrl.startsWith(job.user_id || 'unknown') : false,
      expectedPattern: `${job.user_id}/${isSDXL ? 'sdxl_' : ''}${job.id}_*.png`,
      isMultipleImages: !!imageUrlsArray && imageUrlsArray.length > 1,
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
        is_enhanced: isEnhanced,
        bucket: isSDXL ? `sdxl_image_${quality}` : isEnhanced ? `image7b_${quality}_enhanced` : `image_${quality}`,
        callback_processed_at: new Date().toISOString(),
        file_path_validation: filePathValidation,
        debug_info: {
          original_file_path: filePath,
          image_assets_received: imageAssets,
          image_urls_array: imageUrlsArray,
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

    const { data: updatedImage, error: imageError } = await supabase
      .from('images')
      .update(updateData)
      .eq('id', job.image_id)
      .select()
      .single();

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
        imageUrls: updatedImage.image_urls,
        quality: updatedImage.quality,
        bucket: updatedImage.metadata?.bucket,
        isSDXL: updatedImage.metadata?.is_sdxl,
        isEnhanced: updatedImage.metadata?.is_enhanced
      });

      // Verify the image can be found by AssetService logic
      console.log('🔍 Verifying image accessibility for AssetService:', {
        imageId: updatedImage.id,
        userId: job.user_id,
        imageUrl: updatedImage.image_url,
        imageUrls: updatedImage.image_urls,
        status: updatedImage.status,
        quality: updatedImage.quality,
        metadata: updatedImage.metadata
      });
    }
  } else if (status === 'failed') {
    console.log('❌ Processing failed image job');
    const { error: imageError } = await supabase
      .from('images')
      .update({
        status: 'failed',
        metadata: {
          ...job.metadata || {},
          error_message: errorMessage,
          failed_at: new Date().toISOString()
        }
      })
      .eq('id', job.image_id);

    if (imageError) {
      console.error('❌ Error updating image status to failed:', imageError);
    } else {
      console.log('✅ Image job marked as failed');
    }
  } else if (status === 'processing') {
    console.log('🔄 Processing image job in progress');
    const { error: imageError } = await supabase
      .from('images')
      .update({
        status: 'generating',
        metadata: {
          ...job.metadata || {},
          processing_started_at: new Date().toISOString()
        }
      })
      .eq('id', job.image_id);

    if (imageError) {
      console.error('❌ Error updating image status to generating:', imageError);
    } else {
      console.log('✅ Image job marked as generating');
    }
  }
}

async function handleVideoJobCallback(supabase, job, status, filePath, errorMessage, quality, isEnhanced) {
  console.log('📹 ENHANCED VIDEO CALLBACK DEBUGGING:', {
    jobId: job.id,
    videoId: job.video_id,
    status,
    filePath,
    jobType: job.job_type,
    quality,
    isEnhanced
  });

  if (status === 'completed' && job.video_id && filePath) {
    // Only normalize SDXL video paths, WAN workers upload to bucket root with just filename
    const isSDXLVideo = job.job_type.startsWith('sdxl_');
    const finalVideoPath = isSDXLVideo ? normalizeAssetPath(filePath, job.user_id) : filePath;

    console.log('📹 Video path handling:', {
      originalPath: filePath,
      finalPath: finalVideoPath,
      userId: job.user_id,
      isSDXL: isSDXLVideo,
      jobType: job.job_type
    });

    const { error: videoError } = await supabase
      .from('videos')
      .update({
        status: 'completed',
        video_url: finalVideoPath,
        completed_at: new Date().toISOString(),
        metadata: {
          ...job.metadata || {},
          is_enhanced: isEnhanced,
          bucket: isEnhanced ? `video7b_${quality}_enhanced` : `video_${quality}`,
          callback_processed_at: new Date().toISOString()
        }
      })
      .eq('id', job.video_id);

    if (videoError) {
      console.error('❌ Error updating video:', videoError);
    } else {
      console.log('✅ Video job updated successfully:', {
        path: finalVideoPath,
        isSDXL: isSDXLVideo,
        isEnhanced: isEnhanced
      });
    }
  }

  if (status === 'failed' && job.video_id) {
    const { error: videoError } = await supabase
      .from('videos')
      .update({
        status: 'failed',
        error_message: errorMessage,
        metadata: {
          ...job.metadata || {},
          failed_at: new Date().toISOString()
        }
      })
      .eq('id', job.video_id);

    if (videoError) {
      console.error('❌ Error updating video status to failed:', videoError);
    } else {
      console.log('✅ Video job marked as failed');
    }
  }

  if (status === 'processing' && job.video_id) {
    const { error: videoError } = await supabase
      .from('videos')
      .update({
        status: 'processing',
        metadata: {
          ...job.metadata || {},
          processing_started_at: new Date().toISOString()
        }
      })
      .eq('id', job.video_id);

    if (videoError) {
      console.error('❌ Error updating video status to processing:', videoError);
    } else {
      console.log('✅ Video job marked as processing');
    }
  }
}