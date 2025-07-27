# ChatWorker Enhancement System

## Overview

The `ChatWorkerEnhancement` class has been integrated into `chat_worker.py` to provide a comprehensive fallback and performance optimization layer for prompt enhancement. This system enables intelligent prompting between SDXL and WAN models with edge function integration.

## Architecture

### Core Components

1. **ChatWorkerEnhancement Class**
   - Fallback system prompts for when edge functions are unavailable
   - Performance optimization with caching
   - Worker-level post-processing and quality validation
   - Emergency fallback mechanisms

2. **Integration with ChatWorker**
   - Seamless integration with existing model infrastructure
   - Backward compatibility with legacy enhancement methods
   - Multiple endpoint options for different use cases

## Features

### 🧠 Edge Function Integration
- Accepts intelligent system prompts from edge functions
- Falls back to worker-built prompts when edge functions are unavailable
- Maintains context and enhancement source tracking

### 🔄 Fallback System
- **SDXL Fast**: 75-token optimization with quality tags and anatomical accuracy
- **SDXL High**: 100-120 token advanced quality with studio lighting
- **WAN Fast**: 175-token video prompts with smooth motion and cinematography
- **WAN High**: 250-token cinematic quality with complex motion

### ⚡ Performance Optimization
- **Caching**: Stores recent enhancements to avoid redundant processing
- **Token Compression**: Intelligent compression for SDXL while preserving key elements
- **Quality Validation**: Scoring system for enhancement quality
- **Memory Management**: OOM handling with retry logic

### 🔧 Post-Processing
- **SDXL Optimization**: Validates quality tags, lighting, technical terms, and resolution
- **WAN Optimization**: Validates motion, cinematography, temporal consistency, and quality
- **Token Counting**: Ensures optimal token counts for each model type

## API Endpoints

### Primary Enhancement Endpoints

#### `/enhance` (POST)
- **Default intelligent enhancement endpoint**
- Uses `ChatWorkerEnhancement.enhance_prompt_intelligent()`
- Supports edge function integration and fallback

#### `/enhance/intelligent` (POST)
- **Explicit intelligent enhancement endpoint**
- Same functionality as `/enhance` but with clearer naming

#### `/enhance/legacy` (POST)
- **Backward compatibility endpoint**
- Uses original `enhance_prompt()` method
- Maintains compatibility with existing integrations

### Management Endpoints

#### `/enhancement/info` (GET)
- Returns enhancement system information
- Lists supported job types and features
- Shows cache size and model status

#### `/enhancement/cache/clear` (POST)
- Clears enhancement cache
- Returns cache statistics

## Request Format

### Standard Request
```json
{
  "prompt": "your original prompt",
  "job_type": "sdxl_image_fast",
  "quality": "fast",
  "enhancement_type": "manual"
}
```

### Edge Function Integration
```json
{
  "prompt": "your original prompt",
  "job_type": "sdxl_image_fast",
  "quality": "fast",
  "system_prompt": "edge function provided system prompt",
  "context": {
    "additional_context": "from edge function"
  }
}
```

## Response Format

### Successful Enhancement
```json
{
  "success": true,
  "original_prompt": "original prompt",
  "enhanced_prompt": "enhanced prompt",
  "generation_time": 1.23,
  "enhancement_type": "manual",
  "job_type": "sdxl_image_fast",
  "quality": "fast",
  "enhancement_source": "edge_function|worker_fallback|emergency_fallback",
  "worker_optimizations": {
    "caching": true,
    "post_processing": true,
    "fallback_ready": true
  },
  "quality_score": 0.85,
  "sdxl_optimizations": {
    "has_quality_tags": true,
    "has_lighting": true,
    "has_technical_terms": true,
    "has_resolution": true,
    "token_count": 75
  }
}
```

### Cached Response
```json
{
  "success": true,
  "original_prompt": "original prompt",
  "enhanced_prompt": "enhanced prompt",
  "cache_hit": true,
  "enhancement_source": "worker_fallback",
  "worker_optimizations": {
    "caching": true,
    "post_processing": true,
    "fallback_ready": true
  }
}
```

## Supported Job Types

### SDXL LUSTIFY
- `sdxl_image_fast`: 75-token optimal, quality tags, anatomical accuracy
- `sdxl_image_high`: 100-120 token, advanced quality, studio lighting

### WAN 2.1 Video
- `video_fast`: 175-token, smooth motion, temporal consistency
- `video_high`: 250-token, cinematic quality, complex motion
- `wan_7b_enhanced`: Enhanced mode with 7B model capabilities

## Quality Levels

- `fast`: Optimized for speed and efficiency
- `high`: Maximum quality with extended token limits

## Fallback Hierarchy

1. **Edge Function**: Use provided system prompt if available
2. **Worker Fallback**: Use built-in prompts based on job type and quality
3. **Emergency Fallback**: Basic enhancement with minimal processing

## Performance Features

### Caching Strategy
- Cache key: `{original_prompt}_{job_type}_{quality}`
- Automatic cleanup when cache exceeds 100 entries
- Removes oldest 20 entries during cleanup

### Token Compression
- Priority preservation: quality tags > subject > lighting > technical > style
- Intelligent word selection for SDXL optimization
- Maintains key elements while reducing token count

### Quality Validation
- Scoring system based on model-specific quality indicators
- Normalized scores (0-1) for easy comparison
- Model-specific optimization validation

## Error Handling

### Graceful Degradation
- OOM handling with retry logic
- Emergency fallback for complete failures
- Comprehensive error logging and reporting

### Error Response Format
```json
{
  "success": false,
  "error": "error description",
  "enhanced_prompt": "original prompt (as fallback)"
}
```

## Usage Examples

### Basic Enhancement
```bash
curl -X POST http://localhost:7861/enhance \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "beautiful woman",
    "job_type": "sdxl_image_fast",
    "quality": "fast"
  }'
```

### Edge Function Integration
```bash
curl -X POST http://localhost:7861/enhance \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "beautiful woman",
    "job_type": "sdxl_image_fast",
    "quality": "fast",
    "system_prompt": "Custom edge function system prompt",
    "context": {"user_preferences": "high_detail"}
  }'
```

### Cache Management
```bash
# Get enhancement info
curl http://localhost:7861/enhancement/info

# Clear cache
curl -X POST http://localhost:7861/enhancement/cache/clear
```

## Integration Notes

### Backward Compatibility
- Original `/enhance` endpoint now uses intelligent enhancement
- Legacy endpoint available at `/enhance/legacy`
- Existing integrations continue to work

### Model Requirements
- Requires Qwen 2.5-7B Instruct model
- Automatic model loading and memory management
- OOM handling with cleanup and retry

### Performance Considerations
- Caching reduces redundant processing
- Token compression optimizes for model efficiency
- Quality validation ensures consistent output

## Future Enhancements

- Dynamic prompt template selection
- A/B testing for enhancement strategies
- Advanced caching with TTL
- Real-time quality metrics
- Integration with external quality assessment services 