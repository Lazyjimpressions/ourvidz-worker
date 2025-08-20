# SDXL Worker I2I Contract Analysis

**Date**: August 20, 2025  
**Purpose**: Comprehensive analysis of SDXL worker's I2I detection, parameter handling, and field specifications

## 🎯 **I2I Detection Mechanism**

### **Detection Logic**
The SDXL worker detects I2I mode **purely by the presence of a reference image**, not by pipeline flags:

```python
# In process_job()
reference_image_url = metadata.get('reference_image_url')
reference_image = None
if reference_image_url:
    try:
        reference_image = self.download_image_from_url(reference_image_url)
        # I2I generation triggered
    except Exception as e:
        # Fallback to text-to-image generation
        reference_image = None

# In generate_images_batch()
if reference_image:
    # I2I generation using I2I pipeline
    images, negative_prompt_used = self.generate_with_i2i_pipeline(...)
else:
    # Standard text-to-image generation
    images, used_seed = self.generate_images_batch(...)
```

### **Reference Image Location**
- **✅ Primary**: `metadata.reference_image_url` (string URL)
- **❌ NOT supported**: Top-level `reference_image_url`
- **❌ NOT supported**: `config.image` (base64 or URL)
- **✅ Supported**: Signed Supabase URLs (worker downloads via HTTP)

## 🔧 **Strength/Denoise Parameter Contract**

### **Parameter Reading Order**
1. **Primary**: `metadata.denoise_strength` (0.0-1.0, higher = more change)
2. **Fallback**: `metadata.reference_strength` (0.0-1.0, higher = stronger reference)
3. **Conversion**: `denoise_strength = 1.0 - reference_strength`

### **Directionality**
```python
# Worker reads denoise_strength directly
denoise_strength = metadata.get('denoise_strength')
if denoise_strength is None:
    reference_strength = metadata.get('reference_strength', 0.5)
    denoise_strength = 1.0 - reference_strength  # Convert reference_strength to denoise_strength

# Worker passes denoise_strength to pipeline as 'strength'
generation_kwargs = {
    'strength': denoise_strength,  # Direct mapping
    # ...
}
```

**✅ Confirmed**: `strength` parameter in pipeline = `denoise_strength` from metadata

### **Defaults**
- **If `denoise_strength` missing**: Uses `reference_strength = 0.5` → `denoise_strength = 0.5`
- **If both missing**: `denoise_strength = 0.5` (50% change)

## 🎛️ **Guidance and Steps for I2I**

### **Field Names**
- **Steps**: `config.num_inference_steps` (not `steps`)
- **Guidance**: `config.guidance_scale` (float)

### **Worker-Side Behavior**

#### **Exact Copy Mode** (`exact_copy_mode=true`)
```python
if exact_copy_mode:
    denoise_strength = min(denoise_strength, 0.05)  # Clamp to ≤ 0.05
    guidance_scale = 1.0                            # Fixed at 1.0
    steps = min(max(6, int(denoise_strength * 100)), 10)  # 6-10 steps
    negative_prompt = None                          # Omitted
```

#### **Reference Modify Mode** (`exact_copy_mode=false`)
```python
else:
    # Use parameters as provided by edge function (NO CLAMPING)
    guidance_scale = config['guidance_scale']       # Use as provided
    steps = config['num_inference_steps']           # Use as provided
    negative_prompt = [...]                         # Standard quality prompts
```

### **Fast vs High Settings**
- **No worker-side clamps** in modify mode
- **Uses config values** directly from job configuration
- **Exact copy mode** overrides all settings regardless of fast/high

## 📦 **Batching and Outputs**

### **Batch Size Field**
- **✅ Primary**: `config.num_images` (not `config.batch_size`)
- **Accepted values**: 1, 3, 6 only
- **Validation**: Invalid values default to 1

```python
num_images = config.get('num_images', 1)
if num_images not in [1, 3, 6]:
    logger.warning(f"⚠️ Invalid num_images: {num_images}, defaulting to 1")
    num_images = 1
```

### **Batch Generation**
- **Single generation call** with `num_images_per_prompt = 1`
- **Multiple generators** created for each image
- **Seed handling**: `seed + i` for each image in batch

## 🎯 **Exact Copy Mode Behavior**

### **Mode Detection**
- **Flag**: `metadata.exact_copy_mode` (boolean)
- **Default**: `False`

### **Behavior Overrides**
When `exact_copy_mode=true`:
1. **Denoise strength**: Clamped to ≤ 0.05
2. **Guidance scale**: Fixed at 1.0
3. **Steps**: 6-10 (based on denoise_strength)
4. **Negative prompt**: Omitted (`negative_prompt = None`)
5. **Prompt**: Empty string (`""`)

### **Negative Prompt Handling**
- **Exact copy mode**: `negative_prompt_used = False`
- **Modify mode**: `negative_prompt_used = True`
- **Text-to-image**: Always `negative_prompt_used = True`

## 🌱 **Seed and Determinism**

### **Seed Reading**
- **Field**: `config.seed` (integer)
- **Behavior**: Deterministic if provided, random if missing

```python
seed = config.get('seed')
if seed:
    generators = [torch.Generator(device="cuda").manual_seed(int(seed) + i) for i in range(num_images)]
else:
    random_seed = int(time.time())
    generators = [torch.Generator(device="cuda").manual_seed(random_seed + i) for i in range(num_images)]
    seed = random_seed
```

### **Determinism Factors**
- **Seed**: Primary factor for determinism
- **Denoise strength**: Affects generation but doesn't break determinism
- **Solver**: Uses `unipc` (deterministic)
- **I2I pipeline**: `StableDiffusionXLImg2ImgPipeline` (deterministic with same seed)

## 🔄 **Fallbacks and Compatibility**

### **Parameter Priority**
1. **`metadata.denoise_strength`** (wins if present)
2. **`metadata.reference_strength`** (fallback, converted)

### **Reference Image Priority**
- **Only one location**: `metadata.reference_image_url`
- **No fallback locations** supported
- **No top-level reference_image_url** support

## 📊 **Callback Metadata (for Verification)**

### **Standard Metadata Fields**
```json
{
  "metadata": {
    "width": 1024,
    "height": 1024,
    "format": "png",
    "batch_size": 3,
    "steps": 25,
    "guidance_scale": 7.5,
    "seed": 123456789,
    "file_size_bytes": 2048576,
    "asset_index": 0,
    "negative_prompt_used": true
  }
}
```

### **I2I-Specific Metadata Fields**
```json
{
  "metadata": {
    "denoise_strength": 0.15,
    "pipeline": "img2img",
    "resize_policy": "center_crop",
    "negative_prompt_used": false  // For exact copy mode
  }
}
```

### **Verification Fields Available**
- **`denoise_strength`**: Actual value used (after clamping in exact copy mode)
- **`pipeline`**: Always `"img2img"` for I2I jobs
- **`resize_policy`**: Always `"center_crop"` for I2I jobs
- **`negative_prompt_used`**: Boolean indicating if negative prompt was applied
- **`steps`**: Actual steps used (from config or exact copy mode)
- **`guidance_scale`**: Actual guidance scale used (from config or exact copy mode)

## 📋 **Summary of Field Specifications**

### **Required Fields for I2I**
```json
{
  "id": "job-123",
  "type": "sdxl_image_fast",
  "prompt": "beautiful woman",
  "user_id": "user-123",
  "config": {
    "num_images": 1,
    "num_inference_steps": 25,
    "guidance_scale": 7.5
  },
  "metadata": {
    "reference_image_url": "https://example.com/reference.jpg",
    "denoise_strength": 0.15,
    "exact_copy_mode": false
  }
}
```

### **Field Name Summary**
| **Purpose** | **Field Name** | **Location** | **Type** | **Default** |
|-------------|----------------|--------------|----------|-------------|
| **Reference Image** | `reference_image_url` | `metadata` | string URL | None |
| **Denoise Strength** | `denoise_strength` | `metadata` | float 0.0-1.0 | 0.5 |
| **Exact Copy Mode** | `exact_copy_mode` | `metadata` | boolean | false |
| **Batch Size** | `num_images` | `config` | int (1,3,6) | 1 |
| **Steps** | `num_inference_steps` | `config` | int | job config |
| **Guidance Scale** | `guidance_scale` | `config` | float | job config |
| **Seed** | `seed` | `config` | int | random |

## ✅ **Confirmed Specifications**

### **I2I Detection**
- ✅ **Triggered by**: Presence of `metadata.reference_image_url`
- ✅ **No pipeline flags** used for detection
- ✅ **Signed Supabase URLs** supported

### **Parameter Directionality**
- ✅ **`denoise_strength`**: Higher = more change (0.0 = no change, 1.0 = maximum change)
- ✅ **`strength` parameter**: Direct mapping to `denoise_strength`
- ✅ **Conversion**: `denoise_strength = 1.0 - reference_strength`

### **Exact Copy Mode**
- ✅ **Flag**: `metadata.exact_copy_mode`
- ✅ **Behavior**: Overrides all parameters for safety
- ✅ **Negative prompt**: Omitted in exact copy mode

### **Determinism**
- ✅ **Seed**: `config.seed` for reproducible results
- ✅ **I2I determinism**: Same seed + same denoise_strength = same result

**🎯 All specifications confirmed and ready for edge function implementation.**
