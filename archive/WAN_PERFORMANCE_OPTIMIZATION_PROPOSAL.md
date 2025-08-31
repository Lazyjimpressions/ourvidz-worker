# WAN Video Generation Performance Optimization Proposal

**Date**: August 20, 2025  
**Issue**: Video generation too slow (315.3s for 5.2s video)  
**Goal**: Reduce generation time by 40-60% while maintaining quality

## 📊 **Current Performance Analysis**

### **Observed Performance**
- **Generation Time**: 315.3s (5.25 minutes) for 5.2s video
- **Expected Time**: 135s (2.25 minutes) - **2.3x slower than expected**
- **Generation Rate**: 0.26 fps (83 frames ÷ 315.3s)
- **Quality**: High quality but too slow for user experience

### **Current Settings**
```python
'video_fast': {
    'task': 't2v-1.3B',
    'size': '480*832',
    'sample_steps': 25,        # ⚠️ Too many steps
    'sample_guide_scale': 6.5,
    'sample_solver': 'unipc',  # ⚠️ Slower solver
    'sample_shift': 5.0,       # ⚠️ High shift value
    'frame_num': 83,           # ⚠️ Many frames
    'expected_time': 135,      # ⚠️ Unrealistic expectation
}
```

## 🚀 **Optimization Strategies**

### **Strategy 1: Reduce Inference Steps (Primary)**

**Current**: 25 steps  
**Proposed**: 15-18 steps  
**Impact**: 40% faster generation

```python
'video_fast': {
    'sample_steps': 15,        # Reduced from 25
    'expected_time': 90,       # Reduced from 135
}
```

**Quality Impact**: Minimal - WAN 2.1 can produce good results with 15-18 steps

### **Strategy 2: Optimize Solver Settings**

**Current**: `unipc` solver  
**Proposed**: `euler` or `dpm++`  
**Impact**: 20-30% faster generation

```python
'video_fast': {
    'sample_solver': 'euler',  # Faster than unipc
    'sample_shift': 3.0,       # Reduced from 5.0
}
```

### **Strategy 3: Reduce Frame Count (Secondary)**

**Current**: 83 frames (5.2s)  
**Proposed**: 64 frames (4s)  
**Impact**: 25% faster generation

```python
'video_fast': {
    'frame_num': 64,           # Reduced from 83
    'expected_time': 75,       # Further reduced
}
```

### **Strategy 4: Resolution Optimization**

**Current**: `480*832`  
**Proposed**: `384*640`  
**Impact**: 35% faster generation

```python
'video_fast': {
    'size': '384*640',         # Reduced from 480*832
}
```

## 📋 **Proposed Optimized Settings**

### **Option A: Balanced Optimization (Recommended)**

```python
'video_fast': {
    'task': 't2v-1.3B',
    'size': '480*832',         # Keep current resolution
    'sample_steps': 18,        # Reduced from 25
    'sample_guide_scale': 6.5, # Keep current
    'sample_solver': 'euler',  # Faster solver
    'sample_shift': 3.0,       # Reduced from 5.0
    'frame_num': 64,           # Reduced from 83 (4s video)
    'enhance_prompt': False,
    'expected_time': 75,       # Reduced from 135
    'content_type': 'video',
    'file_extension': 'mp4'
}
```

**Expected Performance**: ~120-150s (2-2.5 minutes) - **50% faster**

### **Option B: Aggressive Optimization**

```python
'video_fast': {
    'task': 't2v-1.3B',
    'size': '384*640',         # Smaller resolution
    'sample_steps': 15,        # Fewer steps
    'sample_guide_scale': 6.0, # Slightly reduced
    'sample_solver': 'euler',  # Faster solver
    'sample_shift': 2.5,       # Lower shift
    'frame_num': 48,           # Fewer frames (3s video)
    'enhance_prompt': False,
    'expected_time': 60,       # Much faster
    'content_type': 'video',
    'file_extension': 'mp4'
}
```

**Expected Performance**: ~90-120s (1.5-2 minutes) - **60% faster**

### **Option C: Quality-First Optimization**

```python
'video_fast': {
    'task': 't2v-1.3B',
    'size': '480*832',         # Keep resolution
    'sample_steps': 20,        # Moderate reduction
    'sample_guide_scale': 6.5, # Keep quality
    'sample_solver': 'dpm++',  # Better than euler
    'sample_shift': 4.0,       # Moderate reduction
    'frame_num': 83,           # Keep frame count
    'enhance_prompt': False,
    'expected_time': 100,      # Moderate improvement
    'content_type': 'video',
    'file_extension': 'mp4'
}
```

**Expected Performance**: ~150-180s (2.5-3 minutes) - **40% faster**

## 🎯 **Recommendation: Option A (Balanced)**

### **Why Option A is Recommended**

1. **Balanced Approach**: Good speed improvement without quality loss
2. **User Experience**: 2-2.5 minutes is acceptable for video generation
3. **Quality Preservation**: Maintains good visual quality
4. **Frame Count**: 4-second videos are still engaging
5. **Resolution**: Keeps current quality level

### **Expected Results**
- **Generation Time**: 120-150s (vs current 315.3s)
- **Speed Improvement**: 50% faster
- **Quality**: Maintained at good level
- **User Experience**: Much more responsive

## 🔧 **Implementation Plan**

### **Phase 1: Quick Wins (Immediate)**
1. Reduce `sample_steps` from 25 to 18
2. Change solver from `unipc` to `euler`
3. Reduce `sample_shift` from 5.0 to 3.0
4. Update expected times

### **Phase 2: Frame Count Optimization (Next)**
1. Reduce `frame_num` from 83 to 64
2. Test quality impact
3. Adjust based on user feedback

### **Phase 3: Resolution Optimization (Future)**
1. Test smaller resolutions
2. Evaluate quality vs speed trade-offs
3. Implement if needed

## 📊 **Quality vs Speed Trade-offs**

| **Setting** | **Speed Impact** | **Quality Impact** | **Recommendation** |
|-------------|------------------|-------------------|-------------------|
| **Steps (25→18)** | +40% faster | Minimal | ✅ **Implement** |
| **Solver (unipc→euler)** | +25% faster | Minimal | ✅ **Implement** |
| **Shift (5.0→3.0)** | +15% faster | Minimal | ✅ **Implement** |
| **Frames (83→64)** | +25% faster | Moderate | ⚠️ **Test first** |
| **Resolution (480→384)** | +35% faster | Noticeable | ❌ **Avoid for now** |

## 🚀 **Expected Performance Improvement**

### **Current Performance**
- **Generation Time**: 315.3s (5.25 minutes)
- **User Experience**: Poor (too slow)

### **Optimized Performance (Option A)**
- **Generation Time**: 120-150s (2-2.5 minutes)
- **Speed Improvement**: 50% faster
- **User Experience**: Good (acceptable wait time)

### **Aggressive Performance (Option B)**
- **Generation Time**: 90-120s (1.5-2 minutes)
- **Speed Improvement**: 60% faster
- **User Experience**: Excellent (fast response)

## ✅ **Next Steps**

1. **Implement Option A** settings in `wan_worker.py`
2. **Test with real prompts** to validate performance
3. **Monitor quality** to ensure no significant degradation
4. **Gather user feedback** on new generation times
5. **Consider Option B** if users want even faster generation

**🎯 Goal**: Reduce video generation time from 5+ minutes to 2-2.5 minutes while maintaining quality.
