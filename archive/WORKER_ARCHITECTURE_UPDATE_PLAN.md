# Worker Architecture Update Plan

**Date:** August 16, 2025  
**Purpose:** Update workers to pure inference architecture with centralized configuration management

---

## 🎯 **Target Architecture**

### **Core Principle**
**Pure Inference Workers + Centralized Configuration Management**

### **Configuration Hierarchy**
1. **Frontend:** User interface and preset selection
2. **Edge Functions:** Business logic, validation, parameter conversion
3. **Supabase:** Persistent data, user profiles, configurations
4. **Workers:** Pure inference execution

---

## 🏗️ **Updated File Structure**

### **Current Structure (Before)**
```
ourvidz-worker/
├── Core Workers/
│   ├── sdxl_worker.py          # Mixed logic + generation
│   ├── chat_worker.py          # Mixed logic + enhancement
│   └── wan_worker.py           # Mixed logic + generation + enhancement
├── System Management/
│   ├── dual_orchestrator.py    # Job routing
│   ├── memory_manager.py       # VRAM management
│   └── worker_registration.py  # URL registration
├── Infrastructure/
│   ├── startup.sh              # System startup
│   ├── wan_generate.py         # WAN generation utilities
│   └── requirements.txt        # Dependencies
└── Documentation/
    ├── README.md
    ├── WORKER_API.md
    ├── CODEBASE_INDEX.md
    ├── CHAT_WORKER_CONSOLIDATED.md
    ├── CLEANUP_SUMMARY.md
    └── ENHANCEMENT_ARCHITECTURE.md
```

### **Target Structure (After)**
```
ourvidz-worker/
├── Core Workers/
│   ├── sdxl_worker.py          # Pure image generation inference
│   ├── chat_worker.py          # Pure enhancement/chat inference
│   └── wan_worker.py           # Pure video generation inference
├── System Management/
│   ├── dual_orchestrator.py    # Job routing (updated)
│   ├── memory_manager.py       # VRAM management (updated)
│   └── worker_registration.py  # URL registration
├── Infrastructure/
│   ├── startup.sh              # System startup (updated)
│   ├── wan_generate.py         # WAN generation utilities
│   └── requirements.txt        # Dependencies
├── Configuration/
│   ├── worker_configs.py       # Worker configuration templates
│   ├── model_configs.py        # Model-specific configurations
│   └── validation_schemas.py   # Request validation schemas
├── Documentation/
│   ├── README.md               # Updated
    ├── WORKER_API.md           # Updated
    ├── CODEBASE_INDEX.md       # Updated
    ├── SYSTEM_SUMMARY.md       # Updated
    ├── CLEANUP_SUMMARY.md
    └── ENHANCEMENT_ARCHITECTURE.md
└── Archive/
    └── [Historical files]
```

---

## 🔧 **Worker Update Plan**

### **Phase 1: SDXL Worker Refactoring**

#### **Current Issues:**
- Mixed business logic and generation
- Built-in quality assumptions
- Hardcoded configurations
- Limited flexibility

#### **Target State:**
- Pure inference engine
- Receives complete parameters
- No built-in restrictions
- Handles any image type/quality

#### **Changes Required:**

**1. Remove Business Logic**
```python
# REMOVE: Built-in quality presets
# REMOVE: Hardcoded configurations
# REMOVE: Content restrictions
# REMOVE: User validation

# ADD: Pure parameter execution
def generate_image(self, request_params):
    """Pure inference - executes provided parameters"""
    # Extract all parameters from request
    steps = request_params.get('steps', 25)
    guidance_scale = request_params.get('guidance_scale', 7.5)
    batch_size = request_params.get('batch_size', 1)
    resolution = request_params.get('resolution', '1024x1024')
    # ... execute with provided parameters
```

**2. Update API Endpoints**
```python
# Current: Mixed endpoints
@app.route('/generate', methods=['POST'])
def generate_with_business_logic():
    # Business logic here

# Target: Pure inference endpoints
@app.route('/generate', methods=['POST'])
def generate_pure_inference():
    """Pure inference - validates and executes provided parameters"""
    # Validate request format
    # Execute with provided parameters
    # Return results
```

**3. Parameter Validation**
```python
# Add comprehensive parameter validation
def validate_generation_params(self, params):
    """Validate all generation parameters"""
    required_fields = ['prompt', 'steps', 'guidance_scale', 'batch_size']
    # Validate each parameter
    # Return validation result
```

### **Phase 2: Chat Worker Refactoring**

#### **Current Issues:**
- "Pure inference" design but mixed logic
- Built-in safety assumptions
- Limited enhancement flexibility

#### **Target State:**
- Pure inference engine
- NSFW-first, unrestricted design
- Flexible enhancement with both Qwen models
- Receives complete configuration

#### **Changes Required:**

**1. Remove Safety Assumptions**
```python
# REMOVE: Built-in safety filtering
# REMOVE: Content restrictions
# REMOVE: "Pure inference" limitations

# ADD: Unrestricted, NSFW-first design
def enhance_prompt(self, request_params):
    """Unrestricted enhancement with provided configuration"""
    enhancement_type = request_params.get('enhancement_type', 'base')
    system_prompt = request_params.get('system_prompt', '')
    # Execute enhancement with provided configuration
```

**2. Update Enhancement Endpoints**
```python
# Current: Limited enhancement
@app.route('/enhance', methods=['POST'])
def enhance_with_limitations():

# Target: Flexible enhancement
@app.route('/enhance/base', methods=['POST'])
def enhance_with_base():
    """Base model enhancement - creative, unrestricted"""

@app.route('/enhance/instruct', methods=['POST'])
def enhance_with_instruct():
    """Instruct model enhancement - conversation-style, unrestricted"""
```

**3. Model Management**
```python
# Keep both Qwen models
# Model selection based on enhancement needs, not safety
def select_model(self, enhancement_type):
    if enhancement_type == 'creative' or enhancement_type == 'adult':
        return 'base'
    elif enhancement_type == 'conversational':
        return 'instruct'
    else:
        return 'base'  # Default to base for NSFW-first
```

### **Phase 3: WAN Worker Refactoring**

#### **Current Issues:**
- Mixed generation and enhancement logic
- Built-in Qwen Base model
- Complex job type handling

#### **Target State:**
- Pure video generation inference
- Receives enhanced prompts from Chat Worker
- No built-in enhancement logic

#### **Changes Required:**

**1. Remove Enhancement Logic**
```python
# REMOVE: Qwen Base model loading
# REMOVE: Enhancement logic
# REMOVE: Enhanced job types
# REMOVE: Auto prompt function

# ADD: Pure generation
def generate_video(self, request_params):
    """Pure video generation inference"""
    # Extract all parameters
    # Execute WAN generation
    # Return results
```

**2. Simplify Job Types**
```python
# Current: Complex job types with enhancement
job_types = {
    'image_fast': {...},
    'image7b_fast_enhanced': {...},  # REMOVE
    'video_high': {...},
    'video7b_high_enhanced': {...}   # REMOVE
}

# Target: Simple generation types
job_types = {
    'image_fast': {...},
    'image_high': {...},
    'video_fast': {...},
    'video_high': {...}
}
```

**3. Update Memory Management**
```python
# Remove Qwen model from memory allocation
# Only WAN model needed
def get_memory_requirements(self):
    return {
        'wan_model': '30GB',
        'total': '30GB'
    }
```

### **Phase 4: Orchestrator Updates**

#### **Changes Required:**

**1. Update Worker Configurations**
```python
# Current: Mixed responsibilities
workers = {
    'sdxl': {'job_types': ['sdxl_image_fast', 'sdxl_image_high']},
    'chat': {'job_types': ['chat_enhance', 'chat_conversation']},
    'wan': {'job_types': ['image_fast', 'image7b_fast_enhanced', ...]}
}

# Target: Pure inference roles
workers = {
    'sdxl': {'role': 'pure_image_generation', 'job_types': ['sdxl_generate']},
    'chat': {'role': 'pure_enhancement', 'job_types': ['enhance_base', 'enhance_instruct', 'chat']},
    'wan': {'role': 'pure_video_generation', 'job_types': ['wan_generate']}
}
```

**2. Update Job Routing**
```python
# Current: Direct routing
def route_job(job_type):
    if 'sdxl' in job_type:
        return 'sdxl_worker'
    elif 'wan' in job_type:
        return 'wan_worker'

# Target: Enhancement → Generation flow
def route_job(job_request):
    if job_request.needs_enhancement:
        # Route to Chat Worker for enhancement
        enhanced_prompt = chat_worker.enhance(job_request)
        job_request.prompt = enhanced_prompt
    
    # Route to target generation worker
    if job_request.target_worker == 'sdxl':
        return sdxl_worker.generate(job_request)
    elif job_request.target_worker == 'wan':
        return wan_worker.generate(job_request)
```

### **Phase 5: Configuration Management**

#### **New Files to Create:**

**1. worker_configs.py**
```python
"""Worker configuration templates and validation"""

class WorkerConfig:
    """Base configuration for all workers"""
    
    @staticmethod
    def get_sdxl_config():
        """SDXL worker configuration template"""
        return {
            'model_path': '/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors',
            'supported_resolutions': ['512x512', '1024x1024', 'custom'],
            'supported_batch_sizes': [1, 3, 6],
            'step_range': [10, 50],
            'guidance_scale_range': [1.0, 20.0]
        }
    
    @staticmethod
    def get_wan_config():
        """WAN worker configuration template"""
        return {
            'model_path': '/workspace/models/wan2.1-t2v-1.3b',
            'supported_resolutions': ['480x832', 'custom'],
            'frame_range': [1, 83],
            'reference_modes': ['none', 'single', 'start', 'end', 'both']
        }
    
    @staticmethod
    def get_chat_config():
        """Chat worker configuration template"""
        return {
            'base_model_path': '/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B',
            'instruct_model_path': '/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct',
            'enhancement_types': ['base', 'instruct'],
            'max_tokens': 2048
        }
```

**2. validation_schemas.py**
```python
"""Request validation schemas for all workers"""

class ValidationSchemas:
    """Validation schemas for worker requests"""
    
    @staticmethod
    def sdxl_generation_schema():
        """SDXL generation request validation"""
        return {
            'required': ['prompt', 'steps', 'guidance_scale', 'batch_size', 'resolution'],
            'optional': ['negative_prompt', 'seed', 'reference_image'],
            'validation_rules': {
                'steps': {'min': 10, 'max': 50},
                'guidance_scale': {'min': 1.0, 'max': 20.0},
                'batch_size': {'allowed': [1, 3, 6]}
            }
        }
    
    @staticmethod
    def wan_generation_schema():
        """WAN generation request validation"""
        return {
            'required': ['prompt', 'job_type', 'frames', 'resolution'],
            'optional': ['reference_mode', 'reference_image', 'seed'],
            'validation_rules': {
                'frames': {'min': 1, 'max': 83},
                'reference_mode': {'allowed': ['none', 'single', 'start', 'end', 'both']}
            }
        }
    
    @staticmethod
    def chat_enhancement_schema():
        """Chat enhancement request validation"""
        return {
            'required': ['prompt', 'enhancement_type', 'target_model'],
            'optional': ['system_prompt', 'quality', 'nsfw_optimization'],
            'validation_rules': {
                'enhancement_type': {'allowed': ['base', 'instruct']},
                'target_model': {'allowed': ['sdxl', 'wan']}
            }
        }
```

---

## 📋 **Implementation Checklist**

### **Phase 1: SDXL Worker**
- [ ] Remove business logic and hardcoded configurations
- [ ] Implement pure inference endpoints
- [ ] Add comprehensive parameter validation
- [ ] Update API documentation
- [ ] Test with various parameter combinations

### **Phase 2: Chat Worker**
- [ ] Remove safety assumptions and restrictions
- [ ] Implement NSFW-first, unrestricted design
- [ ] Add flexible enhancement endpoints (Base/Instruct)
- [ ] Update model selection logic
- [ ] Test enhancement with both models

### **Phase 3: WAN Worker**
- [ ] Remove Qwen Base model and enhancement logic
- [ ] Simplify job types (remove enhanced variants)
- [ ] Implement pure video generation
- [ ] Update memory management
- [ ] Test video generation with enhanced prompts

### **Phase 4: Orchestrator**
- [ ] Update worker configurations
- [ ] Implement enhancement → generation flow
- [ ] Update job routing logic
- [ ] Update memory allocation
- [ ] Test end-to-end workflows

### **Phase 5: Configuration Management**
- [ ] Create worker_configs.py
- [ ] Create validation_schemas.py
- [ ] Update startup.sh with new configurations
- [ ] Test configuration validation
- [ ] Update documentation

### **Phase 6: Documentation**
- [ ] Update SYSTEM_SUMMARY.md
- [ ] Update WORKER_API.md
- [ ] Update CODEBASE_INDEX.md
- [ ] Update README.md
- [ ] Create migration guide

---

## 🎯 **Expected Outcomes**

### **Performance Improvements**
- **Faster Execution:** Pure inference without business logic overhead
- **Better Resource Usage:** Optimized memory allocation
- **Reduced Latency:** Streamlined request processing

### **Architectural Benefits**
- **Clear Separation:** Business logic vs inference execution
- **Better Scalability:** Workers can be easily replicated
- **Improved Maintainability:** Centralized configuration management
- **Enhanced Flexibility:** Any type of generation possible

### **User Experience**
- **Full Customization:** Complete parameter control
- **Consistent Quality:** Unified enhancement through Chat Worker
- **Better Reliability:** Simplified error handling
- **Enhanced Features:** More generation options

---

**🎯 This plan transforms the workers into pure inference engines with centralized configuration management, providing maximum flexibility and scalability.**
