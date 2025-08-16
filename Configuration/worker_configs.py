"""
Worker Configuration Templates
Pure Inference Architecture - August 16, 2025

This module provides configuration templates for all workers in the pure inference setup.
All business logic and configuration management is handled by edge functions and frontend.
Workers are pure inference engines that execute provided parameters.
"""

import os
from typing import Dict, List, Any, Optional

class WorkerConfig:
    """Base configuration for all workers in pure inference architecture"""
    
    @staticmethod
    def get_sdxl_config() -> Dict[str, Any]:
        """SDXL worker configuration template"""
        return {
            'model_path': '/workspace/models/sdxl-lustify/lustifySDXLNSFWSFW_v20.safetensors',
            'model_name': 'LUSTIFY SDXL v2.0',
            'port': 7860,
            'supported_resolutions': ['512x512', '1024x1024', 'custom'],
            'supported_batch_sizes': [1, 3, 6],
            'step_range': {'min': 10, 'max': 50},
            'guidance_scale_range': {'min': 1.0, 'max': 20.0},
            'seed_range': {'min': 0, 'max': 2147483647},
            'memory_requirements': '10GB',
            'role': 'pure_image_generation',
            'capabilities': [
                'batch_processing',
                'reference_images',
                'custom_resolutions',
                'negative_prompts',
                'seed_control'
            ]
        }
    
    @staticmethod
    def get_wan_config() -> Dict[str, Any]:
        """WAN worker configuration template"""
        return {
            'model_path': '/workspace/models/wan2.1-t2v-1.3b',
            'model_name': 'WAN 2.1 T2V 1.3B',
            'port': 7860,  # Shared with SDXL
            'supported_resolutions': ['480x832', 'custom'],
            'frame_range': {'min': 1, 'max': 83},
            'fps_range': {'min': 8, 'max': 24},
            'reference_modes': ['none', 'single', 'start', 'end', 'both'],
            'memory_requirements': '30GB',
            'role': 'pure_video_generation',
            'capabilities': [
                'video_generation',
                'image_generation',
                'reference_frames',
                'custom_resolutions',
                'frame_control',
                'fps_control'
            ]
        }
    
    @staticmethod
    def get_chat_config() -> Dict[str, Any]:
        """Chat worker configuration template"""
        return {
            'base_model_path': '/workspace/models/huggingface_cache/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796',
            'instruct_model_path': '/workspace/models/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28',
            'base_model_name': 'Qwen 2.5-7B Base',
            'instruct_model_name': 'Qwen 2.5-7B Instruct',
            'port': 7861,
            'enhancement_types': ['base', 'instruct'],
            'max_tokens': 2048,
            'memory_requirements': '15GB',
            'role': 'pure_enhancement_and_chat',
            'capabilities': [
                'prompt_enhancement',
                'chat_conversation',
                'base_model_enhancement',
                'instruct_model_enhancement',
                'nsfw_optimization',
                'dynamic_system_prompts'
            ]
        }
    
    @staticmethod
    def get_system_config() -> Dict[str, Any]:
        """System-wide configuration"""
        return {
            'total_vram': '48GB',
            'memory_allocation': {
                'sdxl': '10GB',
                'chat': '15GB',
                'wan': '30GB'
            },
            'startup_priority': {
                'sdxl': 1,
                'chat': 2,
                'wan': 3
            },
            'shared_ports': {
                7860: ['sdxl', 'wan'],
                7861: ['chat']
            },
            'environment_variables': {
                'PYTHONPATH': '/workspace/python_deps/lib/python3.11/site-packages',
                'HF_HOME': '/workspace/models/huggingface_cache',
                'CUDA_VISIBLE_DEVICES': '0'
            }
        }

class ModelConfig:
    """Model-specific configurations for pure inference"""
    
    @staticmethod
    def get_sdxl_model_config() -> Dict[str, Any]:
        """SDXL model-specific configuration"""
        return {
            'attention_slicing': 'auto',
            'enable_xformers': True,
            'compilation': True,
            'dtype': 'float16',
            'safety_checker': None,  # NSFW-first design
            'requires_safety_checking': False,
            'feature_extractor': None,
            'watermarker': None
        }
    
    @staticmethod
    def get_wan_model_config() -> Dict[str, Any]:
        """WAN model-specific configuration"""
        return {
            'task': 't2v-1.3B',
            'compilation': True,
            'dtype': 'float16',
            'attention_slicing': 'auto',
            'enable_xformers': True,
            'thread_safe_timeouts': True,
            'concurrent_futures': True
        }
    
    @staticmethod
    def get_qwen_base_config() -> Dict[str, Any]:
        """Qwen Base model configuration"""
        return {
            'dtype': 'float16',
            'compilation': True,
            'device_map': 'auto',
            'trust_remote_code': True,
            'pad_token': 'eos_token',
            'max_length': 2048
        }
    
    @staticmethod
    def get_qwen_instruct_config() -> Dict[str, Any]:
        """Qwen Instruct model configuration"""
        return {
            'dtype': 'float16',
            'compilation': True,
            'device_map': 'auto',
            'trust_remote_code': True,
            'pad_token': 'eos_token',
            'max_length': 2048
        }

class ValidationConfig:
    """Validation configuration for request parameters"""
    
    @staticmethod
    def get_sdxl_validation_rules() -> Dict[str, Any]:
        """SDXL generation validation rules"""
        return {
            'required_fields': ['prompt'],
            'optional_fields': [
                'steps', 'guidance_scale', 'batch_size', 'resolution',
                'negative_prompt', 'seed', 'reference_image'
            ],
            'validation_rules': {
                'steps': {'type': 'int', 'min': 10, 'max': 50, 'default': 25},
                'guidance_scale': {'type': 'float', 'min': 1.0, 'max': 20.0, 'default': 7.5},
                'batch_size': {'type': 'int', 'allowed': [1, 3, 6], 'default': 1},
                'resolution': {'type': 'str', 'pattern': r'^\d+x\d+$', 'default': '1024x1024'},
                'seed': {'type': 'int', 'min': 0, 'max': 2147483647, 'optional': True},
                'prompt': {'type': 'str', 'min_length': 1, 'max_length': 1000},
                'negative_prompt': {'type': 'str', 'max_length': 1000, 'optional': True}
            }
        }
    
    @staticmethod
    def get_wan_validation_rules() -> Dict[str, Any]:
        """WAN generation validation rules"""
        return {
            'required_fields': ['prompt', 'job_type'],
            'optional_fields': [
                'frames', 'resolution', 'reference_mode', 'reference_image',
                'fps', 'seed'
            ],
            'validation_rules': {
                'frames': {'type': 'int', 'min': 1, 'max': 83, 'default': 83},
                'resolution': {'type': 'str', 'pattern': r'^\d+x\d+$', 'default': '480x832'},
                'reference_mode': {'type': 'str', 'allowed': ['none', 'single', 'start', 'end', 'both'], 'default': 'none'},
                'fps': {'type': 'int', 'min': 8, 'max': 24, 'default': 24},
                'job_type': {'type': 'str', 'allowed': ['image_fast', 'image_high', 'video_fast', 'video_high']},
                'prompt': {'type': 'str', 'min_length': 1, 'max_length': 1000}
            }
        }
    
    @staticmethod
    def get_chat_validation_rules() -> Dict[str, Any]:
        """Chat enhancement validation rules"""
        return {
            'required_fields': ['prompt', 'enhancement_type'],
            'optional_fields': [
                'target_model', 'system_prompt', 'quality', 'nsfw_optimization',
                'max_tokens'
            ],
            'validation_rules': {
                'enhancement_type': {'type': 'str', 'allowed': ['base', 'instruct']},
                'target_model': {'type': 'str', 'allowed': ['sdxl', 'wan'], 'optional': True},
                'quality': {'type': 'str', 'allowed': ['fast', 'high'], 'default': 'fast'},
                'nsfw_optimization': {'type': 'bool', 'default': True},
                'max_tokens': {'type': 'int', 'min': 100, 'max': 2048, 'default': 1024},
                'prompt': {'type': 'str', 'min_length': 1, 'max_length': 1000},
                'system_prompt': {'type': 'str', 'max_length': 2000, 'optional': True}
            }
        }

class PerformanceConfig:
    """Performance and optimization configurations"""
    
    @staticmethod
    def get_optimization_config() -> Dict[str, Any]:
        """System-wide optimization settings"""
        return {
            'torch_compile': True,
            'attention_slicing': 'auto',
            'enable_xformers': True,
            'memory_efficient_attention': True,
            'gradient_checkpointing': False,
            'mixed_precision': 'fp16',
            'cache_dir': '/tmp/model_cache',
            'max_memory': '48GB'
        }
    
    @staticmethod
    def get_timeout_config() -> Dict[str, Any]:
        """Timeout configurations for different operations"""
        return {
            'model_loading': 300,  # 5 minutes
            'generation_timeout': 900,  # 15 minutes
            'enhancement_timeout': 60,  # 1 minute
            'health_check_timeout': 30,  # 30 seconds
            'request_timeout': 600  # 10 minutes
        }
    
    @staticmethod
    def get_retry_config() -> Dict[str, Any]:
        """Retry configurations for error handling"""
        return {
            'max_retries': 3,
            'retry_delay': 5,  # seconds
            'exponential_backoff': True,
            'retryable_errors': [
                'OOM_ERROR',
                'MODEL_LOAD_ERROR',
                'TIMEOUT_ERROR',
                'NETWORK_ERROR'
            ]
        }

# Convenience functions for easy access
def get_worker_config(worker_name: str) -> Dict[str, Any]:
    """Get configuration for a specific worker"""
    configs = {
        'sdxl': WorkerConfig.get_sdxl_config,
        'wan': WorkerConfig.get_wan_config,
        'chat': WorkerConfig.get_chat_config
    }
    return configs.get(worker_name, lambda: {})()
    
def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model-specific configuration"""
    configs = {
        'sdxl': ModelConfig.get_sdxl_model_config,
        'wan': ModelConfig.get_wan_model_config,
        'qwen_base': ModelConfig.get_qwen_base_config,
        'qwen_instruct': ModelConfig.get_qwen_instruct_config
    }
    return configs.get(model_name, lambda: {})()
    
def get_validation_rules(worker_name: str) -> Dict[str, Any]:
    """Get validation rules for a specific worker"""
    rules = {
        'sdxl': ValidationConfig.get_sdxl_validation_rules,
        'wan': ValidationConfig.get_wan_validation_rules,
        'chat': ValidationConfig.get_chat_validation_rules
    }
    return rules.get(worker_name, lambda: {})()
