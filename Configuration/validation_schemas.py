"""
Request Validation Schemas
Pure Inference Architecture - August 16, 2025

This module provides comprehensive validation schemas for all worker requests.
All validation is parameter-focused, not business logic focused.
Workers validate request format and parameters, not user permissions or content.
"""

import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ValidationError(Exception):
    """Custom validation error with detailed information"""
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation error in {field}: {message}")

class ValidationResult:
    """Result of validation operation"""
    def __init__(self, is_valid: bool, errors: List[ValidationError] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, field: str, message: str, value: Any = None):
        """Add a validation error"""
        self.errors.append(ValidationError(field, message, value))
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a validation warning"""
        self.warnings.append(message)

class EnhancementType(Enum):
    """Enhancement type enumeration"""
    BASE = "base"
    INSTRUCT = "instruct"

class JobType(Enum):
    """Job type enumeration"""
    # SDXL job types
    SDXL_GENERATE = "sdxl_generate"
    
    # WAN job types
    WAN_IMAGE_FAST = "image_fast"
    WAN_IMAGE_HIGH = "image_high"
    WAN_VIDEO_FAST = "video_fast"
    WAN_VIDEO_HIGH = "video_high"
    
    # Chat job types
    CHAT_ENHANCE = "enhance"
    CHAT_CONVERSATION = "chat"
    CHAT_UNRESTRICTED = "chat_unrestricted"

@dataclass
class ValidationRule:
    """Individual validation rule"""
    field: str
    rule_type: str
    required: bool = False
    default: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None

class BaseValidator:
    """Base validation class with common validation methods"""
    
    @staticmethod
    def validate_string(value: Any, field: str, min_length: int = None, max_length: int = None, pattern: str = None) -> ValidationResult:
        """Validate string field"""
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            result.add_error(field, f"Expected string, got {type(value).__name__}")
            return result
        
        if min_length and len(value) < min_length:
            result.add_error(field, f"Minimum length {min_length}, got {len(value)}")
        
        if max_length and len(value) > max_length:
            result.add_error(field, f"Maximum length {max_length}, got {len(value)}")
        
        if pattern and not re.match(pattern, value):
            result.add_error(field, f"Pattern mismatch: {pattern}")
        
        return result
    
    @staticmethod
    def validate_integer(value: Any, field: str, min_value: int = None, max_value: int = None, allowed_values: List[int] = None) -> ValidationResult:
        """Validate integer field"""
        result = ValidationResult(True)
        
        if not isinstance(value, int):
            result.add_error(field, f"Expected integer, got {type(value).__name__}")
            return result
        
        if min_value is not None and value < min_value:
            result.add_error(field, f"Minimum value {min_value}, got {value}")
        
        if max_value is not None and value > max_value:
            result.add_error(field, f"Maximum value {max_value}, got {value}")
        
        if allowed_values and value not in allowed_values:
            result.add_error(field, f"Allowed values {allowed_values}, got {value}")
        
        return result
    
    @staticmethod
    def validate_float(value: Any, field: str, min_value: float = None, max_value: float = None) -> ValidationResult:
        """Validate float field"""
        result = ValidationResult(True)
        
        if not isinstance(value, (int, float)):
            result.add_error(field, f"Expected number, got {type(value).__name__}")
            return result
        
        float_value = float(value)
        
        if min_value is not None and float_value < min_value:
            result.add_error(field, f"Minimum value {min_value}, got {float_value}")
        
        if max_value is not None and float_value > max_value:
            result.add_error(field, f"Maximum value {max_value}, got {float_value}")
        
        return result
    
    @staticmethod
    def validate_boolean(value: Any, field: str) -> ValidationResult:
        """Validate boolean field"""
        result = ValidationResult(True)
        
        if not isinstance(value, bool):
            result.add_error(field, f"Expected boolean, got {type(value).__name__}")
        
        return result
    
    @staticmethod
    def validate_enum(value: Any, field: str, enum_class: Enum) -> ValidationResult:
        """Validate enum field"""
        result = ValidationResult(True)
        
        try:
            enum_class(value)
        except ValueError:
            allowed_values = [e.value for e in enum_class]
            result.add_error(field, f"Allowed values {allowed_values}, got {value}")
        
        return result

class SDXLValidator(BaseValidator):
    """SDXL generation request validator"""
    
    @staticmethod
    def validate_generation_request(data: Dict[str, Any]) -> ValidationResult:
        """Validate SDXL generation request"""
        result = ValidationResult(True)
        
        # Required fields
        if 'prompt' not in data:
            result.add_error('prompt', 'Required field missing')
        else:
            prompt_result = SDXLValidator.validate_string(
                data['prompt'], 'prompt', min_length=1, max_length=1000
            )
            result.errors.extend(prompt_result.errors)
            result.warnings.extend(prompt_result.warnings)
        
        # Optional fields with validation
        if 'steps' in data:
            steps_result = SDXLValidator.validate_integer(
                data['steps'], 'steps', min_value=10, max_value=50
            )
            result.errors.extend(steps_result.errors)
        else:
            data['steps'] = 25  # Default value
        
        if 'guidance_scale' in data:
            guidance_result = SDXLValidator.validate_float(
                data['guidance_scale'], 'guidance_scale', min_value=1.0, max_value=20.0
            )
            result.errors.extend(guidance_result.errors)
        else:
            data['guidance_scale'] = 7.5  # Default value
        
        if 'batch_size' in data:
            batch_result = SDXLValidator.validate_integer(
                data['batch_size'], 'batch_size', allowed_values=[1, 3, 6]
            )
            result.errors.extend(batch_result.errors)
        else:
            data['batch_size'] = 1  # Default value
        
        if 'resolution' in data:
            resolution_result = SDXLValidator.validate_string(
                data['resolution'], 'resolution', pattern=r'^\d+x\d+$'
            )
            result.errors.extend(resolution_result.errors)
        else:
            data['resolution'] = '1024x1024'  # Default value
        
        if 'negative_prompt' in data:
            neg_prompt_result = SDXLValidator.validate_string(
                data['negative_prompt'], 'negative_prompt', max_length=1000
            )
            result.errors.extend(neg_prompt_result.errors)
        
        if 'seed' in data:
            seed_result = SDXLValidator.validate_integer(
                data['seed'], 'seed', min_value=0, max_value=2147483647
            )
            result.errors.extend(seed_result.errors)
        
        if 'reference_image' in data:
            # Validate reference image format/URL
            if not isinstance(data['reference_image'], str):
                result.add_error('reference_image', 'Expected string URL or base64')
        
        return result

class WANValidator(BaseValidator):
    """WAN generation request validator"""
    
    @staticmethod
    def validate_generation_request(data: Dict[str, Any]) -> ValidationResult:
        """Validate WAN generation request"""
        result = ValidationResult(True)
        
        # Required fields
        if 'prompt' not in data:
            result.add_error('prompt', 'Required field missing')
        else:
            prompt_result = WANValidator.validate_string(
                data['prompt'], 'prompt', min_length=1, max_length=1000
            )
            result.errors.extend(prompt_result.errors)
        
        if 'job_type' not in data:
            result.add_error('job_type', 'Required field missing')
        else:
            job_type_result = WANValidator.validate_enum(
                data['job_type'], 'job_type', JobType
            )
            result.errors.extend(job_type_result.errors)
        
        # Optional fields with validation
        if 'frames' in data:
            frames_result = WANValidator.validate_integer(
                data['frames'], 'frames', min_value=1, max_value=83
            )
            result.errors.extend(frames_result.errors)
        else:
            data['frames'] = 83  # Default value
        
        if 'resolution' in data:
            resolution_result = WANValidator.validate_string(
                data['resolution'], 'resolution', pattern=r'^\d+x\d+$'
            )
            result.errors.extend(resolution_result.errors)
        else:
            data['resolution'] = '480x832'  # Default value
        
        if 'reference_mode' in data:
            allowed_modes = ['none', 'single', 'start', 'end', 'both']
            if data['reference_mode'] not in allowed_modes:
                result.add_error('reference_mode', f'Allowed values {allowed_modes}, got {data["reference_mode"]}')
        else:
            data['reference_mode'] = 'none'  # Default value
        
        if 'fps' in data:
            fps_result = WANValidator.validate_integer(
                data['fps'], 'fps', min_value=8, max_value=24
            )
            result.errors.extend(fps_result.errors)
        else:
            data['fps'] = 24  # Default value
        
        if 'seed' in data:
            seed_result = WANValidator.validate_integer(
                data['seed'], 'seed', min_value=0, max_value=2147483647
            )
            result.errors.extend(seed_result.errors)
        
        if 'reference_image' in data:
            if not isinstance(data['reference_image'], str):
                result.add_error('reference_image', 'Expected string URL or base64')
        
        return result

class ChatValidator(BaseValidator):
    """Chat enhancement request validator"""
    
    @staticmethod
    def validate_enhancement_request(data: Dict[str, Any]) -> ValidationResult:
        """Validate Chat enhancement request"""
        result = ValidationResult(True)
        
        # Required fields
        if 'prompt' not in data:
            result.add_error('prompt', 'Required field missing')
        else:
            prompt_result = ChatValidator.validate_string(
                data['prompt'], 'prompt', min_length=1, max_length=1000
            )
            result.errors.extend(prompt_result.errors)
        
        if 'enhancement_type' not in data:
            result.add_error('enhancement_type', 'Required field missing')
        else:
            enhancement_result = ChatValidator.validate_enum(
                data['enhancement_type'], 'enhancement_type', EnhancementType
            )
            result.errors.extend(enhancement_result.errors)
        
        # Optional fields with validation
        if 'target_model' in data:
            allowed_models = ['sdxl', 'wan']
            if data['target_model'] not in allowed_models:
                result.add_error('target_model', f'Allowed values {allowed_models}, got {data["target_model"]}')
        
        if 'quality' in data:
            allowed_qualities = ['fast', 'high']
            if data['quality'] not in allowed_qualities:
                result.add_error('quality', f'Allowed values {allowed_qualities}, got {data["quality"]}')
        else:
            data['quality'] = 'fast'  # Default value
        
        if 'nsfw_optimization' in data:
            nsfw_result = ChatValidator.validate_boolean(
                data['nsfw_optimization'], 'nsfw_optimization'
            )
            result.errors.extend(nsfw_result.errors)
        else:
            data['nsfw_optimization'] = True  # Default value
        
        if 'max_tokens' in data:
            tokens_result = ChatValidator.validate_integer(
                data['max_tokens'], 'max_tokens', min_value=100, max_value=2048
            )
            result.errors.extend(tokens_result.errors)
        else:
            data['max_tokens'] = 1024  # Default value
        
        if 'system_prompt' in data:
            system_result = ChatValidator.validate_string(
                data['system_prompt'], 'system_prompt', max_length=2000
            )
            result.errors.extend(system_result.errors)
        
        return result
    
    @staticmethod
    def validate_chat_request(data: Dict[str, Any]) -> ValidationResult:
        """Validate Chat conversation request"""
        result = ValidationResult(True)
        
        # Required fields
        if 'prompt' not in data:
            result.add_error('prompt', 'Required field missing')
        else:
            prompt_result = ChatValidator.validate_string(
                data['prompt'], 'prompt', min_length=1, max_length=1000
            )
            result.errors.extend(prompt_result.errors)
        
        # Optional fields
        if 'system_prompt' in data:
            system_result = ChatValidator.validate_string(
                data['system_prompt'], 'system_prompt', max_length=2000
            )
            result.errors.extend(system_result.errors)
        
        if 'max_tokens' in data:
            tokens_result = ChatValidator.validate_integer(
                data['max_tokens'], 'max_tokens', min_value=100, max_value=2048
            )
            result.errors.extend(tokens_result.errors)
        else:
            data['max_tokens'] = 1024  # Default value
        
        return result

class RequestValidator:
    """Main request validator that routes to specific validators"""
    
    @staticmethod
    def validate_request(worker_type: str, request_type: str, data: Dict[str, Any]) -> ValidationResult:
        """Validate request based on worker type and request type"""
        
        if worker_type == 'sdxl':
            if request_type == 'generate':
                return SDXLValidator.validate_generation_request(data)
            else:
                return ValidationResult(False, [ValidationError('request_type', f'Unknown request type: {request_type}')])
        
        elif worker_type == 'wan':
            if request_type == 'generate':
                return WANValidator.validate_generation_request(data)
            else:
                return ValidationResult(False, [ValidationError('request_type', f'Unknown request type: {request_type}')])
        
        elif worker_type == 'chat':
            if request_type == 'enhance':
                return ChatValidator.validate_enhancement_request(data)
            elif request_type == 'chat':
                return ChatValidator.validate_chat_request(data)
            else:
                return ValidationResult(False, [ValidationError('request_type', f'Unknown request type: {request_type}')])
        
        else:
            return ValidationResult(False, [ValidationError('worker_type', f'Unknown worker type: {worker_type}')])

# Convenience functions
def validate_sdxl_request(data: Dict[str, Any]) -> ValidationResult:
    """Validate SDXL generation request"""
    return RequestValidator.validate_request('sdxl', 'generate', data)

def validate_wan_request(data: Dict[str, Any]) -> ValidationResult:
    """Validate WAN generation request"""
    return RequestValidator.validate_request('wan', 'generate', data)

def validate_chat_enhancement_request(data: Dict[str, Any]) -> ValidationResult:
    """Validate Chat enhancement request"""
    return RequestValidator.validate_request('chat', 'enhance', data)

def validate_chat_conversation_request(data: Dict[str, Any]) -> ValidationResult:
    """Validate Chat conversation request"""
    return RequestValidator.validate_request('chat', 'chat', data)
