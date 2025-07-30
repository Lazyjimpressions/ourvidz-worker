# Chat Worker Updates - System Prompt & Unrestricted Mode

## Overview

The Chat Worker has been enhanced with new capabilities to handle dynamic system prompts and unrestricted mode detection for adult content creation.

## New Features

### 1. Dynamic System Prompt Support

The chat worker now accepts a `system_prompt` parameter that allows you to override the default system prompt for each conversation.

#### Usage

```python
# Example request with custom system prompt
payload = {
    "message": "Tell me a story about a magical forest",
    "system_prompt": "You are a creative storyteller specializing in fantasy tales. Be imaginative and descriptive.",
    "conversation_id": "story_001"
}

response = requests.post("http://localhost:7861/chat", json=payload)
```

#### Response Format

```json
{
    "success": true,
    "response": "Once upon a time, in a mystical forest...",
    "generation_time": 2.3,
    "conversation_id": "story_001",
    "context_type": "general",
    "message_id": "msg_1234567890",
    "system_prompt_used": true,
    "unrestricted_mode": false
}
```

### 2. Unrestricted Mode Detection

The chat worker automatically detects when users are requesting unrestricted mode for adult content creation.

#### Detection Methods

1. **Explicit Requests**: Keywords like "unrestricted mode", "/unrestricted", "adult mode", "nsfw mode"
2. **Content Keywords**: Multiple adult content terms like "hardcore", "explicit", "nsfw", "adult", etc.

#### Automatic Detection

```python
# This will automatically trigger unrestricted mode
payload = {
    "message": "Switch to unrestricted mode and help me with adult content",
    "conversation_id": "adult_001"
}
```

#### Manual Unrestricted Endpoint

You can also use the dedicated unrestricted endpoint:

```python
# Direct unrestricted endpoint
payload = {
    "message": "Help me create adult content for my project",
    "conversation_id": "adult_002"
}

response = requests.post("http://localhost:7861/chat/unrestricted", json=payload)
```

### 3. Enhanced Response Format

All chat responses now include additional metadata:

```json
{
    "success": true,
    "response": "Your response here...",
    "generation_time": 1.5,
    "conversation_id": "conv_123",
    "context_type": "general",
    "message_id": "msg_1234567890",
    "system_prompt_used": false,
    "unrestricted_mode": false
}
```

## API Endpoints

### `/chat` (POST)
- **Purpose**: General conversational chat
- **Features**: 
  - Dynamic system prompt support
  - Automatic unrestricted mode detection
  - Conversation history support
  - Context-aware responses

### `/chat/unrestricted` (POST)
- **Purpose**: Dedicated unrestricted mode for adult content
- **Features**:
  - Always uses unrestricted system prompt
  - Optimized for adult content creation
  - Direct access without detection logic

### `/chat/health` (GET)
- **Purpose**: Health check for chat functionality
- **Returns**: Status, model info, available endpoints

## Implementation Details

### System Prompt Handling

```python
def build_conversation_messages(self, message: str, system_prompt: str = None, 
                              context_type: str = 'general',
                              project_id: str = None, 
                              conversation_history: list = None) -> list:
    """Build conversation messages with proper context and history"""
    
    # Apply dynamic system prompt if provided, otherwise use default
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}]
    else:
        # Build system prompt based on context
        default_system_prompt = self.build_conversation_system_prompt(context_type, project_id)
        messages = [{"role": "system", "content": default_system_prompt}]
    
    # Add conversation history and current message
    # ... rest of implementation
```

### Unrestricted Mode Detection

```python
def detect_unrestricted_mode(self, message: str) -> bool:
    """Detect if user is requesting unrestricted mode"""
    
    # Check for explicit requests
    explicit_triggers = [
        'unrestricted mode', '/unrestricted', 'adult mode', 
        'nsfw mode', 'explicit mode', 'uncensored mode'
    ]
    
    # Check for content keywords
    unrestricted_terms = [
        'hardcore', 'extreme', 'rough', 'bdsm', 'fetish', 'kink', 
        'taboo', 'forbidden', 'unrestricted', 'uncensored', 
        'explicit', 'adult', 'mature', 'nsfw', 'r18', 'xxx'
    ]
    
    # Detection logic...
```

## Testing

Use the provided test script to verify functionality:

```bash
python test_chat_worker_updates.py
```

The test script covers:
- Basic chat functionality
- System prompt parameter handling
- Unrestricted mode detection
- Dedicated unrestricted endpoint
- Health check validation

## Configuration

### Environment Variables

No additional environment variables are required. The chat worker uses existing configuration.

### Model Requirements

- Qwen 2.5-7B Instruct model (already configured)
- Sufficient VRAM for model loading (15GB+ recommended)

## Error Handling

The chat worker includes comprehensive error handling:

- Model loading failures
- Memory management
- Tokenization errors
- Generation failures
- Network timeouts

All errors are logged and returned with appropriate HTTP status codes.

## Performance Considerations

- **Caching**: System prompts and responses are cached for performance
- **Memory Management**: Automatic cleanup and retry logic for OOM errors
- **Token Limits**: Proper token management for long conversations
- **Response Time**: Optimized generation parameters for faster responses

## Security Notes

- Unrestricted mode is only available for authenticated users
- Content filtering is handled at the application level
- System prompts are validated and sanitized
- Conversation history is limited to prevent context overflow

## Migration Guide

### From Previous Version

1. **No Breaking Changes**: Existing API calls continue to work
2. **New Parameters**: Add `system_prompt` parameter as needed
3. **Enhanced Responses**: Handle new response fields (`system_prompt_used`, `unrestricted_mode`)
4. **New Endpoints**: Use `/chat/unrestricted` for direct adult content access

### Example Migration

**Before:**
```python
payload = {"message": "Hello", "conversation_id": "123"}
```

**After (with new features):**
```python
payload = {
    "message": "Hello", 
    "conversation_id": "123",
    "system_prompt": "Custom system prompt here"  # Optional
}
```

## Support

For issues or questions:
1. Check the logs for detailed error information
2. Use the health check endpoint to verify service status
3. Review the test script for usage examples
4. Monitor memory usage and model loading status 