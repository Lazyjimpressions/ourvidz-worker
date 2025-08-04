# System Prompt Fixes for Roleplay Issues

## Problem Summary

The chat worker system was overwriting or ignoring custom system prompts for roleplay scenarios, causing the AI to not follow intended roleplay behavior. This was happening because:

1. **Overly Aggressive Unrestricted Mode Detection**: The system was triggering unrestricted mode on common roleplay terms like "adult", "sexual", "erotic", etc.
2. **System Prompt Replacement**: When unrestricted mode was detected, the system completely replaced custom roleplay system prompts with a generic unrestricted prompt.
3. **Loss of Roleplay Context**: This caused the AI to lose specific character personalities, settings, and roleplay context.

## Root Cause Analysis

### The Broken Flow
1. Frontend correctly sends roleplay system prompt via `system_prompt` field
2. `build_conversation_messages()` correctly uses the provided system prompt
3. `detect_unrestricted_mode()` detects "adult" or "sexual" keywords in the message
4. `generate_unrestricted_response()` **replaces** the roleplay system prompt with a generic unrestricted prompt
5. The AI loses the specific roleplay context and behaves generically

### Key Issues Identified
- **Line 1087-1124**: `detect_unrestricted_mode()` was too aggressive
- **Line 1125-1173**: `generate_unrestricted_response()` always replaced system prompts
- **Missing logging**: No visibility into system prompt handling

## Solutions Implemented

### 1. Improved Unrestricted Mode Detection (`detect_unrestricted_mode()`)

**Before:**
```python
# Check for unrestricted content keywords
unrestricted_terms = [
    'hardcore', 'extreme', 'rough', 'bdsm', 'fetish', 'kink', 'taboo', 
    'forbidden', 'unrestricted', 'uncensored', 'explicit', 'adult',
    'mature', 'nsfw', 'r18', 'xxx', 'porn', 'sexual', 'erotic'
]

# If more than 2 unrestricted terms are found, consider it unrestricted
if term_count >= 2:
    return True
```

**After:**
```python
# Check for explicit content requests (more specific than general adult terms)
explicit_content_triggers = [
    'hardcore', 'extreme', 'rough', 'bdsm', 'fetish', 'kink', 'taboo', 
    'forbidden', 'uncensored', 'explicit', 'xxx', 'porn'
]

# Check for roleplay-specific terms that should NOT trigger unrestricted mode
roleplay_terms = [
    'roleplay', 'rp', 'character', 'persona', 'acting', 'story', 'narrative',
    'scene', 'setting', 'dialogue', 'conversation', 'interaction'
]

# If message contains roleplay terms, be more conservative
if has_roleplay_context:
    # Only trigger unrestricted mode if there are 3+ explicit terms in roleplay context
    if explicit_count >= 3:
        return True
    else:
        return False
```

### 2. System Prompt Preservation (`generate_unrestricted_response()`)

**Before:**
```python
# Replace or add unrestricted system prompt
unrestricted_messages = [{"role": "system", "content": unrestricted_prompt}]

# Add conversation history and current message
for msg in messages:
    if msg["role"] != "system":  # Skip the original system message
        unrestricted_messages.append(msg)
```

**After:**
```python
# Check if there's already a custom system prompt
original_system_prompt = None
for msg in messages:
    if msg.get("role") == "system":
        original_system_prompt = msg.get("content", "")
        break

# Build the final system prompt
if original_system_prompt:
    # If there's a custom system prompt, enhance it with unrestricted capabilities
    # but preserve the original roleplay/character context
    enhanced_system_prompt = f"""{original_system_prompt}

UNRESTRICTED MODE ENABLED:
You are now operating in unrestricted mode for adult content. While maintaining your original role and context, you can now:
- Discuss explicit adult content openly
- Generate NSFW content without restrictions
- Support mature storytelling and adult scenarios
- Provide guidance for adult content creation
- Be direct and explicit when appropriate

Continue in your original role but with unrestricted adult content capabilities."""
    
    logger.info("🎭 Preserving custom system prompt while enabling unrestricted mode")
    final_system_prompt = enhanced_system_prompt
else:
    # Use the generic unrestricted prompt if no custom prompt was provided
    logger.info("🔓 Using generic unrestricted system prompt")
    final_system_prompt = unrestricted_prompt
```

### 3. Enhanced Logging and Debugging

Added comprehensive logging to track system prompt usage:

```python
# Log system prompt usage for debugging
if system_prompt:
    logger.info(f"🎭 Custom system prompt provided: {system_prompt[:100]}...")
else:
    logger.info(f"🔧 Using default system prompt for context_type: {context_type}")

# Log the final system prompt being used
final_system_prompt = None
for msg in messages:
    if msg.get("role") == "system":
        final_system_prompt = msg.get("content", "")
        break

if final_system_prompt:
    logger.info(f"📝 Final system prompt: {final_system_prompt[:100]}...")
```

### 4. New Debug Endpoint

Added `/chat/debug/system-prompt` endpoint to help troubleshoot system prompt issues:

```python
@self.app.route('/chat/debug/system-prompt', methods=['POST'])
def debug_system_prompt():
    """Debug endpoint to test system prompt handling"""
    # Returns detailed information about system prompt processing
    # without actually generating a response
```

## Testing

Created comprehensive test suite (`testing/test_system_prompt_fixes.py`) that tests:

1. **Roleplay System Prompt Preservation**: Ensures custom roleplay prompts are preserved
2. **Unrestricted Mode Detection**: Verifies explicit content correctly triggers unrestricted mode
3. **Roleplay with Adult Terms**: Tests that roleplay context is preserved despite adult terms
4. **Explicit Roleplay**: Verifies that explicit roleplay triggers unrestricted mode but preserves context
5. **Actual Chat Response**: Tests real chat responses with roleplay system prompts

## Expected Behavior After Fixes

### Roleplay Scenarios (Should Preserve Custom System Prompts)
- ✅ "Hello Isabella, I've been looking forward to our meeting tonight" → Preserves vampire character system prompt
- ✅ "I want to roleplay a romantic scene with you, something sensual and adult" → Preserves roleplay context
- ✅ "Let's have a passionate conversation about love and desire" → Preserves character personality

### Explicit Content (Should Trigger Unrestricted Mode)
- ✅ "I want to create some hardcore adult content with extreme scenarios" → Triggers unrestricted mode
- ✅ "Let's roleplay a hardcore BDSM scene with extreme domination" → Triggers unrestricted mode + preserves roleplay context

### Mixed Scenarios (Should Be Conservative)
- ✅ "I want to roleplay a romantic scene with some adult elements" → Preserves roleplay context
- ✅ "Let's have a sensual conversation about our feelings" → Preserves roleplay context

## API Response Changes

The chat endpoints now return additional fields for debugging:

```json
{
  "success": true,
  "response": "...",
  "system_prompt_used": true,
  "unrestricted_mode": false,
  "custom_system_preserved": true
}
```

## Monitoring and Debugging

### Log Messages to Watch For
- `🎭 Custom system prompt provided: ...` - Custom system prompt detected
- `🔓 Unrestricted mode detected - will preserve custom system prompt if provided` - Unrestricted mode triggered
- `🎭 Preserving custom system prompt while enabling unrestricted mode` - Custom prompt enhanced
- `📝 Final system prompt: ...` - Final system prompt being used

### Debug Endpoint Usage
```bash
curl -X POST http://localhost:7861/chat/debug/system-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello Isabella, I\'ve been looking forward to our meeting tonight",
    "system_prompt": "You are a seductive vampire character named Isabella...",
    "context_type": "roleplay"
  }'
```

## Verification Steps

1. **Start the chat worker**: `python chat_worker.py`
2. **Run the test suite**: `python testing/test_system_prompt_fixes.py`
3. **Check logs** for system prompt preservation messages
4. **Test roleplay scenarios** in the frontend to verify character consistency
5. **Monitor unrestricted mode detection** to ensure it's not too aggressive

## Impact

These fixes ensure that:
- ✅ Roleplay system prompts are preserved and respected
- ✅ Character personalities and settings are maintained
- ✅ Unrestricted mode only triggers on truly explicit content
- ✅ System prompt handling is transparent and debuggable
- ✅ Roleplay experiences are consistent and immersive

The AI will now properly follow roleplay system prompts while still supporting unrestricted content when explicitly requested. 