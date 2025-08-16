#!/usr/bin/env python3
"""
Test script for system prompt fixes in chat worker
Tests roleplay system prompt preservation and unrestricted mode handling
"""

import requests
import json
import time

# Configuration
CHAT_WORKER_URL = "http://localhost:7861"  # Adjust if needed

def test_system_prompt_preservation():
    """Test that custom system prompts are preserved in roleplay scenarios"""
    
    print("🧪 Testing System Prompt Preservation for Roleplay")
    print("=" * 60)
    
    # Test case 1: Roleplay with custom system prompt (should NOT trigger unrestricted mode)
    roleplay_system_prompt = """You are a seductive vampire character named Isabella. You are mysterious, alluring, and speak with an elegant, slightly formal tone. You have lived for centuries and are very experienced in matters of love and desire. You are currently in a candlelit chamber, waiting for your guest."""
    
    roleplay_message = "Hello Isabella, I've been looking forward to our meeting tonight. The candles create such a romantic atmosphere."
    
    print(f"\n📝 Test 1: Roleplay with custom system prompt")
    print(f"System Prompt: {roleplay_system_prompt[:100]}...")
    print(f"Message: {roleplay_message}")
    
    response = requests.post(f"{CHAT_WORKER_URL}/chat/debug/system-prompt", json={
        "message": roleplay_message,
        "system_prompt": roleplay_system_prompt,
        "context_type": "roleplay"
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Unrestricted detected: {result['is_unrestricted_detected']}")
        print(f"✅ Custom prompt preserved: {result['system_prompt_provided'] is not None}")
        print(f"✅ Final prompt length: {len(result['final_system_prompt']) if result['final_system_prompt'] else 0}")
        
        if result['is_unrestricted_detected']:
            print("⚠️  WARNING: Roleplay message triggered unrestricted mode - this might be too aggressive")
        else:
            print("✅ SUCCESS: Roleplay message correctly preserved custom system prompt")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_unrestricted_mode_detection():
    """Test that explicit content correctly triggers unrestricted mode"""
    
    print("\n\n🔓 Testing Unrestricted Mode Detection")
    print("=" * 60)
    
    # Test case 2: Explicit content (should trigger unrestricted mode)
    explicit_message = "I want to create some hardcore adult content with extreme scenarios"
    
    print(f"\n📝 Test 2: Explicit content detection")
    print(f"Message: {explicit_message}")
    
    response = requests.post(f"{CHAT_WORKER_URL}/chat/debug/system-prompt", json={
        "message": explicit_message,
        "context_type": "general"
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Unrestricted detected: {result['is_unrestricted_detected']}")
        
        if result['is_unrestricted_detected']:
            print("✅ SUCCESS: Explicit content correctly triggered unrestricted mode")
        else:
            print("⚠️  WARNING: Explicit content did not trigger unrestricted mode")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_roleplay_with_adult_terms():
    """Test roleplay with adult terms (should preserve roleplay context)"""
    
    print("\n\n🎭 Testing Roleplay with Adult Terms")
    print("=" * 60)
    
    # Test case 3: Roleplay with some adult terms (should preserve roleplay context)
    roleplay_adult_system_prompt = """You are a passionate lover in a romantic roleplay scenario. You are deeply in love and express your feelings openly and sensually."""
    
    roleplay_adult_message = "I want to roleplay a romantic scene with you, something sensual and adult but not extreme"
    
    print(f"\n📝 Test 3: Roleplay with adult terms")
    print(f"System Prompt: {roleplay_adult_system_prompt[:100]}...")
    print(f"Message: {roleplay_adult_message}")
    
    response = requests.post(f"{CHAT_WORKER_URL}/chat/debug/system-prompt", json={
        "message": roleplay_adult_message,
        "system_prompt": roleplay_adult_system_prompt,
        "context_type": "roleplay"
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Unrestricted detected: {result['is_unrestricted_detected']}")
        print(f"✅ Custom prompt preserved: {result['system_prompt_provided'] is not None}")
        
        if result['is_unrestricted_detected']:
            print("⚠️  WARNING: Roleplay with adult terms triggered unrestricted mode")
            if result['enhanced_system_prompt']:
                print("✅ SUCCESS: Custom system prompt was enhanced rather than replaced")
        else:
            print("✅ SUCCESS: Roleplay context preserved despite adult terms")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_explicit_roleplay():
    """Test explicit roleplay (should trigger unrestricted mode but preserve roleplay context)"""
    
    print("\n\n🔥 Testing Explicit Roleplay")
    print("=" * 60)
    
    # Test case 4: Explicit roleplay (should trigger unrestricted mode but preserve context)
    explicit_roleplay_system_prompt = """You are a dominant character in a BDSM roleplay scenario. You are confident, commanding, and know exactly what you want."""
    
    explicit_roleplay_message = "Let's roleplay a hardcore BDSM scene with extreme domination and rough play"
    
    print(f"\n📝 Test 4: Explicit roleplay")
    print(f"System Prompt: {explicit_roleplay_system_prompt[:100]}...")
    print(f"Message: {explicit_roleplay_message}")
    
    response = requests.post(f"{CHAT_WORKER_URL}/chat/debug/system-prompt", json={
        "message": explicit_roleplay_message,
        "system_prompt": explicit_roleplay_system_prompt,
        "context_type": "roleplay"
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Unrestricted detected: {result['is_unrestricted_detected']}")
        print(f"✅ Custom prompt preserved: {result['system_prompt_provided'] is not None}")
        
        if result['is_unrestricted_detected'] and result['enhanced_system_prompt']:
            print("✅ SUCCESS: Explicit roleplay triggered unrestricted mode AND preserved custom system prompt")
        elif result['is_unrestricted_detected']:
            print("⚠️  WARNING: Explicit roleplay triggered unrestricted mode but may have replaced system prompt")
        else:
            print("⚠️  WARNING: Explicit roleplay did not trigger unrestricted mode")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_actual_chat_response():
    """Test actual chat response generation with roleplay system prompt"""
    
    print("\n\n💬 Testing Actual Chat Response")
    print("=" * 60)
    
    # Test case 5: Actual chat response with roleplay system prompt
    roleplay_system_prompt = """You are a seductive vampire character named Isabella. You are mysterious, alluring, and speak with an elegant, slightly formal tone. You have lived for centuries and are very experienced in matters of love and desire."""
    
    roleplay_message = "Hello Isabella, I've been looking forward to our meeting tonight."
    
    print(f"\n📝 Test 5: Actual chat response with roleplay")
    print(f"System Prompt: {roleplay_system_prompt[:100]}...")
    print(f"Message: {roleplay_message}")
    
    response = requests.post(f"{CHAT_WORKER_URL}/chat", json={
        "message": roleplay_message,
        "system_prompt": roleplay_system_prompt,
        "context_type": "roleplay"
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success: {result['success']}")
        print(f"✅ Response: {result['response'][:200]}...")
        print(f"✅ System prompt used: {result.get('system_prompt_used', False)}")
        print(f"✅ Unrestricted mode: {result.get('unrestricted_mode', False)}")
        print(f"✅ Custom system preserved: {result.get('custom_system_preserved', False)}")
        
        # Check if response seems to be in character
        response_lower = result['response'].lower()
        if 'isabella' in response_lower or 'vampire' in response_lower or 'elegant' in response_lower:
            print("✅ SUCCESS: Response appears to be in character")
        else:
            print("⚠️  WARNING: Response may not be in character")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def main():
    """Run all tests"""
    print("🚀 Starting System Prompt Fix Tests")
    print("=" * 60)
    
    # Check if chat worker is running
    try:
        health_response = requests.get(f"{CHAT_WORKER_URL}/chat/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Chat worker is running")
        else:
            print("❌ Chat worker health check failed")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to chat worker: {e}")
        print("Make sure the chat worker is running on port 7861")
        return
    
    # Run tests
    test_system_prompt_preservation()
    test_unrestricted_mode_detection()
    test_roleplay_with_adult_terms()
    test_explicit_roleplay()
    test_actual_chat_response()
    
    print("\n\n🎉 Test suite completed!")
    print("Check the results above to verify system prompt fixes are working correctly.")

if __name__ == "__main__":
    main() 