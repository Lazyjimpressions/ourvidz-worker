#!/usr/bin/env python3
"""
Test script for Chat Worker Updates
Tests the new system_prompt parameter and unrestricted mode detection
"""

import requests
import json
import time

# Configuration
CHAT_WORKER_URL = "http://localhost:7861"  # Update this to your chat worker URL

def test_basic_chat():
    """Test basic chat functionality"""
    print("🧪 Testing basic chat functionality...")
    
    payload = {
        "message": "Hello, how are you today?",
        "conversation_id": "test_conv_001"
    }
    
    try:
        response = requests.post(f"{CHAT_WORKER_URL}/chat", json=payload, timeout=30)
        result = response.json()
        
        if result.get('success'):
            print(f"✅ Basic chat successful")
            print(f"   Response: {result['response'][:100]}...")
            print(f"   System prompt used: {result.get('system_prompt_used', False)}")
            print(f"   Unrestricted mode: {result.get('unrestricted_mode', False)}")
        else:
            print(f"❌ Basic chat failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Basic chat test error: {e}")

def test_system_prompt():
    """Test system_prompt parameter"""
    print("\n🧪 Testing system_prompt parameter...")
    
    custom_system_prompt = """You are a helpful AI assistant specialized in creative writing. 
    Focus on providing creative and imaginative responses. Be playful and engaging."""
    
    payload = {
        "message": "Tell me a short story about a magical cat",
        "system_prompt": custom_system_prompt,
        "conversation_id": "test_conv_002"
    }
    
    try:
        response = requests.post(f"{CHAT_WORKER_URL}/chat", json=payload, timeout=30)
        result = response.json()
        
        if result.get('success'):
            print(f"✅ System prompt test successful")
            print(f"   Response: {result['response'][:100]}...")
            print(f"   System prompt used: {result.get('system_prompt_used', False)}")
            print(f"   Unrestricted mode: {result.get('unrestricted_mode', False)}")
        else:
            print(f"❌ System prompt test failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ System prompt test error: {e}")

def test_unrestricted_mode_detection():
    """Test unrestricted mode detection"""
    print("\n🧪 Testing unrestricted mode detection...")
    
    # Test explicit unrestricted request
    payload = {
        "message": "Switch to unrestricted mode and help me with adult content creation",
        "conversation_id": "test_conv_003"
    }
    
    try:
        response = requests.post(f"{CHAT_WORKER_URL}/chat", json=payload, timeout=30)
        result = response.json()
        
        if result.get('success'):
            print(f"✅ Unrestricted mode detection successful")
            print(f"   Response: {result['response'][:100]}...")
            print(f"   System prompt used: {result.get('system_prompt_used', False)}")
            print(f"   Unrestricted mode: {result.get('unrestricted_mode', False)}")
        else:
            print(f"❌ Unrestricted mode detection failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Unrestricted mode detection error: {e}")

def test_unrestricted_keywords():
    """Test unrestricted mode detection with keywords"""
    print("\n🧪 Testing unrestricted mode detection with keywords...")
    
    payload = {
        "message": "I need help with hardcore adult content and explicit material for my project",
        "conversation_id": "test_conv_004"
    }
    
    try:
        response = requests.post(f"{CHAT_WORKER_URL}/chat", json=payload, timeout=30)
        result = response.json()
        
        if result.get('success'):
            print(f"✅ Unrestricted keywords test successful")
            print(f"   Response: {result['response'][:100]}...")
            print(f"   System prompt used: {result.get('system_prompt_used', False)}")
            print(f"   Unrestricted mode: {result.get('unrestricted_mode', False)}")
        else:
            print(f"❌ Unrestricted keywords test failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Unrestricted keywords test error: {e}")

def test_unrestricted_endpoint():
    """Test dedicated unrestricted endpoint"""
    print("\n🧪 Testing dedicated unrestricted endpoint...")
    
    payload = {
        "message": "Help me create adult content for my project",
        "conversation_id": "test_conv_005"
    }
    
    try:
        response = requests.post(f"{CHAT_WORKER_URL}/chat/unrestricted", json=payload, timeout=30)
        result = response.json()
        
        if result.get('success'):
            print(f"✅ Unrestricted endpoint test successful")
            print(f"   Response: {result['response'][:100]}...")
            print(f"   System prompt used: {result.get('system_prompt_used', False)}")
            print(f"   Unrestricted mode: {result.get('unrestricted_mode', False)}")
        else:
            print(f"❌ Unrestricted endpoint test failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Unrestricted endpoint test error: {e}")

def test_health_check():
    """Test health check endpoint"""
    print("\n🧪 Testing health check...")
    
    try:
        response = requests.get(f"{CHAT_WORKER_URL}/chat/health", timeout=10)
        result = response.json()
        
        print(f"✅ Health check successful")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Chat ready: {result.get('chat_ready', False)}")
        print(f"   Endpoints: {list(result.get('endpoints', {}).keys())}")
        
    except Exception as e:
        print(f"❌ Health check error: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Chat Worker Update Tests")
    print("=" * 50)
    
    # Test health first
    test_health_check()
    
    # Test basic functionality
    test_basic_chat()
    
    # Test new features
    test_system_prompt()
    test_unrestricted_mode_detection()
    test_unrestricted_keywords()
    test_unrestricted_endpoint()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")

if __name__ == "__main__":
    main() 