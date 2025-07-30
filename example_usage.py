#!/usr/bin/env python3
"""
Example Usage of Chat Worker Updates
Demonstrates the new system_prompt parameter and unrestricted mode features
"""

import requests
import json

# Configuration
CHAT_WORKER_URL = "http://localhost:7861"

def example_basic_chat():
    """Example of basic chat functionality"""
    print("📝 Example 1: Basic Chat")
    print("-" * 40)
    
    payload = {
        "message": "Hello! Can you help me with a creative writing project?",
        "conversation_id": "example_001"
    }
    
    response = requests.post(f"{CHAT_WORKER_URL}/chat", json=payload)
    result = response.json()
    
    if result.get('success'):
        print(f"✅ Response: {result['response'][:150]}...")
        print(f"   System prompt used: {result.get('system_prompt_used', False)}")
        print(f"   Unrestricted mode: {result.get('unrestricted_mode', False)}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    print()

def example_custom_system_prompt():
    """Example of using custom system prompt"""
    print("📝 Example 2: Custom System Prompt")
    print("-" * 40)
    
    custom_prompt = """You are an expert creative writing coach specializing in science fiction. 
    Your responses should be encouraging, provide specific writing tips, and help develop 
    creative ideas. Be enthusiastic about the user's writing journey."""
    
    payload = {
        "message": "I want to write a sci-fi story about time travel. Can you help me brainstorm some ideas?",
        "system_prompt": custom_prompt,
        "conversation_id": "example_002"
    }
    
    response = requests.post(f"{CHAT_WORKER_URL}/chat", json=payload)
    result = response.json()
    
    if result.get('success'):
        print(f"✅ Response: {result['response'][:150]}...")
        print(f"   System prompt used: {result.get('system_prompt_used', False)}")
        print(f"   Unrestricted mode: {result.get('unrestricted_mode', False)}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    print()

def example_unrestricted_detection():
    """Example of automatic unrestricted mode detection"""
    print("📝 Example 3: Automatic Unrestricted Mode Detection")
    print("-" * 40)
    
    payload = {
        "message": "I need help creating adult content for my project. Can you switch to unrestricted mode?",
        "conversation_id": "example_003"
    }
    
    response = requests.post(f"{CHAT_WORKER_URL}/chat", json=payload)
    result = response.json()
    
    if result.get('success'):
        print(f"✅ Response: {result['response'][:150]}...")
        print(f"   System prompt used: {result.get('system_prompt_used', False)}")
        print(f"   Unrestricted mode: {result.get('unrestricted_mode', False)}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    print()

def example_direct_unrestricted():
    """Example of using the direct unrestricted endpoint"""
    print("📝 Example 4: Direct Unrestricted Endpoint")
    print("-" * 40)
    
    payload = {
        "message": "Help me develop adult content ideas for my creative project",
        "conversation_id": "example_004"
    }
    
    response = requests.post(f"{CHAT_WORKER_URL}/chat/unrestricted", json=payload)
    result = response.json()
    
    if result.get('success'):
        print(f"✅ Response: {result['response'][:150]}...")
        print(f"   System prompt used: {result.get('system_prompt_used', False)}")
        print(f"   Unrestricted mode: {result.get('unrestricted_mode', False)}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    print()

def example_conversation_history():
    """Example of conversation with history"""
    print("📝 Example 5: Conversation with History")
    print("-" * 40)
    
    # First message
    payload1 = {
        "message": "I'm writing a fantasy novel about dragons",
        "conversation_id": "example_005"
    }
    
    response1 = requests.post(f"{CHAT_WORKER_URL}/chat", json=payload1)
    result1 = response1.json()
    
    if result1.get('success'):
        print(f"✅ First response: {result1['response'][:100]}...")
        
        # Second message with history
        conversation_history = [
            {"sender": "user", "content": payload1["message"]},
            {"sender": "assistant", "content": result1["response"]}
        ]
        
        payload2 = {
            "message": "Can you help me develop the main character?",
            "conversation_id": "example_005",
            "conversation_history": conversation_history
        }
        
        response2 = requests.post(f"{CHAT_WORKER_URL}/chat", json=payload2)
        result2 = response2.json()
        
        if result2.get('success'):
            print(f"✅ Second response: {result2['response'][:100]}...")
            print(f"   System prompt used: {result2.get('system_prompt_used', False)}")
            print(f"   Unrestricted mode: {result2.get('unrestricted_mode', False)}")
        else:
            print(f"❌ Error in second message: {result2.get('error', 'Unknown error')}")
    else:
        print(f"❌ Error in first message: {result1.get('error', 'Unknown error')}")
    
    print()

def example_keyword_detection():
    """Example of unrestricted mode detection via keywords"""
    print("📝 Example 6: Keyword-Based Unrestricted Detection")
    print("-" * 40)
    
    payload = {
        "message": "I need help with hardcore adult content and explicit material for my creative project",
        "conversation_id": "example_006"
    }
    
    response = requests.post(f"{CHAT_WORKER_URL}/chat", json=payload)
    result = response.json()
    
    if result.get('success'):
        print(f"✅ Response: {result['response'][:150]}...")
        print(f"   System prompt used: {result.get('system_prompt_used', False)}")
        print(f"   Unrestricted mode: {result.get('unrestricted_mode', False)}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    print()

def check_health():
    """Check chat worker health"""
    print("🏥 Checking Chat Worker Health")
    print("-" * 40)
    
    try:
        response = requests.get(f"{CHAT_WORKER_URL}/chat/health", timeout=10)
        result = response.json()
        
        print(f"✅ Status: {result.get('status', 'unknown')}")
        print(f"✅ Chat ready: {result.get('chat_ready', False)}")
        print(f"✅ Available endpoints: {list(result.get('endpoints', {}).keys())}")
        
        if result.get('model_info'):
            model_info = result['model_info']
            print(f"✅ Model loaded: {model_info.get('loaded', False)}")
            print(f"✅ Model name: {model_info.get('model_name', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    print()

def main():
    """Run all examples"""
    print("🚀 Chat Worker Update Examples")
    print("=" * 50)
    
    # Check health first
    check_health()
    
    # Run examples
    example_basic_chat()
    example_custom_system_prompt()
    example_unrestricted_detection()
    example_direct_unrestricted()
    example_conversation_history()
    example_keyword_detection()
    
    print("=" * 50)
    print("✅ All examples completed!")
    print("\n💡 Tips:")
    print("- Update CHAT_WORKER_URL if your worker is on a different host/port")
    print("- Check the logs for detailed information about each request")
    print("- Use the test script for automated testing")
    print("- Review CHAT_WORKER_UPDATES.md for complete documentation")

if __name__ == "__main__":
    main() 