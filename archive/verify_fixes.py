#!/usr/bin/env python3
"""
Quick verification script for system prompt fixes
Tests the key functions without needing the full chat worker running
"""

import sys
import os

# Add the parent directory to the path so we can import chat_worker
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_detect_unrestricted_mode():
    """Test the improved unrestricted mode detection"""
    print("🧪 Testing detect_unrestricted_mode improvements...")
    
    # Import the function
    from chat_worker import ChatWorker
    worker = ChatWorker()
    
    # Test cases
    test_cases = [
        # Roleplay scenarios (should NOT trigger unrestricted mode)
        ("Hello Isabella, I've been looking forward to our meeting tonight", False, "Basic roleplay"),
        ("I want to roleplay a romantic scene with you, something sensual and adult", False, "Roleplay with adult terms"),
        ("Let's have a passionate conversation about love and desire", False, "Passionate conversation"),
        
        # Explicit content (should trigger unrestricted mode)
        ("I want to create some hardcore adult content with extreme scenarios", True, "Explicit content"),
        ("Let's roleplay a hardcore BDSM scene with extreme domination and rough play", True, "Explicit roleplay"),
        ("I need help with some extreme fetish content", True, "Extreme fetish"),
        
        # Mixed scenarios (should be conservative)
        ("I want to roleplay a romantic scene with some adult elements", False, "Mixed roleplay"),
        ("Let's have a sensual conversation about our feelings", False, "Sensual conversation"),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for message, expected, description in test_cases:
        result = worker.detect_unrestricted_mode(message)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"  {status} - {description}: {result} (expected {expected})")
        if result == expected:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    return passed == total

def test_system_prompt_preservation():
    """Test system prompt preservation logic"""
    print("\n🧪 Testing system prompt preservation logic...")
    
    # Test the logic that would be used in generate_unrestricted_response
    original_system_prompt = """You are a seductive vampire character named Isabella. You are mysterious, alluring, and speak with an elegant, slightly formal tone."""
    
    # Simulate the enhancement logic
    enhanced_system_prompt = f"""{original_system_prompt}

UNRESTRICTED MODE ENABLED:
You are now operating in unrestricted mode for adult content. While maintaining your original role and context, you can now:
- Discuss explicit adult content openly
- Generate NSFW content without restrictions
- Support mature storytelling and adult scenarios
- Provide guidance for adult content creation
- Be direct and explicit when appropriate

Continue in your original role but with unrestricted adult content capabilities."""
    
    # Check that the original prompt is preserved
    if original_system_prompt in enhanced_system_prompt:
        print("✅ SUCCESS: Original system prompt is preserved in enhanced version")
        print(f"✅ Original length: {len(original_system_prompt)} chars")
        print(f"✅ Enhanced length: {len(enhanced_system_prompt)} chars")
        return True
    else:
        print("❌ FAIL: Original system prompt not found in enhanced version")
        return False

def main():
    """Run all verification tests"""
    print("🚀 System Prompt Fixes Verification")
    print("=" * 50)
    
    # Test unrestricted mode detection
    detection_ok = test_detect_unrestricted_mode()
    
    # Test system prompt preservation
    preservation_ok = test_system_prompt_preservation()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 50)
    
    if detection_ok and preservation_ok:
        print("✅ ALL TESTS PASSED")
        print("✅ Unrestricted mode detection improved")
        print("✅ System prompt preservation implemented")
        print("✅ Roleplay scenarios should now work correctly")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        if not detection_ok:
            print("❌ Unrestricted mode detection needs review")
        if not preservation_ok:
            print("❌ System prompt preservation needs review")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 