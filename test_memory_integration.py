#!/usr/bin/env python3
"""
Test script for memory manager integration
Tests the memory endpoints and worker coordination
"""

import requests
import time
import json

def test_worker_memory_endpoints():
    """Test memory endpoints on all workers"""
    print("🧪 Testing Memory Manager Integration")
    print("=" * 50)
    
    # Test endpoints for each worker
    workers = {
        'sdxl': 'http://localhost:7859',
        'wan': 'http://localhost:7860', 
        'chat': 'http://localhost:7861'
    }
    
    results = {}
    
    for worker_name, base_url in workers.items():
        print(f"\n🔍 Testing {worker_name.upper()} Worker")
        print("-" * 30)
        
        # Test health endpoint
        try:
            health_response = requests.get(f"{base_url}/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"✅ Health: {health_data.get('status', 'unknown')}")
            else:
                print(f"❌ Health: HTTP {health_response.status_code}")
        except Exception as e:
            print(f"❌ Health: {e}")
        
        # Test memory status endpoint
        try:
            memory_response = requests.get(f"{base_url}/memory/status", timeout=5)
            if memory_response.status_code == 200:
                memory_data = memory_response.json()
                print(f"✅ Memory Status:")
                print(f"   Total VRAM: {memory_data.get('total_vram', 'N/A'):.1f}GB")
                print(f"   Allocated: {memory_data.get('allocated_vram', 'N/A'):.1f}GB")
                print(f"   Available: {memory_data.get('available_vram', 'N/A'):.1f}GB")
                print(f"   Memory Fraction: {memory_data.get('memory_fraction', 'N/A')}")
                print(f"   Model Loaded: {memory_data.get('model_loaded', 'N/A')}")
                results[worker_name] = memory_data
            else:
                print(f"❌ Memory Status: HTTP {memory_response.status_code}")
        except Exception as e:
            print(f"❌ Memory Status: {e}")
    
    return results

def test_memory_manager():
    """Test memory manager functionality"""
    print(f"\n🧠 Testing Memory Manager")
    print("-" * 30)
    
    try:
        from memory_manager import get_memory_manager
        
        mm = get_memory_manager()
        
        # Test memory report
        report = mm.get_memory_report()
        print(f"✅ Memory Report:")
        print(f"   Total VRAM: {report['total_vram']}GB")
        print(f"   Usable VRAM: {report['usable_vram']}GB")
        print(f"   Current Usage: {report['current_usage']}GB")
        print(f"   Available: {report['available']}GB")
        print(f"   Memory Pressure: {report['memory_pressure']}")
        
        # Test worker status
        print(f"   Worker Status:")
        for worker, loaded in report['worker_status'].items():
            print(f"     {worker}: {'✅ Loaded' if loaded else '❌ Not Loaded'}")
        
        # Test can load checks
        print(f"   Can Load:")
        for worker, can_load in report['can_load'].items():
            print(f"     {worker}: {'✅ Yes' if can_load else '❌ No'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory Manager Test: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Memory Manager Integration Test")
    print("=" * 50)
    
    # Test individual worker endpoints
    worker_results = test_worker_memory_endpoints()
    
    # Test memory manager
    memory_manager_ok = test_memory_manager()
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 50)
    print(f"Workers Tested: {len(worker_results)}")
    print(f"Memory Manager: {'✅ OK' if memory_manager_ok else '❌ Failed'}")
    
    if worker_results:
        total_vram = sum(data.get('total_vram', 0) for data in worker_results.values())
        total_allocated = sum(data.get('allocated_vram', 0) for data in worker_results.values())
        print(f"Total VRAM: {total_vram:.1f}GB")
        print(f"Total Allocated: {total_allocated:.1f}GB")
        print(f"Total Available: {total_vram - total_allocated:.1f}GB")
    
    print(f"\n🎯 Memory Manager Integration: {'✅ COMPLETE' if memory_manager_ok else '❌ INCOMPLETE'}")

if __name__ == "__main__":
    main()
