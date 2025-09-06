#!/usr/bin/env python3
"""
Memory Emergency Handler - Active Memory Management
Handles memory conflicts and enforces memory limits across workers
"""

import os
import sys
import time
import requests
import logging
from typing import Dict, Optional

# Add the current directory to Python path
sys.path.append('/workspace/python_deps/lib/python3.11/site-packages')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryEmergencyHandler:
    def __init__(self):
        """Initialize memory emergency handler"""
        self.worker_urls = self.auto_detect_worker_urls()
        logger.info(f"🧠 Memory Emergency Handler initialized with URLs: {self.worker_urls}")
    
    def auto_detect_worker_urls(self) -> Dict[str, str]:
        """Auto-detect worker URLs from RunPod environment"""
        # Check for RunPod environment
        pod_id = os.getenv('RUNPOD_POD_ID')
        if pod_id:
            urls = {
                'sdxl': f"https://{pod_id}-7859.proxy.runpod.net",
                'wan': f"https://{pod_id}-7860.proxy.runpod.net", 
                'chat': f"https://{pod_id}-7861.proxy.runpod.net"
            }
            logger.info(f"🌐 Auto-detected RunPod URLs: {urls}")
            return urls
        else:
            # Try hostname fallback
            import socket
            hostname = socket.gethostname()
            if '-' in hostname:
                pod_id = hostname.split('-')[0]
                urls = {
                    'sdxl': f"https://{pod_id}-7859.proxy.runpod.net",
                    'wan': f"https://{pod_id}-7860.proxy.runpod.net",
                    'chat': f"https://{pod_id}-7861.proxy.runpod.net"
                }
                logger.info(f"🌐 Auto-detected hostname URLs: {urls}")
                return urls
        
        logger.warning("⚠️ Could not auto-detect worker URLs")
        return {}
    
    def get_worker_memory_status(self, worker: str) -> Optional[Dict]:
        """Get memory status from a worker"""
        if not self.worker_urls.get(worker):
            return None
            
        try:
            url = f"{self.worker_urls[worker]}/memory/status"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"⚠️ Could not get memory status from {worker}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Error getting memory status from {worker}: {e}")
            return None
    
    def force_unload_worker(self, worker: str) -> bool:
        """Force unload a worker's models"""
        if not self.worker_urls.get(worker):
            logger.error(f"❌ No URL configured for {worker} worker")
            return False
            
        try:
            url = f"{self.worker_urls[worker]}/memory/unload"
            response = requests.post(url, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                success = result.get('success', False)
                logger.info(f"{'✅' if success else '❌'} Force unload {worker}: {result.get('message', 'Unknown')}")
                return success
            else:
                logger.error(f"❌ Failed to force unload {worker}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error force unloading {worker}: {e}")
            return False
    
    def get_total_memory_usage(self) -> tuple:
        """Get total memory usage across all workers"""
        total_allocated = 0
        worker_status = {}
        
        for worker in ['sdxl', 'chat', 'wan']:
            status = self.get_worker_memory_status(worker)
            if status:
                allocated = status.get('allocated_vram', 0)
                total_allocated += allocated
                worker_status[worker] = {
                    'allocated': allocated,
                    'model_loaded': status.get('model_loaded', False),
                    'total_vram': status.get('total_vram', 0)
                }
            else:
                worker_status[worker] = {'allocated': 0, 'model_loaded': False, 'total_vram': 0}
        
        return total_allocated, worker_status
    
    def handle_memory_conflict(self, target_worker: str, required_memory_gb: float = 6.0) -> Dict:
        """Handle memory conflict by freeing up space for target worker"""
        logger.warning(f"🚨 MEMORY CONFLICT: Need {required_memory_gb}GB for {target_worker}")
        
        # Get current memory status
        total_allocated, worker_status = self.get_total_memory_usage()
        
        # Get total VRAM (from any worker)
        total_vram = 0
        for worker_data in worker_status.values():
            if worker_data['total_vram'] > 0:
                total_vram = worker_data['total_vram']
                break
        
        available = total_vram - total_allocated
        
        logger.info(f"📊 Current memory status:")
        logger.info(f"   Total VRAM: {total_vram:.2f}GB")
        logger.info(f"   Allocated: {total_allocated:.2f}GB")
        logger.info(f"   Available: {available:.2f}GB")
        logger.info(f"   Required: {required_memory_gb:.2f}GB")
        
        if available >= required_memory_gb:
            logger.info(f"✅ Sufficient memory available for {target_worker}")
            return {'success': True, 'action': 'none', 'reason': 'sufficient_memory'}
        
        # Need to free memory - determine which workers to unload
        workers_to_unload = []
        
        # Priority order: unload chat first (least critical), then WAN, never SDXL
        if target_worker != 'chat' and worker_status['chat']['model_loaded']:
            workers_to_unload.append('chat')
        
        if target_worker != 'wan' and worker_status['wan']['model_loaded']:
            workers_to_unload.append('wan')
        
        logger.info(f"🔄 Will unload workers: {workers_to_unload}")
        
        # Unload workers
        unload_results = {}
        for worker in workers_to_unload:
            logger.info(f"🗑️ Unloading {worker} worker...")
            success = self.force_unload_worker(worker)
            unload_results[worker] = success
        
        # Wait for unload to complete
        time.sleep(3)
        
        # Check memory after unload
        total_allocated_after, worker_status_after = self.get_total_memory_usage()
        available_after = total_vram - total_allocated_after
        
        logger.info(f"📊 Memory after unload:")
        logger.info(f"   Allocated: {total_allocated_after:.2f}GB")
        logger.info(f"   Available: {available_after:.2f}GB")
        
        success = available_after >= required_memory_gb
        
        return {
            'success': success,
            'action': 'unload_workers',
            'workers_unloaded': workers_to_unload,
            'unload_results': unload_results,
            'memory_before': {'allocated': total_allocated, 'available': available},
            'memory_after': {'allocated': total_allocated_after, 'available': available_after},
            'reason': 'memory_freed' if success else 'insufficient_memory_after_unload'
        }
    
    def emergency_memory_cleanup(self) -> Dict:
        """Emergency memory cleanup - unload all non-essential workers"""
        logger.warning("🚨 EMERGENCY MEMORY CLEANUP: Unloading all non-essential workers")
        
        # Unload chat and WAN workers (keep SDXL as it's always needed)
        workers_to_unload = ['chat', 'wan']
        unload_results = {}
        
        for worker in workers_to_unload:
            logger.info(f"🗑️ Emergency unloading {worker} worker...")
            success = self.force_unload_worker(worker)
            unload_results[worker] = success
        
        # Wait for cleanup
        time.sleep(5)
        
        # Check final memory status
        total_allocated, worker_status = self.get_total_memory_usage()
        
        # Get total VRAM
        total_vram = 0
        for worker_data in worker_status.values():
            if worker_data['total_vram'] > 0:
                total_vram = worker_data['total_vram']
                break
        
        available = total_vram - total_allocated
        
        return {
            'success': available > 10,  # Success if we have >10GB available
            'workers_unloaded': workers_to_unload,
            'unload_results': unload_results,
            'final_memory': {
                'total_vram': total_vram,
                'allocated': total_allocated,
                'available': available
            }
        }

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory Emergency Handler')
    parser.add_argument('action', choices=['status', 'conflict', 'cleanup'], 
                       help='Action to perform')
    parser.add_argument('--worker', default='sdxl', 
                       help='Target worker for conflict resolution')
    parser.add_argument('--memory', type=float, default=6.0,
                       help='Required memory in GB')
    
    args = parser.parse_args()
    
    handler = MemoryEmergencyHandler()
    
    if args.action == 'status':
        total_allocated, worker_status = handler.get_total_memory_usage()
        print(f"📊 Memory Status:")
        print(f"   Total Allocated: {total_allocated:.2f}GB")
        for worker, status in worker_status.items():
            print(f"   {worker}: {status['allocated']:.2f}GB allocated, model loaded: {status['model_loaded']}")
    
    elif args.action == 'conflict':
        result = handler.handle_memory_conflict(args.worker, args.memory)
        print(f"🚨 Memory Conflict Resolution: {result}")
    
    elif args.action == 'cleanup':
        result = handler.emergency_memory_cleanup()
        print(f"🧹 Emergency Cleanup: {result}")

if __name__ == "__main__":
    main()
