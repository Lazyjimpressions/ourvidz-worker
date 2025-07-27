#!/usr/bin/env python3
"""
OurVidz Memory Manager - Smart VRAM allocation for triple worker system
Handles: SDXL (always loaded) + Chat (when possible) + WAN (on demand)
"""

import requests
import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, total_vram_gb: float = 49):
        """Initialize memory manager with VRAM capacity"""
        self.total_vram = total_vram_gb
        self.safety_buffer = 2  # GB buffer for safety
        self.usable_vram = total_vram_gb - self.safety_buffer
        
        # Worker memory requirements
        self.worker_memory = {
            'sdxl': 10,      # Always loaded
            'chat': 15,      # Load when possible
            'wan': 30        # Load on demand
        }
        
        # Worker URLs (will be auto-detected)
        self.worker_urls = {
            'sdxl': None,
            'chat': None,
            'wan': None
        }
        
        logger.info(f"🧠 Memory Manager initialized: {total_vram_gb}GB total, {self.usable_vram}GB usable")

    def set_worker_urls(self, urls: Dict[str, str]):
        """Set worker URLs for memory management"""
        self.worker_urls.update(urls)
        logger.info(f"🌐 Worker URLs configured: {self.worker_urls}")

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

    def get_total_memory_usage(self) -> Tuple[float, Dict[str, bool]]:
        """Get total VRAM usage across all workers"""
        total_used = 0
        worker_status = {}
        
        for worker in ['sdxl', 'chat', 'wan']:
            status = self.get_worker_memory_status(worker)
            if status and status.get('model_loaded', False):
                total_used += self.worker_memory[worker]
                worker_status[worker] = True
            else:
                worker_status[worker] = False
        
        logger.info(f"📊 Memory usage: {total_used}GB used, {self.usable_vram - total_used}GB available")
        return total_used, worker_status

    def can_load_worker(self, worker: str) -> bool:
        """Check if worker can be loaded without exceeding memory limits"""
        current_usage, _ = self.get_total_memory_usage()
        required = self.worker_memory[worker]
        
        would_use = current_usage + required
        can_load = would_use <= self.usable_vram
        
        logger.info(f"🔍 Can load {worker}? Current: {current_usage}GB + Required: {required}GB = {would_use}GB (limit: {self.usable_vram}GB) → {'✅' if can_load else '❌'}")
        return can_load

    def prepare_for_wan_job(self) -> bool:
        """Prepare memory for WAN job by unloading chat if necessary"""
        logger.info("🎬 Preparing memory for WAN job...")
        
        if self.can_load_worker('wan'):
            logger.info("✅ Sufficient memory for WAN job")
            return True
        
        # Need to free memory - unload chat worker
        logger.info("⚠️ Insufficient memory, unloading chat worker...")
        return self.unload_worker('chat')

    def prepare_for_chat_request(self) -> str:
        """Prepare for chat request - ensure chat worker is loaded or find alternative"""
        logger.info("🤖 Preparing for chat request...")
        
        # Check if chat worker is already loaded
        _, worker_status = self.get_total_memory_usage()
        
        if worker_status.get('chat', False):
            logger.info("✅ Chat worker already loaded and ready")
            return 'chat'
        
        # Try to load chat worker
        if self.can_load_worker('chat'):
            logger.info("🔄 Loading chat worker...")
            if self.load_worker('chat'):
                return 'chat'
        
        # Fallback to WAN worker
        logger.info("⚠️ Chat worker unavailable, using WAN worker fallback")
        return 'wan'

    def load_worker(self, worker: str) -> bool:
        """Load a worker's models"""
        if not self.worker_urls.get(worker):
            logger.error(f"❌ No URL configured for {worker} worker")
            return False
            
        try:
            url = f"{self.worker_urls[worker]}/memory/load"
            response = requests.post(url, timeout=60)  # Loading can take time
            
            if response.status_code == 200:
                result = response.json()
                success = result.get('success', False)
                logger.info(f"{'✅' if success else '❌'} Load {worker} worker: {result.get('message', 'Unknown')}")
                return success
            else:
                logger.error(f"❌ Failed to load {worker} worker: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading {worker} worker: {e}")
            return False

    def unload_worker(self, worker: str) -> bool:
        """Unload a worker's models"""
        if not self.worker_urls.get(worker):
            logger.error(f"❌ No URL configured for {worker} worker")
            return False
            
        try:
            url = f"{self.worker_urls[worker]}/memory/unload"
            response = requests.post(url, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                success = result.get('success', False)
                logger.info(f"{'✅' if success else '❌'} Unload {worker} worker: {result.get('message', 'Unknown')}")
                return success
            else:
                logger.error(f"❌ Failed to unload {worker} worker: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error unloading {worker} worker: {e}")
            return False

    def cleanup_after_wan_job(self):
        """Cleanup after WAN job completion - reload chat if needed"""
        logger.info("🧹 Cleaning up after WAN job...")
        
        # Check if chat worker needs to be reloaded
        _, worker_status = self.get_total_memory_usage()
        
        if not worker_status.get('chat', False) and self.can_load_worker('chat'):
            logger.info("🔄 Reloading chat worker after WAN job completion")
            self.load_worker('chat')
        else:
            logger.info("ℹ️ Chat worker already loaded or insufficient memory")

    def get_memory_report(self) -> Dict:
        """Get comprehensive memory report"""
        total_used, worker_status = self.get_total_memory_usage()
        
        return {
            'total_vram': self.total_vram,
            'usable_vram': self.usable_vram,
            'safety_buffer': self.safety_buffer,
            'current_usage': total_used,
            'available': self.usable_vram - total_used,
            'worker_status': worker_status,
            'worker_memory_requirements': self.worker_memory,
            'can_load': {
                worker: self.can_load_worker(worker) 
                for worker in ['sdxl', 'chat', 'wan']
            }
        }

# Singleton instance for use in edge functions
memory_manager = MemoryManager()

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    return memory_manager