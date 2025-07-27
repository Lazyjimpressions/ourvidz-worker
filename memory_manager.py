#!/usr/bin/env python3
"""
OurVidz Memory Manager - Smart VRAM allocation for triple worker system
Handles: SDXL (always loaded) + Chat (when possible) + WAN (on demand)

ENHANCED FEATURES:
- Memory pressure detection (critical/high/medium/low)
- Emergency memory management with intelligent fallback
- Force unload capabilities for critical situations
- Predictive loading based on usage patterns
- Comprehensive emergency status reporting
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

    def get_memory_pressure_level(self) -> str:
        """Detect memory pressure level"""
        total_used, _ = self.get_total_memory_usage()
        available = self.usable_vram - total_used
        
        if available < 5:
            return "critical"  # <5GB available
        elif available < 10:
            return "high"      # 5-10GB available
        elif available < 15:
            return "medium"    # 10-15GB available
        else:
            return "low"       # >15GB available

    def force_unload_all_except(self, target: str, reason: str = "emergency") -> bool:
        """Emergency VRAM clearing - use sparingly"""
        logger.warning(f"🚨 EMERGENCY: Force unloading all workers except {target} - Reason: {reason}")
        
        results = {}
        for worker in ['sdxl', 'chat', 'wan']:
            if worker != target:
                logger.warning(f"🗑️ Force unloading {worker}...")
                results[worker] = self.unload_worker(worker)
        
        # Verify we freed enough memory
        time.sleep(2)  # Allow unloading to complete
        if self.can_load_worker(target):
            logger.info(f"✅ Emergency unload successful, {target} can now load")
            return True
        else:
            logger.error(f"❌ Emergency unload failed, {target} still cannot load")
            return False

    def should_preload_chat(self) -> bool:
        """Decide if chat should be preloaded based on usage patterns"""
        # Check memory pressure
        pressure = self.get_memory_pressure_level()
        if pressure in ["critical", "high"]:
            return False
        
        # Check if WAN job is queued (would need to implement queue checking)
        # For now, assume no WAN job is queued if we're considering preloading
        wan_job_queued = False  # TODO: Implement queue checking
        
        # Check if chat was used recently (would need to track this)
        # For now, assume it was used recently
        chat_used_recently = True  # TODO: Implement usage tracking
        
        should_preload = (
            pressure in ["low", "medium"] and
            not wan_job_queued and
            chat_used_recently
        )
        
        logger.info(f"🔍 Should preload chat? Pressure: {pressure}, WAN queued: {wan_job_queued}, Used recently: {chat_used_recently} → {'✅' if should_preload else '❌'}")
        return should_preload

    def get_emergency_memory_status(self) -> Dict:
        """Get detailed memory status for emergency situations"""
        total_used, worker_status = self.get_total_memory_usage()
        pressure = self.get_memory_pressure_level()
        
        return {
            'memory_pressure': pressure,
            'total_used_gb': total_used,
            'available_gb': self.usable_vram - total_used,
            'worker_status': worker_status,
            'can_load_wan': self.can_load_worker('wan'),
            'can_load_chat': self.can_load_worker('chat'),
            'emergency_actions_available': {
                'force_unload_chat': worker_status.get('chat', False),
                'force_unload_sdxl': worker_status.get('sdxl', False),
                'force_unload_all_except_wan': pressure in ["critical", "high"],
                'force_unload_all_except_chat': pressure in ["critical", "high"]
            }
        }

    def handle_emergency_memory_request(self, target_worker: str, reason: str = "emergency") -> Dict:
        """Handle emergency memory requests with intelligent fallback"""
        logger.warning(f"🚨 EMERGENCY MEMORY REQUEST: Need to load {target_worker} - Reason: {reason}")
        
        # Get current status
        status = self.get_emergency_memory_status()
        
        # If we can already load the target, no emergency needed
        if status[f'can_load_{target_worker}']:
            logger.info(f"✅ {target_worker} can be loaded without emergency measures")
            return {
                'success': True,
                'action_taken': 'none',
                'reason': 'sufficient_memory_available',
                'status': status
            }
        
        # Determine best emergency action based on memory pressure
        pressure = status['memory_pressure']
        
        if pressure == "critical":
            # Critical pressure - use nuclear option
            logger.warning("🚨 CRITICAL MEMORY PRESSURE - Using nuclear unload option")
            success = self.force_unload_all_except(target_worker, f"critical_pressure_{reason}")
            return {
                'success': success,
                'action_taken': 'force_unload_all_except',
                'target_worker': target_worker,
                'reason': f'critical_pressure_{reason}',
                'status': self.get_emergency_memory_status()
            }
        
        elif pressure == "high":
            # High pressure - try selective unloading first
            logger.warning("⚠️ HIGH MEMORY PRESSURE - Attempting selective unloading")
            
            # Try unloading chat first (if not the target)
            if target_worker != 'chat' and status['worker_status'].get('chat', False):
                logger.info("🔄 Attempting to unload chat worker first...")
                if self.unload_worker('chat') and self.can_load_worker(target_worker):
                    return {
                        'success': True,
                        'action_taken': 'unload_chat',
                        'reason': f'high_pressure_{reason}',
                        'status': self.get_emergency_memory_status()
                    }
            
            # If selective unloading didn't work, use nuclear option
            success = self.force_unload_all_except(target_worker, f"high_pressure_{reason}")
            return {
                'success': success,
                'action_taken': 'force_unload_all_except',
                'target_worker': target_worker,
                'reason': f'high_pressure_{reason}',
                'status': self.get_emergency_memory_status()
            }
        
        else:
            # Medium/low pressure - should be able to handle normally
            logger.info(f"ℹ️ Memory pressure is {pressure} - should be manageable")
            return {
                'success': False,
                'action_taken': 'none',
                'reason': f'unexpected_pressure_{pressure}',
                'status': status
            }

    def get_memory_report(self) -> Dict:
        """Get comprehensive memory report"""
        total_used, worker_status = self.get_total_memory_usage()
        pressure = self.get_memory_pressure_level()
        
        return {
            'total_vram': self.total_vram,
            'usable_vram': self.usable_vram,
            'safety_buffer': self.safety_buffer,
            'current_usage': total_used,
            'available': self.usable_vram - total_used,
            'memory_pressure': pressure,
            'worker_status': worker_status,
            'worker_memory_requirements': self.worker_memory,
            'can_load': {
                worker: self.can_load_worker(worker) 
                for worker in ['sdxl', 'chat', 'wan']
            },
            'emergency_actions': {
                'force_unload_all_except_wan': pressure in ["critical", "high"],
                'force_unload_all_except_chat': pressure in ["critical", "high"],
                'should_preload_chat': self.should_preload_chat()
            }
        }

# Singleton instance for use in edge functions
memory_manager = MemoryManager()

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    return memory_manager

def emergency_memory_operation(operation: str, target_worker: str = None, reason: str = "emergency") -> Dict:
    """Emergency memory operation interface for edge functions"""
    mm = get_memory_manager()
    
    if operation == "status":
        return mm.get_emergency_memory_status()
    
    elif operation == "force_unload_all_except":
        if not target_worker:
            return {'error': 'target_worker required for force_unload_all_except'}
        return {
            'success': mm.force_unload_all_except(target_worker, reason),
            'operation': operation,
            'target_worker': target_worker,
            'reason': reason
        }
    
    elif operation == "handle_emergency_request":
        if not target_worker:
            return {'error': 'target_worker required for handle_emergency_request'}
        return mm.handle_emergency_memory_request(target_worker, reason)
    
    elif operation == "memory_report":
        return mm.get_memory_report()
    
    else:
        return {'error': f'Unknown operation: {operation}'}