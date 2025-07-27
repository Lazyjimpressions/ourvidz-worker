#!/usr/bin/env python3
"""
Worker Registration System - Auto-register workers with Supabase
Handles: Chat worker registration for enhancement routing
"""

import os
import time
import requests
import logging
from typing import Optional
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class WorkerRegistration:
    def __init__(self, worker_type: str, worker_port: int):
        """Initialize worker registration"""
        self.worker_type = worker_type  # 'sdxl', 'wan', 'chat'
        self.worker_port = worker_port
        self.worker_url = None
        self.supabase = None
        self.registration_id = None
        
        # Initialize Supabase client
        self.setup_supabase()
        
        # Detect worker URL
        self.detect_worker_url()
        
        logger.info(f"🌐 Worker Registration initialized: {worker_type} on port {worker_port}")

    def setup_supabase(self):
        """Setup Supabase client for registration"""
        try:
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            
            if not supabase_url or not supabase_key:
                logger.warning("⚠️ Supabase credentials not found, registration will be skipped")
                return
                
            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("✅ Supabase client initialized for registration")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Supabase client: {e}")

    def detect_worker_url(self):
        """Detect worker URL from RunPod environment"""
        try:
            # Try RUNPOD_POD_ID first
            pod_id = os.getenv('RUNPOD_POD_ID')
            
            if pod_id:
                self.worker_url = f"https://{pod_id}-{self.worker_port}.proxy.runpod.net"
                logger.info(f"🔍 Detected URL from RUNPOD_POD_ID: {self.worker_url}")
                return
            
            # Fallback to hostname
            import socket
            hostname = socket.gethostname()
            
            if '-' in hostname:
                pod_id = hostname.split('-')[0]
                self.worker_url = f"https://{pod_id}-{self.worker_port}.proxy.runpod.net"
                logger.info(f"🔍 Detected URL from hostname: {self.worker_url}")
                return
            
            logger.warning("⚠️ Could not detect RunPod URL, registration will be skipped")
            
        except Exception as e:
            logger.error(f"❌ Error detecting worker URL: {e}")

    def validate_worker_health(self) -> bool:
        """Validate worker is responding to health checks"""
        if not self.worker_url:
            return False
            
        try:
            health_url = f"{self.worker_url}/health"
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"✅ Worker health check passed: {health_url}")
                return True
            else:
                logger.warning(f"⚠️ Worker health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Worker health check error: {e}")
            return False

    def register_worker(self) -> bool:
        """Register worker with Supabase"""
        if not self.supabase or not self.worker_url:
            logger.warning("⚠️ Cannot register worker: missing Supabase client or URL")
            return False
        
        # Validate health first
        if not self.validate_worker_health():
            logger.error("❌ Worker health check failed, cannot register")
            return False
        
        try:
            registration_data = {
                'worker_type': self.worker_type,
                'worker_url': self.worker_url,
                'worker_port': self.worker_port,
                'status': 'active',
                'last_heartbeat': 'now()',
                'capabilities': self.get_worker_capabilities(),
                'metadata': {
                    'version': '1.0.0',
                    'startup_time': time.time(),
                    'environment': 'runpod'
                }
            }
            
            # Insert or update registration
            result = self.supabase.table('worker_registry').upsert(
                registration_data,
                on_conflict='worker_type'
            ).execute()
            
            if result.data:
                self.registration_id = result.data[0]['id']
                logger.info(f"✅ Worker registered successfully: {self.worker_type} -> {self.worker_url}")
                return True
            else:
                logger.error("❌ Worker registration failed: no data returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ Worker registration error: {e}")
            return False

    def get_worker_capabilities(self) -> list:
        """Get worker capabilities for registration"""
        capabilities = {
            'sdxl': ['image_generation', 'fast_generation', 'batch_generation', 'seed_control'],
            'wan': ['video_generation', 'enhanced_jobs', 'base_model_enhancement', 'reference_frames'],
            'chat': ['manual_enhancement', 'instant_response', 'conversational_ai', 'admin_tools']
        }
        
        return capabilities.get(self.worker_type, [])

    def send_heartbeat(self):
        """Send heartbeat to maintain registration"""
        if not self.supabase or not self.registration_id:
            return
            
        try:
            self.supabase.table('worker_registry').update({
                'last_heartbeat': 'now()',
                'status': 'active'
            }).eq('id', self.registration_id).execute()
            
            logger.debug(f"💓 Heartbeat sent for {self.worker_type} worker")
            
        except Exception as e:
            logger.error(f"❌ Heartbeat error: {e}")

    def start_heartbeat_loop(self, interval: int = 30):
        """Start heartbeat loop in background thread"""
        import threading
        
        def heartbeat_loop():
            while True:
                try:
                    time.sleep(interval)
                    self.send_heartbeat()
                except Exception as e:
                    logger.error(f"❌ Heartbeat loop error: {e}")
        
        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        
        logger.info(f"💓 Heartbeat loop started for {self.worker_type} worker (interval: {interval}s)")

    def deregister_worker(self):
        """Deregister worker on shutdown"""
        if not self.supabase or not self.registration_id:
            return
            
        try:
            self.supabase.table('worker_registry').update({
                'status': 'inactive',
                'last_heartbeat': 'now()'
            }).eq('id', self.registration_id).execute()
            
            logger.info(f"✅ Worker deregistered: {self.worker_type}")
            
        except Exception as e:
            logger.error(f"❌ Worker deregistration error: {e}")

# Usage in worker files:

def setup_worker_registration(worker_type: str, worker_port: int) -> Optional[WorkerRegistration]:
    """Setup worker registration for a worker"""
    try:
        registration = WorkerRegistration(worker_type, worker_port)
        
        # Register worker
        if registration.register_worker():
            # Start heartbeat
            registration.start_heartbeat_loop()
            return registration
        else:
            logger.warning(f"⚠️ {worker_type} worker registration failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Worker registration setup failed: {e}")
        return None

# Example usage in chat_worker.py:
"""
# Add this to chat_worker.py __init__ method:
from worker_registration import setup_worker_registration
import atexit

class ChatWorker:
    def __init__(self):
        # ... existing init code ...
        
        # Setup worker registration
        self.registration = setup_worker_registration('chat', 7861)
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def cleanup(self):
        if hasattr(self, 'registration') and self.registration:
            self.registration.deregister_worker()
"""