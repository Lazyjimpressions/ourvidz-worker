#!/usr/bin/env python3
# worker.py - OurVidz GPU Worker for RunPod

import os
import json
import time
import torch
import requests
import subprocess
from PIL import Image
from pathlib import Path
from typing import Optional, List
import traceback

class OurVidzWorker:
    def __init__(self):
        """Initialize the OurVidz worker"""
        self.model_path = Path("/workspace/models")
        
        # Model instances (loaded on demand)
        self.wan_t2v_pipeline = None
        self.wan_i2v_pipeline = None
        self.mistral_model = None
        self.mistral_tokenizer = None
        
        # Environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
        self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        # Auto-shutdown tracking
        self.idle_start_time = None
        self.max_idle_minutes = 10
        
        # Validate environment
        self._validate_environment()
        
        print("🚀 OurVidz Worker initialized")
        self._log_system_info()

    def _validate_environment(self):
        """Validate required environment variables"""
        required_vars = [
            'SUPABASE_URL', 'SUPABASE_SERVICE_KEY',
            'UPSTASH_REDIS_REST_URL', 'UPSTASH_REDIS_REST_TOKEN'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing environment variables: {missing}")

    def _log_system_info(self):
        """Log system information"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        print(f"📁 Model path: {self.model_path}")
        print(f"🔗 Supabase: {self.supabase_url}")

    def _log_gpu_memory(self, context=""):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🔥 GPU Memory {context}- Used: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.1f}GB")

    def load_mistral(self):
        """Load Mistral 7B for prompt enhancement"""
        if self.mistral_model is None:
            print("📝 Loading Mistral 7B...")
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                mistral_path = self.model_path / "mistral"
                
                self.mistral_tokenizer = AutoTokenizer.from_pretrained(
                    mistral_path if mistral_path.exists() else "mistralai/Mistral-7B-Instruct-v0.2",
                    cache_dir=str(mistral_path),
                    local_files_only=mistral_path.exists()
                )
                
                self.mistral_model = AutoModelForCausalLM.from_pretrained(
                    mistral_path if mistral_path.exists() else "mistralai/Mistral-7B-Instruct-v0.2",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=str(mistral_path),
                    local_files_only=mistral_path.exists()
                )
                
                print("✅ Mistral 7B loaded")
                self._log_gpu_memory("after Mistral load")
                
            except Exception as e:
                print(f"❌ Failed to load Mistral 7B: {e}")
                raise

    def unload_mistral(self):
        """Free Mistral memory"""
        if self.mistral_model is not None:
            print("🗑️ Unloading Mistral 7B...")
            del self.mistral_model
            del self.mistral_tokenizer
            self.mistral_model = None
            self.mistral_tokenizer = None
            torch.cuda.empty_cache()
            print("✅ Mistral 7B unloaded")
            self._log_gpu_memory("after Mistral unload")

    def load_wan_t2v(self):
        """Load Wan 2.1 14B Text-to-Video"""
        if self.wan_t2v_pipeline is None:
            print("🎥 Loading Wan 2.1 14B Text-to-Video...")
            try:
                from diffusers import DiffusionPipeline
                
                wan_path = self.model_path / "wan_t2v"
                
                self.wan_t2v_pipeline = DiffusionPipeline.from_pretrained(
