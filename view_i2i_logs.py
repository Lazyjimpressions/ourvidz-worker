#!/usr/bin/env python3
"""
I2I Log Viewer for SDXL Worker
Quick utility to view image-to-image specific logs and settings
"""

import os
import sys
import subprocess
from pathlib import Path

def view_i2i_logs(lines=50, follow=False):
    """View i2i-specific logs"""
    log_file = Path('logs/sdxl_worker_i2i.log')
    
    if not log_file.exists():
        print("❌ I2I log file not found. Run the SDXL worker first to generate logs.")
        return
    
    print("🖼️ SDXL Worker I2I Logs")
    print("=" * 50)
    
    if follow:
        try:
            subprocess.run(['tail', '-f', '-n', str(lines), str(log_file)])
        except KeyboardInterrupt:
            print("\n👋 Stopped following logs")
    else:
        try:
            result = subprocess.run(['tail', '-n', str(lines), str(log_file)], 
                                  capture_output=True, text=True)
            print(result.stdout)
        except Exception as e:
            print(f"❌ Error reading log: {e}")

def search_i2i_settings(pattern):
    """Search for specific i2i settings in logs"""
    log_file = Path('logs/sdxl_worker_i2i.log')
    
    if not log_file.exists():
        print("❌ I2I log file not found.")
        return
    
    print(f"🔍 Searching for '{pattern}' in I2I logs...")
    print("=" * 50)
    
    try:
        result = subprocess.run(['grep', '-i', pattern, str(log_file)], 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        else:
            print(f"No matches found for '{pattern}'")
    except Exception as e:
        print(f"❌ Error searching logs: {e}")

def show_i2i_stats():
    """Show i2i log statistics"""
    log_file = Path('logs/sdxl_worker_i2i.log')
    
    if not log_file.exists():
        print("❌ I2I log file not found.")
        return
    
    try:
        # Count different types of entries
        result = subprocess.run(['grep', '-c', 'IMAGE-TO-IMAGE SETTINGS LOG', str(log_file)], 
                              capture_output=True, text=True)
        settings_count = result.stdout.strip() or '0'
        
        result = subprocess.run(['grep', '-c', 'EFFECTIVE I2I GENERATION SETTINGS', str(log_file)], 
                              capture_output=True, text=True)
        generation_count = result.stdout.strip() or '0'
        
        result = subprocess.run(['grep', '-c', 'EXACT COPY', str(log_file)], 
                              capture_output=True, text=True)
        exact_copy_count = result.stdout.strip() or '0'
        
        result = subprocess.run(['grep', '-c', 'REFERENCE MODIFY', str(log_file)], 
                              capture_output=True, text=True)
        modify_count = result.stdout.strip() or '0'
        
        # Get file size
        size_mb = log_file.stat().st_size / (1024 * 1024)
        
        print("📊 I2I Log Statistics")
        print("=" * 30)
        print(f"📄 Log File: {log_file.name}")
        print(f"📏 Size: {size_mb:.2f} MB")
        print(f"🖼️ I2I Jobs Logged: {settings_count}")
        print(f"⚙️ Generation Settings Logged: {generation_count}")
        print(f"🎯 Exact Copy Jobs: {exact_copy_count}")
        print(f"🔧 Reference Modify Jobs: {modify_count}")
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python view_i2i_logs.py view [lines]     # View recent i2i logs")
        print("  python view_i2i_logs.py follow           # Follow i2i logs in real-time")
        print("  python view_i2i_logs.py search <pattern> # Search i2i logs")
        print("  python view_i2i_logs.py stats            # Show i2i log statistics")
        return
    
    command = sys.argv[1]
    
    if command == 'view':
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        view_i2i_logs(lines)
    elif command == 'follow':
        view_i2i_logs(follow=True)
    elif command == 'search':
        if len(sys.argv) < 3:
            print("❌ Please provide a search pattern")
            return
        pattern = sys.argv[2]
        search_i2i_settings(pattern)
    elif command == 'stats':
        show_i2i_stats()
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == '__main__':
    main()
