#!/usr/bin/env python3
"""
Log Management Script for OurVidz Workers
Provides utilities for viewing, rotating, and managing server logs
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

class LogManager:
    def __init__(self):
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(exist_ok=True)
        
    def list_logs(self):
        """List all available log files"""
        print("📋 Available Log Files:")
        print("-" * 50)
        
        for log_file in self.logs_dir.glob('*.log*'):
            size = log_file.stat().st_size / 1024  # KB
            modified = datetime.fromtimestamp(log_file.stat().st_mtime)
            print(f"📄 {log_file.name}")
            print(f"   Size: {size:.1f} KB")
            print(f"   Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
    
    def view_log(self, log_name, lines=50, follow=False):
        """View log file contents"""
        log_path = self.logs_dir / log_name
        
        if not log_path.exists():
            print(f"❌ Log file not found: {log_name}")
            return
        
        if follow:
            # Follow log in real-time
            try:
                subprocess.run(['tail', '-f', '-n', str(lines), str(log_path)])
            except KeyboardInterrupt:
                print("\n👋 Stopped following log")
        else:
            # Show last N lines
            try:
                result = subprocess.run(['tail', '-n', str(lines), str(log_path)], 
                                      capture_output=True, text=True)
                print(f"📄 Last {lines} lines of {log_name}:")
                print("-" * 50)
                print(result.stdout)
            except Exception as e:
                print(f"❌ Error reading log: {e}")
    
    def search_logs(self, pattern, log_name=None):
        """Search logs for specific patterns"""
        if log_name:
            log_files = [self.logs_dir / log_name]
        else:
            log_files = list(self.logs_dir.glob('*.log'))
        
        print(f"🔍 Searching for '{pattern}' in logs...")
        print("-" * 50)
        
        for log_file in log_files:
            if not log_file.exists():
                continue
                
            try:
                result = subprocess.run(['grep', '-i', pattern, str(log_file)], 
                                      capture_output=True, text=True)
                if result.stdout:
                    print(f"📄 Found in {log_file.name}:")
                    print(result.stdout)
                    print("-" * 30)
            except Exception as e:
                print(f"❌ Error searching {log_file.name}: {e}")
    
    def rotate_logs(self):
        """Manually rotate log files"""
        print("🔄 Rotating log files...")
        
        for log_file in self.logs_dir.glob('*.log'):
            if log_file.stat().st_size > 0:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f"{log_file.stem}_{timestamp}.log"
                backup_path = self.logs_dir / backup_name
                
                try:
                    log_file.rename(backup_path)
                    print(f"✅ Rotated {log_file.name} → {backup_name}")
                except Exception as e:
                    print(f"❌ Failed to rotate {log_file.name}: {e}")
    
    def cleanup_old_logs(self, days=7):
        """Clean up log files older than specified days"""
        print(f"🧹 Cleaning up logs older than {days} days...")
        
        cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
        removed_count = 0
        
        for log_file in self.logs_dir.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    print(f"🗑️ Removed old log: {log_file.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Failed to remove {log_file.name}: {e}")
        
        print(f"✅ Cleaned up {removed_count} old log files")
    
    def get_log_stats(self):
        """Get statistics about log files"""
        print("📊 Log Statistics:")
        print("-" * 50)
        
        total_size = 0
        file_count = 0
        
        for log_file in self.logs_dir.glob('*.log*'):
            size = log_file.stat().st_size
            total_size += size
            file_count += 1
            
            size_mb = size / (1024 * 1024)
            print(f"📄 {log_file.name}: {size_mb:.2f} MB")
        
        total_mb = total_size / (1024 * 1024)
        print(f"\n📊 Total: {file_count} files, {total_mb:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Log Management for OurVidz Workers')
    parser.add_argument('action', choices=['list', 'view', 'search', 'rotate', 'cleanup', 'stats'],
                       help='Action to perform')
    parser.add_argument('--log', '-l', help='Log file name (for view/search)')
    parser.add_argument('--lines', '-n', type=int, default=50, help='Number of lines to show')
    parser.add_argument('--follow', '-f', action='store_true', help='Follow log in real-time')
    parser.add_argument('--pattern', '-p', help='Search pattern')
    parser.add_argument('--days', '-d', type=int, default=7, help='Days to keep logs')
    
    args = parser.parse_args()
    
    log_manager = LogManager()
    
    if args.action == 'list':
        log_manager.list_logs()
    elif args.action == 'view':
        if not args.log:
            print("❌ Please specify log file with --log")
            return
        log_manager.view_log(args.log, args.lines, args.follow)
    elif args.action == 'search':
        if not args.pattern:
            print("❌ Please specify search pattern with --pattern")
            return
        log_manager.search_logs(args.pattern, args.log)
    elif args.action == 'rotate':
        log_manager.rotate_logs()
    elif args.action == 'cleanup':
        log_manager.cleanup_old_logs(args.days)
    elif args.action == 'stats':
        log_manager.get_log_stats()

if __name__ == '__main__':
    main()
