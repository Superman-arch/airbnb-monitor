#!/usr/bin/env python3
"""Service manager for Airbnb monitoring system orchestration."""

import os
import sys
import time
import subprocess
import requests
import json
import logging
from typing import Dict, Any, Optional
import psutil
import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceManager:
    """Manages all monitoring system services."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize service manager."""
        self.config_path = config_path
        self.services = {
            'vlm': {
                'name': 'VLM Server',
                'command': 'python3 vlm_pipeline.py',
                'port': 8080,
                'health_url': 'http://localhost:8080/health',
                'pid': None,
                'required': True
            },
            'inference': {
                'name': 'Roboflow Inference',
                'command': 'docker run -d --rm --name roboflow_inference -p 9001:9001 roboflow/roboflow-inference-server-cpu:latest',
                'port': 9001,
                'health_url': 'http://localhost:9001/',
                'pid': None,
                'required': False  # Optional, has fallback
            },
            'monitor': {
                'name': 'Airbnb Monitor',
                'command': 'python3 run_optimized.py',
                'port': 5000,
                'health_url': 'http://localhost:5000/test',
                'pid': None,
                'required': True
            }
        }
        self.load_config()
        
    def load_config(self):
        """Load configuration from settings.yaml."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check if services are enabled
            self.services['vlm']['required'] = config.get('vlm', {}).get('enabled', True)
            self.services['inference']['required'] = config.get('door_detection', {}).get('use_inference', False)
            
            logger.info(f"Configuration loaded: VLM={self.services['vlm']['required']}, Inference={self.services['inference']['required']}")
        except Exception as e:
            logger.warning(f"Could not load config: {e}, using defaults")
    
    def check_port(self, port: int) -> bool:
        """Check if a port is in use."""
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return True
        return False
    
    def kill_port(self, port: int):
        """Kill process using a port."""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        logger.info(f"Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                        proc.kill()
                        return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    def start_service(self, service_key: str) -> bool:
        """Start a single service."""
        service = self.services[service_key]
        
        if not service['required']:
            logger.info(f"{service['name']} is disabled in config")
            return True
        
        logger.info(f"Starting {service['name']}...")
        
        # Check if port is already in use
        if self.check_port(service['port']):
            logger.warning(f"Port {service['port']} is in use, cleaning up...")
            self.kill_port(service['port'])
            time.sleep(2)
        
        # Start the service
        try:
            if 'docker' in service['command']:
                # Special handling for Docker
                subprocess.run("docker stop roboflow_inference 2>/dev/null", shell=True)
                subprocess.run("docker rm roboflow_inference 2>/dev/null", shell=True)
                process = subprocess.Popen(service['command'], shell=True)
            else:
                # Start Python services
                log_file = f"logs/{service_key}.log"
                os.makedirs("logs", exist_ok=True)
                with open(log_file, 'w') as f:
                    process = subprocess.Popen(
                        service['command'].split(),
                        stdout=f,
                        stderr=subprocess.STDOUT
                    )
            
            service['pid'] = process.pid
            logger.info(f"{service['name']} started with PID: {process.pid}")
            
            # Wait for service to be ready
            if self.wait_for_service(service_key):
                logger.info(f"✓ {service['name']} is ready!")
                return True
            else:
                logger.error(f"✗ {service['name']} failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start {service['name']}: {e}")
            return False
    
    def wait_for_service(self, service_key: str, timeout: int = 30) -> bool:
        """Wait for a service to be ready."""
        service = self.services[service_key]
        
        logger.info(f"Waiting for {service['name']} to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(service['health_url'], timeout=1)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        
        return False
    
    def stop_service(self, service_key: str):
        """Stop a single service."""
        service = self.services[service_key]
        
        if service['pid']:
            try:
                os.kill(service['pid'], signal.SIGTERM)
                logger.info(f"Stopped {service['name']} (PID: {service['pid']})")
            except:
                pass
        
        # Clean up port
        if self.check_port(service['port']):
            self.kill_port(service['port'])
    
    def start_all(self):
        """Start all services in order."""
        logger.info("Starting all services...")
        
        # Start VLM first (if enabled)
        if self.services['vlm']['required']:
            if not self.start_service('vlm'):
                logger.error("VLM failed to start, but continuing...")
        
        # Start Inference server (if enabled)
        if self.services['inference']['required']:
            if not self.start_service('inference'):
                logger.warning("Inference server failed, will use fallback")
        
        # Start main monitor
        if not self.start_service('monitor'):
            logger.error("Monitor failed to start!")
            return False
        
        return True
    
    def stop_all(self):
        """Stop all services."""
        logger.info("Stopping all services...")
        
        for service_key in ['monitor', 'inference', 'vlm']:
            self.stop_service(service_key)
        
        # Clean up Docker
        subprocess.run("docker stop roboflow_inference 2>/dev/null", shell=True)
        subprocess.run("docker rm roboflow_inference 2>/dev/null", shell=True)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        status = {}
        
        for key, service in self.services.items():
            is_running = False
            try:
                response = requests.get(service['health_url'], timeout=1)
                is_running = response.status_code == 200
            except:
                pass
            
            status[key] = {
                'name': service['name'],
                'running': is_running,
                'port': service['port'],
                'required': service['required']
            }
        
        return status
    
    def print_status(self):
        """Print status of all services."""
        print("\n" + "="*50)
        print("Service Status")
        print("="*50)
        
        status = self.get_status()
        for key, info in status.items():
            symbol = "✓" if info['running'] else "✗"
            required = " (required)" if info['required'] else " (optional)"
            print(f"{symbol} {info['name']}: {'Running' if info['running'] else 'Not running'} on port {info['port']}{required}")
        
        # Get IP address
        import socket
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        print("\n" + "="*50)
        print("Access Points")
        print("="*50)
        print(f"Web Dashboard: http://{ip_address}:5000")
        print(f"              http://localhost:5000")
        print("="*50 + "\n")

def main():
    """Main entry point for service manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Airbnb Monitoring Service Manager')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'restart'],
                       help='Action to perform')
    args = parser.parse_args()
    
    manager = ServiceManager()
    
    if args.action == 'start':
        if manager.start_all():
            manager.print_status()
            print("All services started successfully!")
            print("Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping services...")
                manager.stop_all()
        else:
            print("Failed to start services")
            sys.exit(1)
            
    elif args.action == 'stop':
        manager.stop_all()
        print("All services stopped")
        
    elif args.action == 'status':
        manager.print_status()
        
    elif args.action == 'restart':
        manager.stop_all()
        time.sleep(2)
        if manager.start_all():
            manager.print_status()
        else:
            print("Failed to restart services")
            sys.exit(1)

if __name__ == '__main__':
    main()