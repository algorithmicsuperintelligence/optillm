"""
Health checking functionality for providers.
"""
import time
import logging
import threading
from typing import List

logger = logging.getLogger(__name__)

class HealthChecker:
    """Background health checker for providers"""
    
    def __init__(self, providers: List, enabled: bool = True, 
                 interval: int = 30, timeout: int = 5):
        self.providers = providers
        self.enabled = enabled
        self.interval = interval
        self.timeout = timeout
        self.running = False
        self.thread = None
    
    def start(self):
        """Start health checking in background"""
        if not self.enabled:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._check_loop, daemon=True)
        self.thread.start()
        logger.info(f"Health checker started (interval: {self.interval}s)")
    
    def stop(self):
        """Stop health checking"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _check_loop(self):
        """Main health check loop"""
        while self.running:
            for provider in self.providers:
                self._check_provider(provider)
            time.sleep(self.interval)
    
    def _check_provider(self, provider):
        """Check health of a single provider"""
        try:
            # Simple health check - try to get models
            # This creates a minimal request to verify the endpoint is responsive
            provider.client.models.list()
            
            # Mark as healthy
            if not provider.is_healthy:
                logger.info(f"Provider {provider.name} is now healthy")
            provider.is_healthy = True
            provider.last_error = None
            
        except Exception as e:
            # Mark as unhealthy
            if provider.is_healthy:
                logger.warning(f"Provider {provider.name} failed health check: {e}")
            provider.is_healthy = False
            provider.last_error = str(e)