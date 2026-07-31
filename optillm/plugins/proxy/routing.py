"""
Routing strategies for load balancing across providers.
"""
import random
import logging
from typing import List, Optional
from abc import ABC, abstractmethod

# Configure logging for this module
logger = logging.getLogger(__name__)
# Ensure we show debug messages
logging.basicConfig()
logger.setLevel(logging.DEBUG)

class Router(ABC):
    """Abstract base class for routing strategies"""
    
    @abstractmethod
    def select(self, providers: List) -> Optional:
        """Select a provider from the list"""
        pass

class RoundRobinRouter(Router):
    """Round-robin routing strategy"""
    
    def __init__(self, providers: List):
        self.all_providers = providers
        self.index = 0
    
    def select(self, providers: List) -> Optional:
        if not providers:
            logger.debug("Round-robin: No providers available")
            return None
        
        # If only one provider available, return it
        if len(providers) == 1:
            logger.debug(f"Round-robin: Only one provider: {providers[0].name}")
            return providers[0]
        
        logger.debug(f"Round-robin: Starting selection, index={self.index}, providers={[p.name for p in providers]}")
        
        # Find next available provider in round-robin fashion
        attempts = 0
        while attempts < len(self.all_providers):
            # Get provider at current index from all providers
            current_provider = self.all_providers[self.index % len(self.all_providers)]
            next_index = (self.index + 1) % len(self.all_providers)
            
            logger.debug(f"Round-robin: Checking provider {current_provider.name} at index {self.index}")
            
            # Update index for next call
            self.index = next_index
            
            # If this provider is in the available list, use it
            if current_provider in providers:
                logger.debug(f"Round-robin: Selected provider {current_provider.name}")
                return current_provider
            
            attempts += 1
        
        # If we've cycled through all providers, just return first available
        logger.debug(f"Round-robin: Fallback to first available: {providers[0].name}")
        return providers[0]

class WeightedRouter(Router):
    """Weighted random routing strategy"""
    
    def __init__(self, providers: List):
        self.providers = providers
    
    def select(self, providers: List) -> Optional:
        if not providers:
            return None
        
        # Calculate weights
        weights = [p.weight for p in providers]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(providers)
        
        # Weighted random selection
        rand = random.uniform(0, total_weight)
        cumulative = 0
        
        for provider, weight in zip(providers, weights):
            cumulative += weight
            if rand <= cumulative:
                return provider
        
        return providers[-1]

class FailoverRouter(Router):
    """Failover routing - always use first healthy provider"""
    
    def __init__(self, providers: List):
        self.providers = providers
    
    def select(self, providers: List) -> Optional:
        # Return first available provider
        return providers[0] if providers else None

class RouterFactory:
    """Factory for creating routers"""
    
    @staticmethod
    def create(strategy: str, providers: List) -> Router:
        strategies = {
            'round_robin': RoundRobinRouter,
            'weighted': WeightedRouter,
            'failover': FailoverRouter
        }
        
        router_class = strategies.get(strategy, RoundRobinRouter)
        return router_class(providers)