"""
ProxyClient implementation for load balancing across multiple LLM providers.
"""
import time
import logging
from typing import Dict, Optional
import threading
from openai import OpenAI, AzureOpenAI
from optillm.plugins.proxy.routing import RouterFactory
from optillm.plugins.proxy.health import HealthChecker

logger = logging.getLogger(__name__)

class Provider:
    """Wrapper for a provider configuration and client"""
    def __init__(self, config: Dict):
        self.name = config['name']
        self.base_url = config['base_url']
        self.api_key = config['api_key']
        self.weight = config.get('weight', 1)
        self.fallback_only = config.get('fallback_only', False)
        self.model_map = config.get('model_map', {})
        self._client = None
        self.is_healthy = True
        self.last_error = None
        self.latencies = []  # Track recent latencies
        
        # Per-provider concurrency control
        self.max_concurrent = config.get('max_concurrent', None)  # None means no limit
        if self.max_concurrent is not None:
            self._semaphore = threading.Semaphore(self.max_concurrent)
            logger.info(f"Provider {self.name} limited to {self.max_concurrent} concurrent requests")
        else:
            self._semaphore = None
        
    @property
    def client(self):
        """Lazy initialization of OpenAI client"""
        if not self._client:
            if 'azure' in self.base_url.lower():
                # Handle Azure OpenAI
                self._client = AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.base_url,
                    api_version="2024-02-01",
                    max_retries=0  # Disable client retries - we handle them
                )
            elif 'generativelanguage.googleapis.com' in self.base_url:
                # Google AI with standard OpenAI-compatible client
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=0  # Disable client retries - we handle them
                )
            else:
                # Standard OpenAI-compatible client
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=0  # Disable client retries - we handle them
                )
        return self._client
    
    def map_model(self, model: str) -> str:
        """Map requested model to provider-specific name"""
        return self.model_map.get(model, model)
    
    def track_latency(self, latency: float):
        """Track request latency"""
        self.latencies.append(latency)
        if len(self.latencies) > 10:
            self.latencies.pop(0)
    
    def avg_latency(self) -> float:
        """Get average latency"""
        if not self.latencies:
            return 0
        return sum(self.latencies) / len(self.latencies)
    
    def acquire_slot(self, timeout: Optional[float] = None) -> bool:
        """
        Try to acquire a slot for this provider.
        Returns True if acquired, False if timeout or no limit.
        """
        if self._semaphore is None:
            return True  # No limit, always available
        
        return self._semaphore.acquire(blocking=True, timeout=timeout)
    
    def release_slot(self):
        """Release a slot for this provider."""
        if self._semaphore is not None:
            self._semaphore.release()
    
    def available_slots(self) -> Optional[int]:
        """Get number of available slots, None if unlimited."""
        if self._semaphore is None:
            return None
        # Note: _value is internal but there's no public method to check availability
        return self._semaphore._value

class ProxyClient:
    """OpenAI-compatible client that proxies to multiple providers"""
    
    def __init__(self, config: Dict, fallback_client=None):
        self.config = config
        self.fallback_client = fallback_client
        
        # Initialize providers
        self.providers = [
            Provider(p) for p in config.get('providers', [])
        ]
        
        # Filter out fallback-only providers for normal routing
        self.active_providers = [
            p for p in self.providers if not p.fallback_only
        ]
        
        self.fallback_providers = [
            p for p in self.providers if p.fallback_only
        ]
        
        # Initialize router
        strategy = config.get('routing', {}).get('strategy', 'round_robin')
        self.router = RouterFactory.create(strategy, self.active_providers)
        
        # Initialize health checker
        health_config = config.get('routing', {}).get('health_check', {})
        self.health_checker = HealthChecker(
            providers=self.providers,
            enabled=health_config.get('enabled', True),
            interval=health_config.get('interval', 30),
            timeout=health_config.get('timeout', 5)
        )
        
        # Start health checking
        self.health_checker.start()
        
        # Timeout settings
        timeout_config = config.get('timeouts', {})
        self.request_timeout = timeout_config.get('request', 30)  # Default 30 seconds
        self.connect_timeout = timeout_config.get('connect', 5)   # Default 5 seconds
        
        # Queue management settings
        queue_config = config.get('queue', {})
        self.max_concurrent_requests = queue_config.get('max_concurrent', 100)
        self.queue_timeout = queue_config.get('timeout', 60)  # Max time in queue
        self._request_semaphore = threading.Semaphore(self.max_concurrent_requests)
        
        # Monitoring settings
        monitoring = config.get('monitoring', {})
        self.track_latency = monitoring.get('track_latency', True)
        self.track_errors = monitoring.get('track_errors', True)
        
        # Create chat namespace
        self.chat = self._Chat(self)
    
    class _Chat:
        def __init__(self, proxy_client):
            self.proxy_client = proxy_client
            self.completions = proxy_client._Completions(proxy_client)
    
    class _Completions:
        def __init__(self, proxy_client):
            self.proxy_client = proxy_client
            self._system_message_support_cache = {}
        
        def _filter_kwargs(self, kwargs: dict) -> dict:
            """Filter out OptiLLM-specific parameters that shouldn't be sent to providers"""
            optillm_params = {
                'optillm_approach', 'proxy_wrap', 'wrapped_approach', 'wrap',
                'mcts_simulations', 'mcts_exploration', 'mcts_depth',
                'best_of_n', 'rstar_max_depth', 'rstar_num_rollouts', 'rstar_c'
            }
            return {k: v for k, v in kwargs.items() if k not in optillm_params}
        
        def _test_system_message_support(self, provider, model: str) -> bool:
            """Test if a model supports system messages"""
            cache_key = f"{provider.name}:{model}"
            
            if cache_key in self._system_message_support_cache:
                return self._system_message_support_cache[cache_key]
            
            try:
                provider.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "test"},
                        {"role": "user", "content": "hi"}
                    ],
                    max_tokens=1,
                    temperature=0
                )
                self._system_message_support_cache[cache_key] = True
                return True
            except Exception as e:
                error_msg = str(e).lower()
                if any(pattern in error_msg for pattern in [
                    "developer instruction", "system message", "not enabled", "not supported"
                ]):
                    logger.info(f"Provider {provider.name} model {model} does not support system messages")
                    self._system_message_support_cache[cache_key] = False
                    return False
                # Other errors - assume it supports system messages
                self._system_message_support_cache[cache_key] = True
                return True
        
        def _format_messages_for_provider(self, provider, model: str, messages: list) -> list:
            """Format messages based on provider's system message support"""
            # Check if there's a system message
            has_system = any(msg.get("role") == "system" for msg in messages)
            
            if not has_system:
                return messages
            
            # Test system message support
            supports_system = self._test_system_message_support(provider, model)
            
            if supports_system:
                return messages
            
            # Merge system message into first user message
            formatted_messages = []
            system_content = None
            
            for msg in messages:
                if msg.get("role") == "system":
                    system_content = msg.get("content", "")
                elif msg.get("role") == "user":
                    if system_content:
                        # Merge system message with user message
                        formatted_messages.append({
                            "role": "user",
                            "content": f"Instructions: {system_content}\n\nUser: {msg.get('content', '')}"
                        })
                        system_content = None
                    else:
                        formatted_messages.append(msg)
                else:
                    formatted_messages.append(msg)
            
            return formatted_messages
        
        def _make_request_with_timeout(self, provider, request_kwargs):
            """Make a request with timeout handling"""
            # The OpenAI client now supports timeout natively
            try:
                response = provider.client.chat.completions.create(**request_kwargs)
                return response
            except Exception as e:
                # Check if it's a timeout error
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    raise TimeoutError(f"Request to {provider.name} timed out after {self.proxy_client.request_timeout}s")
                raise e
        
        def create(self, **kwargs):
            """Create completion with load balancing, failover, and timeout handling"""
            # Check queue capacity
            if not self.proxy_client._request_semaphore.acquire(blocking=True, timeout=self.proxy_client.queue_timeout):
                raise TimeoutError(f"Request queue timeout after {self.proxy_client.queue_timeout}s - server overloaded")
            
            try:
                model = kwargs.get('model', 'unknown')
                attempted_providers = set()
                errors = []
                
                # Get healthy providers
                healthy_providers = [
                    p for p in self.proxy_client.active_providers 
                    if p.is_healthy
                ]
                
                if not healthy_providers:
                    logger.warning("No healthy providers, trying fallback providers")
                    healthy_providers = self.proxy_client.fallback_providers
                
                # Try routing through healthy providers
                while healthy_providers:
                    available_providers = [p for p in healthy_providers if p not in attempted_providers]
                    if not available_providers:
                        break
                        
                    provider = self.proxy_client.router.select(available_providers)
                    logger.info(f"Router selected provider: {provider.name if provider else 'None'}")
                    
                    if not provider:
                        break
                        
                    attempted_providers.add(provider)
                    
                    # Try to acquire a slot for this provider (with reasonable timeout for queueing)
                    slot_timeout = 10.0  # Wait up to 10 seconds for provider to become available
                    if not provider.acquire_slot(timeout=slot_timeout):
                        logger.debug(f"Provider {provider.name} at max capacity, trying next provider")
                        errors.append((provider.name, "At max concurrent requests"))
                        continue
                    
                    try:
                        # Map model name if needed and filter out OptiLLM-specific parameters
                        request_kwargs = self._filter_kwargs(kwargs.copy())
                        mapped_model = provider.map_model(model)
                        request_kwargs['model'] = mapped_model
                        
                        # Format messages based on provider's system message support
                        if 'messages' in request_kwargs:
                            request_kwargs['messages'] = self._format_messages_for_provider(
                                provider, mapped_model, request_kwargs['messages']
                            )
                        
                        # Add timeout to client if supported
                        request_kwargs['timeout'] = self.proxy_client.request_timeout
                        
                        # Track timing
                        start_time = time.time()
                        
                        # Make request with timeout
                        logger.debug(f"Routing to {provider.name} with {self.proxy_client.request_timeout}s timeout")
                        response = self._make_request_with_timeout(provider, request_kwargs)
                        
                        # Track success
                        latency = time.time() - start_time
                        if self.proxy_client.track_latency:
                            provider.track_latency(latency)
                        
                        logger.info(f"Request succeeded via {provider.name} in {latency:.2f}s")
                        return response
                        
                    except TimeoutError as e:
                        logger.error(f"Provider {provider.name} timed out: {e}")
                        errors.append((provider.name, str(e)))
                        
                        # Mark provider as unhealthy on timeout
                        if self.proxy_client.track_errors:
                            provider.is_healthy = False
                            provider.last_error = f"Timeout: {str(e)}"
                        
                    except Exception as e:
                        logger.error(f"Provider {provider.name} failed: {e}")
                        errors.append((provider.name, str(e)))
                        
                        # Mark provider as unhealthy
                        if self.proxy_client.track_errors:
                            provider.is_healthy = False
                            provider.last_error = str(e)
                    
                    finally:
                        # Always release the provider slot
                        provider.release_slot()
                        logger.debug(f"Released slot for provider {provider.name}")
            
                # All providers failed, try fallback client
                if self.proxy_client.fallback_client:
                    logger.warning("All proxy providers failed, using fallback client")
                    try:
                        fallback_kwargs = self._filter_kwargs(kwargs.copy())
                        fallback_kwargs['timeout'] = self.proxy_client.request_timeout
                        return self.proxy_client.fallback_client.chat.completions.create(**fallback_kwargs)
                    except Exception as e:
                        errors.append(("fallback_client", str(e)))
                
                # Complete failure
                error_msg = f"All providers failed. Errors: {errors}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
            finally:
                # Release semaphore to allow next request
                self.proxy_client._request_semaphore.release()