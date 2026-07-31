import os
"""
Proxy Plugin for OptiLLM - Load balancing and failover for LLM providers

This plugin provides intelligent request routing across multiple LLM providers
with health monitoring, failover, and support for wrapping other approaches.
"""
import logging
import threading
from typing import Tuple, Dict
from optillm.plugins.proxy.config import ProxyConfig
from optillm.plugins.proxy.client import ProxyClient
from optillm.plugins.proxy.approach_handler import ApproachHandler

SLUG = "proxy"
logger = logging.getLogger(__name__)

# Configure logging based on environment
log_level = os.environ.get('OPTILLM_LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level))

# Global proxy client cache to maintain state between requests
_proxy_client_cache = {}

# Global cache for system message support per provider-model combination
_system_message_support_cache: Dict[str, bool] = {}
_cache_lock = threading.RLock()

def _test_system_message_support(proxy_client, model: str) -> bool:
    """
    Test if a model supports system messages by making a minimal test request.
    Returns True if supported, False otherwise.
    """
    try:
        # Try a minimal system message request
        proxy_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "test"},
                {"role": "user", "content": "hi"}
            ],
            max_tokens=1,  # Minimal token generation
            temperature=0
        )
        return True
    except Exception as e:
        error_msg = str(e).lower()
        # Check for known system message rejection patterns
        if any(pattern in error_msg for pattern in [
            "developer instruction", 
            "system message", 
            "not enabled",
            "not supported"
        ]):
            logger.info(f"Model {model} does not support system messages: {str(e)[:100]}")
            return False
        else:
            # If it's a different error, assume system messages are supported
            # but something else went wrong (rate limit, timeout, etc.)
            logger.debug(f"System message test failed for {model}, assuming supported: {str(e)[:100]}")
            return True

def _get_system_message_support(proxy_client, model: str) -> bool:
    """
    Get cached system message support status, testing if not cached.
    Thread-safe with locking.
    """
    # Create a unique cache key based on model and base_url
    cache_key = f"{getattr(proxy_client, '_base_identifier', 'default')}:{model}"
    
    with _cache_lock:
        if cache_key not in _system_message_support_cache:
            logger.debug(f"Testing system message support for {model}")
            _system_message_support_cache[cache_key] = _test_system_message_support(proxy_client, model)
        
        return _system_message_support_cache[cache_key]

def _format_messages_for_model(system_prompt: str, initial_query: str, 
                              supports_system_messages: bool) -> list:
    """
    Format messages based on whether the model supports system messages.
    """
    if supports_system_messages:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_query}
        ]
    else:
        # Merge system prompt into user message
        if system_prompt.strip():
            combined_message = f"{system_prompt}\n\nUser: {initial_query}"
        else:
            combined_message = initial_query
        
        return [{"role": "user", "content": combined_message}]

def run(system_prompt: str, initial_query: str, client, model: str, 
        request_config: dict = None) -> Tuple[str, int]:
    """
    Main proxy plugin entry point.
    
    Supports three usage modes:
    1. Standalone proxy: model="proxy-gpt-4"
    2. Wrapping approach: extra_body={"optillm_approach": "proxy", "proxy_wrap": "moa"}
    3. Combined approach: model="bon&proxy-gpt-4"
    
    Args:
        system_prompt: System message for the LLM
        initial_query: User's query
        client: Original OpenAI client (used as fallback)
        model: Model identifier
        request_config: Additional request configuration
    
    Returns:
        Tuple of (response_text, token_count)
    """
    try:
        # Load configuration
        config = ProxyConfig.load()
        
        if not config.get('providers'):
            logger.warning("No providers configured, falling back to original client")
            # Strip stream parameter to force complete response
            api_config = dict(request_config or {})
            api_config.pop('stream', None)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_query}
                ],
                **api_config
            )
            # Return full response dict to preserve all usage information
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            return response_dict, 0
        
        # Create or reuse proxy client to maintain state (important for round-robin)
        config_key = str(config)  # Simple config-based cache key
        if config_key not in _proxy_client_cache:
            logger.debug("Creating new proxy client instance")
            _proxy_client_cache[config_key] = ProxyClient(
                config=config,
                fallback_client=client
            )
        else:
            logger.debug("Reusing existing proxy client instance")
        
        proxy_client = _proxy_client_cache[config_key]
        
        # Check for wrapped approach in extra_body (recommended method)
        wrapped_approach = None
        if request_config:
            # Support multiple field names for flexibility
            wrapped_approach = (
                request_config.get('proxy_wrap') or 
                request_config.get('wrapped_approach') or
                request_config.get('wrap')
            )
        
        if wrapped_approach:
            logger.info(f"Proxy wrapping approach/plugin: {wrapped_approach}")
            handler = ApproachHandler()
            result = handler.handle(
                wrapped_approach,
                system_prompt,
                initial_query,
                proxy_client,  # Use proxy client instead of original
                model,
                request_config
            )
            
            if result is not None:
                return result
            else:
                logger.warning(f"Approach/plugin '{wrapped_approach}' not found, using direct proxy")
        
        # Check if model contains an approach pattern (backward compatibility)
        if '-' in model and not wrapped_approach:
            parts = model.split('-', 1)
            potential_approach = parts[0]
            actual_model = parts[1] if len(parts) > 1 else model
            
            # Try to handle as approach or plugin
            handler = ApproachHandler()
            result = handler.handle(
                potential_approach,
                system_prompt,
                initial_query,
                proxy_client,
                actual_model,
                request_config
            )
            
            if result is not None:
                logger.info(f"Proxy routing approach/plugin: {potential_approach}")
                return result
        
        # Direct proxy execution with dynamic system message support detection
        logger.info(f"Direct proxy routing for model: {model}")
        
        # Test and cache system message support for this model
        supports_system_messages = _get_system_message_support(proxy_client, model)
        
        # Format messages based on system message support
        messages = _format_messages_for_model(system_prompt, initial_query, supports_system_messages)
        
        if not supports_system_messages:
            logger.info(f"Using fallback message formatting for {model} (no system message support)")
        
        # Strip stream parameter to force complete response
        # server.py will handle converting to SSE streaming format if needed
        api_config = dict(request_config or {})
        api_config.pop('stream', None)

        response = proxy_client.chat.completions.create(
            model=model,
            messages=messages,
            **api_config
        )

        # Return full response dict to preserve all usage information
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
        return response_dict, 0
        
    except Exception as e:
        logger.error(f"Proxy plugin error: {e}", exc_info=True)
        # Fallback to original client
        logger.info("Falling back to original client")
        # Strip stream parameter to force complete response
        api_config = dict(request_config or {})
        api_config.pop('stream', None)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_query}
            ],
            **api_config
        )
        # Return full response dict to preserve all usage information
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
        return response_dict, 0