"""
Proxy Plugin for OptiLLM - Load balancing and failover for LLM providers

This plugin provides intelligent request routing across multiple LLM providers
with health monitoring, failover, and support for wrapping other approaches.
"""
import logging
from typing import Tuple, Optional
from optillm.plugins.proxy.config import ProxyConfig
from optillm.plugins.proxy.client import ProxyClient
from optillm.plugins.proxy.approach_handler import ApproachHandler

SLUG = "proxy"
logger = logging.getLogger(__name__)

# Global proxy client cache to maintain state between requests
_proxy_client_cache = {}

def _apply_log_level(config: dict):
    """Apply monitoring.log_level from the proxy config, if set, to the proxy loggers."""
    log_level = config.get('monitoring', {}).get('log_level')
    if log_level:
        level = getattr(logging, str(log_level).upper(), None)
        if isinstance(level, int):
            logger.setLevel(level)
            logging.getLogger('optillm.plugins.proxy').setLevel(level)

def _request_messages(system_prompt: str, initial_query: str, messages: Optional[list]) -> list:
    """
    Use the original request messages when available so multi-turn structure,
    tool_calls and tool results are preserved. Otherwise rebuild them.
    """
    if messages:
        return messages
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_query}
    ]

def run(system_prompt: str, initial_query: str, client, model: str,
        request_config: dict = None, messages: list = None) -> Tuple[str, int]:
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
        messages: Original request messages, forwarded as-is for direct routing
    
    Returns:
        Tuple of (response_text, token_count)
    """
    try:
        # Load configuration
        config = ProxyConfig.load()
        _apply_log_level(config)
        
        if not config.get('providers'):
            logger.warning(f"No providers configured in {ProxyConfig._config_path}, falling back to original client")
            # Strip stream parameter to force complete response
            api_config = dict(request_config or {})
            api_config.pop('stream', None)

            response = client.chat.completions.create(
                model=model,
                messages=_request_messages(system_prompt, initial_query, messages),
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
        
        # Direct proxy execution. ProxyClient detects per-provider system
        # message support and merges the system prompt when needed.
        logger.info(f"Direct proxy routing for model: {model}")

        # Strip stream parameter to force complete response
        # server.py will handle converting to SSE streaming format if needed
        api_config = dict(request_config or {})
        api_config.pop('stream', None)

        response = proxy_client.chat.completions.create(
            model=model,
            messages=_request_messages(system_prompt, initial_query, messages),
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
            messages=_request_messages(system_prompt, initial_query, messages),
            **api_config
        )
        # Return full response dict to preserve all usage information
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
        return response_dict, 0