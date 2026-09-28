"""
Configuration management for the proxy plugin.
Handles loading, validation, and environment variable interpolation.
"""
import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ProxyConfig:
    """Manages proxy configuration with caching and validation."""
    
    _cached_config: Optional[Dict[str, Any]] = None
    _config_path: Optional[Path] = None
    
    @classmethod
    def load(cls, path: str = None, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load and cache configuration.
        
        Args:
            path: Optional path to config file
            force_reload: Force reload even if cached
            
        Returns:
            Loaded and validated configuration dictionary
        """
        if cls._cached_config and not force_reload:
            return cls._cached_config
            
        if not path:
            # Priority order for config files. The bundled example_config.yaml
            # is deliberately not used: its sample providers would silently
            # receive traffic meant for the user's own endpoints.
            config_locations = [
                Path.home() / ".optillm" / "proxy_config.yaml",
                Path.home() / ".optillm" / "proxy_config.yml",
            ]
            
            for config_path in config_locations:
                if config_path.exists():
                    path = config_path
                    logger.info(f"Using config from: {path}")
                    break
            else:
                # No config found, create default
                path = config_locations[0]
                cls._create_default(path)
        
        cls._config_path = Path(path)
        
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Validate structure
            if not isinstance(config, dict):
                raise ValueError("Configuration must be a dictionary")
            
            # Interpolate environment variables
            config = cls._interpolate_env_vars(config)
            
            # Apply defaults and validate
            config = cls._apply_defaults(config)
            config = cls._validate_config(config)
            
            cls._cached_config = config
            logger.debug(f"Loaded config with {len(config.get('providers', []))} providers")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load proxy config from {path}: {e}")
            return cls._get_minimal_config()
    
    @classmethod
    def reload(cls) -> Dict[str, Any]:
        """Force reload configuration from disk."""
        return cls.load(force_reload=True)
    
    @staticmethod
    def _interpolate_env_vars(obj: Any) -> Any:
        """
        Recursively replace ${VAR} and ${VAR:-default} with environment values.
        
        Args:
            obj: Object to process (dict, list, str, or other)
            
        Returns:
            Processed object with environment variables replaced
        """
        if isinstance(obj, str):
            # Pattern for ${VAR} or ${VAR:-default}
            pattern = re.compile(r'\$\{([^}]+)\}')
            
            def replacer(match):
                var_expr = match.group(1)
                
                # Handle ${VAR:-default} syntax
                if ':-' in var_expr:
                    var_name, default = var_expr.split(':-', 1)
                    value = os.environ.get(var_name.strip(), default)
                else:
                    var_name = var_expr.strip()
                    value = os.environ.get(var_name)
                    
                    if value is None:
                        logger.warning(f"Environment variable ${{{var_name}}} not set")
                        return match.group(0)  # Keep original
                
                return value
            
            return pattern.sub(replacer, obj)
            
        elif isinstance(obj, dict):
            return {k: ProxyConfig._interpolate_env_vars(v) for k, v in obj.items()}
            
        elif isinstance(obj, list):
            return [ProxyConfig._interpolate_env_vars(item) for item in obj]
            
        return obj
    
    @staticmethod
    def _apply_defaults(config: Dict) -> Dict:
        """
        Apply sensible defaults to configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with defaults applied
        """
        # Ensure main sections exist
        config.setdefault('providers', [])
        config.setdefault('routing', {})
        config.setdefault('monitoring', {})
        config.setdefault('timeouts', {})
        config.setdefault('queue', {})
        
        # Routing defaults
        routing = config['routing']
        routing.setdefault('strategy', 'round_robin')
        routing.setdefault('health_check', {})
        
        health_check = routing['health_check']
        health_check.setdefault('enabled', True)
        health_check.setdefault('interval', 30)
        health_check.setdefault('timeout', 5)
        
        # Monitoring defaults
        monitoring = config['monitoring']
        monitoring.setdefault('track_latency', True)
        monitoring.setdefault('track_errors', True)
        
        # Timeout defaults
        timeouts = config['timeouts']
        timeouts.setdefault('request', 30)  # 30 seconds for requests
        timeouts.setdefault('connect', 5)   # 5 seconds for connection
        
        # Queue management defaults
        queue = config['queue']
        queue.setdefault('max_concurrent', 100)  # Max concurrent requests
        queue.setdefault('timeout', 60)  # Max time waiting in queue
        
        # Provider defaults
        for i, provider in enumerate(config['providers']):
            provider.setdefault('name', f"provider_{i}")
            provider.setdefault('weight', 1)
            provider.setdefault('fallback_only', False)
            provider.setdefault('model_map', {})
            # Per-provider concurrency limit (None means no limit)
            provider.setdefault('max_concurrent', None)
        
        return config
    
    @staticmethod
    def _validate_config(config: Dict) -> Dict:
        """
        Validate configuration structure and values.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validated configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate providers
        for provider in config.get('providers', []):
            if 'base_url' not in provider:
                raise ValueError(f"Provider {provider.get('name', 'unknown')} missing base_url")
            if 'api_key' not in provider:
                raise ValueError(f"Provider {provider.get('name', 'unknown')} missing api_key")
            
            # Validate weight
            if provider['weight'] <= 0:
                logger.warning(f"Provider {provider['name']} has invalid weight {provider['weight']}, setting to 1")
                provider['weight'] = 1
            
            # Validate max_concurrent if specified
            if provider.get('max_concurrent') is not None:
                if not isinstance(provider['max_concurrent'], int) or provider['max_concurrent'] <= 0:
                    logger.warning(f"Provider {provider['name']} has invalid max_concurrent {provider['max_concurrent']}, removing limit")
                    provider['max_concurrent'] = None
        
        # Validate routing strategy
        valid_strategies = ['weighted', 'round_robin', 'failover']
        strategy = config['routing']['strategy']
        if strategy not in valid_strategies:
            logger.warning(f"Invalid routing strategy '{strategy}', using 'round_robin'")
            config['routing']['strategy'] = 'round_robin'
        
        return config
    
    @staticmethod
    def _create_default(path: Path):
        """Create default configuration file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        default = """# OptiLLM Proxy Plugin Configuration
# 
# This is an auto-generated configuration file.
# Add your LLM provider endpoints and API keys below.
# 
# Environment variables are supported: ${VAR_NAME} or ${VAR_NAME:-default_value}

providers:
  # Example OpenAI provider (uncomment and configure)
  # - name: openai_primary
  #   base_url: https://api.openai.com/v1
  #   api_key: ${OPENAI_API_KEY}
  #   weight: 1

routing:
  strategy: round_robin  # Options: weighted, round_robin, failover
  health_check:
    enabled: true
    interval: 30  # seconds
    timeout: 5    # seconds

timeouts:
  request: 30     # Maximum time for a request (seconds)
  connect: 5      # Maximum time for connection (seconds)

queue:
  max_concurrent: 100  # Maximum concurrent requests
  timeout: 60          # Maximum time in queue (seconds)

monitoring:
  # log_level: DEBUG  # Overrides the server --log level for proxy logs
  track_latency: true
  track_errors: true

# See proxy/README.md for full documentation
"""
        path.write_text(default)
        logger.info(f"Created default proxy config at {path}")
        logger.info("Please configure your providers in this file")
    
    @staticmethod
    def _get_minimal_config() -> Dict:
        """Return minimal working config as fallback."""
        return {
            'providers': [],
            'routing': {
                'strategy': 'round_robin',
                'health_check': {'enabled': False}
            },
            'timeouts': {
                'request': 30,
                'connect': 5
            },
            'queue': {
                'max_concurrent': 100,
                'timeout': 60
            },
            'monitoring': {
                'track_latency': False,
                'track_errors': True
            }
        }