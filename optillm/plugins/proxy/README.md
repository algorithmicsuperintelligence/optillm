# OptiLLM Proxy Plugin

A sophisticated load balancing and failover plugin for OptiLLM that distributes requests across multiple LLM providers.

## Features

- 🔄 **Load Balancing**: Distribute requests across multiple providers using weighted, round-robin, or failover strategies
- 🏥 **Health Monitoring**: Automatic health checks with provider failover
- 🔌 **Universal Compatibility**: Works with any OptiLLM approach or plugin
- 🌍 **Environment Variables**: Secure configuration with environment variable support
- 📊 **Performance Tracking**: Monitor latency and errors per provider
- 🗺️ **Model Mapping**: Map model names to provider-specific deployments

## Installation

```bash
# Install OptiLLM via pip
pip install optillm

# Verify installation
optillm --version
```

## Quick Start

### 1. Create Configuration

Create `~/.optillm/proxy_config.yaml` (or point `OPTILLM_PROXY_CONFIG` at a config file elsewhere). If neither exists, an empty template is created at `~/.optillm/proxy_config.yaml` and requests go to the server's default client (`--base-url`) until you add providers.

```yaml
providers:
  - name: primary
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    weight: 2
    max_concurrent: 5  # Optional: limit this provider to 5 concurrent requests
    model_map:
      gpt-4: gpt-4-turbo-preview  # Optional: map model names
    
  - name: backup
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY_BACKUP}
    weight: 1
    max_concurrent: 2  # Optional: limit this provider to 2 concurrent requests

routing:
  strategy: weighted  # Options: weighted, round_robin, failover

timeouts:
  request: 30  # Maximum seconds to wait for a provider response
  connect: 5   # Maximum seconds to wait for connection

queue:
  max_concurrent: 100  # Maximum concurrent requests to prevent overload
  timeout: 60         # Maximum seconds a request can wait in queue
```

### 2. Start OptiLLM Server

```bash
# Option A: Use proxy as default for ALL requests (recommended)
optillm --approach proxy

# Option B: Start server normally (use model prefix or extra_body per request)
optillm

# With custom port
optillm --approach proxy --port 8000
```

With `--approach proxy`, `/v1/models` lists the models reported by your configured providers (plus any `model_map` aliases), so you don't need to set `--base-url` as well.

#### Local servers and agent clients

The proxy passes the original messages through unchanged, including `tools`, assistant `tool_calls` and `tool` results, so coding agents (e.g. Crush) work through it. For a local llama.cpp server:

```yaml
providers:
  - name: llamacpp
    base_url: http://localhost:8080/v1
    api_key: none

timeouts:
  request: 300  # local models on long agent prompts can take well over the 30s default
```

### 3. Usage Examples

#### Method 1: Using --approach proxy (Recommended)
```bash
# Start server with proxy as default approach
optillm --approach proxy

# Then make normal requests - proxy handles all routing automatically!
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

#### Method 2: Using Model Prefix (when server started without --approach proxy)
```bash
# Use "proxy-" prefix to activate the proxy plugin
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "proxy-gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

#### Method 3: Using extra_body (when server started without --approach proxy)
```bash
# Use extra_body parameter  
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "extra_body": {
      "optillm_approach": "proxy"
    }
  }'
```

Both methods will:
- Route to one of your configured providers
- Apply model mapping if configured  
- Handle failover automatically

#### Combined Approaches
```bash
# Apply BON sampling, then route through proxy
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bon&proxy-gpt-4",
    "messages": [{"role": "user", "content": "Generate ideas"}]
  }'
```

#### Proxy Wrapping Other Approaches
```bash
# Use proxy to wrap MOA approach
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Solve this problem"}],
    "extra_body": {
      "optillm_approach": "proxy",
      "proxy_wrap": "moa"
    }
  }'
```

## Configuration Reference

### Provider Configuration

Each provider supports the following options:

```yaml
providers:
  - name: provider_name           # Required: Unique identifier
    base_url: https://api.url/v1  # Required: API endpoint
    api_key: ${ENV_VAR}           # Required: API key (supports env vars)
    weight: 2                      # Optional: Weight for weighted routing (default: 1)
    fallback_only: false          # Optional: Use only when primary providers fail
    model_map:                    # Optional: Map model names
      gpt-4: gpt-4-deployment
      gpt-3.5-turbo: gpt-35-turbo
```

### Routing Strategies

```yaml
routing:
  strategy: weighted  # Options: weighted, round_robin, failover
  
  # Health check configuration
  health_check:
    enabled: true     # Enable/disable health checks
    interval: 30      # Seconds between checks
    timeout: 5        # Timeout for health check requests
```

### Timeout and Queue Management

Prevent request queue backup and handle slow/unresponsive backends:

```yaml
timeouts:
  request: 30  # Maximum seconds to wait for provider response (default: 30)
  connect: 5   # Maximum seconds for initial connection (default: 5)

queue:
  max_concurrent: 100  # Maximum concurrent requests (default: 100)
  timeout: 60         # Maximum seconds in queue before rejection (default: 60)
```

**How it works:**
- **Request Timeout**: Each request to a provider has a maximum time limit. If exceeded, the request is cancelled and the next provider is tried.
- **Queue Management**: Limits concurrent requests to prevent memory exhaustion. New requests wait up to `queue.timeout` seconds before being rejected.
- **Automatic Failover**: When a provider times out or returns a server error, it's marked unhealthy and the request automatically fails over to the next available provider. Rejected requests (4xx) don't mark a provider unhealthy. If every provider is unhealthy, the proxy still retries them before falling back to the server's default client.
- **Protection**: Prevents slow backends from causing queue buildup that can crash the proxy server.

### Per-Provider Concurrency Limits

Control the maximum number of concurrent requests each provider can handle:

```yaml
providers:
  - name: slow_server
    base_url: http://192.168.1.100:8080/v1
    api_key: dummy
    max_concurrent: 1  # This server can only handle 1 request at a time
    
  - name: fast_server
    base_url: https://api.fast.com/v1
    api_key: ${API_KEY}
    max_concurrent: 10  # This server can handle 10 concurrent requests
    
  - name: unlimited_server
    base_url: https://api.unlimited.com/v1
    api_key: ${API_KEY}
    # No max_concurrent means no limit for this provider
```

**Use Cases:**
- **Hardware-limited servers**: Set `max_concurrent: 1` for servers that can't handle parallel requests
- **Rate limiting**: Prevent overwhelming providers with too many concurrent requests
- **Resource management**: Balance load across providers with different capacities
- **Cost control**: Limit expensive providers while allowing more requests to cheaper ones

**Behavior:**
- If a provider is at max capacity, the proxy tries the next available provider
- Requests wait briefly (0.5s) for a slot before moving to the next provider
- Works with all routing strategies (weighted, round_robin, failover)

### Environment Variables

The configuration supports flexible environment variable interpolation:

```yaml
# Simple substitution
api_key: ${OPENAI_API_KEY}

# With default value
base_url: ${CUSTOM_ENDPOINT:-https://api.openai.com/v1}

# Nested variables
api_key: ${ENV_PREFIX}_API_KEY
```

## Advanced Usage

### Provider Priority

Control provider selection priority using weights:

```yaml
providers:
  - name: premium
    base_url: https://premium.api/v1
    api_key: ${PREMIUM_KEY}
    weight: 5  # Gets 5x more traffic
    
  - name: standard
    base_url: https://standard.api/v1
    api_key: ${STANDARD_KEY}
    weight: 1  # Baseline traffic
```

### Model-Specific Routing

The proxy automatically maps model names to provider-specific deployments:

```yaml
providers:
  - name: azure
    base_url: ${AZURE_ENDPOINT}
    api_key: ${AZURE_KEY}
    model_map:
      # Request model -> Provider deployment name
      gpt-4: gpt-4-deployment-001
      gpt-4-turbo: gpt-4-turbo-latest
      gpt-3.5-turbo: gpt-35-turbo-deployment
  
  - name: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    # No model_map needed - uses model names as-is
```

With this configuration and `proxy-gpt-4` model requests:
- Request for "proxy-gpt-4" → Azure uses "gpt-4-deployment-001", OpenAI uses "gpt-4"
- Request for "proxy-gpt-3.5-turbo" → Azure uses "gpt-35-turbo-deployment", OpenAI uses "gpt-3.5-turbo"

### Failover Configuration

Set up primary and backup providers:

```yaml
providers:
  # Primary providers (normal traffic)
  - name: primary_1
    base_url: https://api1.com/v1
    api_key: ${KEY_1}
    weight: 3
    
  - name: primary_2
    base_url: https://api2.com/v1
    api_key: ${KEY_2}
    weight: 2
    
  # Backup provider (only on failure)
  - name: emergency_backup
    base_url: https://backup.api/v1
    api_key: ${BACKUP_KEY}
    fallback_only: true  # Only used when all primary providers fail
```

## Monitoring and Debugging

### Logging

Proxy logs follow the server log level (`--log debug` or `OPTILLM_LOG=debug`). To use a different level for the proxy only, set it in the config:

```yaml
monitoring:
  log_level: DEBUG  # Optional. Options: DEBUG, INFO, WARNING, ERROR
  track_latency: true
  track_errors: true
```

### Health Status

The proxy automatically monitors provider health. Failed providers are:
1. Marked as unhealthy after errors
2. Excluded from routing
3. Periodically rechecked for recovery
4. Automatically restored when healthy

### Performance Metrics

When `track_latency` is enabled, the proxy logs:
- Request latency per provider
- Success/failure rates
- Provider selection patterns

## Troubleshooting

### Common Issues

#### 1. "No healthy providers available"
- Check your API keys are correctly set
- Verify base URLs are accessible
- Review health check logs for specific errors
- Ensure at least one provider is configured

#### 2. "Provider X constantly failing"
- Check provider-specific API limits
- Verify model names in model_map
- Test the provider's endpoint directly
- Review error logs for details

#### 3. "Proxy not wrapping approach"
- Ensure using correct extra_body format
- Verify approach/plugin name is correct
- Check that target approach is installed

### Debug Mode

Enable debug logging to see detailed routing decisions:

```bash
optillm --approach proxy --log debug
```

## Best Practices

1. **Multiple API Keys**: Use different API keys per provider for better rate limit distribution
2. **Weight Tuning**: Adjust weights based on provider performance and cost
3. **Health Intervals**: Balance between quick failure detection (short) and API overhead (long)
4. **Fallback Providers**: Always configure at least one fallback provider
5. **Environment Security**: Never commit API keys; always use environment variables

## Performance Tuning

### For High Throughput
```yaml
routing:
  strategy: weighted  # Better distribution than round_robin
  health_check:
    interval: 60     # Reduce health check frequency
    timeout: 10      # Allow longer timeout for stability
```

### For Low Latency
```yaml
routing:
  strategy: failover  # Always use fastest provider
  health_check:
    interval: 10      # Quick failure detection
    timeout: 2        # Fast timeout
```

### For Cost Optimization
```yaml
providers:
  - name: cheap_provider
    weight: 10  # Prefer cheaper provider
    
  - name: expensive_provider
    weight: 1   # Minimize usage
    fallback_only: true  # Or only use on failure
```

## Integration Examples

### With Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Can be any string when using proxy
)

# Method 1: Server started with --approach proxy (recommended)
# Just make normal requests - proxy handles everything!
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

# Method 2: Use proxy with model prefix
response = client.chat.completions.create(
    model="proxy-gpt-4",  # Use "proxy-" prefix
    messages=[{"role": "user", "content": "Hello"}]
)

# Method 3: Use extra_body
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "optillm_approach": "proxy"
    }
)

# Method 4: Proxy wrapping another approach
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "optillm_approach": "proxy",
        "proxy_wrap": "moa"
    }
)
```

### With LangChain
```python
from langchain.llms import OpenAI

# If server started with --approach proxy (recommended)
llm = OpenAI(
    openai_api_base="http://localhost:8000/v1",
    model_name="gpt-4"  # Proxy handles routing automatically
)

# Or use proxy with model prefix
llm = OpenAI(
    openai_api_base="http://localhost:8000/v1",
    model_name="proxy-gpt-4"  # Use "proxy-" prefix
)

response = llm("What is the meaning of life?")
```

## Supported Providers

The proxy works with any OpenAI-compatible API:

- ✅ OpenAI
- ✅ Azure OpenAI
- ✅ Anthropic (via LiteLLM)
- ✅ Google AI (via LiteLLM)
- ✅ Cohere (via LiteLLM)
- ✅ Together AI
- ✅ Anyscale
- ✅ Local models (Ollama, LM Studio, llama.cpp)
- ✅ Any OpenAI-compatible endpoint

## Configuration Examples

### Multi-Cloud Setup
```yaml
providers:
  - name: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    weight: 3
    
  - name: azure
    base_url: ${AZURE_ENDPOINT}
    api_key: ${AZURE_API_KEY}
    weight: 2
    model_map:
      gpt-4: gpt-4-deployment
      
  - name: together
    base_url: https://api.together.xyz/v1
    api_key: ${TOGETHER_API_KEY}
    weight: 1
```

### Local Development Setup
```yaml
providers:
  - name: local_primary
    base_url: http://localhost:8080/v1
    api_key: local
    weight: 1
    
  - name: local_backup
    base_url: http://localhost:8081/v1
    api_key: local
    weight: 1
    
routing:
  strategy: round_robin
  health_check:
    enabled: false  # Disable for local dev
```

### Production Setup
```yaml
providers:
  - name: prod_primary
    base_url: https://api.openai.com/v1
    api_key: ${PROD_OPENAI_KEY_1}
    weight: 5
    
  - name: prod_secondary
    base_url: https://api.openai.com/v1
    api_key: ${PROD_OPENAI_KEY_2}
    weight: 3
    
  - name: prod_fallback
    base_url: ${FALLBACK_ENDPOINT}
    api_key: ${FALLBACK_KEY}
    weight: 1
    fallback_only: true

routing:
  strategy: weighted
  health_check:
    enabled: true
    interval: 30
    timeout: 5

monitoring:
  log_level: WARNING
  track_latency: true
  track_errors: true
```

## Contributing

To add new routing strategies or features:

1. Implement new strategy in `routing.py`
2. Add strategy to RouterFactory
3. Update documentation
4. Add tests

## License

Part of OptiLLM - see main project license.