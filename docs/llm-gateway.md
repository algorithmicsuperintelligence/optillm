# Using OptiLLM as an LLM gateway

OptiLLM's proxy mode gives you one self-hosted, OpenAI-compatible endpoint in front of any number of providers, with load balancing, failover, and health checks. It is free and Apache-2.0 licensed, and you pay your providers directly.

OptiLLM combines routing and inference optimization in the same service. The same request that is routed across providers can also run through [20+ state-of-the-art reasoning techniques](../README.md#techniques), so your gateway can improve answers as well as deliver them.

## Quick start

Create `~/.optillm/proxy_config.yaml`:

```yaml
providers:
  - name: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    weight: 3

  - name: together
    base_url: https://api.together.xyz/v1
    api_key: ${TOGETHER_API_KEY}
    weight: 1
    model_map:
      gpt-4o-mini: meta-llama/Llama-3.3-70B-Instruct-Turbo

  - name: local
    base_url: http://localhost:8080/v1
    api_key: none
    max_concurrent: 2

routing:
  strategy: weighted  # or round_robin, failover
  health_check:
    enabled: true
    interval: 30
```

Start the server with the proxy as the default approach and protect it with an API key:

```bash
optillm --approach proxy --optillm-api-key "$GATEWAY_KEY"
```

Point any OpenAI client at it:

```python
from openai import OpenAI

client = OpenAI(api_key=GATEWAY_KEY, base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
```

If a provider fails, the request moves to the next one, and then to the server's default client (`--base-url`) if one is available.

## Adding inference optimization

Start the server without `--approach proxy` and select the proxy per request, on its own or wrapped around a technique:

```python
# Route only
client.chat.completions.create(model="proxy-gpt-4o-mini", messages=messages)

# Run Mixture of Agents, sending its calls through the proxy
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    extra_body={"optillm_approach": "proxy", "proxy_wrap": "moa"},
)
```

See the [proxy plugin documentation](../optillm/plugins/proxy/README.md) for the full configuration reference.
