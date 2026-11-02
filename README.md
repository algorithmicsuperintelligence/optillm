# OptiLLM

<p align="center">
  <img src="https://raw.githubusercontent.com/algorithmicsuperintelligence/optillm/main/optillm-logo.png" alt="OptiLLM Logo" width="400" />
</p>

<p align="center">
  <strong>2-10x accuracy improvements on reasoning tasks with zero training</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/optillm/"><img src="https://img.shields.io/pypi/v/optillm" alt="PyPI version"></a>
  <a href="https://pepy.tech/projects/optillm"><img src="https://static.pepy.tech/personalized-badge/optillm?period=monthly&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=GREEN&left_text=downloads/month" alt="PyPI Downloads"></a>
  <a href="https://github.com/algorithmicsuperintelligence/optillm/blob/main/LICENSE"><img src="https://img.shields.io/github/license/algorithmicsuperintelligence/optillm" alt="License"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/codelion/optillm">HuggingFace Space</a> |
  <a href="https://colab.research.google.com/drive/1SpuUb8d9xAoTh32M-9wJsB50AOH54EaH?usp=sharing">Colab Demo</a> |
  <a href="https://github.com/algorithmicsuperintelligence/optillm/discussions">Discussions</a>
</p>

**OptiLLM** is an OpenAI API-compatible optimizing inference proxy that implements 20+ state-of-the-art techniques to dramatically improve LLM accuracy and performance on reasoning tasks, without requiring any model training or fine-tuning.

It is possible to beat the frontier models using these techniques across diverse tasks by doing additional compute at inference time. A good example of how to combine such techniques is the [CePO approach](optillm/cepo) from Cerebras.

## Key features

- **Instant improvements**: 2-10x better accuracy on math, coding, and logical reasoning
- **Drop-in replacement**: Works with any OpenAI-compatible API endpoint
- **20+ optimization techniques**: From simple best-of-N to advanced MCTS and planning
- **Zero training required**: Just proxy your existing API calls through OptiLLM
- **Production ready**: Used in production by companies and researchers worldwide
- **Multi-provider**: Supports OpenAI, Anthropic, Google, Cerebras, and 100+ models via LiteLLM

## Quick start

```bash
pip install optillm
export OPENAI_API_KEY="your-key-here"
optillm
```

Point any OpenAI client at the proxy and prefix the model name with a technique:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="moa-gpt-4o-mini",  # This gives you GPT-4o performance from GPT-4o-mini
    messages=[{"role": "user", "content": "Solve: If 2x + 3 = 7, what is x?"}],
)
```

**Before OptiLLM**: "x = 1"  
**After OptiLLM**: "Let me work through this step by step: 2x + 3 = 7, so 2x = 4, therefore x = 2"

## Results

| Technique | Base model | Improvement | Benchmark |
|-----------|------------|-------------|-----------|
| MARS | Gemini 2.5 Flash Lite | +30.0 points | AIME 2025 (43.3 to 73.3) |
| CePO | Llama 3.3 70B | +18.6 points | Math-L5 (51.0 to 69.6) |
| LongCePO | Llama 3.3 70B | +13.6 points | InfiniteBench (58.0 to 71.6) |
| AutoThink | DeepSeek-R1-1.5B | +9.34 points | GPQA-Diamond (21.72 to 31.06) |
| PlanSearch | GPT-4o-mini | +20% pass@5 | LiveCodeBench |
| MOA | GPT-4o-mini | Matches GPT-4 | Arena-Hard-Auto |

See [benchmark results](#benchmark-results) for the full tables.

## Installation

**pip**

```bash
pip install optillm
```

**Docker**

```bash
docker pull ghcr.io/algorithmicsuperintelligence/optillm:latest
docker run -p 8000:8000 ghcr.io/algorithmicsuperintelligence/optillm:latest
```

| Image tag | Contents |
|-----------|----------|
| `latest` | All dependencies for local inference and plugins |
| `latest-proxy` | Lightweight image without local inference |
| `latest-offline` | Self-contained image with pre-downloaded models (spaCy) for fully offline operation |

To build and run from the repository, set your keys in `docker-compose.yaml` or a `.env` file and run `docker compose up -d`.

**From source**

```bash
git clone https://github.com/algorithmicsuperintelligence/optillm.git
cd optillm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python optillm.py
```

## Providers

Set the environment variable for your provider and the proxy picks the matching client.

| Provider | Required environment variables | Notes |
|----------|--------------------------------|-------|
| OptiLLM | `OPTILLM_API_KEY` | Built-in local inference server; supports logprobs and decoding techniques such as `cot_decoding` and `entropy_decoding` |
| OpenAI | `OPENAI_API_KEY` | Also works with any OpenAI-compatible endpoint (e.g. OpenRouter) by setting `--base-url` |
| Cerebras | `CEREBRAS_API_KEY` | See the [Cerebras docs](https://inference-docs.cerebras.ai/introduction) |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`<br>`AZURE_API_VERSION`<br>`AZURE_API_BASE` | |
| Azure OpenAI (Managed Identity) | `AZURE_API_VERSION`<br>`AZURE_API_BASE` | Requires `az login`; see the [Azure docs](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity) |
| LiteLLM | Depends on the model | See the [LiteLLM docs](https://docs.litellm.ai/docs/providers) |

Other models work through [LiteLLM](https://docs.litellm.ai/docs/#litellm-python-sdk). For example, set `GEMINI_API_KEY` and call `moa-gemini/gemini-1.5-flash-002`.

## Usage

### Selecting a technique

When the server runs with the default `auto` approach, a technique can be selected in three ways, in this order of preference:

```python
# 1. Model name prefix
client.chat.completions.create(model="moa-gpt-4o-mini", messages=messages)

# 2. extra_body field
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    extra_body={"optillm_approach": "bon|moa|mcts"},
)

# 3. Tag in the system or user prompt
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "<optillm_approach>re2</optillm_approach> How many r's are there in strawberry?"}],
)
```

If the server is started with a specific `--approach`, set `model` to the model name only.

### Combining techniques

- `&` runs techniques as a pipeline from left to right, using each response as the request to the next stage: `cot_reflection&moa`
- `|` runs techniques in parallel and returns the responses as a list: `bon|moa|mcts`

### How requests flow

![Sequence diagram showing optillm in use](https://raw.githubusercontent.com/algorithmicsuperintelligence/optillm/main/optillm-sequence-diagram.png)

- `A` is an existing tool, framework, or your own code using any OpenAI client SDK.
- `B` is the optillm service, which sends requests to the configured `base_url`.
- `C` is any service providing an OpenAI-compatible chat completions endpoint.

For an LLM without an OpenAI-compatible endpoint, you can use the [LiteLLM proxy server](https://docs.litellm.ai/docs/proxy/quick_start).

### Local inference

Set `OPTILLM_API_KEY` to any value to use the built-in inference server, and use the same value in your client. Any HuggingFace model can be passed as the model name; set `HF_TOKEN` for private models. Add LoRA adapters with the `+` separator and choose one with `active_adapter` (the last one is used by default).

```python
client = OpenAI(api_key="optillm", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct+patched-codes/Llama-3.2-1B-FastApply+patched-codes/Llama-3.2-1B-FixVulns",
    messages=messages,
    temperature=0.2,
    logprobs=True,
    top_logprobs=3,
    extra_body={"active_adapter": "patched-codes/Llama-3.2-1B-FastApply"},
)
```

Alternate decoding techniques are available on the local server:

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=messages,
    temperature=0.2,
    extra_body={
        "decoding": "cot_decoding",  # or "entropy_decoding"
        # CoT specific params
        "k": 10,
        "aggregate_paths": True,
        # OR entropy specific params
        "top_k": 27,
        "min_p": 0.03,
    },
)
```

The server also exposes the legacy `POST /v1/completions` endpoint. Streaming is not supported on it yet; use `/v1/chat/completions` for streaming.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPTILLM_API_KEY" \
  -d '{"model": "meta-llama/Llama-3.2-1B-Instruct", "prompt": "Write a haiku about recursion.", "max_tokens": 64}'
```

### External servers (llama.cpp, ollama)

```bash
export OPENAI_API_KEY="sk-no-key"  # placeholder
./llama-server -c 4096 -m path_to_model
python3 optillm.py --base-url http://localhost:8080/v1
```

> [!WARNING]
> The Anthropic API, llama.cpp-server, and ollama do not support sampling multiple responses from a model, which limits the available approaches to `cot_reflection`, `leap`, `plansearch`, `rstar`, `rto`, `self_consistency`, `re2`, and `z3`. For HuggingFace models, the built-in local inference server supports multiple responses.

## Use as an LLM gateway

The `proxy` plugin turns OptiLLM into a self-hosted gateway with load balancing, failover, and health checks across any OpenAI-compatible providers, with no gateway fees. It can also apply any of the techniques below to the requests it routes, combining routing and optimization in one service. See [Using OptiLLM as an LLM gateway](docs/llm-gateway.md) for setup.

## Techniques

| Approach | Slug | Description |
|----------|------|-------------|
| [MARS (Multi-Agent Reasoning System)](optillm/mars) | `mars` | Multi-agent reasoning with diverse temperature exploration, cross-verification, and iterative improvement |
| [Cerebras Planning and Optimization](optillm/cepo) | `cepo` | Combines Best of N, Chain-of-Thought, Self-Reflection, Self-Improvement, and various prompting techniques |
| CoT with Reflection | `cot_reflection` | Chain-of-thought reasoning with `<thinking>`, `<reflection>` and `<output>` sections |
| PlanSearch | `plansearch` | Search over candidate plans for solving a problem in natural language |
| ReRead | `re2` | Improves reasoning by processing queries twice |
| Self-Consistency | `self_consistency` | An advanced self-consistency method |
| Z3 Solver | `z3` | Uses the Z3 theorem prover for logical reasoning |
| R* Algorithm | `rstar` | The R* algorithm for problem solving |
| LEAP | `leap` | Learns task-specific principles from few-shot examples |
| Round Trip Optimization | `rto` | Optimizes responses through a round-trip process |
| Best of N Sampling | `bon` | Generates multiple responses and selects the best one |
| Mixture of Agents | `moa` | Combines responses from multiple critiques |
| Monte Carlo Tree Search | `mcts` | Uses MCTS for decision-making in chat responses |
| PV Game | `pvg` | Applies a prover-verifier game at inference time |
| [Deep Confidence](optillm/deepconf) | Local only | Confidence-guided reasoning with multiple intensity levels |
| CoT Decoding | Local only | Chain-of-thought decoding to elicit reasoning without explicit prompting |
| Entropy Decoding | Local only | Adaptive sampling based on the uncertainty of tokens during generation |
| Thinkdeeper | Local only | The `reasoning_effort` parameter for reasoning models like DeepSeek R1 |
| [AutoThink](optillm/autothink) | Local only | Combines query complexity classification with steering vectors |

## Plugins

| Plugin | Slug | Description |
|--------|------|-------------|
| [System Prompt Learning](optillm/plugins/spl) | `spl` | Implements what [Andrej Karpathy called the third paradigm](https://x.com/karpathy/status/1921368644069765486) for LLM learning, letting the model acquire problem-solving knowledge and strategies |
| [Deep Think](optillm/plugins/deepthink) | `deepthink` | A Gemini-like Deep Think approach using inference-time scaling for reasoning LLMs |
| [Long-Context Cerebras Planning and Optimization](optillm/plugins/longcepo) | `longcepo` | Combines planning and divide-and-conquer processing of long documents |
| Majority Voting | `majority_voting` | Generates k candidate solutions and selects the most frequent answer (default k=6) |
| [MCP Client](#mcp-plugin) | `mcp` | Model Context Protocol client for using any LLM with any MCP server |
| Router | `router` | Uses the [optillm-modernbert-large](https://huggingface.co/codelion/optillm-modernbert-large) model to route requests to approaches based on the prompt |
| Chain-of-Code | `coc` | Combines CoT with code execution and LLM-based code simulation |
| Memory | `memory` | Short-term memory layer for unbounded context length with any LLM. Set `OPTILLM_MEMORY_FILE` to persist memories across requests |
| Compact | `compact` | Compresses conversation context that exceeds a token budget into a structured summary, keeping recent turns verbatim |
| Privacy | `privacy` | Anonymizes PII in the request and restores it in the response |
| Read URLs | `readurls` | Fetches URLs found in the request and adds their content to the context |
| Execute Code | `executecode` | Runs Python code from requests and LLM responses. **Disabled by default**; see [Security](#security) |
| JSON | `json` | Structured outputs using the outlines library; supports pydantic types and JSON schema |
| GenSelect | `genselect` | Generates multiple candidates and selects the best based on quality criteria |
| Web Search | `web_search` | Performs Google searches using Chrome automation (Selenium) |
| [Deep Research](optillm/plugins/deep_research) | `deep_research` | Test-Time Diffusion Deep Researcher (TTD-DR) for research reports using iterative refinement |
| [Proxy](optillm/plugins/proxy) | `proxy` | Load balancing and failover across LLM providers with health monitoring and round-robin routing |
| [Streaming Hints](optillm/plugins/streaming_hints) | `streaming_hints` | Watches reasoning as it is generated and injects relevant hints from a memory bank at moments of uncertainty |

### MCP plugin

The [Model Context Protocol](https://modelcontextprotocol.io/) plugin connects OptiLLM to MCP servers, making their tools, resources, and prompts available to the model. When the model calls a tool, the plugin executes it and returns the result to the model.

Configure servers in `~/.optillm/mcp_config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"],
      "description": "Local filesystem access"
    },
    "github": {
      "transport": "sse",
      "url": "https://api.githubcopilot.com/mcp",
      "headers": {
        "Authorization": "Bearer ${GITHUB_TOKEN}",
        "Accept": "text/event-stream"
      },
      "description": "GitHub MCP server"
    },
    "remote-http": {
      "transport": "streamable_http",
      "url": "https://api.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${API_TOKEN}"
      },
      "description": "Remote Streamable HTTP MCP server"
    }
  },
  "log_level": "INFO"
}
```

| Transport | Required | Optional |
|-----------|----------|----------|
| `stdio` (default) | `command` | `args`, `env` |
| `sse` | `url` | `headers`, `sse_read_timeout` (default 300.0) |
| `streamable_http` | `url` | `headers`, `sse_read_timeout` (default 300.0) |

All transports accept `description` and `timeout` (connection timeout in seconds, default 5.0). The `websocket` transport was removed in MCP Python SDK 2.x; use `streamable_http` or `sse` instead. String values such as headers support `${VARIABLE_NAME}` expansion, and those variables must be set before starting OptiLLM.

Any of the [official MCP servers](https://modelcontextprotocol.io/examples) can be used. Logs are written to `~/.optillm/logs/mcp_plugin.log`; check there for connection and tool errors.

## Configuration

Parameters can be passed on the command line. Most can also be set as environment variables prefixed with `OPTILLM_`, for example `OPTILLM_APPROACH=mcts`. Run `optillm --help` for the full list.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--approach` | Inference approach to use | `auto` |
| `--model` | OpenAI model to use | `gpt-4o-mini` |
| `--base-url` | Base URL for an OpenAI-compatible endpoint | `""` |
| `--host` | Host address to bind the server | `127.0.0.1` |
| `--port` | Port to run the proxy on | 8000 |
| `--optillm-api-key` | Optional API key for client authentication to optillm | `""` |
| `--n` | Number of final responses to return | 1 |
| `--return-full-response` | Return the full response including the CoT with `<thinking>` tags | `False` |
| `--mcts-simulations` | Number of MCTS simulations | 2 |
| `--mcts-exploration` | Exploration weight for MCTS | 0.2 |
| `--mcts-depth` | Simulation depth for MCTS | 1 |
| `--best-of-n` | Number of samples for the best_of_n approach | 3 |
| `--rstar-max-depth` | Maximum depth for the R* algorithm | 3 |
| `--rstar-num-rollouts` | Number of rollouts for the R* algorithm | 5 |
| `--rstar-c` | Exploration constant for the R* algorithm | 1.4 |
| `--cepo_*` | See CePO parameters below | Various |

<details>
<summary><strong>CePO parameters</strong></summary>

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--cepo_bestofn_n` | Number of responses to be generated in best of n stage | 3 |
| `--cepo_bestofn_temperature` | Temperature for verifier in best of n stage | 0.1 |
| `--cepo_bestofn_max_tokens` | Maximum number of tokens for verifier in best of n stage | 4096 |
| `--cepo_bestofn_rating_type` | Type of rating in best of n stage ("absolute" or "pairwise") | `"absolute"` |
| `--cepo_planning_n` | Number of plans generated in planning stage | 3 |
| `--cepo_planning_m` | Number of attempts to generate n plans in planning stage | 6 |
| `--cepo_planning_temperature_step1` | Temperature for generator in step 1 of planning stage | 0.55 |
| `--cepo_planning_temperature_step2` | Temperature for generator in step 2 of planning stage | 0.25 |
| `--cepo_planning_temperature_direct_resp` | Temperature for generator after step 2 if planning fails and answer directly | 0.1 |
| `--cepo_planning_temperature_step3` | Temperature for generator in step 3 of planning stage | 0.1 |
| `--cepo_planning_temperature_step4` | Temperature for generator in step 4 of planning stage | 0 |
| `--cepo_planning_max_tokens_step1` | Maximum number of tokens in step 1 of planning stage | 4096 |
| `--cepo_planning_max_tokens_step2` | Maximum number of tokens in step 2 of planning stage | 4096 |
| `--cepo_planning_max_tokens_direct_resp` | Maximum number of tokens after step 2 if planning fails and answer directly | 4096 |
| `--cepo_planning_max_tokens_step3` | Maximum number of tokens in step 3 of planning stage | 4096 |
| `--cepo_planning_max_tokens_step4` | Maximum number of tokens in step 4 of planning stage | 4096 |
| `--cepo_use_reasoning_fallback` | Whether to fallback to lower levels of reasoning when higher level fails | False |
| `--cepo_num_of_retries` | Number of retries if llm call fails, 0 for no retries | 0 |
| `--cepo_print_output` | Whether to print the output of each stage | `False` |
| `--cepo_config_file` | Path to CePO configuration file | `None` |
| `--cepo_use_plan_diversity` | Use additional plan diversity step | `False` |
| `--cepo_rating_model` | Specify a model for rating step if different than for completion | `None` |

</details>

## Security

**Network binding and authentication.** The server binds to `127.0.0.1` by default. Use `--host 0.0.0.0` to allow external connections, and only on trusted networks or with authentication enabled. Set `--optillm-api-key` (or `OPTILLM_API_KEY`) to require clients to send `Authorization: Bearer <key>`.

**Code execution.** The `executecode` plugin runs Python from requests and LLM responses in a live Jupyter kernel with no sandboxing, so it is disabled unless you opt in. Only enable it in single-user, air-gapped environments where whoever submits requests is trusted with full access to the host.

```bash
optillm --enable-unsafe-plugins=executecode
# or: export OPTILLM_ENABLE_UNSAFE_PLUGINS=executecode
```

**URL fetching.** The `readurls` plugin blocks requests to non-public destinations (private, loopback, link-local, and reserved addresses such as the cloud metadata IP `169.254.169.254`) to prevent SSRF. Every redirect hop is re-validated and unresolvable hosts are rejected. Only opt out on trusted networks that need internal URL fetching:

```bash
optillm --readurls-allow-internal
# or: export OPTILLM_READURLS_ALLOW_INTERNAL=true
```

The guard validates DNS resolution before each request, but the HTTP client resolves the host again when it connects, so a domain under attacker control could rebind to an internal address between the two lookups. For high-security deployments, also restrict the network egress of the optillm process with a firewall or service-mesh policy.

**SSL verification.** Use a custom CA bundle for self-signed certificates or corporate proxies. Disabling verification is insecure and intended for development only. See [SSL_CONFIGURATION.md](SSL_CONFIGURATION.md) for details.

```bash
optillm --ssl-cert-path /path/to/ca-bundle.crt  # or: export OPTILLM_SSL_CERT_PATH=...
optillm --no-ssl-verify                         # or: export OPTILLM_SSL_VERIFY=false
```

## Benchmark results

### MARS on AIME 2025, IMO 2025, and LiveCodeBench (Oct 2025)

| Benchmark | Approach | Problems | Correct | Accuracy | Improvement |
|-----------|----------|----------|---------|----------|-------------|
| **AIME 2025** | Baseline | 30 | 13 | 43.3% | - |
| **AIME 2025** | **MARS** | 30 | **22** | **73.3%** | **+30.0pp (+69.2%)** |
| **IMO 2025** | Baseline | 6 | 1 | 16.7% | - |
| **IMO 2025** | **MARS** | 6 | **2** | **33.3%** | **+16.7pp (+100%)** |
| **LiveCodeBench v5/v6** | Baseline | 105 | 41 | 39.05% | - |
| **LiveCodeBench v5/v6** | **MARS** | 105 | **53** | **50.48%** | **+11.43pp (+29.3%)** |

Model: google/gemini-2.5-flash-lite-preview-09-2025 via OpenRouter. Configuration: 3 agents, 2-pass verification, thinking tags disabled for proofs.

### CePO on math and code benchmarks (Sep 2025)

| Method                  | AIME 2024 | AIME 2025 |  GPQA  | LiveCodeBench |
| ----------------------: | :-------: | :-------: | :----: | :-----------: |
| Qwen3 8B                |   74.0    |   68.3    |  59.3  |     55.7      |
| CePO (using Qwen3 8B)   |   86.7    |   80.0    |  62.5  |     60.5      |
| Qwen3 32B               |   81.4    |   72.9    |  66.8  |     65.7      |
| CePO (using Qwen3 32B)  | **90.7**  | **83.3**  |  70.0  |   **71.9**    |
| Qwen3 235B              |   85.7    |   81.5    |  71.1  |     70.7      |
| DeepSeek R1             |   79.8    |   70.0    |  71.5  |     64.3      |
| OpenAI o3-mini          |   79.6    |   74.8    |  76.8  |     66.3      |
| Grok3 Think             |   83.9    |   77.3    |**80.2**|     70.6      |

### AutoThink on GPQA-Diamond and MMLU-Pro (May 2025)

| **Model**     | **GPQA-Diamond**            |                          | **MMLU-Pro**               |                          |
|----------------|-----------------------------|--------------------------|----------------------------|--------------------------|
|                | Accuracy (%)                | Avg. Tokens              | Accuracy (%)               | Avg. Tokens              |
| DeepSeek-R1-Distill-Qwen-1.5B    | 21.72                       | 7868.26                  | 25.58                      | 2842.75                  |
| with Fixed Budget | 28.47                     | 3570.00                  | 26.18                      | 1815.67                  |
| **with AutoThink**  | **31.06**                   | **3520.52**              | **26.38**                  | **1792.50**              |

<details>
<summary><strong>Earlier results (2024 to April 2025)</strong></summary>

### LongCePO on LongBench v2 (Apr 2025)

| Model¹                             | Context window | Short samples (up to 32K words) | Medium samples (32–128K words) |
|----------------------------------|----------------|------------------|----------------|
| Llama 3.3 70B Instruct           | 128K           | 36.7 (45.0)               | 27.0 (33.0)            |
| **LongCePO + Llama 3.3 70B Instruct** | **8K**             | **36.8 ± 1.38**        |  **38.7 ± 2.574 (39.735)²**             |
| Mistral-Large-Instruct-2411     | 128K           | 41.7 (46.1)                 | 30.7 (34.9)             |
| o1-mini-2024-09-12               | 128K           | 48.6 (48.9)                | 33.3 (32.9)            |
| Claude-3.5-Sonnet-20241022       | 200K           | 46.1 (53.9)                | 38.6 (41.9)            |
| Llama-4-Maverick-17B-128E-Instruct | 524K         | 32.22 (50.56)                  | 28.84 (41.86)               |

¹ Performance numbers reported by LongBench v2 authors, except for LongCePO and Llama-4-Maverick results.

² Numbers in parentheses for LongCePO indicate accuracy of majority voting from 5 runs.

### LongCePO on HELMET - InfiniteBench En.MC, 128K length (Apr 2025)

| Model   | Accuracy (%) |
|---------|---------------|
| Llama 3.3 70B Instruct  (full context)  | 58.0          |
| **LongCePO + Llama 3.3 70B Instruct (8K context)** | **71.6 ± 1.855 (73.0)¹**  |
| o1-mini-2024-09-12 (full context) | 58.0          |
| gpt-4o-2024-08-06 (full context) | 74.0          |

¹ Numbers in parentheses for LongCePO indicate accuracy of majority voting from 5 runs.

### CePO on math and code benchmarks (Mar 2025)

| Method                         | Math-L5 | MMLU-Pro (Math) | CRUX | LiveCodeBench (pass@1) | Simple QA |
| -----------------------------: | :-----: | :-------------: | :----: | :--------------------: | :-------: |
| Llama 3.3 70B                  |  51.0   |      78.6       |  72.6  |          27.1          |    20.9   |
| Llama 3.1 405B                 |  49.8   |      79.2       |  73.0  |          31.8          |    13.5   |
| CePO (using Llama 3.3 70B)     |  69.6   |      84.8       |  80.1  |          31.9          |  **22.6** |
| QwQ 32B                        |  61.4   |      90.8       |  82.5  |          44.3          |    7.8    |
| CePO (using QwQ 32B)           |  88.1   |    **92.0**     |  86.3  |        **51.5**        |    8.2    |
| DeepSeek R1 Llama              |  83.1   |      82.0       |  84.0  |          47.3          |    14.6   |
| CePO (using DeepSeek R1 Llama) |**90.2** |      84.0       |**89.4**|          47.2          |    15.5   |

### coc-claude-3-5-sonnet-20241022 on AIME 2024 pass@1 (Nov 2024)

| Model | Score |
|-------|-----:|
| o1-mini | 56.67 |
| coc-claude-3-5-sonnet-20241022 | 46.67 |
| coc-gemini/gemini-exp-1121 | 46.67 |
| o1-preview | 40.00 |
| gemini-exp-1114 | 36.67 |
| claude-3-5-sonnet-20241022 | 20.00 |
| gemini-1.5-pro-002 | 20.00 |
| gemini-1.5-flash-002 | 16.67 |

### readurls&memory-gpt-4o-mini on Google FRAMES Benchmark (Oct 2024)

| Model | Accuracy |
| ----- | -------- |
| readurls&memory-gpt-4o-mini | 61.29 |
| gpt-4o-mini | 50.61 |
| readurls&memory-Gemma2-9b | 30.1 |
| Gemma2-9b | 5.1 |
| Gemma2-27b | 30.8 |
| Gemini Flash 1.5 | 66.5 |
| Gemini Pro 1.5 | 72.9 |

### plansearch-gpt-4o-mini on LiveCodeBench (Sep 2024)

| Model                  | pass@1 | pass@5 | pass@10 |
| ---------------------- | ------ | ------ | ------- |
| plansearch-gpt-4o-mini | 44.03  | 59.31  | 63.5    |
| gpt-4o-mini            | 43.9   | 50.61  | 53.25   |
| claude-3.5-sonnet      | 51.3   |        |         |
| gpt-4o-2024-05-13      | 45.2   |        |         |
| gpt-4-turbo-2024-04-09 | 44.2   |        |         |

### moa-gpt-4o-mini on Arena-Hard-Auto (Aug 2024)

![Results showing Mixture of Agents approach using gpt-4o-mini on Arena Hard Auto Benchmark](https://raw.githubusercontent.com/algorithmicsuperintelligence/optillm/main/moa-results.png)

### optillm with Patchwork (July 2024)

We used optillm with [patchwork](https://github.com/patched-codes/patchwork), an open-source framework that automates development work such as PR reviews, bug fixing, and security patching through workflows called patchflows. The mixture of agents approach (moa) improved performance across all supported patchflows.

![Results showing optillm mixture of agents approach used with patchflows](https://raw.githubusercontent.com/algorithmicsuperintelligence/optillm/main/moa-patchwork-results.png)

</details>

## Testing

```bash
pip install -r tests/requirements.txt

# Approach tests
python tests/test.py
python tests/test.py --approaches moa bon mcts
python tests/test.py --single-test "Simple Math Problem"

# Unit and integration tests
./tests/run_tests.sh
pytest tests/test_plugins.py -v
```

Tests run on pull requests through GitHub Actions. See [tests/README.md](tests/README.md) for the test structure and how to write new tests.

## Contributing

- Report bugs by [opening an issue](https://github.com/algorithmicsuperintelligence/optillm/issues/new).
- Share ideas by [starting a discussion](https://github.com/algorithmicsuperintelligence/optillm/discussions).
- Look for a place to start in the [good first issues](https://github.com/algorithmicsuperintelligence/optillm/labels/good%20first%20issue).

## References

- [Eliciting Fine-Tuned Transformer Capabilities via Inference-Time Techniques](https://arxiv.org/abs/2506.08060)
- [AutoThink: efficient inference for reasoning LLMs](https://dx.doi.org/10.2139/ssrn.5253327) - [Implementation](optillm/autothink)
- [Deep Think with Confidence: Confidence-guided reasoning and inference-time scaling](https://arxiv.org/abs/2508.15260) - [Implementation](optillm/deepconf)
- [Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/abs/2402.03620) - [Implementation](optillm/plugins/deepthink)
- [CePO: Empowering Llama with Reasoning using Test-Time Compute](https://cerebras.ai/blog/cepo) - [Implementation](optillm/cepo)
- [LongCePO: Empowering LLMs to efficiently leverage infinite context](https://cerebras.ai/blog/longcepo) - [Implementation](optillm/plugins/longcepo)
- [Chain of Code: Reasoning with a Language Model-Augmented Code Emulator](https://arxiv.org/abs/2312.04474) - [Inspired the implementation of coc plugin](optillm/plugins/coc_plugin.py)
- [Entropy Based Sampling and Parallel CoT Decoding](https://github.com/xjdr-alt/entropix) - [Implementation](optillm/entropy_decoding.py)
- [Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation](https://arxiv.org/abs/2409.12941) - [Evaluation script](scripts/eval_frames_benchmark.py)
- [Writing in the Margins: Better Inference Pattern for Long Context Retrieval](https://www.arxiv.org/abs/2408.14906) - [Inspired the implementation of the memory plugin](optillm/plugins/memory_plugin.py)
- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200) - [Implementation](optillm/cot_decoding.py)
- [Re-Reading Improves Reasoning in Large Language Models](https://arxiv.org/abs/2309.06275) - [Implementation](optillm/reread.py)
- [In-Context Principle Learning from Mistakes](https://arxiv.org/abs/2402.05403) - [Implementation](optillm/leap.py)
- [Planning In Natural Language Improves LLM Search For Code Generation](https://arxiv.org/abs/2409.03733) - [Implementation](optillm/plansearch.py)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) - [Implementation](optillm/self_consistency.py)
- [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://arxiv.org/abs/2408.06195) - [Implementation](optillm/rstar.py)
- [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692) - [Inspired the implementation of moa](optillm/moa.py)
- [Prover-Verifier Games improve legibility of LLM outputs](https://arxiv.org/abs/2407.13692) - [Implementation](optillm/pvg.py)
- [Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning](https://arxiv.org/abs/2405.00451) - [Inspired the implementation of mcts](optillm/mcts.py)
- [Unsupervised Evaluation of Code LLMs with Round-Trip Correctness](https://arxiv.org/abs/2402.08699) - [Inspired the implementation of rto](optillm/rto.py)
- [Patched MOA: optimizing inference for diverse software development tasks](https://arxiv.org/abs/2407.18521) - [Implementation](optillm/moa.py)
- [Patched RTC: evaluating LLMs for diverse software development tasks](https://arxiv.org/abs/2407.16557) - [Implementation](optillm/rto.py)
- [AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset](https://arxiv.org/abs/2504.16891) - [Implementation](optillm/plugins/genselect_plugin.py)
- [Test-Time Diffusion Deep Researcher (TTD-DR): Think More, Research More, Answer Better!](https://arxiv.org/abs/2507.16075v1) - [Implementation](optillm/plugins/deep_research)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{optillm,
  title = {OptiLLM: Optimizing inference proxy for LLMs},
  author = {Asankhaya Sharma},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/algorithmicsuperintelligence/optillm}
}
```
