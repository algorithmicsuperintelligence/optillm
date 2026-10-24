# DeepConf: Deep Think with Confidence

DeepConf is a confidence-aware reasoning approach for large language models that uses model-internal confidence signals to dynamically filter out low-quality reasoning traces during generation, improving both efficiency and accuracy.

## Overview

Based on the paper "Deep Think with Confidence" by Fu et al. (2024), DeepConf implements:

- **Token-level confidence scoring** using entropy and log-probability metrics
- **Online mode with early termination** to save computational resources
- **Warmup phase for threshold calibration** 
- **Consensus-based stopping** when high agreement is reached
- **Weighted majority voting** for final answer selection

## Features

- ✅ **Local models only** - Works with OptiLLM's local inference engine
- ✅ **Two variants**: `low` (aggressive, top 10%) and `high` (conservative, top 90%)
- ✅ **Configurable parameters** for different use cases
- ✅ **Early termination** to reduce token usage by 50-70%
- ✅ **Automatic quality control** without external evaluation

## Usage

### Basic Usage

Set up OptiLLM for local inference:

```bash
export OPTILLM_API_KEY=optillm
python optillm.py --model your-local-model
```

Then make a request with DeepConf decoding:

```python
import openai

client = openai.OpenAI(
    api_key="optillm",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="your-model",
    messages=[
        {"role": "user", "content": "Solve this math problem: What is the derivative of x^3 + 2x^2 - 5x + 1?"}
    ],
    extra_body={
        "decoding": "deepconf",
        "variant": "low",           # "low" or "high"
        "warmup_samples": 16,       # Number of calibration traces
        "max_traces": 64,           # Maximum total traces
        "consensus_threshold": 0.95  # Stop when consensus reached
    }
)

print(response.choices[0].message.content)
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `variant` | `"low"` | Filtering strategy: `"low"` (top 10%, aggressive) or `"high"` (top 90%, conservative) |
| `warmup_samples` | `16` | Number of initial traces for threshold calibration |
| `consensus_threshold` | `0.95` | Stop generation when this level of agreement is reached |
| `max_traces` | `128` | Maximum number of traces to generate |
| `window_size` | `2048` | Sliding window size for group confidence calculation |
| `top_k` | `5` | Number of top tokens for confidence calculation |
| `min_trace_length` | `100` | Minimum tokens before allowing early termination |
| `max_tokens_per_trace` | `4096` | Maximum tokens per individual trace |
| `confidence_metric` | `"lowest_group"` | Trace confidence used for the threshold and as voting weight (`"lowest_group"`, `"bottom_10_percent"` or `"average_confidence"`) |
| `include_stats` | `false` | Include processing statistics in response |

### Advanced Usage

Include statistics in the response for debugging:

```python
response = client.chat.completions.create(
    model="your-model", 
    messages=[...],
    extra_body={
        "decoding": "deepconf",
        "variant": "high",
        "include_stats": true,
        "warmup_samples": 8,
        "max_traces": 32
    }
)
```

## How It Works

1. **Warmup Phase**: Generate initial traces to calibrate confidence threshold
2. **Online Generation**: Generate traces with early termination based on confidence
3. **Consensus Check**: Stop when sufficient agreement is reached
4. **Final Selection**: Use weighted majority voting to select the best answer

### Confidence Metrics

- **Token Entropy**: `H = -∑P(j) log P(j)`
- **Token Confidence**: `C = -(1/k) ∑log P(j)` for top-k tokens
- **Group Confidence**: Sliding window averages over token confidences
- **Trace Confidence**: Average confidence across all tokens in a trace

### Variants

- **DeepConf-low**: Uses 90th percentile threshold (keeps top 10% traces) - more aggressive filtering
- **DeepConf-high**: Uses 10th percentile threshold (keeps top 90% traces) - more conservative filtering

## Performance

DeepConf typically achieves:
- **50-70% reduction in token usage** compared to standard sampling
- **Maintained or improved accuracy** through confidence-based filtering
- **Automatic quality control** without requiring external evaluation models

## Requirements

- Local model inference (PyTorch)
- OptiLLM with `OPTILLM_API_KEY=optillm`
- Compatible with transformer models that provide logits access

## References

- **Paper**: "Deep Think with Confidence" by Fu et al. (2025)
- **arXiv**: https://arxiv.org/abs/2508.15260
- **Authors**: Yichao Fu (UCSD), Xuewei Wang (Meta AI), Yuandong Tian (Meta AI), Jiawei Zhao (Meta AI)
