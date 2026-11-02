# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OptiLLM is an OpenAI API compatible optimizing inference proxy that implements state-of-the-art techniques to improve accuracy and performance of LLMs. It focuses on reasoning improvements for coding, logical, and mathematical queries through inference-time compute optimization.

## Core Architecture

### Main Components

1. **Entry Points**: 
   - `optillm.py` - Main Flask server with inference routing
   - `optillm/inference.py` - Local inference engine with transformer models
   - Setup via `pyproject.toml` with console script `optillm=optillm:main`

2. **Optimization Techniques** (`optillm/`):
   - **Reasoning**: `cot_reflection.py`, `plansearch.py`, `leap.py`, `reread.py` 
   - **Sampling**: `bon.py` (Best of N), `moa.py` (Mixture of Agents), `self_consistency.py`
   - **Search**: `mcts.py` (Monte Carlo Tree Search), `rstar.py` (R* Algorithm)
   - **Verification**: `pvg.py` (Prover-Verifier Game), `z3_solver.py`
   - **Advanced**: `cepo/` (Cerebras Planning & Optimization), `rto.py` (Round Trip)

3. **Decoding Techniques**:
   - `cot_decoding.py` - Chain-of-thought without explicit prompting
   - `entropy_decoding.py` - Adaptive sampling based on token uncertainty
   - `thinkdeeper.py` - Reasoning effort scaling
   - `autothink/` - Query complexity classification with steering vectors

4. **Plugin System** (`optillm/plugins/`):
   - `spl/` - System Prompt Learning (third paradigm learning)
   - `deepthink/` - Gemini-like deep thinking with inference scaling
   - `longcepo/` - Long-context processing with divide-and-conquer
   - `mcp_plugin.py` - Model Context Protocol client
   - `memory_plugin.py` - Short-term memory for unbounded context
   - `privacy_plugin.py` - PII anonymization/deanonymization
   - `executecode_plugin.py` - Code interpreter integration
   - `json_plugin.py` - Structured outputs with outlines library

## Development Commands

### Installation & Setup
```bash
# Development setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Package installation
pip install optillm
```

### Running the Server
```bash
# Basic server (auto approach detection)
python optillm.py

# With specific approach
python optillm.py --approach moa --model gpt-4o-mini

# With external endpoint
python optillm.py --base_url http://localhost:8080/v1

# Docker
docker compose up -d
```

### Testing
```bash
# Run all approach tests
python tests/test.py

# Test specific approaches
python tests/test.py --approaches moa bon mcts

# Test with specific model/endpoint
python tests/test.py --model gpt-4o-mini --base-url http://localhost:8080/v1

# Single test case
python tests/test.py --single-test "specific_test_name"
```

### Evaluation Scripts
```bash
# Math benchmark evaluation
python scripts/eval_math500_benchmark.py

# AIME benchmark
python scripts/eval_aime_benchmark.py

# Arena Hard Auto evaluation  
python scripts/eval_arena_hard_auto_rtc.py

# FRAMES benchmark
python scripts/eval_frames_benchmark.py

# OptiLLM benchmark generation/evaluation
python scripts/gen_optillmbench.py
python scripts/eval_optillmbench.py
```

## Usage Patterns

### Approach Selection (Priority Order)
1. **Model prefix**: `moa-gpt-4o-mini` (approach slug + model name)
2. **extra_body field**: `{"optillm_approach": "bon|moa|mcts"}`
3. **Prompt tags**: `<optillm_approach>re2</optillm_approach>` in system/user prompt

### Approach Combinations
- **Pipeline** (`&`): `cot_reflection&moa` - sequential processing
- **Parallel** (`|`): `bon|moa|mcts` - multiple responses returned as list

### Local Inference
- Set `OPTILLM_API_KEY=optillm` to enable built-in transformer inference
- Supports HuggingFace models with LoRA adapters: `model+lora1+lora2`
- Advanced decoding: `{"decoding": "cot_decoding", "k": 10}`

### Plugin Configuration
- MCP: `~/.optillm/mcp_config.json` for Model Context Protocol servers
- SPL: Built-in system prompt learning for solving strategies
- Memory: Automatic unbounded context via chunking and retrieval
- GenSelect: Quality-based selection from multiple generated candidates

## Key Concepts

### Inference Optimization
The proxy intercepts OpenAI API calls and applies optimization techniques before forwarding to LLM providers (OpenAI, Cerebras, Azure, LiteLLM). Each technique implements specific reasoning or sampling improvements.

### Plugin Architecture
Plugins extend functionality via standardized interfaces. They can modify requests, process responses, add tools, or provide entirely new capabilities like code execution or structured outputs.

### Multi-Provider Support
Automatically detects and routes to appropriate LLM provider based on environment variables (`OPENAI_API_KEY`, `CEREBRAS_API_KEY`, etc.) with fallback to LiteLLM for broader model support.