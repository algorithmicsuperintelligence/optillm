import os
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import torch.nn.functional as F
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import PeftModel, PeftConfig
import bitsandbytes as bnb
from scipy.stats import entropy
from functools import lru_cache
import time
import threading
import traceback
import platform
import sys
import re

from optillm.cot_decoding import cot_decode
from optillm.entropy_decoding import entropy_decode
from optillm.thinkdeeper import thinkdeeper_decode
from optillm.thinkdeeper_mlx import thinkdeeper_decode_mlx
from optillm.autothink import autothink_decode
from optillm.deepconf import deepconf_decode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_reasoning_tokens(text: str, tokenizer=None) -> int:
    """
    Count tokens within <think>...</think> tags in the given text.
    
    Args:
        text: The text to analyze
        tokenizer: Optional tokenizer instance for precise counting
        
    Returns:
        Number of reasoning tokens (0 if no think tags found)
    """
    if not text or not isinstance(text, str):
        return 0
    
    # Extract all content within <think>...</think> tags
    # Handle both complete and truncated think blocks
    
    # First, find all complete <think>...</think> blocks
    complete_pattern = r'<think>(.*?)</think>'
    complete_matches = re.findall(complete_pattern, text, re.DOTALL)
    
    # Then check for unclosed <think> tag (truncated response)
    # This finds <think> that doesn't have a matching </think> after it
    truncated_pattern = r'<think>(?!.*</think>)(.*)$'
    truncated_match = re.search(truncated_pattern, text, re.DOTALL)
    
    # Combine all thinking content
    thinking_content = ''.join(complete_matches)
    if truncated_match:
        thinking_content += truncated_match.group(1)
    
    if not thinking_content:
        return 0
    
    if tokenizer and hasattr(tokenizer, 'encode'):
        # Use tokenizer for precise counting
        try:
            tokens = tokenizer.encode(thinking_content)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Failed to count tokens with tokenizer: {e}")
    
    # Fallback: rough estimation (4 chars per token on average, minimum 1 token for non-empty content)
    content_length = len(thinking_content.strip())
    return max(1, content_length // 4) if content_length > 0 else 0

# MLX Support for Apple Silicon
try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load, generate as mlx_generate
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
    logger.info("MLX framework available")
except ImportError:
    MLX_AVAILABLE = False
    logger.debug("MLX framework not available - falling back to PyTorch")


# Hard ceiling of 4096 by default. Can be lowered via OPTILLM_MAX_TOKENS so a
# single local generation is bounded even when the request (or an approach's
# internal calls) sends no max_tokens -- important for small local models that
# do not reliably emit an EOS token (e.g. the dhara test model), which would
# otherwise ramble up to the full default on every call.
DEFAULT_MAX_NEW_TOKENS = 4096


def _default_max_new_tokens() -> int:
    """Default ``max_new_tokens`` for local generation (env-overridable)."""
    raw = os.environ.get("OPTILLM_MAX_TOKENS")
    if raw is None:
        return DEFAULT_MAX_NEW_TOKENS
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid OPTILLM_MAX_TOKENS=%r; using %d", raw, DEFAULT_MAX_NEW_TOKENS)
        return DEFAULT_MAX_NEW_TOKENS


@dataclass
class ModelConfig:
    base_model_id: str
    adapter_ids: Optional[List[str]] = None
    batch_size: int = 32
    max_cache_size: int = 5
    quantization_bits: int = 4
    device_preference: Optional[str] = None
    # Default generation parameters
    max_new_tokens: int = field(default_factory=_default_max_new_tokens)
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 0.7
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    pad_token_id: Optional[int] = None
    logprobs: bool = False
    # Advanced parameters
    use_memory_efficient_attention: bool = True
    enable_prompt_caching: bool = True
    dynamic_temperature: bool = False


@dataclass
class LogProbsResult:
    """Container for logprobs calculation results"""
    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: List[Dict[str, float]]
    bytes_per_token: List[List[int]]

class LogProbsCalculator:
    """Handles calculation of log probabilities for generated tokens"""
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        
    def _get_bytes_for_token(self, token: str) -> List[int]:
        """Get UTF-8 bytes for a token"""
        try:
            return list(token.encode('utf-8'))
        except UnicodeEncodeError:
            return []

    def _get_top_alternatives(
        self,
        logits: torch.Tensor,
        actual_token_id: int,
        num_alternatives: int
    ) -> Dict[str, float]:
        """Calculate top alternative tokens and their logprobs"""
        probs = F.softmax(logits, dim=-1)
        logprobs = torch.log(probs)
        
        # Get top tokens excluding the actual token
        top_values, top_indices = torch.topk(logprobs, k=num_alternatives + 1)
        
        alternatives = {}
        for value, idx in zip(top_values, top_indices):
            token = self.tokenizer.decode([idx])
            if idx != actual_token_id:  # Skip the actual token
                alternatives[token] = value.item()
                if len(alternatives) >= num_alternatives:
                    break
                    
        return alternatives

    def calculate_logprobs(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_alternatives: int = 5
    ) -> LogProbsResult:
        """Calculate log probabilities for a sequence of tokens"""
        self.model.eval()
        
        with torch.no_grad():
            # Get model outputs for the entire sequence
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits
            
            # Calculate softmax and log probabilities
            probs = F.softmax(logits, dim=-1)
            logprobs = torch.log(probs)
            
            # Process each position
            all_tokens = []
            all_token_logprobs = []
            all_top_logprobs = []
            all_bytes = []
            
            sequence_length = generated_ids.shape[-1]
            
            for pos in range(sequence_length - 1):  # -1 because we look at next token
                next_token_id = generated_ids[0, pos + 1]
                current_logits = logits[0, pos]
                
                # Get token and its logprob
                token = self.tokenizer.decode([next_token_id])
                token_logprob = logprobs[0, pos, next_token_id].item()
                
                # Get top alternative tokens
                top_logprobs = self._get_top_alternatives(
                    current_logits,
                    next_token_id,
                    num_alternatives
                )
                
                # Get bytes for token
                token_bytes = self._get_bytes_for_token(token)
                
                all_tokens.append(token)
                all_token_logprobs.append(token_logprob)
                all_top_logprobs.append(top_logprobs)
                all_bytes.append(token_bytes)
            
            # Add None for the last token
            all_tokens.append(self.tokenizer.decode([generated_ids[0, -1]]))
            all_token_logprobs.append(None)
            all_top_logprobs.append(None)
            all_bytes.append(self._get_bytes_for_token(all_tokens[-1]))
            
            return LogProbsResult(
                tokens=all_tokens,
                token_logprobs=all_token_logprobs,
                top_logprobs=all_top_logprobs,
                bytes_per_token=all_bytes
            )

# MLX Support Functions and Classes

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def should_use_mlx(model_id: str) -> bool:
    """Determine if a model should use MLX instead of PyTorch"""
    if not MLX_AVAILABLE or not is_apple_silicon():
        return False
    
    # Models that should use MLX
    mlx_patterns = [
        "mlx-community/",
        "mlx-",
        "-mlx-"
    ]
    
    # Known problematic models that should prefer MLX on Apple Silicon
    problematic_models = [
        "Qwen/Qwen3-",
        "google/gemma-3-",
        "google/gemma3-"
    ]
    
    model_lower = model_id.lower()
    
    # Direct MLX model detection
    for pattern in mlx_patterns:
        if pattern.lower() in model_lower:
            return True
    
    # Problematic model detection
    for pattern in problematic_models:
        if pattern.lower() in model_lower:
            logger.warning(f"Model {model_id} detected as potentially problematic with MPS backend")
            suggested_mlx = suggest_mlx_alternative(model_id)
            logger.warning(f"Consider using MLX model: {suggested_mlx}")
            # Don't auto-switch, but recommend
            return False
    
    return False

def suggest_mlx_alternative(model_id: str) -> str:
    """Suggest MLX alternative for a given model"""
    mlx_alternatives = {
        # Qwen3 models
        "Qwen/Qwen3-0.6B": "mlx-community/Qwen3-0.6B-4bit",
        "Qwen/Qwen3-1.7B": "mlx-community/Qwen3-1.7B-4bit",
        "Qwen/Qwen3-4B": "mlx-community/Qwen3-4B-4bit",
        "Qwen/Qwen3-8B": "mlx-community/Qwen3-8B-4bit",
        "Qwen/Qwen3-14B": "mlx-community/Qwen3-14B-4bit",
        "Qwen/Qwen3-32B": "mlx-community/Qwen3-32B-4bit",
        
        # Gemma 3 models  
        "google/gemma-3-1b-it": "mlx-community/gemma-3-1b-it-4bit",
        "google/gemma-3-4b-it": "mlx-community/gemma-3-4b-it-4bit",
        "google/gemma-3-12b-it": "mlx-community/gemma-3-12b-it-4bit",
        "google/gemma-3-27b-it": "mlx-community/gemma-3-27b-it-4bit",
    }
    
    return mlx_alternatives.get(model_id, f"mlx-community/{model_id.split('/')[-1]}-4bit")

@dataclass
class MLXModelConfig:
    """Configuration for MLX models"""
    model_id: str
    max_new_tokens: int = field(default_factory=_default_max_new_tokens)
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    enable_prompt_caching: bool = True

class MLXInferencePipeline:
    """MLX-based inference pipeline that mirrors PyTorch pipeline interface"""
    
    def __init__(self, model_config: MLXModelConfig, cache_manager):
        self.model_config = model_config
        self.cache_manager = cache_manager
        self.last_used = time.time()
        
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX framework not available. Install with: pip install mlx-lm")
        
        if not is_apple_silicon():
            raise RuntimeError("MLX framework is only supported on Apple Silicon")
        
        try:
            logger.info(f"Loading MLX model: {model_config.model_id}")
            self.model, self.tokenizer = self._load_mlx_model(model_config.model_id)
            logger.info("MLX model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MLX model: {str(e)}")
            raise
    
    def _load_mlx_model(self, model_id: str):
        """Load MLX model and tokenizer with caching"""
        def _load_model():
            start_time = time.time()
            logger.info(f"Loading MLX model: {model_id}")
            
            try:
                model, tokenizer = mlx_load(model_id)
                load_time = time.time() - start_time
                logger.info(f"MLX model loaded in {load_time:.2f}s")
                return model, tokenizer
            except Exception as e:
                logger.error(f"Error loading MLX model {model_id}: {str(e)}")
                raise
        
        return self.cache_manager.get_or_load_model(f"mlx_{model_id}", _load_model)
    
    def generate(
        self,
        prompt: str,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[int], List[Optional[Dict]]]:
        """Generate text using MLX"""
        start_time = time.time()
        
        if generation_params is None:
            generation_params = {}
        
        # Extract parameters with defaults
        max_tokens = generation_params.get("max_new_tokens", self.model_config.max_new_tokens)
        temperature = generation_params.get("temperature", self.model_config.temperature)
        top_p = generation_params.get("top_p", self.model_config.top_p)
        repetition_penalty = generation_params.get("repetition_penalty", self.model_config.repetition_penalty)
        num_return_sequences = generation_params.get("num_return_sequences", 1)
        
        # Handle seed
        if generation_params.get("seed") is not None:
            mx.random.seed(generation_params["seed"])
        
        responses = []
        token_counts = []
        logprobs_results = []
        
        # Generate multiple sequences if requested
        for _ in range(num_return_sequences):
            try:
                logger.debug(f"Generating with MLX: max_tokens={max_tokens}, temp={temperature}")
                
                # Use robust MLX generation with multiple fallback approaches
                response = self._robust_mlx_generate(
                    prompt, max_tokens, temperature, top_p, repetition_penalty
                )
                
                responses.append(response)
                
                # Count tokens (approximate) - check if response is string
                if isinstance(response, str):
                    token_count = len(self.tokenizer.encode(response))
                else:
                    # Sometimes MLX returns just the new tokens, get the actual text
                    token_count = len(response) if hasattr(response, '__len__') else 0
                token_counts.append(token_count)
                
                # MLX doesn't provide logprobs by default
                logprobs_results.append(None)
                
            except Exception as e:
                logger.error(f"Error during MLX generation: {str(e)}")
                logger.error(f"MLX generation parameters: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
                responses.append("")
                token_counts.append(0)
                logprobs_results.append(None)
        
        generation_time = time.time() - start_time
        logger.info(f"MLX generation completed in {generation_time:.2f}s")
        
        return responses, token_counts, logprobs_results
    
    def _robust_mlx_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float, repetition_penalty: float) -> str:
        """Robust MLX generation using sampler approach"""
        
        try:
            # Create sampler with generation parameters
            sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                min_p=0.0,  # Default min_p
                min_tokens_to_keep=1  # Default min_tokens_to_keep
            )
            
            # Generate using the sampler
            response = mlx_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False
            )
            
            return response
            
        except Exception as e:
            logger.error(f"MLX generation with sampler failed: {str(e)}")
            
            # Fallback: Try minimal parameters without sampler
            try:
                logger.debug("Attempting MLX generation without sampler")
                response = mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=max_tokens,
                    verbose=False
                )
                return response
            except Exception as fallback_e:
                logger.error(f"MLX fallback generation also failed: {str(fallback_e)}")
                raise
    
    def format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format the prompt according to model's chat template"""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}, using fallback")
                return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        else:
            return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    
    def process_batch(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        generation_params: Optional[Dict[str, Any]] = None,
        active_adapter: str = None,
        return_token_count: bool = True
    ) -> Tuple[List[str], List[int]]:
        """
        Process a batch of prompts with MLX-based batch inference
        
        This method provides true batch processing for MLX models, processing multiple
        prompts simultaneously for improved throughput.
        
        Args:
            system_prompts: List of system prompts
            user_prompts: List of user prompts
            generation_params: Generation parameters (temperature, max_tokens, etc.)
            active_adapter: Active adapter (not used in MLX)
            return_token_count: Whether to return token counts
            
        Returns:
            Tuple of (responses, token_counts)
        """
        import time
        
        if generation_params is None:
            generation_params = {}
        
        # Validate inputs
        if len(system_prompts) != len(user_prompts):
            raise ValueError(f"Number of system prompts ({len(system_prompts)}) must match user prompts ({len(user_prompts)})")
        
        if not system_prompts:
            return [], []
        
        batch_size = len(system_prompts)
        logger.info(f"MLX batch processing {batch_size} prompts")
        
        start_time = time.time()
        
        # Format all prompts using chat template
        formatted_prompts = [
            self.format_chat_prompt(system_prompt, user_prompt)
            for system_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]
        
        # Extract parameters
        max_tokens = generation_params.get("max_new_tokens", self.model_config.max_new_tokens)
        temperature = generation_params.get("temperature", self.model_config.temperature)
        top_p = generation_params.get("top_p", self.model_config.top_p)
        repetition_penalty = generation_params.get("repetition_penalty", self.model_config.repetition_penalty)
        n = generation_params.get("num_return_sequences", 1)
        
        # Handle seed
        if generation_params.get("seed") is not None:
            mx.random.seed(generation_params["seed"])
        
        # Since MLX doesn't natively support batch processing, we need to implement it
        # For now, we'll process each prompt individually but with optimized batching structure
        # TODO: Implement true MLX batch processing using custom generation loop
        
        all_responses = []
        token_counts = []
        
        try:
            # Process each prompt (sequential for now, but with batch infrastructure)
            for i, prompt in enumerate(formatted_prompts):
                logger.debug(f"Processing MLX batch item {i+1}/{batch_size}")
                
                # Generate responses for this prompt
                for _ in range(n):
                    try:
                        response = self._robust_mlx_generate(
                            prompt, max_tokens, temperature, top_p, repetition_penalty
                        )
                        
                        all_responses.append(response)
                        
                        # Count tokens (approximate)
                        if isinstance(response, str):
                            token_count = len(self.tokenizer.encode(response))
                        else:
                            token_count = len(response) if hasattr(response, '__len__') else 0
                        token_counts.append(token_count)
                        
                    except Exception as e:
                        logger.error(f"Error generating response for batch item {i+1}: {e}")
                        all_responses.append("")
                        token_counts.append(0)
            
            processing_time = time.time() - start_time
            logger.info(f"MLX batch processing completed in {processing_time:.2f}s")
            
            if return_token_count:
                return all_responses, token_counts
            return all_responses, [0] * len(all_responses)
            
        except Exception as e:
            logger.error(f"MLX batch processing failed: {e}")
            raise
    
    def _batch_tokenize(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Tokenize a batch of prompts with padding
        
        Args:
            prompts: List of text prompts
            
        Returns:
            Dictionary with tokenized inputs suitable for MLX
        """
        # For future implementation of true MLX batching
        # This would handle padding and attention masks for batch processing
        pass
    
    def _batch_generate(self, input_ids, attention_mask, generation_params: Dict) -> List[str]:
        """
        Perform batch generation using MLX model
        
        Args:
            input_ids: Batched input token IDs
            attention_mask: Attention mask for padded sequences
            generation_params: Generation parameters
            
        Returns:
            List of generated responses
        """
        # For future implementation of true MLX batching
        # This would implement the core batch generation logic using MLX arrays
        pass

class MLXManager:
    """Manager for MLX models and operations"""
    
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.available = MLX_AVAILABLE and is_apple_silicon()
        
        if self.available:
            logger.info("MLX manager initialized - Apple Silicon detected")
        else:
            logger.debug("MLX manager not available - requires Apple Silicon and mlx-lm")
    
    def create_pipeline(self, model_id: str, **kwargs) -> MLXInferencePipeline:
        """Create an MLX inference pipeline"""
        if not self.available:
            raise RuntimeError("MLX not available on this platform")
        
        config = MLXModelConfig(
            model_id=model_id,
            **kwargs
        )
        
        return MLXInferencePipeline(config, self.cache_manager)
    
    def is_mlx_model(self, model_id: str) -> bool:
        """Check if model should use MLX"""
        return should_use_mlx(model_id)

class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention using linear attention mechanism.
    Supports automatic fallback to optimized implementations when available.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5
        
        # Standard projections with bias
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Try to import optimized attention implementations
        self.optimized_attention = None
        
        # Try Flash Attention
        try:
            from flash_attn import flash_attn_func
            self.optimized_attention = flash_attn_func
            print("Using Flash Attention")
        except ImportError:
            pass
            
        # Try xFormers
        if self.optimized_attention is None:
            try:
                import xformers.ops as xops
                self.optimized_attention = xops.memory_efficient_attention
                print("Using xFormers attention")
            except ImportError:
                pass

    def _linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Implements linear attention for more memory efficiency"""
        # Scale query
        q = q * self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            # Convert boolean mask to float mask if needed
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.float()
            k = k * attention_mask.unsqueeze(-1)
        
        if causal:
            # Handle causal attention
            batch_size, num_heads, seq_length, head_dim = q.shape
            positions = torch.arange(seq_length, device=q.device)
            causal_mask = positions.view(1, 1, -1, 1) <= positions.view(1, 1, 1, -1)
            k = k * causal_mask.float()
        
        # Linear attention computation
        context = torch.matmul(k.transpose(-2, -1), v)
        out = torch.matmul(q, context)
        
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()
        
        # Project to q, k, v
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = q.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Try using optimized attention if available
        if self.optimized_attention is not None and hidden_states.device.type == 'cuda':
            # Handle attention mask for optimized implementations
            if attention_mask is not None:
                if attention_mask.dtype != torch.bool:
                    attention_mask = attention_mask > 0
                attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
            
            try:
                attn_output = self.optimized_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    causal=causal,
                    scale=self.scale
                )
            except Exception as e:
                print(f"Optimized attention failed, falling back to linear attention: {e}")
                attn_output = self._linear_attention(q, k, v, attention_mask, causal)
        else:
            # Use linear attention for CPU/MPS or when optimized attention is not available
            attn_output = self._linear_attention(q, k, v, attention_mask, causal)
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class PromptCache:
    """Advanced caching system for frequent prompts and responses"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.prompt_stats = defaultdict(lambda: {"count": 0, "success_rate": 0.0})
        
    @lru_cache(maxsize=128)
    def _compute_prompt_signature(self, prompt: str) -> str:
        """Compute a signature for semantic similarity matching"""
        # Simple but effective signature based on key content words
        words = set(prompt.lower().split())
        return " ".join(sorted(list(words)))
    
    def get_cached_response(self, prompt: str, temperature: float, top_p: float) -> Optional[str]:
        """Get cached response with fuzzy matching"""
        signature = self._compute_prompt_signature(prompt)
        
        if signature in self.cache:
            cached_item = self.cache[signature]
            if abs(cached_item["temperature"] - temperature) < 0.1 and abs(cached_item["top_p"] - top_p) < 0.1:
                self.prompt_stats[signature]["count"] += 1
                return cached_item["response"]
        
        return None
    
    def add_to_cache(self, prompt: str, response: str, temperature: float, top_p: float):
        """Add response to cache with metadata"""
        signature = self._compute_prompt_signature(prompt)
        
        self.cache[signature] = {
            "response": response,
            "temperature": temperature,
            "top_p": top_p,
            "timestamp": torch.cuda.current_timestamp() if torch.cuda.is_available() else 0
        }
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def update_stats(self, prompt: str, success: bool):
        """Update prompt success statistics"""
        signature = self._compute_prompt_signature(prompt)
        stats = self.prompt_stats[signature]
        stats["count"] += 1
        stats["success_rate"] = (stats["success_rate"] * (stats["count"] - 1) + float(success)) / stats["count"]

class DynamicTemperature:
    """Implements dynamic temperature scaling based on input characteristics"""
    
    def __init__(self):
        self.token_entropy_cache = {}
    
    def _compute_token_entropy(self, tokens: List[int]) -> float:
        """Compute token distribution entropy"""
        token_counts = np.bincount(tokens)
        probabilities = token_counts / len(tokens)
        return entropy(probabilities)
    
    def get_optimal_temperature(self, prompt: str, tokenizer: AutoTokenizer, base_temperature: float) -> float:
        """Calculate optimal temperature based on prompt characteristics"""
        tokens = tokenizer.encode(prompt)
        
        # Calculate entropy-based scaling
        token_entropy = self._compute_token_entropy(tokens)
        
        # Scale temperature based on prompt entropy and length
        length_factor = np.clip(len(tokens) / 100, 0.5, 2.0)
        entropy_factor = np.clip(token_entropy / 4.0, 0.5, 1.5)
        
        optimal_temperature = base_temperature * length_factor * entropy_factor
        return np.clip(optimal_temperature, 0.1, 2.0)

class CacheManager:
    """
    Singleton cache manager for models and tokenizers.
    Thread-safe but minimizes lock contention.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance
    
    def __init__(self, max_size: int = 5):
        if self._initialized:
            return
            
        with self._lock:
            if not self._initialized:
                logger.info("Initializing CacheManager singleton")
                self.max_size = max_size
                self.model_cache = OrderedDict()
                self.tokenizer_cache = OrderedDict()
                self.adapter_cache = OrderedDict()
                self.model_adapter_map = {}  # Maps model ID to list of loaded adapter IDs
                self.cache_stats = defaultdict(lambda: {"hits": 0, "misses": 0})
                self._initialized = True
                logger.info("CacheManager singleton initialized")
    
    def get_or_load_model(self, model_key: str, loader_fn) -> Tuple[Any, Any]:
        """Get or load model and tokenizer with minimal locking."""
        cached_model = cached_tokenizer = None
        cache_hit = False
        
        with self._lock:
            if model_key in self.model_cache and model_key in self.tokenizer_cache:
                cached_model = self.model_cache[model_key]
                cached_tokenizer = self.tokenizer_cache[model_key]
                self.model_cache.move_to_end(model_key)
                self.tokenizer_cache.move_to_end(model_key)
                self.cache_stats[model_key]["hits"] += 1
                cache_hit = True
                logger.debug(f"Cache hit for model: {model_key}")

        if cache_hit:
            return cached_model, cached_tokenizer

        logger.info(f"Loading model and tokenizer: {model_key}")
        model, tokenizer = loader_fn()
        
        with self._lock:
            if model_key in self.model_cache and model_key in self.tokenizer_cache:
                cached_model = self.model_cache[model_key]
                cached_tokenizer = self.tokenizer_cache[model_key]
                self.cache_stats[model_key]["hits"] += 1
                logger.debug(f"Using already cached model: {model_key}")
                return cached_model, cached_tokenizer
            
            self.model_cache[model_key] = model
            self.tokenizer_cache[model_key] = tokenizer
            self.cache_stats[model_key]["misses"] += 1
            self.model_adapter_map[model_key] = []  # Initialize empty adapter list for new model
            
            self._cleanup_caches()
            
            logger.info(f"Successfully cached model and tokenizer: {model_key}")
            return model, tokenizer
    
    def get_or_load_adapter(self, model_key: str, adapter_key: str, loader_fn):
        """Get or load adapter with enhanced caching."""
        cache_key = f"{model_key}_{adapter_key}"
        
        with self._lock:
            if cache_key in self.adapter_cache:
                adapter = self.adapter_cache[cache_key]
                self.adapter_cache.move_to_end(cache_key)
                logger.debug(f"Cache hit for adapter: {cache_key}")
                return adapter
        
        adapter = loader_fn()
        
        with self._lock:
            self.adapter_cache[cache_key] = adapter
            if model_key not in self.model_adapter_map:
                self.model_adapter_map[model_key] = []
            if adapter_key not in self.model_adapter_map[model_key]:
                self.model_adapter_map[model_key].append(adapter_key)
            self._cleanup_caches()
            logger.info(f"Successfully cached adapter: {cache_key}")
            return adapter
    
    def get_model_adapters(self, model_key: str) -> List[str]:
        """Get list of adapter IDs loaded for a specific model."""
        with self._lock:
            return self.model_adapter_map.get(model_key, [])
    
    def _cleanup_caches(self):
        """Clean up caches if they exceed max size."""
        while len(self.model_cache) > self.max_size:
            model_key, model = self.model_cache.popitem(last=False)
            if hasattr(model, 'cpu'):
                model.cpu()
            # Clean up associated adapters
            if model_key in self.model_adapter_map:
                for adapter_id in self.model_adapter_map[model_key]:
                    cache_key = f"{model_key}_{adapter_id}"
                    if cache_key in self.adapter_cache:
                        self.adapter_cache.pop(cache_key)
                self.model_adapter_map.pop(model_key)
            
        while len(self.tokenizer_cache) > self.max_size:
            self.tokenizer_cache.popitem(last=False)
            
        # Cleanup orphaned adapters
        valid_cache_keys = {
            f"{model_key}_{adapter_id}"
            for model_key, adapter_ids in self.model_adapter_map.items()
            for adapter_id in adapter_ids
        }
        
        orphaned_adapters = [
            key for key in self.adapter_cache.keys()
            if key not in valid_cache_keys
        ]
        
        for key in orphaned_adapters:
            adapter = self.adapter_cache.pop(key)
            if hasattr(adapter, 'cpu'):
                adapter.cpu()
                
        torch.cuda.empty_cache()

    @classmethod
    def get_instance(cls, max_size: int = 5) -> 'CacheManager':
        """Alternative way to get the singleton instance."""
        if cls._instance is None:
            return cls(max_size)
        return cls._instance

class DeviceManager:
    def __init__(self):
        self.available_devices = self._detect_devices()
        self.device_stats = {device: {'memory_used': 0, 'active_models': 0} for device in self.available_devices}

    def _detect_devices(self) -> List[str]:
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
        if torch.backends.mps.is_available():
            devices.append('mps')
        return devices

    def get_optimal_device(self, model_size: int = 0) -> str:
        if not self.available_devices:
            return 'cpu'

        # Prefer CUDA devices if available
        cuda_devices = [d for d in self.available_devices if 'cuda' in d]
        if cuda_devices:
            # Find CUDA device with most free memory
            max_free_memory = 0
            optimal_device = cuda_devices[0]
            
            for device in cuda_devices:
                idx = int(device.split(':')[1])
                free_memory = torch.cuda.get_device_properties(idx).total_memory - torch.cuda.memory_allocated(idx)
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    optimal_device = device
            
            return optimal_device
        
        # Fall back to MPS if available
        if 'mps' in self.available_devices:
            return 'mps'
        
        return 'cpu'

    def track_device_usage(self, device: str, memory_delta: int):
        if device in self.device_stats:
            self.device_stats[device]['memory_used'] += memory_delta

class ModelManager:
    def __init__(self, cache_manager: CacheManager, device_manager: DeviceManager):
        self.cache_manager = cache_manager
        self.device_manager = device_manager
        
    def quantize_model(self, model):
        """Quantize model to 4-bit precision using bitsandbytes"""
        def _replace_linear_layers(module):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(module, name, bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.float16
                    ))
                else:
                    _replace_linear_layers(child)
                    
        _replace_linear_layers(model)
        return model

    def load_base_model(self, model_id: str, quantize: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        def _load_model():
            logger.info(f"Loading base model: {model_id}")
            
            device = self.device_manager.get_optimal_device()
            logger.info(f"Using device: {device}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
            
            # Base kwargs for model loading
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if 'cuda' in device else device
            }
            
            # Configure device-specific optimizations
            if 'cuda' in device:
                compute_capability = torch.cuda.get_device_capability(0)
                if compute_capability[0] >= 8:
                    model_kwargs["torch_dtype"] = torch.bfloat16
                elif compute_capability[0] >= 7:
                    model_kwargs["torch_dtype"] = torch.float16
                    
                # Check for flash attention availability
                try:
                    import flash_attn
                    has_flash_attn = True
                    logger.info("Flash Attention 2 is available")
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                except ImportError:
                    has_flash_attn = False
                    logger.info("Flash Attention 2 is not installed - falling back to default attention")
                    
            elif 'mps' in device:
                # Special handling for Gemma models which have NaN issues with float16 on MPS
                if 'gemma' in model_id.lower():
                    model_kwargs["torch_dtype"] = torch.float32
                    logger.info("Using MPS device with float32 for Gemma model (float16 causes NaN)")
                else:
                    model_kwargs["torch_dtype"] = torch.float16
                    logger.info("Using MPS device with float16 precision")
            else:
                # CPU can use FP16 if available
                if hasattr(torch.cpu, 'has_fp16') and torch.cpu.has_fp16:
                    model_kwargs["torch_dtype"] = torch.float16
                    logger.info("Using CPU device with float16 precision")
                else:
                    model_kwargs["torch_dtype"] = torch.float32
                    logger.info("Using CPU device with float32 precision - FP16 not supported")
            
            # Load model with configured optimizations
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=os.getenv("HF_TOKEN"),
                    **model_kwargs
                )
            except Exception as e:
                if "attn_implementation" in model_kwargs:
                    logger.warning(f"Failed to load model with Flash Attention: {e}")
                    logger.info("Retrying without Flash Attention...")
                    model_kwargs.pop("attn_implementation")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        token=os.getenv("HF_TOKEN"),
                        **model_kwargs
                    )
                elif model_kwargs["torch_dtype"] == torch.float16:
                    # If FP16 fails, fallback to FP32
                    logger.warning(f"Failed to load model with FP16: {e}")
                    logger.info("Falling back to FP32...")
                    model_kwargs["torch_dtype"] = torch.float32
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        token=os.getenv("HF_TOKEN"),
                        **model_kwargs
                    )
            
            logger.info(f"Model loaded successfully with dtype: {model_kwargs['torch_dtype']}")
            
            # Only apply quantization for CUDA devices when not using mixed precision
            if quantize and 'cuda' in device and model_kwargs["torch_dtype"] == torch.float32:
                model = self.quantize_model(model)
            
            return model, tokenizer
            
        return self.cache_manager.get_or_load_model(model_id, _load_model)

class LoRAManager:
    """LoRA manager with enhanced error handling and caching"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.loaded_adapters = {}
        self.adapter_names = {}  # Maps adapter_id to valid adapter name

    def _get_adapter_name(self, adapter_id: str) -> str:
        """Create a valid adapter name from adapter_id."""
        if adapter_id in self.adapter_names:
            return self.adapter_names[adapter_id]
            
        name = adapter_id.replace('.', '_').replace('-', '_')
        name = ''.join(c if c.isalnum() or c == '_' else '' for c in name)
        if name[0].isdigit():
            name = f"adapter_{name}"
            
        self.adapter_names[adapter_id] = name
        return name

    def validate_adapter(self, adapter_id: str) -> bool:
        """Validate if adapter exists and is compatible"""
        try:
            config = PeftConfig.from_pretrained(
                adapter_id,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN")
            )
            return True
        except Exception as e:
            logger.error(f"Error validating adapter {adapter_id}: {str(e)}")
            return False

    def load_adapter(self, base_model: PreTrainedModel, adapter_id: str) -> PreTrainedModel:
        """Load a LoRA adapter with enhanced caching"""
        model_key = base_model.config._name_or_path
        
        def _load_adapter():
            logger.info(f"Loading LoRA adapter: {adapter_id}")
            
            if not self.validate_adapter(adapter_id):
                error_msg = f"Adapter {adapter_id} not found or is not compatible"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            try:
                adapter_name = self._get_adapter_name(adapter_id)
                
                config = PeftConfig.from_pretrained(
                    adapter_id,
                    trust_remote_code=True,
                    token=os.getenv("HF_TOKEN")
                )
                
                model = base_model
                model.add_adapter(
                    config,
                    adapter_name = adapter_name,
                )
                
                if model not in self.loaded_adapters:
                    self.loaded_adapters[model] = []
                if adapter_id not in self.loaded_adapters[model]:
                    self.loaded_adapters[model].append(adapter_id)
                
                return model
                
            except Exception as e:
                error_msg = f"Failed to load adapter {adapter_id}: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

        return self.cache_manager.get_or_load_adapter(model_key, adapter_id, _load_adapter)

    def set_active_adapter(self, model: PeftModel, adapter_id: str = None) -> bool:
        """Set a specific adapter as active with error handling"""
        if not isinstance(model, PeftModel):
            logger.warning("Model is not a PeftModel, cannot set active adapter")
            return False
            
        available_adapters = self.loaded_adapters.get(model, [])
        
        if not available_adapters:
            logger.warning("No adapters loaded in model")
            return False
            
        if adapter_id is None:
            adapter_id = available_adapters[-1]
            
        if adapter_id in available_adapters:
            try:
                model.set_adapter(self._get_adapter_name(adapter_id))
                logger.info(f"Successfully set active adapter to: {adapter_id}")
                return True
            except Exception as e:
                logger.error(f"Error setting adapter {adapter_id}: {str(e)}")
                return False
        else:
            logger.warning(f"Requested adapter {adapter_id} not loaded. Available adapters: {available_adapters}")
            return False
        
class InferencePipeline:
    def __init__(self, model_config: ModelConfig, cache_manager, device_manager, model_manager, lora_manager):
        self.model_config = model_config
        self.cache_manager = cache_manager
        self.device_manager = device_manager
        self.model_manager = model_manager
        self.lora_manager = lora_manager
        self.last_used = time.time()
        
        try:
            self.base_model, self.tokenizer = self.model_manager.load_base_model(
                model_config.base_model_id,
                quantize=model_config.quantization_bits == 4
            )
            
            self.tokenizer = self.setup_tokenizer(self.tokenizer)
            
            if self.base_model.get_input_embeddings().num_embeddings != len(self.tokenizer):
                self.base_model.resize_token_embeddings(len(self.tokenizer))
            
            self.current_model = self.base_model
            
            if model_config.adapter_ids:
                for adapter_id in model_config.adapter_ids:
                    try:
                        self.current_model = self.lora_manager.load_adapter(
                            self.current_model, adapter_id
                        )
                    except Exception as e:
                        logger.error(f"Error loading adapter {adapter_id}: {e}")
                
                # Set active adapter and verify it's set correctly
                if isinstance(self.current_model, PeftModel):
                    success = self.lora_manager.set_active_adapter(self.current_model)
                    if not success:
                        logger.error("Failed to set active adapter")
            
            self.dtype = self.current_model.dtype
            self.optimal_batch_size = self._find_optimal_batch_size()
            
        except Exception as e:
            logger.error(f"Pipeline initialization error: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            raise

    def setup_tokenizer(self, tokenizer: AutoTokenizer) -> AutoTokenizer:
        """Use tokenizer with its default configuration for inference"""
        logger.debug("  a. Starting tokenizer setup")
        
        # Just use existing special tokens without modification
        logger.debug(f"  b. Using tokenizer with vocab size: {len(tokenizer)}")
        logger.debug(f"  c. Special tokens: PAD={tokenizer.pad_token_id}, "
                    f"EOS={tokenizer.eos_token_id}, BOS={tokenizer.bos_token_id}")
        
        return tokenizer

    def _resolve_eos_token_ids(self):
        """Resolve the effective end-of-sequence token id(s) for generation.

        Prefer the model's own ``generation_config.eos_token_id``. Chat models
        commonly set it to the chat-turn end token (e.g. ``<|im_end|>``), which
        can differ from the tokenizer's ``eos_token_id`` (often the base-model
        ``<|end_of_text|>``). Passing only the tokenizer eos to ``generate`` there
        means the model never stops on its real turn-end token and rambles up to
        ``max_new_tokens`` -- e.g. dhara-250m's ChatML ends at ``<|im_end|>`` but
        its tokenizer eos is ``<|end_of_text|>``.

        The tokenizer eos is merged in as a fallback so a model that only emits
        the base eos still terminates. Returns an int, a list of ints, or None.
        """
        ids: List[int] = []
        gen_cfg = getattr(self.current_model, "generation_config", None)
        gc_eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
        if isinstance(gc_eos, int):
            ids.append(gc_eos)
        elif isinstance(gc_eos, (list, tuple)):
            ids.extend(int(x) for x in gc_eos if isinstance(x, int))
        tok_eos = self.tokenizer.eos_token_id
        if isinstance(tok_eos, int):
            ids.append(tok_eos)
        seen = set()
        resolved = [x for x in ids if not (x in seen or seen.add(x))]
        if not resolved:
            return None
        return resolved[0] if len(resolved) == 1 else resolved

    def get_optimized_generation_config(self, generation_params: Optional[Dict[str, Any]] = None) -> Dict:
        """Get optimized generation config"""
        config = {
            "max_new_tokens": generation_params.get("max_new_tokens", _default_max_new_tokens()),
            "do_sample": generation_params.get("temperature", 1.0) > 0,
            "temperature": generation_params.get("temperature", 1.0),
            "top_p": generation_params.get("top_p", 0.95),
            "num_return_sequences": generation_params.get("num_return_sequences", 1),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self._resolve_eos_token_ids(),
            "return_dict_in_generate": True,
            "output_scores": generation_params.get("logprobs", False),
            "use_cache": True
        }
        return config
    
    def generate(
        self,
        prompt: str,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[int]]:
        """Generate completions with optional logprobs"""
        start_time = time.time()

        # First: Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Tokenize with batching disabled for single prompts
        tokenize_start = time.time()
        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.current_model.device)
        logger.info(f"Tokenization time: {time.time() - tokenize_start:.2f}s")
        
        # Extract logprobs parameters
        calculate_logprobs = generation_params.get("logprobs", False)
        top_logprobs = generation_params.get("top_logprobs", 0)
        
        if top_logprobs and not calculate_logprobs:
            raise ValueError("logprobs must be true when top_logprobs is specified")
        
        if top_logprobs and not (0 <= top_logprobs <= 20):
            raise ValueError("top_logprobs must be between 0 and 20")
             
        # Get optimized generation config
        gen_config = self.get_optimized_generation_config(generation_params)

        # Add optional parameters
        if generation_params:
            if generation_params.get("presence_penalty", 0) != 0:
                gen_config["presence_penalty"] = generation_params["presence_penalty"]
            if generation_params.get("frequency_penalty", 0) != 0:
                gen_config["repetition_penalty"] = 1.0 + generation_params["frequency_penalty"]
            if generation_params.get("stop_sequences"):
                gen_config["stopping_criteria"] = self._create_stopping_criteria(
                    generation_params["stop_sequences"],
                    inputs['input_ids'].shape[1]
                )
            if generation_params.get("seed") is not None:
                torch.manual_seed(generation_params["seed"])
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(generation_params["seed"])

        # Generate responses
        generate_start = time.time()
        with torch.inference_mode():  # Faster than no_grad
            outputs = self.current_model.generate(
                **inputs,
                **gen_config
            )
        logger.info(f"Generation time: {time.time() - generate_start:.2f}s")
        
        generated_sequences = outputs.sequences
        input_length = inputs['input_ids'].shape[1]
        
        # Process outputs
        process_start = time.time()
        responses = []
        token_counts = []
        logprobs_results = []
        
        # Process each generated sequence
        for sequence in generated_sequences:
            response_tokens = sequence[input_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(response_text)
            token_counts.append(len(response_tokens))
            
            # Calculate logprobs if requested
            if calculate_logprobs:
                calculator = LogProbsCalculator(self.tokenizer, self.current_model)
                logprobs_result = calculator.calculate_logprobs(
                    input_ids=sequence.unsqueeze(0),
                    generated_ids=sequence.unsqueeze(0),
                    attention_mask=torch.ones_like(sequence).unsqueeze(0),
                    num_alternatives=top_logprobs or 5
                )
                logprobs_results.append({
                    "content": [{
                        "token": token,
                        "logprob": logprob,
                        "bytes": bytes_,
                        "top_logprobs": top_logprobs
                    } for token, logprob, bytes_, top_logprobs in zip(
                        logprobs_result.tokens[input_length:],
                        logprobs_result.token_logprobs[input_length:],
                        logprobs_result.bytes_per_token[input_length:],
                        logprobs_result.top_logprobs[input_length:]
                    )]
                })
            else:
                logprobs_results.append(None)
        
        logger.info(f"Post-processing time: {time.time() - process_start:.2f}s")
        logger.info(f"Total generation time: {time.time() - start_time:.2f}s")
        
        return responses, token_counts, logprobs_results
    
    def setup_efficient_attention(self):
        """Replace standard attention with memory-efficient version"""
        if hasattr(self.current_model, 'config') and hasattr(self.current_model.config, 'hidden_size'):
            hidden_size = self.current_model.config.hidden_size
            num_attention_heads = self.current_model.config.num_attention_heads
            self.efficient_attention = MemoryEfficientAttention(hidden_size, num_attention_heads)
            
            # Monkey patch attention computation if possible
            if hasattr(self.current_model, 'encoder') and hasattr(self.current_model.encoder, 'layer'):
                for layer in self.current_model.encoder.layer:
                    if hasattr(layer, 'attention'):
                        layer.attention.self = self.efficient_attention
            logger.info("Memory-efficient attention mechanism enabled")

    def _find_optimal_batch_size(self, initial_batch_size: int = 1, max_batch_size: int = 128) -> int:
        """Find optimal batch size through binary search with memory monitoring"""
        if not torch.cuda.is_available():
            return initial_batch_size

        device = self.current_model.device
        if 'cuda' not in str(device):
            return initial_batch_size

        left, right = initial_batch_size, max_batch_size
        optimal_size = initial_batch_size
        
        sample_text = "Sample input text for batch size optimization."
        
        while left <= right:
            mid = (left + right) // 2
            try:
                torch.cuda.empty_cache()
                
                inputs = self.tokenizer([sample_text] * mid, 
                                     padding=True, 
                                     truncation=True,
                                     return_tensors="pt").to(device)
                
                with torch.amp.autocast('cuda',dtype=self.dtype):
                    with torch.no_grad():
                        _ = self.current_model.generate(
                            **inputs,
                            max_new_tokens=1,
                            num_return_sequences=1,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                
                optimal_size = mid
                left = mid + 1
                
                memory_used = torch.cuda.memory_allocated(device)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                
                if memory_used > 0.9 * total_memory:
                    break
                
            except torch.cuda.OutOfMemoryError:
                right = mid - 1
                torch.cuda.empty_cache()
        
        return max(1, int(optimal_size * 0.9))

    def optimize_generation_params(self, prompt: str) -> Dict[str, Any]:
        """Optimize generation parameters based on prompt characteristics"""
        base_params = {
            "max_new_tokens": self.model_config.max_new_tokens,
            "do_sample": self.model_config.do_sample,
            "top_p": self.model_config.top_p,
            "top_k": self.model_config.top_k,
            "temperature": self.model_config.temperature,
            "num_return_sequences": self.model_config.num_return_sequences,
            "repetition_penalty": self.model_config.repetition_penalty,
            "pad_token_id": self.model_config.pad_token_id or self.tokenizer.pad_token_id
        }
        
        if self.model_config.dynamic_temperature:
            base_params["temperature"] = self.dynamic_temperature.get_optimal_temperature(
                prompt, self.tokenizer, base_params["temperature"]
            )
        
        return base_params

    def format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format the prompt according to model's chat template"""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Use the model's built-in chat template if available
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback to a generic template
            return f"<|system|>{system_prompt}</s><|user|>{user_prompt}</s><|assistant|>"
        
    def _create_stopping_criteria(self, stop_sequences: List[str], input_length: int):
        """Create stopping criteria for generation"""
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class StopSequenceCriteria(StoppingCriteria):
            def __init__(self, tokenizer, stop_sequences, input_length):
                self.tokenizer = tokenizer
                self.stop_ids = [
                    self.tokenizer.encode(seq, add_special_tokens=False)
                    for seq in stop_sequences
                ]
                self.input_length = input_length

            def __call__(self, input_ids, scores, **kwargs):
                for stop_ids in self.stop_ids:
                    if input_ids[0, -len(stop_ids):].tolist() == stop_ids:
                        return True
                return False

        return StoppingCriteriaList([
            StopSequenceCriteria(
                self.tokenizer,
                stop_sequences,
                input_length=input_length
            )
        ])
    
    def process_batch(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        generation_params: Optional[Dict[str, Any]] = None,
        active_adapter: str = None,
        return_token_count: bool = True
    ) -> Tuple[List[str], List[int]]:
        """Process a batch of prompts with all optimizations"""
        
        # Set the requested adapter if specified
        if isinstance(self.current_model, PeftModel) and active_adapter is not None:
            self.lora_manager.set_active_adapter(self.current_model, active_adapter)

        all_responses = []
        token_counts = []
        
        # Format all prompts using chat template
        formatted_prompts = [
            self.format_chat_prompt(system_prompt, user_prompt)
            for system_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]
        
        # Get number of completions requested
        n = generation_params.get("num_return_sequences", 1) if generation_params else 1
        
        for i in range(0, len(formatted_prompts), self.optimal_batch_size):
            batch_prompts = formatted_prompts[i:i + self.optimal_batch_size]
            batch_system = system_prompts[i:i + self.optimal_batch_size]
            batch_user = user_prompts[i:i + self.optimal_batch_size]
            
            # Check cache first if enabled
            if self.model_config.enable_prompt_caching:
                cached_responses = []
                uncached_indices = []
                
                for idx, prompt in enumerate(batch_prompts):
                    temp = generation_params.get("temperature", self.model_config.temperature) if generation_params else self.model_config.temperature
                    top_p = generation_params.get("top_p", self.model_config.top_p) if generation_params else self.model_config.top_p
                    
                    cached_response = self.cache_manager.prompt_cache.get_cached_response(
                        prompt,
                        temp,
                        top_p
                    )
                    if cached_response is not None:
                        # For cached responses, replicate n times if multiple completions requested
                        cached_responses.extend([cached_response] * n)
                    else:
                        uncached_indices.append(idx)
                
                if uncached_indices:
                    batch_prompts = [batch_prompts[i] for i in uncached_indices]
                else:
                    batch_prompts = []
            
            if batch_prompts:  # If there are any uncached prompts
                # Configure generation parameters
                base_params = {
                    "max_new_tokens": generation_params.get("max_new_tokens", _default_max_new_tokens()) if generation_params else self.model_config.max_new_tokens,
                    "do_sample": generation_params.get("temperature", 1.0) > 0 if generation_params else self.model_config.do_sample,
                    "temperature": generation_params.get("temperature", 1.0) if generation_params else self.model_config.temperature,
                    "top_p": generation_params.get("top_p", 1.0) if generation_params else self.model_config.top_p,
                    "num_return_sequences": n,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self._resolve_eos_token_ids(),
                }

                # Add optional parameters if specified
                if generation_params:
                    if generation_params.get("presence_penalty", 0) != 0:
                        base_params["presence_penalty"] = generation_params["presence_penalty"]
                    if generation_params.get("frequency_penalty", 0) != 0:
                        base_params["repetition_penalty"] = 1.0 + generation_params["frequency_penalty"]
                    if generation_params.get("logit_bias"):
                        base_params["logit_bias"] = generation_params["logit_bias"]
                    if generation_params.get("seed") is not None:
                        torch.manual_seed(generation_params["seed"])
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(generation_params["seed"])
                
                # Tokenize inputs
                inputs = self.tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.current_model.device)

                # Get the length of each input
                input_lengths = inputs['input_ids'].shape[1]

                # Add stopping criteria if specified
                if generation_params and generation_params.get("stop_sequences"):
                    base_params["stopping_criteria"] = self._create_stopping_criteria(
                        generation_params["stop_sequences"],
                        input_lengths
                    )

                # Generate responses
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    with torch.no_grad():
                        outputs = self.current_model.generate(
                            **inputs,
                            **base_params
                        )
                
                # Decode outputs and remove input portion
                batch_responses = []
                batch_token_counts = []
                
                # Handle multiple sequences per input
                num_return_sequences = base_params["num_return_sequences"]
                for i in range(0, len(outputs), num_return_sequences):
                    sequences = outputs[i:i + num_return_sequences]
                    for seq in sequences:
                        response_tokens = seq[input_lengths:]
                        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                        batch_responses.append(response_text)
                        batch_token_counts.append(len(response_tokens))
                
                # Cache new responses if enabled
                if self.model_config.enable_prompt_caching:
                    for prompt, response in zip(batch_prompts, batch_responses[::n]):  # Only cache first response of each input
                        self.cache_manager.prompt_cache.add_to_cache(
                            prompt,
                            response,
                            base_params["temperature"],
                            base_params["top_p"]
                        )
                
                # Merge cached and new responses in correct order
                all_responses.extend(cached_responses)
                if uncached_indices:
                    response_idx = 0
                    for original_idx in range(len(formatted_prompts[i:i + self.optimal_batch_size])):
                        if original_idx in uncached_indices:
                            # Add n responses for this uncached prompt
                            for _ in range(n):
                                while len(all_responses) < original_idx * n + _:
                                    all_responses.append("")
                                if response_idx < len(batch_responses):
                                    all_responses.append(batch_responses[response_idx])
                                    response_idx += 1
                
                if return_token_count:
                    # Count tokens for responses
                    token_counts.extend([0] * len(cached_responses))  # 0 for cached responses
                    token_counts.extend(batch_token_counts)
        
        if return_token_count:
            return all_responses, token_counts
        return all_responses, [0] * len(all_responses)
    
class ChatCompletionMessage:
    def __init__(self, content: str, role: str = "assistant", logprobs: Optional[Dict] = None):
        self.content = content
        self.role = role
        self.logprobs = logprobs

class ChatCompletionChoice:
    def __init__(
        self,
        index: int,
        message: Dict[str, Any],
        finish_reason: str = "stop",
        logprobs: Optional[Dict] = None
    ):
        self.index = index
        self.message = ChatCompletionMessage(**message)
        self.finish_reason = finish_reason
        if logprobs:
            self.message.logprobs = logprobs

class ChatCompletionUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int, reasoning_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.reasoning_tokens = reasoning_tokens

class ChatCompletion:
    def __init__(self, response_dict: Dict):
        self.id = response_dict["id"]
        self.object = response_dict["object"]
        self.created = response_dict["created"]
        self.model = response_dict["model"]
        self.choices = [
            ChatCompletionChoice(
                index=choice["index"],
                message=choice["message"],
                finish_reason=choice["finish_reason"]
            )
            for choice in response_dict["choices"]
        ]
        self.usage = ChatCompletionUsage(**response_dict["usage"])
    
    def model_dump(self) -> Dict:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "logprobs": choice.message.logprobs
                    } if choice.message.logprobs else {
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason
                }
                for choice in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
                "completion_tokens_details": {
                    "reasoning_tokens": getattr(self.usage, 'reasoning_tokens', 0)
                }
            }
        }

class InferenceClient:
    """OpenAI SDK Compatible client for local inference with dynamic model support"""
    
    def __init__(self):
        self.cache_manager = CacheManager.get_instance(max_size=4)
        self.device_manager = DeviceManager()
        self.model_manager = ModelManager(self.cache_manager, self.device_manager)
        self.lora_manager = LoRAManager(self.cache_manager)
        self.mlx_manager = MLXManager(self.cache_manager)
        self.chat = self.Chat(self)
        self.models = self.Models()

    def get_pipeline(self, model: str):
        """Get inference pipeline - automatically chooses MLX or PyTorch based on model"""
        # Check if should use MLX
        if self.mlx_manager.available and should_use_mlx(model):
            logger.info(f"Using MLX pipeline for model: {model}")
            return self.mlx_manager.create_pipeline(model)
        else:
            # Use existing PyTorch pipeline
            logger.info(f"Using PyTorch pipeline for model: {model}")
            model_config = parse_model_string(model)
            return InferencePipeline(
                model_config,
                self.cache_manager,
                self.device_manager,
                self.model_manager,
                self.lora_manager
            )
    
    class Chat:
        """OpenAI-compatible chat interface"""
        def __init__(self, client: 'InferenceClient'):
            self.client = client
            self.completions = self.Completions(client)

        class Completions:
            def __init__(self, client: 'InferenceClient'):
                self.client = client

            def create(
                self,
                messages: List[Dict[str, str]],
                model: str,
                temperature: float = 1.0,
                top_p: float = 1.0,
                n: int = 1,
                stream: bool = False,
                stop: Optional[Union[str, List[str]]] = None,
                max_tokens: Optional[int] = None,
                presence_penalty: float = 0,
                frequency_penalty: float = 0,
                logit_bias: Optional[Dict[str, float]] = None,
                seed: Optional[int] = None,
                logprobs: Optional[bool] = None,
                top_logprobs: Optional[int] = None,
                active_adapter: Optional[Dict[str, Any]] = None,
                decoding: Optional[str] = None,
                # CoT specific params
                k: int = 10,
                num_beams: int = 1,
                length_penalty: float = 1.0,
                no_repeat_ngram_size: int = 0,
                early_stopping: bool = False,
                aggregate_paths: bool = True,
                # Entropy specific params
                top_k: int = 27,
                min_p: float = 0.03,
                # Thinking specific params
                reasoning_effort: str = "low",
                thought_switch_tokens: List[str] = [],
                min_thinking_tokens: Optional[int] = None,
                max_thinking_tokens: Optional[int] = None,
                max_thoughts: Optional[int] = None,
                prefill: str = "",
                start_think_token: str ="<think>",
                end_think_token: str = "</think>",
                **kwargs
            ) -> ChatCompletion:
                """Create a chat completion with OpenAI-compatible parameters"""
                logger.info("Starting chat completion creation")
                if stream:
                    raise NotImplementedError("Streaming is not yet supported")

                logger.info(f"Getting pipeline for model: {model}")
                pipeline = self.client.get_pipeline(model)
                logger.info("Pipeline acquired")

                # Set active adapter if specified
                if active_adapter is not None:
                    logger.info(f"Setting active adapter to: {active_adapter}")
                    pipeline.lora_manager.set_active_adapter(pipeline.current_model, active_adapter)

                responses = []
                logprobs_results = []
                prompt_tokens = 0
                completion_tokens = 0

                try:
                    # Handle specialized decoding approaches
                    if decoding:
                        logger.info(f"Using specialized decoding approach: {decoding}")

                        # Check if this decoding approach is supported for MLX
                        mlx_unsupported_decodings = ["cot_decoding", "entropy_decoding", "autothink", "deepconf"]
                        if isinstance(pipeline, MLXInferencePipeline) and decoding in mlx_unsupported_decodings:
                            logger.warning(f"{decoding} is not supported for MLX models. Falling back to standard generation.")
                            decoding = None
                        
                    if decoding:
                        # For PyTorch pipelines, ensure model is in eval mode and get device
                        # MLX pipelines handle this differently
                        if not isinstance(pipeline, MLXInferencePipeline):
                            pipeline.current_model.eval()
                            device = pipeline.current_model.device
                        else:
                            device = None  # MLX doesn't use torch devices
                        
                        if decoding == "cot_decoding":
                            # Use directly available parameters for CoT
                                cot_params = {
                                    "k": k,
                                    "num_beams": num_beams,
                                    "max_new_tokens": max_tokens if max_tokens is not None else 512,
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "repetition_penalty": 1.0,
                                    "length_penalty": length_penalty,
                                    "no_repeat_ngram_size": no_repeat_ngram_size,
                                    "early_stopping": early_stopping,
                                    "aggregate_paths": aggregate_paths,
                                }
                                
                                result, confidence = cot_decode(
                                    pipeline.current_model,
                                    pipeline.tokenizer,
                                    messages,
                                    **cot_params
                                )
                                responses = [result]
                                logprobs_results = [{"confidence_score": confidence} if confidence is not None else None]
                                completion_tokens = len(pipeline.tokenizer.encode(result))
                            
                        elif decoding == "entropy_decoding":
                            # Ensure model is using full precision
                            original_dtype = pipeline.current_model.dtype
                            pipeline.current_model = pipeline.current_model.to(torch.float32)

                            try:
                                # Configure generator for entropy decoding
                                generator = None
                                if seed is not None:
                                    generator = torch.Generator(device=device)
                                    generator.manual_seed(seed)
                                else:
                                    generator = torch.Generator(device=device)
                                    generator.manual_seed(1337)  # Default seed as in original implementation
                                
                                # Use directly available parameters for entropy decoding
                                entropy_params = {
                                    "max_new_tokens": max_tokens if max_tokens is not None else _default_max_new_tokens(),
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "top_k": top_k,
                                    "min_p": min_p,
                                    "generator": generator
                                }
                                
                                # Disable autocast and run in full precision
                                with torch.amp.autocast('cuda', enabled=False), torch.inference_mode():
                                    result = entropy_decode(
                                        pipeline.current_model,
                                        pipeline.tokenizer,
                                        messages,
                                        **entropy_params
                                    )
                                responses = [result]
                                logprobs_results = [None]
                                completion_tokens = len(pipeline.tokenizer.encode(result))
                            
                            finally:
                                # Restore original dtype
                                pipeline.current_model = pipeline.current_model.to(original_dtype)

                        elif decoding == "thinkdeeper":
                            # Get base config for reasoning effort
                            thinkdeeper_config = get_effort_profile(reasoning_effort, max_tokens)
                            
                            # Override with any custom parameters
                            custom_config = {
                                "min_thinking_tokens": min_thinking_tokens if min_thinking_tokens is not None else thinkdeeper_config["min_thinking_tokens"],
                                "max_thinking_tokens": max_thinking_tokens if max_thinking_tokens is not None else thinkdeeper_config["max_thinking_tokens"],
                                "max_thoughts": max_thoughts if max_thoughts is not None else thinkdeeper_config["max_thoughts"],
                                "thought_switch_tokens": thought_switch_tokens if thought_switch_tokens else thinkdeeper_config["thought_switch_tokens"],
                                "prefill": prefill if prefill else thinkdeeper_config["prefill"],
                                "start_think_token": start_think_token,
                                "end_think_token": end_think_token,
                            }
                            thinkdeeper_config.update(custom_config)

                            # Check if we're using MLX pipeline
                            if isinstance(pipeline, MLXInferencePipeline):
                                logger.info("Using MLX ThinkDeeper implementation")
                                
                                # Ensure we have enough tokens for thinking + response
                                user_max_tokens = max_tokens if max_tokens is not None else 512
                                total_tokens_needed = max_thinking_tokens + 512  # thinking + response buffer
                                adjusted_max_tokens = max(user_max_tokens, total_tokens_needed)
                                
                                # Add max_tokens to thinkdeeper config
                                thinkdeeper_config_with_tokens = thinkdeeper_config.copy()
                                thinkdeeper_config_with_tokens["max_tokens"] = adjusted_max_tokens
                                
                                logger.debug(f"ThinkDeeper tokens: user={user_max_tokens}, thinking={max_thinking_tokens}, adjusted={adjusted_max_tokens}")
                                
                                result, reasoning_tokens = thinkdeeper_decode_mlx(
                                    pipeline.model,
                                    pipeline.tokenizer,
                                    messages,
                                    thinkdeeper_config_with_tokens
                                )
                            else:
                                logger.info("Using PyTorch ThinkDeeper implementation")
                                result, reasoning_tokens = thinkdeeper_decode(
                                    pipeline.current_model,
                                    pipeline.tokenizer,
                                    messages,
                                    thinkdeeper_config
                                )
                            responses = [result]
                            logprobs_results = [None]
                            completion_tokens = len(pipeline.tokenizer.encode(result))
                        elif decoding == "autothink":
                            # Get steering dataset configuration
                                steering_dataset = kwargs.get("steering_dataset", "codelion/Qwen3-0.6B-pts-steering-vectors")
                                target_layer = kwargs.get("target_layer", 19)
                                
                                # Prepare AutoThink configuration
                                autothink_config = {
                                    "steering_dataset": steering_dataset,
                                    "target_layer": target_layer,
                                    "pattern_strengths": kwargs.get("pattern_strengths", {
                                        "depth_and_thoroughness": 2.5,
                                        "numerical_accuracy": 2.0,
                                        "self_correction": 3.0,
                                        "exploration": 2.0,
                                        "organization": 1.5
                                    })
                                }
                                
                                # Process with AutoThink
                                result = autothink_decode(
                                    pipeline.current_model,
                                    pipeline.tokenizer,
                                    messages,
                                    autothink_config
                                )
                                responses = [result]
                                logprobs_results = [None]
                                completion_tokens = len(pipeline.tokenizer.encode(result))
                        elif decoding == "deepconf":
                            # Prepare DeepConf configuration
                            deepconf_config = {
                                "variant": kwargs.get("variant", "low"),  # "low" or "high"
                                "warmup_samples": kwargs.get("warmup_samples", 16),
                                "consensus_threshold": kwargs.get("consensus_threshold", 0.95),
                                "max_traces": kwargs.get("max_traces", 128),
                                "window_size": kwargs.get("window_size", 2048),
                                "top_k": kwargs.get("top_k", 5),
                                "min_trace_length": kwargs.get("min_trace_length", 100),
                                "max_tokens_per_trace": kwargs.get("max_tokens_per_trace", 4096),
                                "temperature": temperature,
                                "confidence_metric": kwargs.get("confidence_metric", "average_confidence"),
                                "include_stats": kwargs.get("include_stats", False)
                            }
                            
                            # Process with DeepConf
                            result, tokens_used = deepconf_decode(
                                pipeline.current_model,
                                pipeline.tokenizer,
                                messages,
                                deepconf_config
                            )
                            responses = [result]
                            logprobs_results = [None]
                            completion_tokens = tokens_used
                        else:
                            raise ValueError(f"Unknown specialized decoding approach: {decoding}")
                        
                        # Calculate prompt tokens for specialized approaches
                        prompt_text = pipeline.tokenizer.apply_chat_template(messages, tokenize=False)
                        prompt_tokens = len(pipeline.tokenizer.encode(prompt_text))
                        
                    else:
                        # Standard generation
                        prompt = pipeline.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        
                        # Set generation parameters
                        generation_params = {
                            "temperature": temperature,
                            "top_p": top_p,
                            "num_return_sequences": n,
                            "max_new_tokens": max_tokens if max_tokens is not None else _default_max_new_tokens(),
                            "presence_penalty": presence_penalty,
                            "frequency_penalty": frequency_penalty,
                            "stop_sequences": [stop] if isinstance(stop, str) else stop,
                            "seed": seed,
                            "logit_bias": logit_bias,
                            "logprobs": logprobs,
                            "top_logprobs": top_logprobs
                        }

                        # Generate responses
                        responses, token_counts, logprobs_results = pipeline.generate(
                            prompt,
                            generation_params=generation_params
                        )
                        
                        prompt_tokens = len(pipeline.tokenizer.encode(prompt))
                        completion_tokens = sum(token_counts)

                    # Calculate reasoning tokens from all responses
                    total_reasoning_tokens = 0
                    for response in responses:
                        total_reasoning_tokens += count_reasoning_tokens(response, pipeline.tokenizer)

                    # Create OpenAI-compatible response format
                    response_dict = {
                        "id": f"chatcmpl-{int(time.time()*1000)}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": idx,
                                "message": {
                                    "role": "assistant",
                                    "content": response,
                                    **({"logprobs": logprob_result} if logprob_result else {})
                                },
                                "finish_reason": "stop"
                            }
                            for idx, (response, logprob_result) in enumerate(zip(responses, logprobs_results))
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": completion_tokens + prompt_tokens,
                            "reasoning_tokens": total_reasoning_tokens
                        }
                    }
                    
                    logger.debug(f"Response : {response_dict}")
                    return ChatCompletion(response_dict)
                    
                except Exception as e:
                    logger.error(f"Error in chat completion: {str(e)}")
                    raise
                
    class Models:
        """OpenAI-compatible models interface"""
        def list(self):
            """Return list of supported models"""
            try:
                import requests
                response = requests.get(
                    "https://huggingface.co/api/models?sort=downloads&direction=-1&filter=text-generation&limit=20"
                )
                models = response.json()
                model_list = []
                
                for model in models:
                    if 'pipeline_tag' in model and model['pipeline_tag'] == 'text-generation':
                        model_list.append({
                            "id": model['id'],
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "huggingface",
                        })
                
                return {"data": model_list, "object": "list"}
            except Exception as e:
                logger.warning(f"Failed to fetch models: {e}")
                return {
                    "data": [{
                        "id": "HuggingFaceTB/SmolLM-135M-Instruct",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "huggingface",
                    }],
                    "object": "list"
                }
            
def create_inference_client() -> InferenceClient:
    """Factory function to create an inference client"""
    return InferenceClient()

def parse_model_string(model: str) -> ModelConfig:
    """Parse the model string to extract base model and adapter IDs"""
    parts = model.split('+')
    base_model_id = parts[0]
    adapter_ids = parts[1:] if len(parts) > 1 else None
    
    return ModelConfig(
        base_model_id=base_model_id,
        adapter_ids=adapter_ids,
        use_memory_efficient_attention=False,
        quantization_bits=0,
        enable_prompt_caching=False,
        dynamic_temperature=False,
    )

def get_effort_profile(reasoning_effort: str, max_tokens: int = 4096) -> dict:
    """Get reasoning effort profile based on specified level and max tokens.
    
    Args:
        reasoning_effort: 'low', 'medium', or 'high'
        max_tokens: Maximum tokens allowed for generation, defaults to 4096
    
    Returns:
        dict: Configuration for the specified reasoning effort level
    """
    # Base profiles with percentages and thought counts
    profiles = {
        "low": {
            "min_tokens_pct": 0.10,  
            "max_tokens_pct": 0.33,  # 33% of max_tokens
            "max_thoughts": 64,
            "thought_switch_tokens": [
                "Wait,",
                "Alternatively,",
                "However,",
                "Additionally,",
            ],
            "prefill": ""
        },
        "medium": {
            "min_tokens_pct": 0.10,  
            "max_tokens_pct": 0.66,  # 66% of max_tokens
            "max_thoughts": 256,
            "thought_switch_tokens": [
                "Wait,",
                "Alternatively,",
                "However,",
                "Additionally,",
            ],
            "prefill": ""
        },
        "high": {
            "min_tokens_pct": 0.10,  
            "max_tokens_pct": 0.90,  # 90% of max_tokens
            "max_thoughts": 512,
            "thought_switch_tokens": [
                "Wait,",
                "Alternatively,",
                "However,",
                "Additionally,",
            ],
            "prefill": ""
        }
    }
    
    # Get base profile or default to medium
    profile = profiles.get(reasoning_effort.lower(), profiles["low"])
    
    # Calculate actual token limits based on max_tokens
    min_thinking_tokens = int(max_tokens * profile["min_tokens_pct"])
    max_thinking_tokens = int(max_tokens * profile["max_tokens_pct"])
    
    # Create final config
    config = {
        "min_thinking_tokens": min_thinking_tokens,
        "max_thinking_tokens": max_thinking_tokens,
        "max_thoughts": profile["max_thoughts"],
        "thought_switch_tokens": profile["thought_switch_tokens"],
        "prefill": profile["prefill"]
    }
    
    return config