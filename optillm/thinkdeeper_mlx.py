"""
MLX-compatible implementation of ThinkDeeper
Provides the same functionality as the PyTorch version but adapted for MLX framework
"""

import random
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    import mlx.core as mx
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

DEFAULT_CONFIG = {
    "min_thinking_tokens": 1024,
    "max_thinking_tokens": 4196,  
    "max_thoughts": 64,  
    "prefill": "",
    "start_think_token": "<think>",
    "end_think_token": "</think>",
    "thought_switch_tokens": [
        "Wait,",
        "Alternatively,",
    ],
}

class MLXThinkDeeperProcessor:
    def __init__(self, config: Dict[str, Any], tokenizer, model):
        self.config = {**DEFAULT_CONFIG, **config}
        self.tokenizer = tokenizer
        self.model = model
        
        # Get token IDs for think markers
        start_tokens = self.tokenizer.encode(self.config['start_think_token'])
        end_tokens = self.tokenizer.encode(self.config['end_think_token'])
        self._start_think_token = start_tokens[0] if len(start_tokens) == 1 else start_tokens[1]
        self.end_think_token = end_tokens[0] if len(end_tokens) == 1 else end_tokens[1]
        
        # Store thought switch markers as token sequences
        self.thought_switch_sequences = []
        for phrase in self.config["thought_switch_tokens"]:
            # Encode without adding special tokens to get exact sequence
            token_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            self.thought_switch_sequences.append(token_ids)
        
        # Track thought switches
        self.thought_count = 0
        self.current_sequence = []  # Track recent tokens for sequence matching
        self.max_sequence_length = max(len(seq) for seq in self.thought_switch_sequences) if self.thought_switch_sequences else 5
        
        # Track total tokens for budget management
        self.total_tokens_generated = 0
        self.max_total_tokens = config.get('max_tokens', 8192)  # Default to 8192 if not specified

    def is_thought_switch(self, token: int) -> bool:
        """Check if adding this token creates a thought switch sequence."""
        # Add new token to current sequence
        self.current_sequence.append(token)
        
        # Keep only the most recent tokens that could match our sequences
        if len(self.current_sequence) > self.max_sequence_length:
            self.current_sequence = self.current_sequence[-self.max_sequence_length:]
        
        # Check if current sequence ends with any thought switch sequence
        for sequence in self.thought_switch_sequences:
            if len(sequence) <= len(self.current_sequence) and \
               self.current_sequence[-len(sequence):] == sequence:
                return True
        
        return False
    
    def reasoning_effort(self, messages) -> str:
        """Generate response with ThinkDeeper's controlled thinking process using MLX"""
        
        # Prepare the messages with thinking token
        thinking_messages = messages.copy()
        thinking_messages.append({
            "role": "assistant", 
            "content": f"{self.config['start_think_token']}\n{self.config['prefill']}"
        })

        # Convert messages to prompt using tokenizer
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                thinking_messages,
                continue_final_message=False,  # This was causing MLX failures!
                tokenize=False,
                add_generation_prompt=True  # Standard generation prompt
            )
        else:
            # Fallback: simple concatenation
            prompt = ""
            for msg in thinking_messages:
                prompt += f"{msg['role']}: {msg['content']}\n"
        
        
        # Initialize tracking variables
        n_thinking_tokens = 0
        seen_end_think = False
        response_chunks = []
        
        # Use MLX generation with custom token-by-token control
        # Since MLX doesn't support token-by-token generation like PyTorch,
        # we'll use a different approach: generate in chunks and check for markers
        
        current_prompt = prompt
        max_chunk_size = 150  # Increase chunk size - MLX may work better with larger chunks
        consecutive_empty_chunks = 0
        max_empty_chunks = 3  # Allow up to 3 consecutive empty chunks before stopping
        
        while (n_thinking_tokens < self.config["max_thinking_tokens"] and 
               self.thought_count < self.config["max_thoughts"] and 
               self.total_tokens_generated < self.max_total_tokens - 512):  # Reserve 512 tokens for final response
            try:
                # Generate a small chunk of tokens
                chunk_response = self._generate_chunk(
                    current_prompt, 
                    max_tokens=min(max_chunk_size, self.config["max_thinking_tokens"] - n_thinking_tokens),
                    temperature=0.6
                )
                
                if not chunk_response or chunk_response.strip() == "":
                    consecutive_empty_chunks += 1
                    
                    if consecutive_empty_chunks >= max_empty_chunks:
                        break
                    
                    # Try with different parameters for next attempt
                    max_chunk_size = min(max_chunk_size + 50, 300)  # Increase chunk size more aggressively
                    continue
                else:
                    # Reset empty chunk counter on successful generation
                    consecutive_empty_chunks = 0
                    max_chunk_size = 150  # Reset chunk size
                    
                    # Update token counts
                    chunk_tokens = len(self.tokenizer.encode(chunk_response))
                    self.total_tokens_generated += chunk_tokens
                
                # Check for end think token in the chunk
                if self.config['end_think_token'] in chunk_response:
                    # Split at the end think token
                    parts = chunk_response.split(self.config['end_think_token'], 1)
                    before_end = parts[0]
                    after_end = parts[1] if len(parts) > 1 else ""
                    
                    response_chunks.append(before_end)
                    n_thinking_tokens += len(self.tokenizer.encode(before_end))
                    
                    # Check if we've reached minimum thinking tokens
                    if n_thinking_tokens < self.config["min_thinking_tokens"]:
                        # Insert thought transition instead of ending
                        transition = random.choice(self.config["thought_switch_tokens"])
                        response_chunks.append(transition)
                        current_prompt += before_end + transition
                        n_thinking_tokens += len(self.tokenizer.encode(transition))
                        self.thought_count += 1
                        continue
                    else:
                        # Natural end - add the end token and continue for conclusion
                        response_chunks.append(self.config['end_think_token'])
                        current_prompt += before_end + self.config['end_think_token']
                        seen_end_think = True
                        
                        # Generate conclusion after thinking
                        if after_end.strip():
                            response_chunks.append(after_end)
                        else:
                            conclusion = self._generate_chunk(current_prompt, max_tokens=200, temperature=0.3)
                            if conclusion:
                                response_chunks.append(conclusion)
                        break
                else:
                    # No end think token found, add the chunk and continue
                    response_chunks.append(chunk_response)
                    current_prompt += chunk_response
                    n_thinking_tokens += len(self.tokenizer.encode(chunk_response))
                    
                    # Check for thought switch patterns in the chunk
                    for phrase in self.config["thought_switch_tokens"]:
                        if phrase in chunk_response:
                            self.thought_count += 1
                            break
                
                # Safety check to avoid infinite loops
                if len(response_chunks) > 100:
                    logger.warning("Too many chunks generated, stopping to avoid infinite loop")
                    break
                    
            except Exception as e:
                logger.error(f"Error during MLX chunk generation: {str(e)}")
                break
        
        # Enforce minimum thinking tokens if not reached
        if not seen_end_think and n_thinking_tokens < self.config["min_thinking_tokens"]:
            while n_thinking_tokens < self.config["min_thinking_tokens"] and self.thought_count < self.config["max_thoughts"]:
                # Add transition and continue thinking
                transition = random.choice(self.config["thought_switch_tokens"])
                response_chunks.append(f" {transition} ")
                current_prompt += f" {transition} "
                
                # Generate more thinking content
                additional_thinking = self._generate_chunk(
                    current_prompt,
                    max_tokens=min(200, self.config["min_thinking_tokens"] - n_thinking_tokens + 100),
                    temperature=0.6
                )
                
                if additional_thinking and additional_thinking.strip():
                    response_chunks.append(additional_thinking)
                    current_prompt += additional_thinking
                    additional_tokens = len(self.tokenizer.encode(additional_thinking))
                    n_thinking_tokens += additional_tokens
                    self.thought_count += 1
                else:
                    # If generation fails, break to avoid infinite loop
                    break
        
        # If we haven't seen end think token, force it
        if not seen_end_think:
            response_chunks.append(self.config['end_think_token'])
            
            # Add a brief conclusion
            try:
                conclusion = self._generate_chunk(
                    current_prompt + self.config['end_think_token'],
                    max_tokens=100,
                    temperature=0.3
                )
                if conclusion:
                    response_chunks.append(conclusion)
            except Exception as e:
                logger.error(f"Error generating conclusion: {str(e)}")
        
        # Join all chunks and create final response
        response_content = "".join(response_chunks)
        full_response = f"{self.config['start_think_token']}\n{self.config['prefill']}{response_content}"
        
        logger.debug(f"MLX Final response length: {len(full_response)} chars, Thinking tokens: {n_thinking_tokens}")
        return full_response, n_thinking_tokens
    
    def _generate_chunk(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate a small chunk of text using MLX with proper sampler"""
        try:
            # Let MLX fail naturally to identify the real issue
            
            # Create sampler with specified thinkdeeper parameters
            sampler = make_sampler(
                temp=temperature,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                min_tokens_to_keep=3
            )
            
            # Use mlx_generate with the sampler
            # Ensure we have minimum tokens to generate - larger minimum for better MLX performance
            actual_max_tokens = max(max_tokens, 30)  # At least 30 tokens for better generation
            
            response = mlx_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=actual_max_tokens,
                sampler=sampler,
                verbose=False
            )
            
            # MLX generate might return just the generated tokens or the full text
            # Check if response starts with the prompt
            if response:
                if response.startswith(prompt):
                    # Response includes the prompt, extract new content
                    new_content = response[len(prompt):]
                else:
                    # Response is just the generated tokens
                    new_content = response
                
                if new_content.strip():  # Only return non-empty content
                    return new_content
            
            return ""
            
        except Exception as e:
            logger.error(f"Error in MLX chunk generation: {str(e)}")
            return ""

def thinkdeeper_decode_mlx(
    model, 
    tokenizer, 
    messages: List[Dict[str, str]], 
    request_config: Dict[str, Any] = None
) -> str:
    """MLX-compatible ThinkDeeper processing function"""
    logger.info("Starting MLX ThinkDeeper processing")
    
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX framework not available for ThinkDeeper processing")
    
    # Extract config from request_config if provided
    config = DEFAULT_CONFIG.copy()
    if request_config:
        # Update only valid keys from DEFAULT_CONFIG
        for key in DEFAULT_CONFIG:
            if key in request_config:
                config[key] = request_config[key]
        
        # Also handle max_tokens which is not in DEFAULT_CONFIG
        if 'max_tokens' in request_config:
            config['max_tokens'] = request_config['max_tokens']

    logger.info(f"MLX ThinkDeeper using config: {config}")
    
    try:
        processor = MLXThinkDeeperProcessor(config, tokenizer, model)
        response, reasoning_tokens = processor.reasoning_effort(messages)
        return response, reasoning_tokens
        
    except Exception as e:
        logger.error(f"Error in MLX ThinkDeeper processing: {str(e)}")
        raise