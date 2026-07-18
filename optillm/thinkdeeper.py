import torch
import random
from transformers import PreTrainedModel, PreTrainedTokenizer, DynamicCache
from typing import Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

class ThinkDeeperProcessor:
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
            logger.debug(f"Encoded '{phrase}' to token sequence: {token_ids}")
            logger.debug(f"Decoded back: {self.tokenizer.decode(token_ids)}")
        
        # Track thought switches
        self.thought_count = 0
        self.current_sequence = []  # Track recent tokens for sequence matching
        self.max_sequence_length = max(len(seq) for seq in self.thought_switch_sequences)

        for phrase, sequence in zip(self.config["thought_switch_tokens"], self.thought_switch_sequences):
            logger.debug(f"Thought switch marker '{phrase}' encoded as: {sequence}")
            logger.debug(f"Decoded back as: {self.tokenizer.decode(sequence)}")

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
        
    @torch.inference_mode()
    def reasoning_effort(self, messages) -> str:
        """Generate response with ThinkDeeper's controlled thinking process"""
        
        messages.append({"role": "assistant", "content": f"{self.config['start_think_token']}\n{self.config['prefill']}"})

        tokens = self.tokenizer.apply_chat_template(
            messages,
            continue_final_message=True,
            return_tensors="pt"
        ).input_ids.to(self.model.device)

        kv = DynamicCache()
        n_thinking_tokens = 0
        seen_end_think = False
        response_chunks = []
        
        while True:
            out = self.model(input_ids=tokens, past_key_values=kv, use_cache=True)
            logits = out.logits[0, -1, :]
            
            # Check if we need to force end thinking
            force_end = (n_thinking_tokens >= self.config["max_thinking_tokens"] or 
                        self.thought_count >= self.config["max_thoughts"])
            
            if force_end and not seen_end_think:
                logger.debug(f"Forcing end think token. Tokens: {n_thinking_tokens}, Thoughts: {self.thought_count}")
                next_token = self.end_think_token
                response_chunks.append(self.tokenizer.decode([next_token]))
                seen_end_think = True
                # Don't break - continue generating but with end_think token forced
                tokens = torch.tensor([[next_token]]).to(tokens.device)
                continue
            else:
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), 1
                ).item()
            
            kv = out.past_key_values
            next_str = self.tokenizer.decode([next_token])
            
            # Check if this is a thought-switching token (only if not in conclusion phase)
            if not seen_end_think and self.is_thought_switch(next_token):
                self.thought_count += 1
                logger.debug(f"Detected thought switch marker. Total thoughts: {self.thought_count}")
                # Clear the sequence after detecting a switch
                self.current_sequence = []

            # Handle natural end think token
            if next_token == self.end_think_token:
                seen_end_think = True
                logger.debug("Found end think token")
                
                # If we haven't reached minimum tokens, continue with thought transition
                if n_thinking_tokens < self.config["min_thinking_tokens"]:
                    replacement = random.choice(self.config["thought_switch_tokens"])
                    logger.debug(f"Inserting thought transition: '{replacement}' (tokens: {n_thinking_tokens})")
                    response_chunks.append(replacement)
                    replacement_tokens = self.tokenizer.encode(replacement)
                    n_thinking_tokens += len(replacement_tokens)
                    tokens = torch.tensor([replacement_tokens]).to(tokens.device)
                    self.thought_count += 1
                    seen_end_think = False
                    continue

            # Handle EOS token
            if next_token == self.model.config.eos_token_id:
                logger.debug("Found eos token")
                if seen_end_think:
                    logger.debug("Reached EOS after end think token - stopping generation")
                    response_chunks.append(next_str)
                    break
                elif n_thinking_tokens < self.config["min_thinking_tokens"]:
                    # Continue with thought transition if under minimum tokens
                    replacement = random.choice(self.config["thought_switch_tokens"])
                    logger.debug(f"Inserting thought transition: '{replacement}' (tokens: {n_thinking_tokens})")
                    response_chunks.append(replacement)
                    replacement_tokens = self.tokenizer.encode(replacement)
                    n_thinking_tokens += len(replacement_tokens)
                    tokens = torch.tensor([replacement_tokens]).to(tokens.device)
                    self.thought_count += 1
                    continue
                else:
                    # Force end think token and continue generating for natural conclusion
                    logger.debug("Reached EOS without end think token - adding end token and continuing generation")
                    response_chunks.append(self.tokenizer.decode([self.end_think_token]))
                    tokens = torch.tensor([[self.end_think_token]]).to(tokens.device)
                    seen_end_think = True
                    continue
            
            # Normal token processing
            response_chunks.append(next_str)
            if not seen_end_think:
                n_thinking_tokens += 1
            tokens = torch.tensor([[next_token]]).to(tokens.device)

        # Join all chunks and add framing tokens
        response = "".join(response_chunks)
        full_response = f"{self.config['start_think_token']}\n{self.config['prefill']}{response}"
        
        logger.debug(f"Final response length: {len(full_response)} chars, Total thoughts: {self.thought_count}, Thinking tokens: {n_thinking_tokens}")
        return full_response, n_thinking_tokens

def thinkdeeper_decode(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    messages: List[Dict[str, str]], 
    request_config: Dict[str, Any] = None
) -> str:
    """Main plugin execution function with ThinkDeeper's controlled thinking process"""
    logger.info("Starting ThinkDeeper processing")
    
    # Extract config from request_config if provided
    config = DEFAULT_CONFIG.copy()
    if request_config:
        # Update only valid keys
        for key in DEFAULT_CONFIG:
            if key in request_config:
                config[key] = request_config[key]

    logger.info(f"Using config: {config}")
    
    try:
        processor = ThinkDeeperProcessor(config, tokenizer, model)
        response, reasoning_tokens = processor.reasoning_effort(messages)
        return response, reasoning_tokens
        
    except Exception as e:
        logger.error(f"Error in ThinkDeeper processing: {str(e)}")
        raise