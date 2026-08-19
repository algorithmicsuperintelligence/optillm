"""
AutoThink processor implementation.

This module implements the AutoThink processor for controlled thinking
with query complexity classification and steering vectors.
"""

import torch
import random
import logging
from transformers import PreTrainedModel, PreTrainedTokenizer, DynamicCache
from typing import Dict, List, Any, Optional, Union, Tuple

from .classifier import ComplexityClassifier
from .steering import SteeringVectorManager, install_steering_hooks, remove_steering_hooks

logger = logging.getLogger(__name__)

# Default configurations
DEFAULT_CONFIG = {
    # General configuration
    "min_thinking_tokens": 256,
    "max_thinking_tokens": 2048,
    "max_thoughts": 64,
    "prefill": "",
    "start_think_token": "<think>",
    "end_think_token": "</think>",
    
    # Complexity-specific configurations
    "high_complexity_min_tokens": 1024, 
    "high_complexity_max_tokens": 4096,
    "low_complexity_min_tokens": 256,
    "low_complexity_max_tokens": 1024,
    
    # Thought switch tokens
    "thought_switch_tokens": [
        "Wait,",
        "Alternatively,",
        "However,",
        "Additionally,",
        "Let's consider,",
        "On second thought,",
        "Actually,",
        "Furthermore,",
        "Looking at it differently,",
        "To be thorough,"
    ],
    
    # Classifier configuration
    "classifier_model": "adaptive-classifier/llm-router",
    "complexity_threshold": 0.6,
    
    # Steering configuration
    "steering_dataset": "",
    "target_layer": 19,
    "pattern_strengths": {
        "depth_and_thoroughness": 2.5,
        "numerical_accuracy": 2.0,
        "self_correction": 3.0,
        "exploration": 2.0,
        "organization": 1.5
    }
}

class AutoThinkProcessor:
    """
    AutoThink processor for controlled thinking with 
    complexity classification and steering vectors.
    """
    
    def __init__(self, config: Dict[str, Any], tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
        """
        Initialize the AutoThink processor.
        
        Args:
            config: Configuration dictionary
            tokenizer: Model tokenizer
            model: Language model
        """
        # Merge default config with provided config
        self.config = {**DEFAULT_CONFIG, **config}
        self.tokenizer = tokenizer
        self.model = model
        
        # Initialize classifier
        self.classifier = ComplexityClassifier(self.config["classifier_model"])
        
        # Get token IDs for think markers
        start_tokens = self.tokenizer.encode(self.config['start_think_token'])
        end_tokens = self.tokenizer.encode(self.config['end_think_token'])
        self._start_think_token = start_tokens[0] if len(start_tokens) == 1 else start_tokens[1]
        self.end_think_token = end_tokens[0] if len(end_tokens) == 1 else end_tokens[1]
        
        # Store thought switch markers as token sequences
        self.thought_switch_sequences = []
        for phrase in self.config["thought_switch_tokens"]:
            token_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            self.thought_switch_sequences.append(token_ids)
            logger.debug(f"Encoded '{phrase}' to token sequence: {token_ids}")
            logger.debug(f"Decoded back: {self.tokenizer.decode(token_ids)}")
        
        # Track thought switches
        self.thought_count = 0
        self.current_sequence = []  # Track recent tokens for sequence matching
        self.max_sequence_length = max(len(seq) for seq in self.thought_switch_sequences)
        
        # Initialize steering vector manager and hooks if dataset is provided
        self.steering_manager = None
        self.steering_hooks = []
        
        if self.config["steering_dataset"]:
            self._setup_steering()
    
    def _setup_steering(self):
        """Set up steering vector management."""
        try:
            # Initialize steering vector manager
            self.steering_manager = SteeringVectorManager(
                dataset_name=self.config["steering_dataset"],
                target_layer=self.config["target_layer"]
            )
            
            # Set pattern strengths
            if "pattern_strengths" in self.config:
                for pattern, strength in self.config["pattern_strengths"].items():
                    self.steering_manager.set_steering_strength(pattern, strength)
            
            # Create tokenized contexts for efficient matching
            self.steering_manager.create_tokenized_contexts(self.tokenizer)
            
            # Install hooks on the model
            self.steering_hooks = install_steering_hooks(
                self.model, 
                self.steering_manager,
                self.tokenizer
            )
            
            logger.info(f"STEERING: Set up steering with {len(self.steering_hooks)} hooks")
        
        except Exception as e:
            logger.error(f"STEERING: Error setting up steering: {e}")
            self.steering_manager = None
            self.steering_hooks = []
    
    def _cleanup_steering(self):
        """Clean up steering hooks."""
        if self.steering_hooks:
            remove_steering_hooks(self.steering_hooks)
            self.steering_hooks = []
            logger.info("STEERING: Hooks removed successfully")
    
    def classify_complexity(self, query: str) -> Tuple[str, float]:
        """
        Classify query complexity.
        
        Args:
            query: The query to classify
            
        Returns:
            Tuple of (complexity_label, confidence_score)
        """
        complexity, confidence = self.classifier.get_complexity_with_confidence(query)
        logger.info(f"Query classified as {complexity} with confidence {confidence:.2f}")
        return complexity, confidence
    
    def get_token_budget(self, complexity: str) -> Tuple[int, int]:
        """
        Get token budget based on complexity.
        
        Args:
            complexity: Complexity label (HIGH or LOW)
            
        Returns:
            Tuple of (min_tokens, max_tokens)
        """
        if complexity == "HIGH":
            return (
                self.config["high_complexity_min_tokens"],
                self.config["high_complexity_max_tokens"]
            )
        else:
            return (
                self.config["low_complexity_min_tokens"],
                self.config["low_complexity_max_tokens"]
            )
    
    def is_thought_switch(self, token: int) -> bool:
        """
        Check if adding this token creates a thought switch sequence.
        
        Args:
            token: Token ID to check
            
        Returns:
            Boolean indicating if this completes a thought switch
        """
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
    def process(self, messages: List[Dict[str, str]]) -> str:
        """
        Process messages with AutoThink's controlled thinking.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Generated response
        """
        try:
            # Extract the query from the messages
            query = self._extract_query(messages)
            
            # Classify query complexity
            complexity, confidence = self.classify_complexity(query)
            
            # Get token budget based on complexity
            min_tokens, max_tokens = self.get_token_budget(complexity)
            logger.info(f"Using token budget: {min_tokens}-{max_tokens} for {complexity} complexity")
            
            # Prepare messages with thinking start token
            thinking_messages = messages.copy()
            thinking_messages.append({
                "role": "assistant", 
                "content": f"{self.config['start_think_token']}\n{self.config['prefill']}"
            })
            
            # Tokenize the messages
            tokens = self.tokenizer.apply_chat_template(
                thinking_messages,
                continue_final_message=True,
                return_tensors="pt"
            ).input_ids.to(self.model.device)
            
            # Reset and update token history in steering hooks
            if self.steering_hooks:
                token_ids = tokens[0].tolist()
                prompt_text = self.tokenizer.decode(token_ids)
                for hook, _ in self.steering_hooks:
                    # Reset the hook state for a new generation
                    hook.reset()
                    # Update both token history and text context buffer
                    hook.update_token_history(token_ids)
                    hook.update_context(prompt_text)
                    # Try to match with a steering vector
                    hook.try_match()
            
            # Generate with controlled thinking
            kv = DynamicCache()
            n_thinking_tokens = 0
            seen_end_think = False
            response_chunks = []
            
            while True:
                out = self.model(input_ids=tokens, past_key_values=kv, use_cache=True)
                logits = out.logits[0, -1, :]
                
                # Check if we need to force end thinking
                force_end = (n_thinking_tokens >= max_tokens or 
                            self.thought_count >= self.config["max_thoughts"])
                
                if force_end and not seen_end_think:
                    logger.debug(f"Forcing end think token. Tokens: {n_thinking_tokens}, Thoughts: {self.thought_count}")
                    next_token = self.end_think_token
                    response_chunks.append(self.tokenizer.decode([next_token]))
                    seen_end_think = True
                    tokens = torch.tensor([[next_token]]).to(tokens.device)
                    continue
                else:
                    next_token = torch.multinomial(
                        torch.softmax(logits, dim=-1), 1
                    ).item()
                
                kv = out.past_key_values
                next_str = self.tokenizer.decode([next_token])
                
                # Update steering hooks with new token
                if self.steering_hooks:
                    for hook, _ in self.steering_hooks:
                        hook.update_token_history([next_token])
                
                # Check if this is a thought-switching token (only if not in conclusion phase)
                if not seen_end_think and self.is_thought_switch(next_token):
                    self.thought_count += 1
                    logger.debug(f"Detected thought switch marker. Total thoughts: {self.thought_count}")
                    self.current_sequence = []
                
                # Handle natural end think token
                if next_token == self.end_think_token:
                    seen_end_think = True
                    logger.debug("Found end think token")
                    
                    # If we haven't reached minimum tokens, continue with thought transition
                    if n_thinking_tokens < min_tokens:
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
                    logger.debug("Found EOS token")
                    if seen_end_think:
                        logger.debug("Reached EOS after end think token - stopping generation")
                        response_chunks.append(next_str)
                        break
                    elif n_thinking_tokens < min_tokens:
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
                
                # Update steering hooks with new token
                if self.steering_hooks:
                    for hook, _ in self.steering_hooks:
                        # Update both token history and text context
                        hook.update_token_history([next_token])
                        hook.update_context(next_str)
                        # Check for matches on EVERY token
                        hook.try_match()
                
                tokens = torch.tensor([[next_token]]).to(tokens.device)
            
            # Reset and clean up steering hooks
            if self.steering_hooks:
                for hook, _ in self.steering_hooks:
                    hook.reset()
                    
            # Clean up steering hooks
            self._cleanup_steering()
            
            # Join all chunks and add framing tokens
            response = "".join(response_chunks)
            full_response = f"{self.config['start_think_token']}\n{self.config['prefill']}{response}"
            
            logger.debug(f"Final response length: {len(full_response)} chars, Total thoughts: {self.thought_count}")
            return full_response
        
        except Exception as e:
            # Clean up steering hooks in case of error
            self._cleanup_steering()
            logger.error(f"Error in AutoThink processing: {str(e)}")
            raise
    
    def _extract_query(self, messages: List[Dict[str, str]]) -> str:
        """
        Extract the query from messages for classification.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Extracted query string
        """
        # Get the last user message
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        
        if user_messages:
            return user_messages[-1]
        
        # Fallback to concatenated messages
        return " ".join(m["content"] for m in messages)
