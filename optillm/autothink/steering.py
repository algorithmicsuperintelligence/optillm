"""
Steering vector manager for AutoThink.

This module provides functionality to load and apply steering vectors
from Hugging Face datasets during inference.
"""

import torch
import logging
import random
import json
import datasets
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class SteeringVectorManager:
    """
    Manager for loading and applying steering vectors from a dataset.
    """
    
    def __init__(
        self, 
        dataset_name: str,
        target_layer: int = 19,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the steering vector manager.
        
        Args:
            dataset_name: Name of the HuggingFace dataset containing steering vectors
            target_layer: Target layer for applying steering vectors
            cache_dir: Directory for caching the dataset
            device: Device to use for tensors
        """
        self.dataset_name = dataset_name
        self.target_layer = target_layer
        self.cache_dir = cache_dir
        self.device = device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        
        # Storage for steering vectors
        self.steering_vectors = []
        self.pattern_to_vectors = {}
        self.tokenized_contexts = {}
        
        # Default steering strengths
        self.default_strength = 2.0
        self.pattern_strengths = {
            "depth_and_thoroughness": 2.5,
            "numerical_accuracy": 2.0, 
            "self_correction": 3.0,
            "exploration": 2.0,
            "organization": 1.5,
            "unknown": 1.0
        }
        
        # If dataset is provided, load it
        if dataset_name:
            self.load_dataset()
        
    def load_dataset(self):
        """Load steering vectors from the HuggingFace dataset."""
        try:
            logger.info(f"Loading steering vectors from dataset: {self.dataset_name}")
            
            # Load the dataset
            dataset = datasets.load_dataset(self.dataset_name, cache_dir=self.cache_dir)
            
            # Get the main split (usually 'train')
            main_split = list(dataset.keys())[0]
            vector_data = dataset[main_split]
            
            # Load each item as a steering vector
            for item in vector_data:
                # Convert dataset item to proper format
                vector = self._process_dataset_item(item)
                if vector:
                    self.steering_vectors.append(vector)
                    
                    # Group by reasoning pattern
                    pattern = vector.get("reasoning_pattern", "unknown")
                    if pattern not in self.pattern_to_vectors:
                        self.pattern_to_vectors[pattern] = []
                    self.pattern_to_vectors[pattern].append(vector)
            
            logger.info(f"Loaded {len(self.steering_vectors)} steering vectors")
            logger.info(f"Found {len(self.pattern_to_vectors)} reasoning patterns: {list(self.pattern_to_vectors.keys())}")
            
            # Log the first vector for debugging
            if self.steering_vectors:
                first_vector = self.steering_vectors[0]
                logger.info(f"First vector sample - pattern: {first_vector.get('reasoning_pattern', 'missing')}")
                if 'pivot_context' in first_vector:
                    context_len = len(first_vector['pivot_context'])
                    logger.info(f"First vector pivot_context length: {context_len}")
        
        except Exception as e:
            logger.error(f"Error loading steering vectors: {e}")
            self.steering_vectors = []
            self.pattern_to_vectors = {}
    
    def _process_dataset_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a dataset item into a steering vector.
        
        Args:
            item: Dataset item
            
        Returns:
            Processed steering vector or None if invalid
        """
        try:
            # Check if item has the required fields
            required_fields = ["pivot_context", "steering_vector", "reasoning_pattern"]
            if not all(field in item for field in required_fields):
                return None
            
            # Convert steering_vector to a proper format if it's a string or list
            steering_vector = item["steering_vector"]
            if isinstance(steering_vector, str):
                # Try to parse JSON string
                try:
                    steering_vector = json.loads(steering_vector)
                except json.JSONDecodeError:
                    # Try comma-separated format
                    steering_vector = [float(x) for x in steering_vector.strip("[]").split(",")]
            
            # Ensure we have a proper list
            if not isinstance(steering_vector, list):
                logger.warning(f"Invalid steering vector format: {type(steering_vector)}")
                return None
            
            # Create the steering vector dictionary
            vector = {
                "pivot_context": item["pivot_context"],
                "pivot_token": item.get("pivot_token", ""),
                "pivot_token_id": item.get("pivot_token_id", -1),
                "prob_before": item.get("prob_before", 0.0),
                "prob_after": item.get("prob_after", 0.0),
                "prob_delta": item.get("prob_delta", 0.0),
                "model_id": item.get("model_id", ""),
                "task_type": item.get("task_type", "unknown"),
                "steering_vector": steering_vector,
                "cluster_id": item.get("cluster_id", -1),
                "reasoning_pattern": item.get("reasoning_pattern", "unknown"),
                "cluster_vector": item.get("cluster_vector", steering_vector),
                "steering_layer": item.get("steering_layer", self.target_layer),
            }
            
            return vector
        
        except Exception as e:
            logger.error(f"Error processing dataset item: {e}")
            return None
    
    def create_tokenized_contexts(self, tokenizer):
        """
        Pre-tokenize context patterns for efficient matching during generation.
        Similar to how guided mode does token-based matching.
        
        Args:
            tokenizer: Tokenizer for encoding contexts
        """
        # Get configurations - use similar defaults as ThinkDeeperProcessor
        max_pts_tokens = 256  # Maximum tokens to store for matching
        
        count = 0
        for vector in self.steering_vectors:
            # Get the context
            context = vector.get("pivot_context", "")
            if not context:
                continue
                
            # Pre-tokenize the context for faster matching during generation
            tokenized_context = tokenizer.encode(context, add_special_tokens=False)
            
            # Keep only up to max_pts_tokens - no point storing more than our history capacity
            if len(tokenized_context) > max_pts_tokens:
                # Get only the last max_pts_tokens (most important for matching)
                tokenized_context = tokenized_context[-max_pts_tokens:]
            
            # Store the tokenized context with its corresponding vector
            tuple_key = tuple(tokenized_context)
            self.tokenized_contexts[tuple_key] = vector
            
            # Store additional shorter versions for partial matching during early generation
            # Create shorter suffixes for early matching when context is still building up
            for suffix_len in [4, 8, 12]:
                if len(tokenized_context) > suffix_len:
                    suffix = tokenized_context[-suffix_len:]
                    suffix_tuple = tuple(suffix)
                    # Only store if not already present (avoid overwriting longer matches)
                    if suffix_tuple not in self.tokenized_contexts:
                        self.tokenized_contexts[suffix_tuple] = vector
            
            count += 1
            
        # Log statistics about the tokenized contexts
        logger.info(f"STEERING: Pre-tokenized {count} contexts into {len(self.tokenized_contexts)} token patterns")
        
        # Count patterns by length for debugging
        length_counts = {}
        for key in self.tokenized_contexts.keys():
            length = len(key)
            if length not in length_counts:
                length_counts[length] = 0
            length_counts[length] += 1
        
        logger.info(f"STEERING: Token pattern length distribution: {sorted(length_counts.items())}")
    
    def get_steering_strength(self, pattern: str) -> float:
        """
        Get the steering strength for a specific pattern.
        
        Args:
            pattern: The reasoning pattern
            
        Returns:
            The steering strength
        """
        return self.pattern_strengths.get(pattern, self.default_strength)
    
    def set_steering_strength(self, pattern: str, strength: float):
        """
        Set the steering strength for a specific pattern.
        
        Args:
            pattern: The reasoning pattern
            strength: The steering strength
        """
        self.pattern_strengths[pattern] = strength
        logger.info(f"STEERING: Set strength for {pattern} to {strength}")
    
    def get_pattern_vectors(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Get all steering vectors for a specific reasoning pattern.
        
        Args:
            pattern: The reasoning pattern
            
        Returns:
            List of steering vectors
        """
        return self.pattern_to_vectors.get(pattern, [])
        
    def get_steering_vector(self, context: str, match_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the most appropriate steering vector for a context.
        
        Args:
            context: The current generation context.
            match_key: Optional key for matching.
            
        Returns:
            Dictionary with steering data or None if no match.
        """
        if match_key is not None:
            # Try exact matching by key
            for vector in self.steering_vectors:
                # Get the last 100 chars of the pivot_context for comparison
                vector_context = vector.get("pivot_context", "")
                vector_key = vector_context[-100:] if len(vector_context) >= 100 else vector_context
                
                # Perform exact match comparison and log for debugging
                if vector_key == match_key:
                    logger.debug(f"STEERING: Context match found for '{vector.get('pivot_token', '')}' with pattern {vector.get('reasoning_pattern', 'unknown')}")
                    return vector
                
                # For first 5 attempts, log debugging info when match fails
                if random.random() < 0.001:  # Log a small random sample for debugging
                    logger.debug(f"STEERING: Match failed - key length: {len(match_key)}, vector key length: {len(vector_key)}")
                    logger.debug(f"STEERING: Match key sample: '{match_key[:20]}...'")
                    logger.debug(f"STEERING: Vector key sample: '{vector_key[:20]}...'")
        
        # If no match found, return None
        return None

class SteeringHook:
    """Hook for applying steering vectors during generation."""
    
    def __init__(self, manager: SteeringVectorManager, layer_num: int, tokenizer=None):
        """
        Initialize the steering hook.
        
        Args:
            manager: The steering vector manager
            layer_num: The layer number to apply steering to
            tokenizer: Tokenizer for token-based matching
        """
        self.manager = manager
        self.layer_num = layer_num
        self.tokenizer = tokenizer
        
        # For text-based matching (original approach)
        self.context_buffer = ""
        
        # For token-based matching (guided-style approach)
        self.token_history = []  # Store token IDs for matching
        self.max_history = 256   # Maximum tokens to keep in history
        
        # State tracking
        self.match_found = False
        self.current_vector = None
        self.last_pattern = None
        
        # Single pattern for entire request
        self.active_pattern = None  # Currently active pattern
        self.generation_started = False
        
        logger.info(f"STEERING: Initialized hook for layer {layer_num}")
    
    def __call__(self, module, input_tensors, output):
        """
        Apply steering to the output of a layer.
        
        Args:
            module: The module being hooked
            input_tensors: The input tensors
            output: The output tensor
            
        Returns:
            Modified output tensor
        """
        # Use a try-except block around the entire function to prevent crashing
        try:
            # Skip if no active pattern is set
            if not self.active_pattern:
                return output
            
            # Apply steering vector (only if we have an active pattern)
            if self.current_vector is not None:
                # Get the appropriate steering strength
                pattern = self.current_vector.get("reasoning_pattern", "unknown")
                strength = self.manager.get_steering_strength(pattern)
                
                # Keep strength within safe bounds - use lower values for better stability
                safe_strength = min(max(strength, 0.1), 2.0)  # Limit between 0.1 and 2.0
                
                # Log when pattern changes
                if pattern != self.last_pattern:
                    logger.info(f"STEERING: Switching to {pattern} reasoning pattern with strength {safe_strength}")
                    self.last_pattern = pattern
                else:
                    # Log periodically that steering is still active
                    if random.random() < 0.05:
                        logger.info(f"STEERING: Still applying {pattern} pattern with strength {safe_strength}")
                
                # Apply the steering vector using our safer function
                try:
                    if isinstance(output, tuple):
                        # Some models return a tuple where the first element is the hidden states
                        hidden_states = output[0]
                        
                        # Apply steering - if it fails, return original
                        try:
                            # Create a new reference for the modified hidden states
                            modified_hidden_states = self._apply_steering_vector(hidden_states, self.current_vector, safe_strength)
                            # Validate the result has the right shape
                            if modified_hidden_states.shape == hidden_states.shape:
                                # Create a new tuple with the modified hidden states
                                return (modified_hidden_states,) + output[1:]
                            else:
                                logger.error(f"STEERING: Modified hidden states have wrong shape. Expected {hidden_states.shape}, got {modified_hidden_states.shape}")
                                return output
                        except Exception as e:
                            logger.error(f"STEERING: Error applying steering to tuple output: {e}")
                            return output
                    else:
                        # Direct tensor output
                        try:
                            # Apply steering directly
                            return self._apply_steering_vector(output, self.current_vector, safe_strength)
                        except Exception as e:
                            logger.error(f"STEERING: Error applying steering to direct output: {e}")
                            return output
                        
                except Exception as e:
                    logger.error(f"STEERING: Unexpected error in steering application: {e}")
                    return output
            
            return output
        except Exception as e:
            logger.error(f"STEERING: Critical error in hook: {e}")
            return output
    
    def _apply_steering_vector(self, hidden_states: torch.Tensor, 
                              steering_vector: Dict[str, Any],
                              scaling_factor: float = 2.0) -> torch.Tensor:
        """
        Apply a steering vector to hidden states.
        
        Args:
            hidden_states: The hidden states tensor
            steering_vector: Dictionary with steering vector data
            scaling_factor: Factor to scale the steering vector by
            
        Returns:
            Modified hidden states tensor
        """
        try:
            # Make a DEEP clone to avoid in-place modification issues
            hidden_states_clone = hidden_states.clone().detach()
            
            # Check what kind of vector we're using
            vector_type = None
            if "steering_vector" in steering_vector:
                vector_data = steering_vector["steering_vector"]
                vector_type = "steering_vector"
            elif "cluster_vector" in steering_vector:
                vector_data = steering_vector["cluster_vector"]
                vector_type = "cluster_vector"
            else:
                logger.warning("STEERING: No valid vector found in steering data")
                return hidden_states  # No steering vector found
            
            # Safely convert vector to tensor
            try:
                vector = torch.tensor(vector_data, 
                                   dtype=hidden_states.dtype, 
                                   device=hidden_states.device)
            except Exception as e:
                logger.error(f"STEERING: Error converting vector to tensor: {e}")
                return hidden_states
            
            # Log vector info
            pattern = steering_vector.get("reasoning_pattern", "unknown")
            logger.debug(f"STEERING: Applying {vector_type} for pattern '{pattern}' with base scaling {scaling_factor}")
            
            # Apply scaling based on prob_delta if available
            if "prob_delta" in steering_vector:
                prob_delta = abs(steering_vector["prob_delta"])
                # Limit the impact of prob_delta to prevent extreme scaling
                prob_delta_capped = min(max(prob_delta, 0.1), 2.0)
                scaling_factor *= prob_delta_capped
                logger.debug(f"STEERING: Adjusted scaling by prob_delta {prob_delta_capped} to {scaling_factor}")
            
            # Check if the token is positive or negative
            is_positive = steering_vector.get("is_positive", True)
            
            # Log tensor shapes and verify compatibility
            hs_shape = hidden_states.shape
            vector_shape = vector.shape
            logger.debug(f"STEERING: hidden_states shape: {hs_shape}, vector shape: {vector_shape}")
            
            # Verify shapes are compatible
            if len(vector_shape) != 1 or vector_shape[0] != hs_shape[-1]:
                logger.error(f"STEERING: Shape mismatch - hidden_states: {hs_shape}, vector: {vector_shape}")
                return hidden_states
            
            # Bound scaling factor for safety - using a tighter range to prevent instability
            safe_scaling = min(max(scaling_factor, 0.0), 3.0)  # Limit between 0 and 3
            
            # Apply steering with safe indexing - with additional safeguards
            try:
                if len(hs_shape) >= 3 and hs_shape[0] > 0 and hs_shape[1] > 0:
                    # Apply to the last token's representation (safe indexing)
                    if is_positive:
                        # For positive tokens, add the vector
                        # Normalize vector first to prevent numerical instability
                        vector_norm = torch.nn.functional.normalize(vector, dim=0)
                        hidden_states_clone[-1, -1, :] = hidden_states_clone[-1, -1, :] + safe_scaling * vector_norm
                    else:
                        # For negative tokens, subtract the vector
                        vector_norm = torch.nn.functional.normalize(vector, dim=0)
                        hidden_states_clone[-1, -1, :] = hidden_states_clone[-1, -1, :] - safe_scaling * vector_norm
                    
                    # Check for NaN or inf values after modification
                    if torch.isnan(hidden_states_clone).any() or torch.isinf(hidden_states_clone).any():
                        logger.error("STEERING: NaN or inf values detected after applying vector, reverting to original")
                        return hidden_states
                else:
                    logger.error(f"STEERING: Hidden states shape not suitable for steering: {hs_shape}")
                    return hidden_states
            except IndexError as e:
                logger.error(f"STEERING: IndexError when applying vector: {e}")
                logger.error(f"STEERING: Indices: [-1, -1, :], tensor shape: {hidden_states.shape}")
                return hidden_states
            
            return hidden_states_clone
        except Exception as e:
            logger.error(f"STEERING: Unexpected error applying steering vector: {e}")
            return hidden_states
    
    def update_context(self, new_tokens: str):
        """
        Update the context buffer with new tokens.
        
        Args:
            new_tokens: New tokens to add to the context.
        """
        # Both methods - text-based and token-based
        if self.tokenizer is not None:
            # Token-based approach (similar to guided mode)
            # Tokenize the new text
            token_ids = self.tokenizer.encode(new_tokens, add_special_tokens=False)
            
            if token_ids:  # Only proceed if we got tokens
                # Add to token history
                self.token_history.extend(token_ids)
                
                # Trim history if needed
                if len(self.token_history) > self.max_history:
                    self.token_history = self.token_history[-self.max_history:]
                
                # Log token updates periodically
                if random.random() < 0.01:
                    logger.debug(f"STEERING: Token history updated, now has {len(self.token_history)} tokens")
        else:
            # Original text-based approach as fallback
            # Update context buffer
            self.context_buffer += new_tokens
            
            # Keep only the last 500 characters
            if len(self.context_buffer) > 500:
                self.context_buffer = self.context_buffer[-500:]
                logger.debug(f"STEERING: Context buffer trimmed to {len(self.context_buffer)} chars")
    
    def update_token_history(self, new_tokens: List[int]):
        """
        Update the token history with new tokens.
        
        Args:
            new_tokens: New token IDs to add
        """
        # Add to token history
        self.token_history.extend(new_tokens)
        
        # Trim history if needed
        if len(self.token_history) > self.max_history:
            self.token_history = self.token_history[-self.max_history:]
        
        # Log token updates periodically
        if random.random() < 0.01:
            logger.debug(f"STEERING: Token history updated, now has {len(self.token_history)} tokens")
            
        """
        Update the context buffer with new tokens.
        
        Args:
            new_tokens: New tokens to add to the context.
        """
        # Both methods - text-based and token-based
        if self.tokenizer is not None:
            # Token-based approach (similar to guided mode)
            # Tokenize the new text
            token_ids = self.tokenizer.encode(new_tokens, add_special_tokens=False)
            
            if token_ids:  # Only proceed if we got tokens
                # Add to token history
                self.token_history.extend(token_ids)
                
                # Trim history if needed
                if len(self.token_history) > self.max_history:
                    self.token_history = self.token_history[-self.max_history:]
                
                # Log token updates periodically
                if random.random() < 0.01:
                    logger.debug(f"STEERING: Token history updated, now has {len(self.token_history)} tokens")
        
        # Text-based approach (always update)
        # Update context buffer
        self.context_buffer += new_tokens
        
        # Keep only the last 500 characters
        if len(self.context_buffer) > 500:
            self.context_buffer = self.context_buffer[-500:]
            logger.debug(f"STEERING: Context buffer trimmed to {len(self.context_buffer)} chars")
    
    def try_match(self):
        """
        Try to match the current context with a steering vector.
        Only allows one pattern to be selected for the entire generation.
        Tries both token-based and text-based matching approaches.
        """
        # If we already have an active pattern, don't try to match again
        if self.active_pattern:
            return False
        
        # Try both token-based and text-based matching
        match_result = False
        
        # First try token-based matching if available
        if self.tokenizer is not None and hasattr(self.manager, 'tokenized_contexts') and self.manager.tokenized_contexts:
            match_result = self._try_token_match()
        
        # If token matching fails, try text-based matching
        if not match_result:
            match_result = self._try_text_match()
            
        # Set generation started flag AFTER trying to match
        self.generation_started = True
            
        # If a match is found, set this as the permanent pattern for this generation
        if match_result and self.current_vector:
            new_pattern = self.current_vector.get("reasoning_pattern", "unknown")
            self.active_pattern = new_pattern
            logger.info(f"STEERING: Selected '{new_pattern}' pattern for this request")
            
        return match_result
    
    def _try_token_match(self):
        """
        Try to match using token-based context (similar to guided mode).
        """
        # Ensure we have enough tokens
        if len(self.token_history) < 4:
            logger.debug(f"STEERING: Not enough tokens to match ({len(self.token_history)})")
            return False
            
        # Track best match
        best_match = {
            'length': 0, 
            'vector': None,
            'is_partial': True
        }
        
        # Log token history periodically
        if random.random() < 0.01:
            history_sample = self.token_history[-5:] if len(self.token_history) >= 5 else self.token_history
            logger.debug(f"STEERING: Token matching with history (last {len(history_sample)} of {len(self.token_history)} tokens): {history_sample}")
        
        # Check for matches in tokenized contexts
        for tokenized_context, vector in self.manager.tokenized_contexts.items():
            token_list = list(tokenized_context)
            token_len = len(token_list)
            
            # Try partial matching for shorter contexts
            if len(self.token_history) < token_len:
                # Only try partial matching if we have enough context tokens (at least 4)
                if len(self.token_history) >= 4:
                    # Calculate how many tokens to match - minimum of context length or 1/2 of token sequence
                    match_len = min(len(self.token_history), max(4, token_len // 2))
                    # Try to match the end of the token sequence with the context tokens
                    if self.token_history[-match_len:] == token_list[-match_len:]:
                        # Track this match - prefer longer matches
                        if match_len > best_match['length']:
                            best_match = {
                                'length': match_len,
                                'vector': vector,
                                'is_partial': True,
                                'match_len': match_len,
                                'token_len': token_len
                            }
            else:
                # Full matching when we have enough tokens
                if self.token_history[-token_len:] == token_list:
                    # Track this match - full matches are preferred
                    if token_len >= best_match['length']:
                        best_match = {
                            'length': token_len,
                            'vector': vector,
                            'is_partial': False,
                            'match_len': token_len,
                            'token_len': token_len
                        }
        
        # Apply best match if found
        if best_match['vector'] is not None:
            match_type = "PARTIAL" if best_match['is_partial'] else "FULL"
            self.match_found = True
            self.current_vector = best_match['vector']
            pattern = best_match['vector'].get("reasoning_pattern", "unknown")
            pivot_token = best_match['vector'].get("pivot_token", "")
            
            logger.info(f"STEERING: Found {match_type} token match ({best_match['match_len']}/{best_match['token_len']} tokens) for {pattern} pattern")
            logger.info(f"STEERING: Pivot token: '{pivot_token}'")
            
            return True
            
        # If no match, try fuzzy matching with 70% similarity threshold
        if len(self.token_history) >= 8 and not self.match_found:
            logger.debug("STEERING: No exact match found, trying fuzzy matching")
            for tokenized_context, vector in self.manager.tokenized_contexts.items():
                token_list = list(tokenized_context)
                token_len = len(token_list)
                
                if token_len >= 8:  # Only try fuzzy matching for contexts with enough tokens
                    match_len = min(len(self.token_history), token_len)
                    last_tokens = self.token_history[-match_len:]
                    context_tokens = token_list[-match_len:]
                    
                    # Count matching tokens
                    matches = sum(1 for a, b in zip(last_tokens, context_tokens) if a == b)
                    similarity = matches / match_len
                    
                    if similarity >= 0.7:  # 70% similarity threshold
                        if match_len > best_match['length']:
                            best_match = {
                                'length': match_len,
                                'vector': vector,
                                'is_partial': True,
                                'match_len': match_len,
                                'token_len': token_len,
                                'similarity': similarity
                            }
                            
            # Apply fuzzy match if found
            if best_match['vector'] is not None:
                self.match_found = True
                self.current_vector = best_match['vector']
                pattern = best_match['vector'].get("reasoning_pattern", "unknown")
                pivot_token = best_match['vector'].get("pivot_token", "")
                similarity = best_match.get('similarity', 0.0)
                
                logger.info(f"STEERING: Found fuzzy match ({similarity:.2f} similarity) for {pattern} pattern")
                logger.info(f"STEERING: Pivot token: '{pivot_token}'")
                
                return True
            
        # If no match, try fuzzy matching with 70% similarity threshold
        if len(self.token_history) >= 8 and not self.match_found:
            logger.debug("STEERING: No exact match found, trying fuzzy matching")
            for tokenized_context, vector in self.manager.tokenized_contexts.items():
                token_list = list(tokenized_context)
                token_len = len(token_list)
                
                if token_len >= 8:  # Only try fuzzy matching for contexts with enough tokens
                    match_len = min(len(self.token_history), token_len)
                    last_tokens = self.token_history[-match_len:]
                    context_tokens = token_list[-match_len:]
                    
                    # Count matching tokens
                    matches = sum(1 for a, b in zip(last_tokens, context_tokens) if a == b)
                    similarity = matches / match_len
                    
                    if similarity >= 0.7:  # 70% similarity threshold
                        if match_len > best_match['length']:
                            best_match = {
                                'length': match_len,
                                'vector': vector,
                                'is_partial': True,
                                'match_len': match_len,
                                'token_len': token_len,
                                'similarity': similarity
                            }
                            
            # Apply fuzzy match if found
            if best_match['vector'] is not None:
                self.match_found = True
                self.current_vector = best_match['vector']
                pattern = best_match['vector'].get("reasoning_pattern", "unknown")
                pivot_token = best_match['vector'].get("pivot_token", "")
                similarity = best_match.get('similarity', 0.0)
                
                logger.info(f"STEERING: Found fuzzy match ({similarity:.2f} similarity) for {pattern} pattern")
                logger.info(f"STEERING: Pivot token: '{pivot_token}'")
                
                return True
        
        return False
    
    def _try_text_match(self):
        """Try to match using text-based context (original approach)."""
        # Skip if context buffer is too short
        if len(self.context_buffer) < 10:  # Require at least 10 chars for matching
            return False
            
        # Get the last 100 characters as the match key
        match_key = self.context_buffer[-100:] if len(self.context_buffer) >= 100 else self.context_buffer
        
        # Log context buffer periodically to debug
        if random.random() < 0.01:  # Log occasionally to avoid spam
            logger.debug(f"STEERING: Current context buffer (last 50 chars): '{self.context_buffer[-50:]}'") 
            logger.debug(f"STEERING: Matching with key (length {len(match_key)}): '{match_key[:20]}...'")
        
        # Try to find a matching steering vector using original matching
        vector = self.manager.get_steering_vector(self.context_buffer, match_key)
        
        if vector is not None:
            self.match_found = True
            self.current_vector = vector
            pattern = vector.get("reasoning_pattern", "unknown")
            pivot_token = vector.get("pivot_token", "")
            logger.info(f"STEERING: Found text match for {pattern} reasoning pattern")
            logger.info(f"STEERING: Pivot token: '{pivot_token}'")
            return True
            
        # Attempt fuzzy text matching as a fallback
        if len(match_key) >= 20:  # Only try for reasonably sized contexts
            # Try each steering vector for approximate match
            best_match = None
            best_similarity = 0.0
            
            for vector in self.manager.steering_vectors:
                vector_context = vector.get("pivot_context", "")
                if not vector_context or len(vector_context) < 20:
                    continue
                    
                # Get the end of the vector context (last 100 chars)
                vector_key = vector_context[-100:] if len(vector_context) >= 100 else vector_context
                
                # Calculate simple character-level similarity
                min_length = min(len(match_key), len(vector_key))
                matching_chars = sum(1 for a, b in zip(match_key, vector_key) if a == b)
                similarity = matching_chars / min_length if min_length > 0 else 0
                
                # Keep track of best match above threshold
                if similarity >= 0.7 and similarity > best_similarity:  # 70% similarity threshold
                    best_similarity = similarity
                    best_match = vector
            
            # Use the best match if found
            if best_match is not None:
                self.match_found = True
                self.current_vector = best_match
                pattern = best_match.get("reasoning_pattern", "unknown")
                pivot_token = best_match.get("pivot_token", "")
                logger.info(f"STEERING: Found fuzzy text match ({best_similarity:.2f} similarity) for {pattern} pattern")
                logger.info(f"STEERING: Pivot token: '{pivot_token}'")
                return True
        
        return False
    
    def reset(self):
        """Reset the hook state for a new generation."""
        self.match_found = False
        self.current_vector = None
        
        # Clear both text and token histories
        self.context_buffer = ""
        self.token_history = []
        
        # Reset pattern and state tracking
        self.last_pattern = None
        self.active_pattern = None
        self.generation_started = False
        
        logger.info("STEERING: Hook state reset for new generation")

def install_steering_hooks(model, manager: SteeringVectorManager, tokenizer=None) -> List[Tuple]:
    """
    Install steering hooks on a model.
    
    Args:
        model: The model to install hooks on
        manager: The steering vector manager
        tokenizer: Tokenizer for token-based matching
        
    Returns:
        List of installed hooks
    """
    hooks = []
    
    # Target layer is specified in the manager
    layer_num = manager.target_layer
    logger.info(f"STEERING: Attempting to install hook on layer {layer_num}")
    
    # First, log model structure to help with debugging
    model_type = type(model).__name__
    logger.info(f"STEERING: Model type is {model_type}")
    if hasattr(model, 'config'):
        logger.info(f"STEERING: Model architecture is {model.config.architectures[0] if hasattr(model.config, 'architectures') else 'unknown'}")
    
    # Find the appropriate module - depends on model architecture
    module = None
    if hasattr(model, 'transformer'):
        logger.info("STEERING: Model has 'transformer' attribute")
        if hasattr(model.transformer, 'h') and layer_num < len(model.transformer.h):
            module = model.transformer.h[layer_num]
            logger.info(f"STEERING: Using transformer.h[{layer_num}]")
    elif hasattr(model, 'model'):
        logger.info("STEERING: Model has 'model' attribute")
        if hasattr(model.model, 'layers') and layer_num < len(model.model.layers):
            module = model.model.layers[layer_num]
            logger.info(f"STEERING: Using model.layers[{layer_num}]")
        elif hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers') and layer_num < len(model.model.decoder.layers):
            module = model.model.decoder.layers[layer_num]
            logger.info(f"STEERING: Using model.decoder.layers[{layer_num}]")
    elif hasattr(model, 'layers') and layer_num < len(model.layers):
        module = model.layers[layer_num]
        logger.info(f"STEERING: Using layers[{layer_num}]")
    
    if module is None:
        logger.error(f"STEERING: Could not find appropriate module for layer {layer_num}")
        logger.error("STEERING: Model structure not compatible with current hook installation logic")
        return []
    
    # Create and register hook
    hook = SteeringHook(manager, layer_num, tokenizer)
    handle = module.register_forward_hook(hook)
    
    # Return both hook object and handle for later removal
    hooks.append((hook, handle))
    
    logger.info(f"STEERING: Installed hook on layer {layer_num} successfully")
    
    return hooks

def remove_steering_hooks(hooks):
    """
    Remove steering hooks from a model.
    
    Args:
        hooks: List of (hook, handle) tuples
    """
    for _, handle in hooks:
        handle.remove()
    
    logger.info(f"STEERING: Removed {len(hooks)} hooks")
