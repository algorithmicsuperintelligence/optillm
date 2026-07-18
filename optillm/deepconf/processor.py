"""
Main DeepConf processor implementation.

Implements the online mode algorithm with:
- Warmup phase for threshold calibration
- Early termination based on confidence
- Consensus-based stopping
- Weighted majority voting
"""

import torch
import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer, DynamicCache
from collections import Counter, defaultdict
import numpy as np

from .confidence import ConfidenceCalculator, ConfidenceThresholdCalibrator

logger = logging.getLogger(__name__)

# Default configuration based on DeepConf paper
DEFAULT_CONFIG = {
    "variant": "low",            # "low" (aggressive) or "high" (conservative)
    "warmup_samples": 16,        # Initial calibration traces
    "consensus_threshold": 0.95, # Stop when consensus reached
    "max_traces": 128,           # Maximum trace budget
    "window_size": 2048,         # Sliding window for group confidence
    "top_k": 5,                  # K for token confidence calculation
    "min_trace_length": 100,     # Minimum tokens before termination allowed
    "max_tokens_per_trace": 4096,  # Maximum tokens per individual trace
    "temperature": 0.7,          # Generation temperature
    "confidence_metric": "average_confidence",  # Metric for threshold calculation
    "include_stats": False,      # Include debugging statistics in response
}

class TraceResult:
    """Container for a single reasoning trace and its confidence statistics."""
    
    def __init__(self, tokens: List[int], text: str, confidence_stats: Dict[str, float]):
        self.tokens = tokens
        self.text = text
        self.confidence_stats = confidence_stats
        self.terminated_early = False

class DeepConfProcessor:
    """
    Main DeepConf processor implementing online mode with early termination.
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                 config: Dict[str, Any] = None):
        """
        Initialize the DeepConf processor.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Initialize components
        self.confidence_calculator = ConfidenceCalculator(
            window_size=self.config["window_size"],
            top_k=self.config["top_k"]
        )
        self.threshold_calibrator = ConfidenceThresholdCalibrator(
            variant=self.config["variant"]
        )
        
        # Track generation state
        self.warmup_traces = []
        self.online_traces = []
        self.confidence_threshold = None
        self.total_tokens_used = 0
        
        logger.info(f"DeepConf processor initialized with variant: {self.config['variant']}")
    
    def reset(self):
        """Reset processor state for new query."""
        self.warmup_traces = []
        self.online_traces = []
        self.confidence_threshold = None
        self.total_tokens_used = 0
        self.confidence_calculator.reset()
    
    def generate_single_trace(self, messages: List[Dict[str, str]], 
                            use_early_termination: bool = False) -> TraceResult:
        """
        Generate a single reasoning trace with optional early termination.
        
        Args:
            messages: Input messages
            use_early_termination: Whether to apply early termination
            
        Returns:
            TraceResult object containing trace and confidence stats
        """
        # Reset confidence calculator for new trace
        self.confidence_calculator.reset()
        
        # Tokenize input messages
        tokens = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).input_ids.to(self.model.device)
        
        # Initialize generation state
        kv_cache = DynamicCache()
        generated_tokens = []
        generated_text_parts = []
        token_count = 0
        terminated_early = False
        
        while token_count < self.config["max_tokens_per_trace"]:
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=tokens, past_key_values=kv_cache, use_cache=True)
                logits = outputs.logits[0, -1, :]  # Get logits for last token
                kv_cache = outputs.past_key_values
            
            # Calculate confidence for current token
            token_confidence = self.confidence_calculator.add_token_confidence(logits)
            
            # Check for early termination (only after minimum trace length)
            if (use_early_termination and 
                token_count >= self.config["min_trace_length"] and
                self.confidence_threshold is not None):
                
                current_group_confidence = self.confidence_calculator.get_current_group_confidence()
                if (current_group_confidence is not None and 
                    current_group_confidence < self.confidence_threshold):
                    logger.debug(f"Early termination at token {token_count}, "
                               f"confidence: {current_group_confidence:.4f} < {self.confidence_threshold:.4f}")
                    terminated_early = True
                    break
            
            # Sample next token
            probs = torch.softmax(logits / self.config["temperature"], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Add token to generation
            generated_tokens.append(next_token)
            token_text = self.tokenizer.decode([next_token])
            generated_text_parts.append(token_text)
            
            # Update tokens for next iteration
            tokens = torch.tensor([[next_token]]).to(self.model.device)
            token_count += 1
        
        # Compile results
        generated_text = "".join(generated_text_parts)
        confidence_stats = self.confidence_calculator.get_trace_statistics()
        
        trace_result = TraceResult(generated_tokens, generated_text, confidence_stats)
        trace_result.terminated_early = terminated_early
        
        self.total_tokens_used += token_count
        
        logger.debug(f"Generated trace: {token_count} tokens, "
                    f"avg confidence: {confidence_stats['average_confidence']:.4f}, "
                    f"early termination: {terminated_early}")
        
        return trace_result
    
    def run_warmup_phase(self, messages: List[Dict[str, str]]) -> None:
        """
        Run the warmup phase to generate initial traces and calibrate threshold.
        
        Args:
            messages: Input messages
        """
        logger.info(f"Starting warmup phase with {self.config['warmup_samples']} traces")
        
        for i in range(self.config['warmup_samples']):
            trace = self.generate_single_trace(messages, use_early_termination=False)
            self.warmup_traces.append(trace)
            self.threshold_calibrator.add_warmup_trace(trace.confidence_stats)
            
            logger.debug(f"Warmup trace {i+1}/{self.config['warmup_samples']} completed")
        
        # Calculate confidence threshold
        self.confidence_threshold = self.threshold_calibrator.calculate_threshold(
            metric=self.config["confidence_metric"]
        )
        
        logger.info(f"Warmup phase completed. Threshold: {self.confidence_threshold:.4f}")
    
    def check_consensus(self, traces: List[TraceResult]) -> Tuple[bool, str, float]:
        """
        Check if consensus has been reached among traces.
        
        Args:
            traces: List of trace results
            
        Returns:
            Tuple of (has_consensus, consensus_answer, consensus_ratio)
        """
        if not traces:
            return False, "", 0.0
        
        # Extract answers from traces (simplified - in practice might need more sophisticated extraction)
        answers = []
        for trace in traces:
            # Simple heuristic: take last sentence or last 50 characters as the "answer"
            answer = trace.text.strip().split('.')[-1].strip()
            if not answer:
                answer = trace.text.strip()[-50:].strip()
            answers.append(answer)
        
        # Count answer frequencies
        answer_counts = Counter(answers)
        most_common_answer, most_common_count = answer_counts.most_common(1)[0]
        
        consensus_ratio = most_common_count / len(answers)
        has_consensus = consensus_ratio >= self.config["consensus_threshold"]
        
        logger.debug(f"Consensus check: {consensus_ratio:.3f} "
                    f"({'✓' if has_consensus else '✗'} >= {self.config['consensus_threshold']})")
        
        return has_consensus, most_common_answer, consensus_ratio
    
    def weighted_majority_vote(self, traces: List[TraceResult]) -> Tuple[str, Dict[str, float]]:
        """
        Perform weighted majority voting based on trace confidences.
        
        Args:
            traces: List of trace results
            
        Returns:
            Tuple of (best_answer, voting_stats)
        """
        if not traces:
            return "", {}
        
        # Group traces by answer and calculate weighted scores
        answer_groups = defaultdict(list)
        for trace in traces:
            # Extract answer (same heuristic as consensus check)
            answer = trace.text.strip().split('.')[-1].strip()
            if not answer:
                answer = trace.text.strip()[-50:].strip()
            answer_groups[answer].append(trace)
        
        # Calculate weighted scores for each answer
        answer_scores = {}
        for answer, group_traces in answer_groups.items():
            # Weight by average confidence
            confidences = [trace.confidence_stats['average_confidence'] for trace in group_traces]
            weighted_score = sum(confidences) / len(confidences)  # Average confidence
            count_weight = len(group_traces) / len(traces)  # Frequency weight
            
            # Combine confidence and frequency
            final_score = weighted_score * 0.7 + count_weight * 0.3
            answer_scores[answer] = final_score
        
        # Select best answer
        best_answer = max(answer_scores.keys(), key=lambda x: answer_scores[x])
        
        voting_stats = {
            "num_unique_answers": len(answer_groups),
            "best_score": answer_scores[best_answer],
            "answer_distribution": {ans: len(traces) for ans, traces in answer_groups.items()}
        }
        
        logger.info(f"Weighted voting completed. Best answer score: {answer_scores[best_answer]:.4f}")
        
        return best_answer, voting_stats
    
    def process_online(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Main online processing with warmup and early termination.
        
        Args:
            messages: Input messages
            
        Returns:
            Tuple of (final_answer, processing_stats)
        """
        self.reset()
        
        logger.info("Starting DeepConf online processing")
        
        # Phase 1: Warmup
        self.run_warmup_phase(messages)
        
        # Phase 2: Online generation with early termination
        logger.info("Starting online generation phase")
        
        all_traces = self.warmup_traces[:]  # Include warmup traces
        
        for trace_num in range(self.config["max_traces"] - self.config["warmup_samples"]):
            # Generate trace with early termination
            trace = self.generate_single_trace(messages, use_early_termination=True)
            all_traces.append(trace)
            self.online_traces.append(trace)
            
            # Check consensus
            has_consensus, consensus_answer, consensus_ratio = self.check_consensus(all_traces)
            
            logger.debug(f"Online trace {trace_num + 1} completed. "
                        f"Total traces: {len(all_traces)}, Consensus: {consensus_ratio:.3f}")
            
            if has_consensus:
                logger.info(f"Consensus reached after {len(all_traces)} traces "
                           f"(ratio: {consensus_ratio:.3f})")
                break
        
        # Phase 3: Final answer selection
        final_answer, voting_stats = self.weighted_majority_vote(all_traces)
        
        # Compile processing statistics
        processing_stats = {
            "total_traces": len(all_traces),
            "warmup_traces": len(self.warmup_traces),
            "online_traces": len(self.online_traces),
            "early_terminations": sum(1 for trace in all_traces if trace.terminated_early),
            "total_tokens_used": self.total_tokens_used,
            "confidence_threshold": self.confidence_threshold,
            "variant": self.config["variant"],
            **voting_stats
        }
        
        logger.info(f"DeepConf processing completed. "
                   f"Traces: {processing_stats['total_traces']}, "
                   f"Tokens: {processing_stats['total_tokens_used']}, "
                   f"Early terminations: {processing_stats['early_terminations']}")
        
        return final_answer, processing_stats