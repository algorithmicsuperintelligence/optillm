"""
Confidence calculation utilities for DeepConf.

Implements various confidence metrics based on token-level probabilities:
- Token Entropy: H = -∑P(j) log P(j)
- Token Confidence: C = -(1/k) ∑log P(j) for top-k tokens
- Group Confidence: Sliding window averages
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ConfidenceCalculator:
    """
    Calculates various confidence metrics for token-level assessment.
    """
    
    def __init__(self, window_size: int = 2048, top_k: int = 5):
        """
        Initialize the confidence calculator.
        
        Args:
            window_size: Size of sliding window for group confidence
            top_k: Number of top tokens for token confidence calculation
        """
        self.window_size = window_size
        self.top_k = top_k
        self.token_confidences = []
        self.group_confidences = []
        
    def reset(self):
        """Reset internal state for new trace."""
        self.token_confidences = []
        self.group_confidences = []
    
    def calculate_token_entropy(self, logits: torch.Tensor) -> float:
        """
        Calculate token entropy: H = -∑P(j) log P(j)
        
        Args:
            logits: Raw logits tensor for current token position
            
        Returns:
            Token entropy value
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Calculate entropy: -∑P(j) log P(j)
        entropy = -(probs * log_probs).sum().item()
        
        return entropy
    
    def calculate_token_confidence(self, logits: torch.Tensor, k: Optional[int] = None) -> float:
        """
        Calculate token confidence: C = -(1/k) ∑log P(j) for top-k tokens
        
        Args:
            logits: Raw logits tensor for current token position
            k: Number of top tokens to consider (default: self.top_k)
            
        Returns:
            Token confidence value
        """
        if k is None:
            k = self.top_k
            
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get top-k log probabilities
        top_log_probs, _ = torch.topk(log_probs, k=k)
        
        # Calculate confidence: -(1/k) ∑log P(j)
        confidence = -top_log_probs.mean().item()
        
        return confidence
    
    def add_token_confidence(self, logits: torch.Tensor) -> float:
        """
        Add a new token's confidence and update group statistics.
        
        Args:
            logits: Raw logits tensor for current token position
            
        Returns:
            Token confidence value
        """
        confidence = self.calculate_token_confidence(logits)
        self.token_confidences.append(confidence)
        
        # Update group confidence if we have enough tokens
        if len(self.token_confidences) >= self.window_size:
            self._update_group_confidence()
            
        return confidence
    
    def _update_group_confidence(self):
        """Update group confidence based on current sliding window."""
        if len(self.token_confidences) < self.window_size:
            return
            
        # Calculate group confidence for current window
        start_idx = len(self.token_confidences) - self.window_size
        window_confidences = self.token_confidences[start_idx:]
        group_confidence = np.mean(window_confidences)
        
        self.group_confidences.append(group_confidence)
    
    def get_current_group_confidence(self) -> Optional[float]:
        """
        Get the most recent group confidence.
        
        Returns:
            Most recent group confidence or None if not available
        """
        if not self.group_confidences:
            return None
        return self.group_confidences[-1]
    
    def get_average_trace_confidence(self) -> float:
        """
        Calculate average confidence across all tokens in the trace.
        
        Returns:
            Average confidence value
        """
        if not self.token_confidences:
            return 0.0
        return np.mean(self.token_confidences)
    
    def get_bottom_10_percent_confidence(self) -> float:
        """
        Calculate average confidence of bottom 10% groups.
        
        Returns:
            Bottom 10% group confidence
        """
        if not self.group_confidences:
            return 0.0
            
        num_bottom = max(1, len(self.group_confidences) // 10)
        sorted_confidences = sorted(self.group_confidences)
        bottom_confidences = sorted_confidences[:num_bottom]
        
        return np.mean(bottom_confidences)
    
    def get_lowest_group_confidence(self) -> float:
        """
        Get the minimum confidence across all groups.
        
        Returns:
            Lowest group confidence
        """
        if not self.group_confidences:
            return 0.0
        return min(self.group_confidences)
    
    def get_trace_statistics(self) -> Dict[str, float]:
        """
        Get comprehensive confidence statistics for the current trace.
        
        Returns:
            Dictionary with various confidence metrics
        """
        return {
            "average_confidence": self.get_average_trace_confidence(),
            "bottom_10_percent": self.get_bottom_10_percent_confidence(),
            "lowest_group": self.get_lowest_group_confidence(),
            "current_group": self.get_current_group_confidence() or 0.0,
            "num_tokens": len(self.token_confidences),
            "num_groups": len(self.group_confidences)
        }

class ConfidenceThresholdCalibrator:
    """
    Calibrates confidence thresholds based on warmup traces.
    """
    
    def __init__(self, variant: str = "low"):
        """
        Initialize the threshold calibrator.
        
        Args:
            variant: "low" (aggressive, top 10%) or "high" (conservative, top 90%)
        """
        self.variant = variant
        self.warmup_confidences = []
    
    def add_warmup_trace(self, confidence_stats: Dict[str, float]):
        """
        Add confidence statistics from a warmup trace.
        
        Args:
            confidence_stats: Dictionary with confidence metrics
        """
        self.warmup_confidences.append(confidence_stats)
    
    def calculate_threshold(self, metric: str = "average_confidence") -> float:
        """
        Calculate the confidence threshold based on warmup traces.
        
        Args:
            metric: Which confidence metric to use for threshold calculation
            
        Returns:
            Calculated threshold value
        """
        if not self.warmup_confidences:
            logger.warning("No warmup traces available for threshold calculation")
            return 0.0
        
        confidences = [stats[metric] for stats in self.warmup_confidences]
        
        if self.variant == "low":
            # DeepConf-low: 90th percentile (keeps top 10%)
            threshold = np.percentile(confidences, 90)
        else:
            # DeepConf-high: 10th percentile (keeps top 90%) 
            threshold = np.percentile(confidences, 10)
        
        logger.info(f"Calculated {self.variant} threshold: {threshold:.4f} for metric: {metric}")
        return threshold
    
    def should_terminate_trace(self, current_confidence: float, threshold: float) -> bool:
        """
        Determine if current trace should be terminated based on confidence.
        
        Args:
            current_confidence: Current confidence value
            threshold: Threshold for termination
            
        Returns:
            True if trace should be terminated
        """
        return current_confidence < threshold