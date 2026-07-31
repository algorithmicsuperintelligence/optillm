"""
Query complexity classifier for AutoThink.

This module provides functionality to classify queries as HIGH or LOW complexity
using the adaptive-classifier model.
"""

import logging
from typing import Tuple, List
import os
import sys

logger = logging.getLogger(__name__)

class ComplexityClassifier:
    """
    Classifies queries as HIGH or LOW complexity for token budget allocation.
    Uses the adaptive-classifier model for classification.
    """
    
    def __init__(self, model_name: str = "adaptive-classifier/llm-router"):
        """
        Initialize the complexity classifier.
        
        Args:
            model_name: HuggingFace model name or path for the classifier
        """
        self.model_name = model_name
        self.classifier = None
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the classification model using adaptive-classifier library."""
        try:
            # Check if adaptive-classifier is installed
            try:
                import adaptive_classifier
            except ImportError:
                logger.info("Installing adaptive-classifier library...")
                os.system(f"{sys.executable} -m pip install adaptive-classifier")
            
            # Import the AdaptiveClassifier class
            from adaptive_classifier import AdaptiveClassifier
            
            logger.info(f"Loading complexity classifier model: {self.model_name}")
            self.classifier = AdaptiveClassifier.from_pretrained(self.model_name)
            logger.info("Classifier loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading complexity classifier: {e}")
            # Fallback to basic classification if model fails to load
            self.classifier = None
    
    def predict(self, text: str) -> List[Tuple[str, float]]:
        """
        Predict the complexity label for a given text.
        
        Args:
            text: The query text to classify
            
        Returns:
            List of (label, score) tuples sorted by confidence
        """
        if self.classifier is None:
            logger.warning("Classifier not loaded. Using fallback classification.")
            return self._fallback_classification(text)
        
        try:
            # Make prediction using the AdaptiveClassifier
            predictions = self.classifier.predict(text)
            logger.debug(f"Classifier predictions: {predictions}")
            
            # Make sure predictions are in the expected format
            if isinstance(predictions, list) and all(isinstance(p, tuple) and len(p) == 2 for p in predictions):
                # Sort by confidence (assuming higher score = higher confidence)
                predictions.sort(key=lambda x: x[1], reverse=True)
                return predictions
            else:
                logger.warning(f"Unexpected prediction format: {predictions}")
                return self._fallback_classification(text)
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return self._fallback_classification(text)
    
    def _fallback_classification(self, text: str) -> List[Tuple[str, float]]:
        """
        Simple heuristic classification when model isn't available.
        
        Args:
            text: The query text
            
        Returns:
            List of (label, score) tuples
        """
        # Count key indicators of complexity
        complexity_indicators = [
            "explain", "analyze", "compare", "evaluate", "synthesize",
            "how", "why", "complex", "detail", "thorough", "comprehensive",
            "step by step", "calculate", "prove", "justify", "multiple",
            "consequences", "implications", "differentiate", "frameworks"
        ]
        
        # Count mentions of complexity indicators
        count = sum(1 for indicator in complexity_indicators if indicator.lower() in text.lower())
        
        # Calculate complexity probability based on count and text length
        text_length_factor = min(len(text) / 100, 2.0)  # Cap at 2.0
        indicator_factor = min(count / 3, 1.5)  # Cap at 1.5
        
        # Combined factor determines HIGH vs LOW
        complexity_score = text_length_factor * indicator_factor
        
        if complexity_score > 1.0:
            return [("HIGH", 0.7), ("LOW", 0.3)]
        else:
            return [("LOW", 0.8), ("HIGH", 0.2)]
    
    def is_high_complexity(self, text: str, threshold: float = 0.5) -> bool:
        """
        Determine if a query is high complexity.
        
        Args:
            text: The query text
            threshold: Confidence threshold for HIGH classification
            
        Returns:
            Boolean indicating if the query is high complexity
        """
        predictions = self.predict(text)
        
        for label, score in predictions:
            if label == "HIGH" and score >= threshold:
                return True
        
        return False
    
    def get_complexity_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Get the complexity label and confidence score.
        
        Args:
            text: The query text
            
        Returns:
            Tuple of (complexity_label, confidence_score)
        """
        predictions = self.predict(text)
        return predictions[0]  # Return highest confidence prediction
