"""
Strategy and StrategyDatabase classes for the System Prompt Learning (SPL) plugin.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from optillm.plugins.spl.config import (
    STRATEGY_DB_PATH,
    STRATEGY_METRICS_PATH,
    VALID_PROBLEM_TYPES,
    MAX_STRATEGIES_PER_TYPE,
    STRATEGY_CREATION_THRESHOLD,
    STRATEGY_MERGING_THRESHOLD,
)

# Setup logging
logger = logging.getLogger(__name__)

class Strategy:
    """Represents a problem-solving strategy learned by the system."""
    
    def __init__(
        self, 
        strategy_id: str, 
        problem_type: str, 
        strategy_text: str, 
        examples: List[str] = None, 
        success_count: int = 0, 
        total_attempts: int = 0,
        created_at: str = None,
        last_used: str = None,
        last_updated: str = None,
        confidence: float = 0.5,
        tags: List[str] = None,
        reasoning_examples: List[str] = None  # Store reasoning examples
    ):
        self.strategy_id = strategy_id
        # Ensure problem_type is always a valid type
        self.problem_type = problem_type if problem_type in VALID_PROBLEM_TYPES else "general_problem"
        self.strategy_text = strategy_text
        self.examples = examples or []
        self.success_count = success_count
        self.total_attempts = total_attempts
        self.created_at = created_at or datetime.now().isoformat()
        self.last_used = last_used
        self.last_updated = last_updated or self.created_at
        self.confidence = confidence
        self.tags = tags or []
        self.reasoning_examples = reasoning_examples or []
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of this strategy."""
        if self.total_attempts == 0:
            return 0.0
        return self.success_count / self.total_attempts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the strategy to a dictionary for serialization."""
        return {
            "strategy_id": self.strategy_id,
            "problem_type": self.problem_type,
            "strategy_text": self.strategy_text,
            "examples": self.examples,
            "success_count": self.success_count,
            "total_attempts": self.total_attempts,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "last_updated": self.last_updated,
            "confidence": self.confidence,
            "tags": self.tags,
            "reasoning_examples": self.reasoning_examples
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Strategy':
        """Create a Strategy instance from a dictionary."""
        return cls(
            strategy_id=data["strategy_id"],
            problem_type=data["problem_type"],
            strategy_text=data["strategy_text"],
            examples=data.get("examples", []),
            success_count=data.get("success_count", 0),
            total_attempts=data.get("total_attempts", 0),
            created_at=data.get("created_at"),
            last_used=data.get("last_used"),
            last_updated=data.get("last_updated"),
            confidence=data.get("confidence", 0.5),
            tags=data.get("tags", []),
            reasoning_examples=data.get("reasoning_examples", [])
        )
    
    def record_attempt(self, success: bool) -> None:
        """Record an attempt to use this strategy."""
        self.total_attempts += 1
        if success:
            self.success_count += 1
        self.last_used = datetime.now().isoformat()
        # Update confidence using a smoothed estimate
        alpha = 0.1  # Learning rate
        self.confidence = (1 - alpha) * self.confidence + alpha * (1.0 if success else 0.0)
    
    def update_strategy(self, new_strategy_text: str) -> None:
        """Update the strategy text with a refined version."""
        self.strategy_text = new_strategy_text
        self.last_updated = datetime.now().isoformat()
    
    def add_reasoning_example(self, reasoning: str) -> None:
        """Add a reasoning example to the strategy."""
        if reasoning and reasoning.strip():
            # Keep only a reasonable number of examples (max 5)
            if len(self.reasoning_examples) >= 5:
                self.reasoning_examples.pop(0)  # Remove oldest example
            self.reasoning_examples.append(reasoning.strip())
    
    def add_example(self, example: str) -> None:
        """Add an example to the strategy."""
        if example and example not in self.examples:
            self.examples.append(example)


class StrategyDatabase:
    """Manages a collection of problem-solving strategies."""
    
    def __init__(self, db_path: str = STRATEGY_DB_PATH, metrics_path: str = STRATEGY_METRICS_PATH):
        self.db_path = db_path
        self.metrics_path = metrics_path
        self.strategies: List[Strategy] = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vectors = None
        self.metrics = {
            "total_queries": 0,
            "strategy_applications": 0,
            "strategies_created": 0,
            "strategies_refined": 0,
            "successful_resolutions": 0,
            "last_strategy_id": 0,
            "reasoning_examples_collected": 0,
            "strategies_merged": 0
        }
        self._load()
    
    def _load(self) -> None:
        """Load strategies and metrics from disk."""
        # Load strategies
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.strategies = [Strategy.from_dict(s) for s in data]
                    
                    # Update last_strategy_id based on loaded strategies
                    for strategy in self.strategies:
                        # Extract the numeric part from strategy_id (e.g., "strategy_42" -> 42)
                        if strategy.strategy_id.startswith("strategy_"):
                            try:
                                strategy_num = int(strategy.strategy_id.split("_")[1])
                                self.metrics["last_strategy_id"] = max(
                                    self.metrics["last_strategy_id"], 
                                    strategy_num
                                )
                            except ValueError:
                                # Skip if the ID isn't in the expected format
                                pass
                            
                logger.info(f"Loaded {len(self.strategies)} strategies from {self.db_path}")
                logger.info(f"Last strategy ID is {self.metrics['last_strategy_id']}")
            except Exception as e:
                logger.error(f"Error loading strategies: {str(e)}")
                self.strategies = []
        
        # Load metrics
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as f:
                    metrics = json.load(f)
                    
                    # Update metrics but keep the last_strategy_id we calculated from the strategies
                    last_id = self.metrics["last_strategy_id"]
                    self.metrics.update(metrics)
                    
                    # Use the larger of the two values to be safe
                    if "last_strategy_id" in metrics:
                        self.metrics["last_strategy_id"] = max(last_id, metrics["last_strategy_id"])
                    
                logger.info(f"Loaded metrics from {self.metrics_path}")
                logger.info(f"Last strategy ID is {self.metrics['last_strategy_id']}")
            except Exception as e:
                logger.error(f"Error loading metrics: {str(e)}")
    
    def _save(self) -> None:
        """Save strategies and metrics to disk."""
        # Save strategies
        try:
            with open(self.db_path, 'w') as f:
                json.dump([s.to_dict() for s in self.strategies], f, indent=2)
            logger.info(f"Saved {len(self.strategies)} strategies to {self.db_path}")
        except Exception as e:
            logger.error(f"Error saving strategies: {str(e)}")
        
        # Save metrics
        try:
            with open(self.metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"Saved metrics to {self.metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def add_strategy(self, strategy: Strategy) -> None:
        """Add a new strategy to the database."""
        # Extract the ID number from the strategy_id
        if strategy.strategy_id.startswith("strategy_"):
            try:
                strategy_num = int(strategy.strategy_id.split("_")[1])
                # Update last_strategy_id if this is a higher number
                self.metrics["last_strategy_id"] = max(self.metrics["last_strategy_id"], strategy_num)
            except ValueError:
                # If the format is unexpected, don't update the counter
                pass
                
        # First, check if this is a new problem type that we don't have any strategies for
        exists = any(s.problem_type == strategy.problem_type for s in self.strategies)
        
        # Add strategy to database
        self.strategies.append(strategy)
        self.vectors = None  # Invalidate vector cache
        self.metrics["strategies_created"] += 1
        self._save()
        
        # Log whether this is a new problem type
        if not exists:
            logger.info(f"Added first strategy for problem type: {strategy.problem_type}")
        else:
            logger.info(f"Added additional strategy for problem type: {strategy.problem_type}")
    
    def get_strategies_for_problem(self, problem_type: str) -> List[Strategy]:
        """Get all strategies for a specific problem type."""
        return [s for s in self.strategies if s.problem_type == problem_type]
    
    def get_strategy_by_id(self, strategy_id: str) -> Optional[Strategy]:
        """Get a strategy by its ID."""
        for strategy in self.strategies:
            if strategy.strategy_id == strategy_id:
                return strategy
        return None
    
    def update_strategy_performance(self, strategy_id: str, success: bool) -> None:
        """Update the performance metrics for a strategy."""
        strategy = self.get_strategy_by_id(strategy_id)
        if strategy:
            strategy.record_attempt(success)
            self.metrics["strategy_applications"] += 1
            if success:
                self.metrics["successful_resolutions"] += 1
            self._save()
    
    def refine_strategy(self, strategy_id: str, refined_text: str) -> None:
        """Refine a strategy based on new insights."""
        strategy = self.get_strategy_by_id(strategy_id)
        if strategy:
            strategy.update_strategy(refined_text)
            self.metrics["strategies_refined"] += 1
            self._save()
    
    def add_reasoning_example(self, strategy_id: str, reasoning: str) -> None:
        """Add a reasoning example to a strategy."""
        strategy = self.get_strategy_by_id(strategy_id)
        if strategy and reasoning:
            strategy.add_reasoning_example(reasoning)
            self.metrics["reasoning_examples_collected"] += 1
            self._save()
    
    def add_example_to_strategy(self, strategy_id: str, example: str) -> None:
        """Add an example to a strategy."""
        strategy = self.get_strategy_by_id(strategy_id)
        if strategy and example:
            strategy.add_example(example)
            self._save()
    
    def get_similar_strategies(self, query: str, n: int = 5) -> List[Tuple[Strategy, float]]:
        """Find strategies similar to a query using TF-IDF similarity."""
        if not self.strategies:
            return []
        
        # Extract strategy texts and update vectorizer
        strategy_texts = [s.strategy_text for s in self.strategies]
        if self.vectors is None or len(self.vectors.shape) == 0 or self.vectors.shape[0] != len(strategy_texts):
            try:
                self.vectors = self.vectorizer.fit_transform(strategy_texts)
            except Exception as e:
                logger.error(f"Error creating strategy vectors: {str(e)}")
                return []
        
        # Convert query to vector and find similarities
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            
            # Get top strategies with their similarity scores
            sorted_indices = similarities.argsort()[::-1]
            return [(self.strategies[i], float(similarities[i])) for i in sorted_indices[:n]]
        except Exception as e:
            logger.error(f"Error finding similar strategies: {str(e)}")
            return []
    
    def find_similar_strategy(self, problem_type: str, query: str, threshold: float = STRATEGY_CREATION_THRESHOLD) -> Optional[Tuple[Strategy, float]]:
        """
        Find a strategy of the same problem type that is similar to the query.
        
        Args:
            problem_type: The problem type to match
            query: The query to find similarity against
            threshold: The similarity threshold to consider a match
            
        Returns:
            Optional[Tuple[Strategy, float]]: The most similar strategy and its similarity score,
                                             or None if no similar strategy is found
        """
        if not self.strategies:
            return None
        
        # Get strategies of the specified problem type
        type_strategies = [s for s in self.strategies if s.problem_type == problem_type]
        if not type_strategies:
            return None
        
        try:
            # Vectorize strategy texts
            strategy_texts = [s.strategy_text for s in type_strategies]
            vectorizer = TfidfVectorizer(stop_words='english')
            vectors = vectorizer.fit_transform(strategy_texts + [query])
            
            # Calculate similarities
            query_vector = vectors[-1]
            strategy_vectors = vectors[:-1]
            similarities = cosine_similarity(query_vector, strategy_vectors).flatten()
            
            # Find the most similar strategy
            if len(similarities) > 0:
                max_idx = similarities.argmax()
                max_similarity = similarities[max_idx]
                
                if max_similarity >= threshold:
                    return (type_strategies[max_idx], float(max_similarity))
        
        except Exception as e:
            logger.error(f"Error finding similar strategy: {str(e)}")
        
        return None
    
    def find_similar_examples(self, problem_type: str, query: str, threshold: float = STRATEGY_CREATION_THRESHOLD) -> Optional[Tuple[Strategy, float]]:
        """
        Find a strategy of the same problem type with examples similar to the query.
        
        Args:
            problem_type: The problem type to match
            query: The query to find similarity against
            threshold: The similarity threshold to consider a match
            
        Returns:
            Optional[Tuple[Strategy, float]]: The strategy with the most similar examples and the similarity score,
                                             or None if no similar strategy is found
        """
        if not self.strategies:
            return None
        
        # Get strategies of the specified problem type
        type_strategies = [s for s in self.strategies if s.problem_type == problem_type]
        if not type_strategies:
            return None
        
        max_similarity = 0.0
        most_similar_strategy = None
        
        try:
            for strategy in type_strategies:
                if not strategy.examples:
                    continue
                
                # Vectorize examples and query
                vectorizer = TfidfVectorizer(stop_words='english')
                vectors = vectorizer.fit_transform(strategy.examples + [query])
                
                # Calculate similarities
                query_vector = vectors[-1]
                example_vectors = vectors[:-1]
                similarities = cosine_similarity(query_vector, example_vectors).flatten()
                
                # Get the maximum similarity for this strategy
                if len(similarities) > 0:
                    strategy_max_similarity = similarities.max()
                    
                    if strategy_max_similarity > max_similarity:
                        max_similarity = strategy_max_similarity
                        most_similar_strategy = strategy
            
            if most_similar_strategy and max_similarity >= threshold:
                return (most_similar_strategy, float(max_similarity))
                
        except Exception as e:
            logger.error(f"Error finding similar examples: {str(e)}")
        
        return None
    
    def get_next_strategy_id(self) -> str:
        """Generate a unique ID for a new strategy."""
        self.metrics["last_strategy_id"] += 1
        new_id = f"strategy_{self.metrics['last_strategy_id']}"
        logger.info(f"Generated new strategy ID: {new_id}")
        return new_id
    
    def increment_query_count(self) -> None:
        """Increment the total query count."""
        self.metrics["total_queries"] += 1
        self._save()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics."""
        return self.metrics.copy()
    
    def prune_strategies(self, min_success_rate: float = 0.3, min_attempts: int = 5) -> int:
        """Prune strategies with poor performance."""
        initial_count = len(self.strategies)
        self.strategies = [
            s for s in self.strategies if 
            s.total_attempts < min_attempts or s.success_rate >= min_success_rate
        ]
        pruned_count = initial_count - len(self.strategies)
        if pruned_count > 0:
            self.vectors = None  # Invalidate vector cache
            self._save()
        return pruned_count
    
    def merge_similar_strategies(self, similarity_threshold: float = STRATEGY_MERGING_THRESHOLD) -> int:
        """Merge strategies that are very similar to each other."""
        if len(self.strategies) <= 1:
            return 0
        
        merged_count = 0
        i = 0
        
        while i < len(self.strategies):
            j = i + 1
            while j < len(self.strategies):
                # Check if strategies are of the same problem type
                if self.strategies[i].problem_type == self.strategies[j].problem_type:
                    # Get strategy texts
                    text_i = self.strategies[i].strategy_text
                    text_j = self.strategies[j].strategy_text
                    
                    # Calculate similarity using TF-IDF
                    vectorizer = TfidfVectorizer(stop_words='english')
                    vectors = vectorizer.fit_transform([text_i, text_j])
                    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                    
                    if similarity >= similarity_threshold:
                        # Merge strategies
                        merged_strategy = self._merge_two_strategies(self.strategies[i], self.strategies[j])
                        # Replace the first strategy with the merged one
                        self.strategies[i] = merged_strategy
                        # Remove the second strategy
                        self.strategies.pop(j)
                        merged_count += 1
                        self.metrics["strategies_merged"] += 1
                        logger.info(f"Merged strategies {self.strategies[i].strategy_id} and {merged_strategy.strategy_id} with similarity {similarity:.2f}")
                    else:
                        j += 1
                else:
                    j += 1
            i += 1
        
        if merged_count > 0:
            self.vectors = None  # Invalidate vector cache
            self._save()
        
        return merged_count
    
    def _merge_two_strategies(self, strategy1: Strategy, strategy2: Strategy) -> Strategy:
        """Merge two similar strategies into one."""
        # Use the strategy with the higher success rate as the base
        if strategy1.success_rate >= strategy2.success_rate:
            base, other = strategy1, strategy2
        else:
            base, other = strategy2, strategy1
        
        # Create a new merged strategy
        merged = Strategy(
            strategy_id=base.strategy_id,
            problem_type=base.problem_type,
            strategy_text=base.strategy_text,
            examples=list(set(base.examples + other.examples)),
            success_count=base.success_count + other.success_count,
            total_attempts=base.total_attempts + other.total_attempts,
            created_at=min(base.created_at, other.created_at) if base.created_at and other.created_at else base.created_at,
            last_used=max(base.last_used, other.last_used) if base.last_used and other.last_used else base.last_used,
            last_updated=datetime.now().isoformat(),
            confidence=(base.confidence + other.confidence) / 2,
            tags=list(set(base.tags + other.tags)),
            reasoning_examples=base.reasoning_examples + other.reasoning_examples
        )
        
        # Keep only a reasonable number of reasoning examples (max 5)
        if len(merged.reasoning_examples) > 5:
            merged.reasoning_examples = merged.reasoning_examples[-5:]
        
        return merged
    
    def limit_strategies_per_type(self, max_per_type: int = MAX_STRATEGIES_PER_TYPE) -> int:
        """
        Limit the number of strategies per problem type to the specified maximum in the database.
        This controls storage limit, not the number of strategies used during inference.
        Keeps the best performing strategies based on success rate and recency.
        
        Args:
            max_per_type: Maximum number of strategies to keep per problem type
            
        Returns:
            int: Number of strategies removed
        """
        # Group strategies by problem type
        strategies_by_type = {}
        for strategy in self.strategies:
            if strategy.problem_type not in strategies_by_type:
                strategies_by_type[strategy.problem_type] = []
            strategies_by_type[strategy.problem_type].append(strategy)
        
        # Keep track of strategies to remove
        to_remove = []
        
        # For each problem type, keep only the best max_per_type strategies
        for problem_type, strategies in strategies_by_type.items():
            if len(strategies) <= max_per_type:
                continue
            
            # Score strategies based on success rate (70%) and recency (30%)
            scored_strategies = []
            for strategy in strategies:
                recency_score = 0
                if strategy.last_used:
                    last_used = datetime.fromisoformat(strategy.last_used)
                    days_since = (datetime.now() - last_used).days
                    recency_score = max(0, 1.0 - min(1.0, days_since / 30.0))
                
                score = (0.7 * strategy.success_rate) + (0.3 * recency_score)
                scored_strategies.append((strategy, score))
            
            # Sort by score (descending)
            scored_strategies.sort(key=lambda x: x[1], reverse=True)
            
            # Mark excess strategies for removal
            for strategy, _ in scored_strategies[max_per_type:]:
                to_remove.append(strategy)
        
        # Remove marked strategies
        initial_count = len(self.strategies)
        self.strategies = [s for s in self.strategies if s not in to_remove]
        removed_count = initial_count - len(self.strategies)
        
        if removed_count > 0:
            self.vectors = None  # Invalidate vector cache
            self._save()
            logger.info(f"Removed {removed_count} excess strategies to maintain max {max_per_type} per type in database (storage limit)")
        
        return removed_count
