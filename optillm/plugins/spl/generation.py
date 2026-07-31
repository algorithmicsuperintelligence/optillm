"""
Functions for generating strategies in the System Prompt Learning (SPL) plugin.
"""

import uuid
import logging
from typing import Tuple, Optional, List

from optillm.plugins.spl.strategy import Strategy, StrategyDatabase
from optillm.plugins.spl.utils import extract_thinking
from optillm.plugins.spl.prompts import (
    PROBLEM_CLASSIFICATION_PROMPT,
    STRATEGY_GENERATION_PROMPT
)
from optillm.plugins.spl.config import (
    VALID_PROBLEM_TYPES,
    DEFAULT_MAX_TOKENS,
    STRATEGY_CREATION_THRESHOLD,
    MAX_STRATEGIES_PER_TYPE
)

# Setup logging
logger = logging.getLogger(__name__)

def classify_problem(content: str, client, model: str) -> str:
    """
    Use the LLM to classify the problem type, ensuring the result is one of the valid types.
    
    Args:
        content: The query/problem to classify
        client: LLM client for making API calls
        model: Model identifier
    
    Returns:
        str: The problem type classification (always a valid type)
    """
    # Format problem types as a comma-separated list
    problem_types_str = ", ".join(VALID_PROBLEM_TYPES[:-1])  # Exclude the general_problem fallback
    
    try:
        messages = [
            {
                "role": "system", 
                "content": PROBLEM_CLASSIFICATION_PROMPT.format(problem_types=problem_types_str)
            },
            {
                "role": "user", 
                "content": (
                    f"Classify the following problem into ONE of these types: {problem_types_str}\n\n"
                    f"Problem: {content}"
                )
            }
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,  # Low temperature for more deterministic output
            max_tokens=DEFAULT_MAX_TOKENS  # Increased token limit for reasoning LLMs
        )
        
        # Extract final response and thinking content
        raw_response = response.choices[0].message.content
        final_response, thinking = extract_thinking(raw_response)
        
        # Clean and normalize the response
        final_response = final_response.strip().lower()
        
        logger.debug(f"Problem classification - raw response: '{raw_response}'")
        logger.debug(f"Problem classification - final response after removing thinking: '{final_response}'")
        
        # Find the exact match from our list of valid types
        for valid_type in VALID_PROBLEM_TYPES:
            if valid_type.lower() == final_response:
                logger.info(f"Classified problem as '{valid_type}' (exact match)")
                return valid_type
        
        # If no exact match, look for partial matches
        for valid_type in VALID_PROBLEM_TYPES:
            if valid_type.lower() in final_response:
                logger.info(f"Classified problem as '{valid_type}' (partial match from '{final_response}')")
                return valid_type
        
        # If still no match, return the general_problem fallback
        logger.warning(f"Could not match '{final_response}' to any valid problem type, using 'general_problem'")
        return "general_problem"
    
    except Exception as e:
        logger.error(f"Error classifying problem: {str(e)}")
        return "general_problem"  # Default fallback

def generate_strategy(problem: str, problem_type: str, client, model: str, db: StrategyDatabase) -> Strategy:
    """
    Generate a new problem-solving strategy using the LLM.
    
    Args:
        problem: The problem that needs a strategy
        problem_type: The type of problem
        client: LLM client for making API calls
        model: Model identifier
        db: The strategy database to use for generating IDs
    
    Returns:
        Strategy: A new strategy for solving this type of problem
    """
    try:
        messages = [
            {
                "role": "system", 
                "content": STRATEGY_GENERATION_PROMPT
            },
            {
                "role": "user", 
                "content": (
                    f"Create a problem-solving strategy for the following {problem_type} problem:\n\n"
                    f"{problem}\n\n"
                    f"This strategy should help solve not just this specific problem, but any {problem_type} problem."
                )
            }
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,  # Medium temperature for creative but focused output
            max_tokens=DEFAULT_MAX_TOKENS  # Increased token limit for reasoning LLMs
        )
        
        response_text = response.choices[0].message.content
        
        # Extract final strategy and thinking
        strategy_text, thinking = extract_thinking(response_text)
        if not strategy_text.strip():
            strategy_text = response_text  # Use full response if extraction failed
        
        logger.debug(f"Generated strategy - raw response: '{response_text}'")
        logger.debug(f"Generated strategy - final text after removing thinking: '{strategy_text}'")
        
        # Create a new strategy object using the provided database for ID generation
        strategy = Strategy(
            strategy_id=db.get_next_strategy_id(),  # Use the provided DB instance
            problem_type=problem_type,
            strategy_text=strategy_text.strip(),
            examples=[problem],
            created_at=None,  # Use default
            reasoning_examples=[thinking] if thinking else []
        )
        
        logger.info(f"Generated new strategy for {problem_type}: ID {strategy.strategy_id}")
        return strategy
    
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}")
        # Create a minimal fallback strategy with a unique ID
        fallback_id = f"fallback_{uuid.uuid4().hex[:8]}"
        logger.info(f"Using fallback strategy with ID: {fallback_id}")
        return Strategy(
            strategy_id=fallback_id,
            problem_type=problem_type,
            strategy_text=(
                f"When solving {problem_type} problems:\n"
                "1. Break down the problem into smaller parts\n"
                "2. Solve each part systematically\n"
                "3. Combine the solutions"
            ),
            examples=[problem]
        )

def should_create_new_strategy(problem_type: str, query: str, existing_strategies: List[Strategy], db: StrategyDatabase) -> Tuple[bool, Optional[Strategy]]:
    """
    Determine whether to create a new strategy or update an existing one.
    
    Args:
        problem_type: The type of problem
        query: The current query/problem
        existing_strategies: Existing strategies for this problem type
        db: Strategy database
    
    Returns:
        Tuple[bool, Optional[Strategy]]: 
            - Boolean indicating if a new strategy should be created
            - The similar strategy to update (if any)
    """
    # If there are no existing strategies, definitely create one
    if not existing_strategies:
        return True, None
    
    # If we already have enough strategies for this problem type (storage limit reached),
    # check if the query is similar to any existing strategy
    if len(existing_strategies) >= MAX_STRATEGIES_PER_TYPE:  # Using storage limit here, not inference limit
        # First, check similarity based on strategy text
        similar_strategy_result = db.find_similar_strategy(problem_type, query)
        if similar_strategy_result:
            similar_strategy, similarity = similar_strategy_result
            logger.info(f"Found similar strategy {similar_strategy.strategy_id} with text similarity {similarity:.2f}")
            return False, similar_strategy
            
        # Next, check similarity based on examples
        similar_examples_result = db.find_similar_examples(problem_type, query)
        if similar_examples_result:
            similar_strategy, similarity = similar_examples_result
            logger.info(f"Found strategy {similar_strategy.strategy_id} with similar examples, similarity {similarity:.2f}")
            return False, similar_strategy
            
        # If we're at or above max strategies and no similar strategy was found,
        # use the strategy with the lowest success rate for this new example
        if existing_strategies:
            # Sort by success rate (ascending)
            existing_strategies.sort(key=lambda s: s.success_rate)
            worst_strategy = existing_strategies[0]
            logger.info(f"At maximum strategies for {problem_type}, updating lowest performing strategy {worst_strategy.strategy_id}")
            return False, worst_strategy
    
    # If we have fewer than the maximum allowed strategies for this type,
    # check strategy similarity before creating a new one
    similar_strategy_result = db.find_similar_strategy(problem_type, query, threshold=STRATEGY_CREATION_THRESHOLD)
    if similar_strategy_result:
        similar_strategy, similarity = similar_strategy_result
        logger.info(f"Found similar strategy {similar_strategy.strategy_id} with text similarity {similarity:.2f}")
        return False, similar_strategy
    
    # If we get here, we should create a new strategy
    logger.info(f"No similar strategy found for {problem_type}, creating a new one")
    return True, None
