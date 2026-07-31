"""
Main implementation of the System Prompt Learning (SPL) plugin.
"""

import time
import logging
from typing import Tuple

from .strategy import StrategyDatabase
from .generation import (
    classify_problem,
    generate_strategy,
    should_create_new_strategy
)
from .evaluation import (
    select_relevant_strategies,
    evaluate_strategy_effectiveness,
    refine_strategy
)
from .utils import (
    extract_thinking,
    augment_system_prompt
)
from .config import (
    DEFAULT_MAX_TOKENS,
    MAINTENANCE_INTERVAL,
    STRATEGY_MERGING_THRESHOLD,
    MAX_STRATEGIES_PER_TYPE,
    MAX_STRATEGIES_FOR_INFERENCE,
    MIN_SUCCESS_RATE_FOR_INFERENCE
)

# Setup logging
logger = logging.getLogger(__name__)

def run_spl(system_prompt: str, initial_query: str, client, model: str, request_config: dict = None) -> Tuple[str, int]:
    """
    Main plugin function that implements system prompt learning.
    
    By default, the plugin runs in inference-only mode, which uses existing strategies without modifying them.
    Setting request_config['spl_learning'] = True enables learning mode to create and refine strategies.
    
    Args:
        system_prompt: The system prompt
        initial_query: The user's query
        client: The LLM client
        model: The model identifier
        request_config: Optional request configuration
                       Can include {'spl_learning': True} to enable learning mode
    
    Returns:
        Tuple[str, int]: The LLM response and token count
    """
    start_time = time.time()
    logger.info(f"Starting SPL plugin execution for query: {initial_query[:100]}...")
    
    # Check if we should enable learning mode
    learning_mode = False
    if request_config and 'spl_learning' in request_config:
        learning_mode = request_config['spl_learning']
        logger.info(f"Running in learning mode: {learning_mode}")
        
    # Initialize the strategy database
    db = StrategyDatabase()
    logger.info(f"Current strategy count: {len(db.strategies)}")
    logger.info(f"Last strategy ID: {db.metrics.get('last_strategy_id', 0)}")
    
    # Only increment query count in learning mode
    if learning_mode:
        db.increment_query_count()
        db._save()  # Save immediately to ensure counter is persisted
    
    # 1. Classify the problem type
    problem_type = classify_problem(initial_query, client, model)
    logger.info(f"Classified problem as: {problem_type}")
    
    # 2. Get existing strategies for this problem type
    existing_strategies = db.get_strategies_for_problem(problem_type)
    logger.info(f"Found {len(existing_strategies)} existing strategies for {problem_type}")
    
    # 3. Determine if we need to create a new strategy or update an existing one
    similar_strategy = None
    
    if learning_mode:
        # In learning mode, check if we should create a new strategy or update an existing one
        should_create, similar_strategy = should_create_new_strategy(
            problem_type, 
            initial_query, 
            existing_strategies, 
            db
        )
        
        if should_create:
            # Create a new strategy
            logger.info(f"Creating new strategy for {problem_type}")
            new_strategy = generate_strategy(initial_query, problem_type, client, model, db)
            db.add_strategy(new_strategy)
            logger.info(f"Added new strategy with ID: {new_strategy.strategy_id}")
        
        elif similar_strategy:
            # Update existing strategy with new example
            logger.info(f"Updating existing strategy {similar_strategy.strategy_id} with new example")
            db.add_example_to_strategy(similar_strategy.strategy_id, initial_query)
    
    # 4. Perform database maintenance (more frequently than before)
    if learning_mode and db.metrics["total_queries"] % MAINTENANCE_INTERVAL == 0:
        # 4.1 Merge similar strategies
        merged_count = db.merge_similar_strategies(similarity_threshold=STRATEGY_MERGING_THRESHOLD)
        logger.info(f"Merged {merged_count} similar strategies")
        
        # 4.2 Limit strategies per problem type (applies storage limit, not inference limit)
        db.limit_strategies_per_type(max_per_type=MAX_STRATEGIES_PER_TYPE)
        
        # 4.3 Prune low-performing strategies
        pruned_count = db.prune_strategies()
        logger.info(f"Pruned {pruned_count} low-performing strategies")
    
    # 5. Re-select strategies (in case the database changed in step 4)
    existing_strategies = db.get_strategies_for_problem(problem_type)
    
    # 6. Select relevant strategies for this problem (using inference limit)
    selected_strategies = select_relevant_strategies(initial_query, problem_type, db, learning_mode, MAX_STRATEGIES_FOR_INFERENCE)
    
    # Log the selected strategies
    for i, strategy in enumerate(selected_strategies, 1):
        logger.info(f"Selected strategy {i}/{MAX_STRATEGIES_FOR_INFERENCE} for inference: {strategy.strategy_id} (success rate: {strategy.success_rate:.2f})")
    
    # 7. Handle situation when no strategies are selected
    if not selected_strategies:
        if not existing_strategies:
            # No strategies exist for this problem type
            logger.info(f"No strategies exist for problem type '{problem_type}'. Enable learning mode with 'spl_learning=True' to create strategies.")
        else:
            # Strategies exist but don't meet the minimum success rate
            logger.info(f"Strategies exist for problem type '{problem_type}' but none meet the minimum success rate threshold of {MIN_SUCCESS_RATE_FOR_INFERENCE:.2f}.")
            logger.info("Enable learning mode with 'spl_learning=True' to improve strategies.")
        
        # Use the original system prompt without augmentation
        logger.info("Running without strategy augmentation - using base system prompt only.")
        augmented_prompt = system_prompt
    else:
        # Normal case - strategies were selected
        # Augment the system prompt with the selected strategies
        augmented_prompt = augment_system_prompt(system_prompt, selected_strategies)
        logger.info(f"Augmented system prompt with {len(selected_strategies)} strategies (inference limit: {MAX_STRATEGIES_FOR_INFERENCE})")
    
    # 9. Forward the request to the LLM with the augmented prompt
    try:
        # Create a copy of request_config without spl_learning
        request_params = {}
        if request_config:
            request_params = {k: v for k, v in request_config.items() if k != 'spl_learning'}
        
        # Ensure max_tokens is set to at least DEFAULT_MAX_TOKENS for reasoning LLMs
        if 'max_tokens' not in request_params:
            request_params['max_tokens'] = DEFAULT_MAX_TOKENS
        elif request_params['max_tokens'] < DEFAULT_MAX_TOKENS:
            request_params['max_tokens'] = DEFAULT_MAX_TOKENS
            
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": augmented_prompt},
                {"role": "user", "content": initial_query}
            ],
            **request_params
        )
        
        completion_tokens = response.usage.completion_tokens
        response_text = response.choices[0].message.content
        
        # Extract final response and thinking content
        final_response, thinking = extract_thinking(response_text)
        
        logger.debug(f"Main response - raw: '{response_text}'")
        if thinking:
            logger.debug(f"Main response - thinking extracted: '{thinking}'")
            logger.debug(f"Main response - final answer after removing thinking: '{final_response}'")
        
        # Only perform learning operations if in learning mode and we have strategies
        if learning_mode:
            if selected_strategies:
                # 10. Evaluate the effectiveness of the strategies
                strategy_effectiveness = evaluate_strategy_effectiveness(
                    final_response,
                    thinking,
                    selected_strategies,
                    client,
                    model
                )
                
                # 11. Update strategy metrics based on effectiveness
                for strategy_id, effective in strategy_effectiveness.items():
                    # Skip temporary fallback strategies
                    if strategy_id != "fallback_temporary":
                        db.update_strategy_performance(strategy_id, effective)
                        logger.info(f"Strategy {strategy_id} effectiveness: {effective}")
                        
                        # If the strategy was effective and thinking was used, add the thinking as a reasoning example
                        if effective and thinking and strategy_id != "fallback_temporary":
                            db.add_reasoning_example(strategy_id, thinking)
                            logger.info(f"Added reasoning example to strategy {strategy_id}")
                
                # 12. Periodically refine strategies (after every 10 uses)
                for strategy in selected_strategies:
                    # Skip temporary fallback strategies
                    if (strategy.strategy_id != "fallback_temporary" and 
                        strategy.total_attempts % 10 == 0 and 
                        strategy.total_attempts > 0):
                        logger.info(f"Refining strategy {strategy.strategy_id} after {strategy.total_attempts} attempts")
                        refined_strategy = refine_strategy(strategy, initial_query, final_response, thinking, client, model)
                        db.refine_strategy(strategy.strategy_id, refined_strategy.strategy_text)
            else:
                logger.info("No strategies to evaluate or refine - consider adding strategies for this problem type")
        else:
            logger.info("Strategy evaluation and refinement skipped (not in learning mode)")
        
        # Log execution time and status after run
        execution_time = time.time() - start_time
        logger.info(f"SPL plugin execution completed in {execution_time:.2f} seconds")
        logger.info(f"Final strategy count: {len(db.strategies)}")
        logger.info(f"Final last strategy ID: {db.metrics.get('last_strategy_id', 0)}")
        
        # Return the original response to preserve the thinking tag format
        return response_text, completion_tokens
    
    except Exception as e:
        logger.error(f"Error in SPL plugin: {str(e)}")
        # Fall back to regular completion on error
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_query}
                ],
                max_tokens=DEFAULT_MAX_TOKENS  # Ensure fallback also uses sufficient tokens
            )
            return response.choices[0].message.content, response.usage.completion_tokens
        except Exception as inner_e:
            logger.error(f"Error in fallback completion: {str(inner_e)}")
            # Return a simple error message if even the fallback fails
            return f"Error processing request: {str(e)}", 0
