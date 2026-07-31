"""
Utility functions for the System Prompt Learning (SPL) plugin.
"""

import re
import logging
from typing import Tuple, Optional, List, Any

from optillm.plugins.spl.prompts import STRATEGY_APPLICATION_PROMPT

# Setup logging
logger = logging.getLogger(__name__)

def extract_thinking(response: str) -> Tuple[str, Optional[str]]:
    """
    Extract thinking content from <think>...</think> tags and the response after.
    
    Args:
        response: The model's response
    
    Returns:
        Tuple[str, Optional[str]]: The cleaned response and the thinking content (if any)
    """
    thinking_content = None
    final_response = response
    
    # Check if there are thinking tags
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, response, re.DOTALL)
    
    if think_matches:
        # Extract thinking content (concatenate if multiple blocks)
        thinking_content = "\n".join(think_matches)
        
        # Extract the response part (everything after the last </think> tag)
        final_parts = response.split('</think>')
        if len(final_parts) > 1:
            final_response = final_parts[-1].strip()
    
    return final_response, thinking_content

def augment_system_prompt(system_prompt: str, strategies: List[Any]) -> str:
    """
    Augment the system prompt with selected strategies and reasoning examples.
    Instructs the LLM to apply the strategies in its solution.
    
    Args:
        system_prompt: The original system prompt
        strategies: A list of strategies to add to the prompt
    
    Returns:
        str: The augmented system prompt
    """
    if not strategies:
        return system_prompt
    
    # Build the individual strategy sections
    strategies_section = ""
    
    for i, strategy in enumerate(strategies, 1):
        strategies_section += f"Strategy {i} for {strategy.problem_type} problems:\n{strategy.strategy_text}\n\n"
        
        # Add a sample reasoning example if available
        if strategy.reasoning_examples:
            # Use the most recent reasoning example (last one)
            reasoning = strategy.reasoning_examples[-1]
            if reasoning:
                strategies_section += f"Example reasoning process:\n<think>\n{reasoning}\n</think>\n\n"
    
    # Format the application prompt with the strategies section
    strategy_prompt = STRATEGY_APPLICATION_PROMPT.format(strategies_section=strategies_section)
    
    # Append the formatted strategy section to the system prompt
    augmented_prompt = system_prompt + "\n\n" + strategy_prompt
    
    return augmented_prompt
