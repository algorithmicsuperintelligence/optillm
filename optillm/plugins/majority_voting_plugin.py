"""
Majority Voting Plugin for OptILLM

Generic implementation that generates multiple candidates and selects
the most common response through simple voting.
"""

import re
import logging
from typing import Tuple, Dict, Any
from collections import Counter

logger = logging.getLogger(__name__)

# Plugin identifier
SLUG = "majority_voting"

# Default configuration
DEFAULT_K = 8
DEFAULT_TEMPERATURE = 0.6  # Unified temperature for consistency


def normalize_response(response: str) -> str:
    """
    Basic normalization for comparing responses.
    Removes extra whitespace, punctuation at ends, and lowercases.
    """
    if not response:
        return ""
    
    # Remove thinking blocks if present
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Basic normalization
    response = response.strip()
    response = response.lower()
    
    # Remove trailing punctuation
    response = response.rstrip('.,;:!?')
    
    # Normalize whitespace
    response = ' '.join(response.split())
    
    return response


def extract_final_answer(response: str) -> str:
    """
    Try to extract just the final answer from a response.
    This is generic and looks for common patterns.
    """
    if not response:
        return response
    
    # Remove thinking blocks
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    
    # Look for common answer patterns
    patterns = [
        r'(?:final answer|answer):\s*(.+?)(?:\n|$)',
        r'(?:the answer is|answer is)\s*(.+?)(?:\n|$)',
        r'###\s*(.+?)(?:\n|$)',  # Common in math problems
        r'^([A-E])\b',  # Single letter at start
        r'\b([A-E])\b\s*$',  # Single letter at end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    # If no pattern found, return the whole response
    return response


def run(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    request_config: Dict[str, Any] = None
) -> Tuple[str, int]:
    """
    Generic majority voting implementation.
    """
    logger.info("Starting majority voting process")
    
    # Extract parameters
    k = request_config.get('k', DEFAULT_K) if request_config else DEFAULT_K
    temperature = request_config.get('temperature', DEFAULT_TEMPERATURE) if request_config else DEFAULT_TEMPERATURE
    max_tokens = request_config.get('max_tokens', 4096) if request_config else 4096
    
    logger.info(f"Generating {k} candidates with temperature={temperature}")
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_query}
    ]
    
    # Generate candidates
    candidates = []
    total_tokens = 0
    
    try:
        # Try parallel generation first
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            n=k,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        candidates = [choice.message.content for choice in response.choices]
        total_tokens = response.usage.completion_tokens
        
    except Exception as e:
        logger.warning(f"Parallel generation failed: {str(e)}")
        # Fallback to sequential
        for i in range(k):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                candidates.append(response.choices[0].message.content)
                total_tokens += response.usage.completion_tokens
            except Exception as err:
                logger.error(f"Error generating candidate {i+1}: {str(err)}")
                continue
    
    if not candidates:
        return "Error: Could not generate any candidates", 0
    
    # Extract and normalize answers for voting
    answer_votes = Counter()
    answer_to_responses = {}
    
    for i, candidate in enumerate(candidates):
        # Try to extract just the answer part
        answer = extract_final_answer(candidate)
        
        # Normalize for comparison
        normalized = normalize_response(answer)
        
        if normalized:
            answer_votes[normalized] += 1
            
            # Keep track of original responses for each normalized answer
            if normalized not in answer_to_responses:
                answer_to_responses[normalized] = []
            answer_to_responses[normalized].append(candidate)
            
            logger.debug(f"Candidate {i+1}: '{answer}' -> '{normalized}'")
        else:
            logger.warning(f"Could not extract/normalize answer from candidate {i+1}")
    
    # Select the most voted answer
    if answer_votes:
        most_common_normalized, count = answer_votes.most_common(1)[0]
        logger.info(f"Most common answer: '{most_common_normalized}' with {count}/{k} votes")
        
        # Return the first original response that mapped to this answer
        winning_responses = answer_to_responses[most_common_normalized]
        return winning_responses[0], total_tokens
    else:
        # If no answers could be extracted, return the first candidate
        logger.warning("No answers could be extracted, returning first candidate")
        return candidates[0], total_tokens