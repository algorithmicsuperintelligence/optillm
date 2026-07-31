import argparse
import json
import os
import logging
import re
import time
import math
from typing import List, Dict, Tuple, Optional, Union
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
import statistics
from collections import defaultdict

# Add sys path to import optillm modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optillm.utils.answer_extraction import extract_answer as unified_extract_answer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://openrouter.ai/api/v1")

client = OpenAI(api_key="optillm", base_url="http://localhost:8001/v1")

SYSTEM_PROMPT = '''You are solving AIME (American Invitational Mathematics Examination) problems.

Important: Always end your solution with the final answer in one of these two formats:

1. \\[
   \\boxed{X}.
   \\]

2. $n=\\boxed{X}$

where X is your integer answer between 0 and 999.'''

# Define the thought transition phrases to track
THOUGHT_TRANSITIONS = [
    "Wait,", 
    "Alternatively,", 
    "However,", 
    "Additionally,"
]

def load_2024_dataset() -> list[dict]:
    """
    Load the 2024 dataset of problems.
    Returns:
        list[dict]: The dataset of problems.
    """
    dataset_original = load_dataset("AI-MO/aimo-validation-aime")
    # Filter out problems that are not from 2024
    dataset = dataset_original["train"].filter(lambda example: "2024" in example["url"])
    logging.debug(f"Filtered dataset size: {len(dataset)}.")
    assert len(dataset) == 30, f"Expected 30 problems after filtering by 2024, but found {len(dataset)}"
    return dataset

def load_2025_dataset() -> list[dict]:
    """
    Load the 2025 dataset of problems from math-ai/aime25.
    Returns:
        list[dict]: The dataset of problems.
    """
    dataset = load_dataset("math-ai/aime25")
    # The AIME 2025 dataset has 30 problems in the "test" split
    dataset = dataset["test"]
    logging.debug(f"Loaded AIME 2025 dataset size: {len(dataset)}.")
    assert len(dataset) == 30, f"Expected 30 problems in AIME 2025, but found {len(dataset)}"
    return dataset

def load_dataset_by_year(year: int) -> list[dict]:
    """
    Load dataset by year (2024 or 2025).
    Returns:
        list[dict]: The dataset of problems.
    """
    if year == 2024:
        return load_2024_dataset()
    elif year == 2025:
        return load_2025_dataset()
    else:
        raise ValueError(f"Unsupported year: {year}. Only 2024 and 2025 are supported.")

def extract_answer(response: str) -> Optional[int]:
    """
    Extract the numerical answer from a math solution response using unified extraction.
    AIME problems expect integer answers between 0 and 999.
    """
    if not response:
        return None

    # Use unified answer extraction with AIME problem context
    extracted_answer = unified_extract_answer(
        response,
        problem_type="aime",
        problem_id=None
    )

    if extracted_answer is None:
        return None

    # Math-verify returns a list of all possible matches
    # Check if extracted_answer is a list and find first valid integer
    if isinstance(extracted_answer, list):
        for item in extracted_answer:
            if isinstance(item, (int, float)):
                answer = int(item)
                if 0 <= answer <= 999:
                    return answer
            elif isinstance(item, str) and item.isdigit():
                answer = int(item)
                if 0 <= answer <= 999:
                    return answer
        return None

    # Convert to integer if needed - AIME answers are always integers
    if isinstance(extracted_answer, (int, float)):
        answer = int(extracted_answer)
        # AIME answers are typically 0-999
        if 0 <= answer <= 999:
            return answer
    elif isinstance(extracted_answer, str) and extracted_answer.isdigit():
        answer = int(extracted_answer)
        if 0 <= answer <= 999:
            return answer

    return None

def analyze_thinking(response: str) -> Dict:
    """
    Analyze thinking patterns in the response.
    Extract tokens between <think> and </think> tags and count thought transitions.
    
    Args:
        response (str): The model's response text
        
    Returns:
        Dict: Analysis metrics including thinking tokens and thought transitions
    """
    # Default result with zero values
    result = {
        "has_think_tags": False,
        "thinking_tokens": 0,
        "thinking_tokens_text": "",
        "total_tokens": len(response.split()),
        "thought_transitions": 0,
        "transition_counts": {phrase: 0 for phrase in THOUGHT_TRANSITIONS},
        "transition_positions": []
    }
    
    # Extract content between <think> and </think> tags
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    think_match = think_pattern.search(response)
    
    if think_match:
        thinking_text = think_match.group(1)
        result["has_think_tags"] = True
        result["thinking_tokens"] = len(thinking_text.split())
        result["thinking_tokens_text"] = thinking_text
        
        # Count thought transitions
        for phrase in THOUGHT_TRANSITIONS:
            # Find all occurrences of each transition phrase
            for match in re.finditer(r'\b' + re.escape(phrase) + r'\b', thinking_text):
                result["transition_counts"][phrase] += 1
                # Record the approximate token position of the transition
                token_position = len(thinking_text[:match.start()].split())
                result["transition_positions"].append((phrase, token_position))
                
        # Sort transition positions by token position
        result["transition_positions"].sort(key=lambda x: x[1])
        
        # Calculate total transitions
        result["thought_transitions"] = sum(result["transition_counts"].values())
    
    return result

def analyze_logits_probs(logprobs_data: List[Dict]) -> Dict:
    """
    Analyze token probability distributions and entropy patterns.
    
    Args:
        logprobs_data: List of dictionaries containing token and logprob information
        
    Returns:
        Dict: Analysis metrics including entropy statistics
    """
    if not logprobs_data:
        return {
            "entropy_stats": None,
            "transition_entropy": None,
            "token_count": 0
        }
    
    token_entropies = []
    token_probs = []
    token_texts = []
    
    # Process each token's logprobs
    for token_info in logprobs_data:
        if not token_info.get("top_logprobs"):
            continue
        
        # Extract probabilities from logprobs
        probs = []
        for token, logprob in token_info["top_logprobs"].items():
            probs.append(math.exp(logprob))
        
        # Normalize probabilities to sum to 1
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p/total_prob for p in probs]
        
        # Calculate entropy: -sum(p_i * log(p_i))
        entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)
        token_entropies.append(entropy)
        token_probs.append(probs[0] if probs else 0)  # Store top token probability
        token_texts.append(token_info["token"])
    
    # Analyze entropy changes around thought transitions
    transition_entropy = {}
    
    for phrase in THOUGHT_TRANSITIONS:
        # Find indices where this transition phrase begins
        transition_indices = []
        
        # Simple approach: find where token texts match the start of the phrase
        for i, token in enumerate(token_texts):
            if phrase.startswith(token) and i < len(token_texts) - 1:
                # Check if this could be the start of the transition phrase
                # This is a simplification; more complex matching would require full tokenization
                transition_indices.append(i)
        
        # Analyze entropy changes around transitions
        if transition_indices:
            before_entropy = []
            after_entropy = []
            
            for idx in transition_indices:
                # Look at 5 tokens before and after transition
                before_window = max(0, idx-5)
                after_window = min(len(token_entropies), idx+5)
                
                if idx > before_window:
                    before_entropy.extend(token_entropies[before_window:idx])
                if after_window > idx:
                    after_entropy.extend(token_entropies[idx:after_window])
            
            transition_entropy[phrase] = {
                "before_mean": statistics.mean(before_entropy) if before_entropy else 0,
                "after_mean": statistics.mean(after_entropy) if after_entropy else 0,
                "count": len(transition_indices)
            }
    
    # Calculate overall entropy statistics
    entropy_stats = {
        "mean": statistics.mean(token_entropies) if token_entropies else 0,
        "median": statistics.median(token_entropies) if token_entropies else 0,
        "max": max(token_entropies) if token_entropies else 0,
        "min": min(token_entropies) if token_entropies else 0,
        "std": statistics.stdev(token_entropies) if len(token_entropies) > 1 else 0
    }
    
    # Calculate entropy per quartile of generation
    if token_entropies:
        quartile_size = max(1, len(token_entropies) // 4)
        entropy_stats["quartiles"] = [
            statistics.mean(token_entropies[i:i+quartile_size])
            for i in range(0, len(token_entropies), quartile_size)
            if i < len(token_entropies)
        ]
    else:
        entropy_stats["quartiles"] = []
    
    return {
        "entropy_stats": entropy_stats,
        "transition_entropy": transition_entropy,
        "token_count": len(token_entropies)
    }

def get_llm_response(problem: str, model: str, analyze_logits: bool = False, extra_body: dict = None) -> Union[str, List[Dict]]:
    """
    Get response from the LLM for a given problem.
    If multiple choices are returned, formats them as attempt dictionaries.
    
    Args:
        problem (str): The problem text
        model (str): The model identifier
        analyze_logits (bool): Whether to request logprobs
        
    Returns:
        Union[str, List[Dict]]: Either a string response or list of attempt dictionaries
    """
    try:
        # Add logprobs parameters if requested
        kwargs = {}
        if analyze_logits:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 3
        
        # Add extra_body if provided
        if extra_body:
            kwargs["extra_body"] = extra_body
        
        response = client.with_options(timeout=6000.0).chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": SYSTEM_PROMPT + problem}
            ],
            max_tokens=64000,
            **kwargs
        )
        
        # Save raw response if logprobs are requested
        if analyze_logits:
            raw_filename = f"results/raw_responses_{model.replace('/', '_')}.json"
            problem_id = hash(problem) % 10000  # Simple hash to identify the problem
            save_raw_response(raw_filename, problem_id, response.model_dump())
        
        # If there's more than one choice, format as attempts
        if len(response.choices) > 1:
            attempts = []
            for i, choice in enumerate(response.choices):
                response_text = choice.message.content.strip()
                predicted_answer = extract_answer(response_text)
                attempt_data = {
                    "attempt_number": i + 1,
                    "response": response_text,
                    "predicted_answer": predicted_answer
                }
                
                # Add logprobs if available
                if analyze_logits and hasattr(choice.message, 'logprobs') and choice.message.logprobs:
                    attempt_data["logprobs"] = choice.message.logprobs
                
                attempts.append(attempt_data)
            return attempts
            
        # If single choice, return as before
        response_text = response.choices[0].message.content.strip()
        
        # If analyzing logits, return as a dictionary with logprobs
        if analyze_logits and hasattr(response.choices[0].message, 'logprobs') and response.choices[0].message.logprobs:
            return {
                "response": response_text,
                "logprobs": response.choices[0].message.logprobs
            }
        
        # Otherwise return just the text
        return response_text
        
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        if "timeout" in str(e).lower():
            logger.error("API call timed out - consider increasing timeout for complex approaches like MARS")
        raise e  # Re-raise instead of silently returning empty string

def make_n_attempts(problem: str, model: str, n: int, analyze_thoughts: bool = False, analyze_logits: bool = False, extra_body: dict = None) -> List[Dict]:
    """
    Make n attempts to solve a problem and return all responses and predictions.
    
    Args:
        problem (str): The problem text
        model (str): The model identifier
        n (int): Number of attempts to make
        analyze_thoughts (bool): Whether to analyze thinking patterns
        analyze_logits (bool): Whether to analyze token probabilities
        
    Returns:
        List[Dict]: List of dictionaries containing response and predicted answer for each attempt
    """
    attempts = []
    remaining_attempts = n
    
    while remaining_attempts > 0:
        try:
            response = get_llm_response(problem, model, analyze_logits, extra_body)
        except Exception as e:
            logger.error(f"Failed to get response for attempt {n - remaining_attempts + 1}: {e}")
            # Create a failed attempt record
            attempt_data = {
                "attempt_number": len(attempts) + 1,
                "response": f"ERROR: {str(e)}",
                "predicted_answer": None,
                "error": str(e)
            }
            attempts.append(attempt_data)
            remaining_attempts -= 1
            continue
        
        # If response is already formatted as attempts
        if isinstance(response, list):
            for attempt in response:
                if analyze_thoughts:
                    attempt["thought_analysis"] = analyze_thinking(attempt["response"])
                if analyze_logits and "logprobs" in attempt:
                    attempt["logit_analysis"] = analyze_logits_probs(attempt["logprobs"]["content"])
            attempts.extend(response)
            remaining_attempts = n - len(attempts)
        elif isinstance(response, dict) and "response" in response:
            # Process dict response with logprobs
            response_text = response["response"]
            predicted_answer = extract_answer(response_text)
            attempt_data = {
                "attempt_number": len(attempts) + 1,
                "response": response_text,
                "predicted_answer": predicted_answer
            }
            if analyze_thoughts:
                attempt_data["thought_analysis"] = analyze_thinking(response_text)
            if analyze_logits and "logprobs" in response:
                attempt_data["logit_analysis"] = analyze_logits_probs(response["logprobs"]["content"])
            attempts.append(attempt_data)
            remaining_attempts -= 1
        else:
            # Process simple string response
            predicted_answer = extract_answer(response)
            attempt_data = {
                "attempt_number": len(attempts) + 1,
                "response": response,
                "predicted_answer": predicted_answer
            }
            if analyze_thoughts:
                attempt_data["thought_analysis"] = analyze_thinking(response)
            attempts.append(attempt_data)
            remaining_attempts -= 1
    
    return attempts

def evaluate_pass_at_n(attempts: List[Dict], correct_answer: int) -> Tuple[bool, Optional[int]]:
    """
    Evaluate if any of the n attempts got the correct answer.
    
    Args:
        attempts (List[Dict]): List of attempt results
        correct_answer (int): The correct answer
        
    Returns:
        Tuple[bool, Optional[int]]: (whether any attempt was correct, first correct attempt number)
    """
    for attempt in attempts:
        if attempt["predicted_answer"] == correct_answer:
            return True, attempt["attempt_number"]
    return False, None

def load_existing_results(filename: str) -> List[Dict]:
    """Load existing results from file if it exists."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_result(filename: str, result: Dict):
    """Save a single result to the results file."""
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def get_last_processed_index(results: List[Dict]) -> int:
    """Get the index of the last processed problem."""
    if not results:
        return -1
    return max(int(r.get('index', -1)) for r in results)

def analyze_results(results: List[Dict], n: int, analyze_thoughts: bool = False, analyze_logits: bool = False):
    """
    Analyze and print summary statistics of the results.
    
    Args:
        results (List[Dict]): List of evaluation results
        n (int): Number of attempts per problem
        analyze_thoughts (bool): Whether to analyze thinking patterns
        analyze_logits (bool): Whether to analyze token probabilities
    """
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    print("\n=== Results Summary ===")
    print(f"Evaluation mode: pass@{n}")
    print(f"Total problems: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Calculate attempt statistics
    successful_attempts = [r['first_correct_attempt'] for r in results if r['is_correct']]
    if successful_attempts:
        avg_attempts = sum(successful_attempts) / len(successful_attempts)
        print("\nFor correct solutions:")
        print(f"Average attempts needed: {avg_attempts:.2f}")
        print("Attempt distribution:")
        for i in range(1, n + 1):
            count = sum(1 for x in successful_attempts if x == i)
            print(f"  Attempt {i}: {count} problems")
    
    if analyze_thoughts:
        print("\n=== Thinking Pattern Analysis ===")
        
        # Collect metrics about thinking patterns for correct vs incorrect attempts
        correct_attempts = []
        incorrect_attempts = []
        
        for result in results:
            for attempt in result['attempts']:
                if 'thought_analysis' in attempt:
                    if result['is_correct'] and attempt['predicted_answer'] == result['correct_answer']:
                        correct_attempts.append(attempt)
                    else:
                        incorrect_attempts.append(attempt)
        
        # Function to calculate statistics for a group of attempts
        def calc_stats(attempts):
            if not attempts:
                return {
                    "count": 0,
                    "avg_thinking_tokens": 0,
                    "avg_thought_transitions": 0,
                    "transition_usage": {phrase: 0 for phrase in THOUGHT_TRANSITIONS},
                    "has_think_tags_pct": 0
                }
            
            thinking_tokens = [a['thought_analysis']['thinking_tokens'] for a in attempts]
            thought_transitions = [a['thought_analysis']['thought_transitions'] for a in attempts]
            has_think_tags = sum(1 for a in attempts if a['thought_analysis']['has_think_tags'])
            
            # Count total transition usage
            transition_usage = defaultdict(int)
            for attempt in attempts:
                for phrase, count in attempt['thought_analysis']['transition_counts'].items():
                    transition_usage[phrase] += count
            
            return {
                "count": len(attempts),
                "avg_thinking_tokens": statistics.mean(thinking_tokens) if thinking_tokens else 0,
                "median_thinking_tokens": statistics.median(thinking_tokens) if thinking_tokens else 0,
                "min_thinking_tokens": min(thinking_tokens) if thinking_tokens else 0,
                "max_thinking_tokens": max(thinking_tokens) if thinking_tokens else 0,
                "avg_thought_transitions": statistics.mean(thought_transitions) if thought_transitions else 0,
                "median_thought_transitions": statistics.median(thought_transitions) if thought_transitions else 0,
                "transition_usage": dict(transition_usage),
                "has_think_tags_pct": (has_think_tags / len(attempts)) * 100 if attempts else 0
            }
        
        # Calculate statistics
        correct_stats = calc_stats(correct_attempts)
        incorrect_stats = calc_stats(incorrect_attempts)
        all_stats = calc_stats(correct_attempts + incorrect_attempts)
        
        # Print statistics
        print(f"\nOverall Thinking Statistics (All {all_stats['count']} Attempts):")
        print(f"- Average thinking tokens: {all_stats['avg_thinking_tokens']:.2f}")
        print(f"- Median thinking tokens: {all_stats['median_thinking_tokens']}")
        print(f"- Range: {all_stats['min_thinking_tokens']} - {all_stats['max_thinking_tokens']} tokens")
        print(f"- Average thought transitions: {all_stats['avg_thought_transitions']:.2f}")
        print(f"- Median thought transitions: {all_stats['median_thought_transitions']}")
        print(f"- Percentage with <think> tags: {all_stats['has_think_tags_pct']:.2f}%")
        print("- Transition phrase usage:")
        for phrase, count in all_stats['transition_usage'].items():
            print(f"  - {phrase}: {count} occurrences")
        
        print(f"\nCorrect Attempts ({correct_stats['count']}):")
        print(f"- Average thinking tokens: {correct_stats['avg_thinking_tokens']:.2f}")
        print(f"- Median thinking tokens: {correct_stats['median_thinking_tokens']}")
        print(f"- Average thought transitions: {correct_stats['avg_thought_transitions']:.2f}")
        print(f"- Median thought transitions: {correct_stats['median_thought_transitions']}")
        print(f"- Percentage with <think> tags: {correct_stats['has_think_tags_pct']:.2f}%")
        print("- Transition phrase usage:")
        for phrase, count in correct_stats['transition_usage'].items():
            print(f"  - {phrase}: {count} occurrences")
        
        print(f"\nIncorrect Attempts ({incorrect_stats['count']}):")
        print(f"- Average thinking tokens: {incorrect_stats['avg_thinking_tokens']:.2f}")
        print(f"- Median thinking tokens: {incorrect_stats['median_thinking_tokens']}")
        print(f"- Average thought transitions: {incorrect_stats['avg_thought_transitions']:.2f}")
        print(f"- Median thought transitions: {incorrect_stats['median_thought_transitions']}")
        print(f"- Percentage with <think> tags: {incorrect_stats['has_think_tags_pct']:.2f}%")
        print("- Transition phrase usage:")
        for phrase, count in incorrect_stats['transition_usage'].items():
            print(f"  - {phrase}: {count} occurrences")
        
        # Calculate correlation between thinking tokens and correctness
        if correct_attempts and incorrect_attempts:
            print("\nCorrelation Analysis:")
            
            # Find problems with both correct and incorrect attempts for comparison
            problems_with_both = defaultdict(lambda: {"correct": [], "incorrect": []})
            
            for result in results:
                problem_id = result['index']
                for attempt in result['attempts']:
                    if 'thought_analysis' in attempt:
                        category = "correct" if attempt['predicted_answer'] == result['correct_answer'] else "incorrect"
                        problems_with_both[problem_id][category].append(attempt)
            
            # Filter to problems that have both correct and incorrect attempts
            valid_problems = {
                k: v for k, v in problems_with_both.items() 
                if v["correct"] and v["incorrect"]
            }
            
            if valid_problems:
                print(f"Found {len(valid_problems)} problems with both correct and incorrect attempts")
                
                # For each problem, compare thinking patterns
                avg_token_diff = []
                avg_transition_diff = []
                
                for problem_id, attempts in valid_problems.items():
                    correct_tokens = [a['thought_analysis']['thinking_tokens'] for a in attempts['correct']]
                    incorrect_tokens = [a['thought_analysis']['thinking_tokens'] for a in attempts['incorrect']]
                    
                    correct_transitions = [a['thought_analysis']['thought_transitions'] for a in attempts['correct']]
                    incorrect_transitions = [a['thought_analysis']['thought_transitions'] for a in attempts['incorrect']]
                    
                    avg_correct_tokens = statistics.mean(correct_tokens) if correct_tokens else 0
                    avg_incorrect_tokens = statistics.mean(incorrect_tokens) if incorrect_tokens else 0
                    
                    avg_correct_transitions = statistics.mean(correct_transitions) if correct_transitions else 0
                    avg_incorrect_transitions = statistics.mean(incorrect_transitions) if incorrect_transitions else 0
                    
                    avg_token_diff.append(avg_correct_tokens - avg_incorrect_tokens)
                    avg_transition_diff.append(avg_correct_transitions - avg_incorrect_transitions)
                
                print(f"Average token difference (correct - incorrect): {statistics.mean(avg_token_diff):.2f}")
                print(f"Average transition difference (correct - incorrect): {statistics.mean(avg_transition_diff):.2f}")
    
    if analyze_logits:
        print("\n=== Logit Analysis ===")
        
        # Collect metrics about logit patterns for correct vs incorrect attempts
        correct_attempts = []
        incorrect_attempts = []
        
        for result in results:
            for attempt in result['attempts']:
                if 'logit_analysis' in attempt:
                    if result['is_correct'] and attempt['predicted_answer'] == result['correct_answer']:
                        correct_attempts.append(attempt)
                    else:
                        incorrect_attempts.append(attempt)
        
        # Function to calculate logit statistics for a group of attempts
        def calc_logit_stats(attempts):
            if not attempts:
                return {
                    "count": 0,
                    "entropy": None,
                    "transitions": None
                }
            
            # Collect all entropy stats
            entropy_means = []
            entropy_stds = []
            entropy_quartiles = []
            transition_entropies = defaultdict(lambda: {"before": [], "after": []})
            
            for attempt in attempts:
                if attempt['logit_analysis'].get('entropy_stats') and attempt['logit_analysis']['entropy_stats'].get('mean'):
                    entropy_means.append(attempt['logit_analysis']['entropy_stats']['mean'])
                    entropy_stds.append(attempt['logit_analysis']['entropy_stats']['std'])
                    
                    if attempt['logit_analysis']['entropy_stats'].get('quartiles'):
                        entropy_quartiles.append(attempt['logit_analysis']['entropy_stats']['quartiles'])
                    
                    # Collect transition entropy data
                    if attempt['logit_analysis'].get('transition_entropy'):
                        for phrase, stats in attempt['logit_analysis']['transition_entropy'].items():
                            if stats.get('before_mean') is not None:
                                transition_entropies[phrase]["before"].append(stats['before_mean'])
                            if stats.get('after_mean') is not None:
                                transition_entropies[phrase]["after"].append(stats['after_mean'])
            
            # Calculate average entropy quartiles
            avg_quartiles = []
            if entropy_quartiles:
                # Ensure all quartile lists have the same length
                max_quartiles = max(len(q) for q in entropy_quartiles)
                padded_quartiles = [q + [0] * (max_quartiles - len(q)) for q in entropy_quartiles]
                
                # Calculate average for each quartile position
                for i in range(max_quartiles):
                    quartile_values = [q[i] for q in padded_quartiles if i < len(q)]
                    avg_quartiles.append(statistics.mean(quartile_values) if quartile_values else 0)
            
            # Calculate statistics for transitions
            transition_stats = {}
            for phrase, values in transition_entropies.items():
                if values["before"] and values["after"]:
                    before_mean = statistics.mean(values["before"])
                    after_mean = statistics.mean(values["after"])
                    transition_stats[phrase] = {
                        "before_mean": before_mean,
                        "after_mean": after_mean,
                        "entropy_change": after_mean - before_mean,
                        "count": len(values["before"])
                    }
            
            return {
                "count": len(attempts),
                "entropy": {
                    "mean": statistics.mean(entropy_means) if entropy_means else 0,
                    "std": statistics.mean(entropy_stds) if entropy_stds else 0,
                    "quartiles": avg_quartiles
                },
                "transitions": transition_stats
            }
        
        # Calculate statistics
        correct_stats = calc_logit_stats(correct_attempts)
        incorrect_stats = calc_logit_stats(incorrect_attempts)
        all_stats = calc_logit_stats(correct_attempts + incorrect_attempts)
        
        # Print statistics
        print(f"\nOverall Logit Statistics (All {all_stats['count']} Attempts):")
        if all_stats['entropy'] and all_stats['entropy']['mean']:
            print(f"- Average entropy: {all_stats['entropy']['mean']:.4f}")
            print(f"- Average entropy std: {all_stats['entropy']['std']:.4f}")
            
            if all_stats['entropy']['quartiles']:
                print("- Entropy by generation quartile:")
                for i, q in enumerate(all_stats['entropy']['quartiles']):
                    print(f"  - Q{i+1}: {q:.4f}")
            
            if all_stats['transitions']:
                print("- Entropy around thought transitions:")
                for phrase, stats in all_stats['transitions'].items():
                    change = stats['entropy_change']
                    change_dir = "increases" if change > 0 else "decreases"
                    print(f"  - {phrase} (n={stats['count']}): Entropy {change_dir} by {abs(change):.4f}")
                    print(f"    - Before: {stats['before_mean']:.4f}, After: {stats['after_mean']:.4f}")
        
        if correct_stats['count'] > 0 and incorrect_stats['count'] > 0:
            print("\nEntropy Comparison (Correct vs Incorrect Attempts):")
            
            if (correct_stats['entropy'] and correct_stats['entropy']['mean'] and 
                incorrect_stats['entropy'] and incorrect_stats['entropy']['mean']):
                
                correct_entropy = correct_stats['entropy']['mean']
                incorrect_entropy = incorrect_stats['entropy']['mean']
                diff = correct_entropy - incorrect_entropy
                
                print(f"- Correct attempts avg entropy: {correct_entropy:.4f}")
                print(f"- Incorrect attempts avg entropy: {incorrect_entropy:.4f}")
                print(f"- Difference (correct - incorrect): {diff:.4f}")
                
                # Compare entropy progression
                if (correct_stats['entropy']['quartiles'] and incorrect_stats['entropy']['quartiles']):
                    print("- Entropy progression through generation:")
                    
                    for i in range(min(len(correct_stats['entropy']['quartiles']), 
                                       len(incorrect_stats['entropy']['quartiles']))):
                        c_q = correct_stats['entropy']['quartiles'][i]
                        i_q = incorrect_stats['entropy']['quartiles'][i]
                        q_diff = c_q - i_q
                        
                        print(f"  - Q{i+1}: Correct: {c_q:.4f}, Incorrect: {i_q:.4f}, Diff: {q_diff:.4f}")
                
                # Compare transitions
                common_transitions = set(correct_stats['transitions'].keys()) & set(incorrect_stats['transitions'].keys())
                
                if common_transitions:
                    print("- Entropy changes around thought transitions:")
                    
                    for phrase in common_transitions:
                        c_stats = correct_stats['transitions'][phrase]
                        i_stats = incorrect_stats['transitions'][phrase]
                        
                        c_change = c_stats['entropy_change']
                        i_change = i_stats['entropy_change']
                        
                        print(f"  - {phrase}:")
                        print(f"    - Correct: {c_stats['before_mean']:.4f} → {c_stats['after_mean']:.4f} (Δ {c_change:.4f})")
                        print(f"    - Incorrect: {i_stats['before_mean']:.4f} → {i_stats['after_mean']:.4f} (Δ {i_change:.4f})")
                        print(f"    - Difference in entropy change: {c_change - i_change:.4f}")
    
    print("\n=== Incorrect Problems ===")
    for r in results:
        if not r['is_correct']:
            print(f"Problem {r['index']}:")
            print(f"Expected: {r['correct_answer']}")
            print("Predicted answers across attempts:", [
                attempt['predicted_answer'] for attempt in r['attempts']
            ])
            print("---")

def save_raw_response(filename: str, problem_id: int, response_data: Dict):
    """Save raw response data (including logprobs) to a separate file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create a timestamped ID for this response
    timestamp = int(time.time())
    response_id = f"{problem_id}_{timestamp}"
    
    # Create or update the raw responses file
    try:
        with open(filename, 'r') as f:
            raw_responses = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raw_responses = {}
    
    # Add this response to the collection
    raw_responses[response_id] = response_data
    
    # Save the updated collection
    with open(filename, 'w') as f:
        json.dump(raw_responses, f)
    
    return response_id

def main(model: str, n_attempts: int, year: int = 2024, analyze_thoughts: bool = False, analyze_logits: bool = False, test_time_compute: bool = False, approach_name: str = None, extra_body: dict = None):
    """Main evaluation function that handles gaps in processed indexes."""
    os.makedirs("results", exist_ok=True)

    # Create suffix based on analysis flags
    suffix_parts = []
    if year != 2024:
        suffix_parts.append(f"aime{year}")
    if analyze_thoughts:
        suffix_parts.append("thought_analysis")
    if analyze_logits:
        suffix_parts.append("logit_analysis")
    if approach_name:
        suffix_parts.append(approach_name)

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    results_file = f"results/evaluation_results_{model.replace('/', '_')}_pass_at_{n_attempts}{suffix}.json"

    dataset = load_dataset_by_year(year)
    existing_results = load_existing_results(results_file)
    
    # Create a set of already processed indexes for efficient lookup
    processed_indexes = {result['index'] for result in existing_results}
    
    for _, item in enumerate(tqdm(dataset, desc="Evaluating problems")):
        id = int(item['id'])
        # Skip if this index has already been processed
        if id in processed_indexes:
            continue
            
        problem_text = item['problem']
        correct_answer = int(item['answer'])

        print(f"\n🔬 Processing Problem {id}: {problem_text[:100]}...")
        print(f"   Expected answer: {correct_answer}")
        if extra_body and 'optillm_approach' in extra_body:
            print(f"   Using approach: {extra_body['optillm_approach']}")

        # Make n attempts for each problem
        attempts = make_n_attempts(problem_text, model, n_attempts, analyze_thoughts, analyze_logits, extra_body)
        is_correct, first_correct = evaluate_pass_at_n(attempts, correct_answer)

        # Report result
        predicted_answers = [attempt.get('predicted_answer') for attempt in attempts]
        print(f"   Predicted: {predicted_answers}")
        if is_correct:
            print("   ✅ CORRECT!")
        else:
            print("   ❌ Incorrect")

        result = {
            "index": id,
            "problem": problem_text,
            "attempts": attempts,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "first_correct_attempt": first_correct
        }
        save_result(results_file, result)
    
    final_results = load_existing_results(results_file)
    analyze_results(final_results, n_attempts, analyze_thoughts, analyze_logits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on AIME problems")
    parser.add_argument("--model", type=str, required=True, help="OpenAI model to use (e.g., gpt-4, gpt-3.5-turbo)")
    parser.add_argument("--n", type=int, default=1, help="Number of attempts per problem (for pass@n evaluation)")
    parser.add_argument("--year", type=int, default=2024, choices=[2024, 2025], help="AIME year to evaluate (2024 or 2025)")
    parser.add_argument("--approach", type=str, help="OptILLM approach to use (e.g., mars, moa, bon)")
    parser.add_argument("--analyze-thoughts", action="store_true", help="Analyze thinking patterns in responses")
    parser.add_argument("--analyze-logits", action="store_true", help="Analyze token probability distributions")
    parser.add_argument("--test-time-compute", action="store_true", help="Evaluate test-time compute scaling approaches")
    args = parser.parse_args()
    
    if args.test_time_compute:
        # Define test-time compute approaches with same config as eval_optillmbench.py
        TEST_TIME_COMPUTE_APPROACHES = [
            # Baseline
            ("none", "Baseline without any optimization", {}),
            
            # Sequential test-time compute using thinkdeeper with controlled thinking budgets
            ("thinkdeeper_2k", "ThinkDeeper with 2K thinking tokens", {
                "decoding": "thinkdeeper",
                "min_thinking_tokens": 2048,
                "max_thinking_tokens": 2560,  # min + 512 for flexibility
                "max_tokens": 3072  # Total budget: max_thinking_tokens + 512
            }),
            ("thinkdeeper_4k", "ThinkDeeper with 4K thinking tokens", {
                "decoding": "thinkdeeper", 
                "min_thinking_tokens": 4096,
                "max_thinking_tokens": 4608,  # min + 512 for flexibility
                "max_tokens": 5120  # Total budget: max_thinking_tokens + 512
            }),
            ("thinkdeeper_8k", "ThinkDeeper with 8K thinking tokens", {
                "decoding": "thinkdeeper",
                "min_thinking_tokens": 8192,
                "max_thinking_tokens": 8704,  # min + 512 for flexibility
                "max_tokens": 9216  # Total budget: max_thinking_tokens + 512
            }),
            
            # Parallel test-time compute using majority voting with different k values
            ("majority_voting_3", "Majority Voting with k=3", {"k": 3}),
            ("majority_voting_6", "Majority Voting with k=6", {"k": 6}),
            ("majority_voting_9", "Majority Voting with k=9", {"k": 9}),
        ]
        
        # Run evaluation for each approach
        for approach_slug, approach_name, extra_body in TEST_TIME_COMPUTE_APPROACHES:
            print(f"\n{'=' * 80}")
            print(f"Evaluating: {approach_name}")
            print(f"Model: {args.model}")
            print(f"Approach: {approach_slug}")
            print(f"Extra body: {extra_body}")
            print(f"{'=' * 80}\n")
            
            main(args.model, args.n, args.year, args.analyze_thoughts, args.analyze_logits,
                 test_time_compute=True, approach_name=approach_slug, extra_body=extra_body)
    else:
        # Handle approach parameter - only set extra_body if approach is not "none"
        extra_body = {"optillm_approach": args.approach} if args.approach and args.approach != "none" else None
        approach_name = args.approach if args.approach and args.approach != "none" else None

        main(args.model, args.n, args.year, args.analyze_thoughts, args.analyze_logits,
             approach_name=approach_name, extra_body=extra_body)