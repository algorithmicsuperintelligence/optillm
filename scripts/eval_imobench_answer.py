"""
Evaluation script for IMO-Bench AnswerBench dataset (400 problems)
Tests model performance on short-answer mathematical problems across 4 categories
"""

import argparse
import json
import os
import logging
import time
import re
import pandas as pd
import requests
from typing import List, Dict
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

# Add sys path to import optillm modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optillm.utils.answer_extraction import extract_answer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset URL
ANSWERBENCH_URL = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench.csv"

SYSTEM_PROMPT = '''You are solving IMO-Bench mathematical problems across algebra, combinatorics, geometry, and number theory.

Key requirements:
1. **Clear reasoning**: Show your work step-by-step
2. **Mathematical rigor**: Justify each step logically
3. **Final answer**: Clearly state your final answer in \\boxed{} format

For different problem types:
- Algebra: Handle functional equations, polynomials, inequalities
- Combinatorics: Use counting techniques, pigeonhole principle, extremal arguments
- Geometry: Apply coordinate systems, trigonometry, or synthetic methods
- Number Theory: Use divisibility, modular arithmetic, prime factorization

Always conclude with your final answer in \\boxed{your_answer} format.'''


def download_answerbench() -> pd.DataFrame:
    """
    Download and parse the AnswerBench CSV dataset
    """
    logger.info("Downloading AnswerBench dataset...")
    try:
        response = requests.get(ANSWERBENCH_URL, timeout=30)
        response.raise_for_status()

        # Save to temp file and load with pandas
        temp_file = "/tmp/answerbench.csv"
        with open(temp_file, 'wb') as f:
            f.write(response.content)

        df = pd.read_csv(temp_file)
        logger.info(f"Loaded {len(df)} problems from AnswerBench")
        return df

    except Exception as e:
        logger.error(f"Error downloading AnswerBench: {e}")
        raise


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison
    """
    if answer is None:
        return ""

    # Convert to string and lowercase
    answer = str(answer).strip().lower()

    # Remove extra whitespace
    answer = re.sub(r'\s+', ' ', answer)

    # Remove common LaTeX formatting
    answer = answer.replace('\\', '')
    answer = answer.replace('$', '')
    answer = answer.replace('{', '').replace('}', '')

    return answer


def compare_answers(predicted: str, ground_truth: str) -> bool:
    """
    Compare predicted answer with ground truth
    Uses both exact match and semantic equivalence
    """
    if not predicted or not ground_truth:
        return False

    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    truth_norm = normalize_answer(ground_truth)

    # Exact match after normalization
    if pred_norm == truth_norm:
        return True

    # Check if one contains the other (for cases like "4" in "c = 4")
    if pred_norm in truth_norm or truth_norm in pred_norm:
        return True

    # Try numeric comparison if possible
    try:
        pred_num = float(re.sub(r'[^0-9.-]', '', predicted))
        truth_num = float(re.sub(r'[^0-9.-]', '', ground_truth))
        if abs(pred_num - truth_num) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    return False


def extract_answer_from_solution(solution: str, problem_id: str = None) -> str:
    """
    Extract the final answer from a solution
    """
    if not solution:
        return None

    # Try unified answer extraction first
    try:
        extracted = extract_answer(solution, problem_type="math")
        if extracted:
            return str(extracted)
    except Exception as e:
        logger.debug(f"Unified extraction failed: {e}")

    # Look for boxed answers
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, solution)
    if boxed_matches:
        return boxed_matches[-1].strip()

    # Look for "final answer" or "answer:" sections
    answer_patterns = [
        r'final answer[:\s]*([^\n]+)',
        r'answer[:\s]*([^\n]+)',
        r'therefore[:\s]*([^\n]+)',
        r'thus[:\s]*([^\n]+)'
    ]

    solution_lower = solution.lower()
    for pattern in answer_patterns:
        matches = re.findall(pattern, solution_lower)
        if matches:
            return matches[-1].strip()

    return None


def get_llm_response(problem: str, model: str, client: OpenAI, extra_body: dict = None, timeout: int = 300) -> Dict:
    """
    Get response from the LLM for a mathematical problem
    """
    try:
        kwargs = {}
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = client.with_options(timeout=timeout).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem}
            ],
            max_tokens=16000,
            temperature=0.1,
            **kwargs
        )

        solution_text = response.choices[0].message.content.strip()
        reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)
        total_tokens = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0

        return {
            "solution": solution_text,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "success": True
        }

    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return {
            "solution": f"Error: {str(e)}",
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "success": False
        }


def save_result(filename: str, result: Dict):
    """Save a single result to the results file with incremental updates"""
    results = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            results = []

    results.append(result)

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


def load_existing_results(filename: str) -> List[Dict]:
    """Load existing results from file if it exists"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def analyze_results(results: List[Dict]):
    """Analyze and print comprehensive statistics"""
    if not results:
        print("No results to analyze")
        return

    total_problems = len(results)
    correct = sum(1 for r in results if r.get('is_correct', False))

    print("\n" + "="*80)
    print("IMO-Bench AnswerBench Evaluation Results")
    print("="*80)
    print(f"Total problems: {total_problems}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total_problems*100:.2f}%")

    # Category breakdown
    categories = {}
    for r in results:
        cat = r.get('category', 'Unknown')
        if cat not in categories:
            categories[cat] = {'total': 0, 'correct': 0}
        categories[cat]['total'] += 1
        if r.get('is_correct', False):
            categories[cat]['correct'] += 1

    print("\nPerformance by Category:")
    print("-" * 60)
    for cat, stats in sorted(categories.items()):
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{cat:20s}: {stats['correct']:3d}/{stats['total']:3d} ({acc:5.1f}%)")

    # Difficulty breakdown if available
    difficulties = {}
    for r in results:
        diff = r.get('difficulty', 'Unknown')
        if diff and diff != 'Unknown':
            if diff not in difficulties:
                difficulties[diff] = {'total': 0, 'correct': 0}
            difficulties[diff]['total'] += 1
            if r.get('is_correct', False):
                difficulties[diff]['correct'] += 1

    if difficulties:
        print("\nPerformance by Difficulty:")
        print("-" * 60)
        for diff, stats in sorted(difficulties.items()):
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"{diff:20s}: {stats['correct']:3d}/{stats['total']:3d} ({acc:5.1f}%)")

    # Token statistics
    total_tokens = sum(r['response'].get('total_tokens', 0) for r in results)
    reasoning_tokens = sum(r['response'].get('reasoning_tokens', 0) for r in results)

    print("\nToken Statistics:")
    print("-" * 60)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Reasoning tokens: {reasoning_tokens:,}")
    print(f"Avg tokens per problem: {total_tokens/total_problems:.0f}")

    # Time statistics
    total_time = sum(r.get('solve_time_seconds', 0) for r in results)
    print(f"\nTotal solve time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Avg time per problem: {total_time/total_problems:.1f}s")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on IMO-Bench AnswerBench")
    parser.add_argument("--model", type=str, required=True,
                       help="Model to use (e.g., google/gemini-2.5-flash-preview-09-2025)")
    parser.add_argument("--base-url", type=str, default="http://localhost:8001/v1",
                       help="Base URL for OptiLLM server")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout in seconds for each problem")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of problems to evaluate (for testing)")
    parser.add_argument("--categories", type=str, default=None,
                       help="Comma-separated list of categories to evaluate (e.g., 'Algebra,Geometry')")

    args = parser.parse_args()

    # Initialize OpenAI client
    client = OpenAI(api_key="optillm", base_url=args.base_url)

    # Setup results directory
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine if using MARS approach
    is_mars = args.model.startswith("mars-")
    approach_name = "mars" if is_mars else "baseline"
    model_name = args.model.replace("mars-", "") if is_mars else args.model

    results_file = f"results/imobench_answer_{approach_name}_{model_name.replace('/', '_')}_{timestamp}.json"

    # Download dataset
    df = download_answerbench()

    # Filter by categories if specified
    if args.categories:
        selected_cats = [c.strip() for c in args.categories.split(',')]
        df = df[df['Category'].isin(selected_cats)]
        print(f"Filtered to categories: {selected_cats}")

    # Limit problems if specified
    if args.limit:
        df = df.head(args.limit)

    print(f"\nEvaluating {len(df)} AnswerBench problems")
    print(f"Model: {args.model}")
    print(f"Approach: {approach_name}")
    print(f"Results will be saved to: {results_file}\n")

    # Evaluate each problem
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Solving problems"):
        problem_id = row.get('Problem ID', f'problem_{idx}')
        problem_text = row['Problem']
        ground_truth = row['Short Answer']
        category = row.get('Category', 'Unknown')
        subcategory = row.get('Subcategory', '')
        difficulty = row.get('Difficulty', '')

        logger.info(f"Evaluating {problem_id}: {category}")

        start_time = time.time()

        # Get LLM response
        response = get_llm_response(
            problem_text,
            args.model,
            client,
            extra_body=None,  # Model prefix handles MARS
            timeout=args.timeout
        )

        solve_time = time.time() - start_time

        # Extract answer
        extracted_answer = extract_answer_from_solution(response['solution'], problem_id)

        # Compare with ground truth
        is_correct = compare_answers(extracted_answer, ground_truth)

        # Compile result
        result = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "approach": approach_name,
            "problem_id": problem_id,
            "category": category,
            "subcategory": subcategory,
            "difficulty": difficulty,
            "problem": problem_text,
            "ground_truth": ground_truth,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
            "response": response,
            "solve_time_seconds": solve_time
        }

        # Save result immediately
        save_result(results_file, result)

        status = "✓" if is_correct else "✗"
        logger.info(f"{status} {problem_id} - Answer: {extracted_answer}")

    # Load and analyze all results
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)

    results = load_existing_results(results_file)
    analyze_results(results)

    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
