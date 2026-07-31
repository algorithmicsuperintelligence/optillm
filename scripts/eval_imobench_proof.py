"""
Evaluation script for IMO-Bench ProofBench dataset (60 problems)
Tests model performance on rigorous mathematical proof construction
Uses IMO25-style verification system for grading
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset URL
PROOFBENCH_URL = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/proofbench.csv"

SYSTEM_PROMPT = '''You are solving IMO-level mathematical proof problems. These require rigorous, complete proofs.

Key requirements:
1. **Complete proofs**: Provide rigorous, step-by-step mathematical proofs
2. **Mathematical rigor**: Every step must be logically justified
3. **Clear structure**: Organize your solution with clear logical flow
4. **Proper notation**: Use correct mathematical notation and formatting
5. **Verification**: Double-check your reasoning and conclusions

For different problem types:
- Functional equations: Consider injectivity, surjectivity, and special values
- Geometry: Use coordinate systems, trigonometry, or synthetic methods as appropriate
- Number theory: Apply divisibility, modular arithmetic, and prime factorization
- Combinatorics: Use counting techniques, pigeonhole principle, and extremal arguments
- Algebra: Handle polynomials, inequalities, and algebraic structures

Provide a complete, rigorous proof that addresses all aspects of the problem.'''

VERIFICATION_PROMPT = """You are an expert mathematician and IMO grader. Your task is to rigorously verify this mathematical solution.

**Grading Scale (0-7 points):**
- **7 points**: Complete, rigorous, correct proof
- **6 points**: Correct approach, minor gaps or notation issues
- **5 points**: Mostly correct, some gaps in rigor
- **3-4 points**: Significant progress, partial solution
- **1-2 points**: Some correct ideas, incomplete
- **0 points**: No progress or completely wrong

**Verification Instructions:**
1. Check logical correctness of each step
2. Verify mathematical rigor and completeness
3. Identify any critical errors or gaps
4. Assess proof structure and clarity

**Problem:**
{problem}

**Solution to verify:**
{solution}

Provide your assessment in the following format:

**SCORE:** [0-7]
**VERDICT:** [Correct/Partially Correct/Incorrect]
**REASONING:** [Detailed explanation of your assessment]
**CRITICAL ERRORS:** [List any critical errors found, or "None"]
**GAPS:** [List any gaps in rigor, or "None"]"""


def download_proofbench() -> pd.DataFrame:
    """
    Download and parse the ProofBench CSV dataset
    """
    logger.info("Downloading ProofBench dataset...")
    try:
        response = requests.get(PROOFBENCH_URL, timeout=30)
        response.raise_for_status()

        # Save to temp file and load with pandas
        temp_file = "/tmp/proofbench.csv"
        with open(temp_file, 'wb') as f:
            f.write(response.content)

        df = pd.read_csv(temp_file)
        logger.info(f"Loaded {len(df)} problems from ProofBench")
        return df

    except Exception as e:
        logger.error(f"Error downloading ProofBench: {e}")
        raise


def verify_proof(problem: str, solution: str, grading_guidelines: str, model: str, client: OpenAI) -> Dict:
    """
    Verify a proof using IMO25-style two-stage verification
    Returns score on 0-7 scale and detailed assessment
    """
    try:
        # Format verification prompt
        verification_text = VERIFICATION_PROMPT.format(
            problem=problem,
            solution=solution
        )

        # Add grading guidelines if available
        if grading_guidelines and pd.notna(grading_guidelines):
            verification_text += f"\n\n**Grading Guidelines:**\n{grading_guidelines}"

        # Get verification response
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert IMO grader. Provide rigorous assessment."},
                {"role": "user", "content": verification_text}
            ],
            max_tokens=4000,
            temperature=0.1
        )

        verification_response = response.choices[0].message.content.strip()

        # Extract score
        score_match = re.search(r'\*\*SCORE:\*\*\s*(\d+)', verification_response)
        score = int(score_match.group(1)) if score_match else 0

        # Extract verdict
        verdict_match = re.search(r'\*\*VERDICT:\*\*\s*([^\n]+)', verification_response)
        verdict = verdict_match.group(1).strip() if verdict_match else "Unknown"

        # Determine if correct (7 points = full marks)
        is_correct = (score == 7)

        # Check for critical errors
        errors_match = re.search(r'\*\*CRITICAL ERRORS:\*\*\s*([^\n]+)', verification_response)
        has_critical_errors = errors_match and "None" not in errors_match.group(1) if errors_match else False

        return {
            "score": score,
            "verdict": verdict,
            "is_correct": is_correct,
            "has_critical_errors": has_critical_errors,
            "verification_response": verification_response,
            "success": True
        }

    except Exception as e:
        logger.error(f"Error in proof verification: {e}")
        return {
            "score": 0,
            "verdict": "Error",
            "is_correct": False,
            "has_critical_errors": True,
            "verification_response": f"Verification error: {str(e)}",
            "success": False
        }


def extract_solution_quality(solution: str) -> Dict:
    """
    Analyze the quality of a mathematical proof
    """
    analysis = {
        "has_proof_structure": False,
        "uses_mathematical_notation": False,
        "has_logical_steps": False,
        "addresses_cases": False,
        "has_conclusion": False,
        "length_score": 0
    }

    if not solution:
        return analysis

    solution_lower = solution.lower()

    # Check for proof structure
    proof_keywords = ["proof:", "solution:", "we prove", "to show", "suppose", "assume", "let", "consider"]
    if any(keyword in solution_lower for keyword in proof_keywords):
        analysis["has_proof_structure"] = True

    # Check for mathematical notation
    math_patterns = [r'\$.*\$', r'\\[a-zA-Z]+', r'\\geq', r'\\leq', r'\\in', r'\\sum', r'\\prod']
    if any(re.search(pattern, solution) for pattern in math_patterns):
        analysis["uses_mathematical_notation"] = True

    # Check for logical flow
    logical_words = ["therefore", "thus", "hence", "consequently", "since", "because", "implies"]
    logical_count = sum(1 for word in logical_words if word in solution_lower)
    if logical_count >= 3:
        analysis["has_logical_steps"] = True

    # Check for case analysis
    case_words = ["case", "if", "when", "suppose"]
    case_count = sum(1 for word in case_words if word in solution_lower)
    if case_count >= 2:
        analysis["addresses_cases"] = True

    # Check for conclusion
    conclusion_words = ["therefore", "thus", "q.e.d", "qed", "proven", "concluded"]
    if any(word in solution_lower for word in conclusion_words):
        analysis["has_conclusion"] = True

    # Length score (normalized)
    analysis["length_score"] = min(len(solution) / 2000, 1.0)

    return analysis


def get_llm_response(problem: str, model: str, client: OpenAI, extra_body: dict = None, timeout: int = 600) -> Dict:
    """
    Get response from the LLM for a proof problem
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
            max_tokens=64000,  # Extended for complex proofs
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
    """Save a single result with incremental updates"""
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
    """Load existing results from file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def calculate_subset_scores(results: List[Dict]) -> Dict:
    """
    Calculate full credit scores for various subsets (Novel, IMO 2024, USAMO 2025)
    Returns dictionary with subset names and their (solved, total, percentage) tuples
    """
    subsets = {
        'Novel': {'full': 0, 'total': 0},
        'IMO 2024': {'full': 0, 'total': 0},
        'USAMO 2025': {'full': 0, 'total': 0}
    }

    for r in results:
        source = r.get('source', '')
        score = r['verification'].get('score', 0)
        is_full = (score == 7)

        # Categorize by source
        if 'Novel Problem' in source:
            subsets['Novel']['total'] += 1
            if is_full:
                subsets['Novel']['full'] += 1
        elif 'IMO 2024' in source:
            subsets['IMO 2024']['total'] += 1
            if is_full:
                subsets['IMO 2024']['full'] += 1
        elif 'USAMO 2025' in source:
            subsets['USAMO 2025']['total'] += 1
            if is_full:
                subsets['USAMO 2025']['full'] += 1

    # Calculate percentages
    subset_stats = {}
    for name, counts in subsets.items():
        total = counts['total']
        full = counts['full']
        pct = (full / total * 100) if total > 0 else 0
        subset_stats[name] = (full, total, pct)

    return subset_stats


def analyze_results(results: List[Dict]):
    """Analyze and print comprehensive statistics with full credit prioritized"""
    if not results:
        print("No results to analyze")
        return

    total_problems = len(results)
    full_marks = sum(1 for r in results if r['verification'].get('score', 0) == 7)
    partial_credit = sum(1 for r in results if 1 <= r['verification'].get('score', 0) <= 6)
    no_credit = total_problems - full_marks - partial_credit

    total_score = sum(r['verification'].get('score', 0) for r in results)
    avg_score = total_score / total_problems

    print("\n" + "="*80)
    print("IMO-Bench ProofBench Evaluation Results")
    print("="*80)

    # ========================================================================
    # SECTION 1: FULL CREDIT SCORES (PRIMARY METRIC)
    # ========================================================================
    print("\n" + "="*80)
    print("FULL CREDIT SCORES (7/7 = Solved) - PRIMARY METRIC")
    print("="*80)
    print(f"\nOverall: {full_marks}/{total_problems} = {full_marks/total_problems*100:.1f}%")

    # Basic vs Advanced breakdown (full credit only)
    basic_full = sum(1 for r in results if 'Basic' in r.get('problem_id', '') and r['verification'].get('score', 0) == 7)
    basic_total = sum(1 for r in results if 'Basic' in r.get('problem_id', ''))
    adv_full = sum(1 for r in results if 'Advanced' in r.get('problem_id', '') and r['verification'].get('score', 0) == 7)
    adv_total = sum(1 for r in results if 'Advanced' in r.get('problem_id', ''))

    print(f"\nBasic problems:    {basic_full}/{basic_total} = {basic_full/basic_total*100 if basic_total > 0 else 0:.1f}%")
    print(f"Advanced problems: {adv_full}/{adv_total} = {adv_full/adv_total*100 if adv_total > 0 else 0:.1f}%")

    # ========================================================================
    # SECTION 2: SUBSET BREAKDOWN (Novel, IMO 2024, USAMO 2025)
    # ========================================================================
    subset_stats = calculate_subset_scores(results)

    if any(total > 0 for _, total, _ in subset_stats.values()):
        print("\n" + "-"*80)
        print("Subset Breakdown (Full Credit Only):")
        print("-"*80)
        for name in ['Novel', 'IMO 2024', 'USAMO 2025']:
            full, total, pct = subset_stats[name]
            if total > 0:
                print(f"{name:15s}: {full}/{total} = {pct:.1f}%")

    # ========================================================================
    # SECTION 3: DETAILED ANALYSIS (Average Scores and Distributions)
    # ========================================================================
    print("\n" + "="*80)
    print("DETAILED ANALYSIS (Average Scores)")
    print("="*80)
    print(f"\nAverage score: {avg_score:.2f}/7 ({avg_score/7*100:.1f}%)")
    print(f"Full credit (7/7): {full_marks} ({full_marks/total_problems*100:.1f}%)")
    print(f"Partial credit (1-6): {partial_credit} ({partial_credit/total_problems*100:.1f}%)")
    print(f"No credit (0): {no_credit} ({no_credit/total_problems*100:.1f}%)")

    # Basic vs Advanced (average scores)
    basic_scores = [r['verification'].get('score', 0) for r in results if 'Basic' in r.get('problem_id', '')]
    adv_scores = [r['verification'].get('score', 0) for r in results if 'Advanced' in r.get('problem_id', '')]

    if basic_scores or adv_scores:
        print("\n" + "-"*80)
        print("Basic vs Advanced (Average Scores):")
        print("-"*80)
        if basic_scores:
            basic_avg = sum(basic_scores) / len(basic_scores)
            print(f"Basic ({len(basic_scores)}):    {basic_avg:.2f}/7 ({basic_avg/7*100:.1f}%)")
        if adv_scores:
            adv_avg = sum(adv_scores) / len(adv_scores)
            print(f"Advanced ({len(adv_scores)}): {adv_avg:.2f}/7 ({adv_avg/7*100:.1f}%)")

    # Category breakdown (average scores)
    categories = {}
    for r in results:
        cat = r.get('category', 'Unknown')
        if cat not in categories:
            categories[cat] = {'total': 0, 'scores': [], 'full': 0}
        categories[cat]['total'] += 1
        score = r['verification'].get('score', 0)
        categories[cat]['scores'].append(score)
        if score == 7:
            categories[cat]['full'] += 1

    if categories:
        print("\n" + "-"*80)
        print("Performance by Category:")
        print("-"*80)
        for cat, stats in sorted(categories.items()):
            avg = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
            full = stats['full']
            total = stats['total']
            print(f"{cat:20s}: Avg {avg:.2f}/7 ({avg/7*100:.1f}%) | Solved {full}/{total} ({full/total*100:.1f}%)")

    # Level breakdown (average scores)
    levels = {}
    for r in results:
        level = r.get('level', 'Unknown')
        if level not in levels:
            levels[level] = {'total': 0, 'scores': [], 'full': 0}
        levels[level]['total'] += 1
        score = r['verification'].get('score', 0)
        levels[level]['scores'].append(score)
        if score == 7:
            levels[level]['full'] += 1

    if levels:
        print("\n" + "-"*80)
        print("Performance by Level:")
        print("-"*80)
        for level, stats in sorted(levels.items()):
            avg = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
            full = stats['full']
            total = stats['total']
            print(f"{level:20s}: Avg {avg:.2f}/7 ({avg/7*100:.1f}%) | Solved {full}/{total} ({full/total*100:.1f}%)")

    # Token statistics
    try:
        total_tokens = sum(r['response'].get('total_tokens', 0) for r in results)
        reasoning_tokens = sum(r['response'].get('reasoning_tokens', 0) for r in results)

        print("\n" + "-"*80)
        print("Token Statistics:")
        print("-"*80)
        print(f"Total tokens: {total_tokens:,}")
        print(f"Reasoning tokens: {reasoning_tokens:,}")
        print(f"Avg tokens per problem: {total_tokens/total_problems:.0f}")
    except (KeyError, TypeError):
        pass  # Skip token stats if data not available

    # Time statistics
    total_time = sum(r.get('solve_time_seconds', 0) for r in results)
    print("\n" + "-"*80)
    print(f"Total solve time: {total_time:.1f}s ({total_time/60:.1f} minutes, {total_time/3600:.1f} hours)")
    print(f"Avg time per problem: {total_time/total_problems:.1f}s")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on IMO-Bench ProofBench")
    parser.add_argument("--model", type=str, required=True,
                       help="Model to use (e.g., google/gemini-2.5-flash-preview-09-2025 or mars-...)")
    parser.add_argument("--base-url", type=str, default="http://localhost:8001/v1",
                       help="Base URL for OptiLLM server")
    parser.add_argument("--verifier-model", type=str, default=None,
                       help="Model to use for verification (defaults to same as solver)")
    parser.add_argument("--timeout", type=int, default=600,
                       help="Timeout in seconds for each problem")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of problems (for testing)")
    parser.add_argument("--subset", type=str, default=None,
                       help="Evaluate only 'basic' or 'advanced' subset")

    args = parser.parse_args()

    # Initialize OpenAI client
    client = OpenAI(api_key="optillm", base_url=args.base_url)

    # Verifier model defaults to solver model
    verifier_model = args.verifier_model or args.model.replace("mars-", "")

    # Setup results directory
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine if using MARS
    is_mars = args.model.startswith("mars-")
    approach_name = "mars" if is_mars else "baseline"
    model_name = args.model.replace("mars-", "") if is_mars else args.model

    results_file = f"results/imobench_proof_{approach_name}_{model_name.replace('/', '_')}_{timestamp}.json"

    # Download dataset
    df = download_proofbench()

    # Filter by subset if specified
    if args.subset:
        if args.subset.lower() == 'basic':
            df = df[df['Level'].str.contains('Basic', case=False, na=False)]
        elif args.subset.lower() == 'advanced':
            df = df[df['Level'].str.contains('Advanced', case=False, na=False)]
        print(f"Filtered to {args.subset} subset")

    # Limit problems if specified
    if args.limit:
        df = df.head(args.limit)

    print(f"\nEvaluating {len(df)} ProofBench problems")
    print(f"Model: {args.model}")
    print(f"Approach: {approach_name}")
    print(f"Verifier: {verifier_model}")
    if is_mars:
        print("MARS Config: use_thinking_tags=False, answer_extraction_mode='none'")
    print(f"Results will be saved to: {results_file}\n")

    # Evaluate each problem
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Solving proof problems"):
        problem_id = row.get('Problem ID', f'problem_{idx}')
        problem_text = row['Problem']
        reference_solution = row.get('Solution', '')
        grading_guidelines = row.get('Grading guidelines', '')
        category = row.get('Category', 'Unknown')
        level = row.get('Level', 'Unknown')
        source = row.get('Source', 'Unknown')

        logger.info(f"Evaluating {problem_id}: {category} ({level})")

        start_time = time.time()

        # Get LLM response (model prefix handles MARS configuration automatically)
        response = get_llm_response(
            problem_text,
            args.model,
            client,
            extra_body=None,
            timeout=args.timeout
        )

        solve_time = time.time() - start_time

        # Verify the proof
        verification = verify_proof(
            problem_text,
            response['solution'],
            grading_guidelines,
            verifier_model,
            client
        )

        # Analyze solution quality
        quality = extract_solution_quality(response['solution'])

        # Compile result
        result = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "approach": approach_name,
            "verifier_model": verifier_model,
            "problem_id": problem_id,
            "category": category,
            "level": level,
            "source": source,
            "problem": problem_text,
            "reference_solution": reference_solution,
            "grading_guidelines": grading_guidelines,
            "response": response,
            "verification": verification,
            "quality": quality,
            "solve_time_seconds": solve_time
        }

        # Save result immediately
        save_result(results_file, result)

        score = verification.get('score', 0)
        status = "✓ SOLVED" if score == 7 else f"✗ {score}/7"
        logger.info(f"{status} {problem_id}")

    # Load and analyze all results
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)

    results = load_existing_results(results_file)
    analyze_results(results)

    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
