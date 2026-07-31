"""
Reference solutions and verification for IMO 2025 problems
Contains actual problems from the official contest and exact answers from Google DeepMind's solutions
"""

import re
from typing import Dict, Any, Optional

# Actual IMO 2025 problems from the official contest
IMO_2025_PROBLEMS = [
    {
        "id": 1,
        "problem": """A line in the plane is called *sunny* if it is not parallel to any of the $x$-axis, the $y$-axis, and the line $x+y=0$.

Let $n\\ge3$ be a given integer. Determine all nonnegative integers $k$ such that there exist $n$ distinct lines in the plane satisfying both the following:
*   for all positive integers $a$ and $b$ with $a+b\\le n+1$, the point $(a,b)$ is on at least one of the lines; and
*   exactly $k$ of the lines are sunny.""",
        "type": "combinatorial_geometry",
        "difficulty": "medium",
        "expected_answer": "{0, 1, 3}",
        "answer_type": "set",
        "key_insights": [
            "reduction_principle",
            "structural_lemma",
            "c_k_analysis",
            "sunny_line_covering"
        ],
        "solution_approach": "reduction_to_specific_case"
    },
    {
        "id": 2,
        "problem": """Let $\\Omega$ and $\\Gamma$ be circles with centers $M$ and $N$, respectively, such that the radius of $\\Omega$ is less than the radius of $\\Gamma$. Suppose circles $\\Omega$ and $\\Gamma$ intersect at two distinct points $A$ and $B$. Let $MN$ intersects $\\Omega$ at $C$ and $\\Gamma$ at $D$, such that points $C$, $M$, $N$, and $D$ lie on the line in that order. Let $P$ be the circumcenter of triangle $ACD$. Line $AP$ intersects $\\Omega$ again at $E\\neq A$. Line $AP$ intersects $\\Gamma$ again at $F\\neq A$. Let $H$ be the orthocenter of triangle $PMN$.

Prove that the line through $H$ parallel to $AP$ is tangent to the circumcircle of triangle $BEF$.

(The orthocenter of a triangle is the point of intersection of its altitudes.)""",
        "type": "geometry",
        "difficulty": "very_hard",
        "expected_answer": "Complete geometric proof",
        "answer_type": "proof",
        "key_insights": [
            "excenter_identification",
            "auxiliary_point_v",
            "orthocenter_tangency",
            "circumcircle_properties"
        ],
        "solution_approach": "synthetic_geometry_with_coordinates"
    },
    {
        "id": 3,
        "problem": """Let $\\mathbb N$ denote the set of positive integers. A function $f:\\mathbb N\\to\\mathbb N$ is said to be bonza if $f(a)$ divides $b^a-f(b)^{f(a)}$ for all positive integers $a$ and $b$.

Determine the smallest real constant $c$ such that $f(n)\\le cn$ for all bonza functions $f$ and all positive integers $n$.""",
        "type": "functional_equation",
        "difficulty": "very_hard",
        "expected_answer": "4",
        "answer_type": "number",
        "key_insights": [
            "classification_lemma",
            "set_s_analysis",
            "upper_bound_proof",
            "construction_example"
        ],
        "solution_approach": "case_analysis_and_construction"
    },
    {
        "id": 4,
        "problem": """A proper divisor of a positive integer $N$ is a positive divisor of $N$ other than $N$ itself.

The infinite sequence $a_1,a_2,\\ldots$ consists of positive integers, each of which has at least three proper divisors. For each $n\\ge1$, the integer $a_{n+1}$ is the sum of three largest proper divisors of $a_n$.

Determine all possible values of $a_1$.""",
        "type": "number_theory",
        "difficulty": "very_hard",
        "expected_answer": "6J·12^K where gcd(J,10)=1",
        "answer_type": "formula",
        "key_insights": [
            "regime_analysis",
            "evolution_dynamics",
            "divisibility_constraints",
            "fixed_point_analysis"
        ],
        "solution_approach": "sequence_analysis_with_regimes"
    },
    {
        "id": 5,
        "problem": """Alice and Bazza are playing the inekoalaty game, a two-player game whose rules depend on a positive real number $\\lambda$ which is known to both players. On the $n$th turn of the game (starting with $n=1$) the following happens:
*   If $n$ is odd, Alice chooses a nonnegative real number $x_n$ such that
$$x_1+x_2+\\cdots+x_n\\le\\lambda n.$$
*   If $n$ is even, Bazza chooses a nonnegative real number $x_n$ such that
$$x_1^2+x_2^2+\\cdots+x_n^2\\le n.$$

If a player cannot choose a suitable number $x_n$, the game ends and the other player wins. If the game goes forever, neither player wins. All chosen numbers are known to both players.

Determine all values of $\\lambda$ for which Alice has a winning strategy and all those for which Bazza has a winning strategy.""",
        "type": "game_theory",
        "difficulty": "hard",
        "expected_answer": "Alice wins if λ > 1/√2, Bazza wins if λ < 1/√2, draw if λ = 1/√2",
        "answer_type": "threshold",
        "key_insights": [
            "budget_analysis",
            "critical_threshold",
            "strategy_construction",
            "drawing_strategies"
        ],
        "solution_approach": "threshold_analysis_with_strategies"
    },
    {
        "id": 6,
        "problem": """Consider a $2025\\times2025$ grid of unit squares. Matilda wishes to place on the grid some rectangular tiles, possibly of difference sizes, such that each side of every tile lies on a grid line and every unit square is covered by at most one tile.

Determine the minimum number of tiles Matilda needs to place so that each row and each column of the grid has exactly one unit square that is not covered by any tile.""",
        "type": "combinatorial_optimization",
        "difficulty": "hard",
        "expected_answer": "2025",
        "answer_type": "number",
        "key_insights": [
            "tiling_constraints",
            "row_column_requirements",
            "optimization_bounds",
            "construction_proof"
        ],
        "solution_approach": "extremal_combinatorics"
    }
]

def verify_answer_format(problem_id: int, solution: str) -> Dict[str, Any]:
    """
    Verify if the solution contains the correct answer format for problems with specific answers
    """
    result = {
        "correct_answer_found": False,
        "extracted_answer": None,
        "answer_score": 0.0,
        "error_message": ""
    }

    solution_clean = solution.lower().replace(" ", "").replace("\n", " ")

    if problem_id == 1:
        # Expected: {0, 1, 3}
        # Look for sets containing 0, 1, 3
        set_patterns = [
            r"\{0,1,3\}",
            r"\{0,\s*1,\s*3\}",
            r"\{1,0,3\}",
            r"\{3,1,0\}",
            # Allow other orderings
            r"\{[013,\s]+\}" # General pattern
        ]

        for pattern in set_patterns:
            if re.search(pattern, solution_clean):
                # Verify it actually contains exactly 0, 1, 3
                numbers = re.findall(r'\d+', re.search(pattern, solution_clean).group())
                if sorted([int(x) for x in numbers]) == [0, 1, 3]:
                    result["correct_answer_found"] = True
                    result["extracted_answer"] = "{0, 1, 3}"
                    result["answer_score"] = 1.0
                    break

    elif problem_id == 3:
        # Expected: 4
        # Look for "c = 4" or "constant is 4" etc.
        if re.search(r"c\s*=\s*4(?![0-9])", solution) or \
           re.search(r"constant.*4(?![0-9])", solution) or \
           re.search(r"answer.*4(?![0-9])", solution):
            result["correct_answer_found"] = True
            result["extracted_answer"] = "4"
            result["answer_score"] = 1.0

    elif problem_id == 4:
        # Expected: 6J·12^K where gcd(J,10)=1
        # Look for the formula pattern
        patterns = [
            r"6j.*12\^k",
            r"6.*j.*12\^k",
            r"a_1\s*=\s*6.*12",
            r"6.*\*.*12\^"
        ]

        for pattern in patterns:
            if re.search(pattern, solution_clean):
                result["correct_answer_found"] = True
                result["extracted_answer"] = "6J·12^K"
                result["answer_score"] = 1.0
                break

    elif problem_id == 5:
        # Expected: threshold at 1/√2
        threshold_found = False
        patterns = [
            r"λ\s*>\s*1/√2",
            r"lambda\s*>\s*1/sqrt\(2\)",
            r"1/√2",
            r"√2/2",
            r"sqrt\(2\)/2"
        ]

        for pattern in patterns:
            if re.search(pattern, solution):
                threshold_found = True
                break

        if threshold_found:
            # Also check for Alice/Bazza winning conditions
            alice_wins = "alice.*win" in solution_clean or "alice.*λ.*>" in solution_clean
            bazza_wins = "bazza.*win" in solution_clean or "bazza.*λ.*<" in solution_clean

            if alice_wins and bazza_wins:
                result["correct_answer_found"] = True
                result["extracted_answer"] = "λ = 1/√2 threshold"
                result["answer_score"] = 1.0

    elif problem_id == 6:
        # Expected: 2025
        if re.search(r"2025", solution) and ("minimum" in solution_clean or "answer" in solution_clean):
            result["correct_answer_found"] = True
            result["extracted_answer"] = "2025"
            result["answer_score"] = 1.0

    return result

def verify_key_insights(problem_id: int, solution: str) -> Dict[str, Any]:
    """
    Check for problem-specific key insights that should appear in correct solutions
    """
    problem_data = next((p for p in IMO_2025_PROBLEMS if p["id"] == problem_id), None)
    if not problem_data:
        return {"insight_score": 0.0, "insights_found": [], "insights_missing": []}

    key_insights = problem_data["key_insights"]
    solution_lower = solution.lower()

    insights_found = []
    insights_missing = []

    # Define keywords for each insight type
    insight_keywords = {
        # Problem 1
        "reduction_principle": ["reduction", "reduce", "specific case"],
        "structural_lemma": ["structural", "lemma", "vertical", "horizontal", "diagonal"],
        "c_k_analysis": ["c(k)", "assertion", "pk can be covered"],
        "sunny_line_covering": ["sunny", "shady", "parallel"],

        # Problem 2
        "excenter_identification": ["excenter", "external", "angle bisector"],
        "auxiliary_point_v": ["auxiliary", "point v", "parallelogram"],
        "orthocenter_tangency": ["orthocenter", "tangent", "perpendicular"],
        "circumcircle_properties": ["circumcircle", "circumcenter"],

        # Problem 3
        "classification_lemma": ["classification", "lemma", "set s"],
        "set_s_analysis": ["s = p", "s = ∅", "s = {2}", "infinite", "finite"],
        "upper_bound_proof": ["upper bound", "f(n) ≤", "c ≤ 4"],
        "construction_example": ["construction", "example", "g(n)"],

        # Problem 4
        "regime_analysis": ["regime", "growth", "boost", "fixed point"],
        "evolution_dynamics": ["evolution", "sequence", "a_{n+1}"],
        "divisibility_constraints": ["6|an", "divisible", "v2", "v3"],
        "fixed_point_analysis": ["fixed point", "stable", "r(n) = 1"],

        # Problem 5
        "budget_analysis": ["budget", "ck", "evolution"],
        "critical_threshold": ["threshold", "1/√2", "critical"],
        "strategy_construction": ["strategy", "alice", "bazza"],
        "drawing_strategies": ["draw", "game continues", "forever"],

        # Problem 6
        "tiling_constraints": ["tile", "rectangular", "cover"],
        "row_column_requirements": ["row", "column", "exactly one"],
        "optimization_bounds": ["minimum", "lower bound", "upper bound"],
        "construction_proof": ["construction", "proof", "achieve"]
    }

    for insight in key_insights:
        if insight in insight_keywords:
            keywords = insight_keywords[insight]
            if any(keyword in solution_lower for keyword in keywords):
                insights_found.append(insight)
            else:
                insights_missing.append(insight)

    insight_score = len(insights_found) / len(key_insights) if key_insights else 0.0

    return {
        "insight_score": insight_score,
        "insights_found": insights_found,
        "insights_missing": insights_missing,
        "total_insights": len(key_insights)
    }

def get_problem_by_id(problem_id: int) -> Optional[Dict[str, Any]]:
    """Get problem data by ID"""
    return next((p for p in IMO_2025_PROBLEMS if p["id"] == problem_id), None)

def get_expected_answer(problem_id: int) -> Optional[str]:
    """Get the expected answer for a problem"""
    problem = get_problem_by_id(problem_id)
    return problem["expected_answer"] if problem else None

def get_answer_type(problem_id: int) -> Optional[str]:
    """Get the answer type for a problem"""
    problem = get_problem_by_id(problem_id)
    return problem["answer_type"] if problem else None