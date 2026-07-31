#!/usr/bin/env python3
"""
MARS (Multi-Agent Reasoning System) IMO25 specific tests
Tests MARS on actual IMO25 problems to analyze failures and improve implementation
"""

import sys
import os
import time
import logging
import io
import unittest

# Add parent directory to path to import optillm modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm.mars import multi_agent_reasoning_system


class MockOpenAIClient:
    """Enhanced mock OpenAI client for IMO25 testing"""

    def __init__(self, response_delay=0.1, reasoning_tokens=2000):
        self.response_delay = response_delay
        self.reasoning_tokens = reasoning_tokens
        self.call_count = 0
        self.call_times = []

    def chat_completions_create(self, **kwargs):
        """Mock completions.create with realistic IMO25 responses"""
        time.time()
        time.sleep(self.response_delay)
        self.call_count += 1
        self.call_times.append(time.time())

        call_count = self.call_count

        class MockUsage:
            def __init__(self, reasoning_tokens):
                self.completion_tokens_details = type('obj', (), {
                    'reasoning_tokens': reasoning_tokens
                })()
                self.total_tokens = reasoning_tokens + 200

        class MockChoice:
            def __init__(self, content):
                self.message = type('obj', (), {
                    'content': content
                })()

        class MockResponse:
            def __init__(self, content, reasoning_tokens):
                self.choices = [MockChoice(content)]
                self.usage = MockUsage(reasoning_tokens)

        # Get problem content from messages
        messages = kwargs.get('messages', [])
        problem_content = ""
        for message in messages:
            problem_content += message.get('content', '')

        # Generate appropriate responses based on problem content and call type
        if "verifying" in problem_content.lower():
            # Verification response
            content = "VERIFICATION: This solution appears CORRECT. The analysis is mathematically sound and the final answer is properly justified. Confidence: 8/10."
        elif "improving" in problem_content.lower():
            # Improvement response
            content = "IMPROVEMENT: The original approach is good but can be enhanced. Here's the improved version with stronger reasoning..."
        elif "bonza" in problem_content.lower():
            # IMO25 Problem 3 - functional equation
            responses = [
                "Looking at this functional equation problem, I need to find the smallest constant c such that f(n) ≤ cn for all bonza functions f. Let me analyze the divisibility condition: f(a) divides b^a - f(b)^f(a). This is a complex functional equation. After careful analysis of the constraints, I believe the minimum constant is c = 4. This can be shown by constructing specific examples and proving upper bounds.",
                "For the bonza function problem, I'll work through the case analysis systematically. A function f: ℕ → ℕ is bonza if f(a) | (b^a - f(b)^f(a)) for all positive integers a,b. Through detailed analysis of the divisibility constraints and construction of extremal examples, the smallest real constant c such that f(n) ≤ cn for all bonza functions is c = 4.",
                "This functional equation requires careful analysis. I'll examine when f(a) divides b^a - f(b)^f(a). By studying specific cases and constructing examples, I can show that the minimal constant c = 4 is both necessary and sufficient. The answer is c = 4."
            ]
            content = responses[call_count % len(responses)]
        elif "three largest proper divisors" in problem_content.lower():
            # IMO25 Problem 4 - number theory sequence
            responses = [
                "For this sequence problem, I need to analyze when a_{n+1} equals the sum of three largest proper divisors of a_n. After examining the dynamics and constraints, the possible values of a_1 are of the form 6J·12^K where gcd(J,10)=1. This follows from regime analysis of the sequence evolution.",
                "Analyzing the sequence where each term is the sum of three largest proper divisors of the previous term. Through careful analysis of the divisibility patterns and sequence behavior, I find that a_1 must have the form a_1 = 6J·12^K where gcd(J,10)=1.",
                "The sequence evolution depends on the three largest proper divisors. After detailed analysis of the constraints and fixed point behavior, the answer is a_1 = 6J·12^K where gcd(J,10)=1."
            ]
            content = responses[call_count % len(responses)]
        elif "alice and bazza" in problem_content.lower():
            # IMO25 Problem 5 - game theory
            responses = [
                "In this inekoalaty game, Alice and Bazza have alternating constraints. Alice wins if λ > 1/√2, Bazza wins if λ < 1/√2, and it's a draw if λ = 1/√2. The critical threshold is λ = 1/√2 ≈ 0.707. This follows from analyzing the budget constraints and optimal strategies.",
                "For the game theory problem, the key is finding the threshold value of λ. Through analysis of the constraints x₁+x₂+...+xₙ ≤ λn and x₁²+x₂²+...+xₙ² ≤ n, the critical value is λ = 1/√2. Alice has a winning strategy when λ > 1/√2.",
                "The inekoalaty game has a critical threshold at λ = 1/√2. Alice wins for λ > 1/√2, Bazza wins for λ < 1/√2, and they draw at λ = 1/√2. This threshold emerges from the constraint analysis."
            ]
            content = responses[call_count % len(responses)]
        elif "2025×2025 grid" in problem_content.lower():
            # IMO25 Problem 6 - combinatorial optimization
            responses = [
                "For the tiling problem on a 2025×2025 grid, Matilda needs to place rectangular tiles such that each row and column has exactly one uncovered unit square. The minimum number of tiles needed is 2025. This can be achieved by strategic tile placement.",
                "In this combinatorial optimization problem, the constraint that each row and each column must have exactly one uncovered square leads to the minimum number of tiles being 2025. This follows from extremal combinatorics arguments.",
                "The minimum number of tiles for the 2025×2025 grid problem is 2025. This can be proven by considering the constraints and constructing an optimal tiling pattern."
            ]
            content = responses[call_count % len(responses)]
        else:
            # General mathematical response
            content = f"Mathematical solution {call_count}: This is a complex problem requiring systematic analysis. Let me work through it step by step with rigorous reasoning and provide a complete solution."

        return MockResponse(content, self.reasoning_tokens)

    @property
    def chat(self):
        return type('obj', (), {
            'completions': type('obj', (), {
                'create': self.chat_completions_create
            })()
        })()


class TestMARSIMO25(unittest.TestCase):
    """Test MARS on specific IMO25 problems"""

    def setUp(self):
        """Set up test fixtures with logging capture"""
        self.system_prompt = "You are a mathematical problem solver capable of handling complex olympiad-level problems."
        self.model = "mock-model"

        # Set up logging capture for detailed analysis
        self.log_capture = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        self.log_handler.setLevel(logging.INFO)

        # Add handler to MARS loggers
        mars_logger = logging.getLogger('optillm.mars')
        mars_logger.addHandler(self.log_handler)
        mars_logger.setLevel(logging.INFO)

        # Store original level to restore later
        self.original_level = mars_logger.level

    def tearDown(self):
        """Clean up test fixtures"""
        mars_logger = logging.getLogger('optillm.mars')
        mars_logger.removeHandler(self.log_handler)
        mars_logger.setLevel(self.original_level)
        self.log_handler.close()

    def get_captured_logs(self):
        """Get the captured log output"""
        return self.log_capture.getvalue()

    def test_imo25_problem3_functional_equation(self):
        """Test MARS on IMO25 Problem 3 - Functional Equation (Expected: c = 4)"""
        problem3 = """Let ℕ denote the set of positive integers. A function f:ℕ→ℕ is said to be bonza if f(a) divides b^a-f(b)^{f(a)} for all positive integers a and b.

Determine the smallest real constant c such that f(n)≤cn for all bonza functions f and all positive integers n."""

        print("\n🧮 Testing MARS on IMO25 Problem 3 (Expected answer: c = 4)...")

        client = MockOpenAIClient(response_delay=0.05, reasoning_tokens=3000)

        start_time = time.time()
        result = multi_agent_reasoning_system(
            self.system_prompt,
            problem3,
            client,
            self.model
        )
        execution_time = time.time() - start_time

        # Verify result structure
        self.assertIsInstance(result, tuple)
        response, tokens = result
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 100, "Response should be substantial for IMO problem")
        self.assertGreater(tokens, 0)

        # Check if the answer "4" appears in the response
        has_answer_4 = "4" in response
        has_constant_c = "c" in response.lower()

        print(f"  📊 Execution time: {execution_time:.2f}s")
        print(f"  📊 Response length: {len(response):,} characters")
        print(f"  📊 Total tokens: {tokens:,}")
        print(f"  📊 API calls made: {client.call_count}")
        print(f"  🎯 Contains answer '4': {has_answer_4}")
        print(f"  🎯 Contains 'constant c': {has_constant_c}")

        # Analyze the logs for answer extraction
        logs = self.get_captured_logs()

        # Look for voting and answer extraction in logs
        voting_logs = [line for line in logs.split('\n') if '🗳️  VOTING' in line]
        synthesis_logs = [line for line in logs.split('\n') if '🤝 SYNTHESIS' in line]

        print(f"  📋 Voting log entries: {len(voting_logs)}")
        print(f"  📋 Synthesis log entries: {len(synthesis_logs)}")

        if voting_logs:
            print(f"  📋 Sample voting log: {voting_logs[0][:100]}...")

        # Check for specific answer extraction patterns
        answer_extraction_logs = [line for line in logs.split('\n') if 'extracted answer' in line.lower()]
        if answer_extraction_logs:
            print(f"  🔍 Answer extraction logs found: {len(answer_extraction_logs)}")
            for log in answer_extraction_logs[:3]:
                print(f"    {log}")

        # Log key parts of the response for analysis
        response_lines = response.split('\n')
        key_lines = [line for line in response_lines if any(keyword in line.lower() for keyword in ['constant', 'c =', 'answer', '= 4', 'therefore'])]
        if key_lines:
            print("  🔑 Key response lines:")
            for line in key_lines[:5]:
                print(f"    {line.strip()}")

        print("✅ IMO25 Problem 3 test completed")

    def test_imo25_problem4_number_theory(self):
        """Test MARS on IMO25 Problem 4 - Number Theory (Expected: 6J·12^K formula)"""
        problem4 = """A proper divisor of a positive integer N is a positive divisor of N other than N itself.

The infinite sequence a_1,a_2,… consists of positive integers, each of which has at least three proper divisors. For each n≥1, the integer a_{n+1} is the sum of three largest proper divisors of a_n.

Determine all possible values of a_1."""

        print("\n🔢 Testing MARS on IMO25 Problem 4 (Expected: 6J·12^K formula)...")

        client = MockOpenAIClient(response_delay=0.05, reasoning_tokens=3000)

        start_time = time.time()
        result = multi_agent_reasoning_system(
            self.system_prompt,
            problem4,
            client,
            self.model
        )
        execution_time = time.time() - start_time

        # Verify result structure
        self.assertIsInstance(result, tuple)
        response, tokens = result
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 100, "Response should be substantial for IMO problem")

        # Check for formula components
        has_formula_6J = "6J" in response or "6j" in response.lower()
        has_formula_12K = "12^K" in response or "12^k" in response.lower()
        has_gcd_condition = "gcd" in response.lower()

        print(f"  📊 Execution time: {execution_time:.2f}s")
        print(f"  📊 Response length: {len(response):,} characters")
        print(f"  🎯 Contains '6J': {has_formula_6J}")
        print(f"  🎯 Contains '12^K': {has_formula_12K}")
        print(f"  🎯 Contains 'gcd': {has_gcd_condition}")

        print("✅ IMO25 Problem 4 test completed")

    def test_answer_extraction_analysis(self):
        """Test answer extraction specifically with controlled responses"""
        print("\n🔍 Testing answer extraction with controlled responses...")

        class ControlledMockClient(MockOpenAIClient):
            def __init__(self):
                super().__init__(response_delay=0.01, reasoning_tokens=1000)
                self.response_index = 0
                self.controlled_responses = [
                    "After careful analysis, I determine that the smallest constant c = 4. This can be proven by construction and bounds analysis.",
                    "The minimum value is c = 4. Therefore, the answer is 4.",
                    "Through systematic analysis, the constant c must equal 4. The final answer is c = 4."
                ]

            def chat_completions_create(self, **kwargs):
                # Override to provide controlled responses with clear answers
                result = super().chat_completions_create(**kwargs)
                if self.response_index < len(self.controlled_responses):
                    result.choices[0].message.content = self.controlled_responses[self.response_index]
                    self.response_index += 1
                return result

        simple_problem = "Find the smallest constant c such that f(n) ≤ cn for all valid functions f."

        client = ControlledMockClient()
        result = multi_agent_reasoning_system(
            self.system_prompt,
            simple_problem,
            client,
            self.model
        )

        response, tokens = result

        # Analyze logs for answer extraction details
        logs = self.get_captured_logs()
        voting_logs = [line for line in logs.split('\n') if 'VOTING' in line and 'extracted answer' in line.lower()]

        print(f"  📊 Response contains '4': {'4' in response}")
        print(f"  📊 Response contains 'c = 4': {'c = 4' in response}")
        print(f"  📋 Voting logs with extraction: {len(voting_logs)}")

        if voting_logs:
            for i, log in enumerate(voting_logs[:3]):
                print(f"    Vote {i+1}: {log}")

        print("✅ Answer extraction analysis completed")


def run_imo25_tests():
    """Run all IMO25 MARS tests"""
    print("Running MARS IMO25 specific tests...")
    print("=" * 80)

    # Run unittest tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMARSIMO25)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 80)

    if result.wasSuccessful():
        print("🎉 All IMO25 tests passed!")
        return True
    else:
        print("❌ Some IMO25 tests failed - analyzing for improvements")
        return False


if __name__ == "__main__":
    success = run_imo25_tests()
    sys.exit(0 if success else 1)