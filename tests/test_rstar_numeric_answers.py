import unittest
from unittest.mock import Mock

from optillm.rstar import Node, RStar


class TestRStarNumericAnswers(unittest.TestCase):
    def setUp(self):
        self.search = RStar("", Mock(), "mock", max_depth=1, num_rollouts=1)

    def test_explicit_answers_keep_the_complete_number(self):
        for answer in ("-3", "+3", "1.25", "-0.25", ".5", "-.5", "1.2e-3", "-2E+4", "42"):
            for prefix in ("The answer is", "The final answer is", "Thus, the answer is"):
                with self.subTest(answer=answer, prefix=prefix):
                    self.assertEqual(
                        self.search.extract_answer(f"{prefix} {answer}."),
                        (answer, 1.0),
                    )

    def test_fallback_keeps_sign_decimal_and_exponent(self):
        for answer in ("-3", "1.25", "-.5", "1.2e-3", "42"):
            with self.subTest(answer=answer):
                self.assertEqual(
                    self.search.extract_answer(f"After 2 steps, we obtain {answer}."),
                    (answer, 0.5),
                )

    def test_answer_without_numbers_remains_unknown(self):
        self.assertEqual(self.search.extract_answer("More reasoning is needed."), ("", 0.0))

    def test_signed_answer_receives_explicit_answer_reward(self):
        node = Node("The final answer is -1.25", "A1")
        self.assertEqual(self.search.evaluate(node), 1.0)

    def test_answer_voting_does_not_merge_different_decimal_values(self):
        states = ["The answer is 1.25", "The answer is 1.75", "The answer is 1.75"]
        answers = [self.search.extract_answer(state) for state in states]
        self.assertEqual(self.search.select_best_answer(answers), "1.75")

    def test_solve_preserves_the_generated_numeric_answer(self):
        self.search.generate_response = Mock(return_value="The final answer is -1.25")

        answer, tokens = self.search.solve("Compute the result.")

        self.assertEqual(answer, "-1.25")
        self.assertEqual(tokens, 0)
