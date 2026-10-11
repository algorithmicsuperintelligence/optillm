import asyncio
import unittest
from unittest.mock import Mock

from optillm.rstar import RStar


class TestRStarBackpropagation(unittest.TestCase):
    def test_sync_rollouts_record_terminal_rewards(self):
        for depth in (0, 2):
            with self.subTest(depth=depth):
                search = RStar("", Mock(), "mock", max_depth=depth, num_rollouts=3)
                search.generate_response = Mock(return_value="reasoning step")
                search.evaluate = Mock(return_value=0.75)

                trajectories = search.mcts("question")

                self.assertEqual(trajectories[0][0].visits, 3)
                self.assertAlmostEqual(trajectories[0][0].value, 2.25)
                for trajectory in trajectories:
                    for node in trajectory:
                        self.assertGreater(node.visits, 0)
                        self.assertAlmostEqual(node.value / node.visits, 0.75)
                    self.assertEqual(search.trajectory_score(trajectory), 0.75)


class TestAsyncRStarBackpropagation(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_trajectories_keep_their_own_rewards(self):
        search = RStar("", Mock(), "mock", max_depth=1, num_rollouts=2)
        responses = iter(["1", "2", "3", "4"])

        async def generate(prompt):
            # Both rollouts enter expansion before either completes it.
            await asyncio.sleep(0)
            return next(responses)

        search.generate_response_async = generate
        search.evaluate = Mock(side_effect=lambda node: 10 - float(node.state))
        search.mutual_consistency = Mock(return_value=True)

        trajectories = await search.mcts_async("question")

        self.assertEqual(len(trajectories), 2)
        self.assertEqual(trajectories[0][0].visits, 2)
        self.assertEqual(trajectories[0][0].value, 13)
        for trajectory in trajectories:
            for node in trajectory[1:]:
                self.assertEqual(node.visits, 1)
                self.assertEqual(node.value, 10 - float(trajectory[-1].state))
        best = search.select_final_trajectory(trajectories)
        self.assertEqual(best[-1].state, "3")

    async def test_zero_depth_counts_each_rollout_once(self):
        search = RStar("", Mock(), "mock", max_depth=0, num_rollouts=3)

        async def generate(prompt):
            await asyncio.sleep(0)
            return "reasoning step"

        search.generate_response_async = generate
        search.evaluate = Mock(return_value=0.5)
        trajectories = await search.mcts_async("question")

        self.assertEqual(trajectories[0][0].visits, 3)
        self.assertEqual(trajectories[0][0].value, 1.5)
        self.assertTrue(all(search.trajectory_score(t) == 0.5 for t in trajectories))
