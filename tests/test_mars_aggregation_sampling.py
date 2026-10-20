"""Check that aggregation retains candidates from different reasoning agents."""

from optillm.mars.aggregator import MARSAggregator
from optillm.mars.workspace import AgentSolution, MARSWorkspace


def solution(agent_id, confidence, score):
    return AgentSolution(
        agent_id=agent_id,
        solution=f"{agent_id}: {confidence}",
        confidence=confidence,
        reasoning_tokens=0,
        total_tokens=0,
        solution_length=0,
        verification_score=score,
    )


def test_diverse_group_selects_best_confidence_from_each_agent():
    workspace = MARSWorkspace("problem", {})
    strongest = [solution("a", 0.4, 0.9), solution("a", 0.8, 0.9), solution("a", 0.6, 0.9)]
    other_agents = [solution("b", 0.7, 0.5), solution("c", 0.6, 0.4)]
    workspace.solutions = strongest + other_agents
    aggregator = MARSAggregator(None, "test-model", {"population_size": 6, "aggregation_size": 3})

    groups = aggregator._sample_solutions_for_aggregation(workspace)

    assert groups[0] == strongest
    assert groups[1] == [strongest[1], *other_agents]


def test_diverse_group_fills_remaining_slots_without_duplicates():
    workspace = MARSWorkspace("problem", {})
    candidates = [solution("a", 0.8, 0.9), solution("a", 0.5, 0.8), solution("b", 0.7, 0.5)]
    workspace.solutions = candidates
    aggregator = MARSAggregator(None, "test-model", {"population_size": 6, "aggregation_size": 3})

    groups = aggregator._sample_solutions_for_aggregation(workspace)

    assert all(candidate in groups[1] for candidate in candidates)
    assert len({id(candidate) for candidate in groups[1]}) == 3
