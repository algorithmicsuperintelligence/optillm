"""Verification feedback must reach synthesized as well as original solutions."""

from datetime import datetime

import pytest

from optillm.mars.workspace import AgentSolution, MARSWorkspace, VerificationResult


@pytest.mark.parametrize("agent_id", [0, "1", "agg_123456", "strategy_iter_2"])
def test_verification_updates_the_full_agent_id(agent_id):
    workspace = MARSWorkspace("problem", {"verification_passes_required": 2, "consensus_threshold": 1})
    solution = AgentSolution(agent_id, "answer", 0.8, 10, 20, 6)
    workspace.add_solution(solution)

    for _ in range(2):
        workspace.add_verification(
            VerificationResult(
                verifier_id=2,
                solution_id=f"agent_{agent_id}_iter_0",
                assessment="CORRECT",
                confidence=0.9,
                issues=[],
                suggestions=[],
                detailed_report="Checked the answer",
                timestamp=datetime.now(),
            )
        )

    assert len(solution.verification_results) == 2
    assert solution.verification_score == 1.0
    assert workspace.get_verified_solutions() == [solution]
    assert workspace.has_consensus()
    assert not workspace.should_continue_iteration()
