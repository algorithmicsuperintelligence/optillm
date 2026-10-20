"""Generated solutions must be verified by peers and refined by their author."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from optillm.mars.agent import MARSAgent
from optillm.mars.verifier import MARSVerifier
from optillm.mars.workspace import MARSWorkspace


def response(text, reasoning_tokens=7):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(total_tokens=20, reasoning_tokens=reasoning_tokens),
    )


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("author_id,verifier_id", [(0, 2), (1, 2), (2, 0)])
def test_generated_solution_is_verified_and_improved_by_the_right_agents(
    parallel, author_id, verifier_id
):
    config = {"max_tokens": 100, "max_verification_attempts": 1}
    workspace = MARSWorkspace("Solve 2x + 5 = 15.", config)
    feedback = "INCORRECT: the subtraction is wrong. Subtract 5 before dividing."
    agents = [MARSAgent(i, Mock(), "test-model", config) for i in range(3)]
    for agent in agents:
        agent.client.chat.completions.create.return_value = response(feedback)

    author = agents[author_id]
    author.client.chat.completions.create.side_effect = [
        response("The answer is 10."),
        response("The answer is 5.", reasoning_tokens=11),
    ]
    solution, _ = author.generate_solution(workspace.problem)
    assert solution.agent_id == str(author_id)
    workspace.add_solution(solution)
    verifier = MARSVerifier(agents, workspace, config)

    if parallel:
        verification = asyncio.run(verifier.verify_solutions_parallel())
    else:
        verification = verifier.verify_solutions()

    detail = verification["verification_details"][0]
    assert detail["verification_results"][0]["verifier_id"] == verifier_id
    assert solution.verification_results[-1]["assessment"] == "INCORRECT"

    if parallel:
        improvement = asyncio.run(verifier.iterative_improvement_parallel())
    else:
        improvement = verifier.iterative_improvement()

    assert improvement == {
        "solutions_improved": 1,
        "improvement_attempts": 1,
        "total_reasoning_tokens": 11,
    }
    assert solution.solution == "The answer is 5."
    assert solution.reasoning_tokens == 18
    assert author.client.chat.completions.create.call_count == 2
    prompt = author.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "The answer is 10." in prompt
    assert feedback in prompt
    for agent in agents:
        if agent.agent_id not in (author_id, verifier_id):
            agent.client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("author_id,verifier_id", [(0, 2), (1, 2), (2, 0)])
def test_integer_ids_still_choose_a_peer_with_a_different_temperature(author_id, verifier_id):
    agents = [MARSAgent(i, Mock(), "test-model", {}) for i in range(3)]
    verifier = MARSVerifier(agents, MARSWorkspace("problem", {}), {})

    assert verifier._select_verifier_agent(author_id) is agents[verifier_id]


def test_aggregated_and_single_agent_solutions_keep_the_fallback():
    agent = MARSAgent(0, Mock(), "test-model", {})
    workspace = MARSWorkspace("problem", {})
    verifier = MARSVerifier([agent], workspace, {})

    assert verifier._select_verifier_agent("0") is agent
    assert verifier._select_verifier_agent("agg_123456") is agent
    assert MARSVerifier([], workspace, {})._select_verifier_agent("0") is None
