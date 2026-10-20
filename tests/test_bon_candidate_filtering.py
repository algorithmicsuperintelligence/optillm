from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from optillm.bon import best_of_n_sampling


def response(*choices):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=reason)
            for text, reason in choices
        ],
        usage=SimpleNamespace(completion_tokens=1),
    )


@pytest.mark.parametrize(
    "invalid_text, reason", [("incomplete answer", "length"), ("   ", "stop")]
)
def test_batched_invalid_candidates_are_never_sent_to_judge(invalid_text, reason):
    rated = []

    def create(**kwargs):
        if kwargs.get("n") == 2:
            return response((invalid_text, reason), ("complete answer", "stop"))
        candidate = kwargs["messages"][-2]["content"]
        rated.append(candidate)
        return response(("10" if candidate == invalid_text else "8", "stop"))

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    answer, tokens = best_of_n_sampling("system", "question", client, "model", n=2)

    assert answer == "complete answer"
    assert rated == ["complete answer"]
    assert tokens == 2


def test_all_invalid_batch_uses_existing_sequential_fallback():
    create = Mock(
        side_effect=[
            response(("partial", "length")),
            response(("complete answer", "stop")),
            response(("9", "stop")),
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    answer, tokens = best_of_n_sampling("system", "question", client, "model", n=1)

    assert answer == "complete answer"
    assert "n" not in create.call_args_list[1].kwargs
    assert tokens == 3
