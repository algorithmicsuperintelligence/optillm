from types import SimpleNamespace

from optillm.plugins.streaming_hints import (
    Hint,
    HintMemoryBank,
    StreamingHintRetriever,
    run_streaming_hints,
)
from optillm.plugins import streaming_hints_plugin

PROBLEM = "A train travels 120 km in 1.5 hours. What is its speed in meters per second?"


def response(text, reason="stop", tokens=10):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=reason)],
        usage=SimpleNamespace(completion_tokens=tokens),
    )


def fake_client(*responses):
    calls = []
    queue = list(responses)

    def create(**kwargs):
        calls.append(kwargs)
        return queue.pop(0)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


def units_bank():
    return HintMemoryBank([
        Hint(id="units", text="Check the units of every quantity.", keywords=["units", "speed", "meters", "hours"]),
        Hint(id="cases", text="Split into cases.", keywords=["count", "cases", "arrangements"]),
    ])


def test_default_bank_loads_and_ranks_relevant_hint_first():
    bank = HintMemoryBank.from_file()
    assert len(bank.hints) >= 10
    assert len({h.id for h in bank.hints}) == len(bank.hints)

    (best, score), = bank.search("how many ways can 5 people be arranged in a row", k=1)
    assert best.id == "math-casework"
    assert score > 0


def test_search_skips_excluded_and_unrelated_queries():
    bank = units_bank()
    assert bank.search(PROBLEM, k=1)[0][0].id == "units"
    assert bank.search(PROBLEM, k=1, exclude=["units"]) == []
    assert bank.search("zzz qqq", k=2) == []


def test_retriever_waits_for_uncertainty_signal_and_gap():
    retriever = StreamingHintRetriever(units_bank(), threshold=0.1, min_gap_chars=20)
    retriever.reset(PROBLEM)

    # A confident sentence is not an injection point.
    assert retriever.feed("The distance is 120 km and the time is 1.5 hours. ") is None

    text = "So the speed is 80. Wait, the answer must be in meters per second. "
    injection = retriever.feed(text)
    assert injection is not None
    assert injection.hint.id == "units"
    assert text[: injection.offset].endswith("meters per second.")
    assert injection.hint.uses == 1

    # Each hint is injected at most once.
    assert retriever.feed("Hmm, let me check that again. " * 3) is None


def test_retriever_handles_sentences_split_across_feeds():
    retriever = StreamingHintRetriever(units_bank(), threshold=0.1, min_gap_chars=0)
    retriever.reset(PROBLEM)

    assert retriever.feed("Hmm, I am not sure about the un") is None
    injection = retriever.feed("its here. More text that is discarded.")
    assert injection is not None
    assert injection.offset == len("its here.")


def test_retriever_respects_max_hints_and_threshold():
    retriever = StreamingHintRetriever(units_bank(), threshold=0.99, min_gap_chars=0)
    retriever.reset(PROBLEM)
    assert retriever.feed("Wait, what are the units of speed here? ") is None

    retriever = StreamingHintRetriever(units_bank(), threshold=0.1, min_gap_chars=0, max_hints=0)
    retriever.reset(PROBLEM)
    assert retriever.feed("Wait, what are the units of speed here? ") is None


def test_run_injects_hint_and_continues_from_cut():
    first = (
        "Speed is distance over time, so 120 / 1.5 = 80. "
        "Wait, the question asks for meters per second. "
        "This tail is discarded."
    )
    client, calls = fake_client(
        response(first, tokens=30),
        response(" 80 km/h is 80000 m / 3600 s, about 22.2 m/s.", tokens=20),
    )

    answer, tokens = run_streaming_hints(
        "system", PROBLEM, client, "model",
        request_config={"hint_threshold": 0.1, "min_gap_chars": 20},
        memory_bank=units_bank(),
    )

    assert tokens == 50
    assert "discarded" not in answer
    assert "<hint>" not in answer
    assert answer.endswith("about 22.2 m/s.")

    assert len(calls) == 2
    assert calls[0]["messages"][1]["content"] == PROBLEM
    continuation = calls[1]["messages"][1]["content"]
    assert continuation.startswith(PROBLEM)
    assert "<hint>Check the units of every quantity.</hint>" in continuation
    assert "meters per second.\n<hint>" in continuation
    assert "discarded" not in continuation


def test_run_show_hints_returns_reasoning_with_hint_tags():
    client, _ = fake_client(
        response("Speed is 80. Wait, the units are meters per second. Tail."),
        response(" So 22.2 m/s."),
    )
    answer, _ = run_streaming_hints(
        "system", PROBLEM, client, "model",
        request_config={"hint_threshold": 0.1, "min_gap_chars": 0, "show_hints": True},
        memory_bank=units_bank(),
    )
    assert "<hint>Check the units of every quantity.</hint>" in answer
    assert answer.endswith("So 22.2 m/s.")


def test_run_continues_after_length_cutoff_and_respects_token_budget():
    client, calls = fake_client(
        response("Part one", reason="length", tokens=6),
        response(" part two", reason="length", tokens=4),
        response(" never requested"),
    )
    answer, tokens = run_streaming_hints(
        "system", "question", client, "model",
        request_config={"max_tokens": 10, "segment_tokens": 6},
        memory_bank=units_bank(),
    )
    assert answer == "Part one part two"
    assert tokens == 10
    assert [c["max_tokens"] for c in calls] == [6, 4]


def test_run_treats_full_budget_as_cut_off_when_finish_reason_is_stop():
    # The local inference server reports "stop" even when max_tokens was hit.
    client, calls = fake_client(
        response("Part one", reason="stop", tokens=6),
        response(" part two", reason="stop", tokens=3),
    )
    answer, tokens = run_streaming_hints(
        "system", "question", client, "model",
        request_config={"segment_tokens": 6},
        memory_bank=units_bank(),
    )
    assert answer == "Part one part two"
    assert tokens == 9
    assert len(calls) == 2


def test_run_without_relevant_hints_is_a_single_call():
    client, calls = fake_client(response("The answer is 4. Wait, yes, 4.", tokens=7))
    answer, tokens = run_streaming_hints(
        "system", "What is 2 + 2?", client, "model",
        request_config={"min_gap_chars": 0, "temperature": 0.3},
        memory_bank=units_bank(),
    )
    assert answer == "The answer is 4. Wait, yes, 4."
    assert tokens == 7
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.3


def test_plugin_entry_point():
    assert streaming_hints_plugin.SLUG == "streaming_hints"
    client, _ = fake_client(response("Done."))
    answer, tokens = streaming_hints_plugin.run("system", "question", client, "model")
    assert answer == "Done."
    assert tokens == 10
