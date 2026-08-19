"""Tests for compact_plugin."""

import os
import pytest
from unittest.mock import MagicMock, patch
from optillm.plugins.compact_plugin import (
    estimate_tokens,
    parse_tagged_conversation,
    reconstruct_tagged,
    _get_config,
    run,
)


def _make_client(summary_text="<summary>Test summary with Key decisions: decided X.</summary>"):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = summary_text
    mock_response.usage.completion_tokens = 80
    client = MagicMock()
    client.chat.completions.create.return_value = mock_response
    client.models.retrieve.return_value = MagicMock(spec=[])
    return client


class TestEstimateTokens:
    def test_short_text(self):
        assert estimate_tokens("hello") >= 1

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_long_text_returns_positive(self):
        assert estimate_tokens("a" * 400) > 0


class TestParseTaggedConversation:
    def test_single_user_message(self):
        assert parse_tagged_conversation("User: hello") == [("user", "hello")]

    def test_conversation_pair(self):
        assert parse_tagged_conversation("User: hi\nAssistant: hello") == [("user", "hi"), ("assistant", "hello")]

    def test_multi_turn(self):
        turns = parse_tagged_conversation("User: q1\nAssistant: a1\nUser: q2\nAssistant: a2")
        assert len(turns) == 4
        assert turns[0] == ("user", "q1")
        assert turns[3] == ("assistant", "a2")

    def test_empty_string(self):
        assert parse_tagged_conversation("") == []

    def test_no_tags(self):
        assert parse_tagged_conversation("just some text") == []


class TestReconstructTagged:
    def test_roundtrip(self):
        result = reconstruct_tagged([("user", "hello"), ("assistant", "world")])
        assert "User: hello" in result
        assert "Assistant: world" in result

    def test_empty(self):
        assert reconstruct_tagged([]) == ""


class TestGetConfig:
    def test_default_when_nothing_set(self):
        assert _get_config(None, "key", "NONEXISTENT_VAR", 42) == 42

    def test_request_config_takes_priority(self):
        assert _get_config({"compact_threshold": 0.5}, "compact_threshold", "COMPACT_THRESHOLD", 0.75) == 0.5

    def test_env_var_as_fallback(self):
        with patch.dict(os.environ, {"COMPACT_KEEP_RECENT": "6"}):
            assert _get_config(None, "compact_keep_recent", "COMPACT_KEEP_RECENT", 4) == 6

    def test_request_config_overrides_env(self):
        with patch.dict(os.environ, {"COMPACT_KEEP_RECENT": "6"}):
            assert _get_config({"compact_keep_recent": 2}, "compact_keep_recent", "COMPACT_KEEP_RECENT", 4) == 2


class TestRun:
    def test_passthrough_short_conversation(self):
        query = "User: hi\nAssistant: hello"
        result, tokens = run("system", query, _make_client(), "gpt-4")
        assert result == query
        assert tokens == 0

    def test_passthrough_few_turns(self):
        turns = []
        for i in range(3):
            turns.append(f"User: question {i}")
            turns.append(f"Assistant: answer {i}")
        query = "\n".join(turns)
        result, tokens = run("system", query, _make_client(), "gpt-4",
                             request_config={"compact_context_window": 100, "compact_threshold": 0.5})
        assert result == query
        assert tokens == 0

    def test_compression_triggered_uses_llm(self):
        turns = []
        for i in range(20):
            turns.append(f"User: this is a longer question number {i} with extra text to increase token count")
            turns.append(f"Assistant: this is a longer answer number {i} with extra text to increase token count")
        query = "\n".join(turns)

        client = _make_client("<summary>\nConversation summary:\n- Scope: 36 messages.\n- Key decisions: decided to use compact.\n- Context: user was testing compression.\n</summary>")
        result, tokens = run("system", query, client, "gpt-4",
                             request_config={"compact_context_window": 200, "compact_threshold": 0.3,
                                             "compact_keep_recent": 4})

        assert tokens == 80  # LLM was called
        assert "[Conversation summary]" in result
        assert "Scope:" in result
        assert "question number 19" in result  # last user turn preserved
        client.chat.completions.create.assert_called_once()

    def test_structured_summary_format(self):
        turns = []
        for i in range(20):
            turns.append(f"User: question {i} " + "x" * 100)
            turns.append(f"Assistant: answer {i} " + "y" * 100)
        query = "\n".join(turns)

        summary_text = "<summary>\nConversation summary:\n- Scope: 36 messages.\n- Key decisions: use plugin.\n- Key files: src/main.py.\n- Context: testing.\n</summary>"
        client = _make_client(summary_text)
        result, tokens = run("system", query, client, "gpt-4",
                             request_config={"compact_context_window": 200, "compact_threshold": 0.1,
                                             "compact_keep_recent": 2})

        assert "Key decisions:" in result
        assert "Key files:" in result

    def test_output_preserves_tag_format(self):
        turns = []
        for i in range(10):
            turns.append(f"User: question {i} " + "x" * 100)
            turns.append(f"Assistant: answer {i} " + "y" * 100)
        query = "\n".join(turns)

        client = _make_client()
        result, tokens = run("system", query, client, "gpt-4",
                             request_config={"compact_context_window": 200, "compact_threshold": 0.1,
                                             "compact_keep_recent": 2})

        parsed = parse_tagged_conversation(result)
        assert len(parsed) >= 2
        assert parsed[0][0] == "user"
        assert "[Conversation summary]" in parsed[0][1]

    def test_env_var_configuration(self):
        turns = []
        for i in range(10):
            turns.append(f"User: question {i} " + "x" * 100)
            turns.append(f"Assistant: answer {i} " + "y" * 100)
        query = "\n".join(turns)

        client = _make_client()
        with patch.dict(os.environ, {"COMPACT_CONTEXT_WINDOW": "200", "COMPACT_THRESHOLD": "0.1",
                                      "COMPACT_KEEP_RECENT": "2"}):
            result, tokens = run("system", query, client, "gpt-4")

        assert tokens == 80
        assert "[Conversation summary]" in result

    def test_llm_failure_falls_back_to_passthrough(self):
        turns = []
        for i in range(10):
            turns.append(f"User: question {i} " + "x" * 100)
            turns.append(f"Assistant: answer {i} " + "y" * 100)
        query = "\n".join(turns)

        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")

        result, tokens = run("system", query, client, "gpt-4",
                             request_config={"compact_context_window": 200, "compact_threshold": 0.1,
                                             "compact_keep_recent": 2})
        assert result == query
        assert tokens == 0

    def test_summary_tag_extraction(self):
        turns = []
        for i in range(10):
            turns.append(f"User: question {i} " + "x" * 100)
            turns.append(f"Assistant: answer {i} " + "y" * 100)
        query = "\n".join(turns)

        raw_llm_output = "Here is the summary:\n<summary>\nConversation summary:\n- Scope: 10 messages.\n- Key decisions: decided X.\n</summary>\nHope this helps!"
        client = _make_client(summary_text=raw_llm_output)

        result, tokens = run("system", query, client, "gpt-4",
                             request_config={"compact_context_window": 200, "compact_threshold": 0.1,
                                             "compact_keep_recent": 2})

        assert "Here is the summary:" not in result
        assert "Hope this helps!" not in result
        assert "Scope:" in result
        assert "Key decisions:" in result

    def test_summary_without_tags_used_raw(self):
        turns = []
        for i in range(10):
            turns.append(f"User: question {i} " + "x" * 100)
            turns.append(f"Assistant: answer {i} " + "y" * 100)
        query = "\n".join(turns)

        client = _make_client(summary_text="Plain summary without XML tags. Key decisions: use compact.")

        result, tokens = run("system", query, client, "gpt-4",
                             request_config={"compact_context_window": 200, "compact_threshold": 0.1,
                                             "compact_keep_recent": 2})

        assert "Plain summary" in result
        assert tokens == 80

    def test_system_prompt_included_in_compression(self):
        turns = []
        for i in range(10):
            turns.append(f"User: question {i} " + "x" * 100)
            turns.append(f"Assistant: answer {i} " + "y" * 100)
        query = "\n".join(turns)

        client = _make_client()
        result, tokens = run("You are a medical coding assistant", query, client, "gpt-4",
                             request_config={"compact_context_window": 200, "compact_threshold": 0.1,
                                             "compact_keep_recent": 2})

        call_args = client.chat.completions.create.call_args
        system_content = call_args.kwargs["messages"][0]["content"]
        assert "medical coding assistant" in system_content

    def test_keep_recent_exceeds_turn_count(self):
        query = "User: hi\nAssistant: hello"
        client = _make_client()
        result, tokens = run("system", query, client, "gpt-4",
                             request_config={"compact_context_window": 1, "compact_threshold": 0.1,
                                             "compact_keep_recent": 100})
        assert result == query
        assert tokens == 0

    def test_threshold_zero_always_triggers(self):
        turns = []
        for i in range(10):
            turns.append(f"User: question {i} " + "x" * 100)
            turns.append(f"Assistant: answer {i} " + "y" * 100)
        query = "\n".join(turns)

        client = _make_client()
        result, tokens = run("system", query, client, "gpt-4",
                             request_config={"compact_context_window": 200, "compact_threshold": 0.0,
                                             "compact_keep_recent": 2})

        assert tokens == 80
        assert "[Conversation summary]" in result

    def test_malformed_env_var_uses_default(self):
        with patch.dict(os.environ, {"COMPACT_THRESHOLD": "not_a_number"}):
            val = _get_config(None, "compact_threshold", "COMPACT_THRESHOLD", 0.75)
        assert val == 0.75

    def test_embedded_tags_not_split(self):
        text = "User: I asked my friend: User: what is Python?\nAssistant: Here is the answer"
        turns = parse_tagged_conversation(text)
        assert len(turns) == 2
        assert "friend: User: what is Python?" in turns[0][1]


class TestGetContextWindow:
    def test_provider_returns_context_length(self):
        from optillm.plugins.compact_plugin import _get_context_window
        model_info = MagicMock()
        model_info.context_length = 32768
        client = MagicMock()
        client.models.retrieve.return_value = model_info
        result = _get_context_window(client, "test-model", None)
        assert result == 32768

    def test_provider_returns_max_context_length(self):
        from optillm.plugins.compact_plugin import _get_context_window
        model_info = MagicMock()
        model_info.context_length = None
        model_info.max_context_length = 65536
        client = MagicMock()
        client.models.retrieve.return_value = model_info
        result = _get_context_window(client, "test-model", None)
        assert result == 65536

    def test_provider_no_context_info_falls_back_to_config(self):
        from optillm.plugins.compact_plugin import _get_context_window
        model_info = MagicMock(spec=[])
        client = MagicMock()
        client.models.retrieve.return_value = model_info
        result = _get_context_window(client, "test-model", {"compact_context_window": 50000})
        assert result == 50000

    def test_provider_api_fails_falls_back_to_default(self):
        from optillm.plugins.compact_plugin import _get_context_window
        client = MagicMock()
        client.models.retrieve.side_effect = Exception("not supported")
        result = _get_context_window(client, "test-model", None)
        assert result == 128000
