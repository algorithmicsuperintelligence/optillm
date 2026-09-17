"""Streaming a completed response must keep tool calls and reasoning.

The proxy and `none` paths get a full chat completion back and, when the client
asked for `stream: true`, re-emit it as SSE. They used to forward only the
message text, so a tool call (which has empty content) reached the client as a
bare `[DONE]`, and agents that stream saw an empty reply.
"""
import json

from optillm.server import generate_streaming_completion


def _chunks(gen):
    out = []
    for frame in gen:
        assert frame.startswith("data: ") and frame.endswith("\n\n")
        body = frame[len("data: "):-2]
        out.append(body if body == "[DONE]" else json.loads(body))
    return out


def test_tool_calls_survive_streaming():
    completion = {
        "id": "abc", "model": "m", "created": 1,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    chunks = _chunks(generate_streaming_completion(completion, "optillm"))
    assert chunks[-1] == "[DONE]"
    first = chunks[0]["choices"][0]
    assert first["finish_reason"] == "tool_calls"
    call = first["delta"]["tool_calls"][0]
    assert call["index"] == 0 and call["function"]["name"] == "get_weather"
    assert chunks[1]["usage"]["total_tokens"] == 8


def test_text_and_reasoning_survive_streaming():
    completion = {"choices": [{"message": {"content": "hi", "reasoning": "thinking"},
                               "finish_reason": "stop"}]}
    first = _chunks(generate_streaming_completion(completion, "optillm"))[0]["choices"][0]
    assert first["delta"]["content"] == "hi"
    assert first["delta"]["reasoning"] == "thinking"


def test_a_list_of_completions_streams_every_choice():
    completions = [{"choices": [{"message": {"content": "a"}}]},
                   {"choices": [{"message": {"content": "b"}}]}]
    chunks = _chunks(generate_streaming_completion(completions, "optillm"))
    assert [c["choices"][0]["delta"]["content"] for c in chunks[:-1]] == ["a", "b"]
