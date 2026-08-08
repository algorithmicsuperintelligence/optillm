import json
import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault(
    "math_verify",
    types.SimpleNamespace(parse=lambda *args, **kwargs: None),
)

from optillm.anthropic import (
    anthropic_request_to_openai,
    approximate_count_tokens,
    generate_anthropic_stream,
    openai_response_to_anthropic,
)


class MockClient:
    def __init__(self, response):
        self.response = response
        self.last_request = None
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, outer):
            self.completions = self.Completions(outer)

        class Completions:
            def __init__(self, outer):
                self.outer = outer

            def create(self, **kwargs):
                self.outer.last_request = kwargs
                return self.outer.response


def text_completion_response(content="Hello"):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


def tool_completion_response():
    return {
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\":\"README.md\"}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
    }


def test_anthropic_request_to_openai_text_and_parameters():
    request = {
        "model": "claude-test",
        "system": "You are concise.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        "max_tokens": 100,
        "temperature": 0.2,
        "top_p": 0.9,
        "stop_sequences": ["END"],
    }

    converted = anthropic_request_to_openai(request)

    assert converted["model"] == "claude-test"
    assert converted["messages"] == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Hi"},
    ]
    assert converted["max_tokens"] == 100
    assert converted["temperature"] == 0.2
    assert converted["top_p"] == 0.9
    assert converted["stop"] == ["END"]


def test_anthropic_request_to_openai_tools_and_results():
    request = {
        "model": "claude-test",
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "read_file"},
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "file contents",
                    }
                ],
            },
        ],
    }

    converted = anthropic_request_to_openai(request)

    assert converted["tools"][0]["function"]["name"] == "read_file"
    assert converted["tool_choice"]["function"]["name"] == "read_file"
    assert converted["messages"][0]["tool_calls"][0]["function"]["name"] == "read_file"
    assert converted["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "file contents",
    }


def test_openai_response_to_anthropic_text():
    converted = openai_response_to_anthropic(text_completion_response("Hello Claude"), "claude-test")

    assert converted["type"] == "message"
    assert converted["role"] == "assistant"
    assert converted["content"] == [{"type": "text", "text": "Hello Claude"}]
    assert converted["stop_reason"] == "end_turn"
    assert converted["usage"] == {"input_tokens": 7, "output_tokens": 3}


def test_openai_response_to_anthropic_tool_use():
    converted = openai_response_to_anthropic(tool_completion_response(), "claude-test")

    assert converted["stop_reason"] == "tool_use"
    assert converted["content"] == [
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "read_file",
            "input": {"path": "README.md"},
        }
    ]


def test_anthropic_stream_events():
    message = openai_response_to_anthropic(text_completion_response("stream me"), "claude-test")
    stream = "".join(generate_anthropic_stream(message))

    assert "event: message_start" in stream
    assert "event: content_block_start" in stream
    assert "event: content_block_delta" in stream
    assert "event: message_delta" in stream
    assert "event: message_stop" in stream


def test_approximate_count_tokens_is_deterministic():
    request = {"system": "You are concise.", "messages": [{"role": "user", "content": "Hello"}]}

    assert approximate_count_tokens(request) == approximate_count_tokens(request)
    assert approximate_count_tokens(request) > 0


def test_anthropic_messages_route(monkeypatch):
    import optillm.server as server

    mock_client = MockClient(text_completion_response("Route works"))
    monkeypatch.setattr(server, "get_config", lambda: (mock_client, "optillm"))
    monkeypatch.setitem(server.server_config, "approach", "none")

    response = server.app.test_client().post(
        "/v1/messages",
        json={
            "model": "claude-test",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["type"] == "message"
    assert body["content"][0]["text"] == "Route works"
    assert mock_client.last_request["model"] == "claude-test"


def test_anthropic_messages_route_stream(monkeypatch):
    import optillm.server as server

    mock_client = MockClient(text_completion_response("Route streams"))
    monkeypatch.setattr(server, "get_config", lambda: (mock_client, "optillm"))
    monkeypatch.setitem(server.server_config, "approach", "none")

    response = server.app.test_client().post(
        "/v1/messages",
        json={
            "model": "claude-test",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    payload = response.data.decode("utf-8")
    assert "event: message_start" in payload
    assert "event: content_block_delta" in payload
    assert "event: message_stop" in payload


def test_anthropic_count_tokens_route():
    import optillm.server as server

    response = server.app.test_client().post(
        "/v1/messages/count_tokens",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 200
    assert response.get_json()["input_tokens"] > 0


def test_anthropic_route_accepts_x_api_key(monkeypatch):
    import optillm.server as server

    mock_client = MockClient(text_completion_response("Authenticated"))
    monkeypatch.setattr(server, "get_config", lambda: (mock_client, "optillm"))
    monkeypatch.setitem(server.server_config, "optillm_api_key", "test-key")
    monkeypatch.setitem(server.server_config, "approach", "none")

    response = server.app.test_client().post(
        "/v1/messages",
        headers={"x-api-key": "test-key"},
        json={
            "model": "claude-test",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    assert response.get_json()["content"][0]["text"] == "Authenticated"


def test_anthropic_tool_requests_bypass_approaches(monkeypatch):
    import optillm.server as server

    mock_client = MockClient(tool_completion_response())
    monkeypatch.setattr(server, "get_config", lambda: (mock_client, "optillm"))
    monkeypatch.setitem(server.server_config, "approach", "moa")

    response = server.app.test_client().post(
        "/v1/messages",
        json={
            "model": "claude-test",
            "max_tokens": 32,
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "messages": [{"role": "user", "content": "Read README.md"}],
        },
    )

    body = response.get_json()
    assert response.status_code == 200
    assert mock_client.last_request["model"] == "claude-test"
    assert body["stop_reason"] == "tool_use"
    assert body["content"][0]["type"] == "tool_use"
