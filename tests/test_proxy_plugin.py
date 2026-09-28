#!/usr/bin/env python3
"""
Tests for the proxy plugin and agent-style (tool calling) passthrough.
Regression tests for issue #330.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optillm.server as server
from optillm.plugins.proxy.client import ProxyClient, list_provider_models
from optillm.plugins.proxy.config import ProxyConfig

TOOLS = [{"type": "function", "function": {"name": "ls", "parameters": {"type": "object", "properties": {}}}}]

AGENT_MESSAGES = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": [{"type": "text", "text": "list files"}]},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "a.py"},
]

TOOL_CALL_COMPLETION = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "local-model",
    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


def parse_sse(body):
    chunks = []
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[len("data: "):]))
    return chunks


class TestMessageHandling(unittest.TestCase):
    def test_parse_conversation_handles_tool_messages(self):
        system_prompt, initial_query, approach = server.parse_conversation(AGENT_MESSAGES)
        self.assertEqual(system_prompt, "You are an agent.")
        self.assertEqual(initial_query, "User: list files")
        self.assertIsNone(approach)

    def test_strip_optillm_approach_tags_keeps_structure(self):
        messages = [{"role": "user", "content": "<optillm_approach>proxy</optillm_approach> hi"}] + AGENT_MESSAGES[2:]
        stripped = server.strip_optillm_approach_tags(messages)
        self.assertEqual(stripped[0]["content"], "hi")
        self.assertEqual(stripped[1:], AGENT_MESSAGES[2:])
        # Original request is not mutated
        self.assertIn("<optillm_approach>", messages[0]["content"])

    def test_streaming_completion_keeps_tool_calls_and_usage(self):
        chunks = parse_sse("".join(server.generate_streaming_completion(TOOL_CALL_COMPLETION, "m", include_usage=True)))
        delta = chunks[0]["choices"][0]["delta"]
        self.assertEqual(delta["tool_calls"][0]["index"], 0)
        self.assertEqual(delta["tool_calls"][0]["function"]["name"], "ls")
        self.assertNotIn("content", delta)
        self.assertEqual(chunks[1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(chunks[2]["usage"]["total_tokens"], 5)


class TestChatCompletionsPassthrough(unittest.TestCase):
    def setUp(self):
        self.original_config = server.server_config.copy()
        self.app = server.app.test_client()
        self.upstream = MagicMock()
        self.upstream.chat.completions.create.return_value = TOOL_CALL_COMPLETION

    def tearDown(self):
        server.server_config.clear()
        server.server_config.update(self.original_config)

    def test_none_approach_forwards_original_messages(self):
        with patch.object(server, "get_config", return_value=(self.upstream, "key")):
            resp = self.app.post("/v1/chat/completions", json={
                "model": "local-model", "messages": AGENT_MESSAGES, "tools": TOOLS,
                "stream": True, "stream_options": {"include_usage": True}})

        kwargs = self.upstream.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["messages"][2]["tool_calls"], AGENT_MESSAGES[2]["tool_calls"])
        self.assertEqual(kwargs["messages"][3]["role"], "tool")
        self.assertEqual(kwargs["tools"], TOOLS)
        self.assertNotIn("stream", kwargs)
        self.assertNotIn("stream_options", kwargs)

        chunks = parse_sse(resp.get_data(as_text=True))
        self.assertEqual(chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "ls")
        self.assertEqual(chunks[1]["choices"][0]["finish_reason"], "tool_calls")


class TestProxyConfig(unittest.TestCase):
    def setUp(self):
        ProxyConfig._cached_config = None
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        ProxyConfig._cached_config = None
        self.tmp.cleanup()

    def test_does_not_fall_back_to_bundled_example(self):
        with patch.object(Path, "home", return_value=Path(self.tmp.name)):
            config = ProxyConfig.load()
        self.assertEqual(config["providers"], [])
        self.assertTrue((Path(self.tmp.name) / ".optillm" / "proxy_config.yaml").exists())


class TestProxyClientFailover(unittest.TestCase):
    def make_client(self, fallback):
        config = ProxyConfig._validate_config(ProxyConfig._apply_defaults({
            "providers": [{"name": "local", "base_url": "http://localhost:8080/v1", "api_key": "none"}],
            "routing": {"health_check": {"enabled": False}},
        }))
        client = ProxyClient(config, fallback_client=fallback)
        provider = client.providers[0]
        provider._client = MagicMock()
        return client, provider

    def test_client_error_does_not_mark_provider_unhealthy(self):
        client, provider = self.make_client(fallback=None)
        error = Exception("bad request")
        error.status_code = 400
        provider._client.chat.completions.create.side_effect = error
        with self.assertRaises(Exception):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        self.assertTrue(provider.is_healthy)

    def test_unhealthy_provider_is_retried_before_default_client(self):
        fallback = MagicMock()
        client, provider = self.make_client(fallback=fallback)
        provider.is_healthy = False
        provider._client.chat.completions.create.return_value = "from provider"
        result = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(result, "from provider")
        fallback.chat.completions.create.assert_not_called()


class TestProxyModels(unittest.TestCase):
    CONFIG = {"providers": [{"name": "local", "base_url": "http://localhost:8080/v1", "api_key": "none",
                             "model_map": {"alias": "local-model"}}]}

    def fake_provider_client(self):
        model = MagicMock()
        model.model_dump.return_value = {"id": "local-model", "object": "model", "created": 0, "owned_by": "llamacpp"}
        fake = MagicMock()
        fake.models.list.return_value.data = [model]
        return fake

    def test_list_provider_models_includes_aliases(self):
        with patch("optillm.plugins.proxy.client.Provider.client", new_callable=PropertyMock,
                   return_value=self.fake_provider_client()):
            models = list_provider_models(self.CONFIG)
        self.assertEqual([m["id"] for m in models], ["local-model", "alias"])

    def test_models_endpoint_uses_proxy_providers(self):
        original_config = server.server_config.copy()
        server.server_config.update({"approach": "proxy", "base_url": ""})
        try:
            with patch.object(server, "get_config", return_value=(MagicMock(), "key")), \
                 patch.object(ProxyConfig, "load", return_value=self.CONFIG), \
                 patch("optillm.plugins.proxy.client.list_provider_models", return_value=[{"id": "local-model"}]):
                resp = server.app.test_client().get("/v1/models")
        finally:
            server.server_config.clear()
            server.server_config.update(original_config)
        self.assertEqual(resp.get_json()["data"], [{"id": "local-model"}])


if __name__ == "__main__":
    unittest.main()
