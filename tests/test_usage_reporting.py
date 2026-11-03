#!/usr/bin/env python3
"""Unit tests for OpenAI-compatible usage reporting helpers."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optillm.server as server


def parse_sse(body: str):
    chunks = []
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[len("data: "):]))
    return chunks


class TestEstimateMessagesTokens(unittest.TestCase):
    def test_counts_string_and_text_parts(self):
        messages = [
            {"role": "system", "content": "abcd"},  # 4 chars -> 1 token
            {"role": "user", "content": [
                {"type": "text", "text": "efghijkl"},  # 8 chars
                {"type": "image_url", "image_url": {"url": " Ignored"}},
            ]},
        ]
        # Total text length 12 -> 3 tokens
        self.assertEqual(server.estimate_messages_tokens(messages), 3)

    def test_empty_messages(self):
        self.assertEqual(server.estimate_messages_tokens([]), 0)
        self.assertEqual(server.estimate_messages_tokens(None), 0)

    def test_short_non_empty_prompt_is_at_least_one_token(self):
        messages = [{"role": "user", "content": "Hi"}]
        self.assertEqual(server.estimate_messages_tokens(messages), 1)


class TestBuildUsage(unittest.TestCase):
    def test_keeps_provided_completion_tokens(self):
        messages = [{"role": "user", "content": "abcdefgh"}]  # 8 chars -> 2 prompt tokens
        usage = server.build_usage(
            messages,
            completion_tokens=42,
            reasoning_tokens=7,
        )
        self.assertEqual(usage["prompt_tokens"], 2)
        self.assertEqual(usage["completion_tokens"], 42)
        self.assertEqual(usage["total_tokens"], 44)
        self.assertEqual(usage["completion_tokens_details"]["reasoning_tokens"], 7)

    def test_estimates_completion_when_missing(self):
        messages = [{"role": "user", "content": "abcd"}]
        usage = server.build_usage(messages, final_content="efghijkl")  # 8 chars -> 2
        self.assertEqual(usage["prompt_tokens"], 1)
        self.assertEqual(usage["completion_tokens"], 2)
        self.assertEqual(usage["total_tokens"], 3)

    def test_estimates_when_completion_tokens_zero_with_content(self):
        messages = [{"role": "user", "content": "abcd"}]
        usage = server.build_usage(
            messages,
            completion_tokens=0,
            final_content="efghijkl",  # 8 chars -> 2
        )
        self.assertEqual(usage["completion_tokens"], 2)
        self.assertEqual(usage["total_tokens"], 3)

    def test_keeps_zero_completion_when_content_empty(self):
        messages = [{"role": "user", "content": "abcd"}]
        usage = server.build_usage(messages, completion_tokens=0, final_content="")
        self.assertEqual(usage["completion_tokens"], 0)


class TestEnsureUsage(unittest.TestCase):
    def test_fills_missing_prompt_tokens(self):
        messages = [{"role": "user", "content": "abcdefgh"}]
        result = {
            "choices": [{"message": {"role": "assistant", "content": "hi there!!"}}],
            "usage": {"completion_tokens": 5},
        }
        out = server.ensure_usage(result, messages)
        self.assertEqual(out["usage"]["prompt_tokens"], 2)
        self.assertEqual(out["usage"]["completion_tokens"], 5)
        self.assertEqual(out["usage"]["total_tokens"], 7)

    def test_does_not_clobber_complete_upstream_usage(self):
        messages = [{"role": "user", "content": "abcdefgh"}]
        result = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 99,
                "completion_tokens": 11,
                "total_tokens": 110,
            },
        }
        out = server.ensure_usage(result, messages)
        self.assertEqual(out["usage"]["prompt_tokens"], 99)
        self.assertEqual(out["usage"]["completion_tokens"], 11)
        self.assertIs(out, result)

    def test_preserves_prompt_tokens_zero(self):
        messages = [{"role": "user", "content": "abcdefgh"}]
        result = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 3,
                "total_tokens": 3,
            },
        }
        out = server.ensure_usage(result, messages)
        self.assertEqual(out["usage"]["prompt_tokens"], 0)
        self.assertEqual(out["usage"]["total_tokens"], 3)
        self.assertIs(out, result)

    def test_fills_missing_total_tokens_when_prompt_present(self):
        messages = [{"role": "user", "content": "abcdefgh"}]
        result = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
            },
        }
        out = server.ensure_usage(result, messages)
        self.assertEqual(out["usage"]["prompt_tokens"], 10)
        self.assertEqual(out["usage"]["completion_tokens"], 4)
        self.assertEqual(out["usage"]["total_tokens"], 14)
        self.assertIsNot(out, result)

    def test_preserves_extra_completion_tokens_details(self):
        messages = [{"role": "user", "content": "abcdefgh"}]
        result = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "completion_tokens": 5,
                "completion_tokens_details": {
                    "reasoning_tokens": 2,
                    "accepted_prediction_tokens": 1,
                },
            },
        }
        out = server.ensure_usage(result, messages)
        details = out["usage"]["completion_tokens_details"]
        self.assertEqual(details["reasoning_tokens"], 2)
        self.assertEqual(details["accepted_prediction_tokens"], 1)


class TestContentsFromCompletion(unittest.TestCase):
    def test_returns_string_for_multiple_choices(self):
        result = {
            "choices": [
                {"message": {"role": "assistant", "content": "one"}},
                {"message": {"role": "assistant", "content": "two"}},
            ]
        }
        content = server._contents_from_completion(result)
        self.assertIsInstance(content, str)
        self.assertIn("one", content)
        self.assertIn("two", content)


class TestGenerateStreamingResponseUsage(unittest.TestCase):
    def test_emits_usage_chunk_when_include_usage(self):
        usage = {
            "prompt_tokens": 2,
            "completion_tokens": 4,
            "total_tokens": 6,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
        body = "".join(server.generate_streaming_response(
            "hello", "test-model", usage=usage, include_usage=True
        ))
        chunks = parse_sse(body)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "hello")
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"], usage)

    def test_omits_usage_chunk_when_include_usage_false(self):
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        body = "".join(server.generate_streaming_response(
            "hello", "test-model", usage=usage, include_usage=False
        ))
        chunks = parse_sse(body)
        self.assertEqual(len(chunks), 1)
        self.assertNotIn("usage", chunks[0])

    def test_empty_list_response(self):
        body = "".join(server.generate_streaming_response([], "test-model"))
        chunks = parse_sse(body)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "")
        self.assertEqual(chunks[0]["choices"][0]["finish_reason"], "stop")


CAPTURED = {}
MOCK_TEXT = "usage reporting answer text"


def fake_chat_with_mcts(system_prompt, initial_query, client, model,
                        num_simulations, exploration_weight, depth,
                        request_config=None, request_id=None):
    CAPTURED["model"] = model
    return MOCK_TEXT, 5


class _FakeIncompleteUsageResp:
    def model_dump(self):
        return {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 3},
        }


class _FakeClient:
    def __init__(self, response=None):
        resp = response or _FakeIncompleteUsageResp()
        self.chat = type("_Chat", (), {
            "completions": type("_Completions", (), {
                "create": lambda self, **k: resp
            })()
        })()


class TestUsageEndpointWiring(unittest.TestCase):
    """HTTP-level checks that proxy/completions expose complete usage."""

    def setUp(self):
        CAPTURED.clear()
        self._originals = {
            "get_config": server.get_config,
            "chat_with_mcts": server.chat_with_mcts,
            "config": dict(server.server_config),
        }
        server.get_config = lambda: (_FakeClient(), "optillm")
        server.chat_with_mcts = fake_chat_with_mcts
        server.server_config["optillm_api_key"] = ""
        server.app.config["TESTING"] = True
        self.client = server.app.test_client()

    def tearDown(self):
        server.get_config = self._originals["get_config"]
        server.chat_with_mcts = self._originals["chat_with_mcts"]
        server.server_config.clear()
        server.server_config.update(self._originals["config"])

    def test_chat_approach_includes_prompt_and_total_tokens(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "optillm_approach": "mcts",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        usage = resp.get_json()["usage"]
        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertEqual(usage["completion_tokens"], 5)
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + 5)

    def test_chat_approach_stream_include_usage_emits_final_chunk(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "optillm_approach": "mcts",
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        chunks = parse_sse(resp.get_data(as_text=True))
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertIn("usage", chunks[-1])
        self.assertGreater(chunks[-1]["usage"]["prompt_tokens"], 0)
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 5)

    def test_chat_passthrough_fills_missing_prompt_tokens(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": [{"role": "user", "content": "abcdefgh"}],
                "optillm_approach": "none",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        usage = resp.get_json()["usage"]
        self.assertEqual(usage["prompt_tokens"], 2)
        self.assertEqual(usage["completion_tokens"], 3)
        self.assertEqual(usage["total_tokens"], 5)

    def test_completions_passthrough_fills_missing_prompt_tokens(self):
        resp = self.client.post(
            "/v1/completions",
            json={
                "model": "local-model",
                "prompt": "abcdefgh",
                "optillm_approach": "none",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        usage = resp.get_json()["usage"]
        self.assertEqual(usage["prompt_tokens"], 2)
        self.assertEqual(usage["completion_tokens"], 3)
        self.assertEqual(usage["total_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
