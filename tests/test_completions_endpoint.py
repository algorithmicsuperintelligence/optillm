"""
Offline unit tests for the legacy /v1/completions endpoint (issue #170).

Uses the Flask test client (no real server port, no network, no model) and
mocks the approach function (chat_with_mcts) following the pattern in
tests/test_approaches.py. Also asserts that /v1/chat/completions keeps its
exact historical response shape.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch

from optillm import server


MOCK_TEXT = "legacy completion text"
CAPTURED = {}


def fake_chat_with_mcts(system_prompt, initial_query, client, model,
                        num_simulations, exploration_weight, depth,
                        request_config=None, request_id=None):
    CAPTURED['model'] = model
    CAPTURED['request_config'] = request_config
    return MOCK_TEXT, 5


def fake_chat_with_mcts_boom(*args, **kwargs):
    raise RuntimeError("approach exploded")


class _FakeResp:
    def model_dump(self):
        return {"choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class _FakeClient:
    def __init__(self):
        self.chat = type("_Chat", (), {
            "completions": type("_Completions", (), {"create": lambda self, **k: _FakeResp()})()
        })()


class CompletionsEndpointTestBase(unittest.TestCase):
    """Shared fixture: mock client + approach, Flask test client."""

    def setUp(self):
        CAPTURED.clear()
        self._originals = {
            'get_config': server.get_config,
            'chat_with_mcts': server.chat_with_mcts,
            'config': dict(server.server_config),
        }
        server.get_config = lambda: (_FakeClient(), 'optillm')
        server.chat_with_mcts = fake_chat_with_mcts
        server.server_config['optillm_api_key'] = ''
        server.app.config['TESTING'] = True
        self.client = server.app.test_client()

    def tearDown(self):
        server.get_config = self._originals['get_config']
        server.chat_with_mcts = self._originals['chat_with_mcts']
        server.server_config.clear()
        server.server_config.update(self._originals['config'])


class TestLegacyCompletionsEndpoint(CompletionsEndpointTestBase):

    def test_basic_text_completion_shape(self):
        resp = self.client.post(
            '/v1/completions',
            json={"model": "gpt-4o-mini",
                  "prompt": "Say hello",
                  "max_tokens": 32,
                  "optillm_approach": "mcts"},
            headers={"Authorization": "Bearer optillm"})

        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        body = resp.get_json()

        # OpenAI legacy completion shape
        self.assertEqual(body["object"], "text_completion")
        self.assertTrue(body["id"].startswith("cmpl-"), body["id"])
        self.assertIsInstance(body["created"], int)
        self.assertEqual(body["model"], "gpt-4o-mini")
        self.assertEqual(len(body["choices"]), 1)
        choice = body["choices"][0]
        self.assertEqual(choice["text"], MOCK_TEXT)
        self.assertEqual(choice["index"], 0)
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertIsNone(choice["logprobs"])

        # usage reports the approach's completion token count
        self.assertEqual(body["usage"]["completion_tokens"], 5)
        self.assertIn("reasoning_tokens", body["usage"]["completion_tokens_details"])

        # optillm_approach routes to the mocked mcts approach; the slug prefix
        # is consumed by parse_combined_approach, so the approach itself sees
        # the bare model name (same dispatch behavior as /v1/chat/completions)
        self.assertEqual(CAPTURED.get("model"), "gpt-4o-mini")

        # request_config carries the token limit through to the approach
        self.assertEqual(CAPTURED["request_config"].get("max_tokens"), 32)

    def test_multiple_prompts_expand_to_multiple_choices(self):
        resp = self.client.post(
            '/v1/completions',
            json={"model": "gpt-4o-mini",
                  "prompt": ["first prompt", "second prompt"],
                  "optillm_approach": "mcts"})

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(len(body["choices"]), 2)
        self.assertEqual([c["text"] for c in body["choices"]], [MOCK_TEXT, MOCK_TEXT])
        self.assertEqual([c["index"] for c in body["choices"]], [0, 1])

    def test_n_expansion_multiple_choices(self):
        resp = self.client.post(
            '/v1/completions',
            json={"model": "gpt-4o-mini",
                  "prompt": "one prompt",
                  "n": 2,
                  "optillm_approach": "mcts"})

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(len(body["choices"]), 2)
        # completion tokens accumulate across the n attempts (2 x 5)
        self.assertEqual(body["usage"]["completion_tokens"], 10)

    def test_stream_not_supported(self):
        resp = self.client.post(
            '/v1/completions',
            json={"model": "gpt-4o-mini",
                  "prompt": "hi",
                  "stream": True,
                  "optillm_approach": "mcts"})

        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_prompt_required(self):
        resp = self.client.post(
            '/v1/completions',
            json={"model": "gpt-4o-mini", "optillm_approach": "mcts"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("prompt", resp.get_json()["error"])

    def test_prompt_type_validation(self):
        resp = self.client.post(
            '/v1/completions',
            json={"model": "gpt-4o-mini", "prompt": [], "optillm_approach": "mcts"})
        self.assertEqual(resp.status_code, 400)

        resp = self.client.post(
            '/v1/completions',
            json={"model": "gpt-4o-mini", "prompt": [1, 2], "optillm_approach": "mcts"})
        self.assertEqual(resp.status_code, 400)

    def test_approach_error_maps_to_500(self):
        server.chat_with_mcts = fake_chat_with_mcts_boom
        resp = self.client.post(
            '/v1/completions',
            json={"model": "gpt-4o-mini",
                  "prompt": "hi",
                  "optillm_approach": "mcts"})
        self.assertEqual(resp.status_code, 500)
        self.assertIn("approach exploded", resp.get_json()["error"])


class TestChatEndpointUnchanged(CompletionsEndpointTestBase):
    """The /v1/chat/completions shape must not change (regression guard)."""

    def test_chat_shape_unchanged(self):
        resp = self.client.post(
            '/v1/chat/completions',
            json={"model": "gpt-4o-mini",
                  "messages": [{"role": "user", "content": "hello"}],
                  "optillm_approach": "mcts"})

        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        body = resp.get_json()
        self.assertNotIn("object", body)
        self.assertNotIn("id", body)
        self.assertEqual(body["model"], "gpt-4o-mini")
        self.assertEqual(len(body["choices"]), 1)
        choice = body["choices"][0]
        self.assertNotIn("text", choice)
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(choice["message"]["content"], MOCK_TEXT)
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(body["usage"]["completion_tokens"], 5)
        self.assertEqual(CAPTURED.get("model"), "gpt-4o-mini")


if __name__ == '__main__':
    unittest.main()
