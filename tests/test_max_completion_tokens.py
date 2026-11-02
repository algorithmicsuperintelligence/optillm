#!/usr/bin/env python3
"""
Regression tests for issue #350: pass-through requests must carry the
max_completion_tokens the client sent and not add a max_tokens copy.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optillm.server as server
from optillm.inference import InferenceClient
from optillm.plugins import proxy_plugin
from optillm.plugins.proxy.config import ProxyConfig

COMPLETION = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "o4-mini",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "8"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
MESSAGES = [{"role": "user", "content": "What is 3 + 5?"}]


class TestNonePassthrough(unittest.TestCase):
    def post(self, **body):
        upstream = MagicMock()
        upstream.chat.completions.create.return_value = COMPLETION
        with patch.object(server, "get_config", return_value=(upstream, "key")):
            resp = server.app.test_client().post(
                "/v1/chat/completions", json={"model": "none-o4-mini", "messages": MESSAGES, **body})
        self.assertEqual(resp.status_code, 200)
        return upstream.chat.completions.create.call_args.kwargs

    def test_max_completion_tokens_is_not_copied_upstream(self):
        kwargs = self.post(max_completion_tokens=1024)
        self.assertEqual(kwargs["max_completion_tokens"], 1024)
        self.assertNotIn("max_tokens", kwargs)

    def test_max_completion_tokens_wins_when_both_are_sent(self):
        kwargs = self.post(max_tokens=10, max_completion_tokens=1024)
        self.assertEqual(kwargs["max_completion_tokens"], 1024)
        self.assertNotIn("max_tokens", kwargs)

    def test_max_tokens_alone_is_forwarded(self):
        kwargs = self.post(max_tokens=10)
        self.assertEqual(kwargs["max_tokens"], 10)
        self.assertNotIn("max_completion_tokens", kwargs)


class TestApproachesStillSeeTheLimit(unittest.TestCase):
    def test_non_passthrough_approach_receives_max_tokens(self):
        """The copy stays in request_config for approaches that read max_tokens."""
        upstream = MagicMock()
        seen = {}

        def fake_approach(approach, system_prompt, initial_query, client, model, request_config, *args, **kwargs):
            seen.update(request_config)
            return "8", 1

        with patch.object(server, "get_config", return_value=(upstream, "key")), \
                patch.object(server, "execute_single_approach", side_effect=fake_approach):
            resp = server.app.test_client().post("/v1/chat/completions", json={
                "model": "moa-o4-mini", "messages": MESSAGES, "max_completion_tokens": 1024})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(seen["max_tokens"], 1024)


class TestProxyPassthrough(unittest.TestCase):
    CONFIG = {"max_completion_tokens": 1024, "max_tokens": 1024, "stream": False, "n": 1}

    def run_plugin(self, **load_patch):
        upstream = MagicMock()
        upstream.chat.completions.create.return_value = COMPLETION
        with patch.object(ProxyConfig, "load", **load_patch):
            proxy_plugin.run("system", "hi", upstream, "o4-mini", dict(self.CONFIG), MESSAGES)
        return upstream.chat.completions.create.call_args.kwargs

    def assert_limit_only(self, kwargs):
        self.assertEqual(kwargs["max_completion_tokens"], 1024)
        self.assertNotIn("max_tokens", kwargs)
        self.assertNotIn("stream", kwargs)

    def test_no_providers_route(self):
        self.assert_limit_only(self.run_plugin(return_value={"providers": []}))

    def test_error_fallback_route(self):
        self.assert_limit_only(self.run_plugin(side_effect=RuntimeError("bad config")))

    def test_provider_route(self):
        config = {"providers": [{"name": "p"}], "routing": {}, "timeouts": {}, "queue": {}}
        proxy_client = MagicMock()
        proxy_client.chat.completions.create.return_value = COMPLETION
        proxy_plugin._proxy_client_cache.clear()
        self.addCleanup(proxy_plugin._proxy_client_cache.clear)
        with patch.object(ProxyConfig, "load", return_value=config), \
                patch.object(proxy_plugin, "ProxyClient", return_value=proxy_client):
            proxy_plugin.run("system", "hi", MagicMock(), "o4-mini", dict(self.CONFIG), MESSAGES)
        self.assert_limit_only(proxy_client.chat.completions.create.call_args.kwargs)


class TestLocalInference(unittest.TestCase):
    """The local client only used to read max_tokens, so it must honor the preferred name."""

    def generated_limit(self, **limits):
        client = InferenceClient.__new__(InferenceClient)
        pipeline = MagicMock()
        pipeline.generate.return_value = (["8"], [1], [None])
        pipeline.tokenizer.encode.return_value = [1, 2, 3]
        client.get_pipeline = MagicMock(return_value=pipeline)
        completions = InferenceClient.Chat.Completions(client)
        with patch("optillm.inference.count_reasoning_tokens", return_value=0):
            completions.create(messages=MESSAGES, model="m", **limits)
        return pipeline.generate.call_args.kwargs["generation_params"]["max_new_tokens"]

    def test_max_completion_tokens_limits_generation(self):
        self.assertEqual(self.generated_limit(max_completion_tokens=7), 7)

    def test_max_completion_tokens_wins_over_max_tokens(self):
        self.assertEqual(self.generated_limit(max_completion_tokens=7, max_tokens=99), 7)

    def test_max_tokens_still_limits_generation(self):
        self.assertEqual(self.generated_limit(max_tokens=99), 99)


if __name__ == "__main__":
    unittest.main()
