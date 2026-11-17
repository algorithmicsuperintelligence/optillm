"""Regression tests for configured fallback providers in the proxy plugin."""

import json
import unittest
from unittest.mock import MagicMock, patch

import httpx
from openai import OpenAI

from optillm.plugins.proxy.client import ProxyClient
from optillm.plugins.proxy.config import ProxyConfig


class TestConfiguredFallbackProviders(unittest.TestCase):
    def make_client(self, strategy="failover", primary_count=1, fallback_count=1):
        providers = [
            {
                "name": f"primary_{index}",
                "base_url": "http://localhost:8080/v1",
                "api_key": "none",
            }
            for index in range(primary_count)
        ] + [
            {
                "name": f"backup_{index}",
                "base_url": "http://localhost:8081/v1",
                "api_key": "none",
                "fallback_only": True,
                "max_concurrent": 1,
                "model_map": {"alias": f"backup-model-{index}"},
            }
            for index in range(fallback_count)
        ]
        config = ProxyConfig._validate_config(
            ProxyConfig._apply_defaults(
                {
                    "providers": providers,
                    "routing": {
                        "strategy": strategy,
                        "health_check": {"enabled": False},
                    },
                    "queue": {"max_concurrent": 1},
                }
            )
        )
        client = ProxyClient(config, fallback_client=MagicMock())
        for provider in client.providers:
            provider._client = MagicMock()
        return client

    def test_primary_failure_uses_configured_backup_for_each_strategy(self):
        for strategy in ("round_robin", "weighted", "failover"):
            for error in (
                RuntimeError("upstream failed"),
                TimeoutError("upstream timed out"),
            ):
                with self.subTest(strategy=strategy, error=type(error).__name__):
                    client = self.make_client(strategy=strategy)
                    primary = client.active_providers[0]
                    backup = client.fallback_providers[0]
                    primary._client.chat.completions.create.side_effect = error
                    backup._client.chat.completions.create.return_value = "from backup"
                    messages = [{"role": "user", "content": "hi"}]

                    result = client.chat.completions.create(
                        model="alias", messages=messages
                    )

                    self.assertEqual(result, "from backup")
                    backup._client.chat.completions.create.assert_called_once_with(
                        model="backup-model-0",
                        messages=messages,
                        timeout=client.request_timeout,
                    )
                    client.fallback_client.chat.completions.create.assert_not_called()
                    self.assertEqual(backup.available_slots(), 1)
                    self.assertTrue(client._request_semaphore.acquire(blocking=False))
                    client._request_semaphore.release()

    def test_primary_success_does_not_use_backup(self):
        client = self.make_client()
        primary = client.active_providers[0]
        primary._client.chat.completions.create.return_value = "from primary"

        self.assertEqual(client.chat.completions.create(model="alias"), "from primary")

        client.fallback_providers[0]._client.chat.completions.create.assert_not_called()
        client.fallback_client.chat.completions.create.assert_not_called()

    def test_http_503_fails_over_with_the_openai_sdk(self):
        client = self.make_client()
        calls = []

        def handle_primary(request):
            calls.append("primary")
            self.assertEqual(json.loads(request.content)["model"], "alias")
            return httpx.Response(503, json={"error": {"message": "unavailable"}})

        def handle_backup(request):
            calls.append("backup")
            self.assertEqual(json.loads(request.content)["model"], "backup-model-0")
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-backup",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "backup-model-0",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        with (
            httpx.Client(transport=httpx.MockTransport(handle_primary)) as primary_http,
            httpx.Client(transport=httpx.MockTransport(handle_backup)) as backup_http,
        ):
            client.active_providers[0]._client = OpenAI(
                api_key="test",
                base_url="https://primary.example/v1",
                http_client=primary_http,
                max_retries=0,
            )
            client.fallback_providers[0]._client = OpenAI(
                api_key="test",
                base_url="https://backup.example/v1",
                http_client=backup_http,
                max_retries=0,
            )

            response = client.chat.completions.create(
                model="alias", messages=[{"role": "user", "content": "hi"}]
            )

        self.assertEqual(response.choices[0].message.content, "ok")
        self.assertEqual(calls, ["primary", "backup"])
        client.fallback_client.chat.completions.create.assert_not_called()

    def test_all_primaries_are_tried_before_backups(self):
        client = self.make_client(primary_count=2, fallback_count=2)
        calls = MagicMock()
        for provider in client.providers:
            calls.attach_mock(provider._client.chat.completions.create, provider.name)
        for provider in client.providers[:-1]:
            provider._client.chat.completions.create.side_effect = RuntimeError(
                f"{provider.name} failed"
            )
        client.providers[-1]._client.chat.completions.create.return_value = "ok"

        self.assertEqual(client.chat.completions.create(model="alias"), "ok")
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            ["primary_0", "primary_1", "backup_0", "backup_1"],
        )
        client.fallback_client.chat.completions.create.assert_not_called()

    def test_primary_at_capacity_uses_backup(self):
        client = self.make_client()
        primary = client.active_providers[0]
        client.fallback_providers[
            0
        ]._client.chat.completions.create.return_value = "from backup"

        with (
            patch.object(primary, "acquire_slot", return_value=False),
            patch.object(primary, "release_slot") as release,
        ):
            self.assertEqual(
                client.chat.completions.create(model="alias"), "from backup"
            )
            primary._client.chat.completions.create.assert_not_called()
            release.assert_not_called()

    def test_default_client_is_last_resort(self):
        client = self.make_client()
        calls = MagicMock()
        for provider in client.providers:
            calls.attach_mock(provider._client.chat.completions.create, provider.name)
            provider._client.chat.completions.create.side_effect = RuntimeError(
                f"{provider.name} failed"
            )
        calls.attach_mock(client.fallback_client.chat.completions.create, "default")
        client.fallback_client.chat.completions.create.return_value = "ok"

        self.assertEqual(client.chat.completions.create(model="alias"), "ok")

        for provider in client.providers:
            provider._client.chat.completions.create.assert_called_once()
        self.assertEqual(
            [call[0] for call in calls.mock_calls], ["primary_0", "backup_0", "default"]
        )

    def test_unhealthy_primaries_use_backup_without_retrying_it(self):
        client = self.make_client()
        client.active_providers[0].is_healthy = False
        client.fallback_providers[
            0
        ]._client.chat.completions.create.side_effect = RuntimeError("backup failed")
        client.fallback_client.chat.completions.create.return_value = "from default"

        self.assertEqual(client.chat.completions.create(model="alias"), "from default")

        client.active_providers[0]._client.chat.completions.create.assert_not_called()
        client.fallback_providers[
            0
        ]._client.chat.completions.create.assert_called_once()

    def test_all_failures_report_backup_error_and_release_slots(self):
        client = self.make_client()
        client.fallback_client = None
        for provider in client.providers:
            provider._client.chat.completions.create.side_effect = RuntimeError(
                f"{provider.name} failed"
            )

        with self.assertRaisesRegex(Exception, "backup_0 failed"):
            client.chat.completions.create(model="alias")

        for provider in client.providers:
            provider._client.chat.completions.create.assert_called_once()
        self.assertEqual(client.fallback_providers[0].available_slots(), 1)
        self.assertTrue(client._request_semaphore.acquire(blocking=False))
        client._request_semaphore.release()


if __name__ == "__main__":
    unittest.main()
