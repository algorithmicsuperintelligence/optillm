#!/usr/bin/env python3
"""
Tests for the router plugin's model caching.

The router plugin classifies each incoming query to pick an optimization
approach. Loading its classifier is expensive and one-time (a ~400M-param
ModernBERT-large deserialize + a HuggingFace Hub round-trip + tokenizer build),
so it must be loaded once and reused across requests rather than reloaded on
every call. These tests pin that behavior down without downloading the real
model by mocking the underlying loaders.
"""

import os
import sys
import threading

# Try to import pytest, but don't fail if it's not available (matches the
# repo convention in tests/test_plugins.py so this runs under CI both ways).
try:
    import pytest
except ImportError:
    pytest = None

from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm.plugins import router_plugin


class _FakeConfig:
    hidden_size = 768


def _install_loader_mocks(stack, from_pretrained_counter):
    """Patch the heavy loaders so load_optillm_model() runs without any network
    or real model weights. `from_pretrained_counter` counts base-model loads."""

    def fake_from_pretrained(*args, **kwargs):
        from_pretrained_counter.append(1)
        base = mock.MagicMock()
        base.config = _FakeConfig()
        return base

    stack.enter_context(
        mock.patch.object(router_plugin.AutoModel, "from_pretrained",
                           side_effect=fake_from_pretrained)
    )
    stack.enter_context(
        mock.patch.object(router_plugin, "hf_hub_download", return_value="/tmp/fake.safetensors")
    )
    stack.enter_context(
        mock.patch.object(router_plugin, "load_model", return_value=None)
    )
    stack.enter_context(
        mock.patch.object(router_plugin.AutoTokenizer, "from_pretrained",
                           return_value=mock.MagicMock())
    )


def _reset_cache():
    router_plugin._model_cache = None


def test_repeated_calls_load_model_once():
    """Calling load_optillm_model() N times must load the base model exactly once."""
    _reset_cache()
    loads = []
    import contextlib
    with contextlib.ExitStack() as stack:
        _install_loader_mocks(stack, loads)

        first = router_plugin.load_optillm_model()
        for _ in range(5):
            again = router_plugin.load_optillm_model()
            # Same cached bundle object returned every time.
            assert again is first, "load_optillm_model() must return the cached bundle"

    assert len(loads) == 1, (
        f"expected exactly 1 base-model load across 6 calls, got {len(loads)} "
        "(model is being reloaded per request instead of cached)"
    )
    _reset_cache()


def test_returns_expected_bundle_shape():
    """The cached value is the (model, tokenizer, device) triple callers expect."""
    _reset_cache()
    import contextlib
    loads = []
    with contextlib.ExitStack() as stack:
        _install_loader_mocks(stack, loads)
        bundle = router_plugin.load_optillm_model()

    assert isinstance(bundle, tuple) and len(bundle) == 3, "expected a 3-tuple bundle"
    model, tokenizer, device = bundle
    assert isinstance(model, router_plugin.OptILMClassifier)
    assert tokenizer is not None
    assert device is not None
    _reset_cache()


def test_concurrent_calls_load_model_once():
    """Under concurrent first-time access the expensive load must run only once."""
    _reset_cache()
    import contextlib
    loads = []
    results = []
    with contextlib.ExitStack() as stack:
        _install_loader_mocks(stack, loads)

        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()  # maximize contention on the first load
            results.append(router_plugin.load_optillm_model())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(loads) == 1, (
        f"expected exactly 1 load under concurrency, got {len(loads)} "
        "(the double-checked lock is not serializing the first load)"
    )
    # Every thread observed the same cached bundle.
    assert all(r is results[0] for r in results), "threads saw different cached bundles"
    _reset_cache()


if __name__ == "__main__":
    for name, fn in [
        ("repeated calls load model once", test_repeated_calls_load_model_once),
        ("returns expected bundle shape", test_returns_expected_bundle_shape),
        ("concurrent calls load model once", test_concurrent_calls_load_model_once),
    ]:
        try:
            fn()
            print(f"✅ {name}")
        except Exception as e:
            print(f"❌ {name}: {e}")
            raise

    print("\nDone!")
