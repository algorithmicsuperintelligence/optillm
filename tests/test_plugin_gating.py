#!/usr/bin/env python3
"""
Tests for the plugin gating that disables code-executing plugins (executecode)
by default and requires an explicit opt-in (issue #302).

All tests are offline: they exercise load_plugins() registration against the
packaged plugin directory and the pure helper functions of the executecode
plugin. No server, LLM client, or Jupyter kernel is started.
"""

import sys
import os

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm import plugin_approaches, load_plugins
from optillm.server import UNSAFE_DEFAULT_DISABLED_PLUGINS, server_config
from optillm.plugins.executecode_plugin import (
    extract_python_code,
    should_execute_request_code,
)

ENV_VAR = "OPTILLM_ENABLE_UNSAFE_PLUGINS"


@pytest.fixture(autouse=True)
def restore_default_plugins(monkeypatch):
    """Make every test start (and leave behind) the default plugin registry."""
    monkeypatch.setitem(server_config, 'enable_unsafe_plugins', '')
    monkeypatch.delenv(ENV_VAR, raising=False)
    yield
    load_plugins()


def test_unsafe_plugin_set_contains_executecode():
    assert "executecode" in UNSAFE_DEFAULT_DISABLED_PLUGINS


def test_executecode_disabled_by_default():
    load_plugins()
    assert "executecode" not in plugin_approaches, \
        "executecode executes arbitrary code and must be opt-in"
    # Gating must not disable unrelated plugins.
    assert "memory" in plugin_approaches
    assert "readurls" in plugin_approaches


def test_executecode_opt_in_via_server_config(monkeypatch):
    monkeypatch.setitem(server_config, 'enable_unsafe_plugins', 'executecode')
    load_plugins()
    assert "executecode" in plugin_approaches


def test_executecode_opt_in_via_env_var(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "executecode")
    load_plugins()
    assert "executecode" in plugin_approaches


def test_opt_in_accepts_comma_separated_list(monkeypatch):
    monkeypatch.setitem(server_config, 'enable_unsafe_plugins', 'memory, executecode ,')
    load_plugins()
    assert "executecode" in plugin_approaches
    assert "memory" in plugin_approaches


def test_disabled_plugin_logs_how_to_enable(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="optillm.server"):
        load_plugins()
    warnings = [r.message for r in caplog.records if "executecode" in r.message]
    assert warnings, "expected a warning explaining why executecode was skipped"
    assert "--enable-unsafe-plugins=executecode" in warnings[0]
    assert ENV_VAR in warnings[0]


def test_extract_python_code_multiple_blocks():
    text = "Before\n```python\nprint(1)\n```\nmiddle\n```python\nprint(2)\n```\nafter"
    assert extract_python_code(text) == ["print(1)", "print(2)"]


def test_extract_python_code_no_blocks():
    assert extract_python_code("no code here at all") == []


def test_extract_python_code_requires_python_tag():
    # A bare fenced block without the `python` tag must not be extracted.
    assert extract_python_code("```\nprint(1)\n```") == []


def test_should_execute_request_code_keyword_hits():
    for query in ["please run this", "execute it", "show the output", "give me the result"]:
        assert should_execute_request_code(query) is True, query


def test_should_execute_request_code_keyword_misses():
    assert should_execute_request_code("what is the capital of France?") is False
    assert should_execute_request_code("") is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
