#!/usr/bin/env python3
"""
Simplified approach tests for CI/CD
Tests the basic structure of approaches without requiring actual model inference
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm.mcts import chat_with_mcts
from optillm.bon import best_of_n_sampling
from optillm.moa import mixture_of_agents
from optillm.self_consistency import advanced_self_consistency_approach
from optillm.reread import re2_approach
from optillm.cot_reflection import cot_reflection
from optillm.plansearch import plansearch
from optillm.leap import leap
from optillm.rto import round_trip_optimization
from optillm.mars import multi_agent_reasoning_system


class MockClient:
    """Mock OpenAI client for testing"""
    def __init__(self):
        self.chat = self.Chat()
    
    class Chat:
        def __init__(self):
            self.completions = self.Completions()
        
        class Completions:
            def create(self, **kwargs):
                class MockChoice:
                    class Message:
                        content = "Test response: 2 + 2 = 4"
                    message = Message()
                    finish_reason = "stop"

                class MockUsage:
                    completion_tokens = 10
                    total_tokens = 20

                class MockResponse:
                    choices = [MockChoice()]
                    usage = MockUsage()

                return MockResponse()


class MockBadClient:
    """Mock client that simulates degraded provider responses.

    mode:
      - "empty_choices": response with an empty choices list
      - "none_content":  a choice whose message.content is None
      - "length":        a choice truncated with finish_reason == "length"
    """
    def __init__(self, mode):
        self.chat = self.Chat(mode)

    class Chat:
        def __init__(self, mode):
            self.completions = self.Completions(mode)

        class Completions:
            def __init__(self, mode):
                self.mode = mode

            def create(self, **kwargs):
                mode = self.mode

                class MockUsage:
                    completion_tokens = 0
                    total_tokens = 0

                if mode == "empty_choices":
                    class MockResponse:
                        choices = []
                        usage = MockUsage()
                    return MockResponse()

                class MockMessage:
                    content = None if mode == "none_content" else "partial output"

                class MockChoice:
                    message = MockMessage()
                    finish_reason = "length" if mode == "length" else "stop"

                class MockResponse:
                    choices = [MockChoice()]
                    usage = MockUsage()

                return MockResponse()


def test_approach_imports():
    """Test that all approaches can be imported"""
    approaches = [
        chat_with_mcts,
        best_of_n_sampling,
        mixture_of_agents,
        advanced_self_consistency_approach,
        re2_approach,
        cot_reflection,
        plansearch,
        leap,
        multi_agent_reasoning_system
    ]
    
    for approach in approaches:
        assert callable(approach), f"{approach.__name__} is not callable"
    
    print("✅ All approaches imported successfully")


def test_basic_approach_calls():
    """Test basic approach calls with mock client"""
    client = MockClient()
    system_prompt = "You are a helpful assistant."
    query = "What is 2 + 2?"
    model = "mock-model"
    
    # Test approaches that should work with mock client
    simple_approaches = [
        ("re2_approach", re2_approach),
        ("cot_reflection", cot_reflection),
        ("leap", leap),
        ("mars", multi_agent_reasoning_system),
    ]
    
    for name, approach_func in simple_approaches:
        try:
            result = approach_func(system_prompt, query, client, model)
            assert result is not None, f"{name} returned None"
            assert isinstance(result, tuple), f"{name} should return a tuple"
            assert len(result) == 2, f"{name} should return (response, tokens)"
            print(f"✅ {name} basic test passed")
        except Exception as e:
            print(f"❌ {name} basic test failed: {e}")


def test_approach_parameters():
    """Test that approaches handle parameters correctly"""
    # Test that approaches accept the expected parameters
    import inspect
    
    approaches = {
        "chat_with_mcts": chat_with_mcts,
        "best_of_n_sampling": best_of_n_sampling,
        "mixture_of_agents": mixture_of_agents,
        "advanced_self_consistency_approach": advanced_self_consistency_approach,
        "re2_approach": re2_approach,
        "cot_reflection": cot_reflection,
        "plansearch": plansearch,
        "leap": leap,
        "multi_agent_reasoning_system": multi_agent_reasoning_system,
    }
    
    for name, func in approaches.items():
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        # Check required parameters
        required_params = ["system_prompt", "initial_query", "client", "model"]
        for param in required_params:
            assert param in params, f"{name} missing required parameter: {param}"
        
        print(f"✅ {name} has correct parameters")


def test_approaches_handle_bad_responses():
    """Approaches must degrade gracefully on empty / None / length-truncated
    provider responses instead of raising raw IndexError/TypeError/AttributeError.

    Without the response-validation guards these calls crash with an
    uncaught IndexError (empty choices) or TypeError/AttributeError (None content
    flowing into extract_output / SequenceMatcher / .strip()).
    """
    system_prompt = "You are a helpful assistant."
    query = "What is 2 + 2?"
    model = "mock-model"

    uncontrolled = (IndexError, TypeError, AttributeError)
    approaches = [
        ("round_trip_optimization", round_trip_optimization),
        ("advanced_self_consistency_approach", advanced_self_consistency_approach),
        ("re2_approach", re2_approach),
        ("leap", leap),
    ]

    for mode in ("empty_choices", "none_content", "length"):
        client = MockBadClient(mode)
        for name, func in approaches:
            try:
                result = func(system_prompt, query, client, model)
                # Graceful degradation: a normal (content, tokens) tuple.
                assert isinstance(result, tuple) and len(result) == 2, \
                    f"{name} ({mode}) returned unexpected {result!r}"
            except uncontrolled as e:
                raise AssertionError(
                    f"{name} ({mode}) raised uncontrolled {type(e).__name__}: {e}")
            except Exception:
                # A controlled, informative error is acceptable hardening behaviour.
                pass

    print("✅ approaches handle bad responses gracefully")


def test_mcts_params_are_request_scoped():
    """MCTS params must be read from the per-request config, not the shared
    global ``server_config`` (issue #304).

    Before the fix, ``proxy`` wrote the request's ``mcts_*`` params into the
    module-level ``server_config`` and ``execute_single_approach`` read them
    back out of it. Under Flask's threaded mode (or gunicorn/uWSGI), two
    concurrent requests race on that global: request B overwrites request A's
    params between A's write and A's read, silently corrupting A's MCTS run.

    This pins the fix: a per-request ``mcts_*`` value must flow through to
    ``chat_with_mcts`` untouched, and the shared global must not be mutated as a
    side effect of running a request.
    """
    import optillm.server as server

    captured = {}

    def fake_chat_with_mcts(system_prompt, initial_query, client, model,
                            num_simulations, exploration_weight, depth,
                            request_config=None, request_id=None):
        captured['num_simulations'] = num_simulations
        captured['exploration_weight'] = exploration_weight
        captured['depth'] = depth
        return "mock mcts response", 0

    original_mcts = server.chat_with_mcts
    saved_defaults = {k: server.server_config[k]
                      for k in ('mcts_simulations', 'mcts_exploration', 'mcts_depth')}
    try:
        server.chat_with_mcts = fake_chat_with_mcts
        # Pin the global defaults to values distinct from the per-request ones,
        # so a read from the global is observably different from a per-request read.
        server.server_config['mcts_simulations'] = 2
        server.server_config['mcts_exploration'] = 0.2
        server.server_config['mcts_depth'] = 1

        # Case 1: a per-request override must reach chat_with_mcts verbatim.
        per_request = {'mcts_simulations': 99, 'mcts_exploration': 0.9, 'mcts_depth': 7}
        server.execute_single_approach(
            'mcts', "You are a helpful assistant.", "What is 2 + 2?",
            MockClient(), "mock-model",
            request_config=dict(per_request), request_id="req-304")

        assert captured.get('num_simulations') == 99, (
            f"expected per-request mcts_simulations=99, got {captured.get('num_simulations')} "
            "— params were read from the shared global instead of request_config (issue #304)")
        assert captured.get('exploration_weight') == 0.9, (
            f"expected per-request mcts_exploration=0.9, got {captured.get('exploration_weight')}")
        assert captured.get('depth') == 7, (
            f"expected per-request mcts_depth=7, got {captured.get('depth')}")

        # Running a request must not mutate the shared global — that mutation is
        # exactly what let concurrent requests corrupt each other.
        assert server.server_config['mcts_simulations'] == 2, (
            "execute_single_approach must not write per-request MCTS params back "
            "into the global server_config (issue #304)")

        # Case 2: with no per-request override (e.g. a user who set --simulations
        # globally via the CLI), the server defaults must still be used.
        captured.clear()
        server.execute_single_approach(
            'mcts', "You are a helpful assistant.", "What is 2 + 2?",
            MockClient(), "mock-model",
            request_config={}, request_id="req-304b")

        assert captured.get('num_simulations') == 2, (
            f"expected the server-default mcts_simulations=2 when no per-request "
            f"value is sent, got {captured.get('num_simulations')}")
        assert captured.get('exploration_weight') == 0.2, (
            f"expected the server-default mcts_exploration=0.2, got {captured.get('exploration_weight')}")
        assert captured.get('depth') == 1, (
            f"expected the server-default mcts_depth=1, got {captured.get('depth')}")
    finally:
        server.chat_with_mcts = original_mcts
        server.server_config.update(saved_defaults)

    print("✅ mcts params are request-scoped (issue #304)")


def test_proxy_does_not_mutate_global_mcts_config():
    """End-to-end guard for the #304 fix at the proxy layer.

    Drives a real request through the Flask ``/v1/chat/completions`` handler with
    per-request MCTS params and asserts the shared global ``server_config`` is
    NOT mutated as a side effect (the mutation is what let concurrent requests
    corrupt each other), while the per-request values still reach ``chat_with_mcts``.
    """
    import optillm.server as server

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

    captured = {}

    def fake_chat_with_mcts(system_prompt, initial_query, client, model,
                            num_simulations, exploration_weight, depth,
                            request_config=None, request_id=None):
        captured['num_simulations'] = num_simulations
        captured['exploration_weight'] = exploration_weight
        captured['depth'] = depth
        return "mock mcts response", 5

    original_get_config = server.get_config
    original_mcts = server.chat_with_mcts
    saved_defaults = {k: server.server_config[k]
                      for k in ('mcts_simulations', 'mcts_exploration', 'mcts_depth')}
    try:
        server.get_config = lambda: (_FakeClient(), 'optillm')
        server.chat_with_mcts = fake_chat_with_mcts
        server.server_config['mcts_simulations'] = 2
        server.server_config['mcts_exploration'] = 0.2
        server.server_config['mcts_depth'] = 1

        server.app.config['TESTING'] = True
        client = server.app.test_client()
        resp = client.post(
            '/v1/chat/completions',
            json={"model": "gpt-4o-mini",
                  "messages": [{"role": "user", "content": "hello"}],
                  "optillm_approach": "mcts",
                  "mcts_simulations": 99, "mcts_exploration": 0.9, "mcts_depth": 7},
            headers={"Authorization": "Bearer optillm"})

        assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.get_data(as_text=True)[:200]}"

        # Per-request params must have reached the MCTS approach.
        assert captured.get('num_simulations') == 99, (
            f"per-request mcts_simulations should reach chat_with_mcts, got {captured.get('num_simulations')}")
        assert captured.get('depth') == 7, (
            f"per-request mcts_depth should reach chat_with_mcts, got {captured.get('depth')}")

        # ...but the shared global must be untouched (issue #304). On the buggy
        # code proxy() wrote these back into server_config, so this would be 99/0.9/7.
        assert server.server_config['mcts_simulations'] == 2, (
            f"proxy must not write per-request mcts_simulations into the global "
            f"server_config, found {server.server_config['mcts_simulations']} (issue #304)")
        assert server.server_config['mcts_exploration'] == 0.2, (
            f"proxy leaked mcts_exploration into the global: {server.server_config['mcts_exploration']}")
        assert server.server_config['mcts_depth'] == 1, (
            f"proxy leaked mcts_depth into the global: {server.server_config['mcts_depth']}")
    finally:
        server.get_config = original_get_config
        server.chat_with_mcts = original_mcts
        server.server_config.update(saved_defaults)

    print("✅ proxy does not mutate global mcts config (issue #304)")


if __name__ == "__main__":
    print("Running approach tests...")

    test_approach_imports()
    test_basic_approach_calls()
    test_approach_parameters()
    test_approaches_handle_bad_responses()
    test_mcts_params_are_request_scoped()
    test_proxy_does_not_mutate_global_mcts_config()

    print("\nAll tests completed!")
