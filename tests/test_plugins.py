#!/usr/bin/env python3
"""
Test plugin functionality
"""

import sys
import os
import importlib

# Try to import pytest, but don't fail if it's not available
try:
    import pytest
except ImportError:
    pytest = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm import plugin_approaches, load_plugins


def test_plugin_module_imports():
    """Test that plugin modules can be imported"""
    plugin_modules = [
        'optillm.plugins.memory_plugin',
        'optillm.plugins.readurls_plugin',
        'optillm.plugins.privacy_plugin',
        'optillm.plugins.genselect_plugin',
        'optillm.plugins.majority_voting_plugin',
        'optillm.plugins.web_search_plugin',
        'optillm.plugins.deep_research_plugin',
        'optillm.plugins.deepthink_plugin',
        'optillm.plugins.longcepo_plugin',
        'optillm.plugins.spl_plugin',
        'optillm.plugins.proxy_plugin',
        'optillm.plugins.mcp_plugin',
        'optillm.plugins.compact_plugin'
    ]
    
    for module_name in plugin_modules:
        try:
            module = importlib.import_module(module_name)
            assert hasattr(module, 'run'), f"{module_name} missing 'run' function"
            assert hasattr(module, 'SLUG'), f"{module_name} missing 'SLUG' attribute"
        except ImportError as e:
            if pytest:
                pytest.fail(f"Failed to import {module_name}: {e}")
            else:
                raise AssertionError(f"Failed to import {module_name}: {e}")


def test_plugin_approach_detection():
    """Test plugin approach detection after loading"""
    # Load plugins first
    load_plugins()
    
    # Check if known plugins are loaded
    expected_plugins = ["memory", "readurls", "privacy", "web_search", "deep_research", "deepthink", "longcepo", "spl", "proxy", "mcp", "compact"]
    for plugin_name in expected_plugins:
        assert plugin_name in plugin_approaches, f"Plugin {plugin_name} not loaded"


def test_memory_plugin_structure():
    """Test memory plugin has required structure"""
    import optillm.plugins.memory_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == "memory"
    assert hasattr(plugin, 'Memory')  # Check for Memory class


def test_memory_plugin_persistence():
    """Test opt-in file-backed persistence for the memory plugin (issue #111).

    Covers the round trip, graceful degradation on a missing/corrupt file, and
    that the default (no persist_path) never touches the filesystem.
    """
    import json
    import tempfile
    from optillm.plugins.memory_plugin import Memory

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "memory.json")

        # Round trip: items written by one instance load into the next.
        m1 = Memory(persist_path=path)
        m1.add("alpha")
        m1.add("beta")
        assert os.path.exists(path), "add() should persist when persist_path is set"

        m2 = Memory(persist_path=path)
        assert m2.items == ["alpha", "beta"], "persisted items should load on init"

        # A missing file is a valid first run, not an error.
        missing = os.path.join(tmp, "does_not_exist.json")
        m3 = Memory(persist_path=missing)
        assert m3.items == []

        # A corrupt file degrades to an empty store without raising.
        corrupt = os.path.join(tmp, "corrupt.json")
        with open(corrupt, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        m4 = Memory(persist_path=corrupt)
        assert m4.items == []

        # A non-list JSON payload is ignored rather than trusted.
        wrong_shape = os.path.join(tmp, "wrong.json")
        with open(wrong_shape, "w", encoding="utf-8") as f:
            json.dump({"items": ["x"]}, f)
        m5 = Memory(persist_path=wrong_shape)
        assert m5.items == []

        # max_size is honoured on load (most recent items win).
        big = os.path.join(tmp, "big.json")
        with open(big, "w", encoding="utf-8") as f:
            json.dump([str(i) for i in range(10)], f)
        m6 = Memory(max_size=3, persist_path=big)
        assert m6.items == ["7", "8", "9"]

        # Default behaviour is unchanged: no persist_path means no file I/O.
        no_persist = os.path.join(tmp, "should_not_be_created.json")
        m7 = Memory()
        m7.add("gamma")
        assert not os.path.exists(no_persist)
        assert m7.persist_path is None


def test_genselect_plugin():
    """Test genselect plugin module"""
    import optillm.plugins.genselect_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'DEFAULT_NUM_CANDIDATES')
    assert plugin.SLUG == "genselect"


def test_majority_voting_plugin():
    """Test majority voting plugin module"""
    import optillm.plugins.majority_voting_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'extract_final_answer')
    assert hasattr(plugin, 'normalize_response')
    assert plugin.SLUG == "majority_voting"


def test_web_search_plugin():
    """Test web search plugin module"""
    import optillm.plugins.web_search_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'GoogleSearcher')
    assert hasattr(plugin, 'extract_search_queries')
    assert plugin.SLUG == "web_search"


def test_deep_research_plugin():
    """Test deep research plugin module"""
    import optillm.plugins.deep_research_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'DeepResearcher')
    assert plugin.SLUG == "deep_research"


def test_deepthink_plugin_imports():
    """Test deepthink plugin and its submodules can be imported"""
    # Test main plugin
    import optillm.plugins.deepthink_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == "deepthink"
    
    # Test submodules can be imported
    from optillm.plugins.deepthink import SelfDiscover, UncertaintyRoutedCoT
    assert SelfDiscover is not None
    assert UncertaintyRoutedCoT is not None


def test_longcepo_plugin():
    """Test longcepo plugin module"""
    import optillm.plugins.longcepo_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == "longcepo"
    
    # Test submodule can be imported
    from optillm.plugins.longcepo import run_longcepo
    assert run_longcepo is not None


def test_spl_plugin():
    """Test spl plugin module"""
    import optillm.plugins.spl_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == "spl"
    
    # Test submodule can be imported
    from optillm.plugins.spl import run_spl
    assert run_spl is not None


def test_proxy_plugin():
    """Test proxy plugin module"""
    import optillm.plugins.proxy_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == "proxy"
    
    # Test proxy submodules can be imported
    from optillm.plugins.proxy import client, config, approach_handler
    assert client is not None
    assert config is not None
    assert approach_handler is not None


def test_proxy_plugin_token_counts():
    """Test that proxy plugin returns complete token usage information"""
    import optillm.plugins.proxy_plugin as plugin
    from unittest.mock import Mock, MagicMock
    
    # Create a mock client with a mock response that has all token counts
    mock_client = Mock()
    mock_response = MagicMock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_response.usage = Mock(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15
    )
    mock_response.model_dump.return_value = {
        'choices': [{'message': {'content': 'Test response'}}],
        'usage': {
            'prompt_tokens': 10,
            'completion_tokens': 5,
            'total_tokens': 15
        }
    }
    mock_client.chat.completions.create.return_value = mock_response
    
    # Run the proxy plugin
    result, _ = plugin.run(
        system_prompt="Test system",
        initial_query="Test query",
        client=mock_client,
        model="test-model"
    )
    
    # Verify the result contains all token counts
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 'usage' in result, "Result should contain usage information"
    assert 'prompt_tokens' in result['usage'], "Usage should contain prompt_tokens"
    assert 'completion_tokens' in result['usage'], "Usage should contain completion_tokens"
    assert 'total_tokens' in result['usage'], "Usage should contain total_tokens"
    assert result['usage']['prompt_tokens'] == 10
    assert result['usage']['completion_tokens'] == 5
    assert result['usage']['total_tokens'] == 15


def test_proxy_plugin_timeout_config():
    """Test that proxy plugin properly configures timeout settings"""
    from optillm.plugins.proxy.config import ProxyConfig
    import tempfile
    import yaml
    
    # Create test config with timeout settings
    config = {
        "providers": [
            {
                "name": "test_provider",
                "base_url": "http://localhost:8000/v1",
                "api_key": "test-key"
            }
        ],
        "timeouts": {
            "request": 10,
            "connect": 3
        },
        "queue": {
            "max_concurrent": 50,
            "timeout": 30
        }
    }
    
    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # Load config with force_reload to bypass any cached config
        loaded_config = ProxyConfig.load(config_path, force_reload=True)

        assert 'timeouts' in loaded_config, "Config should contain timeouts section"
        assert loaded_config['timeouts'].get('request') == 10, "Request timeout should be 10"
        assert loaded_config['timeouts'].get('connect') == 3, "Connect timeout should be 3"

        assert 'queue' in loaded_config, "Config should contain queue section"
        assert loaded_config['queue']['max_concurrent'] == 50, "Max concurrent should be 50"
        assert loaded_config['queue']['timeout'] == 30, "Queue timeout should be 30"

    finally:
        import os
        os.unlink(config_path)
        # Clear the cache to avoid affecting other tests
        ProxyConfig._cached_config = None


def test_proxy_plugin_timeout_handling():
    """Test that proxy plugin handles timeouts correctly"""
    from optillm.plugins.proxy.client import ProxyClient
    from unittest.mock import Mock, patch
    import concurrent.futures

    # Create config with short timeout
    config = {
        "providers": [
            {
                "name": "slow_provider",
                "base_url": "http://localhost:8001/v1",
                "api_key": "test-key-1"
            },
            {
                "name": "fast_provider",
                "base_url": "http://localhost:8002/v1",
                "api_key": "test-key-2"
            }
        ],
        "routing": {
            "strategy": "round_robin",
            "health_check": {"enabled": False}
        },
        "timeouts": {
            "request": 2,
            "connect": 1
        },
        "queue": {
            "max_concurrent": 10,
            "timeout": 5
        }
    }

    # Create proxy client
    proxy_client = ProxyClient(config)

    # Verify timeout settings are loaded
    assert proxy_client.request_timeout == 2, "Request timeout should be 2"
    assert proxy_client.connect_timeout == 1, "Connect timeout should be 1"
    assert proxy_client.max_concurrent_requests == 10, "Max concurrent should be 10"
    assert proxy_client.queue_timeout == 5, "Queue timeout should be 5"


def test_mcp_plugin():
    """Test MCP plugin module"""
    import optillm.plugins.mcp_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'ServerConfig')
    assert hasattr(plugin, 'MCPServer')
    assert hasattr(plugin, 'execute_tool')
    assert plugin.SLUG == "mcp"


def test_compact_plugin():
    """Test compact plugin module"""
    import optillm.plugins.compact_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == "compact"


def test_plugin_subdirectory_imports():
    """Test all plugins with subdirectories can import their submodules"""
    # Test deep_research
    from optillm.plugins.deep_research import DeepResearcher
    assert DeepResearcher is not None
    
    # Test deepthink
    from optillm.plugins.deepthink import SelfDiscover, UncertaintyRoutedCoT
    assert SelfDiscover is not None
    assert UncertaintyRoutedCoT is not None
    
    # Test longcepo
    from optillm.plugins.longcepo import run_longcepo
    assert run_longcepo is not None
    
    # Test spl
    from optillm.plugins.spl import run_spl
    assert run_spl is not None
    
    # Test proxy
    from optillm.plugins.proxy import client, config, approach_handler
    assert client is not None
    assert config is not None
    assert approach_handler is not None


def test_no_relative_import_errors():
    """Test that plugins load without relative import errors"""
    import importlib
    import sys
    
    plugins_with_subdirs = [
        'optillm.plugins.deepthink_plugin',
        'optillm.plugins.deep_research_plugin',
        'optillm.plugins.longcepo_plugin',
        'optillm.plugins.spl_plugin',
        'optillm.plugins.proxy_plugin'
    ]
    
    for plugin_name in plugins_with_subdirs:
        # Clear any previously loaded modules to test fresh import
        modules_to_clear = [k for k in sys.modules.keys() if k.startswith(plugin_name)]
        for mod in modules_to_clear:
            del sys.modules[mod]
        
        try:
            module = importlib.import_module(plugin_name)
            # Try to access the run function to ensure full initialization works
            assert hasattr(module, 'run'), f"{plugin_name} missing run function"
        except ImportError as e:
            if "attempted relative import" in str(e):
                if pytest:
                    pytest.fail(f"Relative import error in {plugin_name}: {e}")
                else:
                    raise AssertionError(f"Relative import error in {plugin_name}: {e}")
            else:
                raise


if __name__ == "__main__":
    print("Running plugin tests...")
    
    try:
        test_plugin_module_imports()
        print("✅ Plugin module imports test passed")
    except Exception as e:
        print(f"❌ Plugin module imports test failed: {e}")
    
    try:
        test_plugin_approach_detection()
        print("✅ Plugin approach detection test passed")
    except Exception as e:
        print(f"❌ Plugin approach detection test failed: {e}")
    
    try:
        test_memory_plugin_structure()
        print("✅ Memory plugin structure test passed")
    except Exception as e:
        print(f"❌ Memory plugin structure test failed: {e}")

    try:
        test_memory_plugin_persistence()
        print("✅ Memory plugin persistence test passed")
    except Exception as e:
        print(f"❌ Memory plugin persistence test failed: {e}")

    try:
        test_genselect_plugin()
        print("✅ GenSelect plugin test passed")
    except Exception as e:
        print(f"❌ GenSelect plugin test failed: {e}")
    
    try:
        test_majority_voting_plugin()
        print("✅ Majority voting plugin test passed")
    except Exception as e:
        print(f"❌ Majority voting plugin test failed: {e}")
    
    try:
        test_web_search_plugin()
        print("✅ Web search plugin test passed")
    except Exception as e:
        print(f"❌ Web search plugin test failed: {e}")
    
    try:
        test_deep_research_plugin()
        print("✅ Deep research plugin test passed")
    except Exception as e:
        print(f"❌ Deep research plugin test failed: {e}")
    
    try:
        test_deepthink_plugin_imports()
        print("✅ Deepthink plugin imports test passed")
    except Exception as e:
        print(f"❌ Deepthink plugin imports test failed: {e}")
    
    try:
        test_longcepo_plugin()
        print("✅ LongCePO plugin test passed")
    except Exception as e:
        print(f"❌ LongCePO plugin test failed: {e}")
    
    try:
        test_spl_plugin()
        print("✅ SPL plugin test passed")
    except Exception as e:
        print(f"❌ SPL plugin test failed: {e}")
    
    try:
        test_proxy_plugin()
        print("✅ Proxy plugin test passed")
    except Exception as e:
        print(f"❌ Proxy plugin test failed: {e}")
    
    try:
        test_proxy_plugin_token_counts()
        print("✅ Proxy plugin token counts test passed")
    except Exception as e:
        print(f"❌ Proxy plugin token counts test failed: {e}")
    
    try:
        test_proxy_plugin_timeout_config()
        print("✅ Proxy plugin timeout config test passed")
    except Exception as e:
        print(f"❌ Proxy plugin timeout config test failed: {e}")
    
    try:
        test_proxy_plugin_timeout_handling()
        print("✅ Proxy plugin timeout handling test passed")
    except Exception as e:
        print(f"❌ Proxy plugin timeout handling test failed: {e}")

    try:
        test_mcp_plugin()
        print("✅ MCP plugin test passed")
    except Exception as e:
        print(f"❌ MCP plugin test failed: {e}")
    
    try:
        test_plugin_subdirectory_imports()
        print("✅ Plugin subdirectory imports test passed")
    except Exception as e:
        print(f"❌ Plugin subdirectory imports test failed: {e}")
    
    try:
        test_no_relative_import_errors()
        print("✅ No relative import errors test passed")
    except Exception as e:
        print(f"❌ Relative import errors test failed: {e}")
    
    print("\nDone!")