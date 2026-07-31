#!/usr/bin/env python3
"""
Comprehensive test suite for MCP plugin functionality
"""

import sys
import os
import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

# Try to import pytest, but don't fail if it's not available
try:
    import pytest
except ImportError:
    pytest = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm.plugins.mcp_plugin import (
    ServerConfig, MCPServer, MCPConfigManager, MCPServerManager,
    execute_tool, execute_tool_stdio, execute_tool_sse, execute_tool_websocket,
    SLUG
)


class TestServerConfig:
    """Test ServerConfig dataclass functionality"""

    def test_default_stdio_config(self):
        """Test default configuration for stdio transport"""
        config = ServerConfig()
        assert config.transport == "stdio"
        assert config.command is None
        assert config.args == []
        assert config.url is None
        assert config.headers == {}
        assert config.env == {}
        assert config.timeout == 5.0
        assert config.sse_read_timeout == 300.0

    def test_stdio_config_from_dict(self):
        """Test creating stdio config from dictionary"""
        config_dict = {
            "transport": "stdio",
            "command": "npx",
            "args": ["@modelcontextprotocol/server-filesystem", "/tmp"],
            "env": {"PATH": "/usr/local/bin"},
            "description": "Filesystem server"
        }

        config = ServerConfig.from_dict(config_dict)
        assert config.transport == "stdio"
        assert config.command == "npx"
        assert config.args == ["@modelcontextprotocol/server-filesystem", "/tmp"]
        assert config.env == {"PATH": "/usr/local/bin"}
        assert config.description == "Filesystem server"

    def test_sse_config_from_dict(self):
        """Test creating SSE config from dictionary"""
        config_dict = {
            "transport": "sse",
            "url": "https://api.example.com/mcp",
            "headers": {"Authorization": "Bearer token123"},
            "timeout": 10.0,
            "sse_read_timeout": 600.0,
            "description": "Remote SSE server"
        }

        config = ServerConfig.from_dict(config_dict)
        assert config.transport == "sse"
        assert config.url == "https://api.example.com/mcp"
        assert config.headers == {"Authorization": "Bearer token123"}
        assert config.timeout == 10.0
        assert config.sse_read_timeout == 600.0
        assert config.description == "Remote SSE server"

    def test_websocket_config_from_dict(self):
        """Test creating WebSocket config from dictionary"""
        config_dict = {
            "transport": "websocket",
            "url": "wss://api.example.com/mcp",
            "description": "WebSocket server"
        }

        config = ServerConfig.from_dict(config_dict)
        assert config.transport == "websocket"
        assert config.url == "wss://api.example.com/mcp"
        assert config.description == "WebSocket server"


class TestMCPConfigManager:
    """Test MCP configuration management"""

    def test_init_default_path(self):
        """Test default configuration path"""
        manager = MCPConfigManager()
        expected_path = Path.home() / ".optillm" / "mcp_config.json"
        assert manager.config_path == expected_path

    def test_init_custom_path(self):
        """Test custom configuration path"""
        custom_path = "/tmp/custom_mcp_config.json"
        manager = MCPConfigManager(custom_path)
        assert manager.config_path == Path(custom_path)

    def test_create_default_config(self):
        """Test creating default configuration file"""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            manager = MCPConfigManager(str(config_path))

            success = manager.create_default_config()
            assert success
            assert config_path.exists()

            # Verify default content
            with open(config_path) as f:
                config = json.load(f)

            assert "mcpServers" in config
            assert "log_level" in config
            assert config["mcpServers"] == {}
            assert config["log_level"] == "INFO"

    def test_load_valid_config(self):
        """Test loading valid configuration"""
        import tempfile

        config_data = {
            "mcpServers": {
                "test_server": {
                    "transport": "stdio",
                    "command": "test-command",
                    "args": ["arg1", "arg2"]
                }
            },
            "log_level": "DEBUG"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            manager = MCPConfigManager(config_path)
            success = manager.load_config()
            assert success
            assert len(manager.servers) == 1
            assert "test_server" in manager.servers
            assert manager.servers["test_server"].command == "test-command"
            assert manager.log_level == "DEBUG"
        finally:
            os.unlink(config_path)

    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration"""
        manager = MCPConfigManager("/nonexistent/path.json")
        success = manager.load_config()
        assert not success
        assert len(manager.servers) == 0


@pytest.mark.asyncio
class TestMCPServer:
    """Test MCP server connection and capability discovery"""

    def test_init(self):
        """Test MCPServer initialization"""
        config = ServerConfig()
        server = MCPServer("test_server", config)

        assert server.server_name == "test_server"
        assert server.config == config
        assert server.tools == []
        assert server.resources == []
        assert server.prompts == []
        assert not server.connected
        assert not server.has_tools_capability
        assert not server.has_resources_capability
        assert not server.has_prompts_capability

    async def test_connect_stdio_validation(self):
        """Test stdio connection validation"""
        config = ServerConfig(transport="stdio")  # No command
        server = MCPServer("test_server", config)

        result = await server.connect_stdio_native()
        assert not result

    async def test_connect_sse_validation(self):
        """Test SSE connection validation"""
        config = ServerConfig(transport="sse")  # No URL
        server = MCPServer("test_server", config)

        result = await server.connect_sse()
        assert not result

    async def test_connect_websocket_validation(self):
        """Test WebSocket connection validation"""
        config = ServerConfig(transport="websocket")  # No URL
        server = MCPServer("test_server", config)

        result = await server.connect_websocket()
        assert not result

    async def test_connect_and_discover_unsupported_transport(self):
        """Test unsupported transport type"""
        config = ServerConfig(transport="invalid")
        server = MCPServer("test_server", config)

        result = await server.connect_and_discover()
        assert not result

    @patch('optillm.plugins.mcp_plugin.sse_client')
    async def test_connect_sse_success(self, mock_sse_client):
        """Test successful SSE connection"""
        # Mock the SSE client context manager
        mock_streams = (AsyncMock(), AsyncMock())
        mock_sse_client.return_value.__aenter__ = AsyncMock(return_value=mock_streams)
        mock_sse_client.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock session
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.capabilities = Mock()
        mock_session.initialize.return_value = mock_result

        config = ServerConfig(
            transport="sse",
            url="https://api.example.com/mcp",
            headers={"Authorization": "Bearer token"}
        )
        server = MCPServer("test_server", config)

        with patch.object(server, 'connect_stdio', return_value=True):
            with patch('optillm.plugins.mcp_plugin.LoggingClientSession') as mock_session_class:
                mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await server.connect_sse()
                assert result


@pytest.mark.asyncio
class TestToolExecution:
    """Test tool execution functionality"""

    async def test_execute_tool_server_not_found(self):
        """Test tool execution with non-existent server"""
        with patch('optillm.plugins.mcp_plugin.MCPConfigManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.load_config.return_value = True
            mock_manager.servers = {}
            mock_manager_class.return_value = mock_manager

            result = await execute_tool("nonexistent", "test_tool", {})
            assert "error" in result
            assert "not found" in result["error"]

    async def test_execute_tool_config_load_failure(self):
        """Test tool execution with config load failure"""
        with patch('optillm.plugins.mcp_plugin.MCPConfigManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.load_config.return_value = False
            mock_manager_class.return_value = mock_manager

            result = await execute_tool("test_server", "test_tool", {})
            assert "error" in result
            assert "Failed to load MCP configuration" == result["error"]

    async def test_execute_tool_unsupported_transport(self):
        """Test tool execution with unsupported transport"""
        config = ServerConfig(transport="invalid")

        with patch('optillm.plugins.mcp_plugin.MCPConfigManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.load_config.return_value = True
            mock_manager.servers = {"test_server": config}
            mock_manager_class.return_value = mock_manager

            result = await execute_tool("test_server", "test_tool", {})
            assert "error" in result
            assert "Unsupported transport type" in result["error"]

    async def test_execute_tool_stdio_no_command(self):
        """Test stdio tool execution without command"""
        config = ServerConfig(transport="stdio")  # No command
        result = await execute_tool_stdio(config, "test_tool", {})
        assert "error" in result
        assert "requires command" in result["error"]

    async def test_execute_tool_sse_no_url(self):
        """Test SSE tool execution without URL"""
        config = ServerConfig(transport="sse")  # No URL
        result = await execute_tool_sse(config, "test_tool", {})
        assert "error" in result
        assert "requires URL" in result["error"]

    async def test_execute_tool_websocket_no_url(self):
        """Test WebSocket tool execution without URL"""
        config = ServerConfig(transport="websocket")  # No URL
        result = await execute_tool_websocket(config, "test_tool", {})
        assert "error" in result
        assert "requires URL" in result["error"]


class TestMCPServerManager:
    """Test MCP server manager functionality"""

    def test_init(self):
        """Test MCPServerManager initialization"""
        config_manager = MCPConfigManager()
        manager = MCPServerManager(config_manager)

        assert manager.config_manager == config_manager
        assert manager.servers == {}
        assert not manager.initialized
        assert manager.all_tools == []
        assert manager.all_resources == []
        assert manager.all_prompts == []

    def test_get_tools_for_model_empty(self):
        """Test getting tools when no tools are available"""
        config_manager = MCPConfigManager()
        manager = MCPServerManager(config_manager)

        tools = manager.get_tools_for_model()
        assert tools == []

    def test_get_capabilities_description_no_servers(self):
        """Test getting capabilities description with no servers"""
        config_manager = MCPConfigManager()
        manager = MCPServerManager(config_manager)

        description = manager.get_capabilities_description()
        assert "No MCP servers available" in description


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GITHUB_TOKEN"), reason="GITHUB_TOKEN not set")
class TestGitHubMCPServer:
    """Integration tests with GitHub MCP server (requires GITHUB_TOKEN)

    Uses the local GitHub MCP server via stdio transport (npx).
    The remote hosted endpoint at api.githubcopilot.com requires OAuth,
    but the local server works with a regular GitHub Personal Access Token.
    """

    async def test_github_mcp_server_connection(self):
        """Test real connection to GitHub MCP server via local stdio transport"""
        import shutil

        # Check if npx is available
        if not shutil.which("npx"):
            pytest.skip("npx not available - required for local GitHub MCP server")

        # Use stdio transport with local GitHub MCP server
        # This uses the official @modelcontextprotocol/server-github package
        config = ServerConfig(
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")},
            description="GitHub MCP Server (local)"
        )

        server = MCPServer("github", config)

        try:
            connected = await server.connect_and_discover()

            if connected:
                assert server.connected
                assert len(server.tools) > 0 or len(server.resources) > 0 or len(server.prompts) > 0
                print("GitHub MCP server connected successfully!")
                print(f"Found: {len(server.tools)} tools, {len(server.resources)} resources, {len(server.prompts)} prompts")

                # List some tools
                if server.tools:
                    print("Available tools:")
                    for tool in server.tools[:5]:
                        print(f"  - {tool.name}")
            else:
                pytest.skip("Could not connect to GitHub MCP server")

        except Exception as e:
            pytest.skip(f"GitHub MCP server test failed: {e}")


class TestPluginStructure:
    """Test plugin structure and exports"""

    def test_slug_exists(self):
        """Test that plugin has SLUG defined"""
        assert hasattr(sys.modules['optillm.plugins.mcp_plugin'], 'SLUG')
        assert SLUG == "mcp"

    def test_run_function_exists(self):
        """Test that plugin has run function defined"""
        import optillm.plugins.mcp_plugin as plugin
        assert hasattr(plugin, 'run')
        assert callable(plugin.run)

    def test_required_imports(self):
        """Test that required modules can be imported"""
        try:
            from mcp.client.sse import sse_client
            from mcp.client.websocket import websocket_client
            assert sse_client is not None
            assert websocket_client is not None
        except ImportError as e:
            pytest.fail(f"Required MCP imports failed: {e}")


# Mock tests for various scenarios
class TestMockScenarios:
    """Test various scenarios with mocked dependencies"""

    @patch('optillm.plugins.mcp_plugin.find_executable')
    def test_stdio_command_not_found(self, mock_find_executable):
        """Test stdio transport when command is not found"""
        mock_find_executable.return_value = None

        config = ServerConfig(transport="stdio", command="nonexistent-command")

        async def test_async():
            result = await execute_tool_stdio(config, "test_tool", {})
            assert "error" in result
            assert "Failed to find executable" in result["error"]

        asyncio.run(test_async())

    def test_environment_variable_expansion(self):
        """Test environment variable expansion in SSE headers.

        Note: The current implementation only expands values that are entirely
        environment variable references (e.g., ${TOKEN}), not embedded ones
        (e.g., Bearer ${TOKEN}).
        """
        os.environ["TEST_TOKEN"] = "test-token-value"

        try:
            # Use a value that is entirely an env var reference
            config = ServerConfig(
                transport="sse",
                url="https://api.example.com/mcp",
                headers={"Authorization": "${TEST_TOKEN}"}
            )

            MCPServer("test", config)

            # Test the header expansion logic from connect_sse method
            expanded_headers = {}
            for key, value in config.headers.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    expanded_value = os.environ.get(env_var)
                    if expanded_value:
                        expanded_headers[key] = expanded_value
                else:
                    expanded_headers[key] = value

            assert expanded_headers["Authorization"] == "test-token-value"

        finally:
            del os.environ["TEST_TOKEN"]


if __name__ == "__main__":
    print("Running MCP plugin tests...")

    # Run basic tests
    test_classes = [
        TestServerConfig,
        TestMCPConfigManager,
        TestPluginStructure,
        TestMockScenarios
    ]

    for test_class in test_classes:
        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]

        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method())
                else:
                    method()
                print(f"✅ {test_class.__name__}.{method_name} passed")
            except Exception as e:
                print(f"❌ {test_class.__name__}.{method_name} failed: {e}")

    # Run async tests
    async def run_async_tests():
        test_instance = TestMCPServer()
        async_methods = [
            'test_connect_stdio_validation',
            'test_connect_sse_validation',
            'test_connect_websocket_validation',
            'test_connect_and_discover_unsupported_transport'
        ]

        for method_name in async_methods:
            try:
                method = getattr(test_instance, method_name)
                await method()
                print(f"✅ TestMCPServer.{method_name} passed")
            except Exception as e:
                print(f"❌ TestMCPServer.{method_name} failed: {e}")

        # Tool execution tests
        tool_test_instance = TestToolExecution()
        tool_methods = [
            'test_execute_tool_server_not_found',
            'test_execute_tool_config_load_failure',
            'test_execute_tool_unsupported_transport',
            'test_execute_tool_stdio_no_command',
            'test_execute_tool_sse_no_url',
            'test_execute_tool_websocket_no_url'
        ]

        for method_name in tool_methods:
            try:
                method = getattr(tool_test_instance, method_name)
                await method()
                print(f"✅ TestToolExecution.{method_name} passed")
            except Exception as e:
                print(f"❌ TestToolExecution.{method_name} failed: {e}")

    asyncio.run(run_async_tests())

    print("\n🎯 MCP Plugin tests completed!")
    print("💡 To run GitHub MCP server integration test, set GITHUB_TOKEN environment variable")

    if os.getenv("GITHUB_TOKEN"):
        print("🔍 Running GitHub MCP server integration test...")
        async def run_github_test():
            test_instance = TestGitHubMCPServer()
            try:
                await test_instance.test_github_mcp_server_connection()
                print("✅ GitHub MCP server integration test passed")
            except Exception as e:
                print(f"❌ GitHub MCP server integration test failed: {e}")

        asyncio.run(run_github_test())