"""
MCP Plugin for OptILLM

This plugin integrates the Model Context Protocol (MCP) with OptILLM,
allowing access to external tools, resources, and prompts through MCP servers.
"""

import os
import json
import logging
import asyncio
import shutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import traceback

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.websocket import websocket_client
from mcp.shared.exceptions import McpError

# Configure logging
LOG_DIR = Path.home() / ".optillm" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "mcp_plugin.log"

# Configure file logger with detailed formatting
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

# Configure console logger with simpler formatting
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))

# Set up the logger
logger = logging.getLogger("optillm.mcp_plugin")
logger.setLevel(logging.DEBUG)  # Set to DEBUG for maximum detail
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Plugin identifier
SLUG = "mcp"

# Add custom logging for MCP communication
def log_mcp_message(direction: str, method: str, params: Any = None, result: Any = None, error: Any = None):
    """Log MCP communication in detail"""
    message_parts = [f"MCP {direction} - Method: {method}"]
    
    if params:
        try:
            params_str = json.dumps(params, indent=2)
            message_parts.append(f"Params: {params_str}")
        except (TypeError, ValueError):
            message_parts.append(f"Params: {params}")
    
    if result:
        try:
            result_str = json.dumps(result, indent=2)
            message_parts.append(f"Result: {result_str}")
        except (TypeError, ValueError):
            message_parts.append(f"Result: {result}")
    
    if error:
        message_parts.append(f"Error: {error}")
    
    logger.debug("\n".join(message_parts))

def find_executable(cmd: str) -> Optional[str]:
    """
    Find the full path to an executable command.
    
    Args:
        cmd: The command to find
        
    Returns:
        Full path to the executable if found, None otherwise
    """
    # First check if it's already a full path
    if os.path.isfile(cmd) and os.access(cmd, os.X_OK):
        return cmd
        
    # Next check if it's in PATH
    cmd_path = shutil.which(cmd)
    if cmd_path:
        logger.info(f"Found {cmd} in PATH at {cmd_path}")
        return cmd_path
    
    # Try common locations
    common_paths = [
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/opt/homebrew/bin",
        os.path.expanduser("~/.npm-global/bin"),
        os.path.expanduser("~/.nvm/current/bin"),
    ]
    
    for path in common_paths:
        full_path = os.path.join(path, cmd)
        if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
            logger.info(f"Found {cmd} at {full_path}")
            return full_path
            
    logger.error(f"Could not find executable: {cmd}")
    return None

@dataclass
class ServerConfig:
    """Configuration for a single MCP server"""
    # Transport type: "stdio" (default), "sse", or "websocket"
    transport: str = "stdio"

    # For stdio transport
    command: Optional[str] = None
    args: List[str] = None

    # For remote transports (SSE/WebSocket)
    url: Optional[str] = None
    headers: Dict[str, str] = None

    # Common fields
    env: Dict[str, str] = None
    description: Optional[str] = None

    # Timeout settings
    timeout: float = 5.0
    sse_read_timeout: float = 300.0

    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.args is None:
            self.args = []
        if self.headers is None:
            self.headers = {}
        if self.env is None:
            self.env = {}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ServerConfig':
        """Create ServerConfig from a dictionary"""
        return cls(
            transport=config.get("transport", "stdio"),
            command=config.get("command"),
            args=config.get("args", []),
            url=config.get("url"),
            headers=config.get("headers", {}),
            env=config.get("env", {}),
            description=config.get("description"),
            timeout=config.get("timeout", 5.0),
            sse_read_timeout=config.get("sse_read_timeout", 300.0)
        )

class MCPConfigManager:
    """Manages MCP configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional custom config path"""
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path.home() / ".optillm" / "mcp_config.json"
        
        # Default configuration
        self.servers: Dict[str, ServerConfig] = {}
        self.log_level: str = "INFO"
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"MCP config file not found at {self.config_path}")
                return False
                
            with open(self.config_path, 'r') as f:
                config_data = f.read()
                logger.debug(f"Raw config data: {config_data}")
                config = json.loads(config_data)
            
            # Set log level
            self.log_level = config.get("log_level", "INFO")
            log_level = getattr(logging, self.log_level.upper(), logging.INFO)
            logger.setLevel(log_level)
            
            # Load server configurations
            servers_config = config.get("mcpServers", {})
            for server_name, server_config in servers_config.items():
                self.servers[server_name] = ServerConfig.from_dict(server_config)
                logger.debug(f"Loaded server config for {server_name}: {server_config}")
            
            logger.info(f"Loaded configuration with {len(self.servers)} servers")
            return True
            
        except Exception as e:
            logger.error(f"Error loading MCP configuration: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def create_default_config(self) -> bool:
        """Create a default configuration file if none exists"""
        try:
            if self.config_path.exists():
                return True
                
            default_config = {
                "mcpServers": {},
                "log_level": "INFO"
            }
            
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
                
            logger.info(f"Created default configuration at {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating default configuration: {e}")
            return False

# Create a custom ClientSession that logs all communication
class LoggingClientSession(ClientSession):
    """A ClientSession that logs all communication"""
    
    async def send_request(self, *args, **kwargs):
        """Log and forward requests"""
        method = args[0]
        params = args[1] if len(args) > 1 else None
        log_mcp_message("REQUEST", method, params)
        
        try:
            result = await super().send_request(*args, **kwargs)
            log_mcp_message("RESPONSE", method, result=result)
            return result
        except Exception as e:
            log_mcp_message("ERROR", method, error=str(e))
            raise
    
    async def send_notification(self, *args, **kwargs):
        """Log and forward notifications"""
        method = args[0]
        params = args[1] if len(args) > 1 else None
        log_mcp_message("NOTIFICATION", method, params)
        
        try:
            await super().send_notification(*args, **kwargs)
        except Exception as e:
            log_mcp_message("ERROR", method, error=str(e))
            raise

class MCPServer:
    """Represents a connection to an MCP server"""

    def __init__(self, server_name: str, config: ServerConfig):
        self.server_name = server_name
        self.config = config
        self.tools = []
        self.resources = []
        self.prompts = []
        self.connected = False
        self.has_tools_capability = False
        self.has_resources_capability = False
        self.has_prompts_capability = False

    async def connect_stdio(self, session: LoggingClientSession) -> bool:
        """Connect to server using stdio transport and discover capabilities"""
        try:
            logger.info(f"Connected to server: {self.server_name}")

            # Initialize session
            logger.debug(f"Initializing MCP session for {self.server_name}")
            result = await session.initialize()
            logger.info(f"Server {self.server_name} initialized with capabilities: {result.capabilities}")
            logger.debug(f"Full initialization result: {result}")

            # Check which capabilities the server supports
            server_capabilities = result.capabilities

            # Discover tools if supported
            if hasattr(server_capabilities, "tools"):
                self.has_tools_capability = True
                logger.info(f"Discovering tools for {self.server_name}")
                try:
                    tools_result = await session.list_tools()
                    self.tools = tools_result.tools
                    logger.info(f"Found {len(self.tools)} tools")
                    logger.debug(f"Tools details: {[t.name for t in self.tools]}")
                except McpError as e:
                    logger.warning(f"Failed to list tools: {e}")

            # Discover resources if supported
            if hasattr(server_capabilities, "resources"):
                self.has_resources_capability = True
                logger.info(f"Discovering resources for {self.server_name}")
                try:
                    resources_result = await session.list_resources()
                    self.resources = resources_result.resources
                    logger.info(f"Found {len(self.resources)} resources")
                    logger.debug(f"Resources details: {[r.uri for r in self.resources]}")
                except McpError as e:
                    logger.warning(f"Failed to list resources: {e}")

            # Discover prompts if supported
            if hasattr(server_capabilities, "prompts"):
                self.has_prompts_capability = True
                logger.info(f"Discovering prompts for {self.server_name}")
                try:
                    prompts_result = await session.list_prompts()
                    self.prompts = prompts_result.prompts
                    logger.info(f"Found {len(self.prompts)} prompts")
                    logger.debug(f"Prompts details: {[p.name for p in self.prompts]}")
                except McpError as e:
                    logger.warning(f"Failed to list prompts: {e}")

            logger.info(f"Server {self.server_name} capabilities: "
                       f"{len(self.tools)} tools, {len(self.resources)} resources, "
                       f"{len(self.prompts)} prompts")
            return True

        except Exception as e:
            logger.error(f"Error during stdio session: {e}")
            logger.error(traceback.format_exc())
            return False

    async def connect_sse(self) -> bool:
        """Connect to server using SSE transport and discover capabilities"""
        logger.info(f"Connecting to SSE server: {self.server_name}")
        logger.debug(f"SSE URL: {self.config.url}")
        logger.debug(f"Headers: {self.config.headers}")

        if not self.config.url:
            logger.error(f"SSE transport requires URL for server {self.server_name}")
            return False

        try:
            # Expand environment variables in headers
            expanded_headers = {}
            for key, value in self.config.headers.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    expanded_value = os.environ.get(env_var)
                    if expanded_value:
                        expanded_headers[key] = expanded_value
                    else:
                        logger.warning(f"Environment variable {env_var} not found for header {key}")
                else:
                    expanded_headers[key] = value

            async with sse_client(
                url=self.config.url,
                headers=expanded_headers,
                timeout=self.config.timeout,
                sse_read_timeout=self.config.sse_read_timeout
            ) as (read_stream, write_stream):
                async with LoggingClientSession(read_stream, write_stream) as session:
                    return await self.connect_stdio(session)

        except Exception as e:
            logger.error(f"Error connecting to SSE server {self.server_name}: {e}")
            logger.error(traceback.format_exc())
            return False

    async def connect_websocket(self) -> bool:
        """Connect to server using WebSocket transport and discover capabilities"""
        logger.info(f"Connecting to WebSocket server: {self.server_name}")
        logger.debug(f"WebSocket URL: {self.config.url}")

        if not self.config.url:
            logger.error(f"WebSocket transport requires URL for server {self.server_name}")
            return False

        try:
            async with websocket_client(self.config.url) as (read_stream, write_stream):
                async with LoggingClientSession(read_stream, write_stream) as session:
                    return await self.connect_stdio(session)

        except Exception as e:
            logger.error(f"Error connecting to WebSocket server {self.server_name}: {e}")
            logger.error(traceback.format_exc())
            return False

    async def connect_stdio_native(self) -> bool:
        """Connect using stdio transport with local executable"""
        logger.debug(f"Server configuration: {vars(self.config)}")

        # Validate stdio configuration
        if not self.config.command:
            logger.error(f"stdio transport requires command for server {self.server_name}")
            return False

        # Find the full path to the command
        full_command = find_executable(self.config.command)
        if not full_command:
            logger.error(f"Failed to find executable for command: {self.config.command}")
            return False

        # Create environment with PATH included
        merged_env = os.environ.copy()
        if self.config.env:
            merged_env.update(self.config.env)

        logger.debug(f"Using command: {full_command}")
        logger.debug(f"Arguments: {self.config.args}")
        logger.debug(f"Environment: {self.config.env}")

        # Create server parameters
        server_params = StdioServerParameters(
            command=full_command,
            args=self.config.args,
            env=merged_env
        )

        try:
            # Start the server separately to see its output
            process = await asyncio.create_subprocess_exec(
                full_command,
                *self.config.args,
                env=merged_env,
                stderr=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE
            )

            # Log startup message from stderr
            async def log_stderr():
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    stderr_text = line.decode().strip()
                    logger.info(f"Server {self.server_name} stderr: {stderr_text}")

            # Log stdout too for debugging
            async def log_stdout():
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    stdout_text = line.decode().strip()
                    logger.debug(f"Server {self.server_name} stdout: {stdout_text}")

            # Start logging tasks
            asyncio.create_task(log_stderr())
            asyncio.create_task(log_stdout())

            # Wait a bit for the server to start up
            logger.debug("Waiting for server to start up...")
            await asyncio.sleep(2)

            # Use the MCP client with proper context management
            logger.debug(f"Establishing MCP client connection to {self.server_name}")
            async with stdio_client(server_params) as (read_stream, write_stream):
                logger.debug("Connection established, creating session")
                # Use our logging session instead of the regular one
                async with LoggingClientSession(read_stream, write_stream) as session:
                    return await self.connect_stdio(session)

        except Exception as e:
            logger.error(f"Error connecting to MCP server {self.server_name}: {e}")
            logger.error(traceback.format_exc())
            return False

    async def connect_and_discover(self) -> bool:
        """Connect to the server and discover capabilities using appropriate transport"""
        logger.info(f"Connecting to MCP server: {self.server_name} using {self.config.transport} transport")

        # Route to appropriate transport method
        try:
            if self.config.transport == "stdio":
                success = await self.connect_stdio_native()
            elif self.config.transport == "sse":
                success = await self.connect_sse()
            elif self.config.transport == "websocket":
                success = await self.connect_websocket()
            else:
                logger.error(f"Unsupported transport type: {self.config.transport}")
                return False

            if success:
                self.connected = True
                logger.info(f"Successfully connected to {self.server_name} via {self.config.transport}")
            else:
                logger.error(f"Failed to connect to {self.server_name} via {self.config.transport}")

            return success

        except Exception as e:
            logger.error(f"Error connecting to MCP server {self.server_name}: {e}")
            logger.error(traceback.format_exc())
            return False

class MCPServerManager:
    """Manages MCP servers and capabilities"""
    
    def __init__(self, config_manager: MCPConfigManager):
        self.config_manager = config_manager
        self.servers: Dict[str, MCPServer] = {}
        self.initialized = False
        
        # Cache of capabilities
        self.all_tools = []
        self.all_resources = []
        self.all_prompts = []
    
    async def initialize(self) -> bool:
        """Initialize and cache all server capabilities"""
        if self.initialized:
            return True
            
        # Create servers
        for server_name, server_config in self.config_manager.servers.items():
            self.servers[server_name] = MCPServer(server_name, server_config)
        
        # Connect to all servers and discover capabilities
        connected_servers = 0
        for server_name, server in self.servers.items():
            success = await server.connect_and_discover()
            if success:
                connected_servers += 1
                # Cache server capabilities
                for tool in server.tools:
                    tool_info = {
                        "server": server_name,
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    }
                    self.all_tools.append(tool_info)
                    logger.debug(f"Cached tool: {tool_info}")
                
                for resource in server.resources:
                    resource_info = {
                        "server": server_name,
                        "uri": resource.uri,
                        "name": resource.name,
                        "description": resource.description
                    }
                    self.all_resources.append(resource_info)
                    logger.debug(f"Cached resource: {resource_info}")
                
                for prompt in server.prompts:
                    prompt_info = {
                        "server": server_name,
                        "name": prompt.name,
                        "description": prompt.description,
                        "arguments": prompt.arguments
                    }
                    self.all_prompts.append(prompt_info)
                    logger.debug(f"Cached prompt: {prompt_info}")
        
        self.initialized = True
        
        # Check if we successfully connected to any servers
        logger.info(f"Connected to {connected_servers}/{len(self.servers)} MCP servers")
        return connected_servers > 0
    
    def get_tools_for_model(self) -> List[Dict[str, Any]]:
        """Get tools in a format suitable for the model's tool-calling API"""
        tools = []
        
        for tool_info in self.all_tools:
            server_name = tool_info["server"]
            tool_name = tool_info["name"]
            
            # Format for model tools API
            tool_entry = {
                "type": "function",
                "function": {
                    "name": f"{server_name}.{tool_name}",
                    "description": tool_info["description"] or f"Tool {tool_name} from server {server_name}",
                    "parameters": tool_info["input_schema"]
                }
            }
            tools.append(tool_entry)
            logger.debug(f"Added tool for model: {tool_entry}")
        
        return tools
    
    def get_capabilities_description(self) -> str:
        """Get a description of all capabilities"""
        if not self.servers:
            return "No MCP servers available."
            
        description_parts = []
        
        for server_name, server in self.servers.items():
            if not server.connected:
                description_parts.append(f"## {server_name}\nServer connection failed or not established.\n")
                continue
                
            server_description = f"## {server_name}\n"
            
            if server.config.description:
                server_description += f"{server.config.description}\n\n"
                
            if server.tools:
                server_description += "### Tools\n"
                for tool in server.tools:
                    server_description += f"- {server_name}.{tool.name}: {tool.description or 'No description'}\n"
                server_description += "\n"
                
            if server.resources:
                server_description += "### Resources\n"
                for resource in server.resources:
                    server_description += f"- {resource.uri}: {resource.name or 'No name'} - {resource.description or 'No description'}\n"
                server_description += "\n"
                
            if server.prompts:
                server_description += "### Prompts\n"
                for prompt in server.prompts:
                    server_description += f"- {prompt.name}: {prompt.description or 'No description'}\n"
                server_description += "\n"
                
            description_parts.append(server_description)
            
        return "\n".join(description_parts)

async def execute_tool_with_session(session: LoggingClientSession, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool using an existing session"""
    try:
        # Initialize the session
        await session.initialize()

        # Call the tool and get the result
        logger.info(f"Calling tool {tool_name} with arguments: {arguments}")
        result = await session.call_tool(tool_name, arguments)

        # Process the result
        content_results = []
        for content in result.content:
            if content.type == "text":
                content_results.append({
                    "type": "text",
                    "text": content.text
                })
                logger.debug(f"Tool result (text): {content.text[:100]}...")
            elif content.type == "image":
                content_results.append({
                    "type": "image",
                    "data": content.data,
                    "mimeType": content.mimeType
                })
                logger.debug(f"Tool result (image): {content.mimeType}")

        return {
            "result": content_results,
            "is_error": result.isError
        }

    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Error executing tool: {str(e)}"}

async def execute_tool(server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool on an MCP server

    This function creates a fresh connection for each tool execution to ensure reliability.
    """
    logger.info(f"Executing tool {tool_name} on server {server_name} with arguments: {arguments}")

    # Load configuration
    config_manager = MCPConfigManager()
    if not config_manager.load_config():
        return {"error": "Failed to load MCP configuration"}

    # Get server configuration
    server_config = config_manager.servers.get(server_name)
    if not server_config:
        return {"error": f"Server {server_name} not found in configuration"}

    # Log the tool call in detail
    logger.debug("Tool call details:")
    logger.debug(f"  Server: {server_name}")
    logger.debug(f"  Tool: {tool_name}")
    logger.debug(f"  Arguments: {json.dumps(arguments, indent=2)}")
    logger.debug(f"  Transport: {server_config.transport}")

    try:
        # Route to appropriate transport
        if server_config.transport == "stdio":
            return await execute_tool_stdio(server_config, tool_name, arguments)
        elif server_config.transport == "sse":
            return await execute_tool_sse(server_config, tool_name, arguments)
        elif server_config.transport == "websocket":
            return await execute_tool_websocket(server_config, tool_name, arguments)
        else:
            return {"error": f"Unsupported transport type: {server_config.transport}"}

    except Exception as e:
        logger.error(f"Error executing tool {tool_name} on server {server_name}: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Error executing tool: {str(e)}"}

async def execute_tool_stdio(server_config: ServerConfig, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute tool using stdio transport"""
    if not server_config.command:
        return {"error": "stdio transport requires command"}

    # Find executable
    full_command = find_executable(server_config.command)
    if not full_command:
        return {"error": f"Failed to find executable for command: {server_config.command}"}

    # Create environment with PATH included
    merged_env = os.environ.copy()
    if server_config.env:
        merged_env.update(server_config.env)

    # Create server parameters
    server_params = StdioServerParameters(
        command=full_command,
        args=server_config.args,
        env=merged_env
    )

    logger.debug(f"  Command: {full_command}")
    logger.debug(f"  Args: {server_config.args}")

    try:
        # Use the MCP client with proper context management
        async with stdio_client(server_params) as (read_stream, write_stream):
            # Use our logging session
            async with LoggingClientSession(read_stream, write_stream) as session:
                return await execute_tool_with_session(session, tool_name, arguments)

    except Exception as e:
        logger.error(f"Error with stdio tool execution: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Error executing tool via stdio: {str(e)}"}

async def execute_tool_sse(server_config: ServerConfig, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute tool using SSE transport"""
    if not server_config.url:
        return {"error": "SSE transport requires URL"}

    try:
        # Expand environment variables in headers
        expanded_headers = {}
        for key, value in server_config.headers.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                expanded_value = os.environ.get(env_var)
                if expanded_value:
                    expanded_headers[key] = expanded_value
                else:
                    logger.warning(f"Environment variable {env_var} not found for header {key}")
            else:
                expanded_headers[key] = value

        logger.debug(f"  URL: {server_config.url}")
        logger.debug(f"  Headers: {list(expanded_headers.keys())}")

        async with sse_client(
            url=server_config.url,
            headers=expanded_headers,
            timeout=server_config.timeout,
            sse_read_timeout=server_config.sse_read_timeout
        ) as (read_stream, write_stream):
            async with LoggingClientSession(read_stream, write_stream) as session:
                return await execute_tool_with_session(session, tool_name, arguments)

    except Exception as e:
        logger.error(f"Error with SSE tool execution: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Error executing tool via SSE: {str(e)}"}

async def execute_tool_websocket(server_config: ServerConfig, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute tool using WebSocket transport"""
    if not server_config.url:
        return {"error": "WebSocket transport requires URL"}

    try:
        logger.debug(f"  URL: {server_config.url}")

        async with websocket_client(server_config.url) as (read_stream, write_stream):
            async with LoggingClientSession(read_stream, write_stream) as session:
                return await execute_tool_with_session(session, tool_name, arguments)

    except Exception as e:
        logger.error(f"Error with WebSocket tool execution: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Error executing tool via WebSocket: {str(e)}"}

async def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    """
    Main plugin execution function called by OptILLM
    
    Args:
        system_prompt: System prompt
        initial_query: User query
        client: OptILLM client
        model: Model identifier
        
    Returns:
        Tuple of (response text, token usage)
    """
    logger.info(f"MCP Plugin run called with model: {model}")
    logger.debug(f"System prompt: {system_prompt[:100]}...")
    logger.debug(f"Initial query: {initial_query}")
    
    try:
        # Load configuration
        config_manager = MCPConfigManager()
        if not config_manager.load_config():
            # Try to create default config
            config_manager.create_default_config()
            # Try loading again
            if not config_manager.load_config():
                logger.error("Failed to load or create MCP configuration")
                # In case of no configuration, pass through the original query
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": initial_query}
                    ],
                    temperature=0.7,
                )
                return response.choices[0].message.content, response.usage.completion_tokens
        
        # Initialize server manager
        server_manager = MCPServerManager(config_manager)
        success = await server_manager.initialize()
        
        if not success:
            logger.warning("Failed to connect to any MCP servers, falling back to default behavior")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_query}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content, response.usage.completion_tokens
        
        # Get tools formatted for the model
        tools = server_manager.get_tools_for_model()
        if not tools:
            logger.warning("No tools available from MCP servers")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_query}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content, response.usage.completion_tokens
        
        # Get capabilities description
        capabilities_description = server_manager.get_capabilities_description()
        logger.debug(f"Capabilities description: {capabilities_description}")
        
        # Enhance system prompt with MCP capabilities
        enhanced_system_prompt = f"{system_prompt}\n\nYou have access to the following MCP capabilities:\n\n{capabilities_description}"
        
        # First request - ask the model what it wants to do
        logger.info("Sending initial request to model")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": initial_query}
            ],
            tools=tools if tools else None,  # Only include tools if available
            temperature=0.7,
        )
        
        # Check if the model wants to use any tools
        response_message = response.choices[0].message
        response_content = response_message.content or ""
        logger.debug(f"Initial model response: {response_content[:100]}...")
        
        # Check for tool calls
        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            logger.info(f"Model requested tool calls: {len(response_message.tool_calls)}")
            for i, tc in enumerate(response_message.tool_calls):
                logger.debug(f"Tool call {i+1}: {tc.function.name} with args: {tc.function.arguments}")
            
            # Create new messages with the original system and user message
            messages = [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": initial_query},
                {"role": "assistant", "content": response_content, "tool_calls": response_message.tool_calls}
            ]
            
            # Process each tool call
            for tool_call in response_message.tool_calls:
                tool_call_id = tool_call.id
                full_tool_name = tool_call.function.name
                
                # Split into server and tool name
                if "." in full_tool_name:
                    server_name, tool_name = full_tool_name.split(".", 1)
                    
                    try:
                        # Parse arguments
                        arguments = json.loads(tool_call.function.arguments)
                        
                        # Execute tool (creates a fresh connection for reliability)
                        result = await execute_tool(server_name, tool_name, arguments)
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(result)
                        })
                        logger.debug(f"Added tool result for {full_tool_name}: {json.dumps(result)[:100]}...")
                    except Exception as e:
                        logger.error(f"Error processing tool call {full_tool_name}: {e}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({"error": f"Error: {str(e)}"})
                        })
                else:
                    # Invalid tool name format
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({"error": f"Invalid tool name format: {full_tool_name}. Expected format: server_name.tool_name"})
                    })
            
            # Send follow-up request with tool results
            logger.info("Sending follow-up request to model with tool results")
            final_response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools if tools else None,  # Keep tools available in case the model wants to make additional calls
                temperature=0.7,
            )
            
            final_message = final_response.choices[0].message
            response_text = final_message.content or ""
            token_usage = final_response.usage.completion_tokens
            logger.debug(f"Final model response: {response_text[:100]}...")
        else:
            # Model didn't call any tools, use its initial response
            response_text = response_content
            token_usage = response.usage.completion_tokens
            logger.info("Model did not request any tool calls")
        
        return response_text, token_usage
        
    except Exception as e:
        logger.error(f"Error in MCP plugin run: {e}")
        logger.error(traceback.format_exc())
        # In case of error, pass through the original query
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_query}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content, response.usage.completion_tokens
