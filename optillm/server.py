import argparse
import logging
import os
import secrets
import time
import traceback
from pathlib import Path
from flask import Flask, request, jsonify
from cerebras.cloud.sdk import Cerebras
from openai import AzureOpenAI, OpenAI
from flask import Response
import json
import importlib
import glob
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, Union, Dict, Any, List
from importlib.metadata import version
from dataclasses import fields

# Import approach modules
from optillm.mcts import chat_with_mcts
from optillm.bon import best_of_n_sampling
from optillm.moa import mixture_of_agents
from optillm.rto import round_trip_optimization
from optillm.self_consistency import advanced_self_consistency_approach
from optillm.pvg import inference_time_pv_game
from optillm.z3_solver import Z3SymPySolverSystem
from optillm.rstar import RStar
from optillm.cot_reflection import cot_reflection
from optillm.plansearch import plansearch
from optillm.leap import leap
from optillm.reread import re2_approach
from optillm.cepo.cepo import cepo, CepoConfig, init_cepo_config
from optillm.mars import multi_agent_reasoning_system
from optillm.batching import RequestBatcher, BatchingError
from optillm.conversation_logger import ConversationLogger
import optillm.conversation_logger

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging_levels = {
    "notset": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Initialize Flask app
app = Flask(__name__)

# Global request batcher (initialized in main() if batch mode enabled)
request_batcher = None

# Global conversation logger (initialized in main() if logging enabled)
conversation_logger = None

def get_config():
    import httpx

    API_KEY = None

    # Create httpx client with SSL configuration
    ssl_verify = server_config.get('ssl_verify', True)
    ssl_cert_path = server_config.get('ssl_cert_path', '')

    # Determine SSL verification setting
    if not ssl_verify:
        logger.warning("SSL certificate verification is DISABLED. This is insecure and should only be used for development.")
        http_client = httpx.Client(verify=False)
    elif ssl_cert_path:
        logger.info(f"Using custom CA certificate bundle: {ssl_cert_path}")
        http_client = httpx.Client(verify=ssl_cert_path)
    else:
        http_client = httpx.Client(verify=True)

    if os.environ.get("OPTILLM_API_KEY"):
        # Use local inference engine
        from optillm.inference import create_inference_client
        API_KEY = os.environ.get("OPTILLM_API_KEY")
        default_client = create_inference_client()
    # Cerebras, OpenAI, Azure, or LiteLLM API configuration
    elif os.environ.get("CEREBRAS_API_KEY"):
        API_KEY = os.environ.get("CEREBRAS_API_KEY")
        base_url = server_config['base_url']
        if base_url != "":
            default_client = Cerebras(api_key=API_KEY, base_url=base_url, http_client=http_client)
        else:
            default_client = Cerebras(api_key=API_KEY, http_client=http_client)
    elif os.environ.get("OPENAI_API_KEY"):
        API_KEY = os.environ.get("OPENAI_API_KEY")
        base_url = server_config['base_url']
        if base_url != "":
            default_client = OpenAI(api_key=API_KEY, base_url=base_url, http_client=http_client)
            logger.info(f"Created OpenAI client with base_url: {base_url}")
        else:
            default_client = OpenAI(api_key=API_KEY, http_client=http_client)
            logger.info("Created OpenAI client without base_url")
    elif os.environ.get("AZURE_OPENAI_API_KEY"):
        API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
        API_VERSION = os.environ.get("AZURE_API_VERSION")
        AZURE_ENDPOINT = os.environ.get("AZURE_API_BASE")
        if API_KEY is not None:
            default_client = AzureOpenAI(
                api_key=API_KEY,
                api_version=API_VERSION,
                azure_endpoint=AZURE_ENDPOINT,
                http_client=http_client
            )
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            azure_credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")
            default_client = AzureOpenAI(
                api_version=API_VERSION,
                azure_endpoint=AZURE_ENDPOINT,
                azure_ad_token_provider=token_provider,
                http_client=http_client
            )
    else:
        # Import the LiteLLM wrapper
        from optillm.litellm_wrapper import LiteLLMWrapper
        default_client = LiteLLMWrapper()
        logger.info("Created LiteLLMWrapper as fallback")
    logger.info(f"Client type: {type(default_client)}")
    return default_client, API_KEY

def count_reasoning_tokens(text: str, tokenizer=None) -> int:
    """
    Count tokens within <think>...</think> tags in the given text.

    Args:
        text: The text to analyze
        tokenizer: Optional tokenizer instance for precise counting

    Returns:
        Number of reasoning tokens (0 if no think tags found)
    """
    if not text or not isinstance(text, str):
        return 0

    # Extract all content within <think>...</think> tags
    # Handle both complete and truncated think blocks

    # First, find all complete <think>...</think> blocks
    complete_pattern = r'<think>(.*?)</think>'
    complete_matches = re.findall(complete_pattern, text, re.DOTALL)

    # Then check for unclosed <think> tag (truncated response)
    # This finds <think> that doesn't have a matching </think> after it
    truncated_pattern = r'<think>(?!.*</think>)(.*)$'
    truncated_match = re.search(truncated_pattern, text, re.DOTALL)

    # Combine all thinking content
    thinking_content = ''.join(complete_matches)
    if truncated_match:
        thinking_content += truncated_match.group(1)

    if not thinking_content:
        return 0

    if tokenizer and hasattr(tokenizer, 'encode'):
        # Use tokenizer for precise counting
        try:
            tokens = tokenizer.encode(thinking_content)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Failed to count tokens with tokenizer: {e}")

    # Fallback: rough estimation (4 chars per token on average, minimum 1 token for non-empty content)
    content_length = len(thinking_content.strip())
    return max(1, content_length // 4) if content_length > 0 else 0

# Server configuration
server_config = {
    'approach': 'none',
    'mcts_simulations': 2,
    'mcts_exploration': 0.2,
    'mcts_depth': 1,
    'best_of_n': 3,
    'model': 'gpt-4o-mini',
    'rstar_max_depth': 3,
    'rstar_num_rollouts': 5,
    'rstar_c': 1.4,
    'n': 1,
    'base_url': '',
    'optillm_api_key': '',
    'return_full_response': False,
    'host': '127.0.0.1',  # Default to localhost for security; use 0.0.0.0 to allow external connections
    'port': 8000,
    'log': 'info',
    'ssl_verify': True,
    'ssl_cert_path': '',
}

# List of known approaches
known_approaches = ["none", "mcts", "bon", "moa", "rto", "z3", "self_consistency",
                   "pvg", "rstar", "cot_reflection", "plansearch", "leap", "re2", "cepo", "mars"]

plugin_approaches = {}

def normalize_message_content(messages):
    """
    Ensure all message content fields are strings, not lists.
    Some models don't handle list-format content correctly.
    """
    normalized_messages = []
    for message in messages:
        normalized_message = message.copy()
        content = message.get('content', '')

        # Convert list content to string if needed
        if isinstance(content, list):
            # Extract text content from the list
            text_content = ' '.join(
                item.get('text', '') for item in content
                if isinstance(item, dict) and item.get('type') == 'text'
            )
            normalized_message['content'] = text_content

        normalized_messages.append(normalized_message)

    return normalized_messages

def none_approach(
    client: Any,
    model: str,
    original_messages: List[Dict[str, str]],
    request_id: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Direct proxy approach that passes through all parameters to the underlying endpoint.

    Args:
        client: OpenAI client instance
        model: Model identifier
        original_messages: Original messages from the request
        request_id: Optional request ID for conversation logging
        **kwargs: Additional parameters to pass through

    Returns:
        Dict[str, Any]: Full OpenAI API response
    """
    # Strip 'none-' prefix from model if present
    if model.startswith('none-'):
        model = model[5:]

    try:
        # Normalize message content to ensure it's always string
        normalized_messages = normalize_message_content(original_messages)

        # Prepare request data for logging
        provider_request = {
            "model": model,
            "messages": normalized_messages,
            **kwargs
        }

        # Make the direct completion call with normalized messages and parameters
        response = client.chat.completions.create(
            model=model,
            messages=normalized_messages,
            **kwargs
        )

        # Convert to dict if it's not already
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response

        # Log the provider call if conversation logging is enabled
        if conversation_logger and request_id:
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)

        return response_dict

    except Exception as e:
        # Log error if conversation logging is enabled
        if conversation_logger and request_id:
            conversation_logger.log_error(request_id, f"Error in none approach: {str(e)}")
        logger.error(f"Error in none approach: {str(e)}")
        raise

def load_plugins():
   # Clear existing plugins first but modify the global dict in place
   plugin_approaches.clear()

   # Get installed package plugins directory
   import optillm
   package_plugin_dir = os.path.join(os.path.dirname(optillm.__file__), 'plugins')

   # Get local project plugins directory
   current_dir = os.getcwd() if server_config.get("plugins_dir", "") == "" else server_config["plugins_dir"]
   local_plugin_dir = os.path.join(current_dir, 'optillm', 'plugins')

   plugin_dirs = []

   # Add package plugin dir
   plugin_dirs.append((package_plugin_dir, "package"))

   # Add local plugin dir only if it's different from package dir
   if local_plugin_dir != package_plugin_dir:
       plugin_dirs.append((local_plugin_dir, "local"))

   for plugin_dir, source in plugin_dirs:
       logger.info(f"Looking for {source} plugins in: {plugin_dir}")

       if not os.path.exists(plugin_dir):
           logger.debug(f"{source.capitalize()} plugin directory not found: {plugin_dir}")
           continue

       plugin_files = glob.glob(os.path.join(plugin_dir, '*.py'))
       if not plugin_files:
           logger.debug(f"No plugin files found in {source} directory: {plugin_dir}")
           continue

       logger.info(f"Found {source} plugin files: {plugin_files}")

       for plugin_file in plugin_files:
           try:
               module_name = os.path.basename(plugin_file)[:-3]  # Remove .py extension
               spec = importlib.util.spec_from_file_location(module_name, plugin_file)
               module = importlib.util.module_from_spec(spec)
               spec.loader.exec_module(module)

               if hasattr(module, 'SLUG') and hasattr(module, 'run'):
                   if module.SLUG in plugin_approaches:
                       logger.info(f"Overriding {source} plugin: {module.SLUG}")
                   plugin_approaches[module.SLUG] = module.run
                   logger.info(f"Loaded {source} plugin: {module.SLUG}")
               else:
                   logger.warning(f"Plugin {module_name} from {source} missing required attributes (SLUG and run)")
           except Exception as e:
               logger.error(f"Error loading {source} plugin {plugin_file}: {str(e)}")

   if not plugin_approaches:
       logger.warning("No plugins loaded from any location")

def get_config_path():
    # Get installed package config directory
    import optillm
    package_config_dir = os.path.join(os.path.dirname(optillm.__file__), 'cepo', 'configs')
    package_config_path = os.path.join(package_config_dir, 'cepo_config.yaml')

    # Get local project config directory
    current_dir = os.getcwd() if server_config.get("config_dir", "") == "" else server_config["config_dir"]
    local_config_dir = os.path.join(current_dir, 'optillm', 'cepo', 'configs')
    local_config_path = os.path.join(local_config_dir, 'cepo_config.yaml')

    # If local config exists and is different from package config, use local
    if os.path.exists(local_config_path) and local_config_path != package_config_path:
        logger.debug(f"Using local config from: {local_config_path}")
        return local_config_path

    # Otherwise use package config
    logger.debug(f"Using package config from: {package_config_path}")
    return package_config_path

def parse_combined_approach(model: str, known_approaches: list, plugin_approaches: dict):
    if model == 'auto':
        return 'SINGLE', ['none'], model

    parts = model.split('-')
    approaches = []
    operation = 'SINGLE'
    model_parts = []
    parsing_approaches = True

    for part in parts:
        if parsing_approaches:
            if part in known_approaches or part in plugin_approaches:
                approaches.append(part)
            elif '&' in part:
                operation = 'AND'
                approaches.extend(part.split('&'))
            elif '|' in part:
                operation = 'OR'
                approaches.extend(part.split('|'))
            else:
                parsing_approaches = False
                model_parts.append(part)
        else:
            model_parts.append(part)

    if not approaches:
        approaches = ['none']
        operation = 'SINGLE'

    actual_model = '-'.join(model_parts)

    return operation, approaches, actual_model

def execute_single_approach(approach, system_prompt, initial_query, client, model, request_config: dict = None, request_id: str = None):
    if approach in known_approaches:
        if approach == 'none':
            # Use the request_config that was already prepared and passed to this function
            kwargs = request_config.copy() if request_config else {}

            # Remove items that are handled separately by the framework
            # Note: 'n' is NOT removed - the none_approach passes it to the client which handles multiple completions
            kwargs.pop('stream', None)  # stream is handled by proxy()

            # Reconstruct original messages from system_prompt and initial_query
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if initial_query:
                messages.append({"role": "user", "content": initial_query})

            logger.debug(f"none_approach kwargs: {kwargs}")
            response = none_approach(original_messages=messages, client=client, model=model, request_id=request_id, **kwargs)
            # For none approach, we return the response and a token count of 0
            # since the full token count is already in the response
            return response, 0
        elif approach == 'mcts':
            # Read the MCTS params from the per-request config (falling back to
            # the server defaults) instead of the shared global server_config.
            # Reading them off the global races with concurrent requests that
            # overwrite it between the write in proxy() and this read (issue #304).
            mcts_config = request_config or {}
            return chat_with_mcts(system_prompt, initial_query, client, model,
                                            mcts_config.get('mcts_simulations', server_config['mcts_simulations']),
                                            mcts_config.get('mcts_exploration', server_config['mcts_exploration']),
                                            mcts_config.get('mcts_depth', server_config['mcts_depth']), request_config, request_id)
        elif approach == 'bon':
            return  best_of_n_sampling(system_prompt, initial_query, client, model, server_config['best_of_n'], request_config, request_id)
        elif approach == 'moa':
            return mixture_of_agents(system_prompt, initial_query, client, model, request_config, request_id)
        elif approach == 'rto':
            return round_trip_optimization(system_prompt, initial_query, client, model, request_config, request_id)
        elif approach == 'z3':
            z3_solver = Z3SymPySolverSystem(system_prompt, client, model, request_config=request_config, request_id=request_id)
            return z3_solver.process_query(initial_query)
        elif approach == "self_consistency":
            return advanced_self_consistency_approach(system_prompt, initial_query, client, model, request_config, request_id)
        elif approach == "pvg":
            return inference_time_pv_game(system_prompt, initial_query, client, model, request_config=request_config, request_id=request_id)
        elif approach == "rstar":
            rstar = RStar(system_prompt, client, model,
                          max_depth=server_config['rstar_max_depth'], num_rollouts=server_config['rstar_num_rollouts'],
                          c=server_config['rstar_c'], request_config=request_config, request_id=request_id)
            return rstar.solve(initial_query)
        elif approach == "cot_reflection":
            return cot_reflection(system_prompt, initial_query, client, model, return_full_response=server_config['return_full_response'], request_config=request_config, request_id=request_id)
        elif approach == 'plansearch':
            return plansearch(system_prompt, initial_query, client, model, n=server_config['n'], request_config=request_config, request_id=request_id)
        elif approach == 'leap':
            return leap(system_prompt, initial_query, client, model, request_config, request_id)
        elif approach == 're2':
            return re2_approach(system_prompt, initial_query, client, model, n=server_config['n'], request_config=request_config, request_id=request_id)
        elif approach == 'cepo':
            return cepo(system_prompt, initial_query, client, model, cepo_config, request_id)
        elif approach == 'mars':
            return multi_agent_reasoning_system(system_prompt, initial_query, client, model, request_config=request_config, request_id=request_id)
    elif approach in plugin_approaches:
        # Check if the plugin accepts request_config
        plugin_func = plugin_approaches[approach]
        import inspect
        sig = inspect.signature(plugin_func)

        # Check if the plugin function is async
        is_async = inspect.iscoroutinefunction(plugin_func)

        if is_async:
            # For async functions, we need to run them in an event loop
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if 'request_config' in sig.parameters:
                    # Plugin supports request_config
                    result = loop.run_until_complete(plugin_func(system_prompt, initial_query, client, model, request_config=request_config))
                else:
                    # Legacy plugin without request_config support
                    result = loop.run_until_complete(plugin_func(system_prompt, initial_query, client, model))
                return result
            finally:
                loop.close()
        else:
            # For synchronous functions, call directly
            if 'request_config' in sig.parameters:
                # Plugin supports request_config
                return plugin_func(system_prompt, initial_query, client, model, request_config=request_config)
            else:
                # Legacy plugin without request_config support
                return plugin_func(system_prompt, initial_query, client, model)
    else:
        raise ValueError(f"Unknown approach: {approach}")

def execute_combined_approaches(approaches, system_prompt, initial_query, client, model, request_config: dict = None):
    final_response = initial_query
    total_tokens = 0
    for approach in approaches:
        response, tokens = execute_single_approach(approach, system_prompt, final_response, client, model, request_config)
        final_response = response
        total_tokens += tokens
    return final_response, total_tokens

async def execute_parallel_approaches(approaches, system_prompt, initial_query, client, model, request_config: dict = None):
    async def run_approach(approach):
        return await asyncio.to_thread(execute_single_approach, approach, system_prompt, initial_query, client, model, request_config)

    tasks = [run_approach(approach) for approach in approaches]
    results = await asyncio.gather(*tasks)
    responses, tokens = zip(*results)
    return list(responses), sum(tokens)

def execute_n_times(n: int, approaches, operation: str, system_prompt: str, initial_query: str, client: Any, model: str,
                     request_config: dict = None, request_id: str = None) -> Tuple[Union[str, List[str]], int]:
    """
    Execute the pipeline n times and return n responses.

    Args:
        n (int): Number of times to run the pipeline
        approaches (list): List of approaches to execute
        operation (str): Operation type ('SINGLE', 'AND', or 'OR')
        system_prompt (str): System prompt
        initial_query (str): Initial query
        client: OpenAI client instance
        model (str): Model identifier

    Returns:
        Tuple[Union[str, List[str]], int]: List of responses and total token count
    """
    responses = []
    total_tokens = 0

    for _ in range(n):
        if operation == 'SINGLE':
            response, tokens = execute_single_approach(approaches[0], system_prompt, initial_query, client, model, request_config, request_id)
        elif operation == 'AND':
            response, tokens = execute_combined_approaches(approaches, system_prompt, initial_query, client, model, request_config)
        elif operation == 'OR':
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response, tokens = loop.run_until_complete(execute_parallel_approaches(approaches, system_prompt, initial_query, client, model, request_config))
            loop.close()
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # If response is already a list (from OR operation), extend responses
        # Otherwise append the single response
        if isinstance(response, list):
            responses.extend(response)
        else:
            responses.append(response)
        total_tokens += tokens

    # If n=1 and we got a single response, return it as is
    # Otherwise return the list of responses
    if n == 1 and len(responses) == 1:
        return responses[0], total_tokens
    return responses, total_tokens

def generate_streaming_response(final_response, model):
    # Generate a unique response ID
    response_id = f"chatcmpl-{int(time.time()*1000)}"
    created = int(time.time())

    # Yield the final response with OpenAI-compatible format
    if isinstance(final_response, list):
        for index, response in enumerate(final_response):
            # First chunk includes role
            yield "data: " + json.dumps({
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"delta": {"role": "assistant", "content": response}, "index": index, "finish_reason": "stop"}],
            }) + "\n\n"
    else:
        # First chunk includes role
        yield "data: " + json.dumps({
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"delta": {"role": "assistant", "content": final_response}, "index": 0, "finish_reason": "stop"}],
        }) + "\n\n"

    # Yield the final message to indicate the stream has ended
    yield "data: [DONE]\n\n"

def extract_contents(response_obj):
    contents = []
    # Handle both single response and list of responses
    responses = response_obj if isinstance(response_obj, list) else [response_obj]

    for response in responses:
        # Extract content from first choice if it exists
        if (response.get('choices') and
            len(response['choices']) > 0 and
            response['choices'][0].get('message') and
            response['choices'][0]['message'].get('content')):
            contents.append(response['choices'][0]['message']['content'])

    return contents

def parse_conversation(messages):
    system_prompt = ""
    conversation = []
    optillm_approach = None

    for message in messages:
        role = message['role']
        content = message['content']

        # Handle content that could be a list or string
        if isinstance(content, list):
            # Extract text content from the list
            text_content = ' '.join(
                item['text'] for item in content
                if isinstance(item, dict) and item.get('type') == 'text'
            )
        else:
            text_content = content

        if role == 'system':
            system_prompt, optillm_approach = extract_optillm_approach(text_content)
        elif role == 'user':
            if not optillm_approach:
                text_content, optillm_approach = extract_optillm_approach(text_content)
            conversation.append(f"User: {text_content}")
        elif role == 'assistant':
            conversation.append(f"Assistant: {text_content}")

    initial_query = "\n".join(conversation)
    return system_prompt, initial_query, optillm_approach

def tagged_conversation_to_messages(response_text):
    """Convert a tagged conversation string or list of strings into a list of messages.
    If the input doesn't contain User:/Assistant: tags, return it as is.

    Args:
        response_text: Either a string containing "User:" and "Assistant:" tags,
                      or a list of such strings.

    Returns:
        If input has tags: A list of message dictionaries.
        If input has no tags: The original input.
    """
    def has_conversation_tags(text):
        return "User:" in text or "Assistant:" in text

    def process_single_response(text):
        if not has_conversation_tags(text):
            return text

        messages = []
        # Split on "User:" or "Assistant:" while keeping the delimiter
        parts = re.split(r'(?=(User:|Assistant:))', text.strip())
        # Remove empty strings
        parts = [p for p in parts if p.strip()]

        for part in parts:
            part = part.strip()
            if part.startswith('User:'):
                messages.append({
                    'role': 'user',
                    'content': part[5:].strip()
                })
            elif part.startswith('Assistant:'):
                messages.append({
                    'role': 'assistant',
                    'content': part[10:].strip()
                })
        return messages

    if isinstance(response_text, list):
        processed = [process_single_response(text) for text in response_text]
        # If none of the responses had tags, return original list
        if all(isinstance(p, str) for p in processed):
            return response_text
        return processed
    else:
        return process_single_response(response_text or "")

def extract_optillm_approach(content):
    match = re.search(r'<optillm_approach>(.*?)</optillm_approach>', content)
    if match:
        approach = match.group(1)
        content = re.sub(r'<optillm_approach>.*?</optillm_approach>', '', content).strip()
        return content, approach
    return content, None

# Optional API key configuration to secure the proxy
@app.before_request
def check_api_key():
    if server_config['optillm_api_key']:
        if request.path == "/health":
            return

        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Invalid Authorization header. Expected format: 'Authorization: Bearer YOUR_API_KEY'"}), 401

        client_key = auth_header.split('Bearer ', 1)[1].strip()
        if not secrets.compare_digest(client_key, server_config['optillm_api_key']):
            return jsonify({"error": "Invalid API key"}), 401

@app.route('/v1/chat/completions', methods=['POST'])
def proxy():
    logger.info('Received request to /v1/chat/completions')
    data = request.get_json()
    auth_header = request.headers.get("Authorization")
    bearer_token = ""

    if auth_header and auth_header.startswith("Bearer "):
        bearer_token = auth_header.split("Bearer ")[1].strip()
        logger.debug(f"Intercepted Bearer Token: {bearer_token}")

    logger.debug(f'Request data: {data}')

    stream = data.get('stream', False)
    messages = data.get('messages', [])
    model = data.get('model', server_config['model'])
    n = data.get('n', server_config['n'])  # Get n value from request or config
    # Extract response_format if present
    response_format = data.get("response_format", None)

    # Handle max_completion_tokens (preferred) and max_tokens (deprecated but supported)
    # Priority: max_completion_tokens > max_tokens
    max_completion_tokens = data.get('max_completion_tokens')
    max_tokens = data.get('max_tokens')

    # Explicit keys that we are already handling
    explicit_keys = {'stream', 'messages', 'model', 'n', 'response_format', 'max_completion_tokens', 'max_tokens'}

    # Copy the rest into request_config
    request_config = {k: v for k, v in data.items() if k not in explicit_keys}

    # Add the explicitly handled ones
    request_config.update({
        "stream": stream,
        "n": n,
    })

    # Only add response_format if it's not None
    if response_format is not None:
        request_config['response_format'] = response_format

    # Add token limits to request_config with proper priority
    if max_completion_tokens is not None:
        request_config['max_completion_tokens'] = max_completion_tokens
        request_config['max_tokens'] = max_completion_tokens  # For backward compatibility with approaches that read max_tokens
    elif max_tokens is not None:
        request_config['max_tokens'] = max_tokens

    optillm_approach = data.get('optillm_approach', server_config['approach'])
    logger.debug(data)
    # MCTS params are resolved per-request in execute_single_approach: any
    # client-sent mcts_* values are already in request_config (copied from the
    # request body above) and the server defaults are used as the fallback there.
    # We deliberately do NOT write them back into the shared global server_config:
    # concurrent requests would overwrite each other's values between the write
    # here and the read in execute_single_approach, silently corrupting results
    # under any threaded WSGI server (and the write also leaked one request's
    # params into every later request). See issue #304.

    system_prompt, initial_query, message_optillm_approach = parse_conversation(messages)

    if message_optillm_approach:
        optillm_approach = message_optillm_approach

    if optillm_approach != "auto":
        model = f"{optillm_approach}-{model}"

    base_url = server_config['base_url']
    default_client, api_key = get_config()

    operation, approaches, model = parse_combined_approach(model, known_approaches, plugin_approaches)

    # Start conversation logging if enabled
    request_id = None
    if conversation_logger and conversation_logger.enabled:
        request_id = conversation_logger.start_conversation(
            client_request={
                'messages': messages,
                'model': data.get('model', server_config['model']),
                'stream': stream,
                'n': n,
                **{k: v for k, v in data.items() if k not in {'messages', 'model', 'stream', 'n'}}
            },
            approach=approaches[0] if len(approaches) == 1 else f"{operation}({','.join(approaches)})",
            model=model
        )

    # Log approach and request start with ID for terminal monitoring
    request_id_str = f' [Request: {request_id}]' if request_id else ''
    logger.info(f'Using approach(es) {approaches}, operation {operation}, with model {model}{request_id_str}')
    if request_id:
        logger.info(f'Request {request_id}: Starting processing')

    if bearer_token != "" and bearer_token.startswith("sk-"):
        api_key = bearer_token
        if base_url != "":
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
    else:
        client = default_client

    try:
        # Route to batch processing if batch mode is enabled
        if request_batcher is not None:
            try:
                # Create request data for batching
                batch_request_data = {
                    'system_prompt': system_prompt,
                    'initial_query': initial_query,
                    'client': client,
                    'model': model,
                    'request_config': request_config,
                    'approaches': approaches,
                    'operation': operation,
                    'n': n,
                    'stream': stream,
                    'optillm_approach': optillm_approach
                }

                logger.debug("Routing request to batch processor")
                result = request_batcher.add_request(batch_request_data)
                return jsonify(result), 200

            except BatchingError as e:
                logger.error(f"Batch processing failed: {e}")
                return jsonify({"error": str(e)}), 500

        # Check if any of the approaches is 'none'
        contains_none = any(approach == 'none' for approach in approaches)

        if operation == 'SINGLE' and approaches[0] == 'none':
            # Pass through the request including the n parameter
            result, completion_tokens = execute_single_approach(approaches[0], system_prompt, initial_query, client, model, request_config, request_id)

            logger.debug(f'Direct proxy response: {result}')

            # Log the final response and finalize conversation logging
            if conversation_logger and request_id:
                conversation_logger.log_final_response(request_id, result)
                conversation_logger.finalize_conversation(request_id)

            if stream:
                if request_id:
                    logger.info(f'Request {request_id}: Completed (streaming response)')
                return Response(generate_streaming_response(extract_contents(result), model), content_type='text/event-stream')
            else :
                if request_id:
                    logger.info(f'Request {request_id}: Completed')
                return jsonify(result), 200

        elif operation == 'AND' or operation == 'OR':
            if contains_none:
                raise ValueError("'none' approach cannot be combined with other approaches")

        # Handle non-none approaches with n attempts
        response, completion_tokens = execute_n_times(n, approaches, operation, system_prompt, initial_query, client, model, request_config, request_id)

        # Check if the response is a full dict (like from proxy plugin or none approach)
        if operation == 'SINGLE' and isinstance(response, dict) and 'choices' in response and 'usage' in response:
            # This is a full response dict, return it directly
            if conversation_logger and request_id:
                conversation_logger.log_final_response(request_id, response)
                conversation_logger.finalize_conversation(request_id)

            if stream:
                if request_id:
                    logger.info(f'Request {request_id}: Completed (streaming response)')
                return Response(generate_streaming_response(extract_contents(response), model), content_type='text/event-stream')
            else:
                if request_id:
                    logger.info(f'Request {request_id}: Completed')
                return jsonify(response), 200

    except Exception as e:
        # Log error to conversation logger if enabled
        if conversation_logger and request_id:
            conversation_logger.log_error(request_id, str(e))
            conversation_logger.finalize_conversation(request_id)

        request_id_str = f' {request_id}' if request_id else ''
        logger.error(f"Error processing request{request_id_str}: {str(e)}")
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.exception("Debug request exception")
        return jsonify({"error": str(e)}), 500

    # Convert tagged conversation to messages format if needed
    if isinstance(response, list):
        processed_response = tagged_conversation_to_messages(response)
        # If processed_response is a list of message lists, extract last message content
        if processed_response != response:  # Only process if format changed
            response = [msg[-1]['content'] if isinstance(msg, list) and msg else msg
                    for msg in processed_response]
        # Otherwise keep original response
    else:
        messages = tagged_conversation_to_messages(response)
        if isinstance(messages, list) and messages:  # Only process if format changed
            response = messages[-1]['content']

    if stream:
        return Response(generate_streaming_response(response, model), content_type='text/event-stream')
    else:
        # Calculate reasoning tokens from the response
        reasoning_tokens = 0
        if isinstance(response, str):
            reasoning_tokens = count_reasoning_tokens(response)
        elif isinstance(response, list) and response:
            # For multiple responses, sum up reasoning tokens from all
            reasoning_tokens = sum(count_reasoning_tokens(resp) for resp in response if isinstance(resp, str))

        response_data = {
            'model': model,
            'choices': [],
            'usage': {
                'completion_tokens': completion_tokens,
                'completion_tokens_details': {
                    'reasoning_tokens': reasoning_tokens
                }
            }
        }

        if isinstance(response, list):
            for index, resp in enumerate(response):
                response_data['choices'].append({
                    'index': index,
                    'message': {
                        'role': 'assistant',
                        'content': resp,
                    },
                    'finish_reason': 'stop'
                })
        else:
            response_data['choices'].append({
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': response,
                },
                'finish_reason': 'stop'
            })

        # Log the final response and finalize conversation logging
        if conversation_logger and request_id:
            conversation_logger.log_final_response(request_id, response_data)
            conversation_logger.finalize_conversation(request_id)

        logger.debug(f'API response: {response_data}')
        if request_id:
            logger.info(f'Request {request_id}: Completed')
        return jsonify(response_data), 200

@app.route('/v1/models', methods=['GET'])
def proxy_models():
    logger.info('Received request to /v1/models')
    default_client, API_KEY = get_config()
    try:
        if server_config['base_url']:
            client = OpenAI(api_key=API_KEY, base_url=server_config['base_url'])
            # For external API, fetch models using the OpenAI client
            models_response = client.models.list()
            # Convert to dict format
            models_data = {
                "object": "list",
                "data": [model.dict() for model in models_response.data]
            }
        else:
            # For local inference, create a models response manually
            current_model = server_config.get('model', 'gpt-3.5-turbo')
            models_data = {
                "object": "list",
                "data": [
                    {
                        "id": current_model,
                        "object": "model",
                        "created": 1677610602,
                        "owned_by": "optillm"
                    }
                ]
            }

        logger.debug('Models retrieved successfully')
        return jsonify(models_data), 200
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        return jsonify({"error": f"Error fetching models: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM inference with various approaches.")

    try:
        from optillm import __version__ as package_version
    except ImportError:
        package_version = "unknown"

    parser.add_argument('--version', action='version',
                       version=f'%(prog)s {package_version}',
                       help="Show program's version number and exit")

    # Define arguments and their corresponding environment variables
    args_env = [
        ("--optillm-api-key", "OPTILLM_API_KEY", str, "", "Optional API key for client authentication to optillm"),
        ("--approach", "OPTILLM_APPROACH", str, "auto", "Inference approach to use", known_approaches + list(plugin_approaches.keys())),
        ("--mcts-simulations", "OPTILLM_SIMULATIONS", int, 2, "Number of MCTS simulations"),
        ("--mcts-exploration", "OPTILLM_EXPLORATION", float, 0.2, "Exploration weight for MCTS"),
        ("--mcts-depth", "OPTILLM_DEPTH", int, 1, "Simulation depth for MCTS"),
        ("--model", "OPTILLM_MODEL", str, "gpt-4o-mini", "OpenAI model to use"),
        ("--rstar-max-depth", "OPTILLM_RSTAR_MAX_DEPTH", int, 3, "Maximum depth for rStar algorithm"),
        ("--rstar-num-rollouts", "OPTILLM_RSTAR_NUM_ROLLOUTS", int, 5, "Number of rollouts for rStar algorithm"),
        ("--rstar-c", "OPTILLM_RSTAR_C", float, 1.4, "Exploration constant for rStar algorithm"),
        ("--n", "OPTILLM_N", int, 1, "Number of final responses to be returned"),
        ("--return-full-response", "OPTILLM_RETURN_FULL_RESPONSE", bool, False, "Return the full response including the CoT with <thinking> tags"),
        ("--host", "OPTILLM_HOST", str, "127.0.0.1", "Host address to bind the server to (use 0.0.0.0 to allow external connections)"),
        ("--port", "OPTILLM_PORT", int, 8000, "Specify the port to run the proxy"),
        ("--log", "OPTILLM_LOG", str, "info", "Specify the logging level", list(logging_levels.keys())),
        ("--launch-gui", "OPTILLM_LAUNCH_GUI", bool, False, "Launch a Gradio chat interface"),
        ("--plugins-dir", "OPTILLM_PLUGINS_DIR", str, "", "Path to the plugins directory"),
        ("--log-conversations", "OPTILLM_LOG_CONVERSATIONS", bool, False, "Enable conversation logging with full metadata"),
        ("--conversation-log-dir", "OPTILLM_CONVERSATION_LOG_DIR", str, str(Path.home() / ".optillm" / "conversations"), "Directory to save conversation logs"),
    ]

    for arg, env, type_, default, help_text, *extra in args_env:
        env_value = os.environ.get(env)
        if env_value is not None:
            if type_ == bool:
                default = env_value.lower() in ('true', '1', 'yes')
            else:
                default = type_(env_value)
        if extra and extra[0]:  # Check if there are choices for this argument
            parser.add_argument(arg, type=type_, default=default, help=help_text, choices=extra[0])
        else:
            if type_ == bool:
                # For boolean flags, use store_true action
                parser.add_argument(arg, action='store_true', default=default, help=help_text)
            else:
                parser.add_argument(arg, type=type_, default=default, help=help_text)

    # Special handling for best_of_n to support both formats
    best_of_n_default = int(os.environ.get("OPTILLM_BEST_OF_N", 3))
    parser.add_argument("--best-of-n", "--best_of_n", dest="best_of_n", type=int, default=best_of_n_default,
                        help="Number of samples for best_of_n approach")

    # Special handling for base_url to support both formats
    base_url_default = os.environ.get("OPTILLM_BASE_URL", "")
    parser.add_argument("--base-url", "--base_url", dest="base_url", type=str, default=base_url_default,
                        help="Base url for OpenAI compatible endpoint")

    # SSL configuration arguments
    ssl_verify_default = os.environ.get("OPTILLM_SSL_VERIFY", "true").lower() in ("true", "1", "yes")
    parser.add_argument("--ssl-verify", dest="ssl_verify", action="store_true" if ssl_verify_default else "store_false",
                        default=ssl_verify_default,
                        help="Enable SSL certificate verification (default: True)")
    parser.add_argument("--no-ssl-verify", dest="ssl_verify", action="store_false",
                        help="Disable SSL certificate verification")

    ssl_cert_path_default = os.environ.get("OPTILLM_SSL_CERT_PATH", "")
    parser.add_argument("--ssl-cert-path", dest="ssl_cert_path", type=str, default=ssl_cert_path_default,
                        help="Path to custom CA certificate bundle for SSL verification")

    # Use the function to get the default path
    default_config_path = get_config_path()

    # Batch mode arguments
    batch_mode_default = os.environ.get("OPTILLM_BATCH_MODE", "false").lower() == "true"
    batch_size_default = int(os.environ.get("OPTILLM_BATCH_SIZE", 4))
    batch_wait_ms_default = int(os.environ.get("OPTILLM_BATCH_WAIT_MS", 50))

    parser.add_argument("--batch-mode", action="store_true", default=batch_mode_default,
                        help="Enable automatic request batching (fail-fast, no fallback)")
    parser.add_argument("--batch-size", type=int, default=batch_size_default,
                        help="Maximum batch size for request batching")
    parser.add_argument("--batch-wait-ms", dest="batch_wait_ms", type=int, default=batch_wait_ms_default,
                        help="Maximum wait time in milliseconds for batch formation")

    # Special handling of all the CePO Configurations
    for field in fields(CepoConfig):
        parser.add_argument(f"--cepo_{field.name}",
                        dest=f"cepo_{field.name}",
                        type=field.type,
                        default=None,
                        help=f"CePO configuration for {field.name}")

    parser.add_argument("--cepo_config_file",
                    dest="cepo_config_file",
                    type=str,
                    default=default_config_path,
                    help="Path to CePO configuration file")

    args = parser.parse_args()

    # Convert argument names to match server_config keys
    args_dict = vars(args)
    for key in list(args_dict.keys()):
        new_key = key.replace("-", "_")
        if new_key != key:
            args_dict[new_key] = args_dict.pop(key)

    return args

def main():
    global server_config
    global cepo_config
    global request_batcher
    global conversation_logger
    # Call this function at the start of main()

    # Load plugins first so they're available in argument parser
    load_plugins()

    args = parse_args()
    # Update server_config with all argument values
    server_config.update(vars(args))

    port = server_config['port']

    # Initialize request batcher if batch mode is enabled
    if server_config.get('batch_mode', False):
        logger.info(f"Batch mode enabled: size={server_config['batch_size']}, "
                   f"wait={server_config['batch_wait_ms']}ms")
        request_batcher = RequestBatcher(
            max_batch_size=server_config['batch_size'],
            max_wait_ms=server_config['batch_wait_ms'],
            enable_logging=True
        )

        # Set up the batch processor function
        def process_batch_requests(batch_requests):
            """
            Process a batch of requests using true batching when possible

            Args:
                batch_requests: List of request data dictionaries

            Returns:
                List of response dictionaries
            """
            import time
            from optillm.batching import BatchingError

            if not batch_requests:
                return []

            logger.info(f"Processing batch of {len(batch_requests)} requests")

            # Check if we can use true batching (all requests compatible and using 'none' approach)
            can_use_true_batching = True
            first_req = batch_requests[0]

            # Check compatibility across all requests
            for req_data in batch_requests:
                if (req_data['stream'] or
                    req_data['approaches'] != first_req['approaches'] or
                    req_data['operation'] != first_req['operation'] or
                    req_data['model'] != first_req['model']):
                    can_use_true_batching = False
                    break

            # For now, implement sequential processing but with proper infrastructure
            # TODO: Implement true PyTorch/MLX batching in next phase
            responses = []

            for i, req_data in enumerate(batch_requests):
                try:
                    logger.debug(f"Processing batch request {i+1}/{len(batch_requests)}")

                    # Extract request parameters
                    system_prompt = req_data['system_prompt']
                    initial_query = req_data['initial_query']
                    client = req_data['client']
                    model = req_data['model']
                    request_config = req_data['request_config']
                    approaches = req_data['approaches']
                    operation = req_data['operation']
                    n = req_data['n']
                    stream = req_data['stream']

                    # Validate request
                    if stream:
                        raise BatchingError("Streaming requests cannot be batched")

                    # Check if any of the approaches is 'none'
                    contains_none = any(approach == 'none' for approach in approaches)

                    if operation == 'SINGLE' and approaches[0] == 'none':
                        # Pass through the request including the n parameter
                        result, completion_tokens = execute_single_approach(
                            approaches[0], system_prompt, initial_query, client, model, request_config)
                    elif operation == 'AND' or operation == 'OR':
                        if contains_none:
                            raise ValueError("'none' approach cannot be combined with other approaches")
                        # Handle non-none approaches with n attempts
                        result, completion_tokens = execute_n_times(
                            n, approaches, operation, system_prompt, initial_query, client, model, request_config)
                    else:
                        # Handle non-none approaches with n attempts
                        result, completion_tokens = execute_n_times(
                            n, approaches, operation, system_prompt, initial_query, client, model, request_config)

                    # Convert tagged conversation to messages format if needed
                    if isinstance(result, list):
                        processed_response = tagged_conversation_to_messages(result)
                        if processed_response != result:  # Only process if format changed
                            result = [msg[-1]['content'] if isinstance(msg, list) and msg else msg
                                    for msg in processed_response]
                    else:
                        messages = tagged_conversation_to_messages(result)
                        if isinstance(messages, list) and messages:  # Only process if format changed
                            result = messages[-1]['content']

                    # Generate the response in OpenAI format
                    if isinstance(result, list):
                        choices = []
                        for j, res in enumerate(result):
                            choices.append({
                                "index": j,
                                "message": {
                                    "role": "assistant",
                                    "content": res
                                },
                                "finish_reason": "stop"
                            })
                    else:
                        choices = [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": result
                            },
                            "finish_reason": "stop"
                        }]

                    response_dict = {
                        "id": f"chatcmpl-{int(time.time()*1000)}-{i}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": choices,
                        "usage": {
                            "prompt_tokens": 0,  # Will be calculated properly later
                            "completion_tokens": completion_tokens if isinstance(completion_tokens, int) else 0,
                            "total_tokens": completion_tokens if isinstance(completion_tokens, int) else 0
                        }
                    }

                    responses.append(response_dict)

                except Exception as e:
                    logger.error(f"Error processing batch request {i+1}: {e}")
                    raise BatchingError(f"Failed to process request {i+1}: {str(e)}")

            logger.info(f"Completed batch processing of {len(responses)} requests")
            return responses

        # Set the processor function on the batcher
        request_batcher.set_processor(process_batch_requests)

    # Set logging level from user request
    logging_level = server_config['log']
    if logging_level in logging_levels.keys():
        logger.setLevel(logging_levels[logging_level])

    # Initialize conversation logger if enabled
    global conversation_logger
    conversation_logger = ConversationLogger(
        log_dir=Path(server_config['conversation_log_dir']),
        enabled=server_config['log_conversations']
    )
    # Set the global logger instance for access from approach modules
    optillm.conversation_logger.set_global_logger(conversation_logger)
    if server_config['log_conversations']:
        logger.info(f"Conversation logging enabled. Logs will be saved to: {server_config['conversation_log_dir']}")

    # set and log the cepo configs
    cepo_config = init_cepo_config(server_config)
    if args.approach == 'cepo':
        logger.info(f"CePO Config: {cepo_config}")

    logger.info(f"Starting server with approach: {server_config['approach']}")
    server_config_clean = server_config.copy()
    if server_config_clean['optillm_api_key']:
        server_config_clean['optillm_api_key'] = '[REDACTED]'
    logger.info(f"Server configuration: {server_config_clean}")

    # Launch GUI if requested
    if server_config.get('launch_gui'):
        try:
            import gradio as gr
            # Start server in a separate thread
            import threading
            host = server_config['host']
            server_thread = threading.Thread(target=app.run, kwargs={'host': host, 'port': port})
            server_thread.daemon = True
            server_thread.start()

            # Configure the base URL for the Gradio interface
            base_url = f"http://localhost:{port}/v1"
            logger.info(f"Launching Gradio interface connected to {base_url}")

            # Create custom chat function with extended timeout
            def chat_with_optillm(message, history):
                import httpx
                from openai import OpenAI

                # Create client with extended timeout and no retries
                custom_client = OpenAI(
                    api_key="optillm",
                    base_url=base_url,
                    timeout=httpx.Timeout(1800.0, connect=5.0),  # 30 min timeout
                    max_retries=0  # No retries - prevents duplicate requests
                )

                # Convert history to messages format
                messages = []
                for h in history:
                    if h[0]:  # User message
                        messages.append({"role": "user", "content": h[0]})
                    if h[1]:  # Assistant message
                        messages.append({"role": "assistant", "content": h[1]})
                messages.append({"role": "user", "content": message})

                # Make request
                try:
                    response = custom_client.chat.completions.create(
                        model=server_config['model'],
                        messages=messages
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    return f"Error: {str(e)}"

            # Create Gradio interface with queue for long operations
            demo = gr.ChatInterface(
                chat_with_optillm,
                title="OptILLM Chat Interface",
                description=f"Connected to OptILLM proxy at {base_url}"
            )
            demo.queue()  # Enable queue to handle long operations properly
            demo.launch(server_name=host, share=False)
        except ImportError:
            logger.error("Gradio is required for GUI. Install it with: pip install gradio")
            return

    app.run(host=server_config['host'], port=port)

if __name__ == "__main__":
    main()
