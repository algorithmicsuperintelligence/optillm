# Version information
__version__ = "0.3.20"

import os as _os

# An empty Hugging Face token is not the same as "no token". huggingface_hub 1.x
# (pulled in by transformers 5) sends an empty token as the literal header
# "Authorization: Bearer ", which the HTTP stack rejects with
# "Illegal header value b'Bearer '". This happens whenever HF_TOKEN is present but
# blank, e.g. a forked-PR CI run where ${{ secrets.HF_TOKEN }} resolves to "".
# Treat any blank HF token env var as unset so anonymous access is used instead.
for _hf_token_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"):
    if _hf_token_var in _os.environ and not _os.environ[_hf_token_var].strip():
        del _os.environ[_hf_token_var]

# Import from server module
from .server import (
    main,
    server_config,
    app,
    known_approaches,
    plugin_approaches,
    parse_combined_approach,
    parse_conversation,
    extract_optillm_approach,
    get_config,
    load_plugins,
    count_reasoning_tokens,
    parse_args,
    execute_single_approach,
    execute_combined_approaches,
    execute_parallel_approaches,
    generate_streaming_response,
)

# List of exported symbols
__all__ = [
    'main',
    'server_config',
    'app',
    'known_approaches',
    'plugin_approaches',
    'parse_combined_approach',
    'parse_conversation',
    'extract_optillm_approach',
    'get_config',
    'load_plugins',
    'count_reasoning_tokens',
    'parse_args',
    'execute_single_approach',
    'execute_combined_approaches',
    'execute_parallel_approaches',
    'generate_streaming_response',
]
