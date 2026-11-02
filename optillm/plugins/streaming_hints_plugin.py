"""
Streaming Hints Plugin for OptiLLM

Injects hints from a memory bank into the model's reasoning while it is being
generated. See optillm/plugins/streaming_hints/README.md for details.
"""

import logging
from typing import Any, Dict, Tuple

from optillm.plugins.streaming_hints import run_streaming_hints

# Plugin identifier for optillm
SLUG = "streaming_hints"

logger = logging.getLogger(__name__)


def run(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    request_config: Dict[str, Any] = None
) -> Tuple[str, int]:
    logger.info("Starting streaming hints")
    return run_streaming_hints(system_prompt, initial_query, client, model, request_config)
