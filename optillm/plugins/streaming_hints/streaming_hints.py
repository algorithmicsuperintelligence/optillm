"""
Streaming hint injection.

Generates the answer in segments of at most `segment_tokens` tokens and feeds
each segment to the retriever as it arrives. When the retriever finds an
injection point with a relevant hint, the segment is cut there, the hint is
appended inside <hint></hint> tags, and generation continues from the cut.
Segments are used instead of a token stream so the approach works with every
client optillm supports, including the local inference server, which does not
stream.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from .memory_bank import HintMemoryBank
from .retriever import StreamingHintRetriever

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "hint_threshold": 0.2,
    "max_hints": 3,
    "min_gap_chars": 200,
    "segment_tokens": 512,
    "max_segments": 8,
    "hints_path": None,
    "show_hints": False,
}

CONTINUE_TEMPLATE = (
    "{query}\n\n"
    "Your reasoning so far is below. Continue it from exactly where it stops, "
    "without repeating it. Text inside <hint></hint> tags was retrieved to help "
    "you; use it if it is relevant.\n\n"
    "<reasoning_so_far>\n{reasoning}\n</reasoning_so_far>"
)


def _parse_config(request_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    request_config = request_config or {}
    return {key: request_config.get(key, default) for key, default in DEFAULT_CONFIG.items()}


def run_streaming_hints(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    request_config: Optional[Dict[str, Any]] = None,
    memory_bank: Optional[HintMemoryBank] = None,
) -> Tuple[str, int]:
    config = _parse_config(request_config)
    request_config = request_config or {}
    max_tokens = request_config.get("max_tokens")
    temperature = request_config.get("temperature")

    retriever = StreamingHintRetriever(
        memory_bank or HintMemoryBank.from_file(config["hints_path"]),
        threshold=config["hint_threshold"],
        max_hints=config["max_hints"],
        min_gap_chars=config["min_gap_chars"],
    )
    retriever.reset(initial_query)

    output = ""  # model text only
    reasoning = ""  # model text with injected hints
    total_tokens = 0

    for segment in range(config["max_segments"]):
        segment_tokens = config["segment_tokens"]
        if max_tokens:
            remaining = max_tokens - total_tokens
            if remaining <= 0:
                break
            segment_tokens = min(segment_tokens, remaining)

        user_content = initial_query if not reasoning else CONTINUE_TEMPLATE.format(
            query=initial_query, reasoning=reasoning
        )
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": segment_tokens,
        }
        if temperature is not None:
            params["temperature"] = temperature

        response = client.chat.completions.create(**params)
        choice = response.choices[0]
        text = choice.message.content or ""
        used_tokens = response.usage.completion_tokens if response.usage is not None else 0
        total_tokens += used_tokens
        # The local inference server always reports "stop", so a segment that
        # used its whole budget is treated as cut off too.
        truncated = choice.finish_reason == "length" or used_tokens >= segment_tokens

        injection = retriever.feed(text)
        if injection:
            kept = text[: injection.offset]
            output += kept
            reasoning += f"{kept}\n<hint>{injection.hint.text}</hint>\n"
            logger.info(
                f"Segment {segment + 1}: injected hint {injection.hint.id} (score {injection.score:.3f})"
            )
            continue

        output += text
        reasoning += text
        if not truncated:
            break

    logger.info(f"Streaming hints done: {len(retriever.used)} hints injected, {total_tokens} completion tokens")
    return (reasoning if config["show_hints"] else output), total_tokens
