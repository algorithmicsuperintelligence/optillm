"""
Compact plugin for OptiLLM.

Automatically compresses conversation context when it exceeds a token budget,
preserving recent turns verbatim and generating a structured summary of older
content — inspired by Claude Code's compact mechanism.

Uses one LLM call to produce a structured summary with:
  Scope, Key decisions, User preferences, Pending work, Key files referenced.
Recent turns are preserved verbatim.

Composable with other approaches via & operator: compact&moa, compact&bon, etc.

Configuration (env vars or request_config):
  COMPACT_CONTEXT_WINDOW / compact_context_window — max context tokens (default: 128000)
  COMPACT_THRESHOLD / compact_threshold — trigger ratio 0.0-1.0 (default: 0.75)
  COMPACT_KEEP_RECENT / compact_keep_recent — turns to preserve verbatim (default: 4)
"""

import os
import re
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

SLUG = "compact"

DEFAULT_CONTEXT_WINDOW = 128000
DEFAULT_THRESHOLD = 0.75
DEFAULT_KEEP_RECENT = 4

COMPACT_SYSTEM_PROMPT = """You are a conversation summarizer. Given a conversation history, produce a structured summary.

Output ONLY this format, nothing else:

<summary>
Conversation summary:
- Scope: {N} earlier messages compacted (user={U}, assistant={A}).
- Key decisions: {list the main decisions or conclusions reached}
- User preferences: {any stated preferences or constraints}
- Pending work: {any remaining tasks or next steps mentioned}
- Key files referenced: {file paths mentioned, if any}
- Context: {a concise paragraph capturing the essential context needed to continue}
</summary>

Rules:
- Be specific: include actual values, names, and file paths — not vague references
- Be concise: each section should be 1-2 lines maximum
- Omit pleasantries, greetings, and filler
- The Context paragraph is the most important part — it should capture everything a new assistant would need to pick up where this left off"""


def _get_config(request_config: Optional[dict], key: str, env_var: str, default):
    val = None
    if request_config:
        val = request_config.get(key)
    if val is None:
        env_val = os.environ.get(env_var)
        if env_val is not None:
            try:
                val = type(default)(env_val)
            except (ValueError, TypeError):
                logger.warning(f"Invalid value for {env_var}: {env_val!r}, using default {default}")
                val = default
    return val if val is not None else default


def _get_context_window(client, model: str, request_config: Optional[dict]) -> int:
    """Get context window size: try provider /models endpoint first, then config fallback."""
    try:
        model_info = client.models.retrieve(model)
        for attr in ("context_length", "max_context_length", "context_window",
                     "max_model_length", "max_position_embeddings"):
            val = getattr(model_info, attr, None)
            if val is not None:
                return int(val)
    except Exception:
        pass

    return _get_config(request_config, "compact_context_window", "COMPACT_CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW)


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except (ImportError, KeyError):
        return max(1, len(text) // 4)


def parse_tagged_conversation(text: str) -> List[Tuple[str, str]]:
    turns = []
    for match in re.finditer(r'^(User:|Assistant:)\s*', text, re.MULTILINE):
        role = "user" if match.group(1) == "User:" else "assistant"
        start = match.end()
        next_match = re.search(r'^(User:|Assistant:)', text[start:], re.MULTILINE)
        if next_match:
            content = text[start:start + next_match.start()].strip()
        else:
            content = text[start:].strip()
        turns.append((role, content))
    return turns


def reconstruct_tagged(turns: List[Tuple[str, str]]) -> str:
    lines = []
    for role, content in turns:
        tag = "User:" if role == "user" else "Assistant:"
        lines.append(f"{tag} {content}")
    return "\n".join(lines)


def compress_with_llm(
    older_turns: List[Tuple[str, str]],
    system_prompt: str,
    client,
    model: str,
) -> Tuple[Optional[str], int]:
    conversation_text = reconstruct_tagged(older_turns)

    system_content = COMPACT_SYSTEM_PROMPT
    if system_prompt:
        system_content += f"\n\nOriginal system context: {system_prompt}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": conversation_text},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=0.3,
        )
    except Exception as e:
        logger.error(f"Compact: LLM compression failed: {e}")
        return None, 0

    raw = response.choices[0].message.content.strip()
    tokens_used = response.usage.completion_tokens if response.usage else 0

    match = re.search(r'<summary>(.*?)</summary>', raw, re.DOTALL)
    if match:
        summary = match.group(1).strip()
    else:
        summary = raw

    return summary, tokens_used


def run(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    request_config: Optional[dict] = None,
) -> Tuple[str, int]:
    context_window = _get_context_window(client, model, request_config)
    threshold = _get_config(request_config, "compact_threshold", "COMPACT_THRESHOLD", DEFAULT_THRESHOLD)
    keep_recent = _get_config(request_config, "compact_keep_recent", "COMPACT_KEEP_RECENT", DEFAULT_KEEP_RECENT)

    token_count = estimate_tokens(initial_query)
    budget = int(context_window * threshold)

    if token_count < budget:
        logger.debug(f"Compact: passthrough ({token_count} tokens < {budget} budget)")
        return initial_query, 0

    turns = parse_tagged_conversation(initial_query)
    if len(turns) <= keep_recent:
        logger.debug(f"Compact: too few turns to compress ({len(turns)} <= {keep_recent})")
        return initial_query, 0

    split_idx = len(turns) - keep_recent
    older_turns = turns[:split_idx]
    recent_turns = turns[split_idx:]

    logger.info(f"Compact: compressing {len(older_turns)} older turns, keeping {len(recent_turns)} recent")

    summary, tokens_used = compress_with_llm(older_turns, system_prompt, client, model)

    if summary is None:
        logger.warning("Compact: compression failed, returning original query")
        return initial_query, 0

    compressed_turns = [("user", f"[Conversation summary]:\n{summary}")]
    compressed_turns.extend(recent_turns)

    result = reconstruct_tagged(compressed_turns)
    new_token_count = estimate_tokens(result)
    logger.info(f"Compact: {token_count} -> {new_token_count} tokens (used {tokens_used} for compression)")

    return result, tokens_used
