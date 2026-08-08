import json
import time
from typing import Any, Dict, Iterable, List, Optional


TEXT_BLOCK_TYPES = {"text"}


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in TEXT_BLOCK_TYPES:
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_result":
                    tool_content = block.get("content", "")
                    parts.append(_content_text(tool_content))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content)


def _anthropic_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _content_text(content)
    if content is None:
        return ""
    return json.dumps(content, separators=(",", ":"))


def _coerce_arguments(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def anthropic_tools_to_openai(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None

    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return openai_tools


def anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if not tool_choice:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none"}:
            return tool_choice
        if tool_choice == "any":
            return "required"
        return tool_choice
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "tool":
            return {
                "type": "function",
                "function": {"name": tool_choice.get("name", "")},
            }
        if choice_type == "any":
            return "required"
        if choice_type in {"auto", "none"}:
            return choice_type
    return tool_choice


def anthropic_messages_to_openai(
    messages: List[Dict[str, Any]],
    system: Any = None,
) -> List[Dict[str, Any]]:
    openai_messages: List[Dict[str, Any]] = []

    system_text = _content_text(system)
    if system_text:
        openai_messages.append({"role": "system", "content": system_text})

    for message in messages or []:
        role = message.get("role")
        content = message.get("content", "")

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            openai_messages.append({"role": role, "content": _content_text(content)})
            continue

        if role == "assistant":
            text_parts = []
            tool_calls = []
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), separators=(",", ":")),
                        },
                    })
            openai_message: Dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(part for part in text_parts if part) or None,
            }
            if tool_calls:
                openai_message["tool_calls"] = tool_calls
            openai_messages.append(openai_message)
            continue

        text_parts = []
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_result":
                if text_parts:
                    openai_messages.append({
                        "role": "user",
                        "content": "\n".join(part for part in text_parts if part),
                    })
                    text_parts = []
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _anthropic_tool_result_content(block.get("content", "")),
                })

        if text_parts:
            openai_messages.append({
                "role": role,
                "content": "\n".join(part for part in text_parts if part),
            })

    return openai_messages


def anthropic_request_to_openai(data: Dict[str, Any]) -> Dict[str, Any]:
    openai_request: Dict[str, Any] = {
        "model": data.get("model"),
        "messages": anthropic_messages_to_openai(data.get("messages", []), data.get("system")),
    }

    passthrough_fields = ("temperature", "top_p", "metadata", "optillm_approach")
    for field in passthrough_fields:
        if field in data:
            openai_request[field] = data[field]

    if "max_tokens" in data:
        openai_request["max_tokens"] = data["max_tokens"]
    if "stop_sequences" in data:
        openai_request["stop"] = data["stop_sequences"]
    if "stream" in data:
        openai_request["stream"] = data["stream"]

    openai_tools = anthropic_tools_to_openai(data.get("tools"))
    if openai_tools:
        openai_request["tools"] = openai_tools

    openai_tool_choice = anthropic_tool_choice_to_openai(data.get("tool_choice"))
    if openai_tool_choice:
        openai_request["tool_choice"] = openai_tool_choice

    return openai_request


def _message_to_dict(message: Any) -> Dict[str, Any]:
    if message is None:
        return {}
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump()

    result = {}
    for attr in ("role", "content", "tool_calls"):
        if hasattr(message, attr):
            result[attr] = getattr(message, attr)
    return result


def _tool_call_to_dict(tool_call: Any) -> Dict[str, Any]:
    if isinstance(tool_call, dict):
        return tool_call
    if hasattr(tool_call, "model_dump"):
        return tool_call.model_dump()

    function = getattr(tool_call, "function", None)
    if hasattr(function, "model_dump"):
        function = function.model_dump()
    elif function is not None and not isinstance(function, dict):
        function = {
            "name": getattr(function, "name", ""),
            "arguments": getattr(function, "arguments", "{}"),
        }

    return {
        "id": getattr(tool_call, "id", ""),
        "type": getattr(tool_call, "type", "function"),
        "function": function or {},
    }


def response_to_dict(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response


def openai_response_to_anthropic(response: Any, model: str) -> Dict[str, Any]:
    response_dict = response_to_dict(response)
    choices = response_dict.get("choices", [])
    choice = choices[0] if choices else {}
    message = _message_to_dict(choice.get("message", {}))
    content_blocks = []

    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        call = _tool_call_to_dict(tool_call)
        function = call.get("function") or {}
        content_blocks.append({
            "type": "tool_use",
            "id": call.get("id", ""),
            "name": function.get("name", ""),
            "input": _coerce_arguments(function.get("arguments")),
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = response_dict.get("usage") or {}
    finish_reason = choice.get("finish_reason", "stop")

    return {
        "id": response_dict.get("id", f"msg_{int(time.time() * 1000)}"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": response_dict.get("model", model),
        "stop_reason": openai_finish_reason_to_anthropic(finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def text_response_to_anthropic(
    response: Any,
    model: str,
    completion_tokens: int = 0,
    input_tokens: int = 0,
) -> Dict[str, Any]:
    if isinstance(response, list):
        text = "\n\n".join(str(item) for item in response)
    else:
        text = str(response or "")

    return {
        "id": f"msg_{int(time.time() * 1000)}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": completion_tokens,
        },
    }


def openai_finish_reason_to_anthropic(finish_reason: Optional[str]) -> str:
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "content_filter":
        return "stop_sequence"
    return "end_turn"


def generate_anthropic_stream(message: Dict[str, Any]) -> Iterable[str]:
    start_message = dict(message)
    start_message["content"] = []
    start_message["stop_reason"] = None
    start_message["stop_sequence"] = None

    yield _sse("message_start", {"type": "message_start", "message": start_message})

    for index, block in enumerate(message.get("content", [])):
        block_type = block.get("type")
        if block_type == "text":
            yield _sse("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            })
            text = block.get("text", "")
            if text:
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                })
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})
        elif block_type == "tool_use":
            yield _sse("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": {},
                },
            })
            input_json = json.dumps(block.get("input", {}), separators=(",", ":"))
            if input_json != "{}":
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": input_json},
                })
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": message.get("stop_reason", "end_turn"),
            "stop_sequence": message.get("stop_sequence"),
        },
        "usage": {"output_tokens": message.get("usage", {}).get("output_tokens", 0)},
    })
    yield _sse("message_stop", {"type": "message_stop"})


def approximate_count_tokens(data: Dict[str, Any]) -> int:
    relevant = {
        "system": data.get("system"),
        "messages": data.get("messages", []),
        "tools": data.get("tools", []),
    }
    serialized = json.dumps(relevant, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return max(1, len(serialized) // 4)


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
