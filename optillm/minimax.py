"""MiniMax defaults, known-host catalogs, and direct API request options."""
import os
import re
from urllib.parse import urlparse

MINIMAX_MODELS = ("MiniMax-M3.1-Flash", "MiniMax-M3")
OPENROUTER_MINIMAX_MODELS = ("minimax/minimax-m3.1-flash:free", "minimax/minimax-m3")


def minimax_provider_selected():
    """Match get_config's provider precedence without changing explicit models."""
    return bool(os.environ.get("MINIMAX_API_KEY")) and not any(
        os.environ.get(key) for key in ("OPTILLM_API_KEY", "CEREBRAS_API_KEY")
    )


def default_model():
    return os.environ.get("OPTILLM_MODEL", MINIMAX_MODELS[0] if minimax_provider_selected() else "gpt-4o-mini")


def is_minimax_client(client):
    """Apply direct API behavior only to verified MiniMax API hosts."""
    return urlparse(str(getattr(client, "base_url", ""))).hostname in {
        "api.minimax.io", "api.minimaxi.com"
    }


def is_minimax_request(client, model):
    return is_minimax_client(client) or (
        urlparse(str(getattr(client, "base_url", ""))).hostname == 'openrouter.ai'
        and model in OPENROUTER_MINIMAX_MODELS
    )


def current_minimax_models(client, models):
    """Filter ordinary text choices, keeping provider metadata and nontext rows."""
    if is_minimax_client(client):
        current = MINIMAX_MODELS
        text_pattern = r'^MiniMax-M\d'
    elif urlparse(str(getattr(client, "base_url", ""))).hostname == 'openrouter.ai':
        current = OPENROUTER_MINIMAX_MODELS
        text_pattern = r'^minimax/minimax-(?:m\d|\d)'
    else:
        return models
    # Only reorder/filter returned rows; never synthesize gateway capabilities.
    preferred = [row for model in current for row in models if row['id'] == model]
    return preferred + [row for row in models if row['id'] not in current and not re.match(text_pattern, row['id'])]


class MiniMaxRequestError(ValueError):
    """A rejected request must not mark a proxy provider unhealthy."""
    status_code = 400


def prepare_minimax_request(model, request_config):
    """Forward MiniMax extensions through the SDK's JSON body escape hatch."""
    extra_body = dict(request_config.get('extra_body') or {})
    for key in ('thinking', 'reasoning_split'):
        if key in request_config:
            extra_body[key] = request_config.pop(key)
    if model == MINIMAX_MODELS[0]:
        if (extra_body.get('thinking') or {}).get('type') == 'disabled':
            raise MiniMaxRequestError("MiniMax-M3.1-Flash requires adaptive thinking")
        effort = extra_body.get('reasoning_effort', request_config.get('reasoning_effort'))
        if effort is not None and effort not in ('low', 'medium', 'high', 'xhigh', 'max'):
            raise MiniMaxRequestError("MiniMax-M3.1-Flash reasoning_effort must be low, medium, high, xhigh, or max")
        if 'service_tier' in request_config or 'service_tier' in extra_body:
            raise MiniMaxRequestError("MiniMax-M3.1-Flash does not support service_tier")
        if extra_body.get('reasoning_split') is False:
            raise MiniMaxRequestError("MiniMax-M3.1-Flash does not support reasoning_split=false")
    if extra_body:
        request_config['extra_body'] = extra_body
