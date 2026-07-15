"""Request router — selects and invokes providers."""

from __future__ import annotations

import time
import uuid

from app.models import ChatRequest, ChatResponse, Provider
from app.providers.registry import ProviderRegistry, get_registry
from app.audit.logger import get_audit_logger

# Virtual model → (provider enum, upstream model name) mapping
_VIRTUAL_MODELS: dict[str, tuple[Provider, str]] = {
    "cofounder-auto": (Provider.QWEN, ""),   # upstream model chosen at routing time
    "cofounder-qwen": (Provider.QWEN, ""),
    "cofounder-step": (Provider.STEP, ""),
}


async def route_chat(request: ChatRequest, user_agent: str | None = None) -> ChatResponse:
    """Route a chat completion request to the best available provider.

    Virtual model names (cofounder-auto, cofounder-qwen, cofounder-step) are
    translated to the appropriate upstream provider.  Upstream model names
    are never exposed to clients.
    """
    registry = get_registry()
    audit = get_audit_logger()
    request_id = f"req-{uuid.uuid4().hex[:16]}"

    # Resolve virtual model to provider and upstream model
    virtual_name = request.model or "cofounder-auto"
    if virtual_name not in _VIRTUAL_MODELS:
        raise ValueError(
            f"Unknown model '{virtual_name}'. "
            f"Valid models: cofounder-auto, cofounder-qwen, cofounder-step"
        )

    target_provider, upstream_model = _VIRTUAL_MODELS[virtual_name]

    # For cofounder-auto, use Qwen as preferred and fall back to Step
    preferred = target_provider

    # If the upstream model wasn't pre-configured, use the provider's default
    resolved_upstream_model = upstream_model

    t0 = time.perf_counter()
    try:
        response, used_provider = await registry.complete_with_fallback(
            preferred=preferred,
            model=resolved_upstream_model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 1024,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Override the response model with the virtual model name
        # and record the actual upstream model used
        response.model = virtual_name
        response.selected_upstream_model = response.selected_upstream_model or response.model

        audit.log_request(
            request_id=request_id,
            provider=used_provider.value,
            model=virtual_name,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            status="success",
            user_agent=user_agent,
        )

        return response

    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        audit.log_request(
            request_id=request_id,
            provider=preferred.value,
            model=virtual_name,
            latency_ms=latency_ms,
            status="error",
            error=str(exc),
            user_agent=user_agent,
        )
        raise
