"""Request router — selects and invokes providers."""

from __future__ import annotations

import time
import uuid

from app.models import ChatRequest, ChatResponse, Provider
from app.providers.registry import ProviderRegistry, get_registry
from app.audit.logger import get_audit_logger


async def route_chat(request: ChatRequest, user_agent: str | None = None) -> ChatResponse:
    """Route a chat completion request to the best available provider."""
    registry = get_registry()
    audit = get_audit_logger()
    request_id = f"req-{uuid.uuid4().hex[:16]}"

    # Determine target provider
    preferred = request.provider or Provider.OPENAI

    t0 = time.perf_counter()
    try:
        response, used_provider = await registry.complete_with_fallback(
            preferred=preferred,
            model=request.model or "gpt-4o-mini",
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 1024,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        audit.log_request(
            request_id=request_id,
            provider=used_provider.value,
            model=response.model,
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
            model=request.model or "unknown",
            latency_ms=latency_ms,
            status="error",
            error=str(exc),
            user_agent=user_agent,
        )
        raise
