"""OpenAI-compatible provider implementation (used by Qwen and Step)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from app.models import ChatMessage, ChatResponse, Provider
from app.providers.base import BaseProvider, ProviderError


class OpenAICompatProvider(BaseProvider):
    """OpenAI-compatible Chat Completions provider.

    Used for Qwen (DashScope compatible-mode) and Step (StepFun).
    """

    def __init__(
        self,
        name: Provider,
        api_key: str | None,
        base_url: str,
        model: str,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/") + "/chat/completions"
        self._model = model

    async def complete(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        api_key = self._api_key
        if not api_key:
            raise ProviderError(
                f"{self.name.value} API key is not configured",
                provider=self.name,
            )

        payload = {
            "model": model or self._model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(self._base_url, json=payload, headers=headers)

        if resp.status_code != 200:
            raise ProviderError(
                f"{self.name.value} returned {resp.status_code}: {resp.text}",
                provider=self.name,
            )

        data: dict[str, Any] = resp.json()
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})

        return self._build_response(
            provider=self.name,
            model=data.get("model", model or self._model),
            content=message.get("content", ""),
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    async def health(self) -> tuple[str, float | None]:
        api_key = self._api_key
        if not api_key:
            return "unavailable", None

        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            import time

            async with httpx.AsyncClient(timeout=10.0) as client:
                t0 = time.perf_counter()
                resp = await client.get(
                    self._base_url.replace("/chat/completions", "/models"), headers=headers
                )
                latency_ms = (time.perf_counter() - t0) * 1000

            if resp.status_code == 200:
                return "healthy", latency_ms
            return "degraded", latency_ms
        except Exception:
            return "unavailable", None
