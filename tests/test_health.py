"""Tests for /health endpoint."""

from __future__ import annotations

import pytest

from app.models import Provider
from app.providers.base import BaseProvider
from app.providers.registry import ProviderRegistry, set_registry


class FakeOpenAI(BaseProvider):
    """Fake OpenAI provider."""

    name = Provider.OPENAI

    async def complete(self, **kwargs):
        from app.models import ChatResponse, ChatChoice, ChatMessage, Usage

        return ChatResponse(
            id="fake",
            provider=Provider.OPENAI,
            model="gpt-test",
            choices=[
                {
                    "index": 0,
                    "message": ChatMessage(role="assistant", content="ok"),
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def health(self):
        return "healthy", 42.0


class FakeQwen(BaseProvider):
    """Fake Qwen provider."""

    name = Provider.QWEN

    async def complete(self, **kwargs):
        from app.models import ChatResponse, ChatChoice, ChatMessage, Usage

        return ChatResponse(
            id="fake-qwen",
            provider=Provider.QWEN,
            model="qwen-turbo",
            choices=[
                {
                    "index": 0,
                    "message": ChatMessage(role="assistant", content="ok"),
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def health(self):
        return "healthy", 55.0


class FakeStep(BaseProvider):
    """Fake Step provider."""

    name = Provider.STEP

    async def complete(self, **kwargs):
        from app.models import ChatResponse, ChatChoice, ChatMessage, Usage

        return ChatResponse(
            id="fake-step",
            provider=Provider.STEP,
            model="step-2-16k",
            choices=[
                {
                    "index": 0,
                    "message": ChatMessage(role="assistant", content="ok"),
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def health(self):
        return "unavailable", None


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        registry = ProviderRegistry()
        registry.register(FakeOpenAI())
        set_registry(registry)

        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        registry = ProviderRegistry()
        registry.register(FakeOpenAI())
        set_registry(registry)

        resp = client.get("/api/health")
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "providers" in data

    def test_health_providers_list(self, client):
        registry = ProviderRegistry()
        registry.register(FakeOpenAI())
        set_registry(registry)

        resp = client.get("/api/health")
        data = resp.json()
        assert isinstance(data["providers"], list)
        for p in data["providers"]:
            assert "provider" in p
            assert "status" in p

    def test_health_degraded_when_provider_unavailable(self, client):
        registry = ProviderRegistry()
        registry.register(FakeOpenAI())
        registry.register(FakeStep())
        set_registry(registry)

        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "cofounder-os-gateway"
        assert "version" in data
