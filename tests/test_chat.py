"""Tests for /v1/chat/completions endpoint."""

from __future__ import annotations

import pytest

from app.models import ChatMessage, Provider
from app.providers.base import BaseProvider
from app.providers.registry import ProviderRegistry, set_registry


class FakeOpenAI(BaseProvider):
    """Fake OpenAI provider."""

    name = Provider.OPENAI

    async def complete(self, **kwargs):
        from app.models import ChatResponse, ChatChoice, ChatMessage, Usage

        return ChatResponse(
            id="chatcmpl-openai",
            provider=Provider.OPENAI,
            model=kwargs.get("model", "gpt-4o-mini"),
            choices=[
                {
                    "index": 0,
                    "message": ChatMessage(role="assistant", content="Mocked OpenAI response"),
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def health(self):
        return "healthy", 42.0


class FakeQwen(BaseProvider):
    """Fake Qwen provider."""

    name = Provider.QWEN

    async def complete(self, **kwargs):
        from app.models import ChatResponse, ChatChoice, ChatMessage, Usage

        return ChatResponse(
            id="chatcmpl-qwen",
            provider=Provider.QWEN,
            model=kwargs.get("model", "qwen-turbo"),
            choices=[
                {
                    "index": 0,
                    "message": ChatMessage(role="assistant", content="Mocked Qwen response"),
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
        )

    async def health(self):
        return "healthy", 60.0


class FakeStep(BaseProvider):
    """Fake Step provider."""

    name = Provider.STEP

    async def complete(self, **kwargs):
        from app.models import ChatResponse, ChatChoice, ChatMessage, Usage

        return ChatResponse(
            id="chatcmpl-step",
            provider=Provider.STEP,
            model=kwargs.get("model", "step-2-16k"),
            choices=[
                {
                    "index": 0,
                    "message": ChatMessage(role="assistant", content="Mocked Step response"),
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=12, completion_tokens=8, total_tokens=20),
        )

    async def health(self):
        return "healthy", 70.0


def _setup_client_with(client, providers):
    """Register the given providers and return them."""
    registry = ProviderRegistry()
    for p in providers:
        registry.register(p)
    set_registry(registry)
    return providers


class TestChatCompletionsEndpoint:
    def test_chat_returns_200_with_openai(self, client):
        _setup_client_with(client, [FakeOpenAI()])

        resp = client.post(
            "/api/v1/chat/completions",
            json={
                "provider": "openai",
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "Say hello"}],
                "temperature": 0.5,
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-test"
        assert "choices" in data
        assert data["choices"][0]["message"]["content"] == "Mocked OpenAI response"

    def test_chat_returns_200_with_qwen(self, client):
        _setup_client_with(client, [FakeQwen()])

        resp = client.post(
            "/api/v1/chat/completions",
            json={
                "provider": "cofounder-qwen",
                "model": "qwen-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()
        assert data["provider"] == "cofounder-qwen"
        assert "Mocked Qwen" in data["choices"][0]["message"]["content"]

    def test_chat_returns_200_with_step(self, client):
        _setup_client_with(client, [FakeStep()])

        resp = client.post(
            "/api/v1/chat/completions",
            json={
                "provider": "cofounder-step",
                "model": "step-2-16k",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()
        assert data["provider"] == "cofounder-step"
        assert "Mocked Step" in data["choices"][0]["message"]["content"]

    def test_chat_returns_usage(self, client):
        _setup_client_with(client, [FakeOpenAI()])

        resp = client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()
        usage = data["usage"]
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    def test_chat_bad_request_empty_messages(self, client):
        _setup_client_with(client, [FakeOpenAI()])

        resp = client.post(
            "/api/v1/chat/completions",
            json={"messages": []},
        )
        assert resp.status_code == 422  # Pydantic validation error

    def test_chat_fallback_when_preferred_unavailable(self, client):
        """When openai is not registered, fallback to qwen."""
        _setup_client_with(client, [FakeQwen()])

        resp = client.post(
            "/api/v1/chat/completions",
            json={
                "provider": "openai",  # prefer openai — not available
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()
        assert data["provider"] == "cofounder-qwen"
        assert "Mocked Qwen" in data["choices"][0]["message"]["content"]

    def test_chat_request_id_header(self, client):
        _setup_client_with(client, [FakeOpenAI()])

        resp = client.post(
            "/api/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        assert "X-Request-ID" in resp.headers
        assert resp.headers["X-Request-ID"].startswith("req-")

    def test_list_models(self, client):
        _setup_client_with(client, [FakeOpenAI(), FakeQwen(), FakeStep()])

        resp = client.get("/api/v1/models", headers={"Authorization": "Bearer test-openai-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        provider_ids = {m["provider"] for m in data}
        assert "openai" in provider_ids
        assert "cofounder-qwen" in provider_ids
        assert "cofounder-step" in provider_ids

    def test_audit_recent_with_token(self, client):
        from app.audit.logger import get_audit_logger

        # Write a record directly
        audit = get_audit_logger()
        audit.log_request(
            request_id="req-zzz",
            provider="cofounder-qwen",
            model="qwen-turbo",
            status="success",
        )

        resp = client.get("/api/audit/recent", headers={"X-Audit-Token": "test-audit-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert "records" in data
        assert data["count"] >= 1
        assert data["records"][-1]["request_id"] == "req-zzz"

    def test_audit_recent_without_token(self, client):
        resp = client.get("/api/audit/recent")
        assert resp.status_code == 401
