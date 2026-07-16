"""Tests for /v1/chat/completions endpoint."""

from __future__ import annotations

import pytest

from app.models import ChatMessage, CofounderOSMetadata, Provider
from app.providers.base import BaseProvider, ProviderError
from app.providers.registry import ProviderRegistry, set_registry


class FakeQwen(BaseProvider):
    """Fake Qwen provider."""

    name = Provider.QWEN

    async def complete(self, **kwargs):
        from app.models import ChatResponse, ChatChoice, ChatMessage, Usage

        return ChatResponse(
            id="chatcmpl-qwen",
            provider=Provider.QWEN,
            model="cofounder-qwen",
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
            model="cofounder-step",
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


class FakeQwenNullContent(BaseProvider):
    """Fake Qwen provider that returns content=null (e.g. tool-call response)."""

    name = Provider.QWEN

    async def complete(self, **kwargs):
        from app.models import ChatResponse, ChatChoice, Usage

        return ChatResponse(
            id="chatcmpl-qwen-null",
            provider=Provider.QWEN,
            model="cofounder-qwen",
            choices=[
                {
                    "index": 0,
                    "message": ChatMessage(role="assistant", content=None),
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=8, completion_tokens=0, total_tokens=8),
        )

    async def health(self):
        return "healthy", 60.0


class FakeStepEmptyContent(BaseProvider):
    """Fake Step provider that returns empty string content."""

    name = Provider.STEP

    async def complete(self, **kwargs):
        from app.models import ChatResponse, ChatChoice, Usage

        return ChatResponse(
            id="chatcmpl-step-empty",
            provider=Provider.STEP,
            model="cofounder-step",
            choices=[
                {
                    "index": 0,
                    "message": ChatMessage(role="assistant", content=""),
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=12, completion_tokens=0, total_tokens=12),
        )

    async def health(self):
        return "healthy", 70.0


class FakeToolCallProvider(BaseProvider):
    """Fake provider returning tool calls with null content."""

    name = Provider.QWEN

    async def complete(self, **kwargs):
        from app.models import ChatResponse, ChatChoice, Usage

        return ChatResponse(
            id="chatcmpl-tool",
            provider=Provider.QWEN,
            model="cofounder-qwen",
            choices=[
                {
                    "index": 0,
                    "message": ChatMessage(
                        role="assistant",
                        content=None,
                    ),
                    "finish_reason": "tool_calls",
                }
            ],
            usage=Usage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
        )

    async def health(self):
        return "healthy", 60.0


class FailingQwen(BaseProvider):
    """Fake Qwen that always raises ProviderError (triggers fallback)."""

    name = Provider.QWEN

    async def complete(self, **kwargs):
        raise ProviderError("qwen down")

    async def health(self):
        return "unavailable", None


class MalformedProvider(BaseProvider):
    """Provider that raises ProviderError with no valid choices."""

    name = Provider.QWEN

    async def complete(self, **kwargs):
        raise ProviderError(
            "upstream returned response with no choices",
            provider=Provider.QWEN,
        )

    async def health(self):
        return "healthy", 60.0


def _setup_client_with(client, providers):
    """Register the given providers and return them."""
    registry = ProviderRegistry()
    for p in providers:
        registry.register(p)
    set_registry(registry)
    return providers


class TestChatCompletionsEndpoint:
    def test_chat_returns_200_with_qwen(self, client):
        _setup_client_with(client, [FakeQwen()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-qwen",
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
            "/v1/chat/completions",
            json={
                "model": "cofounder-step",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()
        assert data["provider"] == "cofounder-step"
        assert "Mocked Step" in data["choices"][0]["message"]["content"]

    def test_chat_returns_usage(self, client):
        _setup_client_with(client, [FakeQwen()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()
        usage = data["usage"]
        assert usage["prompt_tokens"] == 8
        assert usage["completion_tokens"] == 4
        assert usage["total_tokens"] == 12

    def test_chat_bad_request_empty_messages(self, client):
        _setup_client_with(client, [FakeQwen()])

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": []},
        )
        assert resp.status_code == 422  # Pydantic validation error

    def test_chat_fallback_when_preferred_unavailable(self, client):
        """When cofounder-qwen is not registered, fallback to cofounder-step."""
        _setup_client_with(client, [FakeStep()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-qwen",  # prefer qwen — not available
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()
        assert data["provider"] == "cofounder-step"
        assert "Mocked Step" in data["choices"][0]["message"]["content"]

    def test_chat_auto_routes_to_available_provider(self, client):
        """cofounder-auto selects Qwen when only Qwen is registered."""
        _setup_client_with(client, [FakeQwen()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-auto",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()
        assert data["provider"] == "cofounder-qwen"

    def test_chat_auto_falls_back_to_step(self, client):
        """cofounder-auto falls back to Step when Qwen fails."""
        _setup_client_with(client, [FailingQwen(), FakeStep()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-auto",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()
        assert data["provider"] == "cofounder-step"

    def test_chat_request_id_header(self, client):
        _setup_client_with(client, [FakeQwen()])

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        assert "X-Request-ID" in resp.headers
        assert resp.headers["X-Request-ID"].startswith("req-")

    def test_chat_virtual_model_name_in_response(self, client):
        """Response model field contains the virtual model name, not the upstream."""
        _setup_client_with(client, [FakeQwen()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-qwen",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "cofounder-qwen"

    # ── Tests for upstream response validation fix ──────────────────────────

    def test_chat_null_content_from_upstream(self, client):
        """Qwen returning content=null must not cause HTTP 400."""
        _setup_client_with(client, [FakeQwenNullContent()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-qwen",
                "messages": [{"role": "user", "content": "What is the tool result?"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()
        assert data["provider"] == "cofounder-qwen"
        assert data["choices"][0]["message"]["content"] is None

    def test_chat_empty_content_from_upstream(self, client):
        """Step returning content="" must not cause HTTP 400."""
        _setup_client_with(client, [FakeStepEmptyContent()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-step",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()
        assert data["provider"] == "cofounder-step"
        assert data["choices"][0]["message"]["content"] == ""

    def test_chat_tool_call_with_null_content(self, client):
        """Tool-call response with content=null and finish_reason=tool_calls."""
        _setup_client_with(client, [FakeToolCallProvider()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-qwen",
                "messages": [{"role": "user", "content": "Call a tool"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()
        assert data["provider"] == "cofounder-qwen"
        assert data["choices"][0]["message"]["content"] is None
        assert data["choices"][0]["finish_reason"] == "tool_calls"

    def test_chat_malformed_upstream_returns_502(self, client):
        """Upstream returning no choices must yield HTTP 502, not 400."""
        _setup_client_with(client, [MalformedProvider()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-qwen",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 502, f"Got: {resp.text}"
        data = resp.json()
        assert data["error"] == "upstream_error"

    # ── Tests for routing metadata contract ─────────────────────────────────

    def test_chat_metadata_cofounder_qwen(self, client):
        """cofounder-qwen returns complete cofounder_os metadata in HTTP JSON."""
        _setup_client_with(client, [FakeQwen()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-qwen",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()

        # Top-level, not nested under choices/message/original
        assert "cofounder_os" in data
        meta = data["cofounder_os"]
        assert meta is not None

        # Required fields
        assert isinstance(meta["request_id"], str)
        assert len(meta["request_id"]) > 0
        assert meta["selected_provider"] == "qwen"
        assert isinstance(meta["selected_upstream_model"], str)
        assert meta["routing_reason"] == "forced_local"
        assert isinstance(meta["latency_ms"], (int, float))
        assert meta["latency_ms"] >= 0
        assert "version" in meta

    def test_chat_metadata_cofounder_step(self, client):
        """cofounder-step returns correct cofounder_os metadata."""
        _setup_client_with(client, [FakeStep()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-step",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()

        meta = data["cofounder_os"]
        assert meta is not None
        assert meta["selected_provider"] == "step"
        assert meta["routing_reason"] == "forced_deep"
        assert isinstance(meta["selected_upstream_model"], str)
        assert len(meta["selected_upstream_model"]) > 0
        assert isinstance(meta["request_id"], str)
        assert len(meta["request_id"]) > 0

    def test_chat_metadata_cofounder_auto(self, client):
        """cofounder-auto with preferred provider available returns local_default."""
        _setup_client_with(client, [FakeQwen()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-auto",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()

        meta = data["cofounder_os"]
        assert meta is not None
        assert meta["selected_provider"] == "qwen"
        assert meta["routing_reason"] == "local_default"
        assert isinstance(meta["latency_ms"], (int, float))
        assert meta["latency_ms"] >= 0

    def test_chat_metadata_auto_fallback(self, client):
        """cofounder-auto fallback to Step when Qwen fails returns fallback_deep."""
        _setup_client_with(client, [FailingQwen(), FakeStep()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-auto",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()

        meta = data["cofounder_os"]
        assert meta is not None
        assert meta["selected_provider"] == "step"
        assert meta["routing_reason"] == "fallback_deep"

    def test_chat_metadata_is_top_level(self, client):
        """cofounder_os must be a top-level key, not nested under message."""
        _setup_client_with(client, [FakeQwen()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-qwen",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Top-level key
        assert "cofounder_os" in data
        # Not inside choices or message
        message = data["choices"][0]["message"]
        assert "cofounder_os" not in message
        assert "original" not in data

    def test_chat_metadata_nullable_content_preserved(self, client):
        """Metadata must be present even when upstream content is null."""
        _setup_client_with(client, [FakeQwenNullContent()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-qwen",
                "messages": [{"role": "user", "content": "Test"}],
            },
        )
        assert resp.status_code == 200, f"Got: {resp.text}"
        data = resp.json()

        # Content can be null
        assert data["choices"][0]["message"]["content"] is None
        # Metadata is still present
        assert "cofounder_os" in data
        assert data["cofounder_os"] is not None
        assert data["cofounder_os"]["selected_provider"] == "qwen"
        assert data["cofounder_os"]["routing_reason"] == "forced_local"

    def test_chat_metadata_malformed_upstream_502(self, client):
        """Malformed upstream responses return 502 without metadata."""
        _setup_client_with(client, [MalformedProvider()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-qwen",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 502, f"Got: {resp.text}"
        data = resp.json()
        assert data["error"] == "upstream_error"

    def test_chat_virtual_model_and_metadata_together(self, client):
        """Virtual model name and cofounder_os metadata must both be correct."""
        _setup_client_with(client, [FakeQwen()])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cofounder-auto",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "cofounder-auto"
        assert data["cofounder_os"] is not None
        assert "version" in data["cofounder_os"]

    def test_list_models(self, client):
        _setup_client_with(client, [FakeQwen(), FakeStep()])

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        model_ids = {m["id"] for m in data}
        assert "cofounder-auto" in model_ids
        assert "cofounder-qwen" in model_ids
        assert "cofounder-step" in model_ids
        # No upstream model names exposed
        assert "qwen-turbo" not in model_ids
        assert "step-2-16k" not in model_ids
        assert "gpt-4o-mini" not in model_ids
        assert "openai" not in model_ids

    def test_audit_recent_with_token(self, client):
        from app.audit.logger import get_audit_logger

        # Write a record directly
        audit = get_audit_logger()
        audit.log_request(
            request_id="req-zzz",
            provider="cofounder-qwen",
            model="cofounder-qwen",
            status="success",
        )

        resp = client.get("/audit/recent", headers={"X-Audit-Token": "test-audit-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert "records" in data
        assert data["count"] >= 1
        assert data["records"][-1]["request_id"] == "req-zzz"

    def test_audit_recent_without_token(self, client):
        resp = client.get("/audit/recent")
        assert resp.status_code == 401
