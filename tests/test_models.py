"""Tests for Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import ChatMessage, ChatRequest, ChatResponse, Provider, Usage


class TestChatMessage:
    def test_valid_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role.value == "user"
        assert msg.content == "Hello"

    def test_empty_content_fails(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="")

    def test_whitespace_content_fails(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="   ")


class TestChatRequest:
    def test_minimal_request(self):
        req = ChatRequest(messages=[ChatMessage(role="user", content="Hi")])
        assert req.temperature == 0.7
        assert req.max_tokens == 1024
        assert req.stream is False
        assert req.provider is None

    def test_temperature_limits(self):
        with pytest.raises(ValidationError):
            ChatRequest(
                messages=[ChatMessage(role="user", content="Hi")],
                temperature=3.0,
            )


class TestChatResponse:
    def test_build_response(self):
        resp = ChatResponse(
            id="test-id",
            provider=Provider.QWEN,
            model="cofounder-qwen",
            choices=[
                {
                    "index": 0,
                    "message": ChatMessage(role="assistant", content="Hi!"),
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        assert resp.provider == Provider.QWEN
        assert resp.choices[0].message.content == "Hi!"
        assert resp.usage.total_tokens == 8


class TestProvider:
    def test_values(self):
        assert Provider.QWEN.value == "cofounder-qwen"
        assert Provider.STEP.value == "cofounder-step"

    def test_no_openai_enum(self):
        """Provider.OPENAI must not exist."""
        assert not hasattr(Provider, "OPENAI")

    def test_only_two_providers(self):
        """Only Qwen and Step are registered providers."""
        assert len(Provider) == 2
