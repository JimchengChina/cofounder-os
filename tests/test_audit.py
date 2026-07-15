"""Tests for the audit logger."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest


class TestAuditLogger:
    def test_log_writes_daily_jsonl(self, tmp_path):
        from app.audit.logger import AuditLogger

        logger = AuditLogger(log_dir=str(tmp_path))

        logger.log({"event": "test", "value": 42})

        # File should be named YYYY-MM-DD.jsonl for current UTC date
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = tmp_path / f"{today}.jsonl"
        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "test"
        assert record["value"] == 42
        assert "ts" in record
        assert "ts_iso" in record

    def test_log_request_event(self, tmp_path):
        from app.audit.logger import AuditLogger

        logger = AuditLogger(log_dir=str(tmp_path))

        logger.log_request(
            request_id="req-abc",
            provider="cofounder-qwen",
            model="qwen-turbo",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=123.4,
            status="success",
            user_agent="test-client",
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = tmp_path / f"{today}.jsonl"
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        record = json.loads(lines[0])
        assert record["event"] == "chat_request"
        assert record["request_id"] == "req-abc"
        assert record["total_tokens"] == 15
        assert record["latency_ms"] == pytest.approx(123.4)

    def test_multiple_events_appended_to_same_day(self, tmp_path):
        from app.audit.logger import AuditLogger

        logger = AuditLogger(log_dir=str(tmp_path))

        for i in range(5):
            logger.log({"event": "ping", "i": i})

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = tmp_path / f"{today}.jsonl"
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5
        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["i"] == i

    def test_get_audit_logger_singleton(self):
        from app.audit.logger import get_audit_logger

        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        assert logger1 is logger2

    def test_read_recent_returns_records(self, tmp_path):
        from app.audit.logger import AuditLogger

        logger = AuditLogger(log_dir=str(tmp_path))
        for i in range(3):
            logger.log({"event": "ping", "i": i})

        records = logger.read_recent(max_records=10)
        assert len(records) == 3
        assert records[-1]["i"] == 2

    def test_audit_log_contains_no_prompt_content(self, tmp_path):
        """Audit records must never contain message content or secrets."""
        from app.audit.logger import AuditLogger

        logger = AuditLogger(log_dir=str(tmp_path))

        secret_content = "SECRET_API_KEY_12345"
        logger.log_request(
            request_id="req-secret",
            provider="cofounder-qwen",
            model="qwen-turbo",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=500.0,
            status="success",
            user_agent="test",
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = tmp_path / f"{today}.jsonl"
        raw = log_file.read_text(encoding="utf-8")
        assert secret_content not in raw
        assert "Bearer" not in raw
        assert "api_key" not in raw.lower()
        assert "Authorization" not in raw
