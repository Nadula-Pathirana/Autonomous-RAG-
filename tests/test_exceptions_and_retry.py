import pytest

from autonomous_rag.exceptions import friendly_error, is_transient
from autonomous_rag.utils import with_retry


def test_rate_limit_is_transient():
    assert is_transient(RuntimeError("Error 429: rate limit exceeded"))


def test_bad_key_is_not_transient():
    assert not is_transient(RuntimeError("401 invalid api key"))


def test_friendly_messages():
    assert "rate-limited" in friendly_error(RuntimeError("429 rate limit"))
    assert "API key" in friendly_error(RuntimeError("401 Unauthorized"))
    assert "invalid tool call" in friendly_error(RuntimeError("tool_use_failed"))
    assert "logs/app.log" in friendly_error(RuntimeError("something odd"))


def test_retry_succeeds_after_temporary_failures(monkeypatch):
    monkeypatch.setattr("autonomous_rag.utils.time.sleep", lambda seconds: None)
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("429 rate limit")
        return "ok"

    assert with_retry(flaky) == "ok"
    assert calls["count"] == 3


def test_retry_does_not_repeat_permanent_errors(monkeypatch):
    monkeypatch.setattr("autonomous_rag.utils.time.sleep", lambda seconds: None)
    calls = {"count": 0}

    def broken():
        calls["count"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        with_retry(broken)
    assert calls["count"] == 1