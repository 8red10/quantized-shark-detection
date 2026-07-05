"""Tests for qsd_common.notify — no network (httpx MockTransport)."""

from __future__ import annotations

import httpx
import pytest

from qsd_common.notify import TelegramNotifier, notify_on_completion


def _notifier(handler, **kwargs) -> TelegramNotifier:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return TelegramNotifier(token="t", chat_id="c", client=client, **kwargs)


def test_send_message_ok():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url.path == "/bott/sendMessage"
        return httpx.Response(200, json={"ok": True})

    assert _notifier(handler).send_message("hello") is True
    assert len(calls) == 1


def test_send_message_chunks_over_limit():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    # 4096 * 2 + 10 chars -> 3 chunks -> 3 POSTs.
    assert _notifier(handler).send_message("x" * (4096 * 2 + 10)) is True
    assert len(calls) == 3


def test_send_message_failure_is_best_effort():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "description": "bad token"})

    # Returns False, does not raise.
    assert _notifier(handler).send_message("hello") is False


def test_disabled_when_creds_missing():
    # No token/chat -> disabled, no HTTP attempted.
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should not be called")

    tg = TelegramNotifier(token="", chat_id="", client=httpx.Client(
        transport=httpx.MockTransport(handler)))
    assert tg.enabled is False
    assert tg.send_message("hello") is False


def test_send_document_bytes():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["ctype"] = request.headers.get("content-type", "")
        return httpx.Response(200, json={"ok": True})

    ok = _notifier(handler).send_document(b"col1,col2\n1,2\n", filename="results.csv")
    assert ok is True
    assert seen["path"] == "/bott/sendDocument"
    assert "multipart/form-data" in seen["ctype"]


def test_notify_on_completion_success_sends_message():
    texts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts.append(dict(httpx.QueryParams(request.content.decode())).get("text", ""))
        return httpx.Response(200, json={"ok": True})

    tg = _notifier(handler)
    with notify_on_completion("job", notifier=tg):
        pass
    assert any("completed" in t for t in texts)


def test_notify_on_completion_failure_sends_and_reraises():
    texts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts.append(dict(httpx.QueryParams(request.content.decode())).get("text", ""))
        return httpx.Response(200, json={"ok": True})

    tg = _notifier(handler)
    with pytest.raises(ValueError):
        with notify_on_completion("job", notifier=tg):
            raise ValueError("boom")
    assert any("failed" in t and "boom" in t for t in texts)
