"""Telegram notifications shared by all stages.

Primary use case: alert when an unattended cloud-GPU training run finishes (or
fails), with the freedom to attach reports, plots, and result files. Configured
from env — ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` — which Doppler injects
at runtime.

All sends are **best-effort**: failures are caught, logged, and returned as
``False`` (never raised) and every request has a timeout, so a completed run is
never killed or hung by a notification problem.
"""

from __future__ import annotations

import contextlib
import io
import os
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import httpx

from qsd_common.utils import get_logger

log = get_logger(__name__)

# Telegram API limits.
_MAX_MSG = 4096
_MAX_CAPTION = 1024

# Input a file argument may take: a path, raw bytes, or an open binary stream.
FileInput = str | Path | bytes | BinaryIO


class TelegramNotifier:
    """Send Telegram messages/photos/documents to a fixed chat.

    Prefer :meth:`from_env`. A notifier with an empty token or chat id is
    *disabled*: its sends log a warning and no-op ``False`` instead of raising,
    so missing credentials never crash a run.
    """

    def __init__(
        self,
        token: str | None,
        chat_id: str | None,
        *,
        timeout: float = 15.0,
        parse_mode: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.token = token or ""
        self.chat_id = chat_id or ""
        self.timeout = timeout
        self.parse_mode = parse_mode
        self._client = client

    @classmethod
    def from_env(cls, **kwargs: object) -> TelegramNotifier:
        """Build a notifier from ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID``."""
        return cls(
            os.environ.get("TELEGRAM_BOT_TOKEN"),
            os.environ.get("TELEGRAM_CHAT_ID"),
            **kwargs,  # type: ignore[arg-type]
        )

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    # -- public API ---------------------------------------------------------

    def send_message(
        self, text: str, *, parse_mode: str | None = None, silent: bool = False
    ) -> bool:
        """Send text, auto-split into <=4096-char chunks. Returns overall success."""
        if not self._ready():
            return False
        ok = True
        for chunk in _chunk(text, _MAX_MSG):
            ok &= self._post(
                "sendMessage",
                data={
                    "chat_id": self.chat_id,
                    "text": chunk,
                    "parse_mode": parse_mode if parse_mode is not None else self.parse_mode,
                    "disable_notification": silent,
                },
            )
        return ok

    def send_photo(
        self, photo: FileInput, *, caption: str | None = None, silent: bool = False
    ) -> bool:
        """Send an image (path, bytes, or binary stream), e.g. a saved plot."""
        return self._send_file("sendPhoto", "photo", photo, caption, silent, "image.png")

    def send_document(
        self,
        document: FileInput,
        *,
        caption: str | None = None,
        filename: str | None = None,
        silent: bool = False,
    ) -> bool:
        """Send an arbitrary file (report, CSV/JSON, weights, ...)."""
        return self._send_file(
            "sendDocument", "document", document, caption, silent, filename or "document.bin"
        )

    # -- internals ----------------------------------------------------------

    def _send_file(
        self,
        method: str,
        field: str,
        file: FileInput,
        caption: str | None,
        silent: bool,
        default_name: str,
    ) -> bool:
        if not self._ready():
            return False
        name, content = _read_file(file, default_name)
        data: dict[str, object] = {"chat_id": self.chat_id, "disable_notification": silent}
        overflow: str | None = None
        if caption:
            data["caption"] = caption[:_MAX_CAPTION]
            if self.parse_mode:
                data["parse_mode"] = self.parse_mode
            overflow = caption[_MAX_CAPTION:] or None
        ok = self._post(method, data=data, files={field: (name, content)})
        # A caption longer than Telegram allows is delivered as a follow-up message.
        if overflow:
            ok &= self.send_message(overflow, silent=silent)
        return ok

    def _ready(self) -> bool:
        if not self.enabled:
            log.warning(
                "Telegram notifier disabled (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID unset); "
                "skipping send."
            )
            return False
        return True

    def _post(
        self,
        method: str,
        *,
        data: dict[str, object] | None = None,
        files: dict[str, object] | None = None,
    ) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        # Drop None values so Telegram uses its defaults.
        payload = {k: v for k, v in (data or {}).items() if v is not None}
        try:
            if self._client is not None:
                resp = self._client.post(url, data=payload, files=files, timeout=self.timeout)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, data=payload, files=files)
            resp.raise_for_status()
            body = resp.json()
            if not body.get("ok", False):
                log.warning("Telegram %s rejected: %s", method, body.get("description"))
                return False
            return True
        except Exception as exc:  # best-effort: never propagate
            log.warning("Telegram %s failed: %s", method, exc)
            return False


# -- module-level convenience (default notifier from env) -------------------

_default: TelegramNotifier | None = None


def _default_notifier() -> TelegramNotifier:
    global _default
    if _default is None:
        _default = TelegramNotifier.from_env()
    return _default


def send_message(text: str, **kwargs: object) -> bool:
    return _default_notifier().send_message(text, **kwargs)  # type: ignore[arg-type]


def send_photo(photo: FileInput, **kwargs: object) -> bool:
    return _default_notifier().send_photo(photo, **kwargs)  # type: ignore[arg-type]


def send_document(document: FileInput, **kwargs: object) -> bool:
    return _default_notifier().send_document(document, **kwargs)  # type: ignore[arg-type]


@contextlib.contextmanager
def notify_on_completion(
    name: str,
    *,
    notifier: TelegramNotifier | None = None,
    send_start: bool = False,
) -> Iterator[TelegramNotifier]:
    """Alert on the outcome of a block of work (usable as ``with`` or decorator).

    Sends a success message on clean exit and a failure message (with a trimmed
    traceback) on exception before re-raising. Yields the notifier so the body
    can attach plots/files, e.g.::

        with notify_on_completion("train-ultralytics") as tg:
            metrics = train(...)
            tg.send_photo("runs/pr_curve.png", caption="PR curve")
    """
    tg = notifier or _default_notifier()
    start = time.monotonic()
    if send_start:
        tg.send_message(f"▶️ {name} started")
    try:
        yield tg
    except BaseException as exc:
        elapsed = _format_elapsed(time.monotonic() - start)
        tb = traceback.format_exc()
        tg.send_message(
            f"❌ {name} failed after {elapsed}: {type(exc).__name__}: {exc}\n\n"
            f"{tb[-1500:]}"
        )
        raise
    else:
        elapsed = _format_elapsed(time.monotonic() - start)
        tg.send_message(f"✅ {name} completed in {elapsed}")


# -- helpers ----------------------------------------------------------------


def _chunk(text: str, size: int) -> list[str]:
    if not text:
        return [""]
    return [text[i : i + size] for i in range(0, len(text), size)]


def _read_file(file: FileInput, default_name: str) -> tuple[str, bytes]:
    if isinstance(file, (str, Path)):
        p = Path(file)
        return p.name, p.read_bytes()
    if isinstance(file, bytes):
        return default_name, file
    if isinstance(file, io.IOBase):
        content = file.read()
        name = Path(getattr(file, "name", default_name)).name
        return name, content if isinstance(content, bytes) else str(content).encode()
    raise TypeError(f"unsupported file input: {type(file)!r}")


def _format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
