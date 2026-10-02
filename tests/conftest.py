"""Shared test helpers: the extension's own modules and a fake OpenRouter upstream.

No test talks to the network. ``FakeOpenRouter`` replaces
``runtime._request_json`` (the single HTTP seam) and answers by path,
recording every request, so a test can assert exactly what would be sent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
# main.py imports ``runtime`` as a top-level module, exactly as the host runs it
# (``python3 main.py`` from the bundle root).
sys.path.insert(0, str(ROOT))

import main as main_module  # noqa: E402
import runtime as runtime_module  # noqa: E402


@pytest.fixture
def runtime():
    return runtime_module


@pytest.fixture
def main():
    return main_module


class FakeOpenRouter:
    """Answers ``GET /models`` and ``POST /chat/completions`` from canned data."""

    def __init__(self) -> None:
        self.models: list[dict[str, Any]] = []
        self.completions: list[Any] = []
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, *, method: str, headers: dict, body: Any = None):
        self.calls.append({"url": url, "method": method, "headers": dict(headers), "body": body})
        if url.endswith("/models"):
            return {"data": list(self.models)}
        if url.endswith("/chat/completions"):
            reply = self.completions.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        raise AssertionError(f"unexpected upstream call {method} {url}")


@pytest.fixture
def openrouter(monkeypatch, runtime) -> FakeOpenRouter:
    fake = FakeOpenRouter()
    monkeypatch.setattr(runtime, "_request_json", fake)
    return fake


def connection(api_key: str = "key-1") -> dict[str, Any]:
    """The request's ``connection`` block, as main.py builds it from host resources."""
    return {"connection_config": {}, "credentials": {"api_key": api_key} if api_key else {}}


def completion(message: dict[str, Any], *, finish_reason: str = "stop", **usage: Any) -> dict:
    """A complete non-streaming /chat/completions answer."""
    return {
        "choices": [{"message": {"role": "assistant", **message}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16, **usage},
    }
