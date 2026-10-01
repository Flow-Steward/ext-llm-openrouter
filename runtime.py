"""Runnable remote HTTPS runtime for the OpenRouter reference bundle.

Run with ``FS_EXTENSION_TOKEN=... FS_EXTENSION_PORT=8090 python3 runtime.py``.
OpenRouter's upstream is fixed; the platform supplies only account-scoped
credentials per invocation and this runtime never persists them.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import math
import os
import re
import secrets
import socket
import ssl
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request

MAX_RUNTIME_BODY_BYTES = 5 * 1024 * 1024
MAX_DISCOVERY_PAGE_SIZE = 200
MAX_DISCOVERY_SNAPSHOTS = 4
DISCOVERY_SNAPSHOT_TTL_SECONDS = 90
MAX_ACTION_IDEMPOTENCY_ENTRIES = 256
ACTION_IDEMPOTENCY_TTL_SECONDS = 300
ACTION_IDEMPOTENCY_WAIT_SECONDS = 65
_discovery_snapshots: dict[str, tuple[float, list[Any]]] = {}
_discovery_snapshots_lock = threading.Lock()
_action_idempotency: dict[str, dict[str, Any]] = {}
_action_idempotency_lock = threading.Lock()


def _execute_idempotent_action(
    request: dict[str, Any],
    action_id: str,
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    key = str(request.get("idempotency_key") or "").strip()
    if not key or len(key) > 512:
        raise RuntimeError("invalid_response")
    cache_key = hashlib.sha256(key.encode()).hexdigest()
    try:
        canonical = json.dumps(
            {"action": action_id, "request": request},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid_response") from exc
    fingerprint = hashlib.sha256(canonical).hexdigest()
    now = time.monotonic()
    with _action_idempotency_lock:
        for expired_key in [
            candidate
            for candidate, entry in _action_idempotency.items()
            if entry["deadline"] <= now and entry["event"].is_set()
        ]:
            _action_idempotency.pop(expired_key, None)
        entry = _action_idempotency.get(cache_key)
        if entry is not None:
            if entry["fingerprint"] != fingerprint:
                raise RuntimeError("invalid_response")
            owner = False
        else:
            if len(_action_idempotency) >= MAX_ACTION_IDEMPOTENCY_ENTRIES:
                raise RuntimeError("rate_limited")
            entry = {
                "deadline": now + ACTION_IDEMPOTENCY_TTL_SECONDS,
                "event": threading.Event(),
                "fingerprint": fingerprint,
                "result": None,
                "error": "",
            }
            _action_idempotency[cache_key] = entry
            owner = True
    if owner:
        try:
            entry["result"] = operation()
        except Exception as exc:
            entry["error"] = str(exc).split(":", 1)[0] or "upstream_error"
        finally:
            entry["event"].set()
    elif not entry["event"].wait(timeout=ACTION_IDEMPOTENCY_WAIT_SECONDS):
        raise RuntimeError("timeout")
    if entry["error"]:
        raise RuntimeError(entry["error"])
    result = entry["result"]
    if not isinstance(result, dict):
        raise RuntimeError("invalid_response")
    return result


def _read_bounded(stream: Any) -> bytes:
    raw = stream.read(MAX_RUNTIME_BODY_BYTES + 1)
    if len(raw) > MAX_RUNTIME_BODY_BYTES:
        raise RuntimeError("invalid_response")
    return raw


def _read_request_body(stream: Any, content_length: int) -> bytes:
    if content_length < 0 or content_length > MAX_RUNTIME_BODY_BYTES:
        raise RuntimeError("invalid_response")
    raw = stream.read(content_length)
    if len(raw) != content_length:
        raise RuntimeError("invalid_response")
    return raw


def _public_https_url(raw_url: str) -> tuple[str, str, int, str]:
    """Validate an upstream URL and return its hostname and pinned public IP."""
    try:
        parsed = urlsplit(str(raw_url or ""))
        hostname = (parsed.hostname or "").rstrip(".").lower()
        port = parsed.port or 443
    except ValueError as exc:
        raise RuntimeError("model_unavailable") from exc
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username
        or parsed.password
        or hostname == "localhost"
        or hostname.endswith(".local")
    ):
        raise RuntimeError("model_unavailable")
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise RuntimeError("upstream_error") from exc
    if not addresses:
        raise RuntimeError("model_unavailable")
    public_ips: list[str] = []
    for _family, _socktype, _proto, _canonname, address in addresses:
        try:
            candidate = ipaddress.ip_address(address[0])
        except ValueError as exc:
            raise RuntimeError("model_unavailable") from exc
        if (
            not candidate.is_global
            or candidate.is_private
            or candidate.is_reserved
            or candidate.is_link_local
            or candidate.is_loopback
            or candidate.is_multicast
            or candidate.is_unspecified
            or getattr(candidate, "is_site_local", False)
        ):
            raise RuntimeError("model_unavailable")
        public_ips.append(str(candidate))
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return hostname, public_ips[0], port, path


class _PinnedHTTPSResponse:
    """Context-managed response that owns its pinned HTTPS connection."""

    def __init__(
        self,
        connection: http.client.HTTPSConnection,
        response: http.client.HTTPResponse,
    ) -> None:
        self._connection = connection
        self._response = response
        self.status = response.status

    def read(self, size: int = -1) -> bytes:
        return self._response.read(size)

    def __enter__(self) -> _PinnedHTTPSResponse:
        return self

    def __exit__(self, *_args: Any) -> None:
        self._connection.close()


def urlopen(request: Request, *, timeout: int) -> _PinnedHTTPSResponse:
    """Open one no-redirect HTTPS request pinned to a validated public address."""
    hostname, pinned_ip, port, path = _public_https_url(request.full_url)
    connection = http.client.HTTPSConnection(
        hostname,
        port,
        timeout=timeout,
        context=ssl.create_default_context(),
    )
    # HTTPSConnection keeps ``hostname`` for TLS SNI/certificate validation.
    # Override only its TCP dial target with the public address just validated.
    connection._create_connection = lambda _address, dial_timeout, source_address=None: (
        socket.create_connection(  # type: ignore[attr-defined]
            (pinned_ip, port), dial_timeout, source_address
        )
    )
    host_header = f"[{hostname}]" if ":" in hostname else hostname
    if port != 443:
        host_header = f"{host_header}:{port}"
    try:
        connection.request(
            request.get_method(),
            path,
            body=request.data,
            headers={**dict(request.header_items()), "Host": host_header},
        )
        response = connection.getresponse()
        if response.status < 200 or response.status >= 300:
            # Classify from the status before touching a body that may be
            # oversized or stall until timeout. The connection is closed by
            # the exception cleanup below, and redirects are never followed.
            raise HTTPError(
                request.full_url,
                response.status,
                "upstream rejected",
                getattr(response, "headers", {}),
                None,
            )
        return _PinnedHTTPSResponse(connection, response)
    except Exception:
        connection.close()
        raise


def _request_json(
    url: str, *, method: str, headers: dict[str, str], body: Any = None
) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    request = Request(url, data=data, method=method, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            parsed = json.loads(_read_bounded(response).decode())
    except HTTPError as exc:
        category = {
            401: "authentication_error",
            402: "billing_quota_exceeded",
            403: "authentication_error",
            404: "model_unavailable",
            408: "timeout",
            429: "rate_limited",
        }.get(exc.code, "upstream_error")
        raise RuntimeError(category) from exc
    except http.client.HTTPException as exc:
        status = int(str(exc)) if str(exc).isdigit() else 0
        category = {
            401: "authentication_error",
            402: "billing_quota_exceeded",
            403: "authentication_error",
            404: "model_unavailable",
            408: "timeout",
            429: "rate_limited",
        }.get(status, "upstream_error")
        raise RuntimeError(category) from exc
    except OSError as exc:
        raise RuntimeError(
            "timeout" if "timed out" in str(exc).lower() else "upstream_error"
        ) from exc
    except ValueError as exc:
        raise RuntimeError("invalid_response") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("invalid_response")
    return parsed


def _failure_response(code: str, *, request_semantics: str) -> dict[str, Any]:
    """Expose only provider-neutral retry facts to the host orchestrator."""
    safe_code = str(code).split(":", 1)[0]
    retryable_code = safe_code in {"rate_limited", "timeout", "upstream_error"}
    definite_refusal = safe_code in {
        "authentication_error",
        "billing_quota_exceeded",
        "model_unavailable",
        "rate_limited",
    }
    definitely_no_effect = request_semantics == "safe_read" or definite_refusal
    return {
        "ok": False,
        "errors": [{"code": safe_code}],
        "failure_class": "transient" if safe_code in {"timeout", "upstream_error"} else "provider",
        "retryable": retryable_code and definitely_no_effect,
        "definitely_no_external_effect": definitely_no_effect,
        "external_effect_status": "failed" if definitely_no_effect else "timeout_unknown",
        "provider_error_code": safe_code,
    }


def _connection(request: dict[str, Any]) -> tuple[str, dict[str, str]]:
    connection = request.get("connection") if isinstance(request, dict) else {}
    credentials = connection.get("credentials") if isinstance(connection, dict) else {}
    base_url = "https://openrouter.ai/api/v1"
    api_key = str((credentials or {}).get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("authentication_error")
    return base_url, {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _pagination(request: dict[str, Any]) -> tuple[str, int, int]:
    query = request.get("query") if isinstance(request, dict) else {}
    query = query if isinstance(query, dict) else {}
    cursor = str(query.get("cursor") or "").strip()
    token = ""
    offset = 0
    if cursor:
        token, separator, raw_offset = cursor.partition(":")
        if not separator or not token or not raw_offset.isdigit():
            raise RuntimeError("invalid_response")
        offset = int(raw_offset)
    try:
        requested_limit = int(query.get("limit") or MAX_DISCOVERY_PAGE_SIZE)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid_response") from exc
    if offset < 0 or requested_limit < 1:
        raise RuntimeError("invalid_response")
    return token, offset, min(requested_limit, MAX_DISCOVERY_PAGE_SIZE)


def _new_discovery_snapshot(data: list[Any]) -> str:
    now = time.monotonic()
    token = secrets.token_urlsafe(24)
    with _discovery_snapshots_lock:
        expired = [key for key, (deadline, _) in _discovery_snapshots.items() if deadline <= now]
        for key in expired:
            _discovery_snapshots.pop(key, None)
        if len(_discovery_snapshots) >= MAX_DISCOVERY_SNAPSHOTS:
            raise RuntimeError("rate_limited")
        _discovery_snapshots[token] = (now + DISCOVERY_SNAPSHOT_TTL_SECONDS, list(data))
    return token


def _read_discovery_snapshot(token: str) -> list[Any]:
    now = time.monotonic()
    with _discovery_snapshots_lock:
        entry = _discovery_snapshots.get(token)
        if entry is None or entry[0] <= now:
            _discovery_snapshots.pop(token, None)
            raise RuntimeError("invalid_response")
        return entry[1]


def _drop_discovery_snapshot(token: str) -> None:
    with _discovery_snapshots_lock:
        _discovery_snapshots.pop(token, None)


def _json_content(value: Any) -> Any:
    return value if isinstance(value, (str, list)) else json.dumps(value, separators=(",", ":"))


def _openai_messages(messages: Any) -> list[dict[str, Any]]:
    """Translate the host's canonical message wire into OpenAI-compatible messages."""
    result: list[dict[str, Any]] = []
    for raw in messages if isinstance(messages, list) else []:
        if not isinstance(raw, dict):
            continue
        item = {key: raw[key] for key in ("role", "name", "tool_call_id") if key in raw}
        item["content"] = _json_content(raw.get("content", ""))
        if isinstance(raw.get("cache_control"), (dict, list)):
            item["cache_control"] = raw["cache_control"]
        continuation = raw.get("continuation")
        if isinstance(continuation, dict) and "reasoning_details" in continuation:
            # OpenRouter requires this opaque state to be replayed verbatim on
            # the assistant turn before a tool result.
            item["reasoning_details"] = continuation["reasoning_details"]
        if isinstance(continuation, dict):
            for key in ("reasoning", "reasoning_content"):
                if key in continuation:
                    item[key] = continuation[key]
        calls: list[dict[str, Any]] = []
        for call in raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []:
            if not isinstance(call, dict):
                continue
            call_id, name = str(call.get("id") or "").strip(), str(call.get("name") or "").strip()
            arguments = call.get("arguments")
            if call_id and name and arguments is not None:
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, separators=(",", ":"))
                            if not isinstance(arguments, str)
                            else arguments,
                        },
                    }
                )
        if calls:
            item["tool_calls"] = calls
        result.append(item)
    return result


def _canonical_message(value: Any) -> dict[str, Any]:
    message = dict(value) if isinstance(value, dict) else {}
    continuation = {
        key: message.pop(key)
        for key in ("reasoning_details", "reasoning", "reasoning_content")
        if key in message
    }
    if continuation:
        message["continuation"] = continuation
    calls: list[dict[str, Any]] = []
    for call in message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            with suppress(json.JSONDecodeError):
                arguments = json.loads(arguments)
        call_id, name = str(call.get("id") or "").strip(), str(function.get("name") or "").strip()
        if call_id and name and isinstance(arguments, dict):
            calls.append({"id": call_id, "name": name, "arguments": arguments})
    if calls:
        message["tool_calls"] = calls
    else:
        message.pop("tool_calls", None)
    return message


def _owner_label(item: dict[str, Any], owner_key: str) -> str:
    """Derive a stable human label from OpenRouter's model display metadata."""
    display_name = str(item.get("name") or "").strip()
    publisher, separator, model_name = display_name.partition(":")
    publisher = publisher.strip()
    normalized_publisher = re.sub(r"[^a-z0-9]+", "-", publisher.lower()).strip("-")
    family = re.split(r"[\s:/-]+", model_name.strip(), maxsplit=1)[0] if separator else ""
    normalized_family = re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-")
    if publisher and normalized_publisher == owner_key:
        if publisher.lower().endswith(".ai") and family:
            return f"{publisher} / {family.upper()}"
        return publisher
    if (
        publisher
        and normalized_family
        and owner_key == f"{normalized_publisher}-{normalized_family}"
    ):
        return f"{publisher} {family.title()}"
    return " ".join(part.upper() if part == "ai" else part.title() for part in owner_key.split("-"))


def list_models(request: dict[str, Any]) -> dict[str, Any]:
    base_url, headers = _connection(request)
    token, offset, limit = _pagination(request)
    if token:
        data = _read_discovery_snapshot(token)
    else:
        data = _request_json(f"{base_url}/models", method="GET", headers=headers).get("data", [])
        if not isinstance(data, list):
            raise RuntimeError("invalid_response")
    models = []
    page_data = data[offset : offset + limit]
    for item in page_data:
        if not isinstance(item, dict) or not str(item.get("id") or "").strip():
            continue
        params = item.get("supported_parameters") or []
        caps = ["chat"]
        if "tools" in params:
            caps.append("tools")
        if "structured_outputs" in params or "response_format" in params:
            caps.append("structured_output")
        canonical_slug = str(item.get("canonical_slug") or "").strip()
        owner_key = canonical_slug.split("/", 1)[0]
        # This is an explicit per-model discovery capability.  It is not a
        # routing requirement, but lets the host preserve Anthropic cache
        # blocks for OpenRouter routes known to support them.
        if canonical_slug.startswith("anthropic/") and "prompt_caching" not in caps:
            caps.append("prompt_caching")
        models.append(
            {
                "model_id": item["id"],
                "label": item.get("name") or item["id"],
                "upstream_owner_key": owner_key,
                "upstream_owner_label": _owner_label(item, owner_key),
                "capabilities": caps,
                "metadata": {"supported_parameters": params, "canonical_slug": canonical_slug},
                "upstream_revision": canonical_slug or None,
            }
        )
    next_offset = offset + len(page_data)
    if next_offset < len(data):
        token = token or _new_discovery_snapshot(data)
        next_cursor = f"{token}:{next_offset}"
    else:
        if token:
            _drop_discovery_snapshot(token)
        next_cursor = ""
    return {"models": models, "next_cursor": next_cursor, "snapshot_complete": True}


def chat(request: dict[str, Any], *, structured: bool = False) -> dict[str, Any]:
    payload = dict(request.get("input") or {})
    base_url, headers = _connection(request)
    body = {
        key: payload[key]
        for key in (
            "model_id",
            "temperature",
            "max_tokens",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
        )
        if key in payload
    }
    body["messages"] = _openai_messages(payload.get("messages"))
    body["model"] = body.pop("model_id")
    # Discovery metadata is advisory. Require OpenRouter to select a route
    # that actually accepts the requested tools/strict response format.
    if structured or body.get("tools"):
        body["provider"] = {"require_parameters": True}
    if structured:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": payload.get("schema_name") or "Schema",
                "strict": bool(payload.get("strict", True)),
                "schema": payload.get("schema") or {},
            },
        }
    response = _request_json(
        f"{base_url}/chat/completions", method="POST", headers=headers, body=body
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("invalid_response")
    choice = choices[0]
    if (
        not isinstance(choice, dict)
        or not isinstance(choice.get("message"), dict)
        or choice.get("error") is not None
        or choice.get("finish_reason") == "error"
    ):
        raise RuntimeError("invalid_response")
    message = _canonical_message(choice["message"])
    usage = response.get("usage") or {}
    canonical_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cache_creation_input_tokens": (
            usage.get("cache_creation_input_tokens", 0)
            or (usage.get("prompt_tokens_details") or {}).get("cache_write_tokens", 0)
        ),
        "cache_read_input_tokens": (
            usage.get("cache_read_input_tokens", 0)
            or (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        ),
    }
    reported_cost = usage.get("cost")
    if (
        not isinstance(reported_cost, bool)
        and isinstance(reported_cost, (int, float))
        and math.isfinite(float(reported_cost))
        and float(reported_cost) >= 0
    ):
        canonical_usage["cost_usd"] = float(reported_cost)
    result = {
        "message": message or {},
        "finish_reason": choice.get("finish_reason") or "",
        "usage": canonical_usage,
    }
    if structured:
        result["structured_output"] = (message or {}).get("content")
    return result


class Handler(BaseHTTPRequestHandler):
    token = os.environ.get("FS_EXTENSION_TOKEN", "")

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode()
        if len(raw) > MAX_RUNTIME_BODY_BYTES:
            status = 502
            raw = b'{"ok":false,"errors":[{"code":"invalid_response"}]}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self._send(
            200,
            {"ok": True, "status": "ready", "extension_id": "flowsteward.openrouter"}
            if self.path == "/health"
            else {"ok": False},
        )

    def do_POST(self) -> None:
        if not self.token:
            self._send(503, {"ok": False, "errors": [{"code": "runtime_auth_not_configured"}]})
            return
        if self.headers.get("Authorization") != f"Bearer {self.token}":
            self._send(401, {"ok": False, "errors": [{"code": "unauthorized"}]})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(_read_request_body(self.rfile, content_length).decode() or "{}")
            if self.path == "/handshake":
                result = {
                    "extension_id": "flowsteward.openrouter",
                    "contract_version": "extension_host_v1",
                }
            elif self.path == "/queries/llm.list_models":
                result = list_models(body)
            elif self.path == "/actions/llm.chat":
                result = _execute_idempotent_action(body, "llm.chat", lambda: chat(body))
            elif self.path == "/actions/llm.chat_structured":
                result = _execute_idempotent_action(
                    body, "llm.chat_structured", lambda: chat(body, structured=True)
                )
            elif self.path == "/events":
                result = {"accepted": True}
            else:
                self._send(404, {"ok": False, "errors": [{"code": "not_found"}]})
                return
            self._send(200, {"ok": True, "result": result, "errors": []})
        except Exception as exc:
            semantics = "safe_read" if self.path == "/queries/llm.list_models" else "mutation"
            self._send(502, _failure_response(str(exc), request_semantics=semantics))

    def log_message(self, *_: Any) -> None:
        return


def _server_address() -> tuple[str, int]:
    host = str(os.environ.get("FS_EXTENSION_BIND_HOST") or "127.0.0.1").strip()
    if not host:
        raise RuntimeError("FS_EXTENSION_BIND_HOST must not be empty")
    return host, int(os.environ.get("FS_EXTENSION_PORT", "8090"))


if __name__ == "__main__":
    if not Handler.token:
        raise SystemExit("FS_EXTENSION_TOKEN is required for the OpenRouter remote runtime")
    ThreadingHTTPServer(_server_address(), Handler).serve_forever()
