"""The HTTP layer: one fixed public upstream, DNS pinning, no redirects, bounds.

Every upstream request goes through ``runtime._request_json`` and ``urlopen``.
These tests replace only the socket layer (``http.client.HTTPSConnection`` and
DNS), so the URL checks, status mapping and body bounds under test are real.
"""

from __future__ import annotations

import socket

import pytest


def test_connection_does_not_allow_openrouter_upstream_override(runtime) -> None:
    base_url, _headers = runtime._connection(
        {
            "connection": {
                "connection_config": {"upstream_base_url": "https://untrusted.example"},
                "credentials": {"api_key": "secret"},
            }
        }
    )
    assert base_url == "https://openrouter.ai/api/v1"


@pytest.mark.parametrize(
    ("status_code", "category"),
    [(402, "billing_quota_exceeded"), (429, "rate_limited")],
)
def test_http_errors_are_normalized_to_safe_categories(
    runtime, monkeypatch, status_code: int, category: str
) -> None:

    monkeypatch.setattr(
        runtime,
        "_public_https_url",
        lambda _url: ("openrouter.ai", "8.8.8.8", 443, "/api/v1/models"),
    )

    class _Response:
        status = status_code

        def read(self, _size=-1):
            return b'{"error":"upstream rejected"}'

    class _Connection:
        def __init__(self, *_args, **_kwargs):
            self._create_connection = None

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return _Response()

        def close(self):
            return None

    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _Connection)
    try:
        runtime._request_json("https://openrouter.ai/api/v1/models", method="GET", headers={})
    except RuntimeError as exc:
        assert str(exc) == category
    else:  # pragma: no cover - failure diagnostics
        raise AssertionError("HTTP error was not raised")


@pytest.mark.parametrize(
    ("status_code", "category"),
    ((402, "billing_quota_exceeded"), (429, "rate_limited")),
)
def test_http_status_category_precedes_oversized_error_body(
    runtime, monkeypatch, status_code: int, category: str
) -> None:
    oversized = b"x" * (runtime.MAX_RUNTIME_BODY_BYTES + 1)
    closed = False
    reads = 0
    monkeypatch.setattr(
        runtime,
        "_public_https_url",
        lambda _url: ("openrouter.ai", "8.8.8.8", 443, "/api/v1/models"),
    )

    class _Response:
        status = status_code
        headers = ()

        def read(self, _size=-1):
            nonlocal reads
            reads += 1
            return oversized

    class _Connection:
        def __init__(self, *_args, **_kwargs):
            self._create_connection = None

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return _Response()

        def close(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _Connection)

    with pytest.raises(RuntimeError, match=category):
        runtime._request_json("https://openrouter.ai/api/v1/models", method="GET", headers={})

    assert reads == 0
    assert closed is True


def test_http_status_category_precedes_stalled_error_body(runtime, monkeypatch) -> None:
    closed = False
    reads = 0
    monkeypatch.setattr(
        runtime,
        "_public_https_url",
        lambda _url: ("openrouter.ai", "8.8.8.8", 443, "/api/v1/models"),
    )

    class _Response:
        status = 429
        headers = ()

        def read(self, _size=-1):
            nonlocal reads
            reads += 1
            raise TimeoutError("timed out")

    class _Connection:
        def __init__(self, *_args, **_kwargs):
            self._create_connection = None

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return _Response()

        def close(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _Connection)

    with pytest.raises(RuntimeError, match="rate_limited"):
        runtime._request_json("https://openrouter.ai/api/v1/models", method="GET", headers={})

    assert reads == 0
    assert closed is True


def test_request_json_preserves_the_urlopen_test_seam(runtime, monkeypatch) -> None:
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size=-1):
            return b'{"data":[]}'

    def open_request(request, *, timeout):
        captured.update(request=request, timeout=timeout)
        return _Response()

    monkeypatch.setattr(runtime, "urlopen", open_request)

    result = runtime._request_json(
        "https://openrouter.ai/api/v1/models",
        method="GET",
        headers={"Authorization": "Bearer provider-secret"},
    )

    assert result == {"data": []}
    assert captured["request"].full_url == "https://openrouter.ai/api/v1/models"
    assert captured["request"].get_header("Authorization") == "Bearer provider-secret"
    assert captured["timeout"] == 60


@pytest.mark.parametrize(
    "value",
    (
        "http://openrouter.ai/api/v1/models",
        "https://localhost/api/v1/models",
        "https://router.local/api/v1/models",
    ),
)
def test_openrouter_rejects_non_public_https_hostnames(runtime, value: str) -> None:

    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._public_https_url(value)


@pytest.mark.parametrize(
    ("address", "family"),
    (
        ("127.0.0.1", socket.AF_INET),
        ("10.0.0.1", socket.AF_INET),
        ("169.254.1.1", socket.AF_INET),
        ("203.0.113.8", socket.AF_INET),
        ("100.64.0.1", socket.AF_INET),
        ("224.0.0.1", socket.AF_INET),
        ("5f00::1", socket.AF_INET6),
        ("fec0::1", socket.AF_INET6),
        ("ff02::1", socket.AF_INET6),
    ),
)
def test_openrouter_rejects_non_public_dns_answers(
    runtime, monkeypatch, address: str, family: socket.AddressFamily
) -> None:
    monkeypatch.setattr(
        runtime.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(family, socket.SOCK_STREAM, 6, "", (address, 443))],
    )

    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._public_https_url("https://openrouter.ai/api/v1/models")


@pytest.mark.parametrize(
    ("unsafe_address", "family"),
    (
        ("127.0.0.1", socket.AF_INET),
        ("224.0.0.1", socket.AF_INET),
        ("5f00::1", socket.AF_INET6),
        ("fec0::1", socket.AF_INET6),
        ("ff02::1", socket.AF_INET6),
    ),
)
def test_openrouter_rejects_mixed_public_and_non_public_dns_answers(
    runtime, monkeypatch, unsafe_address: str, family: socket.AddressFamily
) -> None:
    monkeypatch.setattr(
        runtime.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (family, socket.SOCK_STREAM, 6, "", (unsafe_address, 443)),
        ],
    )

    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._public_https_url("https://openrouter.ai/api/v1/models")


def test_openrouter_accepts_public_ipv4_and_ipv6_dns_answers(runtime, monkeypatch) -> None:
    monkeypatch.setattr(
        runtime.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                6,
                "",
                ("2606:4700:4700::1111", 443),
            ),
        ],
    )

    target = runtime._public_https_url("https://openrouter.ai/api/v1/models")

    assert target == ("openrouter.ai", "8.8.8.8", 443, "/api/v1/models")


def test_openrouter_rejects_missing_dns_answers(runtime, monkeypatch) -> None:
    monkeypatch.setattr(runtime.socket, "getaddrinfo", lambda *_args, **_kwargs: [])

    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._public_https_url("https://openrouter.ai/api/v1/models")


def test_openrouter_normalizes_dns_resolution_failures(runtime, monkeypatch) -> None:

    def fail_resolution(*_args, **_kwargs):
        raise socket.gaierror("missing")

    monkeypatch.setattr(runtime.socket, "getaddrinfo", fail_resolution)

    with pytest.raises(RuntimeError, match="upstream_error"):
        runtime._public_https_url("https://openrouter.ai/api/v1/models")


def test_openrouter_reresolves_dns_for_every_request_and_rejects_rebinding(
    runtime, monkeypatch
) -> None:
    answers = iter(
        [
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
        ]
    )
    resolutions = 0
    connections = 0

    def resolve(*_args, **_kwargs):
        nonlocal resolutions
        resolutions += 1
        return next(answers)

    class _Response:
        status = 200

        def read(self, _size=-1):
            return b"{}"

    class _Connection:
        def __init__(self, *_args, **_kwargs):
            nonlocal connections
            connections += 1
            self._create_connection = None

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return _Response()

        def close(self):
            return None

    monkeypatch.setattr(runtime.socket, "getaddrinfo", resolve)
    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _Connection)

    runtime._request_json("https://openrouter.ai/api/v1/models", method="GET", headers={})
    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._request_json("https://openrouter.ai/api/v1/models", method="GET", headers={})

    assert resolutions == 2
    assert connections == 1


def test_openrouter_pins_tcp_while_preserving_tls_hostname_and_host_header(
    runtime, monkeypatch
) -> None:
    captured = {}
    monkeypatch.setattr(
        runtime,
        "_public_https_url",
        lambda _url: ("openrouter.ai", "8.8.8.8", 443, "/api/v1/models?x=1"),
    )

    class _Response:
        status = 200

        def read(self, _size=-1):
            return b'{"data":[]}'

    class _Connection:
        def __init__(self, host, port, *, timeout, context):
            captured.update(
                tls_hostname=host,
                port=port,
                timeout=timeout,
                context=context,
            )
            self._create_connection = None

        def request(self, method, path, *, body, headers):
            captured.update(method=method, path=path, body=body, headers=headers)
            self._create_connection(("openrouter.ai", 443), 60, None)

        def getresponse(self):
            return _Response()

        def close(self):
            captured["closed"] = True

    def create_connection(address, timeout, source_address=None):
        captured.update(
            tcp_address=address,
            tcp_timeout=timeout,
            source_address=source_address,
        )
        return object()

    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _Connection)
    monkeypatch.setattr(runtime.socket, "create_connection", create_connection)

    result = runtime._request_json(
        "https://openrouter.ai/api/v1/models?x=1",
        method="GET",
        headers={"Authorization": "Bearer provider-secret"},
    )

    assert result == {"data": []}
    assert captured["tls_hostname"] == "openrouter.ai"
    assert captured["tcp_address"] == ("8.8.8.8", 443)
    assert captured["headers"] == {
        "Authorization": "Bearer provider-secret",
        "Host": "openrouter.ai",
    }
    assert captured["path"] == "/api/v1/models?x=1"
    assert captured["closed"] is True


def test_openrouter_request_refuses_redirect_before_forwarding_authorization(
    runtime, monkeypatch
) -> None:
    requests = []
    monkeypatch.setattr(
        runtime,
        "_public_https_url",
        lambda _url: ("openrouter.ai", "8.8.8.8", 443, "/api/v1/models"),
    )

    class _Response:
        status = 302

        def read(self, _size=-1):
            return b"{}"

    class _Connection:
        def __init__(self, *_args, **_kwargs):
            self._create_connection = None

        def request(self, method, path, *, body, headers):
            requests.append((method, path, body, headers))

        def getresponse(self):
            return _Response()

        def close(self):
            return None

    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _Connection)

    with pytest.raises(RuntimeError, match="upstream_error"):
        runtime._request_json(
            "https://openrouter.ai/api/v1/models",
            method="GET",
            headers={"Authorization": "Bearer provider-secret"},
        )

    assert len(requests) == 1
    assert requests[0][3]["Authorization"] == "Bearer provider-secret"


def test_request_and_upstream_bodies_are_bounded(runtime, monkeypatch) -> None:
    oversized = b"x" * (runtime.MAX_RUNTIME_BODY_BYTES + 1)

    monkeypatch.setattr(
        runtime,
        "_public_https_url",
        lambda _url: ("openrouter.ai", "8.8.8.8", 443, "/api/v1/models"),
    )

    class _Response:
        status = 200

        def read(self, _size=-1):
            return oversized

    class _Connection:
        def __init__(self, *_args, **_kwargs):
            self._create_connection = None

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return _Response()

        def close(self):
            return None

    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _Connection)
    try:
        runtime._request_json("https://openrouter.ai/api/v1/models", method="GET", headers={})
    except RuntimeError as exc:
        assert str(exc) == "invalid_response"
    else:  # pragma: no cover - failure diagnostics
        raise AssertionError("oversized upstream response was accepted")


class _StatusResponse:
    headers = ()

    def __init__(self, status: int) -> None:
        self.status = status

    def read(self, _size: int = -1) -> bytes:
        return b"{}"


def _fixed_connection(status: int = 200, error: Exception | None = None):
    class _Connection:
        def __init__(self, *_args, **_kwargs):
            self._create_connection = None

        def request(self, *_args, **_kwargs):
            if error is not None:
                raise error

        def getresponse(self):
            return _StatusResponse(status)

        def close(self):
            return None

    return _Connection


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (400, "upstream_error"),
        (401, "authentication_error"),
        (402, "billing_quota_exceeded"),
        (403, "authentication_error"),
        (404, "model_unavailable"),  # also: no endpoint accepts the parameters
        (408, "timeout"),
        (429, "rate_limited"),
        (500, "upstream_error"),
        (502, "upstream_error"),
    ],
)
def test_every_http_status_maps_to_one_safe_category(
    runtime, monkeypatch, status: int, category: str
) -> None:
    monkeypatch.setattr(
        runtime, "_public_https_url", lambda _url: ("openrouter.ai", "8.8.8.8", 443, "/")
    )
    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _fixed_connection(status))

    with pytest.raises(RuntimeError) as caught:
        runtime._request_json("https://openrouter.ai/api/v1/models", method="GET", headers={})

    assert str(caught.value) == category


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (TimeoutError("timed out"), "timeout"),
        (ConnectionResetError("reset by peer"), "upstream_error"),
        (OSError("network is unreachable"), "upstream_error"),
    ],
)
def test_network_failures_map_to_safe_categories(runtime, monkeypatch, error, category) -> None:
    monkeypatch.setattr(
        runtime, "_public_https_url", lambda _url: ("openrouter.ai", "8.8.8.8", 443, "/")
    )
    monkeypatch.setattr(runtime.http.client, "HTTPSConnection", _fixed_connection(error=error))

    with pytest.raises(RuntimeError) as caught:
        runtime._request_json("https://openrouter.ai/api/v1/models", method="GET", headers={})

    assert str(caught.value) == category
