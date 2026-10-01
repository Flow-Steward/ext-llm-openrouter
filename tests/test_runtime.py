from __future__ import annotations

import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

import pytest
import yaml


def _runtime():
    path = Path(__file__).parents[1] / "runtime.py"
    spec = importlib.util.spec_from_file_location("openrouter_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_failure_response_retries_discovery_but_not_ambiguous_chat() -> None:
    runtime = _runtime()
    discovery = runtime._failure_response("rate_limited", request_semantics="safe_read")
    chat = runtime._failure_response("timeout", request_semantics="mutation")

    assert (
        discovery.items()
        >= {
            "failure_class": "provider",
            "retryable": True,
            "definitely_no_external_effect": True,
            "external_effect_status": "failed",
            "provider_error_code": "rate_limited",
        }.items()
    )
    assert (
        chat.items()
        >= {
            "failure_class": "transient",
            "retryable": False,
            "definitely_no_external_effect": False,
            "external_effect_status": "timeout_unknown",
            "provider_error_code": "timeout",
        }.items()
    )


def test_shipped_manifest_exposes_llm_provider_contract() -> None:
    # Structural check only: bundle tests must stay platform-independent, so
    # the typed-parse guarantee lives in the platform suite
    # (tests/unit/test_llm_provider_extensions.py).
    manifest = yaml.safe_load(
        (Path(__file__).parents[1] / "extension.yaml").read_text(encoding="utf-8")
    )

    contract = manifest["runtime"]["extension_contract_v2"]["llm_provider"]

    assert contract["schema_version"] == "llm_provider_extension_v1"
    assert contract["runtime_mode"] == "remote_https"
    assert contract["optional_provider_secrets"] == []
    assert set(contract["required_provider_secrets"]) == {"api_key", "runtime_bearer_token"}
    assert contract["operations"]["list_models"]["query_id"] == "llm.list_models"
    assert contract["operations"]["chat"]["action_id"] == "llm.chat"


def test_list_models_fixture_is_a_complete_snapshot() -> None:
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "llm.list_models.json").read_text()
    )
    assert fixture["result"]["snapshot_complete"] is True


def test_chat_action_reuses_completed_result_for_same_idempotency_key() -> None:
    runtime = _runtime()
    calls = 0
    request = {"idempotency_key": "chat:account:model:request", "request": {"model": "m"}}

    def invoke():
        nonlocal calls
        calls += 1
        return {"message": {"role": "assistant", "content": "ok"}}

    first = runtime._execute_idempotent_action(request, "llm.chat", invoke)
    second = runtime._execute_idempotent_action(request, "llm.chat", invoke)

    assert first == second
    assert calls == 1


def test_chat_action_coalesces_concurrent_requests() -> None:
    runtime = _runtime()
    calls = 0
    request = {"idempotency_key": "chat:concurrent", "request": {"model": "m"}}
    started = threading.Event()
    release = threading.Event()
    results: list[dict] = []

    def invoke():
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=1)
        return {"message": {"role": "assistant", "content": "ok"}}

    def run():
        results.append(runtime._execute_idempotent_action(request, "llm.chat", invoke))

    first = threading.Thread(target=run)
    second = threading.Thread(target=run)
    first.start()
    assert started.wait(timeout=1)
    second.start()
    time.sleep(0.02)
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert calls == 1
    assert len(results) == 2
    assert results[0] == results[1]


def test_discovery_preserves_openrouter_slug_and_capabilities(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda *_args, **_kwargs: {
            "data": [
                {
                    "id": "deepseek/deepseek-chat",
                    "name": "DeepSeek: DeepSeek Chat",
                    "canonical_slug": "deepseek/deepseek-chat",
                    "supported_parameters": ["tools", "structured_outputs"],
                }
            ]
        },
    )
    result = runtime.list_models({"connection": {"credentials": {"api_key": "secret"}}})
    assert result["models"][0]["model_id"] == "deepseek/deepseek-chat"
    assert result["models"][0]["upstream_revision"] == "deepseek/deepseek-chat"
    assert set(result["models"][0]["capabilities"]) == {"chat", "tools", "structured_output"}
    assert result["models"][0]["upstream_owner_label"] == "DeepSeek"


def test_discovery_derives_documented_owner_labels_from_model_metadata(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda *_args, **_kwargs: {
            "data": [
                {
                    "id": "z-ai/glm-4.5",
                    "name": "Z.AI: GLM 4.5",
                    "canonical_slug": "z-ai/glm-4.5",
                },
                {
                    "id": "meta-llama/llama-3.3-70b-instruct",
                    "name": "Meta: Llama 3.3 70B Instruct",
                    "canonical_slug": "meta-llama/llama-3.3-70b-instruct",
                },
            ]
        },
    )

    result = runtime.list_models({"connection": {"credentials": {"api_key": "secret"}}})

    assert [model["upstream_owner_label"] for model in result["models"]] == [
        "Z.AI / GLM",
        "Meta Llama",
    ]


def test_discovery_uses_canonical_slug_as_revision_when_id_is_an_alias(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda *_args, **_kwargs: {
            "data": [{"id": "vendor/latest", "canonical_slug": "vendor/model-2026-07"}]
        },
    )
    result = runtime.list_models({"connection": {"credentials": {"api_key": "secret"}}})
    assert result["models"][0]["model_id"] == "vendor/latest"
    assert result["models"][0]["upstream_revision"] == "vendor/model-2026-07"


def test_discovery_paginates_the_openrouter_catalog(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda *_args, **_kwargs: {
            "data": [
                {"id": f"owner/model-{index}", "canonical_slug": f"owner/model-{index}"}
                for index in range(205)
            ]
        },
    )
    request = {
        "connection": {"credentials": {"api_key": "secret"}},
        "query": {"cursor": "", "limit": 200},
    }

    first = runtime.list_models(request)
    second = runtime.list_models(
        {**request, "query": {"cursor": first["next_cursor"], "limit": 200}}
    )

    assert len(first["models"]) == 200
    assert first["next_cursor"].endswith(":200")
    assert len(second["models"]) == 5
    assert second["next_cursor"] == ""


def test_discovery_pages_share_one_immutable_upstream_snapshot(monkeypatch) -> None:
    runtime = _runtime()
    upstream_calls = 0

    def fake_request(*_args, **_kwargs):
        nonlocal upstream_calls
        upstream_calls += 1
        start = 0 if upstream_calls == 1 else 1
        return {
            "data": [
                {"id": f"owner/model-{index}", "canonical_slug": f"owner/model-{index}"}
                for index in range(start, 205)
            ]
        }

    monkeypatch.setattr(runtime, "_request_json", fake_request)
    request = {
        "connection": {"credentials": {"api_key": "secret"}},
        "query": {"cursor": "", "limit": 200},
    }

    first = runtime.list_models(request)
    second = runtime.list_models(
        {**request, "query": {"cursor": first["next_cursor"], "limit": 200}}
    )

    assert upstream_calls == 1
    assert [model["model_id"] for model in second["models"]] == [
        f"owner/model-{index}" for index in range(200, 205)
    ]


def test_discovery_rejects_new_session_instead_of_evicting_active_cursor(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.setattr(runtime, "MAX_DISCOVERY_SNAPSHOTS", 4)
    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda *_args, **_kwargs: {
            "data": [
                {"id": "owner/model-1", "canonical_slug": "owner/model-1"},
                {"id": "owner/model-2", "canonical_slug": "owner/model-2"},
            ]
        },
    )
    request = {
        "connection": {"credentials": {"api_key": "secret"}},
        "query": {"cursor": "", "limit": 1},
    }

    cursors = [runtime.list_models(request)["next_cursor"] for _ in range(4)]
    with pytest.raises(RuntimeError, match="rate_limited"):
        runtime.list_models(request)

    resumed = runtime.list_models({**request, "query": {"cursor": cursors[0], "limit": 1}})
    assert [model["model_id"] for model in resumed["models"]] == ["owner/model-2"]


def test_reference_runtime_binds_loopback_by_default(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.delenv("FS_EXTENSION_BIND_HOST", raising=False)
    monkeypatch.setenv("FS_EXTENSION_PORT", "8099")

    assert runtime._server_address() == ("127.0.0.1", 8099)


def test_structured_chat_forwards_exact_schema(monkeypatch) -> None:
    runtime = _runtime()
    captured = {}

    def fake_request(_url, **kwargs):
        captured["body"] = kwargs["body"]
        return {
            "choices": [{"message": {"content": '{"status":"ok"}'}, "finish_reason": "stop"}],
            "usage": {},
        }

    monkeypatch.setattr(
        runtime,
        "_request_json",
        fake_request,
    )
    runtime.chat(
        {
            "connection": {"credentials": {"api_key": "secret"}},
            "input": {
                "model_id": "deepseek/deepseek-chat",
                "messages": [],
                "schema": {"type": "object", "additionalProperties": False},
            },
        },
        structured=True,
    )
    assert (
        captured["body"]["response_format"]["json_schema"]["schema"]["additionalProperties"]
        is False
    )
    assert captured["body"]["provider"] == {"require_parameters": True}


@pytest.mark.parametrize("reported_cost", [0, 0.0123])
def test_chat_preserves_finite_nonnegative_upstream_cost(monkeypatch, reported_cost) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda *_args, **_kwargs: {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cost": reported_cost,
            },
        },
    )

    result = runtime.chat(
        {
            "connection": {"credentials": {"api_key": "secret"}},
            "input": {"model_id": "owner/model", "messages": []},
        }
    )

    assert result["usage"]["cost_usd"] == reported_cost


@pytest.mark.parametrize("reported_cost", [-1, float("inf"), float("nan"), "invalid"])
def test_chat_omits_invalid_upstream_cost(monkeypatch, reported_cost) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda *_args, **_kwargs: {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": reported_cost},
        },
    )

    result = runtime.chat(
        {
            "connection": {"credentials": {"api_key": "secret"}},
            "input": {"model_id": "owner/model", "messages": []},
        }
    )

    assert "cost_usd" not in result["usage"]


@pytest.mark.parametrize(
    "response",
    [
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {"content": "ok"}}, {"message": {"content": "extra"}}]},
        {
            "choices": [
                {
                    "message": {"content": "partial"},
                    "finish_reason": "error",
                    "error": {"code": 502, "message": "provider disconnected"},
                }
            ]
        },
    ],
)
def test_chat_rejects_malformed_non_streaming_completion(monkeypatch, response) -> None:
    runtime = _runtime()
    monkeypatch.setattr(runtime, "_request_json", lambda *_args, **_kwargs: response)

    with pytest.raises(RuntimeError, match="invalid_response"):
        runtime.chat(
            {
                "connection": {"credentials": {"api_key": "secret"}},
                "input": {"model_id": "owner/model", "messages": []},
            }
        )


def test_chat_translates_canonical_tool_wire_to_openai_and_back(monkeypatch) -> None:
    runtime = _runtime()
    captured = {}

    def fake_request(_url, **kwargs):
        captured["body"] = kwargs["body"]
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_probe",
                                "type": "function",
                                "function": {
                                    "name": "flow_steward_capability_probe",
                                    "arguments": '{"key":"probe"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(runtime, "_request_json", fake_request)
    result = runtime.chat(
        {
            "connection": {"credentials": {"api_key": "secret"}},
            "input": {
                "model_id": "deepseek/deepseek-chat",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "flow_steward_capability_probe",
                            "description": "Return the probe value.",
                            "parameters": {
                                "type": "object",
                                "properties": {"key": {"type": "string"}},
                                "required": ["key"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_previous", "name": "lookup", "arguments": {"q": "x"}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_previous", "content": {"value": "ok"}},
                ],
            },
        }
    )

    upstream_calls = captured["body"]["messages"][0]["tool_calls"]
    assert upstream_calls == [
        {
            "id": "call_previous",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }
    ]
    assert captured["body"]["messages"][1]["tool_call_id"] == "call_previous"
    assert captured["body"]["tools"][0]["function"]["parameters"]["required"] == ["key"]
    assert result["message"]["tool_calls"] == [
        {"id": "call_probe", "name": "flow_steward_capability_probe", "arguments": {"key": "probe"}}
    ]


def test_chat_preserves_reasoning_continuation_for_tool_replay(monkeypatch) -> None:
    runtime = _runtime()
    captured = {}

    def fake_request(_url, **kwargs):
        captured["body"] = kwargs["body"]
        return {"choices": [{"message": {"content": "", "reasoning_details": [{"id": "r1"}]}}]}

    monkeypatch.setattr(runtime, "_request_json", fake_request)
    result = runtime.chat(
        {
            "connection": {"credentials": {"api_key": "secret"}},
            "input": {
                "model_id": "deepseek/deepseek-chat",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "continuation": {"reasoning_details": [{"id": "prior"}]},
                    }
                ],
            },
        }
    )
    assert captured["body"]["messages"][0]["reasoning_details"] == [{"id": "prior"}]
    assert result["message"]["continuation"] == {"reasoning_details": [{"id": "r1"}]}


def test_connection_does_not_allow_openrouter_upstream_override() -> None:
    runtime = _runtime()
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
    monkeypatch, status_code: int, category: str
) -> None:
    runtime = _runtime()

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
    monkeypatch, status_code: int, category: str
) -> None:
    runtime = _runtime()
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


def test_http_status_category_precedes_stalled_error_body(monkeypatch) -> None:
    runtime = _runtime()
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


def test_request_json_preserves_the_urlopen_test_seam(monkeypatch) -> None:
    runtime = _runtime()
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
def test_openrouter_rejects_non_public_https_hostnames(value: str) -> None:
    runtime = _runtime()

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
    monkeypatch, address: str, family: socket.AddressFamily
) -> None:
    runtime = _runtime()
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
    monkeypatch, unsafe_address: str, family: socket.AddressFamily
) -> None:
    runtime = _runtime()
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


def test_openrouter_accepts_public_ipv4_and_ipv6_dns_answers(monkeypatch) -> None:
    runtime = _runtime()
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


def test_openrouter_rejects_missing_dns_answers(monkeypatch) -> None:
    runtime = _runtime()
    monkeypatch.setattr(runtime.socket, "getaddrinfo", lambda *_args, **_kwargs: [])

    with pytest.raises(RuntimeError, match="model_unavailable"):
        runtime._public_https_url("https://openrouter.ai/api/v1/models")


def test_openrouter_normalizes_dns_resolution_failures(monkeypatch) -> None:
    runtime = _runtime()

    def fail_resolution(*_args, **_kwargs):
        raise socket.gaierror("missing")

    monkeypatch.setattr(runtime.socket, "getaddrinfo", fail_resolution)

    with pytest.raises(RuntimeError, match="upstream_error"):
        runtime._public_https_url("https://openrouter.ai/api/v1/models")


def test_openrouter_reresolves_dns_for_every_request_and_rejects_rebinding(monkeypatch) -> None:
    runtime = _runtime()
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


def test_openrouter_pins_tcp_while_preserving_tls_hostname_and_host_header(monkeypatch) -> None:
    runtime = _runtime()
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


def test_openrouter_request_refuses_redirect_before_forwarding_authorization(monkeypatch) -> None:
    runtime = _runtime()
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


def test_request_and_upstream_bodies_are_bounded(monkeypatch) -> None:
    import io

    runtime = _runtime()
    oversized = b"x" * (runtime.MAX_RUNTIME_BODY_BYTES + 1)

    try:
        runtime._read_request_body(io.BytesIO(oversized), len(oversized))
    except RuntimeError as exc:
        assert str(exc) == "invalid_response"
    else:  # pragma: no cover - failure diagnostics
        raise AssertionError("oversized runtime request was accepted")

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
