"""``llm.list_models``: one page of OpenRouter's catalog per call.

Each call is a separate subprocess, so pagination is stateless: the cursor
carries a fingerprint of the catalog plus an offset, and every page re-reads
``/models`` to prove it is still the same catalog.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import connection

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _catalog(count: int) -> list[dict]:
    return [{"id": f"owner/model-{i}", "canonical_slug": f"owner/model-{i}"} for i in range(count)]


def _list(runtime, *, cursor: str = "", limit: int = 200, api_key: str = "key-1"):
    return runtime.list_models(
        {"connection": connection(api_key), "query": {"cursor": cursor, "limit": limit}}
    )


def test_models_carry_owner_grouping_and_capabilities(runtime, openrouter) -> None:
    openrouter.models = [
        {
            "id": "openai/gpt-4o-mini",
            "name": "OpenAI: GPT-4o-mini",
            "canonical_slug": "openai/gpt-4o-mini",
            "supported_parameters": ["tools", "tool_choice", "structured_outputs"],
        }
    ]

    result = _list(runtime)

    assert result == json.loads((FIXTURES / "llm.list_models.json").read_text())["result"]
    key_check, call = openrouter.calls
    assert key_check["url"] == "https://openrouter.ai/api/v1/key"
    assert call["url"] == "https://openrouter.ai/api/v1/models"
    assert all(c["headers"]["Authorization"] == "Bearer key-1" for c in openrouter.calls)


@pytest.mark.parametrize(
    ("parameters", "capabilities"),
    [
        (["tools", "structured_outputs"], ["chat", "tools", "structured_output"]),
        # JSON mode alone is not strict structured output.
        (["tools", "response_format"], ["chat", "tools"]),
        (["tools"], ["chat", "tools"]),
        ([], ["chat"]),
    ],
)
def test_capabilities_follow_supported_parameters(
    runtime, openrouter, parameters, capabilities
) -> None:
    openrouter.models = [{"id": "o/m", "canonical_slug": "o/m", "supported_parameters": parameters}]

    assert _list(runtime)["models"][0]["capabilities"] == capabilities


def test_anthropic_routes_advertise_prompt_caching(runtime, openrouter) -> None:
    openrouter.models = [{"id": "anthropic/claude-x", "canonical_slug": "anthropic/claude-x"}]

    assert "prompt_caching" in _list(runtime)["models"][0]["capabilities"]


@pytest.mark.parametrize(
    ("model", "label"),
    [
        (
            {"id": "z-ai/glm-4.5", "name": "Z.AI: GLM 4.5", "canonical_slug": "z-ai/glm-4.5"},
            "Z.AI / GLM",
        ),
        (
            {
                "id": "meta-llama/llama-3.3-70b-instruct",
                "name": "Meta: Llama 3.3 70B Instruct",
                "canonical_slug": "meta-llama/llama-3.3-70b-instruct",
            },
            "Meta Llama",
        ),
        ({"id": "acme-ai/x", "canonical_slug": "acme-ai/x"}, "Acme AI"),
    ],
)
def test_owner_labels_come_from_model_metadata(runtime, openrouter, model, label) -> None:
    openrouter.models = [model]

    assert _list(runtime)["models"][0]["upstream_owner_label"] == label


def test_an_alias_id_keeps_its_canonical_slug_as_revision(runtime, openrouter) -> None:
    openrouter.models = [{"id": "vendor/latest", "canonical_slug": "vendor/model-2026-07"}]

    model = _list(runtime)["models"][0]

    assert model["model_id"] == "vendor/latest"
    assert model["upstream_revision"] == "vendor/model-2026-07"


def test_entries_without_an_id_are_skipped(runtime, openrouter) -> None:
    openrouter.models = [{"name": "nameless"}, "not an object", {"id": "o/m"}]

    assert [model["model_id"] for model in _list(runtime)["models"]] == ["o/m"]


def test_pages_are_stateless_and_bound_to_one_catalog(runtime, openrouter) -> None:
    openrouter.models = _catalog(205)

    first = _list(runtime)
    second = _list(runtime, cursor=first["next_cursor"])

    assert len(first["models"]) == 200
    assert first["next_cursor"].endswith(":200")
    assert [m["model_id"] for m in second["models"]] == [
        f"owner/model-{i}" for i in range(200, 205)
    ]
    assert second["next_cursor"] == ""
    # Every page re-reads the catalog; the key is checked once, on the first page.
    assert [c["url"].rsplit("/", 1)[1] for c in openrouter.calls] == ["key", "models", "models"]


def test_a_catalog_that_changes_between_pages_is_refused(runtime, openrouter) -> None:
    openrouter.models = _catalog(205)
    first = _list(runtime)
    openrouter.models = _catalog(205)[1:]

    with pytest.raises(RuntimeError, match="invalid_response"):
        _list(runtime, cursor=first["next_cursor"])


def test_the_page_size_is_capped(runtime, openrouter) -> None:
    openrouter.models = _catalog(runtime.MAX_DISCOVERY_PAGE_SIZE + 1)

    assert len(_list(runtime, limit=10_000)["models"]) == runtime.MAX_DISCOVERY_PAGE_SIZE


@pytest.mark.parametrize("cursor", ["no-separator", ":5", "abc:", "abc:-1", "abc:x"])
def test_a_malformed_cursor_is_invalid(runtime, openrouter, cursor) -> None:
    with pytest.raises(RuntimeError, match="invalid_response"):
        _list(runtime, cursor=cursor)


def test_a_missing_api_key_fails_before_any_request(runtime, openrouter) -> None:
    with pytest.raises(RuntimeError, match="authentication_error"):
        _list(runtime, api_key="")
    assert openrouter.calls == []


def test_a_catalog_without_a_model_list_is_invalid(monkeypatch, runtime) -> None:
    monkeypatch.setattr(runtime, "_request_json", lambda *_a, **_k: {"data": "nope"})

    with pytest.raises(RuntimeError, match="invalid_response"):
        _list(runtime)


def test_a_key_openrouter_rejects_fails_discovery_instead_of_every_model(
    runtime, openrouter
) -> None:
    # GET /models is public: without this check the catalog loads with a wrong
    # key and every model then fails verification with "authentication failed".
    openrouter.models = _catalog(3)
    openrouter.key = RuntimeError("authentication_error")
    with pytest.raises(RuntimeError, match="^authentication_error$"):
        _list(runtime, api_key="sk-or-v1-partly-copied")
    assert [c["url"].rsplit("/", 1)[1] for c in openrouter.calls] == ["key"]


@pytest.mark.parametrize("no_verdict", ["upstream_error", "timeout", "rate_limited"])
def test_a_key_check_without_a_verdict_does_not_block_discovery(
    runtime, openrouter, no_verdict: str
) -> None:
    openrouter.models = _catalog(1)
    openrouter.key = RuntimeError(no_verdict)
    assert len(_list(runtime)["models"]) == 1
