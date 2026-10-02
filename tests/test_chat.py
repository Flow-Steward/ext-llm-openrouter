"""``llm.chat`` and ``llm.chat_structured`` against OpenRouter's chat completions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import completion, connection

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look a value up.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
    },
}
SCHEMA = {
    "title": "answer",
    "type": "object",
    "properties": {
        "status": {"type": "string", "const": "ok"},
        "const": {"type": "integer"},  # a property may itself be called "const"
    },
    "required": ["status", "const"],
    "additionalProperties": False,
}


def _chat(runtime, structured: bool = False, **payload):
    request = {
        "connection": connection(),
        "input": {"model_id": "openai/gpt-4o-mini", "messages": [], **payload},
    }
    return runtime.chat(request, structured=structured)


def test_a_plain_chat_returns_the_canonical_result(runtime, openrouter) -> None:
    openrouter.completions = [completion({"content": "basic-ok"}, cost=0.0000031)]

    result = _chat(
        runtime,
        messages=[{"role": "user", "content": "Return exactly basic-ok."}],
        temperature=0,
        max_tokens=64,
    )

    assert result == json.loads((FIXTURES / "llm.chat.json").read_text())["result"]
    (call,) = openrouter.calls
    assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer key-1"
    assert call["body"] == {
        "model": "openai/gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Return exactly basic-ok."}],
    }


def test_tool_calls_and_results_translate_both_ways(runtime, openrouter) -> None:
    openrouter.completions = [
        completion(
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"key":"probe"}'},
                    }
                ],
            },
            finish_reason="tool_calls",
        )
    ]

    result = _chat(
        runtime,
        tools=[TOOL],
        tool_choice="required",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_0", "name": "lookup", "arguments": {"key": "x"}}],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": {"value": "probe-ok"}},
        ],
    )

    sent = openrouter.calls[0]["body"]
    assert sent["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"key":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": '{"value":"probe-ok"}'},
    ]
    assert sent["tools"] == [TOOL]
    assert sent["tool_choice"] == "required"
    # Tools are only routed to endpoints that accept them.
    assert sent["provider"] == {"require_parameters": True}
    assert result["message"]["tool_calls"] == [
        {"id": "call_1", "name": "lookup", "arguments": {"key": "probe"}}
    ]
    assert result["finish_reason"] == "tool_calls"


def test_parallel_tool_calls_is_enforced_on_the_response_not_sent(runtime, openrouter) -> None:
    # Sending it together with require_parameters makes OpenRouter answer 404
    # "No endpoints found that can handle the requested parameters".
    call = {"id": "c", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    openrouter.completions = [
        completion({"tool_calls": [call]}),
        completion({"tool_calls": [call, {**call, "id": "d"}]}),
    ]

    _chat(runtime, tools=[TOOL], parallel_tool_calls=False)
    assert "parallel_tool_calls" not in openrouter.calls[0]["body"]
    with pytest.raises(RuntimeError, match="capability_mismatch"):
        _chat(runtime, tools=[TOOL], parallel_tool_calls=False)


def test_reasoning_is_replayed_verbatim_on_the_next_turn(runtime, openrouter) -> None:
    openrouter.completions = [completion({"content": "", "reasoning_details": [{"id": "r1"}]})]

    result = _chat(
        runtime,
        messages=[
            {
                "role": "assistant",
                "content": "",
                "continuation": {"reasoning_details": [{"id": "r0"}]},
            }
        ],
    )

    assert openrouter.calls[0]["body"]["messages"][0]["reasoning_details"] == [{"id": "r0"}]
    assert result["message"]["continuation"] == {"reasoning_details": [{"id": "r1"}]}


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"cost": 0}, 0.0),
        ({"cost": 0.0123}, 0.0123),
        ({"cost": -1}, None),
        ({"cost": float("inf")}, None),
        ({"cost": "free"}, None),
        ({"cost": True}, None),
    ],
)
def test_only_a_finite_nonnegative_cost_is_reported(runtime, openrouter, usage, expected) -> None:
    openrouter.completions = [completion({"content": "ok"}, **usage)]

    assert _chat(runtime)["usage"].get("cost_usd") == expected


def test_cache_accounting_is_carried_over(runtime, openrouter) -> None:
    openrouter.completions = [
        completion(
            {"content": "ok"},
            prompt_tokens_details={"cached_tokens": 7, "cache_write_tokens": 3},
        )
    ]

    usage = _chat(runtime)["usage"]

    assert (usage["cache_read_input_tokens"], usage["cache_creation_input_tokens"]) == (7, 3)


@pytest.mark.parametrize(
    "reply",
    [
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {"content": "a"}}, {"message": {"content": "b"}}]},
        {"choices": [{"message": {"content": "partial"}, "finish_reason": "error"}]},
        {"choices": [{"message": {"content": "x"}, "error": {"code": 502}}]},
        {},
    ],
)
def test_a_malformed_completion_is_invalid(runtime, openrouter, reply) -> None:
    openrouter.completions = [reply]

    with pytest.raises(RuntimeError, match="invalid_response"):
        _chat(runtime)


@pytest.mark.parametrize(
    "error",
    ["authentication_error", "billing_quota_exceeded", "model_unavailable", "rate_limited"],
)
def test_upstream_failures_propagate_their_category(runtime, openrouter, error) -> None:
    openrouter.completions = [RuntimeError(error)]

    with pytest.raises(RuntimeError, match=error):
        _chat(runtime)


# --------------------------------------------------------------------------- #
# Structured output                                                            #
# --------------------------------------------------------------------------- #


def test_structured_output_uses_a_strict_json_schema(runtime, openrouter) -> None:
    openrouter.completions = [completion({"content": '{"status":"ok","const":7}'})]

    result = _chat(runtime, structured=True, schema=SCHEMA, schema_name="answer")

    assert result == json.loads((FIXTURES / "llm.chat_structured.json").read_text())["result"]
    sent = openrouter.calls[0]["body"]
    assert sent["provider"] == {"require_parameters": True}
    response_format = sent["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "answer"
    assert response_format["json_schema"]["strict"] is True
    # ``const`` becomes ``enum`` (some upstreams drop const); names stay names.
    properties = response_format["json_schema"]["schema"]["properties"]
    assert properties["status"] == {"type": "string", "enum": ["ok"]}
    assert properties["const"] == {"type": "integer"}


def test_strict_mode_gets_a_schema_strict_upstreams_accept(runtime, openrouter) -> None:
    # A Pydantic model's schema leaves fields with defaults out of "required".
    # OpenAI and Azure refuse that in strict mode with HTTP 400, which surfaced
    # as upstream_error in the AI Builder's intent step.
    pydantic_style = {
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "note": {"type": "string", "default": ""},
                },
                "required": ["id"],
            }
        },
        "type": "object",
        "properties": {
            "goal": {"type": "string"},
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}, "default": []},
            "extras": {
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "additionalProperties": True,
            },
        },
        "required": ["goal"],
    }
    openrouter.completions = [completion({"content": "{}"})]

    _chat(runtime, structured=True, schema=pydantic_style)

    sent = openrouter.calls[0]["body"]["response_format"]["json_schema"]["schema"]
    assert sent["required"] == ["goal", "items", "extras"]
    assert sent["additionalProperties"] is False
    assert sent["$defs"]["Item"]["required"] == ["id", "note"]
    assert sent["$defs"]["Item"]["additionalProperties"] is False
    assert sent["properties"]["extras"]["additionalProperties"] is False
    # Defaults and other literals are carried over untouched.
    assert sent["properties"]["items"]["default"] == []


def test_strict_can_be_turned_off_by_the_host(runtime, openrouter) -> None:
    openrouter.completions = [completion({"content": "{}"})]

    _chat(runtime, structured=True, schema=SCHEMA | {"required": ["status"]}, strict=False)

    response_format = openrouter.calls[0]["body"]["response_format"]["json_schema"]
    assert response_format["strict"] is False
    assert response_format["schema"]["required"] == ["status"]  # sent as given


@pytest.mark.parametrize("schema", [None, {}, "object"])
def test_structured_output_needs_a_schema(runtime, openrouter, schema) -> None:
    with pytest.raises(RuntimeError, match="capability_mismatch"):
        _chat(runtime, structured=True, schema=schema)
    assert openrouter.calls == []


def test_portable_schema_never_loosens_or_rewrites_values(runtime) -> None:
    schema = {
        "$defs": {"Item": {"properties": {"id": {"const": 7}}}},
        "properties": {"mode": {"const": "a", "enum": ["a", "b"]}},
        "default": {"const": "kept verbatim"},
    }

    assert runtime._portable_schema(schema) == {
        "$defs": {"Item": {"properties": {"id": {"enum": [7]}}}},
        "properties": {"mode": {"const": "a", "enum": ["a", "b"]}},
        "default": {"const": "kept verbatim"},
    }
