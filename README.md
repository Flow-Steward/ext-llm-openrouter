# OpenRouter LLM Provider for Flow Steward

Use OpenRouter-hosted models (OpenAI, Google, Anthropic, Meta, DeepSeek and
hundreds more) in Flow Steward agents with one OpenRouter API key. Models you
add appear in Flow Steward's model catalog, grouped by the company that makes
them: you can pick them for an agent or make one a project's default model.

- Extension id: `flowsteward.openrouter`
- Kind: `tool_provider` with the `llm_provider` contract (`llm_provider_extension_v1`)
- Runs as an ordinary Flow Steward extension subprocess. There is no separate
  service to deploy, no runtime URL and no runtime token.
- Needs Flow Steward's extension host contract 1.3.0 or newer
  (`runtime.compatibility.platform_min: 1.3.0`); an older host marks the bundle
  incompatible. 1.3.0 shows the help text under the API key field
  (`provider_secret_help`).

## Requirements

This release speaks the host contract `extension_host_v2` with its final
execution scope (`group_id`, `for_each_id`, `step_id`, `step_execution_id`)
and is built on `flowsteward-extension-sdk` 0.4.0. A request with another
contract, a missing scope key, or both a Group and a For each as its owner is
refused (`invalid_host_envelope`) before OpenRouter is called. Install it on a
Flow Steward Core that speaks this contract; an older Core does not run it.

## Setup

1. Install the extension from the Marketplace (or **Install extension from
   file** with the release ZIP).
2. Open the extension page. The **LLM provider** panel asks for one field:

   | Field | Required | Meaning |
   | --- | --- | --- |
   | **API key** (`api_key`) | yes | Your key from [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys). It starts with `sk-or-v1-`; copy it whole with the copy button. |

   The key is stored encrypted, scoped to your Flow Steward account, and only
   reaches this extension when it calls OpenRouter. It is sent as
   `Authorization: Bearer <key>` to the fixed upstream
   `https://openrouter.ai/api/v1`. There is no base URL to configure, so the key
   cannot be pointed anywhere else.
3. Click **Discover models** to load the catalog. Discovery first checks the
   key with `GET /key`, which runs no model: a rejected key (wrong, revoked,
   or partly copied) stops discovery with "The provider rejected the API key"
   instead of loading the public catalog and failing every model later.
   Use the search box to find a model by name, for example `gpt-4o-mini` or
   `claude`. Models that can't meet Flow Steward's requirements are hidden
   until you choose **Show all models**.
4. Tick the models you want. Flow Steward verifies each one (see below) and
   enables it once it passes.
5. Choose a verified model on an agent, or set it as a project's default model.

## How models are listed

`llm.list_models` reads `GET /models`. For each model:

- **Capabilities** come from `supported_parameters`: `chat` always, `tools` when
  it lists `tools`, and `structured_output` when it lists `structured_outputs`
  (strict JSON Schema; `response_format` alone is often only JSON mode). Flow
  Steward only enables models with all three.
  Anthropic models also report `prompt_caching`.
- **Grouping**: `upstream_owner_key` is the first part of `canonical_slug`
  (`openai` in `openai/gpt-4o-mini`), and the label comes from the model's
  display name ("OpenAI: GPT-4o-mini" gives "OpenAI"). The catalog shows
  "OpenAI (OpenRouter)", "Google (OpenRouter)", and so on.
- **Revision**: `canonical_slug`. When OpenRouter repoints an alias id, Flow
  Steward marks that model's verification stale.

Pages hold 200 models. Every page is a separate subprocess, so the cursor holds
a fingerprint of the catalog plus an offset. If the catalog changes between
pages, discovery fails with `invalid_response` and Flow Steward keeps the
previous catalog. Run **Discover models** again.

## Which models can work

Flow Steward routes work only to models that can chat, call tools and answer
in its structured format. The list of required capabilities is kept in one
place in Flow Steward (`REQUIRED_ROUTING_CAPABILITIES`) and checked live by
verification.

This extension reports a model's capabilities from OpenRouter's
`supported_parameters`: `tools` for tool calling and `structured_outputs` for
structured output. On 2 October 2026, 341 of OpenRouter's 465 models reported
both; the other 124 are hidden by default. Reporting both does not mean a
model passes: verification also checks that it really calls the tool and
fills in the required values.

Models the provider says lack a required capability are marked, for example
"Not supported: no tool calling", and hidden by default; **Show all models**
lists them too. What the provider reports is only a first filter.
Verification is the final authority: a hidden model can still be tried, and a
listed model can still fail.

## How models are verified

Flow Steward, not this extension, verifies a model before it can be enabled. It
runs a fixed probe: a plain chat, a forced tool call, a tool-result replay, and
two structured-output answers checked against a JSON Schema with exact
(`const`) values. A model that fails any step stays disabled and shows the
reason. A verification lasts 30 days.

## OpenRouter behaviour this extension accounts for

- **Routing to endpoints that support the request.** Requests with tools or a
  structured-output schema set `provider.require_parameters: true`, so
  OpenRouter only routes to endpoints that accept those parameters instead of
  silently ignoring them.
- **`parallel_tool_calls` is not forwarded.** Combined with
  `require_parameters`, it makes OpenRouter answer `404` ("No endpoints found
  that can handle the requested parameters"), even for models that call tools
  well, such as `openai/gpt-4o-mini`. When the host asks for one call at a time,
  the response is checked instead: two or more calls fail with
  `capability_mismatch`.
- **Structured output** uses `response_format: {type: json_schema, json_schema:
  {name, strict, schema}}`, with `strict: true` unless the host turns it off.
- **Strict mode needs every property in `required`.** OpenAI and Azure refuse
  (400) a strict schema whose objects leave some properties out of `required`
  or allow extra properties, which is how Pydantic describes fields with
  defaults. In strict mode every object lists all of its properties as
  required and sets `additionalProperties: false`. The model then states
  defaults explicitly, and its answer is still valid against the host's
  original schema.
- **Some upstreams drop `const`.** Gemini behind OpenRouter keeps `enum` but
  ignores `const`, so `const: x` is rewritten as the equivalent `enum: [x]`. A
  `const` beside an explicit `enum` is left alone so the schema is never
  loosened. Flow Steward still validates the answer against the original
  schema.
- **Reasoning state** (`reasoning_details`, `reasoning`, `reasoning_content`) is
  returned to the host and replayed verbatim on the next turn, as OpenRouter
  requires for tool use with reasoning models.
- **Cost**: OpenRouter's reported `usage.cost` is passed on as `cost_usd` when
  it is a finite, non-negative number.

## Network safety

The upstream is fixed. Every request still re-resolves `openrouter.ai`, refuses
the call if any DNS answer is non-public (loopback, private, link-local, CGNAT,
multicast, reserved), and pins the connection to the checked address, while TLS
verifies the hostname. Redirects are never followed, so the key cannot be
forwarded elsewhere. Upstream bodies are capped at 5 MiB, and an error status is
classified before its body is read.

## Errors

Every failure leaves the extension as one of these categories, never as
upstream text: `authentication_error` (401/403, or no key), `billing_quota_exceeded`
(402), `model_unavailable` (404, including "no endpoint accepts these
parameters"), `rate_limited` (429), `timeout`, `capability_mismatch`,
`invalid_response`, `upstream_error`. Model discovery can be retried safely. A
failed chat is not retried automatically, because it may already have run and
been billed upstream.

## Files

| File | Purpose |
| --- | --- |
| `extension.yaml` | Manifest: subprocess entrypoint and the `llm_provider` contract |
| `main.py` | Subprocess entrypoint: reads the host request on stdin and dispatches `llm.list_models`, `llm.chat`, `llm.chat_structured` |
| `runtime.py` | OpenRouter HTTP client and the three operations |
| `health.py` | Health command run by the host |
| `fixtures/` | Canonical results of each operation; the tests compare against them |
| `ui/` | Extension page shown in Flow Steward |

### Host contract in one paragraph

The host runs `python3 main.py` with one JSON request on stdin. Its root
`contract_version` is `extension_host_v2` and `runtime_context` carries the
four execution-scope keys; anything else is refused before any work.
`mode` is `query` or `action`, the operation is in `query.query_id` /
`action.action_id`, and its arguments are in `query.params` / `action.input`. The
API key arrives as a runtime resource,
`runtime_context.resources.provider_secrets.api_key`. The extension prints
`{"ok": true, "result": …}` and exits 0. On failure it prints
`{"ok": false, "error_code", "error", "errors": [{"code", "message"}]}` and exits
2, which the host treats as a provider error rather than a crash.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install pytest==8.3.5 PyYAML==6.0.3 dev-wheels/flowsteward_extension_sdk-*.whl
python -m pytest -q                                    # every handler, with mocked HTTP
python .github/scripts/package_extension.py            # dist/flowsteward.openrouter-<version>.zip
python .github/scripts/catalog_check.py dist/*.zip     # what the catalog would say
```

These are the same steps CI runs. No test touches the network. Each test
replaces `runtime._request_json`, or the socket layer for the transport tests.
To validate the bundle with Flow Steward itself:

```bash
flow-steward extensions validate flowsteward.openrouter --root <directory containing this bundle>
```

Releases: bump `version` in `extension.yaml`, then push a matching tag
(`v1.1.0`). The release workflow publishes the archive CI built.
