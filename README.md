# OpenRouter LLM Provider for Flow Steward

Use OpenRouter-hosted models (OpenAI, Google, Anthropic, Meta, DeepSeek and
hundreds more) in Flow Steward agents with one OpenRouter API key. Models you
add appear in Flow Steward's model catalog, grouped by the company that makes
them: you can pick them for an agent or make one a project's default model.

- Extension id: `flowsteward.openrouter`
- Kind: `tool_provider` with the `llm_provider` contract (`llm_provider_extension_v1`)
- Runs as an ordinary Flow Steward extension subprocess. There is no separate
  service to deploy, no runtime URL and no runtime token.

## Setup

1. Install the extension from the Marketplace (or **Install extension from
   file** with the release ZIP).
2. Open the extension page. The **LLM provider** panel asks for one field:

   | Field | Required | Meaning |
   | --- | --- | --- |
   | **API key** (`api_key`) | yes | Your key from [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) |

   The key is stored encrypted, scoped to your Flow Steward account, and only
   reaches this extension when it calls OpenRouter. It is sent as
   `Authorization: Bearer <key>` to the fixed upstream
   `https://openrouter.ai/api/v1`. There is no base URL to configure, so the key
   cannot be pointed anywhere else.
3. Click **Discover models** to load the catalog.
4. Tick the models you want. Flow Steward verifies each one (see below) and
   enables it once it passes.
5. Choose a verified model on an agent, or set it as a project's default model.

## How models are listed

`llm.list_models` reads `GET /models`. For each model:

- **Capabilities** come from `supported_parameters`: `chat` always, `tools` when
  it lists `tools`, and `structured_output` when it lists `structured_outputs`
  or `response_format`. Flow Steward only enables models with all three.
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

The host runs `python3 main.py` with one JSON request on stdin.
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
