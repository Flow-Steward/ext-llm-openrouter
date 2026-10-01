# OpenRouter LLM Provider — reference extension

Status: active reference bundle. It includes a runnable stdlib remote runtime,
canonical wire handlers, fixtures, and contract tests. Hosting the runtime is
an operator responsibility and is intentionally outside this bundle.

## Install boundary

The manifest deliberately contains no default runtime URL. Installation requires
an explicit public HTTPS `remote_base_url`; this is the URL of this extension
runtime, not `openrouter.ai`. The OpenRouter upstream base URL is fixed by the
runtime and the account-scoped `api_key` is passed through the existing
encrypted `secret_refs` flow. Never place a private, LAN, localhost, or HTTP
endpoint in either configuration.

The bundled stdlib process is an origin server and binds to `127.0.0.1:8090`
by default. Production hosting must place it behind a reverse proxy or load
balancer that terminates TLS and exposes the configured public HTTPS
`remote_base_url`. Set `FS_EXTENSION_BIND_HOST` only when the deployment network
requires a different origin bind address; never publish the plain HTTP origin
directly to the internet.

For every credential-bearing OpenRouter request, the runtime resolves the fixed
`openrouter.ai` hostname again, requires every DNS answer to be a public address,
and pins that request's TCP connection to one validated address while retaining
`openrouter.ai` for TLS certificate verification, SNI, and the HTTP `Host` header.
Empty, private, reserved, link-local, loopback, multicast, unspecified, IPv6
site-local, or mixed DNS results are rejected. Redirect responses are never
followed, so the provider API key is sent only to the validated fixed upstream
request target.

A remote HTTPS LLM provider extension that exposes OpenRouter-hosted models to
the Flow Steward model catalog through the `llm_provider` extension contract
(`runtime.extension_contract_v2.llm_provider`, schema `llm_provider_extension_v1`).

## Contract

- **supplier_key / supplier_label**: `openrouter` / `OpenRouter`
- **runtime_mode**: `remote_https` (the only mode supported in v1)
- **credentials**: configured through the existing platform provider-secret flow
  (`POST /api/llm-provider-extensions/{install_id}/credentials`, account-scoped,
  stored in `secret_refs`). `required_provider_secrets`:
  `["api_key", "runtime_bearer_token"]`. The reference runtime must be started
  with the matching `FS_EXTENSION_TOKEN`; `runtime_bearer_token` is sent only as
  the host-to-runtime Authorization header and is never forwarded upstream.
- **operations**:
  - `llm.list_models` (query) — discover models, paginated via `cursor`/`limit`.
  - `llm.chat` (action) — chat completion.
  - `llm.chat_structured` (action) — structured output for
    `with_structured_output(schema, ...)`.
- **action idempotency**: `llm.chat` and `llm.chat_structured` require the host
  `idempotency_key`. The single-process reference runtime coalesces concurrent
  calls and retains the completed result or safe error for five minutes in a
  bounded in-memory cache, so a transport retry cannot repeat an accepted
  upstream generation while that runtime process remains alive.
- **model_grouping**: `upstream_owner` — models are grouped into provider groups
  by their `upstream_owner_key` / `upstream_owner_label`.

## Grouping & naming

Aggregator models are grouped by upstream owner and labelled
`"<owner> (<supplier>)"`, composed from descriptor metadata (never hardcoded):

| `model_id` (preserved exactly) | Provider group label    |
| ------------------------------ | ----------------------- |
| `deepseek/deepseek-chat`       | `DeepSeek (OpenRouter)` |
| `z-ai/glm-4.5`                 | `Z.AI / GLM (OpenRouter)` |
| `meta-llama/llama-3.3-70b-instruct` | `Meta Llama (OpenRouter)` |

`model_id` is sent to the remote runtime verbatim — e.g. `deepseek/deepseek-chat`.
Models without `structured_output` support (e.g. the Llama entry above) are
stored on discovery but cannot be enabled for global routing in v1.

## Fixtures

`fixtures/` contains canonical `llm.list_models`, `llm.chat`, and
`llm.chat_structured` responses. They are exercised by
`tests/unit/test_llm_provider_extensions.py` to prove exact model-ID
preservation and grouping, and the remote-runtime round-trip.
