# Changelog

## 1.2.1

- The logo (`icon.png`) now has a transparent background; in 1.2.0 the Flow Steward wave sat on a black box on dark themes.

## 1.2.0

- Logo: `icon.png`, the Flow Steward wave mark.
- Speaks the final extension host contract `extension_host_v2`
  (`runtime.contract_version`) and is built on `flowsteward-extension-sdk` 0.4.0.
  A request of another contract, with a missing execution-scope key
  (`group_id`, `for_each_id`, `step_id`, `step_execution_id`), with both a Group
  and a For each as its owner, or with a step but no step execution, is refused
  with `invalid_host_envelope` before OpenRouter is called. An older Flow Steward
  does not run this release.
- Works the same for a step at the root, in a Group or in a For each, and for
  calls outside a workflow (provider settings, model discovery).

## 1.1.1

- Checks the API key before listing models: **Discover models** stops with
  "authentication failed" when OpenRouter rejects the key, instead of loading
  the public catalog and failing every model at verification. Uses `GET /key`,
  which runs no model.
- Explains the API key field in Flow Steward (`provider_secret_help`). Needs
  extension host contract 1.3.0 (`platform_min: 1.3.0`).

## 1.1.0

- Runs as an ordinary Flow Steward extension subprocess and asks only for an
  OpenRouter API key.
- Needs extension host contract 1.1.0.
