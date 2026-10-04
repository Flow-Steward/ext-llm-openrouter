# Changelog

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
