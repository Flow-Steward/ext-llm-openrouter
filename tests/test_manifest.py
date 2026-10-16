"""Contract tests: the manifest, the entrypoint and the code agree.

Flow Steward reads ``extension.yaml``; these tests fail when the manifest
promises something the code does not do (or the other way round).
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = yaml.safe_load((ROOT / "extension.yaml").read_text())
CONTRACT = MANIFEST["runtime"]["extension_contract_v2"]["llm_provider"]


def test_it_runs_as_an_ordinary_extension_subprocess() -> None:
    entrypoint = MANIFEST["entrypoint"]

    assert entrypoint["mode"] == "subprocess"
    assert entrypoint["command"] == ["python3", "main.py"]
    assert (ROOT / "main.py").is_file()
    assert CONTRACT["runtime_mode"] == "subprocess"
    # No separately hosted runtime: nothing to point at, nothing to protect.
    assert "remote" not in MANIFEST["runtime"]
    assert "remote" not in entrypoint


def test_it_asks_only_for_an_api_key() -> None:
    assert CONTRACT["required_provider_secrets"] == ["api_key"]
    assert CONTRACT["optional_provider_secrets"] == []
    # OpenRouter's upstream is fixed: no URL to configure.
    assert CONTRACT.get("required_connection_config", []) == []
    assert CONTRACT.get("optional_connection_config", []) == []


def test_the_code_reads_exactly_the_declared_settings(runtime) -> None:
    source = (ROOT / "runtime.py").read_text()
    read_settings = set(re.findall(r'\.get\("(api_key|\w+_url|\w+_token)"\)', source))

    assert read_settings == {"api_key"}
    assert runtime.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


def test_every_bound_operation_has_a_handler(main) -> None:
    operations = CONTRACT["operations"]

    assert set(main.QUERIES) == {operations["list_models"]["query_id"]}
    assert set(main.ACTIONS) == {
        operations["chat"]["action_id"],
        operations["structured_chat"]["action_id"],
    }


def test_models_are_grouped_by_their_upstream_owner() -> None:
    assert CONTRACT["model_grouping"] == {
        "strategy": "upstream_owner",
        "owner_key_field": "upstream_owner_key",
        "owner_label_field": "upstream_owner_label",
    }


def test_enablement_requires_the_capabilities_the_host_verifies() -> None:
    assert set(CONTRACT["required_capabilities_for_enablement"]) == {
        "chat",
        "tools",
        "structured_output",
    }
    assert CONTRACT["schema_version"] == "llm_provider_extension_v1"


def test_the_chat_timeout_outlasts_the_upstream_request() -> None:
    # The subprocess must not be killed while a 60 s upstream call is open.
    assert MANIFEST["entrypoint"]["timeout_seconds"] > 60


@pytest.mark.parametrize("path", ["health.py", "ui/ui_manifest.yaml", "README.md"])
def test_referenced_files_exist(path) -> None:
    assert (ROOT / path).is_file()


def test_ui_pages_listed_in_the_ui_manifest_exist() -> None:
    ui = yaml.safe_load((ROOT / "ui" / "ui_manifest.yaml").read_text())

    for page in ui["pages"]:
        assert (ROOT / "ui" / page["ref"]).is_file()


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", MANIFEST["version"])


def test_it_requires_a_host_that_shows_setting_help() -> None:
    # Extension host contract 1.1.0 runs an llm_provider as a subprocess; 1.3.0
    # shows provider_secret_help under the field. An older host marks the
    # bundle incompatible ("below required minimum") instead of misreading it.
    assert MANIFEST["runtime"]["compatibility"]["platform_min"] == "1.3.0"


def test_key_help_names_only_declared_secrets_and_fits_the_host_limit() -> None:
    help_text = CONTRACT["provider_secret_help"]
    declared = set(CONTRACT["required_provider_secrets"]) | set(
        CONTRACT["optional_provider_secrets"]
    )
    assert set(help_text) == {"api_key"} <= declared
    assert all(isinstance(t, str) and 0 < len(t) <= 300 for t in help_text.values())


def test_logo_is_a_square_png_within_the_host_limits() -> None:
    # Core requires icon.png at the bundle root: PNG, square, 32-512 px, <= 100 KB.
    data = (ROOT / "icon.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    assert width == height
    assert 32 <= width <= 512
    assert len(data) <= 100 * 1024
