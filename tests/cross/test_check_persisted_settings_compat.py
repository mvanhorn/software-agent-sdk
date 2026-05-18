"""Tests for the persisted settings compatibility check script."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")


def _load_script_module(name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / ".github" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_prod = _load_script_module("check_persisted_settings_compat")
PersistedSettingsCompatError = _prod.PersistedSettingsCompatError
FixtureCase = _prod.FixtureCase
SURFACES = _prod.SURFACES
collect_fixture_cases = _prod.collect_fixture_cases
get_pypi_baseline_version = _prod.get_pypi_baseline_version
validate_fixture_cases = _prod.validate_fixture_cases


def test_collect_fixture_cases_and_validate_current_repo_fixtures() -> None:
    cases = collect_fixture_cases()

    validate_fixture_cases(cases)

    versions_by_surface: dict[str, set[int]] = {}
    for case in cases:
        versions_by_surface.setdefault(case.surface_key, set()).add(case.version)

    assert versions_by_surface == {
        "agent_settings": {1, 2, 3},
        "conversation_settings": {1},
        "persisted_settings": {1},
    }


def test_validate_fixture_cases_requires_every_schema_version() -> None:
    cases = [case for case in collect_fixture_cases() if case.version != 2]

    with pytest.raises(
        PersistedSettingsCompatError,
        match="Missing persisted settings fixtures for AgentSettings: v2",
    ):
        validate_fixture_cases(cases)


def test_collect_fixture_cases_rejects_mismatched_directory_version(
    tmp_path: Path,
) -> None:
    root = tmp_path / "persisted_settings_baselines"
    version_dir = root / "v2"
    version_dir.mkdir(parents=True)
    (version_dir / "conversation_settings.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "max_iterations": 123,
                "confirmation_mode": False,
                "security_analyzer": "llm",
            }
        )
    )

    with pytest.raises(
        PersistedSettingsCompatError,
        match="has schema_version 1, but is stored under v2",
    ):
        collect_fixture_cases(root)


def test_validate_fixture_cases_surfaces_loader_guidance_on_failure() -> None:
    bad_case = FixtureCase(
        path=Path(
            "tests/sdk/persisted_settings_baselines/v1/conversation_settings.json"
        ),
        surface_key="conversation_settings",
        version=1,
        payload={
            "schema_version": 1,
            "max_iterations": 0,
            "confirmation_mode": True,
            "security_analyzer": "llm",
        },
    )

    with pytest.raises(
        PersistedSettingsCompatError,
        match="_CONVERSATION_SETTINGS_MIGRATIONS",
    ):
        validate_fixture_cases(
            [bad_case],
            surfaces={"conversation_settings": SURFACES["conversation_settings"]},
        )


def test_get_pypi_baseline_version_prefers_current_or_previous(monkeypatch) -> None:
    payload = {"releases": {"1.0.0": [], "1.1.0": []}}

    class _DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(
        _prod.urllib.request, "urlopen", lambda *_args, **_kwargs: _DummyResponse()
    )

    assert get_pypi_baseline_version("openhands-sdk", "1.1.0") == "1.1.0"
    assert get_pypi_baseline_version("openhands-sdk", "1.2.0") == "1.1.0"
