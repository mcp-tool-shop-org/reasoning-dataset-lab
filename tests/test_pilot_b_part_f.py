"""Tests for Part F: Python range parity between reasoning-dataset-lab and role-set."""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version


def _load_requires_python_from_pyproject(path: Path) -> str:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    req = data.get("project", {}).get("requires-python")
    assert isinstance(req, str) and req.strip(), f"Missing project.requires-python in {path}"
    return req.strip()


def _load_role_set_requires_python() -> str:
    local_role_set_pyproject = Path("E:/AI/role-set/pyproject.toml")
    if local_role_set_pyproject.exists():
        return _load_requires_python_from_pyproject(local_role_set_pyproject)
    meta_req = importlib.metadata.metadata("role-set").get("Requires-Python")
    assert isinstance(meta_req, str) and meta_req.strip(), "Missing Requires-Python on role-set"
    return meta_req.strip()


def check_lab_requires_python_subset_of_role_set(
    lab_spec_str: str,
    role_set_spec_str: str,
) -> list[str]:
    """Return any candidate Python versions allowed by lab_spec_str that role_set_spec_str refuses."""
    lab_spec = SpecifierSet(lab_spec_str)
    role_set_spec = SpecifierSet(role_set_spec_str)
    probe_versions = [
        f"3.{minor}.{patch}"
        for minor in range(8, 20)
        for patch in (0, 1, 5, 13)
    ]
    disallowed_by_role_set: list[str] = []
    for v_str in probe_versions:
        ver = Version(v_str)
        if ver in lab_spec and ver not in role_set_spec:
            disallowed_by_role_set.append(v_str)
    return disallowed_by_role_set


def test_lab_requires_python_matches_role_set_and_allows_no_refused_python() -> None:
    lab_pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    lab_req = _load_requires_python_from_pyproject(lab_pyproject)
    role_set_req = _load_role_set_requires_python()

    assert lab_req == ">=3.12,<3.13", (
        f"Expected lab requires-python to be '>=3.12,<3.13', got {lab_req!r} "
        f"(role-set requires-python is {role_set_req!r})"
    )
    leaked = check_lab_requires_python_subset_of_role_set(lab_req, role_set_req)
    assert leaked == [], (
        f"Lab requires-python ({lab_req!r}) allows Python version(s) refused by "
        f"role-set ({role_set_req!r}): {leaked[:6]}"
    )

    # Also verify that if a lab range allowed >=3.12 without <3.13, the check catches 3.13+
    caught = check_lab_requires_python_subset_of_role_set(">=3.12", role_set_req)
    assert "3.13.0" in caught
