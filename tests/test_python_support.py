"""reasoning-dataset-lab supports current Python: 3.13 and newer, 3.15 the main target."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NL = chr(10)
OLD_ROLE_SET = "2b268401f89437df3aa5226ac8579d90598f851b"  # role-set <=3.12 (requires-python >=3.12,<3.13)


def test_requires_python_python_version_file_and_lock_agree_on_current_python():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.13"
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.15"
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert lock["requires-python"] == ">=3.13"


def test_role_set_is_pinned_to_a_commit_that_allows_current_python_and_the_lock_agrees():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pins = re.findall(r"role-set @ git\+https://github.com/mcp-tool-shop-org/role-set@([0-9a-f]{40})", pyproject)
    assert len(pins) == 1 and pins[0] != OLD_ROLE_SET
    assert pins[0] in (ROOT / "uv.lock").read_text(encoding="utf-8")


def test_ci_runs_the_suite_on_3_15_with_the_check_name_kept_and_on_3_13_and_3_14():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert NL + "  test:" + NL in ci
    assert NL + "  test-py313:" + NL in ci
    assert NL + "  test-py314:" + NL in ci
    assert 'python-version: "3.12"' not in ci
    test_job = ci.split(NL + "  test:" + NL, 1)[1].split(NL + "  test-py313:", 1)[0]
    assert 'python-version: "3.15"' in test_job
    assert "- name: Atlas check" in test_job
    for leg, version in (("test-py313", "3.13"), ("test-py314", "3.14")):
        body = ci.split(NL + "  " + leg + ":" + NL, 1)[1].split(NL + "  test-py3", 1)[0]
        assert f'python-version: "{version}"' in body
        assert "- name: Atlas check" not in body
