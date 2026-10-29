"""reasoning-dataset-lab supports current Python: 3.13 and newer, 3.15 the main target."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NL = chr(10)


def test_requires_python_python_version_file_and_lock_agree_on_current_python():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.13"
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.15"
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert lock["requires-python"] == ">=3.13"


def test_role_set_is_pinned_to_a_pypi_release_that_allows_current_python_and_the_lock_agrees():
    from reasoning_dataset_lab.generators import role_set as adapter

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "git+https://" not in pyproject  # PyPI refuses git-URL dependencies
    assert re.findall(r'"role-set==([0-9.]+)"', pyproject) == [adapter.ROLE_SET_RELEASE]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    (locked,) = [p for p in lock["package"] if p["name"] == "role-set"]
    assert locked["version"] == adapter.ROLE_SET_RELEASE
    assert "git" not in locked["source"]
    assert locked["source"].get("registry") == "https://pypi.org/simple"


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
        assert "codecov" not in body.lower() and "--no-cov" in body  # coverage and the upload run on the 3.15 leg only
    assert "codecov/codecov-action@" in test_job and "--cov-fail-under=90" in test_job


def test_every_action_in_the_workflow_is_pinned_by_commit_sha_with_a_version_comment():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    uses = re.findall(r"uses: (\S+)(.*)", ci)
    assert uses
    for ref, rest in uses:
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", ref), f"{ref} is not pinned by commit sha"
        assert re.search(r"# v\d", rest), f"{ref} has no version comment"


def test_the_installed_role_set_is_the_pinned_release_and_the_citations_name_the_source_commit():
    import importlib.metadata

    import role_set
    from reasoning_dataset_lab.generators import role_set as adapter

    assert importlib.metadata.version("role-set") == adapter.ROLE_SET_RELEASE, "run uv sync --frozen"
    assert role_set.__version__ == adapter.ROLE_SET_RELEASE
    assert adapter.ROLE_SET_FULL_SHA.startswith(adapter.ROLE_SET_COMMIT)
    assert len(adapter.ROLE_SET_RELEASE_COMMIT) == 40
    # the citations name the source commit the receipts name (role-set 0.3.0's src is identical to it, see the adapter)
    for path in (ROOT / "src" / "reasoning_dataset_lab").rglob("*.py"):
        for cited in re.findall(r"@([0-9a-f]{7})", path.read_text(encoding="utf-8")):
            assert cited in (adapter.ROLE_SET_COMMIT, adapter.ASPIRE_SI_GATES_COMMIT), f"{path.name} cites {cited}"
