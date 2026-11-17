"""The site's npm audit gate (scripts/check-audit-allowlist.mjs, ported from brand) fails closed.

Synthetic `npm audit --json` (v2) reports are judged by the real script through `node`. Allowlists in these tests are
copies of the committed one with the expiry moved, so the tests do not rot when the real exception expires.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check-audit-allowlist.mjs"
ALLOWLIST = ROOT / ".github" / "audit-allowlist.json"
GHSA = "GHSA-fc5q-39xc-vjpf"
SOURCE = 1247710
NODE = shutil.which("node")

if NODE is None:
    if os.environ.get("CI"):
        raise RuntimeError("node is required for the audit-gate tests in CI")
    pytest.skip("node is not installed", allow_module_level=True)


def _advisory(source=SOURCE, name="picomatch", ghsa=GHSA, rng="<=4.0.7", severity="high"):
    return {
        "source": source,
        "name": name,
        "dependency": name,
        "title": "synthetic advisory",
        "url": f"https://github.com/advisories/{ghsa}",
        "severity": severity,
        "cwe": ["CWE-1333"],
        "range": rng,
    }


def _vuln(name, via, severity="high", fix=False, rng="<=4.0.7"):
    return {
        "name": name,
        "severity": severity,
        "isDirect": False,
        "via": via,
        "effects": [],
        "range": rng,
        "nodes": [f"node_modules/{name}"],
        "fixAvailable": fix,
    }


def _report(vulns):
    return {"auditReportVersion": 2, "vulnerabilities": vulns, "metadata": {}}


def _allowed_report(**picomatch):
    return _report(
        {
            "picomatch": _vuln("picomatch", [_advisory()], **picomatch),
            "tinyglobby": _vuln("tinyglobby", ["picomatch"], rng="*"),
        }
    )


def _allowlist(expires="2999-12-31", **changes):
    doc = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    entry = doc["exceptions"][0]
    entry["expiresOn"] = expires
    for key, value in changes.items():
        if value is None:
            entry.pop(key)
        else:
            entry[key] = value
    return doc


LOCK = {"packages": {"node_modules/@astrojs/starlight": {"version": "0.42.7"}, "node_modules/picomatch": {"version": "4.0.7"}}}
DOWNGRADE = {"name": "@astrojs/starlight", "version": "0.39.3", "isSemVerMajor": True}


def _run(tmp_path, report, exit_code, allowlist=None, lock=LOCK):
    report_path = tmp_path / "audit-report.json"
    if isinstance(report, str):
        report_path.write_text(report, encoding="utf-8")
    else:
        report_path.write_text(json.dumps(report), encoding="utf-8")
    allow_path = tmp_path / "audit-allowlist.json"
    allow_path.write_text(json.dumps(allowlist if allowlist is not None else _allowlist()), encoding="utf-8")
    args = [NODE, str(SCRIPT), str(report_path), str(allow_path), str(exit_code)]
    if lock is not None:
        lock_path = tmp_path / "package-lock.json"
        lock_path.write_text(lock if isinstance(lock, str) else json.dumps(lock), encoding="utf-8")
        args.append(str(lock_path))
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def test_a_clean_audit_passes(tmp_path):
    res = _run(tmp_path, _report({}), 0)
    assert res.returncode == 0, res.stderr
    assert "no high/critical advisories" in res.stdout


def test_an_allowed_only_audit_passes_and_names_the_advisory_and_its_dates(tmp_path):
    res = _run(tmp_path, _allowed_report(), 1)
    assert res.returncode == 0, res.stderr
    assert f"advisory {SOURCE} ({GHSA}) in picomatch <=4.0.7" in res.stdout
    assert "added 2026-11-17, expires 2999-12-31" in res.stdout


@pytest.mark.parametrize("severity", ["high", "critical"])
def test_a_different_advisory_fails(tmp_path, severity):
    report = _allowed_report()
    report["vulnerabilities"]["lodash"] = _vuln(
        "lodash", [_advisory(source=1000001, name="lodash", ghsa="GHSA-29mw-wpgm-hmr9", severity=severity)], severity
    )
    res = _run(tmp_path, report, 1)
    assert res.returncode != 0
    assert "no valid exception: 1000001" in res.stderr


def test_a_picomatch_advisory_with_a_different_id_fails(tmp_path):
    report = _report({"picomatch": _vuln("picomatch", [_advisory(source=1000002, ghsa="GHSA-952p-6rrq-rcjv")])})
    res = _run(tmp_path, report, 1)
    assert res.returncode != 0
    assert "no valid exception: 1000002" in res.stderr


@pytest.mark.parametrize("other_severity", ["moderate", "high"])
def test_the_allowed_advisory_plus_any_other_on_the_package_fails(tmp_path, other_severity):
    other = _advisory(source=1000003, ghsa="GHSA-952p-6rrq-rcjv", severity=other_severity, rng="<2.3.1")
    report = _report({"picomatch": _vuln("picomatch", [_advisory(), other])})
    res = _run(tmp_path, report, 1)
    assert res.returncode != 0
    assert "no valid exception: 1000003" in res.stderr


def test_a_transitive_entry_naming_a_package_the_report_does_not_list_fails(tmp_path):
    report = _allowed_report()
    report["vulnerabilities"]["tinyglobby"]["via"] = ["picomatch", "ghost-package"]
    res = _run(tmp_path, report, 1)
    assert res.returncode != 0
    assert "tinyglobby -> ghost-package" in res.stderr


def test_a_transitive_entry_is_judged_by_the_advisories_it_resolves_to(tmp_path):
    # tinyglobby carries no advisory of its own; it resolves through micromatch to an advisory nobody excepted.
    report = _allowed_report()
    report["vulnerabilities"]["tinyglobby"]["via"] = ["micromatch"]
    report["vulnerabilities"]["micromatch"] = _vuln(
        "micromatch", [_advisory(source=1000004, name="micromatch", ghsa="GHSA-952p-6rrq-rcjv", severity="moderate")], "moderate"
    )
    res = _run(tmp_path, report, 1)
    assert res.returncode != 0
    assert "no valid exception: 1000004" in res.stderr


def test_a_transitive_entry_that_resolves_to_no_advisory_fails(tmp_path):
    report = _report({"a": _vuln("a", ["b"]), "b": _vuln("b", ["a"])})
    res = _run(tmp_path, report, 1)
    assert res.returncode != 0
    assert "resolves to no advisory" in res.stderr


def test_the_exception_stops_matching_once_the_reported_range_changes(tmp_path):
    report = _allowed_report()
    report["vulnerabilities"]["picomatch"]["via"][0]["range"] = "<=4.0.8"
    res = _run(tmp_path, report, 1)
    assert res.returncode != 0
    assert 'reported range is "<=4.0.8"' in res.stderr


def test_whitespace_in_the_reported_range_is_ignored(tmp_path):
    report = _allowed_report()
    report["vulnerabilities"]["picomatch"]["via"][0]["range"] = "<= 4.0.7"
    res = _run(tmp_path, report, 1)
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize(
    "fix",
    [
        True,
        {"name": "picomatch", "version": "4.0.8", "isSemVerMajor": False},
        {"name": "astro", "version": "7.5.0", "isSemVerMajor": False},
        {"name": "picomatch", "version": "5.0.0", "isSemVerMajor": True},
        {"name": "@astrojs/starlight", "version": "1.0.0", "isSemVerMajor": True},  # an upgrade, not a downgrade
        {"name": "@astrojs/starlight", "version": "0.39.3-beta.1", "isSemVerMajor": True},  # unparseable version
        {"name": "@astrojs/starlight", "isSemVerMajor": True},  # no version
        {"name": "not-in-the-lockfile", "version": "0.0.1", "isSemVerMajor": True},
        "yes",
    ],
    ids=["true", "picomatch-minor", "parent-minor", "picomatch-major", "parent-major-upgrade",
         "prerelease", "no-version", "unlocked-parent", "string"],
)
def test_the_exception_stops_matching_once_npm_reports_a_fix(tmp_path, fix):
    res = _run(tmp_path, _allowed_report(fix=fix), 1)
    assert res.returncode != 0
    assert "npm reports a fix for picomatch" in res.stderr


def test_a_semver_major_downgrade_of_another_package_is_not_a_fix(tmp_path):
    # The real 2026-11-17 report: npm's only "fix" for picomatch is forcing @astrojs/starlight from 0.42.7 down to 0.39.3.
    res = _run(tmp_path, _allowed_report(fix=DOWNGRADE), 1)
    assert res.returncode == 0, res.stderr
    assert "semver-major downgrade of @astrojs/starlight 0.42.7 -> 0.39.3, not a fix for picomatch" in res.stdout


def test_a_semver_major_suggestion_without_a_lockfile_counts_as_a_fix(tmp_path):
    res = _run(tmp_path, _allowed_report(fix=DOWNGRADE), 1, lock=None)
    assert res.returncode != 0
    assert "npm reports a fix for picomatch" in res.stderr


@pytest.mark.parametrize("lock", ["{not json", json.dumps({"lockfileVersion": 3})], ids=["not-json", "no-packages"])
def test_an_unreadable_lockfile_fails(tmp_path, lock):
    res = _run(tmp_path, _allowed_report(fix=DOWNGRADE), 1, lock=lock)
    assert res.returncode != 0
    assert "Failing closed" in res.stderr


def _real_shape_report():
    """The shape of the real site report on 2026-11-17: picomatch (two copies) reached through the astro toolchain."""
    via = {
        "@astrojs/internal-helpers": ["picomatch"],
        "@astrojs/markdown-satteri": ["@astrojs/internal-helpers"],
        "@astrojs/mdx": ["@astrojs/internal-helpers", "@astrojs/markdown-satteri"],
        "@astrojs/starlight": ["@astrojs/markdown-satteri", "@astrojs/mdx"],
        "anymatch": ["picomatch"],
        "astro": ["@astrojs/internal-helpers", "@astrojs/markdown-satteri", "picomatch", "tinyglobby", "unstorage", "vite"],
        "fdir": ["picomatch"],
        "tinyglobby": ["fdir", "picomatch"],
        "unstorage": ["anymatch"],
        "vite": ["picomatch", "tinyglobby"],
    }
    vulns = {name: _vuln(name, v, fix=DOWNGRADE if name.startswith("@astrojs/") else True, rng="*") for name, v in via.items()}
    vulns["picomatch"] = _vuln("picomatch", [_advisory()], fix=DOWNGRADE)
    return _report(vulns)


def test_the_real_report_shape_passes_with_only_the_picomatch_advisory(tmp_path):
    res = _run(tmp_path, _real_shape_report(), 1)
    assert res.returncode == 0, res.stderr
    assert res.stdout.count(f"advisory {SOURCE} ") == 1


def test_the_real_report_shape_fails_once_astro_carries_its_own_low_and_moderate_advisories(tmp_path):
    # astro is "high" only through picomatch; its own advisories still count, whatever their severity.
    report = _real_shape_report()
    report["vulnerabilities"]["astro"]["via"] += [
        _advisory(source=1246685, name="astro", ghsa="GHSA-mh32-jxcr-mh9f", rng="<=7.4.4", severity="low"),
        _advisory(source=1246686, name="astro", ghsa="GHSA-r3h5-4585-7x3w", rng=">=5.14.3 <=7.4.6", severity="moderate"),
    ]
    res = _run(tmp_path, report, 1)
    assert res.returncode != 0
    assert "no valid exception: 1246685, 1246686" in res.stderr


def test_an_expired_exception_fails(tmp_path):
    res = _run(tmp_path, _allowed_report(), 1, _allowlist(expires="2026-11-16"))
    assert res.returncode != 0
    assert "its exception expired on 2026-11-16" in res.stderr


def test_an_exception_naming_another_package_fails(tmp_path):
    res = _run(tmp_path, _allowed_report(), 1, _allowlist(package="micromatch"))
    assert res.returncode != 0
    assert "the exception names micromatch" in res.stderr


def test_an_exception_without_a_range_does_not_apply(tmp_path):
    res = _run(tmp_path, _allowed_report(), 1, _allowlist(range=None))
    assert res.returncode != 0
    assert "it does not apply" in res.stderr


@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.parametrize(
    "report",
    [
        "{not json",
        "",
        json.dumps({"error": {"code": "ENOTFOUND", "summary": "network"}}),
        json.dumps({"auditReportVersion": 2}),
        json.dumps(_report({"picomatch": {"severity": "high", "via": "picomatch"}})),
        json.dumps(_report({"picomatch": {"severity": "severe", "via": [_advisory()]}})),
    ],
    ids=["not-json", "empty", "npm-error", "no-vulnerabilities", "via-not-a-list", "unknown-severity"],
)
def test_a_malformed_audit_report_fails(tmp_path, report, exit_code):
    res = _run(tmp_path, report, exit_code)
    assert res.returncode != 0


def test_a_nonzero_npm_exit_with_nothing_flagged_fails(tmp_path):
    res = _run(tmp_path, _report({}), 1)
    assert res.returncode != 0
    assert "Failing closed" in res.stderr


def test_the_committed_allowlist_holds_exactly_the_one_dated_exception():
    doc = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    assert doc["description"].startswith("Reviewed, time-boxed npm-audit exceptions")
    (entry,) = doc["exceptions"]
    assert set(entry) == {"id", "package", "range", "reason", "approvedBy", "addedOn", "expiresOn"}
    assert entry["id"] == str(SOURCE)
    assert entry["package"] == "picomatch"
    assert entry["range"] == "<=4.0.7"
    assert entry["approvedBy"] == "mcp-tool-shop"
    assert entry["addedOn"] == "2026-11-17"
    assert entry["expiresOn"] == "2026-12-17"
    assert entry["reason"].startswith(GHSA + " ")
    assert "Remove this entry in the PR that bumps to a patched picomatch." in entry["reason"]


def test_ci_runs_the_gate_in_the_site_audit_step_and_triggers_on_its_files():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "npm audit --audit-level=high --json > audit-report.json" in ci
    assert (
        'node ../scripts/check-audit-allowlist.mjs audit-report.json ../.github/audit-allowlist.json "$AUDIT_EXIT"'
        " package-lock.json\n"
    ) in ci
    assert "npm ci && npm audit --audit-level=high" not in ci
    assert "      - scripts/**\n" in ci
    assert "      - .github/audit-allowlist.json\n" in ci
