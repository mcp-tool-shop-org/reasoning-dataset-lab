"""Manifest verification and optional local-git re-derivation for historical test fixtures.

Each historical regression test reads only the committed snippet fixtures under
`tests/fixtures/`. When the source commits are present in the local git object
store (full clone/worktree), `verify_and_rederive_historical_fixtures` also
re-derives every snippet from `git show <source_sha>:<path>` and checks that its
bytes and SHA-256 match the committed fixture and manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIXTURES_DIR.parents[1]
MANIFEST_PATH = FIXTURES_DIR / "historical_git_fixtures_manifest.json"


def load_historical_fixtures_manifest(
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    """Load and validate the schema of `historical_git_fixtures_manifest.json`."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "Manifest root must be a JSON object"
    fixtures = data.get("fixtures")
    assert isinstance(fixtures, list) and fixtures, "Manifest must contain a non-empty 'fixtures' list"
    for entry in fixtures:
        assert isinstance(entry, dict)
        for req_key in ("fixture", "source_sha", "path", "start_line", "end_line", "sha256"):
            assert req_key in entry, f"Missing required key {req_key!r} in manifest entry {entry!r}"
        assert isinstance(entry["fixture"], str) and entry["fixture"]
        assert isinstance(entry["source_sha"], str) and len(entry["source_sha"]) == 40
        assert isinstance(entry["path"], str) and entry["path"]
        assert isinstance(entry["start_line"], int) and entry["start_line"] >= 1
        assert isinstance(entry["end_line"], int) and entry["end_line"] >= entry["start_line"]
        assert isinstance(entry["sha256"], str) and len(entry["sha256"]) == 64
    return data


def verify_committed_fixtures_against_manifest(
    manifest_path: Path = MANIFEST_PATH,
    fixtures_dir: Path = FIXTURES_DIR,
) -> list[dict[str, Any]]:
    """Verify every committed fixture file exists and matches its manifest SHA-256."""
    manifest = load_historical_fixtures_manifest(manifest_path)
    entries: list[dict[str, Any]] = list(manifest["fixtures"])
    for entry in entries:
        fixture_file = fixtures_dir / entry["fixture"]
        assert fixture_file.is_file(), f"Missing committed fixture file: {fixture_file}"
        raw_bytes = fixture_file.read_bytes()
        assert b"\r\n" not in raw_bytes, f"Fixture {fixture_file.name} must use LF line endings"
        actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        assert actual_sha256 == entry["sha256"], (
            f"SHA-256 mismatch for {fixture_file.name}: "
            f"expected {entry['sha256']}, got {actual_sha256}"
        )
        lines = raw_bytes.decode("utf-8").splitlines()
        expected_line_count = entry["end_line"] - entry["start_line"] + 1
        assert len(lines) == expected_line_count, (
            f"Line count mismatch for {fixture_file.name}: "
            f"expected {expected_line_count}, got {len(lines)}"
        )
    return entries


def is_commit_present_locally(commit_sha: str, repo_root: Path = REPO_ROOT) -> bool:
    """Return True if `commit_sha` exists in the local git object store (no network)."""
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit_sha}^{{commit}}"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    return proc.returncode == 0


def derive_fixture_snippet_from_git(
    entry: dict[str, Any],
    repo_root: Path = REPO_ROOT,
) -> tuple[str, str]:
    """Extract lines `start_line..end_line` from `<source_sha>:<path>` and return `(text, sha256)`."""
    source_sha = entry["source_sha"]
    rel_path = entry["path"]
    start_line = int(entry["start_line"])
    end_line = int(entry["end_line"])
    raw_source = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{source_sha}:{rel_path}"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    ).stdout
    source_lines = raw_source.splitlines()
    assert 1 <= start_line <= end_line <= len(source_lines), (
        f"Invalid span {start_line}..{end_line} for {source_sha}:{rel_path} "
        f"({len(source_lines)} lines)"
    )
    snippet = "\n".join(source_lines[start_line - 1 : end_line]) + "\n"
    digest = hashlib.sha256(snippet.encode("utf-8")).hexdigest()
    return snippet, digest


def rederive_and_check_all_from_git(
    manifest_path: Path = MANIFEST_PATH,
    fixtures_dir: Path = FIXTURES_DIR,
    repo_root: Path = REPO_ROOT,
    *,
    write_files: bool = False,
) -> tuple[bool, list[str]]:
    """Re-derive all fixtures when their source commits exist locally.

    Returns `(all_commits_present, missing_shas)`.
    """
    manifest = load_historical_fixtures_manifest(manifest_path)
    entries: list[dict[str, Any]] = list(manifest["fixtures"])
    unique_shas = sorted({str(e["source_sha"]) for e in entries})
    missing_shas = [
        sha for sha in unique_shas if not is_commit_present_locally(sha, repo_root=repo_root)
    ]
    if missing_shas:
        return False, missing_shas

    for entry in entries:
        snippet, digest = derive_fixture_snippet_from_git(entry, repo_root=repo_root)
        fixture_file = fixtures_dir / entry["fixture"]
        if write_files:
            fixture_file.write_text(snippet, encoding="utf-8", newline="\n")
            entry["sha256"] = digest
        else:
            assert digest == entry["sha256"], (
                f"Re-derived SHA-256 mismatch for {entry['fixture']} from "
                f"{entry['source_sha']}:{entry['path']}:{entry['start_line']}-{entry['end_line']}: "
                f"expected {entry['sha256']}, got {digest}"
            )
            committed_text = fixture_file.read_text(encoding="utf-8")
            assert committed_text == snippet, (
                f"Committed fixture {entry['fixture']} differs from "
                f"{entry['source_sha']}:{entry['path']}:{entry['start_line']}-{entry['end_line']}"
            )

    if write_files:
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return True, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify or re-derive historical git fixtures under tests/fixtures/."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Re-derive fixture snippets from local git commits and update files + manifest SHA-256s.",
    )
    args = parser.parse_args(argv)
    if args.write:
        ok, missing = rederive_and_check_all_from_git(write_files=True)
        if not ok:
            raise SystemExit(f"Cannot --write fixtures; missing local commits: {missing}")
    verify_committed_fixtures_against_manifest()
    ok, missing = rederive_and_check_all_from_git(write_files=False)
    if not ok:
        print(f"MANIFEST_SHA256_OK (git re-derivation skipped; missing commits: {missing})")
        return 0
    print("MANIFEST_AND_GIT_REDERIVATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
