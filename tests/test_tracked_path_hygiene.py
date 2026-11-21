"""Check that tracked files outside `atlas/` do not contain user-home or internal paths."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b.bakeoff_part import check_output_dir_outside_repo

ROOT = Path(__file__).resolve().parents[1]
_INTERNAL_FOLDER = "".join(("rnd", "-local"))

_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "windows_user_dir",
        re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\+|/)Users(?:\\+|/)[^\\/\s'\"`<>|?*]+", re.IGNORECASE),
    ),
    (
        "posix_user_dir",
        re.compile(r"(?<![A-Za-z0-9._-])/(?:home|Users)/[^/\s'\"`<>()\[\]{}]+/"),
    ),
    (
        "internal_folder",
        re.compile(re.escape(_INTERNAL_FOLDER), re.IGNORECASE),
    ),
)


def _tracked_non_atlas_files(root: Path) -> list[str]:
    out = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"],
        stdin=subprocess.DEVNULL,
    )
    paths = [p for p in out.decode("utf-8", errors="replace").split("\0") if p]
    return [rel for rel in paths if not rel.startswith("atlas/")]


def _scan_tracked_files(root: Path) -> list[str]:
    """Return `path:line` hits for forbidden path patterns across non-binary tracked files outside `atlas/`."""
    hits: list[str] = []
    for rel in _tracked_non_atlas_files(root):
        path = root / rel
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if b"\x00" in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if any(pattern.search(line) for _name, pattern in _FORBIDDEN_PATTERNS):
                hits.append(f"{rel}:{line_no}")
    return hits


def test_tracked_files_contain_no_user_or_internal_paths() -> None:
    tracked = _tracked_non_atlas_files(ROOT)
    assert "tests/test_tracked_path_hygiene.py" in tracked
    hits = _scan_tracked_files(ROOT)
    assert not hits, "Forbidden path pattern found in tracked file(s): " + ", ".join(hits)


def test_path_hygiene_scanner_catches_all_three_patterns_skips_binaries_and_hides_text(
    tmp_path: Path,
) -> None:
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)],
        stdin=subprocess.DEVNULL,
        check=True,
    )
    (tmp_path / "win.txt").write_text(
        "\n".join(
            [
                "clean header",
                ":".join(("D", "\\" + "Users" + "\\acct_01\\work")),
                ":".join(("c", "/" + "Users" + "/acct_02/work")),
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (tmp_path / "posix.txt").write_text(
        "\n".join(
            [
                "path=/" + "home" + "/acct_01/work",
                "other=/" + "Users" + "/acct_02/work",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (tmp_path / "internal.txt").write_text(
        "\n".join(
            [
                "line1",
                "line2",
                "line3",
                "E:/work/" + "".join(("rnd", "-local")) + "/part/",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01" + _INTERNAL_FOLDER.encode("utf-8"))
    (tmp_path / "atlas").mkdir()
    (tmp_path / "atlas" / "ignored.json").write_text(_INTERNAL_FOLDER + "\n", encoding="utf-8", newline="\n")
    (tmp_path / "clean.txt").write_text("<external-dir>/part/\n", encoding="utf-8", newline="\n")

    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "-A"],
        stdin=subprocess.DEVNULL,
        check=True,
    )

    hits = _scan_tracked_files(tmp_path)
    assert hits == [
        "internal.txt:4",
        "posix.txt:1",
        "posix.txt:2",
        "win.txt:2",
        "win.txt:3",
    ]
    joined = "\n".join(hits)
    assert "acct_01" not in joined
    assert "acct_02" not in joined
    assert _INTERNAL_FOLDER not in joined


def test_output_dir_inside_repo_hint_uses_neutral_placeholder(tmp_path: Path) -> None:
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    inside_dir = fake_repo / "sub" / "part"
    try:
        check_output_dir_outside_repo(inside_dir, repo_root=fake_repo)
    except RdlabError as exc:
        assert exc.code == "BAKEOFF_OUTPUT_INSIDE_REPO"
        assert exc.hint == "Write the bake-off part to an external directory outside the repo (e.g. <external-dir>/part/)."
    else:
        raise AssertionError("Expected RdlabError for inside-repo output dir")
