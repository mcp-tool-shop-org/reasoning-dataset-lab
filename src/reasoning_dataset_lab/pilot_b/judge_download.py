"""Per-judge weight download child entrypoint (`python -m reasoning_dataset_lab.pilot_b.judge_download`).

Runs inside the target judge's synced CPython 3.15 virtualenv (`venv-d3` for `d3-mini`,
`venv-judge` for the other four judges). Imports only the Python standard library and
`huggingface_hub` (plus optional `hf_xet` version inspection) — never `torch`,
`transformers`, `pydantic`, or `bakeoff_judge`.

Responsibilities:
1. Read `huggingface_hub.__version__` and the installed `hf_xet` version inside the child
   venv and verify they match the pair pinned in that venv's lock file
   (`venv-judge`: `huggingface-hub==2.4.0`, `hf-xet==1.8.0`;
    `venv-d3`:    `huggingface-hub==1.33.0`, `hf-xet==1.8.0`).
   Refuse with `DOWNLOAD_STACK_VERSION_MISMATCH` (`rc=2`) on any mismatch.
2. Download the single judge repository at its pinned 40-char commit SHA via
   `huggingface_hub.snapshot_download` with `max_workers` and
   `HF_XET_NUM_CONCURRENT_RANGE_GETS` applied.
3. Atomically write the JSON download done-file (`judge_id`, `repo_id`, `revision`,
   `venv`, `snapshot_dir`, `files_downloaded`, `files`, `bytes`, `seconds`,
   `huggingface_hub_version`, `hf_xet_version`, `max_workers`,
   `hf_xet_num_concurrent_range_gets`) and exit so all `hf-xet` and Rust/Tokio
   allocations are released before the next child starts.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import huggingface_hub  # type: ignore[import-not-found]


def detect_download_stack_versions() -> tuple[str, str]:
    """Return `(huggingface_hub_version, hf_xet_version)` inside the active virtualenv."""
    hf_hub_ver = str(getattr(huggingface_hub, "__version__", "") or "").strip()
    if not hf_hub_ver:
        try:
            hf_hub_ver = importlib.metadata.version("huggingface-hub").strip()
        except importlib.metadata.PackageNotFoundError:
            hf_hub_ver = "unknown"

    hf_xet_ver = ""
    try:
        import hf_xet  # type: ignore[import-not-found]

        hf_xet_ver = str(getattr(hf_xet, "__version__", "") or "").strip()
    except ImportError:
        hf_xet_ver = ""
    if not hf_xet_ver:
        try:
            hf_xet_ver = importlib.metadata.version("hf-xet").strip()
        except importlib.metadata.PackageNotFoundError:
            hf_xet_ver = "unknown"

    return hf_hub_ver, hf_xet_ver


def _resolve_snapshot_dir(
    repo_id: str,
    revision: str,
    hf_home: Path,
    returned_path: str | Path | None = None,
) -> Path:
    """Resolve the on-disk snapshot directory for `repo_id` at `revision`."""
    candidates: list[Path] = []
    if returned_path is not None:
        rp = Path(returned_path)
        if rp.is_dir():
            candidates.append(rp)
    owner, repo = str(repo_id).split("/", 1)
    model_dir_name = f"models--{owner}--{repo}"
    candidates.append(hf_home / "hub" / model_dir_name / "snapshots" / str(revision))
    candidates.append(hf_home / model_dir_name / "snapshots" / str(revision))
    for cand in candidates:
        if cand.is_dir():
            return cand
    return candidates[0]


def _collect_snapshot_files(snap_dir: Path) -> tuple[list[dict[str, Any]], int]:
    """Enumerate regular files under `snap_dir` and return `(files_list, total_bytes)`."""
    if not snap_dir.is_dir():
        return [], 0
    files_list: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(snap_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(snap_dir).as_posix()
            size_b = int(path.stat().st_size)
            files_list.append({"path": rel, "bytes": size_b})
            total_bytes += size_b
    return files_list, total_bytes


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `python -m reasoning_dataset_lab.pilot_b.judge_download`."""
    parser = argparse.ArgumentParser(
        prog="python -m reasoning_dataset_lab.pilot_b.judge_download",
        description="Download one bake-off judge snapshot inside its synced virtualenv.",
    )
    parser.add_argument("--judge", required=True, help="Canonical judge ID.")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID (<owner>/<repo>).")
    parser.add_argument("--revision", required=True, help="Pinned 40-char commit SHA.")
    parser.add_argument("--hf-home", type=Path, required=True, help="HF_HOME directory.")
    parser.add_argument("--done-file", type=Path, required=True, help="Output done-file JSON path.")
    parser.add_argument("--venv", default="", help="Virtualenv name (venv-judge or venv-d3).")
    parser.add_argument("--max-workers", type=int, default=2, help="snapshot_download max_workers (>= 1).")
    parser.add_argument(
        "--hf-xet-num-concurrent-range-gets",
        type=int,
        default=4,
        help="HF_XET_NUM_CONCURRENT_RANGE_GETS value (>= 0; 0 unsets the env var).",
    )
    parser.add_argument(
        "--expect-huggingface-hub-version",
        default="",
        help="Expected huggingface_hub.__version__ pinned in the venv lock.",
    )
    parser.add_argument(
        "--expect-hf-xet-version",
        default="",
        help="Expected hf_xet version pinned in the venv lock.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Skip live network download unless _IS_FAKE_HF_HUB is active.",
    )
    return parser


def main_download_runner(argv: Sequence[str] | None = None) -> int:
    """Execute the single-judge weight download in the child process and write the done-file."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    hf_hub_ver, hf_xet_ver = detect_download_stack_versions()
    expect_hub = str(args.expect_huggingface_hub_version or "").strip()
    expect_xet = str(args.expect_hf_xet_version or "").strip()

    if (expect_hub and hf_hub_ver != expect_hub) or (
        expect_xet and hf_xet_ver != expect_xet
    ):
        sys.stderr.write(
            f"DOWNLOAD_STACK_VERSION_MISMATCH: judge={args.judge} venv={args.venv} "
            f"expected huggingface-hub=={expect_hub}, hf-xet=={expect_xet}; "
            f"got huggingface-hub=={hf_hub_ver!r}, hf-xet=={hf_xet_ver!r}\n"
        )
        return 2

    if int(args.hf_xet_num_concurrent_range_gets) == 0:
        os.environ.pop("HF_XET_NUM_CONCURRENT_RANGE_GETS", None)
    else:
        os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = str(
            int(args.hf_xet_num_concurrent_range_gets)
        )

    hf_home = Path(args.hf_home)
    hub_cache = hf_home / "hub"
    dl_fn = getattr(huggingface_hub, "snapshot_download", None)
    is_fake_hub = bool(getattr(huggingface_hub, "_IS_FAKE_HF_HUB", False))

    t0 = time.monotonic()
    returned_snap: str | Path | None = None
    if dl_fn is not None and (not args.dryrun or is_fake_hub):
        hub_cache.mkdir(parents=True, exist_ok=True)
        dl_kwargs: dict[str, Any] = {
            "repo_id": str(args.repo_id),
            "revision": str(args.revision),
            "cache_dir": str(hub_cache),
        }
        sig = inspect.signature(dl_fn)
        accepts_mw = "max_workers" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_mw:
            dl_kwargs["max_workers"] = int(args.max_workers)
        returned_snap = dl_fn(**dl_kwargs)

    elapsed_s = max(0.0, time.monotonic() - t0)
    snap_dir = _resolve_snapshot_dir(
        str(args.repo_id),
        str(args.revision),
        hf_home,
        returned_snap,
    )
    files_list, total_bytes = _collect_snapshot_files(snap_dir)

    done_file = Path(args.done_file)
    done_file.parent.mkdir(parents=True, exist_ok=True)
    done_doc: dict[str, Any] = {
        "judge_id": str(args.judge),
        "repo_id": str(args.repo_id),
        "revision": str(args.revision),
        "venv": str(args.venv),
        "snapshot_dir": snap_dir.as_posix(),
        "files_downloaded": len(files_list),
        "files": files_list,
        "bytes": int(total_bytes),
        "seconds": round(elapsed_s, 6),
        "max_workers": int(args.max_workers),
        "hf_xet_num_concurrent_range_gets": int(args.hf_xet_num_concurrent_range_gets),
        "huggingface_hub_version": hf_hub_ver,
        "hf_xet_version": hf_xet_ver,
    }
    tmp_done = done_file.with_suffix(done_file.suffix + ".tmp")
    tmp_done.write_text(
        json.dumps(done_doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(tmp_done, done_file)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main_download_runner())
