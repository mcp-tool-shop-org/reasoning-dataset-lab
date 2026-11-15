#!/usr/bin/env python3
"""Generate and verify the five-judge weights table in `scripts/colab/REAL_SERVICE_ASSUMPTIONS.md`
directly from the five `<owner>__<repo>.sha256.json` receipts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RECEIPTS_DIR = SCRIPT_DIR / "receipts"
DEFAULT_DOC_PATH = SCRIPT_DIR / "REAL_SERVICE_ASSUMPTIONS.md"

BEGIN_MARKER = "<!-- BEGIN_GENERATED_WEIGHTS_TABLE -->"
END_MARKER = "<!-- END_GENERATED_WEIGHTS_TABLE -->"

GIB_DIVISOR = float(1024**3)


def _load_judge_specs() -> dict[str, Any]:
    src_dir = REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from reasoning_dataset_lab.pilot_b.bakeoff_judge import BAKEOFF_JUDGE_SPECS

    return dict(BAKEOFF_JUDGE_SPECS)


def load_and_verify_receipt_rows(
    receipts_dir: Path = DEFAULT_RECEIPTS_DIR,
) -> tuple[list[dict[str, Any]], int, float]:
    """Read the 5 `<owner>__<repo>.sha256.json` receipts, cross-check against `BAKEOFF_JUDGE_SPECS`,
    and return `(rows, sum_total_bytes, sum_seconds)` ordered by receipt filename.
    """
    specs = _load_judge_specs()
    spec_by_filename: dict[str, Any] = {}
    for spec in specs.values():
        owner, repo = spec.repo_id.split("/", 1)
        spec_by_filename[f"{owner}__{repo}.sha256.json"] = spec

    rows: list[dict[str, Any]] = []
    sum_bytes = 0
    sum_seconds = 0.0

    for fname in sorted(spec_by_filename.keys()):
        spec = spec_by_filename[fname]
        rpath = Path(receipts_dir) / fname
        if not rpath.is_file():
            raise FileNotFoundError(f"Missing receipt file: {rpath}")
        doc = json.loads(rpath.read_text(encoding="utf-8"))
        rec_repo = str(doc.get("repo") or "")
        rec_rev = str(doc.get("revision") or "")
        if rec_repo != spec.repo_id or rec_rev != spec.revision:
            raise ValueError(
                f"Receipt {fname} ({rec_repo} @ {rec_rev}) does not match "
                f"BAKEOFF_JUDGE_SPECS[{spec.judge_id!r}] ({spec.repo_id} @ {spec.revision})"
            )
        files_map = doc.get("files")
        if not isinstance(files_map, dict) or not files_map:
            raise ValueError(f"Receipt {fname} has empty or missing 'files' map")
        files_sum = sum(int(meta["bytes"]) for meta in files_map.values())
        total_bytes = int(doc["total_bytes"])
        if files_sum != total_bytes:
            raise ValueError(
                f"Receipt {fname} files byte sum ({files_sum}) != total_bytes ({total_bytes})"
            )
        seconds = float(doc["seconds"])
        rows.append(
            {
                "judge_id": spec.judge_id,
                "receipt_filename": fname,
                "repo": rec_repo,
                "revision": rec_rev,
                "total_bytes": total_bytes,
                "gib": total_bytes / GIB_DIVISOR,
                "seconds": seconds,
            }
        )
        sum_bytes += total_bytes
        sum_seconds += seconds

    return rows, sum_bytes, round(sum_seconds, 1)


def render_weights_table_block(receipts_dir: Path = DEFAULT_RECEIPTS_DIR) -> str:
    """Render the Markdown weights table block from the five `<owner>__<repo>.sha256.json` receipts."""
    rows, sum_bytes, sum_seconds = load_and_verify_receipt_rows(receipts_dir)
    lines = [
        BEGIN_MARKER,
        "| Judge ID | HuggingFace Repo (`revision`) | `total_bytes` (bytes) | Size (GiB) | Rig Download `seconds` (s) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['judge_id']}` | `{r['repo']}` (`{r['revision']}`) | "
            f"`{r['total_bytes']:,}` | `{r['gib']:.2f} GiB` | `{r['seconds']:.1f}` |"
        )
    sum_gib = sum_bytes / GIB_DIVISOR
    rounded_s = int(round(sum_seconds))
    lines.append(
        f"| **Total (5 judges)** | | **`{sum_bytes:,}`** | **`{sum_gib:.2f} GiB`** | "
        f"**`{sum_seconds:.1f}` (`{rounded_s}` s)** |"
    )
    lines.append(END_MARKER)
    return "\n".join(lines)


def extract_weights_table_block(doc_text: str) -> str:
    """Extract the `<!-- BEGIN_GENERATED_WEIGHTS_TABLE --> ... <!-- END_GENERATED_WEIGHTS_TABLE -->` block from `doc_text`."""
    start_idx = doc_text.find(BEGIN_MARKER)
    end_idx = doc_text.find(END_MARKER)
    if start_idx < 0 or end_idx < 0 or end_idx < start_idx:
        raise ValueError(
            f"Document is missing {BEGIN_MARKER} ... {END_MARKER} markers"
        )
    return doc_text[start_idx : end_idx + len(END_MARKER)]


def load_weights_rows_from_receipts(
    receipts_dir: Path = DEFAULT_RECEIPTS_DIR,
) -> list[dict[str, Any]]:
    """Return verified receipt rows ordered by receipt filename."""
    rows, _, _ = load_and_verify_receipt_rows(receipts_dir)
    return rows


def verify_doc_matches_receipts(
    doc_path: Path = DEFAULT_DOC_PATH,
    receipts_dir: Path = DEFAULT_RECEIPTS_DIR,
) -> bool:
    """Return True if the weights table block in `doc_path` matches the receipts in `receipts_dir` byte-for-byte."""
    rendered = render_weights_table_block(receipts_dir)
    doc_text = Path(doc_path).read_text(encoding="utf-8")
    committed = extract_weights_table_block(doc_text)
    return committed == rendered


def update_doc_weights_table(
    doc_path: Path = DEFAULT_DOC_PATH,
    receipts_dir: Path = DEFAULT_RECEIPTS_DIR,
) -> str:
    """Update the generated weights table block in `doc_path` from `receipts_dir` and return the rendered block."""
    rendered = render_weights_table_block(receipts_dir)
    doc_text = Path(doc_path).read_text(encoding="utf-8")
    current_block = extract_weights_table_block(doc_text)
    updated_text = doc_text.replace(current_block, rendered, 1)
    Path(doc_path).write_text(updated_text, encoding="utf-8", newline="\n")
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify the weights table in REAL_SERVICE_ASSUMPTIONS.md from the 5 receipts."
    )
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        default=DEFAULT_RECEIPTS_DIR,
        help="Directory containing the 5 <owner>__<repo>.sha256.json receipts.",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=DEFAULT_DOC_PATH,
        help="Path to REAL_SERVICE_ASSUMPTIONS.md.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed doc matches the generated table without modifying the doc.",
    )
    args = parser.parse_args(argv)

    rendered = render_weights_table_block(args.receipts_dir)
    if args.check:
        doc_text = args.doc.read_text(encoding="utf-8")
        committed = extract_weights_table_block(doc_text)
        if committed != rendered:
            sys.stderr.write(
                "Weights table in REAL_SERVICE_ASSUMPTIONS.md does not match receipts!\n"
            )
            return 1
        print("WEIGHTS_TABLE_OK")
        return 0

    update_doc_weights_table(args.doc, args.receipts_dir)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
