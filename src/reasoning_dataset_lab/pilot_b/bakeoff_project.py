"""The upload projection of a sealed bake-off part (`rdlab bakeoff-project`).

A Colab judge window uploads only `{id, material, claim}` per item (the VM's field allow-list), the whole part in the
part's order for a counted window. This module writes that projection from a part directory (`part.json`,
`checksums.txt`, `items.jsonl`) to a file outside the part, after checking the part's own files against its
`checksums.txt`, and records the projection's SHA-256. The launcher's binding (`vm_judges.bind_items_to_part`) accepts
exactly this output: the same ids in the same order, `material` and `claim` equal to the part's row by row.

It never reads a sealed store, never writes into the part directory or under any `sealed_store/`, and never overwrites
an existing file.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from reasoning_dataset_lab.errors import refused_error

PROJECTION_FIELDS: tuple[str, ...] = ("id", "material", "claim")
_CHECKSUM_LINE_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9._-]+)$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_checksums(part_dir: Path) -> dict[str, str]:
    path = part_dir / "checksums.txt"
    if not path.is_file():
        raise refused_error(
            "PROJECTION_PART_INVALID",
            f"The part has no checksums.txt at {path}.",
            "Pass the sealed part directory (part.json, checksums.txt, items.jsonl), e.g. .../part_bakeoff_sealed.",
        )
    out: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        m = _CHECKSUM_LINE_RE.match(line)
        if m is None or m.group(2) in out:
            raise refused_error(
                "PROJECTION_PART_INVALID",
                f"checksums.txt line {lineno} is not a unique '<sha256>  <name>' line.",
                "Restore the part's checksums.txt as the builder wrote it.",
            )
        out[m.group(2)] = m.group(1)
    return out


def _forbidden_out(out_path: Path, part_dir: Path) -> str | None:
    resolved_out = out_path.resolve()
    resolved_part = part_dir.resolve()
    if resolved_out == resolved_part or resolved_part in resolved_out.parents:
        return "inside the part directory"
    if any(p.lower() == "sealed_store" for p in resolved_out.parts):
        return "under a sealed_store directory"
    return None


def project_part_items(part_dir: Path, out_path: Path) -> dict[str, Any]:
    """Write the `{id, material, claim}` projection of every item of the part in `part_dir`, in the part's order, to
    `out_path`, and `<out_path>.sha256` beside it. Returns the record printed by the CLI: part id, counts, and the
    SHA-256 of `part.json`, the part's `items.jsonl` and the projection.

    Refuses `PROJECTION_OUT_FORBIDDEN` when `out_path` is inside the part directory or under any `sealed_store/`,
    `PROJECTION_OUT_EXISTS` when it (or its `.sha256`) already exists, and `PROJECTION_PART_INVALID` when the part's
    `part.json` or `items.jsonl` does not hash to its `checksums.txt` line, `part.json` has no `part_id` /
    `items_count`, the row count differs from `items_count`, or a row lacks a string `id`, `material` or `claim`.
    """
    part = Path(part_dir)
    out = Path(out_path)
    sha_out = out.with_name(out.name + ".sha256")
    why = _forbidden_out(out, part)
    if why is not None:
        raise refused_error(
            "PROJECTION_OUT_FORBIDDEN",
            f"The projection may not be written {why} ({out}).",
            "Write it to a separate inputs directory outside the part and outside every sealed_store/.",
        )
    for f in (out, sha_out):
        if f.exists():
            raise refused_error(
                "PROJECTION_OUT_EXISTS",
                f"{f} already exists; the projection never overwrites a file.",
                "Choose a new --out path or remove the old projection first.",
            )
    if any(p.lower() == "sealed_store" for p in part.resolve().parts):
        raise refused_error(
            "PROJECTION_PART_INVALID",
            f"{part} is under a sealed_store directory; the projection reads only the part's public files.",
            "Pass the part directory (part_bakeoff_sealed), not the sealed store.",
        )
    sums = _read_checksums(part)
    part_json_bytes = _read_listed(part, "part.json", sums)
    items_bytes = _read_listed(part, "items.jsonl", sums)
    try:
        meta = json.loads(part_json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise refused_error("PROJECTION_PART_INVALID", f"part.json does not parse: {exc}.", "Restore the part.") from exc
    part_id = meta.get("part_id") if isinstance(meta, dict) else None
    items_count = meta.get("items_count") if isinstance(meta, dict) else None
    if not isinstance(part_id, str) or not part_id.strip() or isinstance(items_count, bool) or not isinstance(items_count, int):
        raise refused_error(
            "PROJECTION_PART_INVALID",
            "part.json has no string part_id or no integer items_count.",
            "Pass a part built by rdlab bakeoff-part.",
        )
    lines: list[str] = []
    for lineno, raw in enumerate(items_bytes.decode("utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise refused_error(
                "PROJECTION_PART_INVALID", f"items.jsonl line {lineno} is not a JSON row.", "Restore the part."
            ) from exc
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in PROJECTION_FIELDS):
            raise refused_error(
                "PROJECTION_PART_INVALID",
                f"items.jsonl line {lineno} lacks a string id, material or claim.",
                "Restore the part.",
            )
        lines.append(json.dumps({k: row[k] for k in PROJECTION_FIELDS}, ensure_ascii=False))
    if len(lines) != items_count:
        raise refused_error(
            "PROJECTION_PART_INVALID",
            f"items.jsonl holds {len(lines)} item(s) but part.json items_count is {items_count}.",
            "Restore the part.",
        )
    ids = [json.loads(ln)["id"] for ln in lines]
    if len(set(ids)) != len(ids):
        raise refused_error("PROJECTION_PART_INVALID", "items.jsonl repeats an item id.", "Restore the part.")
    data = ("\n".join(lines) + "\n").encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as fh:
        fh.write(data)
    projection_sha = _sha256_bytes(data)
    with sha_out.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{projection_sha}  {out.name}\n")
    return {
        "status": "ok",
        "part_id": part_id.strip(),
        "items_count": items_count,
        "projected_items": len(lines),
        "fields": list(PROJECTION_FIELDS),
        "part_json_sha256": sums["part.json"],
        "items_jsonl_sha256": sums["items.jsonl"],
        "out": str(out),
        "projection_sha256": projection_sha,
        "sha256_file": str(sha_out),
    }


def _read_listed(part: Path, name: str, sums: dict[str, str]) -> bytes:
    path = part / name
    if name not in sums or not path.is_file():
        raise refused_error(
            "PROJECTION_PART_INVALID",
            f"{name} is missing from the part or not listed in its checksums.txt.",
            "Pass the sealed part directory (part.json, checksums.txt, items.jsonl).",
        )
    data = path.read_bytes()
    if _sha256_bytes(data) != sums[name]:
        raise refused_error(
            "PROJECTION_PART_INVALID",
            f"{name} does not hash to its checksums.txt line.",
            "The part was changed after it was built; rebuild or restore it.",
        )
    return data
