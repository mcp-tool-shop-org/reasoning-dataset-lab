"""The label-blind LC-1 anchor set for a lock-check window (`rdlab bakeoff-lockcheck-anchors`).

A lock-check window's `--anchor-items` are 50 items of the counted part, one per triad, five per domain, chosen by a
fixed rule that reads no label. This module writes them as the upload projection (`{id, material, claim}`, the
`bakeoff-project` allow-list) plus a manifest of the rule and its counts.

The rule (`rdlab-lc1-anchors-v1`):
- **Pool:** the counted part's item ids minus the 30 determinism-canary item ids of `canary_config.json`
  (`canary_all_30_item_ids`); "canary" names only those determinism items.
- **Key:** `sha256(seed + "|" + item_id)` in hex, with the seed `rdlab-lc1-anchors-v1|20261030|bakeoff-d-t10-anchored-sealed`.
- **Selection:** within each domain, sort the pool by `(key, item_id)` and take the first 5, skipping any item whose
  triad is already taken: 5 items x 10 domains, one item per triad, 50 triads.
- **Output order:** the 50 sorted by `(key, item_id)`.
- **Self-check:** 50 items, 10 domains x 5, 50 distinct triads; anything else refuses `LC1_ANCHORS_SPLIT_IMPOSSIBLE`.

What it reads from each row of the part's own `items.jsonl` (checked against the part's `checksums.txt`, never a
sealed store): `id`; `domain` (the item's subject area, the same for the three items of a triad); and `material`,
only through its SHA-256, which groups the three items that share one material into a triad. It never reads the key,
class, gold, `triad_id`, `trait`, `kind`, `closure`, `proof_basis`, `key_basis`, `split` or `messages` fields, the item
number, `n mod 3` or a position in the part. `claim` is copied into the projection only, never used to choose.

After LC-1, the 25 lowest-margin of these 50 are recorded as the window's **low-margin anchors** (the shard
manifest's second anchor half; its JSON field names are unchanged).
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reasoning_dataset_lab.errors import refused_error
from reasoning_dataset_lab.pilot_b.bakeoff_project import (
    PROJECTION_FIELDS,
    _forbidden_out,
    _read_checksums,
    _read_listed,
)

LC1_ANCHOR_RULE_ID = "rdlab-lc1-anchors-v1"
LC1_ANCHOR_SEED = "rdlab-lc1-anchors-v1|20261030|bakeoff-d-t10-anchored-sealed"
LC1_ANCHORS_PER_DOMAIN = 5
LC1_ANCHOR_DOMAIN_COUNT = 10
LC1_ANCHOR_COUNT = LC1_ANCHORS_PER_DOMAIN * LC1_ANCHOR_DOMAIN_COUNT
DEFAULT_CANARY_CONFIG_PATH = Path(__file__).resolve().parent / "canary_config.json"
# The only row fields the selection reads (`material` only as its SHA-256).
LC1_SELECTION_FIELDS: tuple[str, ...] = ("id", "domain", "material")


def lc1_anchor_key(item_id: str, *, seed: str = LC1_ANCHOR_SEED) -> str:
    """The rule's per-item key: `sha256(seed + "|" + item_id)`, hex."""
    return hashlib.sha256(f"{seed}|{item_id}".encode("utf-8")).hexdigest()


def sorted_ids_sha256(ids: Sequence[str]) -> str:
    """SHA-256 of the id list as written in the manifest (`"\\n".join(ids) + "\\n"`, UTF-8)."""
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def load_canary_item_ids(config_path: Path | None = None) -> list[str]:
    """The 30 determinism-canary item ids (`canary_all_30_item_ids`) of `canary_config.json`."""
    target = Path(config_path) if config_path is not None else DEFAULT_CANARY_CONFIG_PATH
    try:
        doc = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise refused_error(
            "LC1_ANCHORS_CANARY_CONFIG_INVALID",
            f"Cannot read the canary config {target}: {type(exc).__name__}.",
            "Pass the canary_config.json of the counted part (src/reasoning_dataset_lab/pilot_b/canary_config.json).",
        ) from exc
    ids = doc.get("canary_all_30_item_ids") if isinstance(doc, dict) else None
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x for x in ids) or len(set(ids)) != len(ids):
        raise refused_error(
            "LC1_ANCHORS_CANARY_CONFIG_INVALID",
            f"{target} has no list of distinct canary_all_30_item_ids.",
            "Pass the canary_config.json of the counted part.",
        )
    return list(ids)


def select_lc1_anchors(
    rows: Iterable[Mapping[str, Any]],
    canary_item_ids: Iterable[str],
    *,
    seed: str = LC1_ANCHOR_SEED,
    per_domain: int = LC1_ANCHORS_PER_DOMAIN,
    domain_count: int = LC1_ANCHOR_DOMAIN_COUNT,
) -> dict[str, Any]:
    """Apply the LC-1 rule to `rows` (dicts; only `id`, `domain` and the SHA-256 of `material` are read) and return the
    selection: the 50 ids in output order, and the counts the manifest records. Refuses `LC1_ANCHORS_SPLIT_IMPOSSIBLE`
    when the pool cannot give `domain_count` domains x `per_domain` items on distinct triads, and
    `LC1_ANCHORS_PART_INVALID` when a row lacks a field or a triad spans two domains."""
    canary = set(canary_item_ids)
    triad_of: dict[str, str] = {}
    domain_of: dict[str, str] = {}
    triad_domain: dict[str, str] = {}
    all_ids: list[str] = []
    for lineno, row in enumerate(rows, start=1):
        vals = {k: row.get(k) for k in LC1_SELECTION_FIELDS}
        if not all(isinstance(v, str) and v for v in vals.values()):
            raise refused_error(
                "LC1_ANCHORS_PART_INVALID",
                f"items.jsonl row {lineno} lacks a string id, domain or material.",
                "Pass the counted part as the builder wrote it.",
            )
        iid, dom = str(vals["id"]), str(vals["domain"])
        if iid in triad_of:
            raise refused_error("LC1_ANCHORS_PART_INVALID", "items.jsonl repeats an item id.", "Restore the part.")
        triad = hashlib.sha256(str(vals["material"]).encode("utf-8")).hexdigest()
        if triad_domain.setdefault(triad, dom) != dom:
            raise refused_error(
                "LC1_ANCHORS_PART_INVALID",
                "Two items that share one material (one triad) carry different domains.",
                "Pass the counted part as the builder wrote it.",
            )
        triad_of[iid] = triad
        domain_of[iid] = dom
        all_ids.append(iid)
    missing_canary = sorted(canary - set(all_ids))
    if missing_canary:
        raise refused_error(
            "LC1_ANCHORS_CANARY_NOT_IN_PART",
            f"{len(missing_canary)} determinism-canary id(s) from the canary config are not in this part.",
            "Pass the counted part the canary config was drawn from.",
        )
    pool = [iid for iid in all_ids if iid not in canary]
    keys = {iid: lc1_anchor_key(iid, seed=seed) for iid in pool}
    by_domain: dict[str, list[str]] = defaultdict(list)
    for iid in pool:
        by_domain[domain_of[iid]].append(iid)

    taken_triads: set[str] = set()
    chosen: list[str] = []
    per_domain_counts: dict[str, int] = {}
    for dom in sorted(by_domain):
        n = 0
        for iid in sorted(by_domain[dom], key=lambda x: (keys[x], x)):
            if n >= per_domain:
                break
            if triad_of[iid] in taken_triads:
                continue
            taken_triads.add(triad_of[iid])
            chosen.append(iid)
            n += 1
        per_domain_counts[dom] = n

    expected_total = per_domain * domain_count
    chosen_triads = {triad_of[i] for i in chosen}
    if (
        len(by_domain) != domain_count
        or len(chosen) != expected_total
        or any(c != per_domain for c in per_domain_counts.values())
        or len(chosen_triads) != expected_total
    ):
        raise refused_error(
            "LC1_ANCHORS_SPLIT_IMPOSSIBLE",
            f"The pool gives {len(chosen)} item(s) over {len(by_domain)} domain(s) and {len(chosen_triads)} triad(s); "
            f"the rule needs {domain_count} domains x {per_domain} items on {expected_total} distinct triads "
            f"(per domain: {dict(sorted(per_domain_counts.items()))}).",
            "Pass the counted part and its canary config; the rule has no fallback.",
        )
    ordered = sorted(chosen, key=lambda x: (keys[x], x))
    per_triad_counts: dict[str, int] = defaultdict(int)
    for iid in ordered:
        per_triad_counts[triad_of[iid]] += 1
    return {
        "item_ids": ordered,
        "pool_size": len(pool),
        "part_item_count": len(all_ids),
        "canary_excluded_count": len(canary),
        "per_domain_counts": dict(sorted(per_domain_counts.items())),
        "triad_count": len(chosen_triads),
        "per_triad_count_histogram": {str(k): v for k, v in sorted(_histogram(per_triad_counts.values()).items())},
        "keys_by_id": {iid: keys[iid] for iid in ordered},
    }


def _histogram(values: Iterable[int]) -> dict[int, int]:
    out: dict[int, int] = defaultdict(int)
    for v in values:
        out[int(v)] += 1
    return dict(out)


def build_lc1_anchor_items(
    part_dir: Path,
    out_path: Path,
    *,
    manifest_path: Path | None = None,
    canary_config_path: Path | None = None,
) -> dict[str, Any]:
    """Write the 50 LC-1 anchors of the counted part in `part_dir` as the upload projection to `out_path` (with
    `<out_path>.sha256`), and the rule's manifest to `manifest_path` (default `<out_path>.manifest.json`). Returns the
    manifest. Never overwrites a file, never writes inside the part or under a `sealed_store/`."""
    part = Path(part_dir)
    out = Path(out_path)
    sha_out = out.with_name(out.name + ".sha256")
    mf_out = Path(manifest_path) if manifest_path is not None else out.with_name(out.name + ".manifest.json")
    for f in (out, sha_out, mf_out):
        why = _forbidden_out(f, part)
        if why is not None:
            raise refused_error(
                "LC1_ANCHORS_OUT_FORBIDDEN",
                f"The anchor files may not be written {why} ({f}).",
                "Write them to a separate inputs directory outside the part and outside every sealed_store/.",
            )
        if f.exists():
            raise refused_error(
                "LC1_ANCHORS_OUT_EXISTS",
                f"{f} already exists; the anchor builder never overwrites a file.",
                "Choose a new --out path or remove the old files first.",
            )
    if any(p.lower() == "sealed_store" for p in part.resolve().parts):
        raise refused_error(
            "LC1_ANCHORS_PART_INVALID",
            f"{part} is under a sealed_store directory; the builder reads only the part's public files.",
            "Pass the part directory (part_bakeoff_sealed), not the sealed store.",
        )
    sums = _read_checksums(part)
    part_json_bytes = _read_listed(part, "part.json", sums)
    items_bytes = _read_listed(part, "items.jsonl", sums)
    try:
        meta = json.loads(part_json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise refused_error("LC1_ANCHORS_PART_INVALID", "part.json does not parse.", "Restore the part.") from exc
    part_id = meta.get("part_id") if isinstance(meta, dict) else None
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(items_bytes.decode("utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except ValueError as exc:
            raise refused_error(
                "LC1_ANCHORS_PART_INVALID", f"items.jsonl line {lineno} is not a JSON row.", "Restore the part."
            ) from exc
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in PROJECTION_FIELDS):
            raise refused_error(
                "LC1_ANCHORS_PART_INVALID",
                f"items.jsonl line {lineno} lacks a string id, material or claim.",
                "Restore the part.",
            )
        # Keep only the fields the rule and the projection use; nothing else of the row goes further.
        rows.append({k: row.get(k) for k in (*LC1_SELECTION_FIELDS, "claim")})
    sel = select_lc1_anchors(rows, load_canary_item_ids(canary_config_path))
    by_id = {r["id"]: r for r in rows}
    lines = [json.dumps({k: by_id[iid][k] for k in PROJECTION_FIELDS}, ensure_ascii=False) for iid in sel["item_ids"]]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as fh:
        fh.write(data)
    projection_sha = hashlib.sha256(data).hexdigest()
    with sha_out.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{projection_sha}  {out.name}\n")
    manifest = {
        "schema_version": "1.0",
        "kind": "lc1_anchor_manifest",
        "rule_id": LC1_ANCHOR_RULE_ID,
        "seed": LC1_ANCHOR_SEED,
        "key": 'sha256(seed + "|" + item_id), hex',
        "selection": (
            f"per domain, sort the pool by (key, item_id), take the first {LC1_ANCHORS_PER_DOMAIN}, skipping a triad "
            "already taken; output sorted by (key, item_id)"
        ),
        "fields_read": ["id", "domain", "sha256(material)"],
        "triad_grouping": "items sharing one material (by its SHA-256)",
        "part_id": part_id,
        "part_json_sha256": sums["part.json"],
        "items_jsonl_sha256": sums["items.jsonl"],
        "part_item_count": sel["part_item_count"],
        "canary_excluded_count": sel["canary_excluded_count"],
        "pool_size": sel["pool_size"],
        "anchor_count": len(sel["item_ids"]),
        "domain_count": len(sel["per_domain_counts"]),
        "per_domain_counts": sel["per_domain_counts"],
        "triad_count": sel["triad_count"],
        "per_triad_count_histogram": sel["per_triad_count_histogram"],
        "sorted_item_ids_sha256": sorted_ids_sha256(sel["item_ids"]),
        "item_ids": sel["item_ids"],
        "projection_fields": list(PROJECTION_FIELDS),
        "projection_sha256": projection_sha,
        "projection_file": out.name,
        "low_margin_anchors": "after LC-1, the 25 lowest-margin of these 50 are recorded as the low-margin anchors",
    }
    with mf_out.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**manifest, "out": str(out), "sha256_file": str(sha_out), "manifest_file": str(mf_out)}
