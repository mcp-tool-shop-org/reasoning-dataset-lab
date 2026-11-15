#!/usr/bin/env python3
"""Build a download receipt (`<owner>__<repo>.sha256.json`) from the Hugging Face API, without downloading weights.

For a judge whose weights have never been pulled to the rig (the D5c judges d3-flash and the NLI encoder), the
receipt is built from the Hub's own metadata at the pinned revision:

- `GET https://huggingface.co/api/models/<repo>/tree/<rev>?recursive=true&expand=true` lists every file;
- an LFS file (the weights, large assets) takes `lfs.oid` (its sha256) and `lfs.size` from that listing;
- a non-LFS file (configs, tokenizer text, remote code) is fetched from
  `https://huggingface.co/<repo>/resolve/<rev>/<path>`, then hashed and sized here.

The receipt has the same shape as the others in `scripts/colab/receipts/` (`repo`, `revision`, `files`, `total_bytes`,
`seconds`), plus `"source": "hf-api-metadata"` and `"seconds": null`: nothing was downloaded, so there is no rig
download time. An LFS oid is the Hub's claim, not bytes we hashed. It is enforced where it matters: on the VM,
`verify_snapshot_against_receipt` sizes and hashes every downloaded file against this receipt and refuses a mismatch
(`WEIGHT_SIZE_MISMATCH`, `BAKEOFF_JUDGE_SNAPSHOT_SHA_MISMATCH`) before any model is loaded.

Usage: python scripts/colab/build_metadata_receipt.py <repo_id> <revision> [--out-dir scripts/colab/receipts]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

METADATA_RECEIPT_SOURCE = "hf-api-metadata"
HF_ENDPOINT = "https://huggingface.co"
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "receipts"


class MetadataReceiptError(ValueError):
    """A tree listing or file that cannot make a sound receipt."""


def tree_url(repo_id: str, revision: str) -> str:
    return f"{HF_ENDPOINT}/api/models/{repo_id}/tree/{revision}?recursive=true&expand=true"


def resolve_url(repo_id: str, revision: str, path: str) -> str:
    return f"{HF_ENDPOINT}/{repo_id}/resolve/{revision}/{urllib.parse.quote(path)}"


def _http_get(url: str) -> bytes:  # pragma: no cover - network
    req = urllib.request.Request(url, headers={"User-Agent": "rdlab-metadata-receipt/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def build_metadata_receipt(
    repo_id: str,
    revision: str,
    *,
    fetch: Callable[[str], bytes] = _http_get,
) -> dict[str, Any]:
    """The receipt for `repo_id` at `revision`: LFS entries from the listing, non-LFS entries hashed after a fetch.
    Refuses (MetadataReceiptError) a revision that is not a 40-hex SHA, an empty or malformed listing, an LFS entry
    without a 64-hex oid or with a size that disagrees with the listing, and a non-LFS file whose fetched length is not
    the listed size."""
    if not _REVISION_RE.fullmatch(revision):
        raise MetadataReceiptError(f"revision must be a 40-hex commit SHA, got {revision!r}")
    listing = json.loads(fetch(tree_url(repo_id, revision)).decode("utf-8"))
    if not isinstance(listing, list) or not listing:
        raise MetadataReceiptError(f"empty or malformed tree listing for {repo_id}@{revision}")
    files: dict[str, dict[str, Any]] = {}
    for entry in listing:
        if not isinstance(entry, Mapping) or entry.get("type") != "file":
            continue
        path = str(entry.get("path") or "")
        if not path or path.startswith("/") or ".." in path.split("/"):
            raise MetadataReceiptError(f"unsafe path {path!r} in tree listing")
        lfs = entry.get("lfs")
        if isinstance(lfs, Mapping):
            oid, size = str(lfs.get("oid") or "").lower(), lfs.get("size")
            if not _SHA256_RE.fullmatch(oid) or not isinstance(size, int) or size != entry.get("size"):
                raise MetadataReceiptError(f"LFS entry {path!r} has no sha256 oid or a size that disagrees")
            files[path] = {"bytes": size, "sha256": oid}
        else:
            body = fetch(resolve_url(repo_id, revision, path))
            if len(body) != entry.get("size"):
                raise MetadataReceiptError(f"{path!r}: fetched {len(body)} bytes, listing says {entry.get('size')}")
            files[path] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    if not files:
        raise MetadataReceiptError(f"no files in tree listing for {repo_id}@{revision}")
    ordered = dict(sorted(files.items()))
    return {
        "files": ordered,
        "repo": repo_id,
        "revision": revision,
        "seconds": None,
        "source": METADATA_RECEIPT_SOURCE,
        "total_bytes": sum(int(v["bytes"]) for v in ordered.values()),
    }


def receipt_filename(repo_id: str) -> str:
    owner, repo = repo_id.split("/", 1)
    return f"{owner}__{repo}.sha256.json"


def write_receipt(doc: Mapping[str, Any], out_dir: Path) -> Path:
    out = Path(out_dir) / receipt_filename(str(doc["repo"]))
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper over the tested builder
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("repo_id")
    p.add_argument("revision")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)
    try:
        doc = build_metadata_receipt(args.repo_id, args.revision)
    except MetadataReceiptError as exc:
        print(f"METADATA_RECEIPT_REFUSED: {exc}", file=sys.stderr)
        return 2
    path = write_receipt(doc, args.out_dir)
    print(f"wrote {path} files={len(doc['files'])} total_bytes={doc['total_bytes']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
