#!/usr/bin/env python3
"""Run the boot gatekeeper locally, exactly as the Colab VM runs it, against a local HF home.

Same subprocess script (`vm_judges._BOOT_GATEKEEPER_SUBPROCESS_SCRIPT`, through `vm_judges.run_boot_gatekeeper_subprocess`),
same prompts (the goldens' `synthetic_prompts`), same judge (the goldens' `judge_id`), same seed and the real transport
(`TransformersJudgeTransport`, offline, weights verified against their receipt first). It prints each prompt's
`output_token_ids_sha256` and whether it matches the golden for `--card-type`, and writes a JSON receipt.

It needs a GPU: the goldens are per GPU card type, so a CPU run is refused (`LOCAL_GATEKEEPER_CPU_REFUSED`), and a run
that sees no NVIDIA device is refused before any model load (`LOCAL_GATEKEEPER_NO_GPU`). Run it only under a card
grant. The weights are read from `--hf-home` (`<hf_home>/hub/models--<owner>--<repo>/snapshots/<revision>`), never
downloaded: the child runs with `HF_HUB_OFFLINE=1`.

    python scripts/colab/local_boot_gatekeeper.py --hf-home <dir> --receipts-dir scripts/colab/receipts \\
        --card-type RTX5090 --out <receipt.json> [--python <venv python>] [--goldens <file>] [--window-mode time-one]

Exit codes: 0 every prompt matches its golden, or the card type has no golden (recorded as `no_golden`); 1 a hash
differs from its golden; 2 refused before the run (arguments, device, receipt, --out) or the gatekeeper subprocess failed.

`--out` must lie outside the git repository (so a run can never overwrite `scripts/colab/boot_gatekeeper_goldens.json`
or any other tracked file) and must not exist unless `--overwrite` is given (`LOCAL_GATEKEEPER_OUT_REFUSED`, checked
before anything else; nothing is written). The receipt and the printout carry the gatekeeper's per-phase seconds
(torch import, verify_ready re-hash, model load, generation 1 and 2, total) and the subprocess's measured elapsed time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for _p in (str(REPO_ROOT / "src"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vm_judges  # noqa: E402

LOCAL_GATEKEEPER_CPU_REFUSED = "LOCAL_GATEKEEPER_CPU_REFUSED"
LOCAL_GATEKEEPER_NO_GPU = "LOCAL_GATEKEEPER_NO_GPU"
LOCAL_GATEKEEPER_BAD_ARGS = "LOCAL_GATEKEEPER_BAD_ARGS"
LOCAL_GATEKEEPER_OUT_REFUSED = "LOCAL_GATEKEEPER_OUT_REFUSED"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="local_boot_gatekeeper.py",
        description="Run the VM's boot gatekeeper locally on a GPU against a local HF home and compare with the goldens.",
    )
    parser.add_argument("--hf-home", type=Path, required=True, help="HF home holding the gatekeeper judge's snapshot")
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        default=SCRIPT_DIR / "receipts",
        help="Directory with <owner>__<repo>.sha256.json receipts (default: scripts/colab/receipts)",
    )
    parser.add_argument("--card-type", required=True, help="Goldens card type to compare with (e.g. A100, L4)")
    parser.add_argument(
        "--out", type=Path, required=True, help="Where to write the JSON receipt (outside the repository)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing --out file")
    parser.add_argument("--goldens", type=Path, default=None, help="Goldens file (default: the repo's)")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python of the judge venv (torch + transformers per judge.lock.txt); default: this interpreter",
    )
    parser.add_argument("--device", default="cuda", help="Device for the transport (default cuda; cpu is refused)")
    parser.add_argument(
        "--window-mode",
        default="time-one",
        choices=list(vm_judges.BOOT_GATEKEEPER_REAL_MODES),
        help="The window mode the transport is built with (default time-one, as the failed window)",
    )
    parser.add_argument("--seed", type=int, default=vm_judges.DEFAULT_SCHEDULE_SEED)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(vm_judges.BOOT_GATEKEEPER_SUBPROCESS_TIMEOUT_SECONDS),
        help="Subprocess timeout (default: the VM's)",
    )
    return parser.parse_args(argv)


def check_out_path(out: Path, *, overwrite: bool = False, repo_root: Path = REPO_ROOT) -> Path:
    """Refuse an `--out` inside the git repository (symlinks resolved), or an existing file without `--overwrite`."""
    resolved = Path(out).expanduser().resolve()
    repo_resolved = Path(repo_root).resolve()
    if resolved == repo_resolved or repo_resolved in resolved.parents:
        raise vm_judges.RefusalError(
            LOCAL_GATEKEEPER_OUT_REFUSED,
            f"--out {out} resolves inside the repository ({repo_resolved}); write the receipt outside it, so no run "
            "can overwrite the goldens or any other repository file",
        )
    if resolved.exists() and not overwrite:
        raise vm_judges.RefusalError(
            LOCAL_GATEKEEPER_OUT_REFUSED,
            f"--out {out} already exists; pass --overwrite to replace it",
        )
    if resolved.exists() and not resolved.is_file():
        raise vm_judges.RefusalError(LOCAL_GATEKEEPER_OUT_REFUSED, f"--out {out} exists and is not a file")
    return resolved


def check_device(device: str, *, env: dict[str, str] | None = None) -> None:
    """Refuse a CPU device, and a CUDA device when no NVIDIA GPU is visible, without initialising CUDA."""
    src_env = os.environ if env is None else env
    norm = str(device or "").strip().lower()
    if not norm.startswith("cuda"):
        raise vm_judges.RefusalError(
            LOCAL_GATEKEEPER_CPU_REFUSED,
            f"device {device!r}: the goldens are per GPU card type, so the gatekeeper is not run on a CPU",
        )
    visible = src_env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in ("", "-1"):
        raise vm_judges.RefusalError(
            LOCAL_GATEKEEPER_NO_GPU,
            f"CUDA_VISIBLE_DEVICES={visible!r} hides every GPU",
        )
    if shutil.which("nvidia-smi", path=src_env.get("PATH")) is None:
        raise vm_judges.RefusalError(LOCAL_GATEKEEPER_NO_GPU, "nvidia-smi is not on PATH: no NVIDIA driver visible")


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    goldens_path = vm_judges.resolve_boot_gatekeeper_goldens_path(REPO_ROOT, args.goldens)
    judge_id = vm_judges.resolve_boot_gatekeeper_judge_id(REPO_ROOT, goldens_path)
    goldens_doc = json.loads(goldens_path.read_text(encoding="utf-8"))
    card_types = goldens_doc.get("card_types") if isinstance(goldens_doc.get("card_types"), dict) else {}
    card_entry = card_types.get(args.card_type)
    if not args.hf_home.is_dir():
        raise vm_judges.RefusalError(LOCAL_GATEKEEPER_BAD_ARGS, f"--hf-home {args.hf_home} is not a directory")
    if not args.receipts_dir.is_dir():
        raise vm_judges.RefusalError(LOCAL_GATEKEEPER_BAD_ARGS, f"--receipts-dir {args.receipts_dir} is not a directory")
    if not args.python.is_file():
        raise vm_judges.RefusalError(LOCAL_GATEKEEPER_BAD_ARGS, f"--python {args.python} is not a file")
    vm_judges.preflight_boot_gatekeeper_judge(judge_id, [judge_id], receipts_dir=args.receipts_dir)
    check_device(args.device)

    prompts = vm_judges.boot_gatekeeper_prompts(goldens_doc)
    timings: dict[str, Any] = {}
    t0 = time.monotonic()
    hashes = vm_judges.run_boot_gatekeeper_subprocess(
        venv_py=args.python,
        judge_id=judge_id,
        prompts=prompts,
        seed=args.seed,
        mode=args.window_mode,
        repo_root=REPO_ROOT,
        receipts_dir=args.receipts_dir,
        hf_home=args.hf_home,
        child_hf_home=args.hf_home,
        device=args.device,
        timeout_seconds=args.timeout_seconds,
        timings_out=timings,
    )
    seconds = round(time.monotonic() - t0, 3)
    per_prompt: list[dict[str, Any]] = []
    for sp in prompts:
        pid = str(sp["prompt_id"])
        golden = vm_judges.boot_gatekeeper_expected_sha(card_entry, pid)
        actual = hashes.get(pid)
        per_prompt.append(
            {
                "prompt_id": pid,
                "output_token_ids_sha256": actual,
                "golden": golden,
                "match": None if golden is None else bool(actual is not None and actual.lower() == golden),
                **(timings.get("generation_by_prompt") or {}).get(
                    pid,
                    {"output_token_count": None, "max_new_tokens": None},
                ),
            }
        )
    if any(p["match"] is None for p in per_prompt):
        status, rc = "no_golden", 0
    elif all(p["match"] for p in per_prompt):
        status, rc = "match", 0
    else:
        status, rc = "mismatch", 1
    return rc, {
        "status": status,
        "judge_id": judge_id,
        "card_type": args.card_type,
        "card_entry_status": card_entry.get("status") if isinstance(card_entry, dict) else None,
        "goldens_sha256": hashlib.sha256(goldens_path.read_bytes()).hexdigest(),
        "window_mode": args.window_mode,
        "device": args.device,
        "seed": args.seed,
        "seconds": seconds,
        "subprocess_timings": {k: v for k, v in timings.items() if k != "generation_by_prompt"},
        "prompts": per_prompt,
    }


def format_timings(timings: dict[str, Any]) -> str:
    """One line: each phase's seconds (`-` when the child did not report it), the measured elapsed and the timeout."""
    phases = timings.get("phase_seconds") or {}
    parts = [f"{k}={'-' if v is None else f'{float(v):.3f}'}" for k, v in phases.items()]
    return (
        "timings " + " ".join(parts)
        + f" subprocess_elapsed={float(timings.get('subprocess_elapsed_seconds') or 0.0):.3f}"
        + f" timeout={float(timings.get('timeout_seconds') or 0.0):g}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        out_path = check_out_path(args.out, overwrite=args.overwrite)
    except vm_judges.RefusalError as exc:
        sys.stderr.write(f"{exc.failure_class}: {exc.detail}\n")
        return 2
    receipt: dict[str, Any] = {
        "tool": "local_boot_gatekeeper",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "vm_judges_sha256": hashlib.sha256((SCRIPT_DIR / "vm_judges.py").read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        rc, result = run(args)
        receipt.update(result)
    except vm_judges.RefusalError as exc:
        rc = 2
        receipt.update(
            {
                "status": "refused",
                "failure_class": exc.failure_class,
                "detail": vm_judges.scrub_log_text(str(exc.detail)),
                "exception_line": getattr(exc, "exception_line", None),
            }
        )
    receipt["rc"] = rc
    receipt["ended_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if receipt["status"] == "refused":
        sys.stderr.write(f"{receipt['failure_class']}: {receipt['detail']}\n")
    else:
        for p in receipt["prompts"]:
            verdict = "no golden" if p["match"] is None else ("match" if p["match"] else "MISMATCH")
            print(
                f"{p['prompt_id']} {p['output_token_ids_sha256']} {verdict} "
                f"tokens={p.get('output_token_count')} max_new_tokens={p.get('max_new_tokens')}"
            )
        print(f"status={receipt['status']} judge_id={receipt['judge_id']} card_type={receipt['card_type']}")
        print(format_timings(receipt["subprocess_timings"]))
    print(f"receipt={args.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
