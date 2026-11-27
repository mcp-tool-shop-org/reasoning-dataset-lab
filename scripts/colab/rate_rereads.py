#!/usr/bin/env python3
"""Mid-run `colab usage` re-read logger for `launch_judges.sh` (observational only).

`colab usage` reports the account rate and active assignment count. The launcher calls `init` once at open and `step`
on each re-read. Every read (including bad reads, floor under-reads, and rates above `--max-rate`) is logged and
recorded in `rate_rereads.jsonl`, never used to trip or shorten a running window.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any

RATE_STABLE_TOLERANCE = 0.01
_EPS = 1e-9

_RATE_RE = re.compile(r"Usage\s+rate\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*h(?:r|our)?\b", re.IGNORECASE)
_COUNT_RE = re.compile(r"Active\s+assignments\s*:\s*([0-9]+)\b", re.IGNORECASE)


def parse_usage(text: str) -> tuple[float | None, int | None]:
    """The account rate (units/h) and the active assignment count from `colab usage` output; None when absent."""
    rate_m = _RATE_RE.search(text or "")
    count_m = _COUNT_RE.search(text or "")
    return (
        float(rate_m.group(1)) if rate_m else None,
        int(count_m.group(1)) if count_m else None,
    )


def _r(v: float | None) -> float | None:
    return None if v is None else round(float(v), 4)


def init_state(
    *,
    pre_account_rate: float,
    window_rate: float,
    open_count: int | None,
    max_rate: float | None,
    floor: float = 0.0,
) -> dict[str, Any]:
    return {
        "floor": float(floor),
        "neighbour_baseline": float(pre_account_rate),
        "pre_account_rate": float(pre_account_rate),
        "our_last_stable_rate": float(window_rate),
        "highest_stable_rate": float(window_rate),
        "open_window_rate": float(window_rate),
        "settled_count": open_count,
        "open_count": open_count,
        "pending": None,
        "prev_our_rate": float(window_rate),
        "bad_reads_in_a_row": 0,
        "max_rate": max_rate,
        "reads": 0,
    }


def step(state: dict[str, Any], rate: float | None, count: int | None) -> dict[str, Any]:
    """Apply one re-read to `state` (in place) and return its log record (never trips)."""
    state["reads"] = int(state.get("reads", 0)) + 1
    rec: dict[str, Any] = {"read": state["reads"], "account_rate": _r(rate), "count": count}
    problem: str | None = None
    if rate is None:
        problem = "rate_missing"
    elif count is None:
        problem = "count_missing"
    elif count <= 0:
        problem = "count_zero"
    elif count == state.get("settled_count") and (
        rate - float(state["neighbour_baseline"]) < float(state.get("floor") or 0.0) - _EPS
    ):
        problem = "rate_below_floor"
    if problem is not None:
        state["bad_reads_in_a_row"] = int(state["bad_reads_in_a_row"]) + 1
        state["pending"] = None
        state["prev_our_rate"] = None
        rec["problem"] = problem
        rec["our_rate"] = None
        if problem == "rate_below_floor" and rate is not None:
            rec["our_rate_below_floor"] = _r(rate - float(state["neighbour_baseline"]))
        if problem == "rate_below_floor":
            rec["decision"] = "RATE_BELOW_FLOOR"
        elif state["bad_reads_in_a_row"] >= 3:
            rec["decision"] = "ASSIGNMENTS_UNREADABLE"
        else:
            rec["decision"] = "bad_read_recorded"
    else:
        state["bad_reads_in_a_row"] = 0
        settled = state.get("settled_count")
        if count != settled:
            state["prev_our_rate"] = None
            pend = state.get("pending")
            if pend is None or pend.get("count") != count:
                state["pending"] = {"count": count, "account_rate": rate}
                rec["decision"] = "count_changed_pending"
            elif abs(rate - float(pend["account_rate"])) <= RATE_STABLE_TOLERANCE + _EPS:
                state["neighbour_baseline"] = rate - float(state["highest_stable_rate"])
                state["settled_count"] = count
                state["pending"] = None
                state["prev_our_rate"] = float(state["highest_stable_rate"])
                rec["decision"] = "rebaselined"
            else:
                pend["account_rate"] = rate
                rec["decision"] = "settling"
            if rec["decision"] == "rebaselined":
                rec["our_rate"] = _r(rate - float(state["neighbour_baseline"]))
            else:
                rec["our_rate"] = None
                rec["our_rate_unsettled"] = _r(rate - float(state["neighbour_baseline"]))
        else:
            state["pending"] = None
            our = rate - float(state["neighbour_baseline"])
            rec["our_rate"] = _r(our)
            prev = state.get("prev_our_rate")
            if prev is not None and abs(our - float(prev)) <= RATE_STABLE_TOLERANCE + _EPS:
                state["our_last_stable_rate"] = our
                rec["decision"] = "stable"
                if our > float(state["highest_stable_rate"]) + _EPS:
                    state["highest_stable_rate"] = our
                    rec["decision"] = "rise_logged"
                max_rate = state.get("max_rate")
                if max_rate is not None and our > float(max_rate) + _EPS:
                    rec["decision"] = "RATE_ABOVE_MAX"
            else:
                rec["decision"] = "candidate"
            state["prev_our_rate"] = our
    rec["neighbour_baseline"] = _r(state["neighbour_baseline"])
    rec["settled_count"] = state.get("settled_count")
    rec["highest_stable_rate"] = _r(state["highest_stable_rate"])
    rec["bad_reads_in_a_row"] = int(state["bad_reads_in_a_row"])
    return rec


def summary(state: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "reads": records,
        "read_count": len(records),
        "open_window_rate_per_hour": _r(state.get("open_window_rate")),
        "open_assignments_count": state.get("open_count"),
        "highest_stable_rate_per_hour": _r(state.get("highest_stable_rate")),
        "trip": None,
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def _opt_float(raw: str) -> float | None:
    raw = (raw or "").strip()
    return float(raw) if raw else None


def _opt_int(raw: str) -> int | None:
    raw = (raw or "").strip()
    return int(raw) if raw else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_parse = sub.add_parser("parse", help="print '<rate> <count>' ('-' when absent) from usage text on stdin")
    p_parse.set_defaults(cmd="parse")
    p_init = sub.add_parser("init")
    p_init.add_argument("--state", required=True)
    p_init.add_argument("--pre", required=True)
    p_init.add_argument("--window", required=True)
    p_init.add_argument("--count", default="")
    p_init.add_argument("--max-rate", default="")
    p_init.add_argument("--floor", default="0")
    p_step = sub.add_parser("step", help="one re-read; usage text on stdin (empty when the read failed)")
    p_step.add_argument("--state", required=True)
    p_step.add_argument("--records", required=True)
    p_step.add_argument("--elapsed", default="")
    p_sum = sub.add_parser("summary")
    p_sum.add_argument("--state", required=True)
    p_sum.add_argument("--records", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "parse":
        rate, count = parse_usage(sys.stdin.read())
        print(f"{'-' if rate is None else f'{rate:.4f}'} {'-' if count is None else count}")
        return 0
    if args.cmd == "init":
        st = init_state(
            pre_account_rate=float(args.pre),
            window_rate=float(args.window),
            open_count=_opt_int(args.count),
            max_rate=_opt_float(args.max_rate),
            floor=float(args.floor or 0.0),
        )
        _save(Path(args.state), st)
        return 0
    if args.cmd == "step":
        state_path = Path(args.state)
        st = _load(state_path)
        rate, count = parse_usage(sys.stdin.read())
        rec = step(st, rate, count)
        rec["at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if (args.elapsed or "").strip():
            rec["elapsed_s"] = int(args.elapsed)
        with Path(args.records).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        _save(state_path, st)
        print(rec["decision"])
        return 0
    st = _load(Path(args.state))
    records: list[dict[str, Any]] = []
    rp = Path(args.records)
    if rp.is_file():
        for ln in rp.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                records.append(json.loads(ln))
    print(json.dumps(summary(st, records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
