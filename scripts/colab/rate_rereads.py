#!/usr/bin/env python3
"""Mid-run `colab usage` re-reads for `launch_judges.sh`'s cost-cap monitor.

`colab usage` reports the ACCOUNT's rate (every open assignment) and the number of active assignments. The window's
own rate is the account rate minus a neighbour baseline, which starts as the pre-open account rate. The launcher calls
`init` once, after the window rate is read stable at open, and `step` once per re-read (at most every 60 s):

- A read whose assignment count differs from the last settled count is never judged. When two consecutive reads at
  the new count agree within 0.01, the neighbour baseline becomes account - the HIGHEST stable rate of ours seen
  (`rebaselined`): the conservative direction, our apparent rate can only come out higher.
- Only a read at the settled count may raise our rate, and only once two such reads in a row agree within 0.01 (a
  stable rate). A stable rise reprices the remaining runtime at max(--max-rate, highest stable rate seen): it can
  shorten, never lengthen. A stable rate above --max-rate trips `RATE_ABOVE_MAX`.
- A read with no rate, no count or a count of 0, or one whose own rate at the settled count is below the window's
  rate floor (`GPU_RATE_FLOOR` on a GPU card), is a bad read: recorded (`bad_read_recorded`), never accepted. Three in a
  row trip `ASSIGNMENTS_UNREADABLE`.

Every read is appended to the records file (numbers and decisions only: the usage text itself is never stored).
`step` prints one line for the shell: `<decision> <max_runtime_s> <trip_class or ->`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

RATE_STABLE_TOLERANCE = 0.01
BAD_READS_TO_TRIP = 3
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


def pricing_rate(state: dict[str, Any]) -> float:
    return max(float(state.get("max_rate") or 0.0), float(state["highest_stable_rate"]))


def runtime_for(state: dict[str, Any]) -> int:
    return max(1, int(math.ceil((float(state["cap_units"]) * 3600.0) / pricing_rate(state) - 1e-12)))


def init_state(
    *,
    pre_account_rate: float,
    window_rate: float,
    open_count: int | None,
    max_rate: float | None,
    cap_units: float,
    max_runtime_s: int,
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
        "cap_units": float(cap_units),
        "max_runtime_s": int(max_runtime_s),
        "open_max_runtime_s": int(max_runtime_s),
        "reads": 0,
        "trip": None,
    }


def step(state: dict[str, Any], rate: float | None, count: int | None) -> dict[str, Any]:
    """Apply one re-read to `state` (in place) and return its record."""
    state["reads"] = int(state.get("reads", 0)) + 1
    rec: dict[str, Any] = {"read": state["reads"], "account_rate": _r(rate), "count": count}
    old_runtime = int(state["max_runtime_s"])
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
        # Our own rate below the card's floor is not our rate (a neighbour closing while the count lags, or a stale
        # read): never accepted, so it can never become the "last stable" rate a re-baseline would use.
        problem = "rate_below_floor"
    if problem is not None:
        state["bad_reads_in_a_row"] = int(state["bad_reads_in_a_row"]) + 1
        state["pending"] = None
        state["prev_our_rate"] = None
        rec["problem"] = problem
        rec["our_rate"] = None
        if problem == "rate_below_floor" and rate is not None:
            rec["our_rate_below_floor"] = _r(rate - float(state["neighbour_baseline"]))
        if state["bad_reads_in_a_row"] >= BAD_READS_TO_TRIP:
            rec["decision"] = "ASSIGNMENTS_UNREADABLE"
            state["trip"] = {
                "failure_class": "ASSIGNMENTS_UNREADABLE",
                "reason": (
                    f"ASSIGNMENTS_UNREADABLE: {state['bad_reads_in_a_row']} usage re-reads in a row with no rate, no "
                    f"assignment count or a count of 0 while the window is open (last: {problem})"
                ),
            }
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
                # The highest stable rate of ours seen, not the last one: our apparent rate can only come out higher.
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
                # Not judged: our rate is unknown until the count settles (account - old baseline, for the record).
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
                    state["max_runtime_s"] = min(old_runtime, runtime_for(state))
                    rec["decision"] = "rise_repriced"
                max_rate = state.get("max_rate")
                if max_rate is not None and our > float(max_rate) + _EPS:
                    rec["decision"] = "RATE_ABOVE_MAX"
                    state["trip"] = {
                        "failure_class": "RATE_ABOVE_MAX",
                        "reason": (
                            f"RATE_ABOVE_MAX: the window's rate rose mid-run to {our:.4f}/h (account {rate:.4f}/h minus "
                            f"neighbour baseline {float(state['neighbour_baseline']):.4f}/h, {count} active "
                            f"assignments, unchanged) above --max-rate {float(max_rate):g}/h"
                        ),
                    }
            else:
                rec["decision"] = "candidate"
            state["prev_our_rate"] = our
    rec["neighbour_baseline"] = _r(state["neighbour_baseline"])
    rec["settled_count"] = state.get("settled_count")
    rec["highest_stable_rate"] = _r(state["highest_stable_rate"])
    rec["pricing_rate"] = _r(pricing_rate(state))
    rec["max_runtime_s"] = int(state["max_runtime_s"])
    rec["runtime_shortened"] = int(state["max_runtime_s"]) < old_runtime
    rec["bad_reads_in_a_row"] = int(state["bad_reads_in_a_row"])
    return rec


def summary(state: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    trip = state.get("trip") or {}
    return {
        "reads": records,
        "read_count": len(records),
        "open_window_rate_per_hour": _r(state.get("open_window_rate")),
        "open_assignments_count": state.get("open_count"),
        "highest_stable_rate_per_hour": _r(state.get("highest_stable_rate")),
        "pricing_rate_per_hour": _r(pricing_rate(state)),
        "open_max_runtime_s": state.get("open_max_runtime_s"),
        "max_runtime_s": state.get("max_runtime_s"),
        "trip": trip.get("failure_class"),
        "trip_reason": trip.get("reason"),
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
    p_init.add_argument("--cap", required=True)
    p_init.add_argument("--max-runtime", required=True)
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
            cap_units=float(args.cap),
            max_runtime_s=int(args.max_runtime),
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
        trip = (st.get("trip") or {}).get("failure_class") or "-"
        print(f"{rec['decision']} {st['max_runtime_s']} {trip}")
        return 0
    # summary
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
