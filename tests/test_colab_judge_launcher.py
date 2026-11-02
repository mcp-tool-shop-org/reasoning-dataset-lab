from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_COLAB_DIR = REPO_ROOT / "scripts" / "colab"
LAUNCH_JUDGES_SH = SCRIPTS_COLAB_DIR / "launch_judges.sh"
VM_JUDGES_PY = SCRIPTS_COLAB_DIR / "vm_judges.py"
README_COLAB_MD = SCRIPTS_COLAB_DIR / "README.md"
JUDGE_LOCK_TXT = SCRIPTS_COLAB_DIR / "locks" / "judge.lock.txt"
D3_LOCK_TXT = SCRIPTS_COLAB_DIR / "locks" / "d3.lock.txt"

if str(SCRIPTS_COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_COLAB_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vm_judges  # noqa: E402
from reasoning_dataset_lab.errors import RdlabError  # noqa: E402
from reasoning_dataset_lab.pilot_b import bakeoff_judge_score  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (  # noqa: E402
    BAKEOFF_JUDGE_SPECS,
    D3_MINI_REVIEWED_REVISION,
)
from tests.fixtures import fake_colab  # noqa: E402

PYYAML_EXPECTED_SDIST_SHA256 = (
    "d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f"
)
EXPECTED_JUDGE_LOCK_SHA256 = (
    "608f56ab8eae6852718438366ff0ce74ef6aafb66db75faef034ae0bb5bdcf10"
)
EXPECTED_D3_LOCK_SHA256 = (
    "08801f9ac5e3cfb14ce819b051aeef388bea3e564e7b99bddc10d91537ab78e2"
)


def _to_wsl_path(path: Path) -> str:
    return path.resolve().as_posix()


def _make_synthetic_judge_snapshots_and_receipts(base_dir: Path) -> tuple[Path, Path]:
    """Create synthetic local HF snapshots and matching `<owner>__<repo>.sha256.json` receipts for all 5 judges."""
    receipts_dir = base_dir / "receipts"
    hf_home = base_dir / "hf_home"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    for judge_id, spec in sorted(BAKEOFF_JUDGE_SPECS.items()):
        owner, repo = spec.repo_id.split("/", 1)
        snap_dir = (
            hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
        )
        snap_dir.mkdir(parents=True, exist_ok=True)

        files_to_create: dict[str, bytes] = {
            "config.json": json.dumps(
                {"model_type": judge_id, "revision": spec.revision}
            ).encode("utf-8"),
            "model.safetensors": f"weights-for-{judge_id}-{spec.revision}".encode(
                "utf-8"
            ),
        }
        for rc_file in spec.remote_code_files:
            files_to_create[rc_file] = f"# remote code stub {rc_file}\n".encode("utf-8")

        files_manifest: dict[str, dict[str, Any]] = {}
        for rel_name, content in sorted(files_to_create.items()):
            fpath = snap_dir / rel_name
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_bytes(content)
            files_manifest[rel_name] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }

        receipt_doc = {
            "repo": spec.repo_id,
            "revision": spec.revision,
            "files": files_manifest,
        }
        receipt_path = receipts_dir / f"{owner}__{repo}.sha256.json"
        receipt_path.write_text(
            json.dumps(receipt_doc, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    return receipts_dir, hf_home


def _make_synthetic_public_items(base_dir: Path, count: int = 12) -> Path:
    """Create a synthetic public `items.jsonl` with `count` items and a `manifest.json` marking `pilot-40b`."""
    part_dir = base_dir / "part_public"
    part_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"part_id": "pilot-40b", "schema_version": "1.0"}
    (part_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    items_path = part_dir / "items.jsonl"
    rows = []
    for idx in range(count):
        rows.append(
            json.dumps(
                {
                    "id": f"item-{idx:04d}",
                    "material": (
                        f"1. Alpha premise {idx} establishes the first hop for entity {idx}.\n"
                        f"2. Beta premise {idx} links the first hop to outcome {idx}."
                    ),
                    "claim": (
                        f"Unique synthetic claim statement number {idx:04d} regarding alpha beta outcome {idx}."
                    ),
                }
            )
        )
    items_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return items_path


def _base_launcher_exports(
    env_paths: dict[str, Path],
    *,
    open_lock: Path,
    receipts_dir: Path,
    hf_home: Path,
    items_file: Path,
    judge_lock_sha: str = EXPECTED_JUDGE_LOCK_SHA256,
    d3_lock_sha: str = EXPECTED_D3_LOCK_SHA256,
) -> str:
    return (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_paths['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(open_lock)}'; "
        f"export RDLAB_RECEIPTS_DIR='{_to_wsl_path(receipts_dir)}'; "
        f"export HF_HOME='{_to_wsl_path(hf_home)}'; "
        f"export RDLAB_ITEMS_FILE='{_to_wsl_path(items_file)}'; "
        f"export RDLAB_JUDGE_LOCK_SHA256='{judge_lock_sha}'; "
        f"export RDLAB_D3_LOCK_SHA256='{d3_lock_sha}'; "
        f"export RDLAB_COLAB_IS_FAKE=1; "
        f"export VM_DRYRUN=1; "
    )


_STDIN_CMD_BOUNDARY_TOKENS = frozenset({"|", "||", "&&", ";", "&", ")"})


def _shlex_tokens(line: str) -> list[str]:
    import shlex

    unwrapped = re.sub(r'([A-Za-z_][A-Za-z0-9_]*=)"\$\((.*)\)"', r"\1 \2", line)
    unwrapped = re.sub(r"\$\(([^()]*)\)", r" \1 ", unwrapped)
    lex = shlex.shlex(unwrapped, posix=True, punctuation_chars="|&;<>()")
    lex.whitespace_split = True
    try:
        return list(lex)
    except ValueError:
        return unwrapped.split()


def check_colab_invocations_have_stdin_devnull(
    script_text: str,
    script_name: str = "launch_judges.sh",
) -> tuple[list[tuple[int, str, str]], list[str]]:
    """Parse a bash script and fail if any `colab` or `run_colab` invocation lacks `</dev/null`."""
    checked: list[tuple[int, str, str]] = []
    errors: list[str] = []

    joined = re.sub(r"\\\n", " ", script_text)
    for lineno, raw_line in enumerate(joined.splitlines(), start=1):
        stripped = raw_line.lstrip()
        if (
            stripped.startswith("#")
            or stripped.startswith("printf ")
            or stripped.startswith("echo ")
        ):
            continue
        code = raw_line.split("#", 1)[0]
        toks = _shlex_tokens(code)
        for i, tok in enumerate(toks):
            if tok not in ("colab", "run_colab"):
                continue
            if i + 1 >= len(toks):
                continue
            nxt = toks[i + 1]
            if nxt == "()" or (i > 0 and toks[i - 1] == "function"):
                continue
            is_subcmd = bool(re.fullmatch(r"[a-z][a-z0-9-]*", nxt))
            is_passthrough = nxt in ("$@", '"$@"')
            if not (is_subcmd or is_passthrough):
                continue
            rest = toks[i + 2 :]
            cmd_slice: list[str] = []
            for r in rest:
                if r in _STDIN_CMD_BOUNDARY_TOKENS:
                    break
                cmd_slice.append(r)
            has_devnull = False
            for k, r in enumerate(cmd_slice):
                if r in ("</dev/null", "0</dev/null"):
                    has_devnull = True
                    break
                if (
                    r in ("<", "0<")
                    and k + 1 < len(cmd_slice)
                    and cmd_slice[k + 1] == "/dev/null"
                ):
                    has_devnull = True
                    break
            checked.append((lineno, tok, nxt))
            if not has_devnull:
                errors.append(
                    f"{script_name}:{lineno}: {tok} {nxt}: missing </dev/null stdin redirect"
                )
    return checked, errors


def test_locks_pin_pyyaml_sdist_and_wheels_only_for_all_others(tmp_path: Path) -> None:
    assert JUDGE_LOCK_TXT.is_file()
    assert D3_LOCK_TXT.is_file()
    assert vm_judges.EXPECTED_JUDGE_LOCK_SHA256 == EXPECTED_JUDGE_LOCK_SHA256
    assert vm_judges.EXPECTED_D3_LOCK_SHA256 == EXPECTED_D3_LOCK_SHA256
    assert vm_judges.PYYAML_SDIST_SHA256 == PYYAML_EXPECTED_SDIST_SHA256

    for lock_path, expected_sha in (
        (JUDGE_LOCK_TXT, EXPECTED_JUDGE_LOCK_SHA256),
        (D3_LOCK_TXT, EXPECTED_D3_LOCK_SHA256),
    ):
        report = vm_judges.verify_lock_sdist_and_hashes(lock_path)
        assert report["lock_sha256"] == expected_sha
        assert report["pyyaml_version"] == "6.0.3"
        assert report["pyyaml_hashes"] == [PYYAML_EXPECTED_SDIST_SHA256]
        assert report["pyyaml_distributions"] == ["pyyaml-6.0.3.tar.gz"]
        assert report["non_pyyaml_sdists"] == []
        assert report["package_count"] >= 50

    # Negative check: any non-pyyaml sdist or wrong pyyaml hash must refuse
    bad_lock = tmp_path / "bad.lock.txt"
    bad_lock.write_text(
        JUDGE_LOCK_TXT.read_text(encoding="utf-8").replace(
            "einops-0.8.2-py3-none-any.whl"
            if "einops" in JUDGE_LOCK_TXT.read_text(encoding="utf-8")
            else "click-8.5.0-py3-none-any.whl",
            "click-8.5.0.tar.gz",
        ),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.verify_lock_sdist_and_hashes(bad_lock)
    assert exc_info.value.failure_class == "LOCK_NON_WHEEL_FORBIDDEN"


def test_dry_mode_matches_live_mode_command_list_against_fake(tmp_path: Path) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    out_dir = tmp_path / "out_match"
    open_lock = tmp_path / "open.lock"
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)
    wsl_out = _to_wsl_path(out_dir)

    exports = _base_launcher_exports(
        env_paths,
        open_lock=open_lock,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )

    # 1. Compare --stub --dry vs --stub live
    proc_dry = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{wsl_launch}' --stub --dry --out-dir '{wsl_out}' --run-id match-stub-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_dry.returncode == 0, f"stderr={proc_dry.stderr}"
    dry_lines = [ln.strip() for ln in proc_dry.stdout.splitlines() if ln.strip()]
    assert not env_paths["calls_log"].exists()

    proc_live = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{wsl_launch}' --stub --out-dir '{wsl_out}' --run-id match-stub-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_live.returncode == 0, f"stdout={proc_live.stdout}\nstderr={proc_live.stderr}"
    calls = [
        json.loads(ln)
        for ln in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    live_lines = [" ".join(["colab", c["subcmd"], *c["args"]]) for c in calls]
    assert dry_lines == live_lines

    # 2. Compare --session-smoke --dry vs --session-smoke live
    env_smoke = fake_colab.setup_fake_colab_environment(tmp_path / "smoke")
    out_smoke = tmp_path / "out_smoke"
    wsl_out_smoke = _to_wsl_path(out_smoke)
    exports_smoke = _base_launcher_exports(
        env_smoke,
        open_lock=tmp_path / "smoke.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_smoke_dry = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_smoke} bash '{wsl_launch}' --session-smoke --dry --out-dir '{wsl_out_smoke}' --run-id smoke-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_smoke_dry.returncode == 0
    smoke_dry_lines = [
        ln.strip() for ln in proc_smoke_dry.stdout.splitlines() if ln.strip()
    ]

    proc_smoke_live = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_smoke} bash '{wsl_launch}' --session-smoke --out-dir '{wsl_out_smoke}' --run-id smoke-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_smoke_live.returncode == 0
    smoke_calls = [
        json.loads(ln)
        for ln in env_smoke["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    smoke_live_lines = [
        " ".join(["colab", c["subcmd"], *c["args"]]) for c in smoke_calls
    ]
    assert smoke_dry_lines == smoke_live_lines


def test_static_guard_all_colab_calls_have_stdin_devnull() -> None:
    script_text = LAUNCH_JUDGES_SH.read_text(encoding="utf-8")
    checked, errors = check_colab_invocations_have_stdin_devnull(script_text)
    assert errors == [], f"Missing </dev/null on colab calls: {errors}"
    assert len(checked) >= 12
    targets = {nxt for _, _, nxt in checked}
    assert {"sessions", "new", "exec", "upload", "download", "stop", "$@"}.issubset(
        targets
    )

    negative_fixture = "\n".join(
        [
            'out="$(colab "$@" 2>"$err_file")" || rc=$?',
            'PRE_SESSIONS="$(run_colab sessions || true)"',
            'run_colab new -s "$SESSION_NAME" >/dev/null',
            'timeout "$GATE_TIMEOUT_SECONDS" colab exec -s "$S" --env RDLAB_STEP=gate -f "$VM_SCRIPT" --timeout="$GATE_TIMEOUT_SECONDS" 2>&1',
            'timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$REPO_TAR" "/content/repo_bundle.tar.gz" 2>&1',
            'colab download -s "$1" "/content/result.tar.gz" "$2/result.tar.gz" >/dev/null 2>&1 || true',
            'colab stop -s "$S" >/dev/null 2>&1 || true',
            'colab stop -s "$S" </dev/null >/dev/null 2>&1 || true',
        ]
    )
    neg_checked, neg_errors = check_colab_invocations_have_stdin_devnull(
        negative_fixture, script_name="negative_fixture.sh"
    )
    assert len(neg_checked) == 8
    assert len(neg_errors) == 7


def test_name_reuse_refused_and_session_stopped_in_trap_on_success_failure_and_sigterm(
    tmp_path: Path,
) -> None:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)

    # 1. Name reuse refused
    env_reuse = fake_colab.setup_fake_colab_environment(
        tmp_path / "reuse",
        neighbour_sessions=[
            {"session_id": "sess-neighbour-keep", "gpu": "T4"},
            {"session_id": "rdlab-reuse-01", "gpu": "CPU"},
        ],
    )
    exports_reuse = _base_launcher_exports(
        env_reuse,
        open_lock=tmp_path / "reuse.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_reuse = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_reuse} bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(tmp_path / 'out_reuse')}' --run-id reuse-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_reuse.returncode != 0
    assert "SESSION_NAME_IN_USE" in proc_reuse.stderr
    assert not env_reuse["stopped_file"].exists()

    # 2. Session stopped in trap on success
    env_ok = fake_colab.setup_fake_colab_environment(tmp_path / "ok")
    exports_ok = _base_launcher_exports(
        env_ok,
        open_lock=tmp_path / "ok.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_ok = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_ok} bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(tmp_path / 'out_ok')}' --run-id trap-ok-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_ok.returncode == 0
    stopped_ok = [
        json.loads(ln)["session_id"]
        for ln in env_ok["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert stopped_ok == ["rdlab-trap-ok-01"]
    assert not (env_ok["colab_locks_dir"] / "rdlab-trap-ok-01.lock").exists()

    # 3. Session stopped in trap on failure
    env_fail = fake_colab.setup_fake_colab_environment(tmp_path / "fail")
    exports_fail = _base_launcher_exports(
        env_fail,
        open_lock=tmp_path / "fail.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_fail = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_fail} export FAKE_COLAB_OVERRIDE_STATUS_FAILED=1; "
            f"bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(tmp_path / 'out_fail')}' --run-id trap-fail-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_fail.returncode != 0
    stopped_fail = [
        json.loads(ln)["session_id"]
        for ln in env_fail["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert stopped_fail == ["rdlab-trap-fail-01"]
    assert not (env_fail["colab_locks_dir"] / "rdlab-trap-fail-01.lock").exists()

    # 4. Session stopped in trap on SIGTERM
    env_term = fake_colab.setup_fake_colab_environment(tmp_path / "term")
    exports_term = _base_launcher_exports(
        env_term,
        open_lock=tmp_path / "term.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    out_term = tmp_path / "out_term"
    term_cmd = (
        f"{exports_term} export FAKE_COLAB_EXEC_SLEEP=30; "
        f"bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(out_term)}' --run-id trap-term-01 & "
        "lpid=$!; "
        f"for _ in $(seq 1 100); do [[ -f '{_to_wsl_path(env_term['colab_locks_dir'] / 'rdlab-trap-term-01.lock')}' ]] && break; sleep 0.05; done; "
        "sleep 0.2; kill -TERM $lpid; wait $lpid"
    )
    proc_term = subprocess.run(
        ["bash", "-c", term_cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_term.returncode != 0
    stopped_term = [
        json.loads(ln)["session_id"]
        for ln in env_term["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert "rdlab-trap-term-01" in stopped_term
    assert not (env_term["colab_locks_dir"] / "rdlab-trap-term-01.lock").exists()


def test_status_json_overrides_exec_rc_zero(tmp_path: Path) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    out_dir = tmp_path / "out_status_override"
    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "open.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )

    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} export FAKE_COLAB_OVERRIDE_STATUS_FAILED=1; export FAKE_COLAB_FORCE_EXEC_RC_ZERO=1; "
            f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id status-override-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 2
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["outcome"] == "failed"
    assert "FORCED_STATUS_OVERRIDE" in outcome["why"]


def test_lock_sha_missing_or_mismatch_and_weights_sha_mismatch_extra_or_missing_file_refuse(
    tmp_path: Path,
) -> None:
    # 1. Missing RDLAB_D3_LOCK_SHA256 or RDLAB_JUDGE_LOCK_SHA256 refuses
    with pytest.raises(vm_judges.RefusalError) as exc_missing_d3:
        vm_judges.verify_lock_digests(
            JUDGE_LOCK_TXT,
            D3_LOCK_TXT,
            env_judge_sha256=EXPECTED_JUDGE_LOCK_SHA256,
            env_d3_sha256="",
        )
    assert exc_missing_d3.value.failure_class == "LOCK_SHA256_MISSING"

    with pytest.raises(vm_judges.RefusalError) as exc_missing_judge:
        vm_judges.verify_lock_digests(
            JUDGE_LOCK_TXT,
            D3_LOCK_TXT,
            env_judge_sha256=None,
            env_d3_sha256=EXPECTED_D3_LOCK_SHA256,
        )
    assert exc_missing_judge.value.failure_class == "LOCK_SHA256_MISSING"

    # 2. Mismatched RDLAB_D3_LOCK_SHA256 or RDLAB_JUDGE_LOCK_SHA256 refuses
    with pytest.raises(vm_judges.RefusalError) as exc_mismatch_d3:
        vm_judges.verify_lock_digests(
            JUDGE_LOCK_TXT,
            D3_LOCK_TXT,
            env_judge_sha256=EXPECTED_JUDGE_LOCK_SHA256,
            env_d3_sha256="0" * 64,
        )
    assert exc_mismatch_d3.value.failure_class == "LOCK_SHA256_MISMATCH"

    with pytest.raises(vm_judges.RefusalError) as exc_mismatch_judge:
        vm_judges.verify_lock_digests(
            JUDGE_LOCK_TXT,
            D3_LOCK_TXT,
            env_judge_sha256="f" * 64,
            env_d3_sha256=EXPECTED_D3_LOCK_SHA256,
        )
    assert exc_mismatch_judge.value.failure_class == "LOCK_SHA256_MISMATCH"

    # Exact match passes
    verified_locks = vm_judges.verify_lock_digests(
        JUDGE_LOCK_TXT,
        D3_LOCK_TXT,
        env_judge_sha256=EXPECTED_JUDGE_LOCK_SHA256,
        env_d3_sha256=EXPECTED_D3_LOCK_SHA256,
    )
    assert verified_locks["judge_lock_sha256"] == EXPECTED_JUDGE_LOCK_SHA256
    assert verified_locks["d3_lock_sha256"] == EXPECTED_D3_LOCK_SHA256

    # 3. Weights verification: clean pass, then sha mismatch, extra file, and missing file refuse whole run
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "weights"
    )
    verified_snaps = vm_judges.verify_all_judge_snapshots(receipts_dir, hf_home)
    assert len(verified_snaps) == 5

    d3_spec = BAKEOFF_JUDGE_SPECS["d3-mini"]
    owner, repo = d3_spec.repo_id.split("/", 1)
    d3_snap = (
        hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / d3_spec.revision
    )
    orig_weights = (d3_snap / "model.safetensors").read_bytes()

    # (a) SHA mismatch
    (d3_snap / "model.safetensors").write_bytes(orig_weights + b"-tampered")
    with pytest.raises((vm_judges.RefusalError, RdlabError)):
        vm_judges.verify_all_judge_snapshots(receipts_dir, hf_home)
    (d3_snap / "model.safetensors").write_bytes(orig_weights)

    # (b) Extra file
    extra_file = d3_snap / "unlisted_extra.bin"
    extra_file.write_bytes(b"extra")
    with pytest.raises((vm_judges.RefusalError, RdlabError)):
        vm_judges.verify_all_judge_snapshots(receipts_dir, hf_home)
    extra_file.unlink()

    # (c) Missing file
    (d3_snap / "model.safetensors").unlink()
    with pytest.raises((vm_judges.RefusalError, RdlabError)):
        vm_judges.verify_all_judge_snapshots(receipts_dir, hf_home)


def test_key_path_upload_refuses_and_deleted_items_receipt_recorded(
    tmp_path: Path,
) -> None:
    # 1. Key or sealed-store upload paths refuse in both Python helper and launch_judges.sh
    for bad_path in (
        "sealed_store/items.jsonl",
        "/tmp/run/sealed_store/keys.sealed.jsonl",
        "/tmp/run/keys/part_keys.json",
        "/tmp/run/keys.jsonl",
        "/tmp/run/items.sealed.jsonl",
    ):
        with pytest.raises(vm_judges.RefusalError) as exc_up:
            vm_judges.check_upload_path_allowed(bad_path)
        assert exc_up.value.failure_class == "FORBIDDEN_KEY_OR_SEALED_UPLOAD"

    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "key_up")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "key_up"
    )
    sealed_dir = tmp_path / "key_up" / "sealed_store"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    bad_items_file = sealed_dir / "keys.sealed.jsonl"
    bad_items_file.write_text('{"id":"item-0000"}\n', encoding="utf-8", newline="\n")

    exports_bad = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "key_up.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=bad_items_file,
    )
    proc_bad = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_bad} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(tmp_path / 'out_key_up')}' --run-id key-upload-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_bad.returncode != 0
    assert "FORBIDDEN_KEY_OR_SEALED_UPLOAD" in proc_bad.stderr
    assert not env_paths["calls_log"].exists()

    # 2. Valid public item file is uploaded, then deleted on the VM, and receipt records `test ! -e` per file
    env_ok = fake_colab.setup_fake_colab_environment(tmp_path / "del_ok")
    items_file = _make_synthetic_public_items(tmp_path / "del_ok")
    out_ok = tmp_path / "out_del_ok"
    exports_ok = _base_launcher_exports(
        env_ok,
        open_lock=tmp_path / "del_ok.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_ok = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_ok} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_ok)}' --run-id del-items-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_ok.returncode == 0, f"stderr={proc_ok.stderr}"
    receipt = json.loads((out_ok / "vm_receipt.json").read_text(encoding="utf-8"))
    del_receipt = receipt["deleted_items_receipt"]
    assert del_receipt["all_absent"] is True
    assert len(del_receipt["files"]) >= 1
    assert all(
        f["absent"] is True and f["test_not_e_rc"] == 0 for f in del_receipt["files"]
    )
    vm_uploaded_item = (
        env_ok["state_dir"] / "vm_fs" / "rdlab_items" / "items.jsonl"
    )
    assert not vm_uploaded_item.exists()


def _make_synthetic_sealed_keys(base_dir: Path, count: int = 12) -> Path:
    """Create a synthetic `sealed_store/keys.sealed.jsonl` file matching `_make_synthetic_public_items`."""
    sealed_dir = base_dir / "sealed_store"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    keys_path = sealed_dir / "keys.sealed.jsonl"
    rows = [
        json.dumps(
            {
                "id": f"item-{idx:04d}",
                "triad_id": f"T10-D-synth-anchored-{idx // 3:03d}",
                "triad_handle": f"triad-{idx // 3:016x}",
                "key": "yes" if idx % 3 == 0 else "no",
                "class": "yes" if idx % 3 == 0 else "no",
                "key_source": "construction",
                "value_set_id": f"synth/vs-{idx // 3:03d}",
                "closure": "closed",
                "split": "dev" if idx < 3 else "test",
                "domain": "synth",
            }
        )
        for idx in range(count)
    ]
    keys_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return keys_path


def _instrument_key_reader(
    monkeypatch: pytest.MonkeyPatch,
    state_dir: Path,
) -> Any:
    """Instrument `bakeoff_judge_score.load_sealed_keys` so it records its own access timestamp and sequence number."""
    orig_load_sealed_keys = bakeoff_judge_score.load_sealed_keys

    def _instrumented_load_sealed_keys(keys_path: Path) -> dict[str, Any]:
        seq = len(fake_colab.read_file_access_log(state_dir))
        access_ts = time.time()
        fake_colab.record_file_access(
            state_dir,
            "read_keys",
            keys_path,
            detail=f"seq={seq};ts={access_ts:.6f}",
        )
        return orig_load_sealed_keys(keys_path)

    monkeypatch.setattr(
        bakeoff_judge_score, "load_sealed_keys", _instrumented_load_sealed_keys
    )
    return _instrumented_load_sealed_keys


def assert_outputs_closed_before_key_access(
    state_dir: Path,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assert that `write_outputs_closed` occurred strictly before the first key access in `file_access.jsonl`."""
    raw_events = fake_colab.read_file_access_log(state_dir)
    annotated = [{**ev, "seq": idx} for idx, ev in enumerate(raw_events)]

    closed_events = [
        ev for ev in annotated if ev.get("event") == "write_outputs_closed"
    ]
    key_events = [
        ev
        for ev in annotated
        if ev.get("event") == "read_keys"
        or "sealed_store" in str(ev.get("path", ""))
        or "keys.sealed" in str(ev.get("path", ""))
    ]
    assert closed_events, "No write_outputs_closed event recorded in file_access.jsonl"
    assert key_events, "No key access event recorded in file_access.jsonl"

    closed_ev = closed_events[0]
    first_key_ev = key_events[0]
    closed_seq = int(closed_ev["seq"])
    first_key_seq = int(first_key_ev["seq"])
    closed_ts = float(closed_ev["ts"])
    first_key_ts = float(first_key_ev["ts"])

    assert closed_seq < first_key_seq and closed_ts <= first_key_ts, (
        f"outputs-closed write (seq={closed_seq}, ts={closed_ts:.6f}) must precede "
        f"first key access (seq={first_key_seq}, ts={first_key_ts:.6f}, path={first_key_ev.get('path')})"
    )

    for ev in annotated[:first_key_seq]:
        assert "sealed_store" not in str(ev.get("path", ""))
        assert "keys.sealed" not in str(ev.get("path", ""))

    closed_path = out_dir / "outputs-closed.json"
    assert closed_path.is_file()
    closed_doc = json.loads(closed_path.read_text(encoding="utf-8"))
    assert closed_doc["outputs_closed_sha256"] == closed_ev["detail"]
    return closed_ev, first_key_ev


def test_outputs_closed_written_before_any_key_path_touched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    keys_file = _make_synthetic_sealed_keys(tmp_path)
    out_dir = tmp_path / "out_closed_order"
    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "open.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )

    _instrument_key_reader(monkeypatch, env_paths["state_dir"])

    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id closed-order-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"

    closed_path = out_dir / "outputs-closed.json"
    assert closed_path.is_file()
    closed_doc = json.loads(closed_path.read_text(encoding="utf-8"))
    assert "outputs_closed_sha256" in closed_doc
    assert "schedule.json" in closed_doc["files"]
    assert "generations.jsonl" in closed_doc["files"]

    # Run the instrumented key-reading step after the launcher closes outputs
    loaded_keys = bakeoff_judge_score.load_sealed_keys(keys_file)
    assert len(loaded_keys) == 12

    closed_ev, first_key_ev = assert_outputs_closed_before_key_access(
        env_paths["state_dir"], out_dir
    )
    assert closed_ev["seq"] < first_key_ev["seq"]
    assert float(closed_ev["ts"]) <= float(first_key_ev["ts"])
    assert f"seq={first_key_ev['seq']};" in str(first_key_ev["detail"])


def test_outputs_closed_before_keys_negative_swapped_order_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "swapped_state"
    out_dir = tmp_path / "out_swapped"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "schedule.json").write_text('{"seed": 1}\n', encoding="utf-8", newline="\n")
    (out_dir / "generations.jsonl").write_text('{"judge_id": "mu-judge"}\n', encoding="utf-8", newline="\n")
    keys_file = _make_synthetic_sealed_keys(tmp_path)

    _instrument_key_reader(monkeypatch, state_dir)

    # Swapped order: read keys BEFORE writing outputs-closed.json
    loaded_keys = bakeoff_judge_score.load_sealed_keys(keys_file)
    assert len(loaded_keys) == 12
    vm_judges.write_outputs_closed(out_dir, state_dir=state_dir)

    with pytest.raises(
        AssertionError,
        match=r"outputs-closed write \(seq=1, .*\) must precede first key access \(seq=0,",
    ):
        assert_outputs_closed_before_key_access(state_dir, out_dir)


def test_log_guard_trips_on_planted_claim_and_planted_key_filename(
    tmp_path: Path,
) -> None:
    items_file = _make_synthetic_public_items(tmp_path, count=4)
    claim_hashes = vm_judges.collect_claim_ngram_sha256s(items_file)
    assert len(claim_hashes) >= 4

    # 1. Clean log passes
    clean_log = tmp_path / "clean.log"
    clean_log.write_text(
        "2026-11-02T00:00:00Z WINDOW_OK session=rdlab-test\n",
        encoding="utf-8",
        newline="\n",
    )
    vm_judges.verify_rig_logs_and_outputs_clean(
        [clean_log], claim_ngram_sha256s=claim_hashes
    )

    # 2. Planted claim text trips log guard without printing the claim text in the exception message
    planted_claim_log = tmp_path / "planted_claim.log"
    leaked_phrase = "Unique synthetic claim statement number 0002 regarding alpha beta outcome 2."
    planted_claim_log.write_text(
        f"debug output: {leaked_phrase}\n", encoding="utf-8", newline="\n"
    )
    with pytest.raises(vm_judges.RefusalError) as exc_claim:
        vm_judges.verify_rig_logs_and_outputs_clean(
            [planted_claim_log], claim_ngram_sha256s=claim_hashes
        )
    assert exc_claim.value.failure_class == "LOG_GUARD_CLAIM_LEAK"
    assert leaked_phrase not in str(exc_claim.value)
    assert "Unique synthetic claim" not in str(exc_claim.value)

    # 3. Planted key-file name (`sealed_store` and `keys.sealed`) trips log guard
    for marker in ("sealed_store", "keys.sealed"):
        planted_key_log = tmp_path / f"planted_{marker}.log"
        planted_key_log.write_text(
            f"touched path /tmp/{marker}/file\n", encoding="utf-8", newline="\n"
        )
        with pytest.raises(vm_judges.RefusalError) as exc_key:
            vm_judges.verify_rig_logs_and_outputs_clean(
                [planted_key_log], claim_ngram_sha256s=claim_hashes
            )
        assert exc_key.value.failure_class == "LOG_GUARD_KEY_FILENAME_LEAK"

    # 4. End-to-end launcher test: planted leak in VM log marks outcome `failed` and excludes files from receipt
    env_leak = fake_colab.setup_fake_colab_environment(tmp_path / "e2e_leak")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "e2e_leak"
    )
    out_leak = tmp_path / "out_e2e_leak"
    exports_leak = _base_launcher_exports(
        env_leak,
        open_lock=tmp_path / "e2e_leak.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_leak = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_leak} export RDLAB_TEST_PLANT_LOG_LEAK='keys.sealed'; "
            f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_leak)}' --run-id leak-guard-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_leak.returncode != 0
    outcome_leak = json.loads(
        (out_leak / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_leak["outcome"] == "failed"
    assert "log guard" in outcome_leak["why"].lower()
    assert not (out_leak / "outputs-closed.json").exists()


def test_colab_message_line_is_not_a_session(tmp_path: Path) -> None:
    empty_line = fake_colab.EMPTY_SESSIONS_LINE
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)

    # 1. When `colab sessions` prints `[colab] No active sessions found on server.` before `new`,
    #    it is treated as an empty list (not a session), and when it prints that same line after `new`,
    #    mid-session verification fails because no session row matched `[rdlab-...]`.
    env_a = fake_colab.setup_fake_colab_environment(
        tmp_path / "msg_a", neighbour_sessions=[]
    )
    out_a = tmp_path / "out_msg_a"
    cmd_a = (
        f"export PATH='{_to_wsl_path(env_a['fake_bin'])}:{_to_wsl_path(env_a['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_a['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_a['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'msg_a.lock')}'; "
        f"export FAKE_COLAB_SESSIONS_OVERRIDE='{empty_line}'; "
        f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_a)}' --run-id msg-only-01"
    )
    proc_a = subprocess.run(
        ["bash", "-c", cmd_a],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_a.returncode != 0
    assert "SESSION_NAME_IN_USE" not in proc_a.stderr
    outcome_a = json.loads((out_a / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome_a["outcome"] == "failed"
    assert outcome_a["format_verified_on_real_cli"] is False

    # 2. Even `--run-id colab` becomes `rdlab-colab`, so `[colab] ...` never matches `[$SESSION_NAME]`
    env_b = fake_colab.setup_fake_colab_environment(
        tmp_path / "msg_b", neighbour_sessions=[]
    )
    out_b = tmp_path / "out_msg_b"
    cmd_b = (
        f"export PATH='{_to_wsl_path(env_b['fake_bin'])}:{_to_wsl_path(env_b['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_b['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_b['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'msg_b.lock')}'; "
        f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_b)}' --run-id colab"
    )
    proc_b = subprocess.run(
        ["bash", "-c", cmd_b],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_b.returncode == 0, f"stderr={proc_b.stderr}"
    outcome_b = json.loads((out_b / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome_b["outcome"] == "ok"


def test_outcome_format_verified_flag_is_false_under_fake(tmp_path: Path) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path)
    out_dir = tmp_path / "out_fake_flag"
    cmd = (
        f"export PATH='{_to_wsl_path(env_paths['fake_bin'])}:{_to_wsl_path(env_paths['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_paths['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'flag.lock')}'; "
        f"export RDLAB_COLAB_IS_FAKE=1; "
        f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --session-smoke --out-dir '{_to_wsl_path(out_dir)}' --run-id fake-flag-01"
    )
    proc = subprocess.run(
        ["bash", "-c", cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["format_verified_on_real_cli"] is False
    assert (
        outcome["sessions_output_shape"]["mid_sessions"]["matches_session_row_shape"]
        is True
    )
    assert (
        outcome["sessions_output_shape"]["format_verified_on_real_cli"] is False
    )


def test_no_test_reads_tty_launcher_calls_with_stdin_closed(tmp_path: Path) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    out_dir = tmp_path / "out_no_tty"
    caller_stdin_file = tmp_path / "caller_stdin.txt"
    caller_stdin_file.write_text("UNCONSUMED_CALLER_STDIN\n", encoding="utf-8", newline="\n")

    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "no_tty.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    cmd = (
        f"{exports} export FAKE_COLAB_READ_STDIN=1; "
        f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id no-tty-01 < /dev/null"
    )
    proc = subprocess.run(
        ["bash", "-c", cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    calls = [
        json.loads(ln)
        for ln in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(calls) >= 10
    assert all(c["stdin_is_devnull"] is True for c in calls)


def test_fake_colab_contract_and_time_one_mode_summary_and_mass_readout(
    tmp_path: Path,
) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path)
    wsl_fake_colab = _to_wsl_path(env_paths["fake_bin"] / "colab")
    wsl_state = _to_wsl_path(env_paths["state_dir"])

    # 1. Fake rejects calls outside the recorded contract
    for bad_args in (
        "cp -s rdlab-1 /content/a /tmp/b",
        "new -s rdlab-1 --json",
        "exec -s rdlab-1 -- 'echo hi'",
        "--unknown-global sessions",
    ):
        res = subprocess.run(
            [
                "bash",
                "-c",
                f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; '{wsl_fake_colab}' {bad_args}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert res.returncode == 2

    # 2. Time-one mode runs all 5 judges on 10 pilot items in both orders (AB, BA),
    #    records load time, median, p95, peak VRAM (5s interval), dtype, canary drift,
    #    and records pilot mass check as a harness readout (is_result=False), with wall 1500s and cap 2.5.
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path, count=12)
    out_t1 = tmp_path / "out_time_one"
    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "t1.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_t1 = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --time-one --out-dir '{_to_wsl_path(out_t1)}' --run-id t1-run-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_t1.returncode == 0, f"stdout={proc_t1.stdout}\nstderr={proc_t1.stderr}"

    outcome = json.loads((out_t1 / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["hardware"] == "A100"
    assert outcome["cap"] == 2.5
    assert outcome["wall_seconds"] == 1500
    assert outcome["outcome"] == "ok"

    receipt = json.loads((out_t1 / "vm_receipt.json").read_text(encoding="utf-8"))
    t1_summary = receipt["time_one_summary"]
    assert t1_summary["items_per_judge"] == 10
    assert t1_summary["orders"] == ["AB", "BA"]
    assert set(t1_summary["judges"].keys()) == set(BAKEOFF_JUDGE_SPECS.keys())
    for jid, jmetrics in t1_summary["judges"].items():
        assert jmetrics["calls_count"] == 20
        assert "load_time_seconds" in jmetrics
        assert "per_call_median_seconds" in jmetrics
        assert "per_call_p95_seconds" in jmetrics
        assert "peak_vram_mib" in jmetrics
        assert jmetrics["vram_sample_interval_seconds"] == 5
        assert jmetrics["dtype"] == "bfloat16"
        assert "canary_drift" in jmetrics
        expected_venv = "venv-d3" if jid == "d3-mini" else "venv-judge"
        assert jmetrics["venv"] == expected_venv

    mass_readout = receipt["pilot_mass_check"]
    assert mass_readout["kind"] == "harness_readout"
    assert mass_readout["is_result"] is False
    assert mass_readout["part_id"] == "pilot-40b"
    assert set(mass_readout["judges"].keys()) == set(BAKEOFF_JUDGE_SPECS.keys())


def _write_single_item_file(path: Path, row: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8", newline="\n")
    return path


def _assert_rig_dry_and_live_refuse_items_file(
    tmp_path: Path,
    items_file: Path,
    *,
    expected_field: str,
    forbidden_marker: str,
    tag: str,
) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / f"snap_{tag}"
    )
    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / f"{tag}.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)
    out_dry = tmp_path / f"out_dry_{tag}"
    proc_dry = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{wsl_launch}' --stub --dry --out-dir '{_to_wsl_path(out_dry)}' --run-id dry-{tag}",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_dry.returncode != 0
    assert proc_dry.stdout.strip() == ""
    assert "ITEMS_FIELD_NOT_ALLOWED" in proc_dry.stderr
    assert expected_field in proc_dry.stderr
    assert forbidden_marker not in proc_dry.stdout
    assert forbidden_marker not in proc_dry.stderr
    assert not env_paths["calls_log"].exists()

    out_live = tmp_path / f"out_live_{tag}"
    proc_live = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(out_live)}' --run-id live-{tag}",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_live.returncode != 0
    assert "ITEMS_FIELD_NOT_ALLOWED" in proc_live.stderr
    assert expected_field in proc_live.stderr
    assert forbidden_marker not in proc_live.stdout
    assert forbidden_marker not in proc_live.stderr
    assert not env_paths["calls_log"].exists()


def test_items_allowlist_row_with_key_basis_refuses(tmp_path: Path) -> None:
    marker = "PLANTED_KEY_BASIS_SECRET_VALUE_7001"
    items_file = _write_single_item_file(
        tmp_path / "part_public" / "items.jsonl",
        {
            "id": "item-0001",
            "material": "1. Alpha premise.\n2. Beta premise.",
            "claim": "Alpha beta claim.",
            "key_basis": marker,
        },
    )
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.collect_claim_ngram_sha256s(items_file)
    assert exc_info.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
    assert "key_basis" in str(exc_info.value)
    assert marker not in str(exc_info.value)

    _assert_rig_dry_and_live_refuse_items_file(
        tmp_path,
        items_file,
        expected_field="key_basis",
        forbidden_marker=marker,
        tag="key-basis",
    )


def test_items_allowlist_row_with_proof_basis_refuses(tmp_path: Path) -> None:
    marker = "PLANTED_PROOF_BASIS_SECRET_VALUE_7002"
    items_file = _write_single_item_file(
        tmp_path / "part_public" / "items.jsonl",
        {
            "id": "item-0001",
            "material": "1. Alpha premise.\n2. Beta premise.",
            "claim": "Alpha beta claim.",
            "proof_basis": marker,
        },
    )
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.collect_claim_ngram_sha256s(items_file)
    assert exc_info.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
    assert "proof_basis" in str(exc_info.value)
    assert marker not in str(exc_info.value)

    _assert_rig_dry_and_live_refuse_items_file(
        tmp_path,
        items_file,
        expected_field="proof_basis",
        forbidden_marker=marker,
        tag="proof-basis",
    )


def test_items_allowlist_row_with_triad_id_refuses(tmp_path: Path) -> None:
    marker = "PLANTED_TRIAD_ID_SECRET_VALUE_7003"
    items_file = _write_single_item_file(
        tmp_path / "part_public" / "items.jsonl",
        {
            "id": "item-0001",
            "material": "1. Alpha premise.\n2. Beta premise.",
            "claim": "Alpha beta claim.",
            "triad_id": marker,
        },
    )
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.collect_claim_ngram_sha256s(items_file)
    assert exc_info.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
    assert "triad_id" in str(exc_info.value)
    assert marker not in str(exc_info.value)

    _assert_rig_dry_and_live_refuse_items_file(
        tmp_path,
        items_file,
        expected_field="triad_id",
        forbidden_marker=marker,
        tag="triad-id",
    )


def test_items_allowlist_row_missing_claim_refuses(tmp_path: Path) -> None:
    marker = "PLANTED_MISSING_CLAIM_ROW_SECRET_7004"
    items_file = _write_single_item_file(
        tmp_path / "part_public" / "items.jsonl",
        {
            "id": "item-0001",
            "material": marker,
        },
    )
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.collect_claim_ngram_sha256s(items_file)
    assert exc_info.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
    assert "claim" in str(exc_info.value)
    assert marker not in str(exc_info.value)

    _assert_rig_dry_and_live_refuse_items_file(
        tmp_path,
        items_file,
        expected_field="claim",
        forbidden_marker=marker,
        tag="missing-claim",
    )

    non_obj_marker = "PLANTED_NON_OBJECT_SECRET_7005"
    non_obj_file = _write_single_item_file(
        tmp_path / "part_non_obj" / "items.jsonl",
        [non_obj_marker],
    )
    with pytest.raises(vm_judges.RefusalError) as exc_non_obj:
        vm_judges.collect_claim_ngram_sha256s(non_obj_file)
    assert exc_non_obj.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
    assert non_obj_marker not in str(exc_non_obj.value)


def test_items_allowlist_exact_id_material_claim_row_passes(tmp_path: Path) -> None:
    items_file = _write_single_item_file(
        tmp_path / "part_public" / "items.jsonl",
        {
            "id": "item-0001",
            "material": "1. Alpha premise establishes hop one.\n2. Beta premise links hop one to hop two.",
            "claim": "Alpha beta projection claim with only three fields.",
        },
    )
    assert vm_judges.ALLOWED_ITEM_FIELDS == frozenset({"id", "material", "claim"})
    assert vm_judges.verify_items_allowlist(items_file) == 1
    hashes = vm_judges.collect_claim_ngram_sha256s(items_file)
    assert len(hashes) >= 2


def test_items_allowlist_refusal_message_contains_field_name_and_not_value(
    tmp_path: Path,
) -> None:
    planted_marker = "MARKER_NEVER_PRINT_THIS_VALUE_9918273645"
    for bad_field in ("key_basis", "proof_basis", "triad_id", "key", "class", "verdict"):
        f_path = _write_single_item_file(
            tmp_path / f"case_{bad_field}" / "items.jsonl",
            {
                "id": "item-0001",
                "material": "1. Alpha premise.\n2. Beta premise.",
                "claim": "Alpha beta claim.",
                bad_field: planted_marker,
            },
        )
        with pytest.raises(vm_judges.RefusalError) as exc_info:
            vm_judges.collect_claim_ngram_sha256s(f_path)
        assert exc_info.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
        assert bad_field in str(exc_info.value)
        assert planted_marker not in str(exc_info.value)
        assert planted_marker not in exc_info.value.detail
        with pytest.raises(vm_judges.RefusalError) as exc_allow:
            vm_judges.verify_items_allowlist(f_path)
        assert exc_allow.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
        assert bad_field in str(exc_allow.value)
        assert planted_marker not in str(exc_allow.value)

    # Defence-in-depth on VM side: vm_judges.main("--step run") refuses an unverified items file
    # and never writes the marker value into status.json or vm_receipt.json.
    vm_dir = tmp_path / "vm_defence"
    vm_items = _write_single_item_file(
        vm_dir / "items" / "items.jsonl",
        {
            "id": "item-0001",
            "material": "1. Alpha premise.\n2. Beta premise.",
            "claim": "Alpha beta claim.",
            "key_basis": planted_marker,
        },
    )
    vm_run_dir = vm_dir / "run"
    vm_status = vm_dir / "status.json"
    rc = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "stub",
            "--repo-dir",
            str(REPO_ROOT),
            "--items-file",
            str(vm_items),
            "--run-dir",
            str(vm_run_dir),
            "--work-dir",
            str(vm_dir / "work"),
            "--receipts-dir",
            str(vm_dir / "receipts"),
            "--hf-home",
            str(vm_dir / "hf"),
            "--result-tar",
            str(vm_dir / "result.tar.gz"),
            "--status-file",
            str(vm_status),
            "--dryrun",
        ]
    )
    assert rc == 2
    status_text = vm_status.read_text(encoding="utf-8")
    receipt_text = (vm_run_dir / "vm_receipt.json").read_text(encoding="utf-8")
    assert "ITEMS_FIELD_NOT_ALLOWED" in status_text
    assert "key_basis" in status_text
    assert planted_marker not in status_text
    assert planted_marker not in receipt_text
    assert not vm_items.exists()


def test_kernel_exec_without_dunder_file_fails_when_dunder_file_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vm_src = VM_JUDGES_PY.read_text(encoding="utf-8")
    status_file = tmp_path / "gate_status.json"
    monkeypatch.setenv("RDLAB_STEP", "gate")
    monkeypatch.setenv("RDLAB_STATUS_FILE", str(status_file))
    monkeypatch.setenv("FAKE_COLAB_DISK_FREE_BYTES", "120000000000")
    monkeypatch.setattr(
        sys, "argv", ["ipykernel_launcher.py", "-f", "/tmp/kernel-cell.json"]
    )
    ns: dict[str, Any] = {"__name__": "__main__"}
    assert "__file__" not in ns
    exec(compile(vm_src, "<cell>", "exec"), ns)
    assert "__file__" not in vm_src
    assert ns.get("CELL_EXIT_CODE") == 0
    status_doc = json.loads(status_file.read_text(encoding="utf-8"))
    assert status_doc["step"] == "gate"
    assert status_doc["rc"] == 0
    assert status_doc["status"] == "ok"


def test_kernel_exec_gate_and_run_steps_complete(tmp_path: Path) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path, neighbour_sessions=[]
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path, count=4)
    out_dir = tmp_path / "out_kernel_exec"
    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "kernel.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    wsl_fake_colab = _to_wsl_path(env_paths["fake_bin"] / "colab")
    wsl_vm_script = _to_wsl_path(VM_JUDGES_PY)

    # Verify fake_colab's `exec -f` runs in a fresh namespace without `__file__`
    probe_script = tmp_path / "probe_no_dunder_file.py"
    probe_script.write_text(
        'if "__file__" in globals():\n'
        '    raise RuntimeError("__file__ must not be defined in kernel exec")\n'
        'print("KERNEL_CELL_NO_FILE_OK")\n',
        encoding="utf-8",
        newline="\n",
    )
    proc_probe = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} '{wsl_fake_colab}' new -s rdlab-probe </dev/null && "
            f"'{wsl_fake_colab}' exec -s rdlab-probe -f '{_to_wsl_path(probe_script)}' --timeout=30 </dev/null && "
            f"'{wsl_fake_colab}' stop -s rdlab-probe </dev/null",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_probe.returncode == 0, f"stderr={proc_probe.stderr}"
    assert "KERNEL_CELL_NO_FILE_OK" in proc_probe.stdout

    # Run full launcher --stub pipeline (both gate step before repo bundle upload and run step) under kernel-exec
    proc_stub = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id kernel-stub-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_stub.returncode == 0, (
        f"stdout={proc_stub.stdout}\nstderr={proc_stub.stderr}\n"
        f"cli_log={(out_dir / 'colab-cli.log').read_text(encoding='utf-8') if (out_dir / 'colab-cli.log').is_file() else ''}"
    )
    cli_log_text = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "VM_GATE_OK" in cli_log_text
    status_doc = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status_doc["status"] == "ok"
    assert status_doc["rc"] == 0
    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert receipt_doc["status"] == "ok"
    outcome_doc = json.loads(
        (out_dir / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_doc["outcome"] == "ok"
    assert Path(wsl_vm_script).is_file()


def test_items_upload_fails_at_6436a3a6_gate_and_passes_after_fix(
    tmp_path: Path,
) -> None:
    """Failing-first regression test: at 6436a3a6 (gate does not create /content/rdlab_items),
    `--stub` fails at `/content/rdlab_items/items.jsonl` upload with HTTP 500; after fix, it passes.
    """
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path, count=4)
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)

    # 1. Simulate 6436a3a6 vm_judges.py where `--step gate` prints VM_GATE_OK without creating /content/rdlab_items
    legacy_vm_script = tmp_path / "vm_judges_6436a3a6.py"
    legacy_src = VM_JUDGES_PY.read_text(encoding="utf-8").replace(
        "if args.step == \"gate\":\n        return run_gate(args)",
        'if args.step == "gate":\n'
        "        import platform\n"
        "        if args.status_file is not None:\n"
        '            write_vm_status_file(args.status_file, step="gate", rc=0, reason="VM_GATE_OK", status="ok")\n'
        '        print(f"VM_GATE_OK {platform.platform()}", flush=True)\n'
        "        return 0",
    )
    legacy_vm_script.write_text(legacy_src, encoding="utf-8", newline="\n")

    env_old = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_6436a3a6", neighbour_sessions=[]
    )
    out_old = tmp_path / "out_6436a3a6"
    exports_old = _base_launcher_exports(
        env_old,
        open_lock=tmp_path / "old.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_old = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_old} export RDLAB_VM_SCRIPT='{_to_wsl_path(legacy_vm_script)}'; "
            f"bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(out_old)}' --run-id stub-old-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_old.returncode != 0
    old_cli_log = (out_old / "colab-cli.log").read_text(encoding="utf-8")
    assert "Uploaded " in old_cli_log and "-> /content/repo_bundle.tar.gz" in old_cli_log
    assert "-> /content/receipts_bundle.tar.gz" in old_cli_log
    assert (
        "[colab] Upload failed: 500 Server Error: Internal Server Error for url:"
        in old_cli_log
    )
    assert "/api/contents/content/rdlab_items/items.jsonl" in old_cli_log
    old_outcome = json.loads(
        (out_old / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert old_outcome["outcome"] == "failed"
    assert "/content/rdlab_items/items.jsonl" in old_outcome["why"]

    # 2. With fixed vm_judges.py, `--step gate` creates `/content/rdlab_items` and logs it; `--stub` passes
    env_new = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_fixed", neighbour_sessions=[]
    )
    out_new = tmp_path / "out_fixed"
    exports_new = _base_launcher_exports(
        env_new,
        open_lock=tmp_path / "new.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_new = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_new} bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(out_new)}' --run-id stub-fixed-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_new.returncode == 0, f"stderr={proc_new.stderr}"
    new_cli_log = (out_new / "colab-cli.log").read_text(encoding="utf-8")
    assert "VM_GATE_OK " in new_cli_log
    assert "items_dir=/content/rdlab_items" in new_cli_log
    assert "required_bytes=69654163200" in new_cli_log
    assert (env_new["state_dir"] / "vm_fs" / "rdlab_items").is_dir()
    new_outcome = json.loads(
        (out_new / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert new_outcome["outcome"] == "ok"


def test_gate_disk_preflight_reports_free_space_and_refuses_when_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify disk preflight arithmetic, free-space reporting, and `VM_DISK_INSUFFICIENT` refusal."""
    import shutil as _shutil

    assert vm_judges.EXPECTED_WEIGHTS_TOTAL_BYTES == 49_451_597_039
    assert vm_judges.EXPECTED_WEIGHTS_DOWNLOAD_SECONDS == 797.4
    assert vm_judges.VENV_JUDGE_WHEEL_BYTES == 2_779_391_889
    assert vm_judges.VENV_D3_WHEEL_BYTES == 2_779_228_374
    assert vm_judges.VENVS_ESTIMATED_BYTES == 11_117_240_526
    assert vm_judges.REQUIRED_VM_DISK_BYTES == 69_654_163_200

    req_info = vm_judges.compute_required_vm_disk_bytes()
    assert req_info == {
        "weights_total_bytes": 49_451_597_039,
        "venvs_estimated_bytes": 11_117_240_526,
        "subtotal_bytes": 60_568_837_565,
        "required_bytes": 69_654_163_200,
    }

    # Receipt-driven calculation when all 5 receipts have `total_bytes`
    rec_dir = tmp_path / "real_sized_receipts"
    rec_dir.mkdir(parents=True, exist_ok=True)
    exact_receipt_totals = [
        ("flowaicom__Flow-Judge-v0.1.sha256.json", 7_644_593_918),
        ("ibm-granite__granite-guardian-4.1-8b.sha256.json", 16_770_923_290),
        ("nvidia__NVIDIA-Nemotron-3-Nano-4B-BF16.sha256.json", 7_964_584_688),
        ("qybaihe__mu-judge-nemotron-nano-4b.sha256.json", 7_964_239_519),
        ("vllm-sr__d3-mini.sha256.json", 9_107_255_624),
    ]
    for fname, tbytes in exact_receipt_totals:
        (rec_dir / fname).write_text(
            json.dumps({"total_bytes": tbytes, "files": {}}) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    assert (
        vm_judges.compute_required_vm_disk_bytes(rec_dir)["required_bytes"]
        == 69_654_163_200
    )

    # Direct unit check using monkeypatched shutil.disk_usage at exact boundary
    monkeypatch.delenv("FAKE_COLAB_DISK_FREE_BYTES", raising=False)
    monkeypatch.delenv("FAKE_COLAB_CONTENT_DISK_FREE_BYTES", raising=False)
    monkeypatch.delenv("FAKE_COLAB_HF_DISK_FREE_BYTES", raising=False)

    fake_usage_ok = _shutil._ntuple_diskusage(
        100_000_000_000, 30_345_836_800, 69_654_163_200
    )
    monkeypatch.setattr(_shutil, "disk_usage", lambda _p: fake_usage_ok)
    report_ok = vm_judges.check_vm_disk_preflight(
        content_dir=tmp_path, hf_home=tmp_path / "hf_home", receipts_dir=rec_dir
    )
    assert report_ok["content_free_bytes"] == 69_654_163_200
    assert report_ok["hf_home_free_bytes"] == 69_654_163_200
    assert report_ok["required_bytes"] == 69_654_163_200

    fake_usage_low = _shutil._ntuple_diskusage(
        100_000_000_000, 30_345_836_801, 69_654_163_199
    )
    monkeypatch.setattr(_shutil, "disk_usage", lambda _p: fake_usage_low)
    with pytest.raises(vm_judges.RefusalError) as exc_low:
        vm_judges.check_vm_disk_preflight(
            content_dir=tmp_path, hf_home=tmp_path / "hf_home", receipts_dir=rec_dir
        )
    assert exc_low.value.failure_class == "VM_DISK_INSUFFICIENT"
    assert "69654163200" in str(exc_low.value)

    # End-to-end launcher test with low disk on /content (and FAKE_COLAB_FORCE_EXEC_RC_ZERO=1)
    env_low = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_low_disk", neighbour_sessions=[]
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "low_disk_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "low_disk_data", count=4)
    out_low = tmp_path / "out_low_disk"
    exports_low = _base_launcher_exports(
        env_low,
        open_lock=tmp_path / "low_disk.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_low = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_low} export FAKE_COLAB_DISK_FREE_BYTES=50000000000; "
            f"export FAKE_COLAB_FORCE_EXEC_RC_ZERO=1; "
            f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_low)}' --run-id low-disk-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_low.returncode == 2
    assert "VM_DISK_INSUFFICIENT" in proc_low.stderr
    low_outcome = json.loads(
        (out_low / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert low_outcome["outcome"] == "failed"
    assert "VM_DISK_INSUFFICIENT" in low_outcome["why"]
    # Verify no uploads occurred when gate refused on low disk
    calls = [
        json.loads(ln)["subcmd"]
        for ln in env_low["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert "upload" not in calls
    assert "stop" in calls

    # Also test low disk on HF_HOME specifically while /content has sufficient disk
    env_hf_low = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_hf_low", neighbour_sessions=[]
    )
    out_hf_low = tmp_path / "out_hf_low"
    exports_hf_low = _base_launcher_exports(
        env_hf_low,
        open_lock=tmp_path / "hf_low.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_hf_low = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_hf_low} export FAKE_COLAB_CONTENT_DISK_FREE_BYTES=120000000000; "
            f"export FAKE_COLAB_HF_DISK_FREE_BYTES=10000000000; "
            f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_hf_low)}' --run-id hf-low-disk-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_hf_low.returncode == 2
    assert "VM_DISK_INSUFFICIENT" in proc_hf_low.stderr


def test_upload_enforces_parent_dir_overwrite_large_file_path_length_and_missing_local(
    tmp_path: Path,
) -> None:
    """Verify `colab upload` enforces parent dir existence, missing local file, overwrite, size cap, and path length."""
    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path, neighbour_sessions=[{"session_id": "rdlab-up-01", "gpu": "CPU"}]
    )
    wsl_colab = _to_wsl_path(env_paths["fake_bin"] / "colab")
    wsl_state = _to_wsl_path(env_paths["state_dir"])
    vm_fs = env_paths["state_dir"] / "vm_fs"

    src_file = tmp_path / "payload.txt"
    src_file.write_text("v1-payload\n", encoding="utf-8", newline="\n")
    wsl_src = _to_wsl_path(src_file)

    def _run_upload(local_arg: str, remote_arg: str, extra_env: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; {extra_env} "
                f"'{wsl_colab}' upload -s rdlab-up-01 '{local_arg}' '{remote_arg}' </dev/null",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    # 1. Missing parent directory on VM -> HTTP 500
    res_no_parent = _run_upload(wsl_src, "/content/missing_dir/payload.txt")
    assert res_no_parent.returncode == 1
    assert "500 Server Error" in res_no_parent.stdout
    assert not (vm_fs / "missing_dir").exists()

    # 2. Missing local file -> exits 1
    res_no_local = _run_upload(_to_wsl_path(tmp_path / "does_not_exist.txt"), "/content/payload.txt")
    assert res_no_local.returncode == 1
    assert "Local file" in res_no_local.stdout and "not found" in res_no_local.stdout

    # 3. Upload to /content/payload.txt succeeds, and second upload overwrites in place
    res_ok1 = _run_upload(wsl_src, "/content/payload.txt")
    assert res_ok1.returncode == 0
    assert (vm_fs / "payload.txt").read_text(encoding="utf-8") == "v1-payload\n"

    src_file.write_text("v2-overwritten\n", encoding="utf-8", newline="\n")
    res_ok2 = _run_upload(wsl_src, "/content/payload.txt")
    assert res_ok2.returncode == 0
    assert (vm_fs / "payload.txt").read_text(encoding="utf-8") == "v2-overwritten\n"

    # 4. Upload targeting an existing directory on VM -> HTTP 500
    (vm_fs / "existing_subdir").mkdir(parents=True, exist_ok=True)
    res_is_dir = _run_upload(wsl_src, "/content/existing_subdir")
    assert res_is_dir.returncode == 1
    assert "500 Server Error" in res_is_dir.stdout

    # 5. Large file exceeding FAKE_COLAB_MAX_UPLOAD_BYTES -> HTTP 413
    res_too_large = _run_upload(
        wsl_src,
        "/content/payload.txt",
        extra_env="export FAKE_COLAB_MAX_UPLOAD_BYTES=4;",
    )
    assert res_too_large.returncode == 1
    assert "413 Client Error: Payload Too Large" in res_too_large.stdout

    # 6. Path component > 255 bytes or path > FAKE_COLAB_MAX_PATH_LEN -> HTTP 400
    long_name = "a" * 256 + ".txt"
    res_long_comp = _run_upload(wsl_src, f"/content/{long_name}")
    assert res_long_comp.returncode == 1
    assert "400 Client Error: Bad Request (path too long)" in res_long_comp.stdout

    res_long_total = _run_upload(
        wsl_src,
        "/content/payload.txt",
        extra_env="export FAKE_COLAB_MAX_PATH_LEN=12;",
    )
    assert res_long_total.returncode == 1
    assert "400 Client Error: Bad Request (path too long)" in res_long_total.stdout


def test_download_enforces_missing_remote_local_parent_overwrite_and_truncated_corrupt_archive(
    tmp_path: Path,
) -> None:
    """Verify `colab download` enforces missing remote file, local parent dir, overwrite, and truncated download detection."""
    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path, neighbour_sessions=[{"session_id": "rdlab-dl-01", "gpu": "CPU"}]
    )
    wsl_colab = _to_wsl_path(env_paths["fake_bin"] / "colab")
    wsl_state = _to_wsl_path(env_paths["state_dir"])
    vm_fs = env_paths["state_dir"] / "vm_fs"

    def _run_download(remote_arg: str, local_arg: str, extra_env: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; {extra_env} "
                f"'{wsl_colab}' download -s rdlab-dl-01 '{remote_arg}' '{local_arg}' </dev/null",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    # 1. Missing remote file -> exits 1 and does not create local file
    missing_dst = tmp_path / "missing_out.json"
    res_missing = _run_download("/content/nonexistent.json", _to_wsl_path(missing_dst))
    assert res_missing.returncode == 1
    assert "File or directory not found" in res_missing.stderr
    assert not missing_dst.exists()

    # 2. Remote path is a directory -> exits 1
    (vm_fs / "a_dir").mkdir(parents=True, exist_ok=True)
    res_dir = _run_download("/content/a_dir", _to_wsl_path(tmp_path / "dir_out"))
    assert res_dir.returncode == 1
    assert "path is a directory" in res_dir.stderr

    # 3. Missing local parent directory -> exits 1 (does not auto-mkdir local parent)
    (vm_fs / "sample.txt").write_text("remote-v1\n", encoding="utf-8", newline="\n")
    no_parent_dst = tmp_path / "no_such_local_parent" / "sample.txt"
    res_no_local_parent = _run_download("/content/sample.txt", _to_wsl_path(no_parent_dst))
    assert res_no_local_parent.returncode == 1
    assert "No such file or directory" in res_no_local_parent.stderr
    assert not no_parent_dst.exists()

    # 4. Existing local file is overwritten in place
    local_dst = tmp_path / "sample_local.txt"
    local_dst.write_text("stale-local\n", encoding="utf-8", newline="\n")
    res_overwrite = _run_download("/content/sample.txt", _to_wsl_path(local_dst))
    assert res_overwrite.returncode == 0
    assert local_dst.read_text(encoding="utf-8") == "remote-v1\n"

    # 5. End-to-end launcher refuses truncated/interrupted download of status.json or result.tar.gz
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "trunc_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "trunc_data", count=4)
    for trunc_target in ("status.json", "result.tar.gz"):
        tag = trunc_target.replace(".", "_")
        env_t = fake_colab.setup_fake_colab_environment(
            tmp_path / f"env_{tag}", neighbour_sessions=[]
        )
        out_t = tmp_path / f"out_{tag}"
        exports_t = _base_launcher_exports(
            env_t,
            open_lock=tmp_path / f"{tag}.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
        )
        proc_t = subprocess.run(
            [
                "bash",
                "-c",
                f"{exports_t} export FAKE_COLAB_TRUNCATE_DOWNLOAD='{trunc_target}'; "
                f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_t)}' --run-id trunc-{tag}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc_t.returncode != 0
        outcome_t = json.loads(
            (out_t / "outcome.draft.json").read_text(encoding="utf-8")
        )
        assert outcome_t["outcome"] == "failed"
        assert not (out_t / "outputs-closed.json").exists()


def test_exec_env_quoting_persistence_timeout_and_shared_kernel_gate_to_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `colab exec` `--env` validation/quoting/persistence, `--timeout`, and shared-kernel gate->run execution."""
    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path, neighbour_sessions=[{"session_id": "rdlab-exec-01", "gpu": "CPU"}]
    )
    wsl_colab = _to_wsl_path(env_paths["fake_bin"] / "colab")
    wsl_state = _to_wsl_path(env_paths["state_dir"])

    probe_py = tmp_path / "env_probe.py"
    probe_py.write_text(
        "import json, os\n"
        "print(json.dumps({\n"
        "    'EQ_VAL': os.environ.get('EQ_VAL'),\n"
        "    'SPACE_VAL': os.environ.get('SPACE_VAL'),\n"
        "    'PERSIST_KEY': os.environ.get('PERSIST_KEY'),\n"
        "    'UNPASSED_RIG': os.environ.get('RDLAB_OUT_DIR'),\n"
        "    'PRIOR_COUNT': globals().get('_PRIOR_EXEC_COUNT'),\n"
        "}))\n",
        encoding="utf-8",
        newline="\n",
    )
    wsl_probe = _to_wsl_path(probe_py)

    # 1. Invalid --env variable name rejected with rc=2
    res_bad_env = subprocess.run(
        [
            "bash",
            "-c",
            f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; "
            f"'{wsl_colab}' exec -s rdlab-exec-01 --env '123INVALID=val' -f '{wsl_probe}' </dev/null",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res_bad_env.returncode == 2
    assert "Invalid --env variable name" in res_bad_env.stderr

    # 2. Invalid --timeout rejected with rc=2
    for bad_timeout in ("0", "-5", "not_a_number"):
        res_bad_to = subprocess.run(
            [
                "bash",
                "-c",
                f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; "
                f"'{wsl_colab}' exec -s rdlab-exec-01 -f '{wsl_probe}' --timeout={bad_timeout} </dev/null",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert res_bad_to.returncode == 2

    # 3. Call 1 sets EQ_VAL (contains '='), SPACE_VAL (contains spaces and '='), and PERSIST_KEY;
    #    unpassed rig variable RDLAB_OUT_DIR is stripped.
    res_call1 = subprocess.run(
        [
            "bash",
            "-c",
            f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; "
            f"export RDLAB_OUT_DIR='/rig/should_not_leak'; "
            f"'{wsl_colab}' exec -s rdlab-exec-01 "
            f"--env 'EQ_VAL=alpha=beta==gamma' "
            f"--env 'SPACE_VAL=hello world = ok' "
            f"--env 'PERSIST_KEY=from_call_1' "
            f"-f '{wsl_probe}' --timeout=30 </dev/null",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res_call1.returncode == 0, f"stderr={res_call1.stderr}"
    out1 = json.loads(res_call1.stdout.strip().splitlines()[-1])
    assert out1["EQ_VAL"] == "alpha=beta==gamma"
    assert out1["SPACE_VAL"] == "hello world = ok"
    assert out1["PERSIST_KEY"] == "from_call_1"
    assert out1["UNPASSED_RIG"] is None
    assert out1["PRIOR_COUNT"] == 0

    # 4. Call 2 on the same session sees PERSIST_KEY persisted from Call 1, and can overwrite EQ_VAL
    res_call2 = subprocess.run(
        [
            "bash",
            "-c",
            f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; "
            f"'{wsl_colab}' exec -s rdlab-exec-01 "
            f"--env 'EQ_VAL=overwritten_in_call_2' "
            f"-f '{wsl_probe}' --timeout=30 </dev/null",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res_call2.returncode == 0
    out2 = json.loads(res_call2.stdout.strip().splitlines()[-1])
    assert out2["EQ_VAL"] == "overwritten_in_call_2"
    assert out2["PERSIST_KEY"] == "from_call_1"
    assert out2["PRIOR_COUNT"] == 1

    # 5. Execute vm_judges.py `--step gate` then `--step run` in the EXACT SAME Python kernel namespace (`ns`)
    shared_vm_fs = tmp_path / "shared_kernel_vm_fs"
    shared_vm_fs.mkdir(parents=True, exist_ok=True)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "shared_kernel_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "shared_kernel_data", count=4)

    vm_src = VM_JUDGES_PY.read_text(encoding="utf-8")
    shared_ns: dict[str, Any] = {"__name__": "__main__"}
    monkeypatch.setattr(
        sys, "argv", ["ipykernel_launcher.py", "-f", "/tmp/kernel-shared.json"]
    )
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(shared_vm_fs))
    monkeypatch.setenv("FAKE_COLAB_DISK_FREE_BYTES", "120000000000")
    monkeypatch.setenv("RDLAB_STATUS_FILE", str(shared_vm_fs / "status.json"))
    monkeypatch.setenv(
        "RDLAB_ITEMS_FILE", str(shared_vm_fs / "rdlab_items" / "items.jsonl")
    )
    monkeypatch.setenv("RDLAB_STEP", "gate")

    exec(compile(vm_src, "<cell>", "exec"), shared_ns)
    assert shared_ns.get("CELL_EXIT_CODE") == 0
    assert (shared_vm_fs / "rdlab_items").is_dir()

    # Now upload items into the directory created by gate, and execute `--step run` in the SAME `shared_ns`
    (shared_vm_fs / "rdlab_items" / "items.jsonl").write_bytes(items_file.read_bytes())
    (shared_vm_fs / "rdlab_items" / "manifest.json").write_bytes(
        (items_file.parent / "manifest.json").read_bytes()
    )
    monkeypatch.setenv("RDLAB_STEP", "run")
    monkeypatch.setenv("RDLAB_MODE", "stub")
    monkeypatch.setenv("VM_DRYRUN", "1")
    monkeypatch.setenv("RDLAB_REPO_DIR", str(REPO_ROOT))
    monkeypatch.setenv("RDLAB_RECEIPTS_DIR", str(receipts_dir))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("RDLAB_RUN_DIR", str(shared_vm_fs / "rdlab_run"))
    monkeypatch.setenv("RDLAB_WORK_DIR", str(shared_vm_fs / "rdlab_work"))
    monkeypatch.setenv("RDLAB_RESULT_TAR", str(shared_vm_fs / "result.tar.gz"))
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", EXPECTED_D3_LOCK_SHA256)

    exec(compile(vm_src, "<cell>", "exec"), shared_ns)
    assert shared_ns.get("CELL_EXIT_CODE") == 0
    run_status = json.loads(
        (shared_vm_fs / "status.json").read_text(encoding="utf-8")
    )
    assert run_status["step"] == "build_allowlisted_result_tar"
    assert run_status["rc"] == 0
    assert run_status["status"] == "ok"


def test_stop_and_sessions_race_and_orphan_row_after_stop(tmp_path: Path) -> None:
    """Verify stop/start race (`FAKE_COLAB_KEEP_SESSION_ON_STOP=1` and `[?]` orphan row) and `[colab]` prefix lines."""
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)

    # 1. Session still listed as `[rdlab-...]` after `colab stop` -> fails with clear reason
    env_keep = fake_colab.setup_fake_colab_environment(
        tmp_path / "keep", neighbour_sessions=[]
    )
    out_keep = tmp_path / "out_keep"
    cmd_keep = (
        f"export PATH='{_to_wsl_path(env_keep['fake_bin'])}:{_to_wsl_path(env_keep['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_keep['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_keep['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'keep.lock')}'; "
        f"export FAKE_COLAB_KEEP_SESSION_ON_STOP=1; "
        f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_keep)}' --run-id race-keep-01"
    )
    proc_keep = subprocess.run(
        ["bash", "-c", cmd_keep],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_keep.returncode == 1
    outcome_keep = json.loads(
        (out_keep / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_keep["outcome"] == "failed"
    assert "still present after colab stop" in outcome_keep["why"]

    # 2. Session listed as `[?]` on server after local session was removed by `colab stop` -> fails
    env_orphan = fake_colab.setup_fake_colab_environment(
        tmp_path / "orphan", neighbour_sessions=[]
    )
    out_orphan = tmp_path / "out_orphan"
    cmd_orphan = (
        f"export PATH='{_to_wsl_path(env_orphan['fake_bin'])}:{_to_wsl_path(env_orphan['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_orphan['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_orphan['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'orphan.lock')}'; "
        f"export FAKE_COLAB_ORPHAN_ON_STOP=1; "
        f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_orphan)}' --run-id race-orphan-01"
    )
    proc_orphan = subprocess.run(
        ["bash", "-c", cmd_orphan],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_orphan.returncode == 1
    outcome_orphan = json.loads(
        (out_orphan / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_orphan["outcome"] == "failed"
    assert "[?]" in outcome_orphan["why"]

    # 3. `[colab]` informational prefix lines (pruned stale session / auto-update notice) do not break session-smoke
    env_prefix = fake_colab.setup_fake_colab_environment(
        tmp_path / "prefix", neighbour_sessions=[]
    )
    out_prefix = tmp_path / "out_prefix"
    cmd_prefix = (
        f"export PATH='{_to_wsl_path(env_prefix['fake_bin'])}:{_to_wsl_path(env_prefix['sentinel_bin'])}':\"$PATH\"; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_prefix['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_prefix['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'prefix.lock')}'; "
        f"export FAKE_COLAB_SESSIONS_PREFIX_LINE='[colab] Pruned 1 stale local session(s).'; "
        f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_prefix)}' --run-id prefix-ok-01"
    )
    proc_prefix = subprocess.run(
        ["bash", "-c", cmd_prefix],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_prefix.returncode == 0, f"stderr={proc_prefix.stderr}"
    outcome_prefix = json.loads(
        (out_prefix / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_prefix["outcome"] == "ok"


def test_real_service_assumptions_doc_table_and_line_citations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `scripts/colab/REAL_SERVICE_ASSUMPTIONS.md` row counts, required numbers, and `path:line` citations."""
    assumptions_md = SCRIPTS_COLAB_DIR / "REAL_SERVICE_ASSUMPTIONS.md"
    assert assumptions_md.is_file()
    text = assumptions_md.read_text(encoding="utf-8")

    # Verify required constants and arithmetic appear in REAL_SERVICE_ASSUMPTIONS.md
    for required_token in (
        "49,451,597,039",
        "46.06 GiB",
        "797.4",
        "797 s",
        "2,779,391,889",
        "2,779,228,374",
        "11,117,240,526",
        "69,654,163,200",
        "VM_DISK_INSUFFICIENT",
    ):
        assert required_token in text, f"Missing {required_token!r} in REAL_SERVICE_ASSUMPTIONS.md"

    table_rows = [
        ln
        for ln in text.splitlines()
        if re.match(r"^\|\s*\d+\s*\|", ln.strip())
    ]
    assert len(table_rows) == 20
    enforced_rows = [ln for ln in table_rows if "**Fake now enforces**" in ln]
    stub_rows = [ln for ln in table_rows if "**Left for the stub**" in ln]
    assert len(enforced_rows) == 18
    assert len(stub_rows) == 2

    # Verify every `path:line` or `path:start-end` citation in REAL_SERVICE_ASSUMPTIONS.md is valid
    citations = re.findall(
        r"`((?:scripts|tests|src)/[A-Za-z0-9_./-]+):(\d+)(?:-(\d+))?`", text
    )
    assert len(citations) >= 25
    for rel_path, start_s, end_s in citations:
        full_p = REPO_ROOT / rel_path
        assert full_p.is_file(), f"Cited path does not exist: {rel_path}"
        line_count = len(full_p.read_text(encoding="utf-8").splitlines())
        start_ln = int(start_s)
        end_ln = int(end_s) if end_s else start_ln
        assert 1 <= start_ln <= end_ln <= line_count, (
            f"Invalid line range {rel_path}:{start_s}-{end_s} (file has {line_count} lines)"
        )

    # Also exercise edge branches in vm_judges.py disk preflight and run_gate for 100% patch coverage
    assert vm_judges.resolve_content_dir(tmp_path) == tmp_path
    monkeypatch.delenv("RDLAB_VM_CONTENT_DIR", raising=False)
    assert vm_judges.resolve_content_dir() == Path("/content")

    # Receipts with `files[*].bytes` fallback and a malformed JSON file
    fallback_rec = tmp_path / "fallback_receipts"
    fallback_rec.mkdir(parents=True, exist_ok=True)
    for idx in range(5):
        (fallback_rec / f"o__r{idx}.sha256.json").write_text(
            json.dumps({"files": {"w.bin": {"bytes": 10_000_000_000}}}) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    (fallback_rec / "z__bad.sha256.json").write_text("{bad json", encoding="utf-8")
    fb_res = vm_judges.compute_required_vm_disk_bytes(fallback_rec)
    assert fb_res["weights_total_bytes"] == 50_000_000_000

    # `main(["--step", "gate", ...])` refusal writing status.json directly on low disk
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(tmp_path / "gate_content"))
    monkeypatch.setenv("FAKE_COLAB_DISK_FREE_BYTES", "1000")
    gate_status = tmp_path / "gate_refused_status.json"
    rc_gate = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            str(tmp_path / "gate_content" / "rdlab_items" / "items.jsonl"),
            "--hf-home",
            str(tmp_path / "gate_content" / "hf_home"),
            "--status-file",
            str(gate_status),
        ]
    )
    assert rc_gate == 2
    gate_doc = json.loads(gate_status.read_text(encoding="utf-8"))
    assert gate_doc["step"] == "gate"
    assert gate_doc["rc"] == 2
    assert gate_doc["failure_class"] == "VM_DISK_INSUFFICIENT"



