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
                    "part_id": "pilot-40b",
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


def test_outputs_closed_written_before_any_key_path_touched(tmp_path: Path) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    out_dir = tmp_path / "out_closed_order"
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

    # Simulate subsequent rig-only scorer touching keys AFTER launcher completes
    fake_colab.record_file_access(
        env_paths["state_dir"],
        "read_keys",
        tmp_path / "sealed_store" / "keys.sealed.jsonl",
    )

    events = fake_colab.read_file_access_log(env_paths["state_dir"])
    event_names = [e["event"] for e in events]
    assert "write_outputs_closed" in event_names
    assert "read_keys" in event_names
    closed_idx = event_names.index("write_outputs_closed")
    key_idx = event_names.index("read_keys")
    assert closed_idx < key_idx
    # And no event prior to read_keys touched any key or sealed_store path
    for ev in events[:key_idx]:
        assert "sealed_store" not in ev["path"]
        assert "keys.sealed" not in ev["path"]


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
