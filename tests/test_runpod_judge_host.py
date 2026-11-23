"""Tests for the RunPod offrig host path in the judge bake-off launcher."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_JUDGES_SH = REPO_ROOT / "scripts" / "colab" / "launch_judges.sh"
FAKE_OFFRIG_MCP_PY = REPO_ROOT / "tests" / "fixtures" / "fake_offrig_mcp.py"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "colab"))
import offrig_host  # noqa: E402
import vm_judges  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_part import run_bakeoff_part_build  # noqa: E402


def _to_bash_path(path: Path | str) -> str:
    """Convert a host filesystem path to the POSIX path visible inside `bash`."""
    raw = str(path).replace("\\", "/")
    if os.name == "nt" and len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
        drive = raw[0].lower()
        rest = raw[2:].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return raw


def _make_synthetic_items_with_part(root: Path, *, repo_root: Path = REPO_ROOT) -> tuple[Path, Path]:
    part_res = run_bakeoff_part_build(
        output_dir=root / "part_build",
        triad_count=2,
        stub_mode=True,
        mock_writer=True,
        shuffle_seed=(1 << 127) | 101,
        grant_id="test-grant-runpod-host",
        repo_root=repo_root,
    )
    part_dir = Path(part_res["part_dir"])
    src_rows = [
        json.loads(ln)
        for ln in (part_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    items_path = root / "projected_items" / "items.jsonl"
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(
        "\n".join(
            json.dumps({"id": r["id"], "material": r["material"], "claim": r["claim"]})
            for r in src_rows
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return items_path, part_dir


def _make_synthetic_receipts(root: Path) -> Path:
    receipts_dir = root / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    for judge_id, spec in sorted(vm_judges.BAKEOFF_JUDGE_SPECS.items()):
        owner, repo = spec.repo_id.split("/", 1)
        weights = f"weights-for-{judge_id}-{spec.revision}".encode("utf-8")
        files_manifest: dict[str, dict[str, Any]] = {
            "model.safetensors": {
                "sha256": hashlib.sha256(weights).hexdigest(),
                "bytes": len(weights),
            }
        }
        doc = {
            "repo": spec.repo_id,
            "revision": spec.revision,
            "files": files_manifest,
        }
        (receipts_dir / f"{owner}__{repo}.sha256.json").write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return receipts_dir


def _init_clean_git_repo(repo_dir: Path) -> None:
    for fname in (".gitignore", "pyproject.toml", "README.md"):
        src_f = REPO_ROOT / fname
        if src_f.is_file():
            shutil.copyfile(src_f, repo_dir / fname)
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Offline Test"],
        ["git", "add", "."],
        ["git", "commit", "-q", "-m", "Clean test fixture tree"],
    ):
        subprocess.run(cmd, cwd=repo_dir, stdin=subprocess.DEVNULL, check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def clean_repo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a clean git worktree copy of the repo scripts and source for launcher staging."""
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts/colab"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode == 0 and not status.stdout.strip():
        return REPO_ROOT

    clone_dir = tmp_path_factory.mktemp("clean_repo")
    for rel in ("src", "scripts/colab"):
        shutil.copytree(REPO_ROOT / rel, clone_dir / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _init_clean_git_repo(clone_dir)
    return clone_dir


def _write_offrig_config_toml(
    config_dir: Path,
    *,
    profile: str = "rdlab-judge",
    image: str = offrig_host.DEFAULT_RUNPOD_IMAGE_REF,
    include_job: bool = True,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg_toml = config_dir / "config.toml"
    lines = [
        "[[profiles]]",
        f'name = "{profile}"',
        'tier = "medium"',
        'gpu_type_ids = ["NVIDIA RTX PRO 6000 Blackwell Server Edition"]',
        "gpu_count = 1",
        "volume_gb = 200",
        "container_disk_gb = 60",
        "context_length = 0",
        "models = []",
        "wait_for_gpu_minutes = 20",
        "parallel = 1",
        'min_cuda = "13.0"',
        "min_vram_gb = 90",
    ]
    if include_job:
        lines.extend(
            [
                "",
                "[profiles.job]",
                f'image = "{image}"',
                'min_cuda = "12.8"',
            ]
        )
    cfg_toml.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return cfg_toml


def _setup_fake_offrig_env(
    root: Path,
    *,
    repo_root: Path,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    vm_root_dir = root / "pod_vm_root"
    vm_root_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "fake_offrig_cfg.json"
    log_path = root / "fake_offrig_calls.jsonl"
    state_path = root / "fake_offrig_state.json"
    offrig_cfg_dir = root / "offrig_cfg"
    _write_offrig_config_toml(
        offrig_cfg_dir,
        image=str((config_overrides or {}).get("profile_image") or offrig_host.DEFAULT_RUNPOD_IMAGE_REF),
    )

    cfg: dict[str, Any] = {
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "nvidia_smi_stdout": "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 580.95.05\n",
        "nvidia_smi_exit_code": 0,
        "pod_id": "pod-fake-6000-01",
        "plan_id": 101,
        "job_id": 201,
        "cost_per_hr": 1.89,
        "spent_usd": 0.42,
        "pod_vm_root": "/workspace/job",
        "vm_root_dir": _to_bash_path(vm_root_dir),
    }
    if config_overrides:
        cfg.update(config_overrides)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")

    fake_mcp_script = repo_root / "tests" / "fixtures" / "fake_offrig_mcp.py"
    if not fake_mcp_script.is_file():
        fake_mcp_script = FAKE_OFFRIG_MCP_PY
    shim_sh = root / "fake-offrig-mcp.sh"
    shim_sh.write_text(
        f"#!/usr/bin/env bash\n"
        f"exec python3 {shlex.quote(_to_bash_path(fake_mcp_script))} \"$@\"\n",
        encoding="utf-8",
        newline="\n",
    )
    shim_sh.chmod(0o755)
    return {
        "cfg_path": cfg_path,
        "log_path": log_path,
        "state_path": state_path,
        "vm_root_dir": vm_root_dir,
        "offrig_config_dir": offrig_cfg_dir,
        "shim_sh": shim_sh,
        "open_lock": root / "open.lock",
    }


def _read_call_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []
    return [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _run_bash_launcher(
    *,
    repo_root: Path,
    args: list[str],
    env_vars: dict[str, str],
    timeout: int = 60,
    include_lock_envs: bool = True,
) -> subprocess.CompletedProcess[str]:
    launch_sh = repo_root / "scripts" / "colab" / "launch_judges.sh"
    merged_env = dict(env_vars)
    if include_lock_envs:
        merged_env.setdefault("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
        merged_env.setdefault("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(merged_env.items()))
    cmd_args = " ".join(shlex.quote(a) for a in args)
    clean_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    bash_cmd = f"export PATH={shlex.quote(clean_path)}; {exports} bash {shlex.quote(_to_bash_path(launch_sh))} {cmd_args}"
    return subprocess.run(
        ["bash", "-c", bash_cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _extract_step_envs_from_dry_output(stdout_text: str) -> dict[str, tuple[str, dict[str, str]]]:
    """Parse `(script, env_dict)` for `gate`, `probe`, and `run` invocations in `--dry` stdout."""
    steps: dict[str, tuple[str, dict[str, str]]] = {}
    for line in stdout_text.splitlines():
        tokens = shlex.split(line)
        if not tokens:
            continue
        env_map: dict[str, str] = {}
        i = 0
        while i < len(tokens):
            if tokens[i] == "--env" and i + 1 < len(tokens) and "=" in tokens[i + 1]:
                k, v = tokens[i + 1].split("=", 1)
                env_map[k] = v
                i += 2
            else:
                i += 1
        step_name = env_map.get("RDLAB_STEP")
        if step_name not in ("alive", "gate", "probe", "run"):
            continue
        script_path = ""
        for idx, tok in enumerate(tokens):
            if tok == "-f" and idx + 1 < len(tokens):
                script_path = tokens[idx + 1]
                break
        if not script_path:
            for tok in tokens:
                if tok.endswith("/vm_judges.py"):
                    script_path = tok
                    break
        steps[step_name] = (script_path, env_map)
    return steps


def test_static_guard_all_colab_calls_have_stdin_devnull() -> None:
    spec = importlib.util.spec_from_file_location(
        "test_colab_judge_launcher", REPO_ROOT / "tests" / "test_colab_judge_launcher.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    checked, errors = mod.check_colab_invocations_have_stdin_devnull(
        LAUNCH_JUDGES_SH.read_text(encoding="utf-8")
    )
    assert errors == [], f"Missing </dev/null on colab calls: {errors}"
    assert len(checked) >= 12


def test_colab_dry_run_unchanged_against_base_commit(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    import re as _re

    golden_path = REPO_ROOT / "tests" / "fixtures" / "colab_dry_599276bb.json"
    golden_doc = json.loads(golden_path.read_text(encoding="utf-8"))

    git_has_base = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "-e", "599276bb:scripts/colab/launch_judges.sh"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    base_repo: Path | None = None
    if git_has_base.returncode == 0:
        base_bytes = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", "599276bb:scripts/colab/launch_judges.sh"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
        ).stdout
        assert hashlib.sha256(base_bytes).hexdigest() == golden_doc["base_launch_judges_sh_sha256"]
        base_repo = tmp_path / "base_repo"
        shutil.copytree(clean_repo_root / "src", base_repo / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(clean_repo_root / "scripts" / "colab", base_repo / "scripts" / "colab", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (base_repo / "scripts" / "colab" / "launch_judges.sh").write_bytes(base_bytes)
        _init_clean_git_repo(base_repo)

    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    scenarios = [
        ("session_smoke", ["--session-smoke", "--run-id", "parity-smoke"]),
        ("probe", ["--probe", "--run-id", "parity-probe"]),
        (
            "time_one",
            [
                "--time-one",
                "--card",
                "A100",
                "--shape",
                "standard",
                "--max-rate",
                "5",
                "--run-id",
                "parity-t1",
                "--items-file",
                _to_bash_path(items_path),
                "--part-dir",
                _to_bash_path(part_dir),
                "--receipts-dir",
                _to_bash_path(receipts_dir),
            ],
        ),
    ]

    def _norm_stdout(txt: str, repo_p: Path, out_p: Path) -> str:
        s = txt.replace(_to_bash_path(repo_p), "<REPO>").replace(_to_bash_path(out_p), "<OUT>")
        s = _re.sub(r"RDLAB_KIT_HEAD_SHA=[0-9a-f]{40}", "RDLAB_KIT_HEAD_SHA=<SHA>", s)
        s = _re.sub(r"RDLAB_KIT_VM_JUDGES_PY_SHA256=[0-9a-f]{64}", "RDLAB_KIT_VM_JUDGES_PY_SHA256=<VM_PY_SHA>", s)
        s = _re.sub(r"RDLAB_KIT_LAUNCH_JUDGES_SH_SHA256=[0-9a-f]{64}", "RDLAB_KIT_LAUNCH_JUDGES_SH_SHA256=<SH_SHA>", s)
        return s

    def _norm_stderr(txt: str) -> str:
        lines = [ln for ln in txt.splitlines() if not ln.startswith("KIT_IDENTITY")]
        return _re.sub(r"part_json_sha256=[0-9a-f]{64}", "part_json_sha256=<PART_SHA>", "\n".join(lines))

    for scenario_name, extra_args in scenarios:
        head_out = tmp_path / f"head_out_{scenario_name}"
        common_env = {
            "RDLAB_RUN_DIR_BASE": _to_bash_path(tmp_path),
        }
        res_head = _run_bash_launcher(
            repo_root=clean_repo_root,
            args=["--dry", "--out-dir", _to_bash_path(head_out), *extra_args],
            env_vars=common_env,
        )
        expected_golden = golden_doc["scenarios"][scenario_name]
        assert res_head.returncode == expected_golden["returncode"] == 0, res_head.stderr
        assert _norm_stdout(res_head.stdout, clean_repo_root, head_out) == expected_golden["stdout"]
        assert _norm_stderr(res_head.stderr) == expected_golden["stderr"]

        if base_repo is not None:
            base_out = tmp_path / f"base_out_{scenario_name}"
            res_base = _run_bash_launcher(
                repo_root=base_repo,
                args=["--dry", "--out-dir", _to_bash_path(base_out), *extra_args],
                env_vars=common_env,
            )
            assert res_base.returncode == 0, res_base.stderr
            assert _norm_stdout(res_head.stdout, clean_repo_root, head_out) == _norm_stdout(res_base.stdout, base_repo, base_out)
            assert _norm_stderr(res_head.stderr) == _norm_stderr(res_base.stderr)


def test_colab_and_runpod_dry_parity_identical_bundles_and_allowlisted_env_diff(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    colab_out = tmp_path / "out_colab"
    runpod_out = tmp_path / "out_runpod"
    shared_env = {
        "RDLAB_DRY_STAGE": "1",
        "SOURCE_DATE_EPOCH": "1700000000",
        "RDLAB_OFFRIG_PROFILE": "rdlab-judge",
    }
    shared_args = [
        "--dry",
        "--time-one",
        "--max-rate",
        "5",
        "--cap",
        "2.5",
        "--run-id",
        "parity-win-01",
        "--items-file",
        _to_bash_path(items_path),
        "--part-dir",
        _to_bash_path(part_dir),
        "--receipts-dir",
        _to_bash_path(receipts_dir),
    ]
    res_colab = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "colab", "--card", "G4", "--shape", "standard", "--out-dir", _to_bash_path(colab_out), *shared_args],
        env_vars=shared_env,
    )
    res_runpod = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--out-dir", _to_bash_path(runpod_out), *shared_args],
        env_vars=shared_env,
    )
    assert res_colab.returncode == 0, res_colab.stderr
    assert res_runpod.returncode == 0, res_runpod.stderr

    # 1. Byte-identical staged artifacts
    colab_stage = colab_out / "stage"
    runpod_stage = runpod_out / "stage"
    colab_files = sorted(p.relative_to(colab_stage).as_posix() for p in colab_stage.rglob("*") if p.is_file())
    runpod_files = sorted(p.relative_to(runpod_stage).as_posix() for p in runpod_stage.rglob("*") if p.is_file())
    assert colab_files == runpod_files
    assert "repo_bundle.tar.gz" in colab_files
    assert "receipts_bundle.tar.gz" in colab_files
    assert "items.jsonl" in colab_files
    for rel in colab_files:
        c_sha = hashlib.sha256((colab_stage / rel).read_bytes()).hexdigest()
        r_sha = hashlib.sha256((runpod_stage / rel).read_bytes()).hexdigest()
        assert c_sha == r_sha, f"Staged file mismatch for {rel}: {c_sha} != {r_sha}"

    # 2. Identical env and script invocation outside explicit VM-root & host/image allowlist
    colab_steps = _extract_step_envs_from_dry_output(res_colab.stdout)
    runpod_steps = _extract_step_envs_from_dry_output(res_runpod.stdout)

    vm_root_path_keys = {
        "RDLAB_REPO_DIR",
        "RDLAB_REPO_TAR",
        "RDLAB_RECEIPTS_DIR",
        "RDLAB_RECEIPTS_TAR",
        "RDLAB_ITEMS_FILE",
        "RDLAB_RUN_DIR",
        "RDLAB_RESULT_TAR",
        "RDLAB_STATUS_FILE",
    }
    host_only_keys = {
        "RDLAB_VM_ROOT",
        "RDLAB_HOST",
        "RDLAB_POD_ID",
        "RDLAB_RUNPOD_IMAGE",
        "RDLAB_RUNPOD_IMAGE_TAG",
        "RDLAB_RUNPOD_IMAGE_DIGEST",
        "RDLAB_CARD",
        "RDLAB_SHAPE",
        "RDLAB_CARD_TYPE",
    }

    for step in ("gate", "run"):
        c_script, c_env = colab_steps[step]
        r_script, r_env = runpod_steps[step]
        assert Path(c_script).name == Path(r_script).name == "vm_judges.py"
        if step == "run":
            assert r_script == "/workspace/job/vm_judges.py"
        all_keys = set(c_env) | set(r_env)
        for key in sorted(all_keys):
            if key in vm_root_path_keys:
                assert c_env[key].startswith("/content/")
                assert r_env[key] == "/workspace/job/" + c_env[key][len("/content/") :]
            elif key in host_only_keys:
                continue
            else:
                assert c_env.get(key) == r_env.get(key), f"Step {step} env {key!r} drifted: {c_env.get(key)!r} != {r_env.get(key)!r}"

    assert runpod_steps["run"][1]["RDLAB_HOST"] == "runpod"
    assert runpod_steps["run"][1]["RDLAB_VM_ROOT"] == "/workspace/job"
    assert runpod_steps["run"][1]["RDLAB_CARD_TYPE"] == "RUNPOD-RTX-PRO-6000-SERVER"
    assert runpod_steps["run"][1]["RDLAB_RUNPOD_IMAGE_TAG"] == offrig_host.DEFAULT_RUNPOD_IMAGE_TAG
    assert runpod_steps["run"][1]["RDLAB_RUNPOD_IMAGE_DIGEST"] == offrig_host.DEFAULT_RUNPOD_IMAGE_DIGEST


def test_runpod_success_receipts_cost_runtime_and_download_duration(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    env_info = _setup_fake_offrig_env(tmp_path / "offrig", repo_root=clean_repo_root)
    out_dir = tmp_path / "out_ok"

    res = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--container-disk-gb",
            "80",
            "--run-id",
            "rp-ok-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_dir),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_info["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_info["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_info["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_info["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_info["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"

    calls = _read_call_log(env_info["log_path"])
    tools = [c["tool"] for c in calls]
    assert tools[0] == "offrig_plan"
    assert calls[0]["arguments"]["no_fallback"] is True
    assert calls[0]["arguments"]["container_disk_gb"] == 80
    assert calls[0]["arguments"]["wait_minutes"] == offrig_host.DEFAULT_LAUNCH_WAIT_MINUTES
    assert calls[0]["arguments"]["max_hours"] == pytest.approx(offrig_host.compute_watchdog_max_hours(1500))
    assert "offrig_launch" in tools
    assert "offrig_job" in tools
    assert "offrig_shutdown" in tools
    assert tools[-1] == "offrig_shutdown"

    vm_receipt = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert vm_receipt["host"] == "runpod"
    assert vm_receipt["pod_id"] == "pod-fake-6000-01"
    assert vm_receipt["hardware"] == "RUNPOD-RTX-PRO-6000-SERVER"
    assert "image" not in vm_receipt
    assert vm_receipt["image_requested"]["tag"] == offrig_host.DEFAULT_RUNPOD_IMAGE_TAG
    assert vm_receipt["image_requested"]["digest"] == offrig_host.DEFAULT_RUNPOD_IMAGE_DIGEST
    assert vm_receipt["image_source"] == "offrig profile"
    assert vm_receipt["model_download_seconds"] == pytest.approx(21.0)
    assert vm_receipt["model_download_seconds_by_judge"] == {"flow-judge-v0.1": 12.4, "mu-judge": 8.6}
    rt = vm_receipt["runtime_environment"]
    assert rt["host_gpu_name"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert rt["driver_version"] == "580.95.05"
    assert rt["cuda_version"] == "13.0"
    assert rt["python_version"] == "3.12.11"
    assert rt["system_python_version"]
    assert rt["vllm_version"] == "0.34.0"

    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["outcome"] == "ok"
    assert outcome["host"] == "runpod"
    assert outcome["venue"] == "runpod"
    assert outcome["pod_id"] == "pod-fake-6000-01"
    assert "image" not in outcome
    assert outcome["image_requested"]["tag"] == offrig_host.DEFAULT_RUNPOD_IMAGE_TAG
    assert outcome["image_source"] == "offrig profile"
    assert outcome["cost_usd"] == pytest.approx(0.42)
    assert outcome["cost"] == pytest.approx(0.42)
    assert outcome["cost_currency"] == "USD"
    assert outcome["cost_basis"] == "offrig_shutdown_spent_usd"
    assert outcome["model_download_seconds"] == pytest.approx(21.0)
    assert outcome["watchdog_deadline"]
    assert (out_dir / "outputs-closed.json").is_file()


@pytest.mark.parametrize(
    ("label", "cfg_overrides", "expected_code"),
    [
        (
            "wrong_gpu",
            {
                "plan_gpus": "1x NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "gpu": "NVIDIA A100-SXM4-80GB",
                "nvidia_smi_stdout": "NVIDIA A100-SXM4-80GB, 81920, 580.95.05\n",
            },
            "RUNPOD_GPU_MISMATCH",
        ),
        (
            "workstation_edition",
            {
                "plan_gpus": "1x NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "nvidia_smi_stdout": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 580.95.05\n",
            },
            "RUNPOD_GPU_WORKSTATION_REFUSED",
        ),
        (
            "low_vram",
            {
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "nvidia_smi_stdout": "NVIDIA RTX PRO 6000 Blackwell Server Edition, 49152, 580.95.05\n",
            },
            "RUNPOD_VRAM_INSUFFICIENT",
        ),
        (
            "offrig_vs_smi_disagreement",
            {
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "nvidia_smi_stdout": "NVIDIA L4, 23034, 580.95.05\n",
            },
            "RUNPOD_GPU_DISAGREEMENT",
        ),
    ],
)
def test_runpod_card_gate_refusals_shut_down_pod(
    label: str,
    cfg_overrides: dict[str, Any],
    expected_code: str,
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    env_info = _setup_fake_offrig_env(
        tmp_path / f"offrig_{label}",
        repo_root=clean_repo_root,
        config_overrides=cfg_overrides,
    )
    out_dir = tmp_path / f"out_{label}"

    res = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            f"rp-{label}",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_dir),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_info["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_info["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_info["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_info["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_info["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res.returncode == 2
    assert expected_code in res.stderr

    calls = _read_call_log(env_info["log_path"])
    tools = [c["tool"] for c in calls]
    assert "offrig_put" not in tools
    assert tools[-1] == "offrig_shutdown"

    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["outcome"] == "failed"
    assert outcome["failure_class"] == expected_code
    assert outcome["pod_shutdown"] is True
    assert outcome["cost_usd"] == pytest.approx(0.42)


def test_runpod_shutdown_on_failed_step_with_retry_and_timeout_and_sigterm(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    # 1. Failed gate step + transient first shutdown failure retried cleanly
    env_fail = _setup_fake_offrig_env(
        tmp_path / "fail_step",
        repo_root=clean_repo_root,
        config_overrides={"fail_steps": {"gate": 2}, "fail_shutdown_attempts": 1},
    )
    out_fail = tmp_path / "out_fail"
    res_fail = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-fail-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_fail),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_fail["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_fail["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_fail["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_fail["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_fail["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_fail.returncode == 2
    assert "VM_DISK_INSUFFICIENT" in res_fail.stderr
    fail_state = json.loads(env_fail["state_path"].read_text(encoding="utf-8"))
    assert fail_state["shutdown"] is True
    assert fail_state["shutdown_attempts"] == 2
    assert json.loads((out_fail / "outcome.draft.json").read_text(encoding="utf-8"))["cost_usd"] == pytest.approx(0.42)

    # 2. Wall timeout on hanging `run` step
    env_to = _setup_fake_offrig_env(
        tmp_path / "timeout",
        repo_root=clean_repo_root,
        config_overrides={"hang_steps": ["run"], "spent_usd": 0.77},
    )
    out_to = tmp_path / "out_to"
    res_to = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-to-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_to),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_to["offrig_config_dir"]),
            "RDLAB_WALL_SECONDS": "2",
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_to["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_to["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_to["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_to["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.05",
        },
        timeout=30,
    )
    assert res_to.returncode == 124
    to_calls = _read_call_log(env_to["log_path"])
    assert any(c["tool"] == "offrig_exec" and c["arguments"].get("action") == "stop" for c in to_calls)
    to_state = json.loads(env_to["state_path"].read_text(encoding="utf-8"))
    assert to_state["shutdown"] is True
    assert to_state["shutdown_reason"] == "wall_timeout"
    assert json.loads((out_to / "outcome.draft.json").read_text(encoding="utf-8"))["cost_usd"] == pytest.approx(0.77)

    # 3. SIGTERM while `run` is active shuts down the pod cleanly
    env_sig = _setup_fake_offrig_env(
        tmp_path / "sigterm",
        repo_root=clean_repo_root,
        config_overrides={"hang_steps": ["run"], "spent_usd": 0.31},
    )
    out_sig = tmp_path / "out_sig"
    launch_sh = clean_repo_root / "scripts" / "colab" / "launch_judges.sh"
    sig_env = {
        "OFFRIG_CONFIG_DIR": _to_bash_path(env_sig["offrig_config_dir"]),
        "RDLAB_WALL_SECONDS": "300",
        "RDLAB_JUDGE_LOCK_SHA256": vm_judges.EXPECTED_JUDGE_LOCK_SHA256,
        "RDLAB_D3_LOCK_SHA256": vm_judges.EXPECTED_D3_LOCK_SHA256,
        "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_sig["shim_sh"]),
        "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
        "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_sig["cfg_path"]),
        "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_sig["log_path"]),
        "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_sig["state_path"]),
        "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        "RDLAB_OFFRIG_POLL_INTERVAL": "0.05",
    }
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(sig_env.items()))
    clean_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    cmd_args = " ".join(
        shlex.quote(a)
        for a in [
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-sig-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_sig),
        ]
    )
    proc = subprocess.Popen(
        [
            "bash",
            "-c",
            f"export PATH={shlex.quote(clean_path)}; {exports} bash {shlex.quote(_to_bash_path(launch_sh))} {cmd_args}",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            calls = _read_call_log(env_sig["log_path"])
            if any(
                c["tool"] == "offrig_exec"
                and c["arguments"].get("action") == "start"
                and c["arguments"].get("name") == "run"
                for c in calls
            ):
                break
            assert proc.poll() is None, f"process exited early: rc={proc.returncode}"
            time.sleep(0.05)
        else:
            raise AssertionError("timed out waiting for `run` step start")

        pid_file = out_sig / ".rig_launcher.pid"
        assert pid_file.is_file()
        subprocess.run(
            ["bash", "-c", f"kill -TERM {shlex.quote(pid_file.read_text(encoding='utf-8').strip())}"],
            stdin=subprocess.DEVNULL,
            check=True,
        )
        proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 143
    sig_state = json.loads(env_sig["state_path"].read_text(encoding="utf-8"))
    assert sig_state["shutdown"] is True
    assert sig_state["shutdown_reason"] == "sigterm_interrupt"
    assert json.loads((out_sig / "outcome.draft.json").read_text(encoding="utf-8"))["cost_usd"] == pytest.approx(0.31)


def test_runpod_refuses_missing_offrig_or_failing_commands_before_paid_launch(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    # 1. Invalid CLI options before any host call
    bad_host = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "unknown", "--out-dir", _to_bash_path(tmp_path / "bad_host")],
        env_vars={},
    )
    assert bad_host.returncode == 2
    assert "HOST_INVALID" in bad_host.stderr

    bad_disk = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--container-disk-gb", "0", "--out-dir", _to_bash_path(tmp_path / "bad_disk")],
        env_vars={},
    )
    assert bad_disk.returncode == 2
    assert "CONTAINER_DISK_INVALID" in bad_disk.stderr

    # 2. Missing offrig-mcp binary
    out_missing = tmp_path / "out_missing_bin"
    res_missing = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-missing-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_missing),
        ],
        env_vars={},
    )
    assert res_missing.returncode == 2
    assert "OFFRIG_NOT_FOUND" in res_missing.stderr
    out_missing_doc = json.loads((out_missing / "outcome.draft.json").read_text(encoding="utf-8"))
    assert out_missing_doc["failure_class"] == "OFFRIG_NOT_FOUND"
    assert out_missing_doc["cost_usd"] == 0.0

    # 3. Failing `offrig_plan` refuses before `offrig_launch`
    env_plan_fail = _setup_fake_offrig_env(
        tmp_path / "plan_fail",
        repo_root=clean_repo_root,
        config_overrides={"fail_tool": "offrig_plan", "fail_message": "no offers matching RTX PRO 6000"},
    )
    out_plan_fail = tmp_path / "out_plan_fail"
    res_plan_fail = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-planfail-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_plan_fail),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_plan_fail["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_plan_fail["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_plan_fail["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_plan_fail["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_plan_fail["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res_plan_fail.returncode == 2
    assert "OFFRIG_PLAN_FAILED" in res_plan_fail.stderr
    plan_tools = [c["tool"] for c in _read_call_log(env_plan_fail["log_path"])]
    assert plan_tools == ["offrig_plan"]
    assert json.loads((out_plan_fail / "outcome.draft.json").read_text(encoding="utf-8"))["cost_usd"] == 0.0

    # 4. Failing `offrig_launch` triggers `offrig_shutdown` once launch was attempted for a plan id
    env_launch_fail = _setup_fake_offrig_env(
        tmp_path / "launch_fail",
        repo_root=clean_repo_root,
        config_overrides={"fail_tool": "offrig_launch", "fail_message": "pod creation denied"},
    )
    out_launch_fail = tmp_path / "out_launch_fail"
    res_launch_fail = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-launchfail-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_launch_fail),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_launch_fail["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_launch_fail["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_launch_fail["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_launch_fail["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_launch_fail["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res_launch_fail.returncode == 2
    assert "OFFRIG_LAUNCH_FAILED" in res_launch_fail.stderr
    launch_tools = [c["tool"] for c in _read_call_log(env_launch_fail["log_path"])]
    assert launch_tools == ["offrig_plan", "offrig_launch", "offrig_shutdown"]
    assert json.loads((out_launch_fail / "outcome.draft.json").read_text(encoding="utf-8"))["pod_shutdown"] is True


def test_runpod_session_smoke_and_probe(tmp_path: Path, clean_repo_root: Path) -> None:
    for mode_flag, run_id, exp_outcome in (
        ("--session-smoke", "rp-smoke-01", "ok"),
        ("--probe", "rp-probe-01", "probe"),
    ):
        env_info = _setup_fake_offrig_env(tmp_path / run_id, repo_root=clean_repo_root)
        out_dir = tmp_path / f"out_{run_id}"
        res = _run_bash_launcher(
            repo_root=clean_repo_root,
            args=["--host", "runpod", mode_flag, "--run-id", run_id, "--out-dir", _to_bash_path(out_dir)],
            env_vars={
                "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_info["shim_sh"]),
                "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
                "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_info["cfg_path"]),
                "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_info["log_path"]),
                "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_info["state_path"]),
                "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
                "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
            },
        )
        assert res.returncode == 0, res.stderr
        assert json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))["outcome"] == exp_outcome


def _write_unit_mcp_shim(tmp_path: Path) -> Path:
    if os.name == "nt":
        shim = tmp_path / "fake-mcp.cmd"
        shim.write_text(
            f'@echo off\r\n"{sys.executable}" "{FAKE_OFFRIG_MCP_PY}" %*\r\n',
            encoding="utf-8",
        )
    else:
        shim = tmp_path / "fake-mcp.sh"
        shim.write_text(
            f"#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE_OFFRIG_MCP_PY))} \"$@\"\n",
            encoding="utf-8",
            newline="\n",
        )
        shim.chmod(0o755)
    return shim


def test_offrig_host_unit_and_cli_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    assert offrig_host.split_image_ref("vllm/vllm-openai:v0.34.0@sha256:" + "a" * 64) == (
        "vllm/vllm-openai:v0.34.0",
        "sha256:" + "a" * 64,
    )
    assert offrig_host.split_image_ref("vllm/vllm-openai:v0.34.0") == ("vllm/vllm-openai:v0.34.0", None)
    assert offrig_host.compute_watchdog_max_hours(1500) == pytest.approx(1.0834)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PLAN_FAILED"):
        offrig_host.compute_watchdog_max_hours(0)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PLAN_FAILED"):
        offrig_host.compute_watchdog_max_hours(60, launch_wait_minutes=-1)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_DEADLINE_INVALID"):
        offrig_host.parse_offrig_deadline_utc("not-a-deadline")
    assert offrig_host.parse_offrig_deadline_utc("2026-11-23 13:05 UTC") == pytest.approx(1795439100.0)
    rem_budget = offrig_host.compute_remaining_run_budget_seconds(
        "2026-11-23 13:05 UTC",
        1795437900.0,
    )
    assert rem_budget["remaining_before_deadline_seconds"] == 1200
    assert rem_budget["reserve_seconds"] == 300
    assert rem_budget["remaining_run_seconds"] == 900

    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_UNREADABLE"):
        offrig_host.parse_nvidia_smi_gpu_csv("")
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_UNREADABLE"):
        offrig_host.parse_nvidia_smi_gpu_csv("bad-line")
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_UNREADABLE"):
        offrig_host.parse_nvidia_smi_gpu_csv("GPU, notanint, 580.1")
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_UNREADABLE"):
        offrig_host.parse_nvidia_smi_gpu_csv("GPU, 0 MiB, 580.1")
    parsed_smi = offrig_host.parse_nvidia_smi_gpu_csv("NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 580.95.05\n")
    assert parsed_smi["vram_mib"] == 97887

    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_UNREADABLE"):
        offrig_host.verify_runpod_gpu_gate("", "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 580.1")

    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_NOT_FOUND"):
        offrig_host.resolve_offrig_mcp_bin({"RDLAB_OFFRIG_MCP_BIN": str(tmp_path / "does-not-exist")})
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_NOT_FOUND"):
        offrig_host.resolve_offrig_mcp_bin({"PATH": str(tmp_path)})

    cmd_str = offrig_host.format_env_shell_command(["RDLAB_STEP=gate"], "/workspace/job/vm_judges.py", ["--mode", "time-one"])
    assert "RDLAB_STEP=gate" in cmd_str and "/workspace/job/vm_judges.py" in cmd_str
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_EXEC_FAILED"):
        offrig_host.format_env_shell_command(["NO_EQUALS"], "/workspace/job/vm_judges.py")
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_EXEC_FAILED"):
        offrig_host.format_env_shell_command(["1BAD=val"], "/workspace/job/vm_judges.py")

    # Exercise `_extract_tool_payload` branches
    sc_ok = offrig_host._extract_tool_payload("offrig_plan", {"structuredContent": {"plan_id": 7}})
    assert sc_ok["plan_id"] == 7
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PUT_FAILED"):
        offrig_host._extract_tool_payload(
            "offrig_put",
            {"isError": True, "content": [{"type": "text", "text": "raw failure"}]},
        )

    # Full unit test of `offrig_host.main` & `OffrigMcpClient` using host-native fake MCP shim
    shim = _write_unit_mcp_shim(tmp_path)
    vm_root = tmp_path / "unit_vm_root"
    cfg_p = tmp_path / "unit_cfg.json"
    state_p = tmp_path / "unit_state.json"
    log_p = tmp_path / "unit_calls.jsonl"
    cfg_p.write_text(
        json.dumps(
            {
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "nvidia_smi_stdout": "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 580.95.05\n",
                "pod_id": "pod-unit-01",
                "plan_id": 77,
                "job_id": 88,
                "cost_per_hr": 2.0,
                "spent_usd": 0.15,
                "pod_vm_root": "/workspace/job",
                "vm_root_dir": _to_bash_path(vm_root) if os.name == "posix" else str(vm_root),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RDLAB_OFFRIG_MCP_BIN", str(shim))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_CONFIG", str(cfg_p))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_STATE", str(state_p))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_LOG", str(log_p))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_REPO_ROOT", str(REPO_ROOT))
    unit_cfg_dir = tmp_path / "unit_offrig_cfg"
    _write_offrig_config_toml(unit_cfg_dir)
    monkeypatch.setenv("OFFRIG_CONFIG_DIR", str(unit_cfg_dir))

    assert offrig_host.main(["check-bin"]) == 0
    assert str(shim) in capsys.readouterr().out

    assert offrig_host.main(["check-path-visibility", "--bin", str(shim), "--path", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["compatible"] is True

    assert offrig_host.main(
        ["check-profile-image", "--profile", "rdlab-judge", "--image", offrig_host.DEFAULT_RUNPOD_IMAGE_REF]
    ) == 0
    assert json.loads(capsys.readouterr().out)["match"] is True

    assert offrig_host.main(
        ["remaining-run-budget", "--deadline", "2026-11-23 13:05 UTC", "--now-epoch", "1795437900"]
    ) == 0
    assert json.loads(capsys.readouterr().out)["remaining_run_seconds"] == 900

    assert offrig_host.main(["plan", "--project", str(REPO_ROOT), "--profile", "job", "--wall-seconds", "600", "--container-disk-gb", "60", "--note", "unit"]) == 0
    assert json.loads(capsys.readouterr().out)["plan_id"] == 77

    assert offrig_host.main(["launch-and-wait", "--project", str(REPO_ROOT), "--plan-id", "77", "--poll-interval", "0.01"]) == 0
    assert json.loads(capsys.readouterr().out)["pod_id"] == "pod-unit-01"

    assert offrig_host.main(["verify-gpu", "--project", str(REPO_ROOT), "--plan-id", "77", "--offrig-gpu", "NVIDIA RTX PRO 6000 Blackwell Server Edition"]) == 0
    assert json.loads(capsys.readouterr().out)["match"] is True

    local_src = tmp_path / "local_up.txt"
    local_src.write_text("hello", encoding="utf-8")
    assert offrig_host.main(["put", "--project", str(REPO_ROOT), "--plan-id", "77", "--local", str(local_src), "--pod", "/workspace/job/up.txt"]) == 0
    assert offrig_host.main(["exec-run", "--project", str(REPO_ROOT), "--plan-id", "77", "--command", "mkdir -p /workspace/job/sub"]) == 0
    local_dl = tmp_path / "dl.txt"
    assert offrig_host.main(["get", "--project", str(REPO_ROOT), "--plan-id", "77", "--pod", "/workspace/job/up.txt", "--local", str(local_dl)]) == 0
    assert local_dl.read_text(encoding="utf-8") == "hello"

    assert offrig_host.main(
        [
            "exec-vm-step",
            "--project",
            str(REPO_ROOT),
            "--plan-id",
            "77",
            "--job-name",
            "probe",
            "--vm-script",
            "/workspace/job/vm_judges.py",
            "--env",
            "RDLAB_STEP=probe",
            "--save-log",
            str(tmp_path / "saved_probe.log"),
        ]
    ) == 0
    capsys.readouterr()

    assert offrig_host.main(["shutdown", "--project", str(REPO_ROOT), "--plan-id", "77", "--reason", "unit_done"]) == 0
    assert json.loads(capsys.readouterr().out)["spent"] == pytest.approx(0.15)

    # Exercise timeout and error branches in `exec_vm_step`, `launch_and_wait`, `verify_pod_gpu`, `OffrigMcpClient`
    with offrig_host.OffrigMcpClient(project_dir=REPO_ROOT) as client:
        with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_COMMAND_FAILED"):
            client._request("nonexistent/method", {})
    cfg_p.write_text(
        json.dumps(
            {
                "pod_vm_root": "/workspace/job",
                "vm_root_dir": str(vm_root),
                "job_state": "failed",
                "job_error": "capacity gone",
                "nvidia_smi_exit_code": 1,
                "nvidia_smi_stderr": "driver lost",
                "hang_steps": ["hang"],
            }
        ),
        encoding="utf-8",
    )
    with offrig_host.OffrigMcpClient(project_dir=REPO_ROOT) as client:
        with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_JOB_FAILED"):
            offrig_host.launch_and_wait(client, plan_id=77, poll_interval=0.01)
        with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_UNREADABLE"):
            offrig_host.verify_pod_gpu(client, plan_id=77, offrig_gpu="NVIDIA RTX PRO 6000 Blackwell Server Edition")
        rc_to, _tail = offrig_host.exec_vm_step(
            client,
            plan_id=77,
            job_name="hang",
            vm_script="/workspace/job/vm_judges.py",
            env_pairs=["RDLAB_STEP=run"],
            timeout_seconds=0,
            poll_interval=0.01,
            save_log=str(tmp_path / "hang.log"),
        )
        assert rc_to == 124

    monkeypatch.setenv("RDLAB_OFFRIG_MCP_BIN", str(tmp_path / "absent"))
    assert offrig_host.main(["check-bin"]) == 2
    assert "OFFRIG_NOT_FOUND" in capsys.readouterr().err


def test_vm_judges_runpod_receipt_runtime_and_content_root_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uv_env = vm_judges.build_vm_uv_env(
        base_env={"UV_SYSTEM_PYTHON": "1", "UV_OVERRIDE": "/tmp/override.txt"},
        content_dir=tmp_path,
    )
    assert "UV_SYSTEM_PYTHON" not in uv_env
    assert "UV_OVERRIDE" not in uv_env
    assert "RDLAB_VM_CONTENT_DIR" in vm_judges.FORBIDDEN_COUNTED_TEST_HOOK_NAMES

    monkeypatch.delenv("RDLAB_VM_CONTENT_DIR", raising=False)
    monkeypatch.setenv("RDLAB_HOST", "runpod")
    monkeypatch.delenv("RDLAB_VM_ROOT", raising=False)
    assert vm_judges.resolve_content_dir() == Path("/workspace/job")
    monkeypatch.setenv("RDLAB_VM_ROOT", str(tmp_path / "runpod_root"))
    assert vm_judges.resolve_content_dir() == tmp_path / "runpod_root"

    # Image split from tag+digest and bare image ref (explicitly labelled as requested + offrig profile)
    rp_fields = vm_judges.resolve_host_receipt_fields(
        {
            "RDLAB_HOST": "runpod",
            "RDLAB_POD_ID": "pod-abc",
            "RDLAB_RUNPOD_IMAGE": "vllm/vllm-openai:v0.34.0@sha256:" + "b" * 64,
        }
    )
    assert rp_fields == {
        "host": "runpod",
        "pod_id": "pod-abc",
        "image_requested": {
            "ref": "vllm/vllm-openai:v0.34.0@sha256:" + "b" * 64,
            "tag": "vllm/vllm-openai:v0.34.0",
            "digest": "sha256:" + "b" * 64,
        },
        "image_source": "offrig profile",
    }
    bare_tag = vm_judges.resolve_host_receipt_fields({"RDLAB_RUNPOD_IMAGE": "vllm/vllm-openai:v0.34.0"})
    assert bare_tag["image_requested"] == {
        "ref": "vllm/vllm-openai:v0.34.0",
        "tag": "vllm/vllm-openai:v0.34.0",
        "digest": None,
    }
    assert bare_tag["image_source"] == "offrig profile"
    assert vm_judges.resolve_host_receipt_fields({}) == {
        "host": "colab",
        "pod_id": None,
        "image_requested": None,
        "image_source": None,
    }

    dl_dur = vm_judges.compute_model_download_duration(
        [
            {"judge_id": "j1", "seconds": 4.25},
            {"judge_id": "j2", "elapsed_seconds": "5.75"},
            {"judge_id": "j3", "seconds": None},
            {"judge_id": "j4", "seconds": "invalid"},
        ]
    )
    assert dl_dur["model_download_seconds"] == pytest.approx(10.0)
    assert dl_dur["model_download_seconds_by_judge"] == {"j1": 4.25, "j2": 5.75}

    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(tmp_path / "custom_root"))
    parsed = vm_judges.parse_args([])
    assert parsed.items_file == tmp_path / "custom_root" / "rdlab_items" / "items.jsonl"
    assert parsed.receipts_dir == tmp_path / "custom_root" / "rdlab_receipts"
    assert parsed.run_dir == tmp_path / "custom_root" / "rdlab_run"
    assert parsed.work_dir == tmp_path / "custom_root" / "rdlab_work"

    monkeypatch.setenv("RDLAB_HOST", "runpod")
    monkeypatch.setenv("RDLAB_CARD_TYPE", "RUNPOD-RTX-PRO-6000-SERVER")
    monkeypatch.setattr(
        vm_judges,
        "read_vm_gpu_name",
        lambda: ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "ok"),
    )
    card_chk = vm_judges.verify_vm_card_type("time-one", dryrun=False, until=None)
    assert card_chk is not None
    assert card_chk["source"] == "offrig pod + nvidia-smi"
    assert card_chk["card_type"] == "RUNPOD-RTX-PRO-6000-SERVER"

    # RunPod VM check refuses Workstation Edition and refuses Colab G4 card_type
    monkeypatch.setattr(
        vm_judges,
        "read_vm_gpu_name",
        lambda: ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "ok"),
    )
    with pytest.raises(vm_judges.RefusalError) as ws_exc:
        vm_judges.verify_vm_card_type("time-one", dryrun=False, until=None)
    assert ws_exc.value.failure_class == vm_judges.CARD_TYPE_MISMATCH

    monkeypatch.setenv("RDLAB_CARD_TYPE", "G4")
    monkeypatch.setattr(
        vm_judges,
        "read_vm_gpu_name",
        lambda: ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "ok"),
    )
    with pytest.raises(vm_judges.RefusalError) as g4_exc:
        vm_judges.verify_vm_card_type("time-one", dryrun=False, until=None)
    assert g4_exc.value.failure_class == vm_judges.CARD_TYPE_MISMATCH
    monkeypatch.setenv("RDLAB_CARD_TYPE", "RUNPOD-RTX-PRO-6000-SERVER")

    # Runtime environment collection against a mocked `nvidia-smi` and fake venv python
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        smi_path = fake_bin / "nvidia-smi.cmd"
        smi_path.write_text(
            "@echo off\r\n"
            'if "%1"=="--query-gpu=name,driver_version" (\r\n'
            "  echo NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.95.05\r\n"
            ") else (\r\n"
            "  echo NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0\r\n"
            ")\r\n",
            encoding="utf-8",
        )
    else:
        smi_path = fake_bin / "nvidia-smi"
        smi_path.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "${1:-}" == "--query-gpu=name,driver_version" ]]; then\n'
            '  echo "NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.95.05"\n'
            "else\n"
            '  echo "NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0"\n'
            "fi\n",
            encoding="utf-8",
            newline="\n",
        )
        smi_path.chmod(0o755)

    work_dir = tmp_path / "work"
    vllm_py = work_dir / vm_judges.VLLM_VENV_NAME / "bin" / "python"
    vllm_py.parent.mkdir(parents=True, exist_ok=True)
    vllm_py.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}/usr/bin{os.pathsep}/bin")

    orig_run_vm_sub = vm_judges.run_vm_subprocess

    import platform as _platform

    def _patched_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if Path(cmd[0]) == vllm_py:
            if len(cmd) >= 3 and "platform" in cmd[2]:
                return subprocess.CompletedProcess(cmd, 0, stdout="3.12.11\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="0.34.0\n", stderr="")
        return orig_run_vm_sub(cmd, **_kwargs)

    monkeypatch.setattr(vm_judges, "run_vm_subprocess", _patched_run)
    rt = vm_judges.collect_runtime_environment(work_dir=work_dir)
    assert rt["host_gpu_name"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert rt["driver_version"] == "580.95.05"
    assert rt["cuda_version"] == "13.0"
    assert rt["python_version"] == "3.12.11"
    assert rt["system_python_version"] == _platform.python_version()
    assert rt["vllm_version"] == "0.34.0"

    assert vm_judges._read_vllm_python_version(work_dir=None) is None

    import importlib.metadata as importlib_metadata

    monkeypatch.setattr(importlib_metadata, "version", lambda pkg: "0.34.0" if pkg == "vllm" else "0")
    assert vm_judges._read_vllm_version(work_dir=None) == "0.34.0"
    def _raise_missing(_pkg: str) -> str:
        raise importlib_metadata.PackageNotFoundError("vllm")
    monkeypatch.setattr(importlib_metadata, "version", _raise_missing)
    assert vm_judges._read_vllm_version(work_dir=None) is None

    receipt_built = vm_judges.build_vm_receipt_doc(
        mode="time-one",
        judge_download_records=[{"judge_id": "mu-judge", "seconds": 3.5}],
        judge_order=["mu-judge"],
        work_dir=work_dir,
        environ={
            "RDLAB_HOST": "runpod",
            "RDLAB_POD_ID": "pod-direct",
            "RDLAB_RUNPOD_IMAGE": offrig_host.DEFAULT_RUNPOD_IMAGE_REF,
            "RDLAB_CARD_TYPE": "G4",
        },
    )
    assert receipt_built["host"] == "runpod"
    assert receipt_built["pod_id"] == "pod-direct"
    assert receipt_built["model_download_seconds"] == pytest.approx(3.5)


def test_offrig_host_error_and_edge_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib
    import queue

    # GPU gate direct refusal branches and plain name without "nvidia " prefix
    assert offrig_host.normalize_gpu_name("RTX PRO 6000 Blackwell Server Edition") == "rtx pro 6000 blackwell server edition"
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_DISAGREEMENT"):
        offrig_host.verify_runpod_gpu_gate("NVIDIA L4", "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 580.1")
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_WORKSTATION_REFUSED"):
        offrig_host.verify_runpod_gpu_gate("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 580.1")
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_MISMATCH"):
        offrig_host.verify_runpod_gpu_gate("NVIDIA H100 PCIe", "NVIDIA H100 PCIe, 81559, 580.1")
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_VRAM_INSUFFICIENT"):
        offrig_host.verify_runpod_gpu_gate("NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA RTX PRO 6000 Blackwell Server Edition, 49152, 580.1")

    # resolve_offrig_mcp_bin via PATH
    path_bin_dir = tmp_path / "path_bins"
    path_bin_dir.mkdir(parents=True, exist_ok=True)
    path_exe = path_bin_dir / ("offrig-mcp.exe" if os.name == "nt" else "offrig-mcp")
    path_exe.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")
    path_exe.chmod(0o755)
    assert Path(offrig_host.resolve_offrig_mcp_bin({"PATH": str(path_bin_dir)})) == path_exe

    # Client start OSError & unstarted _send / close
    c_missing = offrig_host.OffrigMcpClient(project_dir=tmp_path, mcp_bin=str(tmp_path / "no-such-bin"))
    c_missing.close()
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_COMMAND_FAILED"):
        c_missing.start()
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_COMMAND_FAILED"):
        c_missing._send({"jsonrpc": "2.0"})

    # Client stdin OSError, stdout EOF, invalid JSON, timeout, non-dict result, and forced close
    class _BrokenStream:
        def write(self, _data: str) -> int:
            raise OSError("broken pipe")
        def flush(self) -> None:
            pass
        def close(self) -> None:
            raise OSError("already closed")

    class _TimeoutProc:
        def __init__(self) -> None:
            self.stdin = _BrokenStream()
            self.wait_calls = 0
        def poll(self) -> int:
            return 1
        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            if timeout is not None and self.wait_calls <= 2:
                raise subprocess.TimeoutExpired("fake", timeout)
            return 0
        def terminate(self) -> None:
            pass
        def kill(self) -> None:
            pass

    c_err = offrig_host.OffrigMcpClient(project_dir=tmp_path, mcp_bin=str(path_exe))
    c_err._proc = _TimeoutProc()  # type: ignore[assignment]
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_COMMAND_FAILED"):
        c_err._send({"jsonrpc": "2.0"})

    class _GoodStdin:
        def write(self, _s: str) -> int:
            return len(_s)
        def flush(self) -> None:
            pass
        def close(self) -> None:
            pass

    c_err._proc.stdin = _GoodStdin()  # type: ignore[assignment]
    c_err._stderr_lines = ["fatal startup error"]
    c_err._lines.put(None)
    with pytest.raises(offrig_host.OffrigHostError, match="closed stdout unexpectedly"):
        c_err._request("initialize", {}, timeout=1.0)

    c_err._lines = queue.Queue()
    c_err._lines.put("not-json\n")
    with pytest.raises(offrig_host.OffrigHostError, match="invalid JSON"):
        c_err._request("initialize", {}, timeout=1.0)

    c_err._lines = queue.Queue()
    c_err._lines.put("\n")
    c_err._lines.put(json.dumps({"jsonrpc": "2.0", "id": 999999, "result": {}}))
    c_err._lines.put(json.dumps({"jsonrpc": "2.0", "id": c_err._next_id, "result": ["not-a-dict"]}))
    with pytest.raises(offrig_host.OffrigHostError, match="non-object result"):
        c_err._request("initialize", {}, timeout=1.0)

    c_err._lines = queue.Queue()
    with pytest.raises(offrig_host.OffrigHostError, match="timed out"):
        c_err._request("initialize", {}, timeout=0.01)

    c_err._read_stderr_loop(["stderr-line\n"])
    assert "stderr-line" in c_err._stderr_lines
    c_err.close()

    # Fake client testing `plan_pod` missing plan_id, `launch_and_wait` poll -> missing pod_id & timeout
    class _FakeUnitClient:
        def __init__(self, responses: list[dict[str, Any]]) -> None:
            self._responses = list(responses)
        def __enter__(self) -> _FakeUnitClient:
            return self
        def __exit__(self, *_a: object) -> None:
            pass
        def call_tool(self, _name: str, _args: Any, **_kw: Any) -> dict[str, Any]:
            return self._responses.pop(0)

    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PLAN_FAILED"):
        offrig_host.plan_pod(_FakeUnitClient([{}]), profile="job", wall_seconds=60)  # type: ignore[arg-type]

    # Missing pod_id when done (with nested progress.rented)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_JOB_FAILED"):
        offrig_host.launch_and_wait(
            _FakeUnitClient(
                [
                    {"job_id": 1},
                    {"job_id": 1, "state": "running"},
                    {"job_id": 1, "state": "done", "progress": {"rented": {"gpu": "X"}}},
                ]
            ),  # type: ignore[arg-type]
            plan_id=1,
            poll_interval=0.01,
            timeout_seconds=5.0,
        )

    # Timed-out launch_and_wait
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_JOB_FAILED"):
        offrig_host.launch_and_wait(
            _FakeUnitClient([{"job_id": 1}, {"job_id": 1, "state": "running"}]),  # type: ignore[arg-type]
            plan_id=1,
            poll_interval=0.01,
            timeout_seconds=0.0,
        )

    # CLI `exec-run` stdout/stderr + timeout -> 124 and `exec-vm-step` log_tail without trailing newline
    monkeypatch.setattr(
        offrig_host,
        "OffrigMcpClient",
        lambda **_kw: _FakeUnitClient(
            [
                {"stdout": "out-msg\n", "stderr": "err-msg\n", "timed_out": True, "exit_code": None},
                {"started": True},
                {"status": {"state": "exited", "code": 0, "log_tail": "no-newline"}},
            ]
        ),
    )
    rc_to = offrig_host.main(["exec-run", "--project", str(tmp_path), "--plan-id", "1", "--command", "sleep 5"])
    assert rc_to == 124
    captured_run = capsys.readouterr()
    assert "out-msg" in captured_run.out and "err-msg" in captured_run.err

    rc_step = offrig_host.main(
        [
            "exec-vm-step",
            "--project",
            str(tmp_path),
            "--plan-id",
            "1",
            "--job-name",
            "s",
            "--vm-script",
            "/workspace/job/vm_judges.py",
        ]
    )
    assert rc_step == 0
    assert capsys.readouterr().out == "no-newline\n"
    importlib.reload(offrig_host)


@pytest.mark.parametrize("dry", [False, True], ids=["live", "dry"])
def test_runpod_refuses_stub_run_id_and_accepts_real_run_id(
    dry: bool,
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    env_bad = _setup_fake_offrig_env(tmp_path / "offrig_bad", repo_root=clean_repo_root)
    bad_args = [
        "--host",
        "runpod",
        "--time-one",
        "--run-id",
        "rdlab-d5c-qwen36-g4-stub-a4-23",
        "--items-file",
        _to_bash_path(items_path),
        "--part-dir",
        _to_bash_path(part_dir),
        "--receipts-dir",
        _to_bash_path(receipts_dir),
        "--out-dir",
        _to_bash_path(tmp_path / "out_bad"),
    ]
    if dry:
        bad_args.insert(0, "--dry")
    res_bad = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=bad_args,
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_bad["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_bad["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_bad["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_bad["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_bad["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_bad.returncode == 2
    assert "WINDOW_ID_STUB_REFUSED" in res_bad.stderr
    assert _read_call_log(env_bad["log_path"]) == []

    env_ok = _setup_fake_offrig_env(tmp_path / "offrig_ok", repo_root=clean_repo_root)
    ok_args = [
        "--host",
        "runpod",
        "--time-one",
        "--run-id",
        "rdlab-d5c-qwen36-runpod-time-a1",
        "--items-file",
        _to_bash_path(items_path),
        "--part-dir",
        _to_bash_path(part_dir),
        "--receipts-dir",
        _to_bash_path(receipts_dir),
        "--out-dir",
        _to_bash_path(tmp_path / "out_ok"),
    ]
    if dry:
        ok_args.insert(0, "--dry")
    res_ok = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=ok_args,
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_ok["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_ok["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_ok["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_ok["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_ok["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_ok.returncode == 0, f"stderr={res_ok.stderr}\nstdout={res_ok.stdout}"
    assert "WINDOW_ID_STUB_REFUSED" not in res_ok.stderr
    if not dry:
        assert "offrig_plan" in [c["tool"] for c in _read_call_log(env_ok["log_path"])]


def test_offrig_profile_image_verification_and_pre_plan_refusal(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    cfg_dir = tmp_path / "cfg_unit"
    assert offrig_host.resolve_offrig_config_path({"OFFRIG_CONFIG_DIR": str(cfg_dir)}) == cfg_dir / "config.toml"
    posix_cfg = offrig_host.resolve_offrig_config_path({"XDG_CONFIG_HOME": str(tmp_path / "xdg")})
    if not sys.platform.startswith(("win", "darwin")):
        assert posix_cfg == tmp_path / "xdg" / "offrig" / "config.toml"

    # Missing config.toml -> OFFRIG_CONFIG_UNREADABLE
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_CONFIG_UNREADABLE"):
        offrig_host.read_offrig_profile_image("rdlab-judge", config_path=cfg_dir / "config.toml")

    # Malformed TOML -> OFFRIG_CONFIG_UNREADABLE
    cfg_dir.mkdir(parents=True, exist_ok=True)
    bad_toml = cfg_dir / "bad.toml"
    bad_toml.write_text("[[profiles]\n", encoding="utf-8")
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_CONFIG_UNREADABLE"):
        offrig_host.read_offrig_profile_image("rdlab-judge", config_path=bad_toml)

    # Missing [profiles.job] or wrong profile -> OFFRIG_PROFILE_NOT_FOUND
    no_job = cfg_dir / "no_job.toml"
    no_job.write_text('[[profiles]]\nname = "rdlab-judge"\n', encoding="utf-8")
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PROFILE_NOT_FOUND"):
        offrig_host.read_offrig_profile_image("rdlab-judge", config_path=no_job)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PROFILE_NOT_FOUND"):
        offrig_host.read_offrig_profile_image("other-profile", config_path=no_job)

    # Mismatch -> OFFRIG_PROFILE_IMAGE_MISMATCH
    cfg_file = _write_offrig_config_toml(cfg_dir, image="vllm/vllm-openai:v0.33.0")
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PROFILE_IMAGE_MISMATCH"):
        offrig_host.verify_offrig_profile_image(
            "rdlab-judge",
            offrig_host.DEFAULT_RUNPOD_IMAGE_REF,
            config_path=cfg_file,
        )
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PROFILE_IMAGE_MISMATCH"):
        offrig_host.verify_offrig_profile_image("rdlab-judge", "", config_path=cfg_file)

    assert offrig_host.main(["check-profile-image", "--profile", "rdlab-judge", "--image", "vllm/vllm-openai:v0.33.0", "--config-path", str(cfg_file)]) == 0

    # Launcher refuses before `offrig_plan` when profile job.image mismatches requested image
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    env_mis = _setup_fake_offrig_env(
        tmp_path / "offrig_mis",
        repo_root=clean_repo_root,
        config_overrides={"profile_image": "vllm/vllm-openai:v0.31.0"},
    )
    out_mis = tmp_path / "out_mis"
    res_mis = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-img-mis-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_mis),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_mis["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_mis["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_mis["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_mis["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_mis["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res_mis.returncode == 2
    assert "OFFRIG_PROFILE_IMAGE_MISMATCH" in res_mis.stderr
    assert _read_call_log(env_mis["log_path"]) == []


def test_runpod_deadline_covers_window_and_fake_clock_bounds_run_keeping_fetch_before_deadline(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    assert offrig_host.DEFAULT_LAUNCH_WAIT_MINUTES == 20
    assert offrig_host.DEFAULT_IMAGE_PULL_SETUP_SECONDS == 900
    assert offrig_host.DEFAULT_FETCH_ALLOWANCE_SECONDS == 180
    assert offrig_host.DEFAULT_SHUTDOWN_RESERVE_SECONDS == 120
    assert offrig_host.compute_watchdog_max_hours(1500) == pytest.approx(1.0834)
    assert offrig_host.parse_offrig_deadline_utc("2026-11-23T13:05:00Z") == pytest.approx(1795439100.0)
    assert offrig_host.parse_offrig_deadline_utc("2026-11-23T13:05:00") == pytest.approx(1795439100.0)

    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    # 1. Slow setup consumes 3000s of 3900s deadline -> 900s before deadline -> 600s run budget (540s VM wall), fetch+shutdown still finish before deadline
    t0 = 1795435200  # 2026-11-23 12:00 UTC
    clock_file = tmp_path / "fake_clock.txt"
    clock_file.write_text(f"{t0}\n", encoding="utf-8")
    env_slow = _setup_fake_offrig_env(
        tmp_path / "slow_setup",
        repo_root=clean_repo_root,
        config_overrides={"advance_clock_on_step": {"gate": 3000}},
    )
    out_slow = tmp_path / "out_slow"
    res_slow = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-slow-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_slow),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_slow["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_slow["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_slow["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_slow["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_slow["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_slow.returncode == 0, f"stderr={res_slow.stderr}\nstdout={res_slow.stdout}"
    run_env_doc = json.loads((env_slow["vm_root_dir"] / "run_window_env.json").read_text(encoding="utf-8"))
    assert run_env_doc["env"]["RDLAB_WALL_SECONDS"] == "540"
    cli_log_txt = (out_slow / "colab-cli.log").read_text(encoding="utf-8")
    assert "RUNPOD_RUN_BOUNDED_BY_DEADLINE" in cli_log_txt
    tools_slow = [c["tool"] for c in _read_call_log(env_slow["log_path"])]
    assert "offrig_get" in tools_slow and tools_slow[-1] == "offrig_shutdown"
    assert (out_slow / "vm_receipt.json").is_file()
    final_clock = int(clock_file.read_text(encoding="utf-8").strip())
    deadline_ep = offrig_host.parse_offrig_deadline_utc("2026-11-23 13:05 UTC")
    assert final_clock < deadline_ep

    # 2. Setup consumes entire margin beyond reserve -> skips `run`, runs fetch & shutdown with RUNPOD_DEADLINE_EXHAUSTED
    clock_ex = tmp_path / "fake_clock_ex.txt"
    clock_ex.write_text(f"{t0}\n", encoding="utf-8")
    env_ex = _setup_fake_offrig_env(
        tmp_path / "ex_setup",
        repo_root=clean_repo_root,
        config_overrides={"advance_clock_on_step": {"gate": 3700}},
    )
    out_ex = tmp_path / "out_ex"
    res_ex = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-exhausted-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_ex),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_ex["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_ex),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_ex["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_ex["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_ex["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_ex["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_ex.returncode == 2
    assert "RUNPOD_DEADLINE_EXHAUSTED_BEFORE_RUN" in (out_ex / "colab-cli.log").read_text(encoding="utf-8")
    assert not (env_ex["vm_root_dir"] / "run_window_env.json").exists()
    assert json.loads(env_ex["state_path"].read_text(encoding="utf-8"))["shutdown"] is True
    assert json.loads((out_ex / "outcome.draft.json").read_text(encoding="utf-8"))["failure_class"] == "RUNPOD_DEADLINE_EXHAUSTED"


@pytest.mark.parametrize(
    ("case_label", "cfg_overrides", "extra_env", "expected_code"),
    [
        (
            "client_timeout",
            {"sleep_on_tool": "offrig_launch", "sleep_seconds": 0.8},
            {"RDLAB_OFFRIG_LAUNCH_TIMEOUT_SECONDS": "0.15"},
            "OFFRIG_MCP_TIMEOUT",
        ),
        (
            "sidecar_crash",
            {"crash_on_tool": "offrig_launch"},
            {},
            "OFFRIG_MCP_EOF",
        ),
        (
            "bad_json",
            {"bad_json_on_tool": "offrig_launch"},
            {},
            "OFFRIG_MCP_BAD_JSON",
        ),
        (
            "job_poll_failure",
            {"job_state": "failed", "job_error": "capacity lost during wait"},
            {},
            "OFFRIG_JOB_FAILED",
        ),
        (
            "watchdog_start_failed",
            {"watchdog": "FAILED to start: fork denied; shut down by the deadline yourself"},
            {},
            "OFFRIG_WATCHDOG_START_FAILED",
        ),
    ],
)
def test_runpod_shutdown_called_on_every_launch_failure_class(
    case_label: str,
    cfg_overrides: dict[str, Any],
    extra_env: dict[str, str],
    expected_code: str,
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    env_info = _setup_fake_offrig_env(
        tmp_path / f"launch_{case_label}",
        repo_root=clean_repo_root,
        config_overrides=cfg_overrides,
    )
    out_dir = tmp_path / f"out_{case_label}"
    env_vars = {
        "OFFRIG_CONFIG_DIR": _to_bash_path(env_info["offrig_config_dir"]),
        "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_info["shim_sh"]),
        "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
        "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_info["cfg_path"]),
        "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_info["log_path"]),
        "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_info["state_path"]),
        "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        **extra_env,
    }
    res = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            f"rp-lf-{case_label}",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_dir),
        ],
        env_vars=env_vars,
    )
    assert res.returncode == 2
    assert expected_code in res.stderr
    calls = _read_call_log(env_info["log_path"])
    shutdown_calls = [c for c in calls if c["tool"] == "offrig_shutdown"]
    assert shutdown_calls, calls
    assert all(int(c["arguments"]["plan_id"]) == 101 for c in shutdown_calls)
    assert json.loads(env_info["state_path"].read_text(encoding="utf-8"))["shutdown"] is True
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["pod_shutdown"] is True
    assert outcome["failure_class"] == expected_code


def test_runpod_plan_gpu_list_gate_and_probe_gpu_gate_and_status_poll_retries(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    # 1. `verify_plan_gpu_list` refuses Workstation Edition & unknown/empty GPU list
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_PLAN_WORKSTATION_REFUSED"):
        offrig_host.verify_plan_gpu_list(
            {
                "plan_id": 1,
                "gpus": "1x NVIDIA RTX PRO 6000 Blackwell Server Edition | NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            }
        )
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_PLAN_GPU_MISMATCH"):
        offrig_host.verify_plan_gpu_list({"plan_id": 1, "gpu_types": ["NVIDIA A100-SXM4-80GB"]})
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_PLAN_GPU_MISMATCH"):
        offrig_host.verify_plan_gpu_list({"plan_id": 1})

    # Launcher refuses plan with Workstation Edition BEFORE `offrig_launch`
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    env_ws_plan = _setup_fake_offrig_env(
        tmp_path / "ws_plan",
        repo_root=clean_repo_root,
        config_overrides={
            "plan_gpus": "1x NVIDIA RTX PRO 6000 Blackwell Server Edition | NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        },
    )
    out_ws_plan = tmp_path / "out_ws_plan"
    res_ws_plan = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-ws-plan-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_ws_plan),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_ws_plan["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_ws_plan["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_ws_plan["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_ws_plan["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_ws_plan["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res_ws_plan.returncode == 2
    assert "RUNPOD_PLAN_WORKSTATION_REFUSED" in res_ws_plan.stderr
    assert [c["tool"] for c in _read_call_log(env_ws_plan["log_path"])] == ["offrig_plan"]

    # 2. Probe path runs `verify-gpu`, refuses Workstation Edition, and shuts down pod
    env_probe_ws = _setup_fake_offrig_env(
        tmp_path / "probe_ws",
        repo_root=clean_repo_root,
        config_overrides={
            "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "plan_gpus": "1x NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "nvidia_smi_stdout": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 580.95.05\n",
        },
    )
    # Use matching offrig GPU Workstation on post-launch metadata as well so RUNPOD_GPU_WORKSTATION_REFUSED triggers
    cfg_ws = json.loads(env_probe_ws["cfg_path"].read_text(encoding="utf-8"))
    cfg_ws["gpu"] = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    cfg_ws["plan_gpus"] = "1x NVIDIA RTX PRO 6000 Blackwell Server Edition"
    env_probe_ws["cfg_path"].write_text(json.dumps(cfg_ws, indent=2) + "\n", encoding="utf-8")
    out_probe_ws = tmp_path / "out_probe_ws"
    res_probe_ws = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--probe", "--run-id", "rp-probe-ws-01", "--out-dir", _to_bash_path(out_probe_ws)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_probe_ws["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_probe_ws["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_probe_ws["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_probe_ws["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_probe_ws["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_probe_ws.returncode == 2
    assert "RUNPOD_GPU_WORKSTATION_REFUSED" in res_probe_ws.stderr
    assert json.loads(env_probe_ws["state_path"].read_text(encoding="utf-8"))["shutdown"] is True

    # 3. `exec_vm_step` status poll retries transiently then recovers, and fails once exhausted
    class _FlakyStatusClient:
        def __init__(self, fail_times: int) -> None:
            self.fail_times = fail_times
            self.status_calls = 0

        def call_tool(self, _tool: str, args: dict[str, Any], **_kw: Any) -> dict[str, Any]:
            if args.get("action") == "start":
                return {"started": True}
            self.status_calls += 1
            if self.status_calls <= self.fail_times:
                raise offrig_host.OffrigHostError("OFFRIG_MCP_TIMEOUT", "transient status poll timeout")
            return {"status": {"state": "exited", "code": 0, "log_tail": "OK\n"}}

    flaky_ok = _FlakyStatusClient(fail_times=2)
    rc_flaky, tail_flaky = offrig_host.exec_vm_step(
        flaky_ok,  # type: ignore[arg-type]
        plan_id=1,
        job_name="run",
        vm_script="/workspace/job/vm_judges.py",
        env_pairs=["RDLAB_STEP=run"],
        timeout_seconds=30,
        poll_interval=0.01,
        status_poll_retries=3,
    )
    assert rc_flaky == 0 and tail_flaky == "OK\n" and flaky_ok.status_calls == 3

    flaky_fail = _FlakyStatusClient(fail_times=5)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_MCP_TIMEOUT"):
        offrig_host.exec_vm_step(
            flaky_fail,  # type: ignore[arg-type]
            plan_id=1,
            job_name="run",
            vm_script="/workspace/job/vm_judges.py",
            env_pairs=["RDLAB_STEP=run"],
            timeout_seconds=30,
            poll_interval=0.01,
            status_poll_retries=2,
        )
    assert flaky_fail.status_calls == 3


def test_runpod_card_type_pre_plan_goldens_refusal_path_visibility_and_sde_scope(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    # 1. Pre-plan goldens refusal on RunPod counted AND lock-check even when Colab `G4` goldens are promoted
    goldens_doc = json.loads((clean_repo_root / "scripts" / "colab" / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    goldens_doc["card_types"]["G4"] = {
        "status": "filled",
        "prompts": {sp["prompt_id"]: {"output_token_ids_sha256": "c" * 64} for sp in goldens_doc["synthetic_prompts"]},
    }
    goldens_doc["promoted_provenance"] = {
        "window_id": "rdlab-g4-prom-01",
        "session_env_sha256": "a" * 64,
        "session_fingerprint_sha256": "b" * 64,
        "mode": "lock-check",
        "lock_check_phase": "anchors",
        "card_type": "G4",
        "hash_source": "real_transport",
        "stub_transport": False,
    }
    goldens_doc["goldens_body_sha256"] = vm_judges.compute_goldens_body_sha256(goldens_doc)
    g4_goldens_path = tmp_path / "g4_only_goldens.json"
    g4_goldens_path.write_text(json.dumps(goldens_doc), encoding="utf-8")

    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    for mode_name, extra_mode_args in (
        ("counted", ["--counted", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir)]),
        (
            "lock-check",
            [
                "--lock-check",
                "--lock-check-phase",
                "anchors",
                "--anchor-items",
                _to_bash_path(items_path),
                "--p0a-items",
                _to_bash_path(items_path),
                "--p0a-part-dir",
                _to_bash_path(part_dir),
                "--counted-items",
                _to_bash_path(items_path),
            ],
        ),
    ):
        env_g = _setup_fake_offrig_env(tmp_path / f"gold_{mode_name}", repo_root=clean_repo_root)
        res_g = _run_bash_launcher(
            repo_root=clean_repo_root,
            args=[
                "--host",
                "runpod",
                *extra_mode_args,
                "--run-id",
                f"rp-no-golden-{mode_name}",
                "--receipts-dir",
                _to_bash_path(receipts_dir),
                "--boot-gatekeeper-goldens",
                _to_bash_path(g4_goldens_path),
                "--out-dir",
                _to_bash_path(tmp_path / f"out_gold_{mode_name}"),
            ],
            env_vars={
                "OFFRIG_CONFIG_DIR": _to_bash_path(env_g["offrig_config_dir"]),
                "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_g["shim_sh"]),
                "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
                "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_g["cfg_path"]),
                "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_g["log_path"]),
                "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_g["state_path"]),
                "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            },
        )
        assert res_g.returncode == 2
        assert "BOOT_GATEKEEPER_GOLDEN_MISSING" in res_g.stderr
        assert vm_judges.RUNPOD_SERVER_CARD_TYPE in res_g.stderr
        assert _read_call_log(env_g["log_path"]) == []

    # 2. Path visibility preflight: Windows `.exe` binary given POSIX paths refused before `offrig_plan`
    fake_exe = tmp_path / "offrig-mcp.exe"
    fake_exe.write_bytes(b"MZ\x90\x00" + b"\x00" * 60)
    fake_exe.chmod(0o755)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PATH_INCOMPATIBLE"):
        offrig_host.check_offrig_binary_path_visibility(fake_exe, "/workspace/rdlab")
    env_vis = _setup_fake_offrig_env(tmp_path / "vis", repo_root=clean_repo_root)
    res_vis = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-vis-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(tmp_path / "out_vis"),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_vis["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(fake_exe),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_vis["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_vis["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_vis["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res_vis.returncode == 2
    assert "OFFRIG_PATH_INCOMPATIBLE" in res_vis.stderr
    assert _read_call_log(env_vis["log_path"]) == []

    # 3. `SOURCE_DATE_EPOCH` is scoped to parity (`--dry` + `RDLAB_DRY_STAGE=1`), so non-dry staging never freezes timestamps
    part_dir_bash = _to_bash_path(part_dir)
    # Extract `stage_window_artifacts` inline Python invocation behaviour via clean live `launch_judges.sh --stub` with SOURCE_DATE_EPOCH
    fake_bin_dir = tmp_path / "colab_fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_colab_exe = fake_bin_dir / "colab"
    fake_colab_exe.write_text("#!/usr/bin/env bash\necho '[]'\nexit 0\n", encoding="utf-8", newline="\n")
    fake_colab_exe.chmod(0o755)
    out_sde = tmp_path / "out_sde_live"
    sde_bash = (
        f"export PATH={shlex.quote(_to_bash_path(fake_bin_dir) + ':/usr/bin:/bin')}; "
        f"export RDLAB_JUDGE_LOCK_SHA256={shlex.quote(vm_judges.EXPECTED_JUDGE_LOCK_SHA256)}; "
        f"export RDLAB_D3_LOCK_SHA256={shlex.quote(vm_judges.EXPECTED_D3_LOCK_SHA256)}; "
        f"export SOURCE_DATE_EPOCH=1700000000; export RDLAB_DRY_STAGE=1; "
        f"bash {shlex.quote(_to_bash_path(clean_repo_root / 'scripts' / 'colab' / 'launch_judges.sh'))} "
        f"--stub --run-id sde-live-chk --items-file {shlex.quote(_to_bash_path(items_path))} "
        f"--part-dir {shlex.quote(part_dir_bash)} --receipts-dir {shlex.quote(_to_bash_path(receipts_dir))} "
        f"--out-dir {shlex.quote(_to_bash_path(out_sde))}"
    )
    subprocess.run(["bash", "-c", sde_bash], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    rig_binding = out_sde / "launcher_part_binding.json"
    assert rig_binding.is_file()
    assert json.loads(rig_binding.read_text(encoding="utf-8"))["checked_at"] != "2023-11-14T22:13:20Z"


def test_offrig_host_and_vm_judges_defensive_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover defensive parsing, config-table, watchdog-abort, poll-deadline, and GPU branches."""
    with pytest.raises(offrig_host.OffrigHostError) as exc_empty:
        offrig_host.parse_offrig_deadline_utc("   ")
    assert exc_empty.value.code == "OFFRIG_DEADLINE_INVALID"

    with pytest.raises(offrig_host.OffrigHostError) as exc_bad_utc:
        offrig_host.parse_offrig_deadline_utc("2026-11-99 99:99 UTC")
    assert exc_bad_utc.value.code == "OFFRIG_DEADLINE_INVALID"

    win_appdata = tmp_path / "win_appdata"
    monkeypatch.setattr(offrig_host.sys, "platform", "win32")
    assert (
        offrig_host.resolve_offrig_config_path(environ={"APPDATA": str(win_appdata)})
        == win_appdata / "offrig" / "config.toml"
    )
    monkeypatch.setattr(offrig_host.sys, "platform", "darwin")
    mac_cfg = offrig_host.resolve_offrig_config_path(environ={})
    assert mac_cfg.parts[-4:] == ("Library", "Application Support", "offrig", "config.toml")
    monkeypatch.setattr(offrig_host.sys, "platform", "linux")
    xdg_cfg = tmp_path / "xdg_cfg"
    assert (
        offrig_host.resolve_offrig_config_path(environ={"XDG_CONFIG_HOME": str(xdg_cfg)})
        == xdg_cfg / "offrig" / "config.toml"
    )

    expected_img = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    dict_cfg = tmp_path / "dict_profiles.toml"
    dict_cfg.write_text(
        f'[profiles.judge-host]\n[profiles.judge-host.job]\nimage = "{expected_img}"\n',
        encoding="utf-8",
    )
    verified_img = offrig_host.verify_offrig_profile_image(
        "judge-host", expected_img, config_path=dict_cfg
    )
    assert verified_img["requested_image"] == expected_img
    assert verified_img["profile_image"] == expected_img
    assert verified_img["source"] == "offrig profile"
    assert verified_img["match"] is True

    missing_bin_res = offrig_host.check_offrig_binary_path_visibility(
        tmp_path / "nonexistent-offrig",
        str(tmp_path / "sample.jsonl"),
    )
    assert missing_bin_res["compatible"] is True
    assert missing_bin_res["windows_binary"] is False

    fake_exe = tmp_path / "offrig.exe"
    fake_exe.write_bytes(b"MZ")
    with pytest.raises(offrig_host.OffrigHostError) as exc_empty_sample:
        offrig_host.check_offrig_binary_path_visibility(fake_exe, "")
    assert exc_empty_sample.value.code == "OFFRIG_PATH_INCOMPATIBLE"

    single_gpu = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    plan_gpu_summary = offrig_host.verify_plan_gpu_list({"plan_id": 1, "gpu": single_gpu})
    assert single_gpu in plan_gpu_summary["planned_gpus"]
    assert plan_gpu_summary["verified"] is True

    class _FakeWatchdogClient:
        def __init__(self, *, fail_shutdown: bool) -> None:
            self.fail_shutdown = fail_shutdown
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def call_tool(self, tool_name: str, arguments: dict[str, Any], *, timeout: float = 120.0) -> dict[str, Any]:
            self.calls.append((tool_name, dict(arguments)))
            if tool_name == "offrig_launch":
                return {
                    "plan_id": int(arguments["plan_id"]),
                    "status": "running",
                    "watchdog": "FAILED to start: simulated",
                }
            if tool_name == "offrig_status":
                return {"status": "running", "state": "running"}
            if tool_name == "offrig_shutdown":
                if self.fail_shutdown:
                    raise offrig_host.OffrigHostError("OFFRIG_MCP_TIMEOUT", "shutdown timeout")
                return {"ok": True}
            if tool_name == "offrig_exec":
                if arguments.get("action") == "status":
                    fake_now[0] += 5.0
                    raise offrig_host.OffrigHostError("OFFRIG_MCP_TIMEOUT", "job poll timeout")
                return {"job_id": "job-99"}
            raise AssertionError(f"unexpected tool {tool_name}")

    fake_now = [100.0]
    for fail_shutdown in (False, True):
        client = _FakeWatchdogClient(fail_shutdown=fail_shutdown)
        with pytest.raises(offrig_host.OffrigHostError) as exc_wd:
            offrig_host.launch_and_wait(client, plan_id=11, poll_interval=0.0, timeout_seconds=5.0)
        assert exc_wd.value.code == "OFFRIG_WATCHDOG_START_FAILED"
        assert any(name == "offrig_shutdown" and args["plan_id"] == 11 for name, args in client.calls)

    exec_client = _FakeWatchdogClient(fail_shutdown=False)
    monkeypatch.setattr(offrig_host.time, "monotonic", lambda: fake_now[0])
    rc_step, tail_step = offrig_host.exec_vm_step(
        exec_client,
        plan_id=11,
        job_name="step-deadline",
        vm_script="scripts/colab/run_judges.sh",
        env_pairs=(),
        timeout_seconds=1,
        poll_interval=0.0,
        status_poll_retries=5,
    )
    assert rc_step == 124
    assert tail_step == ""
    assert any(
        name == "offrig_exec" and args.get("action") == "stop"
        for name, args in exec_client.calls
    )

    assert vm_judges.card_type_from_gpu_name("NVIDIA L4", host="runpod") is None

