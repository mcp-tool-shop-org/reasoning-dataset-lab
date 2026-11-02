#!/usr/bin/env python3
"""Contract-checked fake Colab CLI for offline judge launcher tests.

Enforces the recorded Colab CLI contract against `colab_help.txt` and
`colab-cli-contract.json`:
- subcommands and flags checked against saved `colab --help` output;
- session row shape `[<SESSION>] <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN>`;
- empty-list line `[colab] No active sessions found on server.`;
- records CLI call log (`calls.jsonl`) and file access order (`file_access.jsonl`).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = FIXTURES_DIR / "colab-cli-contract.json"
HELP_FIXTURE_PATH = FIXTURES_DIR / "colab_help.txt"

SESSION_ROW_SHAPE = (
    "[<SESSION>] <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN>"
)
EMPTY_SESSIONS_LINE = "[colab] No active sessions found on server."
DEFAULT_SESSION_ROW_TEMPLATE = (
    "[{sid}] m-s-fake-{sid} | hw: {gpu} | status: READY | age: 1m"
)

GLOBAL_VALUE_FLAGS: frozenset[str] = frozenset(
    {"--auth", "--client-oauth-config", "-c", "--config"}
)
VALUE_OPTS_BY_CMD: dict[str, frozenset[str]] = {
    "new": frozenset({"-s", "--session", "--gpu", "--tpu"}),
    "exec": frozenset(
        {"-s", "--session", "-f", "--file", "--timeout", "--env", "--output-image"}
    ),
    "upload": frozenset({"-s", "--session"}),
    "download": frozenset({"-s", "--session"}),
    "stop": frozenset({"-s", "--session"}),
    "sessions": frozenset(),
    "usage": frozenset(),
    "status": frozenset({"-s", "--session"}),
}


def mask_session_line_shape(line: str, target_session: str = "") -> str:
    """Replace alphanumeric tokens with `<SESSION>` (if equal to target_session) or `<TOKEN>`."""

    def _repl(match: re.Match[str]) -> str:
        tok = match.group(0)
        if target_session and tok == target_session:
            return "<SESSION>"
        return "<TOKEN>"

    return re.sub(r"[A-Za-z0-9_.-]+", _repl, line.strip())


def parse_help_fixture(help_text: str) -> dict[str, set[str]]:
    """Parse subcommand sections and options from saved `colab --help` fixture text."""
    sections: dict[str, str] = {}
    current_name = "top"
    buf: list[str] = []
    for line in help_text.splitlines():
        m = re.match(r"^=== colab(?:\s+([a-z0-9-]+))?\s+--help ===$", line.strip())
        if m:
            sections[current_name] = "\n".join(buf)
            current_name = m.group(1) or "top"
            buf = []
        else:
            buf.append(line)
    sections[current_name] = "\n".join(buf)

    parsed: dict[str, set[str]] = {}
    for sec_name, body in sections.items():
        opts = set(re.findall(r"(?<!\w)(--[a-z0-9-]+|-[a-zA-Z])(?!\w)", body))
        parsed[sec_name] = opts
    return parsed


def load_and_verify_contract(
    contract_path: Path = CONTRACT_PATH,
    help_path: Path = HELP_FIXTURE_PATH,
) -> dict[str, Any]:
    """Load `colab-cli-contract.json` and verify all subcommands and flags against `colab_help.txt`."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    help_text = help_path.read_text(encoding="utf-8")
    parsed_help = parse_help_fixture(help_text)

    if contract.get("session_row_shape") != SESSION_ROW_SHAPE:
        raise ValueError(
            f"Contract session_row_shape mismatch: {contract.get('session_row_shape')!r}"
        )
    if contract.get("empty_sessions_line") != EMPTY_SESSIONS_LINE:
        raise ValueError(
            f"Contract empty_sessions_line mismatch: {contract.get('empty_sessions_line')!r}"
        )

    top_opts = parsed_help.get("top", set())
    for gopt in contract.get("global_options", []):
        if gopt not in top_opts:
            raise ValueError(f"Global option {gopt!r} missing from colab --help fixture")

    for subcmd, spec in contract.get("commands", {}).items():
        if subcmd not in parsed_help:
            raise ValueError(f"Subcommand {subcmd!r} missing from colab_help.txt fixture")
        sub_opts = parsed_help[subcmd]
        for opt in spec.get("options", []):
            if opt not in sub_opts:
                raise ValueError(
                    f"Option {opt!r} for subcommand {subcmd!r} missing from colab_help.txt"
                )
    return contract


def record_file_access(
    state_dir: Path,
    event: str,
    path: str | Path,
    *,
    detail: str = "",
) -> None:
    """Append a timestamped file-access event to `file_access.jsonl` in `state_dir`."""
    state_dir.mkdir(parents=True, exist_ok=True)
    access_log = state_dir / "file_access.jsonl"
    entry = {
        "event": event,
        "path": str(path).replace("\\", "/"),
        "detail": detail,
        "ts": time.time(),
    }
    with access_log.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry) + "\n")


def read_file_access_log(state_dir: Path) -> list[dict[str, Any]]:
    """Read recorded file-access events in order from `state_dir/file_access.jsonl`."""
    access_log = state_dir / "file_access.jsonl"
    if not access_log.is_file():
        return []
    return [
        json.loads(line)
        for line in access_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def setup_fake_colab_environment(
    base_dir: Path,
    *,
    neighbour_sessions: list[dict[str, str]] | None = None,
) -> dict[str, Path]:
    """Create a front-of-PATH fake `colab` wrapper and a sentinel real-colab trap in `base_dir`."""
    fake_bin = base_dir / "fake_bin"
    sentinel_bin = base_dir / "sentinel_real_bin"
    state_dir = base_dir / "fake_colab_state"
    colab_locks_dir = base_dir / "colab_locks"
    fake_bin.mkdir(parents=True, exist_ok=True)
    sentinel_bin.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    colab_locks_dir.mkdir(parents=True, exist_ok=True)

    initial_sessions = (
        neighbour_sessions
        if neighbour_sessions is not None
        else [{"session_id": "sess-neighbour-keep", "gpu": "T4"}]
    )
    sessions_file = state_dir / "sessions.json"
    sessions_file.write_text(
        json.dumps(initial_sessions),
        encoding="utf-8",
        newline="\n",
    )

    sentinel_called = state_dir / "REAL_COLAB_CALLED"
    sentinel_script = sentinel_bin / "colab"
    sentinel_script.write_text(
        f"#!/usr/bin/env bash\ntouch '{_to_posix(sentinel_called)}'\nexit 99\n",
        encoding="utf-8",
        newline="\n",
    )
    sentinel_script.chmod(sentinel_script.stat().st_mode | stat.S_IEXEC)

    fake_colab_py_posix = _to_posix(Path(__file__).resolve())
    fake_colab_sh = fake_bin / "colab"
    fake_colab_sh.write_text(
        "#!/usr/bin/env bash\n"
        "export RDLAB_COLAB_IS_FAKE=1\n"
        f"exec python3 '{fake_colab_py_posix}' \"$@\"\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_colab_sh.chmod(fake_colab_sh.stat().st_mode | stat.S_IEXEC)

    return {
        "fake_bin": fake_bin,
        "sentinel_bin": sentinel_bin,
        "state_dir": state_dir,
        "colab_locks_dir": colab_locks_dir,
        "sentinel_called": sentinel_called,
        "calls_log": state_dir / "calls.jsonl",
        "file_access_log": state_dir / "file_access.jsonl",
        "sessions_file": sessions_file,
        "stopped_file": state_dir / "stopped_sessions.jsonl",
    }


def _to_posix(path: Path) -> str:
    return path.resolve().as_posix()


def _check_stdin_is_devnull() -> bool:
    try:
        return os.path.samefile("/proc/self/fd/0", "/dev/null")
    except Exception:
        try:
            s0 = os.fstat(0)
            sn = os.stat("/dev/null")
            return bool(s0.st_rdev == sn.st_rdev and s0.st_rdev != 0)
        except Exception:
            return False


def main(argv: list[str] | None = None) -> int:
    os.environ["RDLAB_COLAB_IS_FAKE"] = "1"
    contract = load_and_verify_contract()
    commands_contract: dict[str, Any] = contract["commands"]
    global_options: set[str] = set(contract.get("global_options", []))

    state_dir_raw = os.environ.get("FAKE_COLAB_STATE_DIR")
    state_dir = Path(state_dir_raw) if state_dir_raw else Path("/tmp/rdlab-fake-colab-state")
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "is_fake_cli").write_text("1\n", encoding="utf-8", newline="\n")

    vm_fs = state_dir / "vm_fs"
    vm_fs.mkdir(parents=True, exist_ok=True)
    calls_log = state_dir / "calls.jsonl"
    sessions_file = state_dir / "sessions.json"
    if not sessions_file.is_file():
        sessions_file.write_text("[]\n", encoding="utf-8", newline="\n")

    lock_file = Path(os.environ.get("RDLAB_OPEN_LOCK", "/tmp/colab-window-open.lock"))
    reconcile_lock_dir = Path(os.environ.get("COLAB_LOCK_DIR", "/tmp/colab-locks"))

    raw_args = list(sys.argv[1:] if argv is None else argv)

    idx = 0
    top_help_requested = False
    while idx < len(raw_args) and raw_args[idx].startswith("-"):
        gopt = raw_args[idx].split("=", 1)[0]
        if gopt not in global_options:
            sys.stderr.write(f"No such global option: {raw_args[idx]}\n")
            return 2
        if gopt in ("-h", "--help"):
            top_help_requested = True
        if gopt in GLOBAL_VALUE_FLAGS and "=" not in raw_args[idx]:
            idx += 2
        else:
            idx += 1

    if top_help_requested and idx >= len(raw_args):
        print("Usage: colab [OPTIONS] COMMAND [ARGS]...")
        print(f"Options: {' '.join(sorted(global_options))}")
        print(f"Commands: {' '.join(sorted(commands_contract.keys()))}")
        return 0

    if idx >= len(raw_args):
        sys.stderr.write("Missing command.\n")
        return 2

    subcmd = raw_args[idx]
    args = raw_args[idx + 1 :]

    if subcmd == "version":
        if args:
            sys.stderr.write(f"Got unexpected extra arguments ({' '.join(args)})\n")
            return 2
        print(f"Version: {contract['version'].lstrip('v')}")
        return 0

    if subcmd not in commands_contract:
        sys.stderr.write(
            f"No such command '{subcmd}' in {contract['cli']} {contract['version']}.\n"
        )
        return 2

    allowed_opts = set(commands_contract[subcmd]["options"])
    value_opts = VALUE_OPTS_BY_CMD.get(subcmd, frozenset())

    parsed_opts: dict[str, list[str]] = {}
    positionals: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--":
            sys.stderr.write(
                f"colab {subcmd} --: not an option of {contract['cli']} {contract['version']} "
                "(trailing shell commands are rejected)\n"
            )
            return 2
        if tok.startswith("-") and len(tok) > 1:
            if "=" in tok:
                opt_name, opt_val = tok.split("=", 1)
                if opt_name not in allowed_opts:
                    sys.stderr.write(
                        f"colab {subcmd} {opt_name}: not an option of {contract['cli']} {contract['version']} "
                        f"(allowed: {' '.join(sorted(allowed_opts))})\n"
                    )
                    return 2
                parsed_opts.setdefault(opt_name, []).append(opt_val)
                i += 1
            else:
                opt_name = tok
                if opt_name not in allowed_opts:
                    sys.stderr.write(
                        f"colab {subcmd} {opt_name}: not an option of {contract['cli']} {contract['version']} "
                        f"(allowed: {' '.join(sorted(allowed_opts))})\n"
                    )
                    return 2
                if opt_name in value_opts:
                    if i + 1 >= len(args):
                        sys.stderr.write(f"Option '{opt_name}' requires an argument.\n")
                        return 2
                    parsed_opts.setdefault(opt_name, []).append(args[i + 1])
                    i += 2
                else:
                    parsed_opts.setdefault(opt_name, []).append("true")
                    i += 1
        else:
            positionals.append(tok)
            i += 1

    if "-h" in parsed_opts or "--help" in parsed_opts:
        print(commands_contract[subcmd]["usage"])
        print(f"Options: {' '.join(sorted(allowed_opts))}")
        return 0

    def _get_opt(*names: str) -> str | None:
        for n in names:
            vals = parsed_opts.get(n)
            if vals:
                return vals[-1]
        return None

    lock_held = False
    if lock_file.exists():
        try:
            res = subprocess.run(["flock", "-n", str(lock_file), "true"], check=False)
            lock_held = res.returncode != 0
        except Exception:
            lock_held = False

    reconcile_locks_present = (
        sorted(p.name for p in reconcile_lock_dir.glob("*.lock"))
        if reconcile_lock_dir.is_dir()
        else []
    )
    stdin_is_devnull = _check_stdin_is_devnull()

    if os.environ.get("FAKE_COLAB_READ_STDIN") == "1":
        _ = sys.stdin.read()

    record = {
        "subcmd": subcmd,
        "args": args,
        "lock_held": lock_held,
        "reconcile_locks": reconcile_locks_present,
        "stdin_is_devnull": stdin_is_devnull,
        "ts": time.time(),
    }
    with calls_log.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record) + "\n")

    # Emit synthetic tokens on stderr to test log scrubbing
    hf_prefix = "hf" + "_"
    jwt_prefix = "eyJ" + "hbGciOiJIUzI1NiJ9." + "eyJ" + "zdWIiOiJmYWtlIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    sys.stderr.write(
        "debug proxy https://colab.research.google.com/?colab-runtime-proxy-token=FAKE_PROXY_SECRET_777 "
        "auth=Bearer FAKE_BEARER_SECRET_888 oauth=ya29.FAKE_OAUTH_999 "
        f"jwt={jwt_prefix} "
        f"hf={hf_prefix}FakeSecretTokenValue1234567890\n"
    )

    sessions: list[dict[str, str]] = json.loads(sessions_file.read_text(encoding="utf-8"))

    if subcmd == "sessions":
        if positionals:
            sys.stderr.write(
                f"Got unexpected extra arguments ({' '.join(positionals)})\n"
            )
            return 2
        override_out = os.environ.get("FAKE_COLAB_SESSIONS_OVERRIDE")
        if override_out is not None:
            print(override_out)
            return 0
        prefix_line = os.environ.get("FAKE_COLAB_SESSIONS_PREFIX_LINE")
        if not sessions:
            print(prefix_line if prefix_line is not None else EMPTY_SESSIONS_LINE)
            return 0
        if prefix_line is not None:
            print(prefix_line)
        row_tpl = os.environ.get(
            "FAKE_COLAB_SESSIONS_TEMPLATE", DEFAULT_SESSION_ROW_TEMPLATE
        )
        enforce_shape = os.environ.get("FAKE_COLAB_ALLOW_NONCONTRACT_ROW_SHAPE") != "1"
        next_sessions: list[dict[str, Any]] = []
        sessions_changed = False
        for s in sessions:
            sid = s.get("session_id", "unknown")
            gpu = s.get("gpu", "CPU")
            rendered = row_tpl.format(sid=sid, gpu=gpu)
            if enforce_shape and sid != "?":
                masked = mask_session_line_shape(rendered, target_session=sid)
                if masked != SESSION_ROW_SHAPE:
                    sys.stderr.write(
                        f"Fake session row violated contract shape: got {masked!r}, expected {SESSION_ROW_SHAPE!r}\n"
                    )
                    return 2
            print(rendered)
            rem = s.get("remaining_reads")
            if isinstance(rem, int):
                sessions_changed = True
                if rem > 1:
                    updated_s = dict(s)
                    updated_s["remaining_reads"] = rem - 1
                    next_sessions.append(updated_s)
            else:
                next_sessions.append(s)
        if sessions_changed:
            sessions_file.write_text(
                json.dumps(next_sessions), encoding="utf-8", newline="\n"
            )
        return 0

    if subcmd == "new":
        if positionals:
            sys.stderr.write(
                f"Got unexpected extra arguments ({' '.join(positionals)})\n"
            )
            return 2
        sid = _get_opt("-s", "--session") or "rdlab-default-session"
        if os.environ.get("FAKE_COLAB_FAIL_NEW") == "1":
            sys.stderr.write(f"Failed to create session [{sid}]\n")
            return 1
        if any(s.get("session_id") == sid for s in sessions):
            sys.stderr.write(f"Session [{sid}] already exists\n")
            return 1
        gpu = _get_opt("--gpu") or "CPU"
        sessions.append({"session_id": sid, "gpu": gpu})
        sessions_file.write_text(json.dumps(sessions), encoding="utf-8", newline="\n")
        print(f"Created session [{sid}] ({gpu})")
        return 0

    if subcmd == "upload":
        sid = _get_opt("-s", "--session")
        if not sid or len(positionals) != 2:
            sys.stderr.write("Usage: colab upload -s SESSION <local_path> <remote_path>\n")
            return 2
        src, dst = positionals[0], positionals[1]
        if not any(s.get("session_id") == sid for s in sessions):
            sys.stderr.write(f"Session {sid} not found\n")
            return 1
        if not dst.startswith("/content/"):
            sys.stderr.write(f"Remote path must be under /content/: {dst}\n")
            return 1
        vm_dst = vm_fs / dst.removeprefix("/content/")
        if not Path(src).is_file():
            print(f"[colab] Local file '{src}' not found.")
            return 1
        max_path_len = int(os.environ.get("FAKE_COLAB_MAX_PATH_LEN", "1024"))
        if len(dst) > max_path_len or any(
            len(part) > 255 for part in dst.strip("/").split("/") if part
        ):
            print(
                f"[colab] Upload failed: 400 Client Error: Bad Request (path too long) for url: "
                f"https://8080-m-s-fake-{sid}.prod.r.colab.dev/api/contents/{dst.lstrip('/')}?"
                f"authuser=0&colab-runtime-proxy-token=FAKE_PROXY_SECRET_777"
            )
            return 1
        max_upload_bytes = int(
            os.environ.get("FAKE_COLAB_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024))
        )
        if Path(src).stat().st_size > max_upload_bytes:
            print(
                f"[colab] Upload failed: 413 Client Error: Payload Too Large for url: "
                f"https://8080-m-s-fake-{sid}.prod.r.colab.dev/api/contents/{dst.lstrip('/')}?"
                f"authuser=0&colab-runtime-proxy-token=FAKE_PROXY_SECRET_777"
            )
            return 1
        if not vm_dst.parent.is_dir() or vm_dst.is_dir():
            print(
                f"[colab] Upload failed: 500 Server Error: Internal Server Error for url: "
                f"https://8080-m-s-fake-{sid}.prod.r.colab.dev/api/contents/{dst.lstrip('/')}?"
                f"authuser=0&colab-runtime-proxy-token=FAKE_PROXY_SECRET_777"
            )
            return 1
        shutil.copyfile(src, vm_dst)
        record_file_access(state_dir, "upload", src, detail=f"to:{dst}")
        print(f"Uploaded {src} -> {dst}")
        return 0

    if subcmd == "download":
        sid = _get_opt("-s", "--session")
        if not sid or len(positionals) != 2:
            sys.stderr.write("Usage: colab download -s SESSION <remote_path> <local_path>\n")
            return 2
        src, dst = positionals[0], positionals[1]
        if not any(s.get("session_id") == sid for s in sessions):
            sys.stderr.write(f"Session {sid} not found\n")
            return 1
        if not src.startswith("/content/"):
            sys.stderr.write(f"Remote path must be under /content/: {src}\n")
            return 1
        vm_src = vm_fs / src.removeprefix("/content/")
        if vm_src.is_dir():
            sys.stderr.write(
                f"[colab] Download failed: Cannot download '{src}': path is a directory.\n"
            )
            return 1
        if not vm_src.is_file():
            sys.stderr.write(
                f"[colab] Download failed: File or directory not found: {src}\n"
            )
            return 1
        dst_path = Path(dst)
        if not dst_path.parent.is_dir():
            sys.stderr.write(
                f"[colab] Download failed: [Errno 2] No such file or directory: '{dst}'\n"
            )
            return 1
        truncate_target = os.environ.get("FAKE_COLAB_TRUNCATE_DOWNLOAD", "").strip()
        if truncate_target and truncate_target in ("1", vm_src.name, dst_path.name):
            raw_bytes = vm_src.read_bytes()
            dst_path.write_bytes(raw_bytes[: min(8, max(1, len(raw_bytes) // 4))])
        else:
            shutil.copyfile(vm_src, dst_path)
        record_file_access(state_dir, "download", dst, detail=f"from:{src}")
        print(f"Downloaded {src} -> {dst}")
        return 0

    if subcmd == "exec":
        if positionals:
            sys.stderr.write(
                f"colab exec does not take positional shell arguments ({' '.join(positionals)})\n"
            )
            return 2
        sid = _get_opt("-s", "--session")
        py_file = _get_opt("-f", "--file")
        if not sid or not py_file:
            sys.stderr.write("colab exec requires -s/--session and -f/--file\n")
            return 2
        if not any(s.get("session_id") == sid for s in sessions):
            sys.stderr.write(f"Session {sid} not found\n")
            return 1
        py_path = Path(py_file)
        if not py_path.is_file() or py_path.suffix != ".py":
            sys.stderr.write(
                f"colab exec -f must point to an existing Python (.py) file: {py_file}\n"
            )
            return 2
        timeout_raw = _get_opt("--timeout")
        try:
            timeout_s = float(timeout_raw) if timeout_raw is not None else 30.0
        except ValueError:
            sys.stderr.write(f"Invalid --timeout={timeout_raw}\n")
            return 2
        if timeout_s <= 0:
            sys.stderr.write(f"Invalid --timeout={timeout_raw}\n")
            return 2

        session_env_file = state_dir / f"session_env_{sid}.json"
        persisted_env: dict[str, str] = {}
        if session_env_file.is_file():
            try:
                persisted_env = json.loads(
                    session_env_file.read_text(encoding="utf-8")
                )
            except Exception:
                persisted_env = {}

        for kv in parsed_opts.get("--env", []):
            if "=" not in kv:
                sys.stderr.write(f"Invalid --env value (expected KEY=VALUE): {kv}\n")
                return 2
            k, v = kv.split("=", 1)
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
                sys.stderr.write(f"Invalid --env variable name: {k}\n")
                return 2
            persisted_env[k] = v

        session_env_file.write_text(
            json.dumps(persisted_env, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        exec_env = os.environ.copy()
        for k_existing in list(exec_env):
            if k_existing.startswith("HF_TOKEN") or k_existing == "HUGGING_FACE_HUB_TOKEN":
                exec_env.pop(k_existing, None)
        for rig_only_key in (
            "HF_HOME",
            "RDLAB_ITEMS_FILE",
            "RDLAB_RECEIPTS_DIR",
            "RDLAB_OUT_DIR",
            "RDLAB_REPO_TAR",
            "RDLAB_RECEIPTS_TAR",
            "RDLAB_RESUME_TAR",
            "RDLAB_RUN_DIR",
            "RDLAB_WORK_DIR",
            "RDLAB_RESULT_TAR",
        ):
            exec_env.pop(rig_only_key, None)

        exec_env["RDLAB_VM_CONTENT_DIR"] = str(vm_fs)
        exec_env.setdefault("RDLAB_STATUS_FILE", str(vm_fs / "status.json"))
        exec_env.setdefault(
            "RDLAB_ITEMS_FILE", str(vm_fs / "rdlab_items" / "items.jsonl")
        )
        exec_env.setdefault(
            "FAKE_COLAB_DISK_FREE_BYTES",
            os.environ.get("FAKE_COLAB_DISK_FREE_BYTES", "120000000000"),
        )
        for k, v in persisted_env.items():
            mapped_v = v
            if v.startswith("/content/") and not v.startswith("/content/drive"):
                mapped_v = str(vm_fs / v.removeprefix("/content/"))
            elif v == "/content":
                mapped_v = str(vm_fs)
            exec_env[k] = mapped_v

        (vm_fs / ".cache").mkdir(parents=True, exist_ok=True)
        sleep_s = float(os.environ.get("FAKE_COLAB_EXEC_SLEEP", "0"))
        step = exec_env.get("RDLAB_STEP", "run")
        if step == "gate" and os.environ.get("FAKE_COLAB_UNWRITABLE_ITEMS_DIR") == "1":
            blocker = vm_fs / "rdlab_items"
            if blocker.is_dir():
                shutil.rmtree(blocker)
            blocker.write_text("file-blocking-mkdir\n", encoding="utf-8", newline="\n")

        session_cells_file = state_dir / f"session_cells_{sid}.json"
        prior_cells: list[dict[str, Any]] = []
        if session_cells_file.is_file():
            try:
                prior_cells = json.loads(
                    session_cells_file.read_text(encoding="utf-8")
                )
            except Exception:
                prior_cells = []

        if step == "run" and os.environ.get("FAKE_COLAB_CRASH_RUN_BEFORE_STATUS") == "1":
            sys.stderr.write(
                "Traceback (most recent call last): cell crashed before writing status.json\n"
            )
            prior_cells.append(
                {
                    "step": step,
                    "py_file": str(py_path),
                    "env": dict(persisted_env),
                    "rc": 1,
                }
            )
            session_cells_file.write_text(
                json.dumps(prior_cells, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            if os.environ.get("FAKE_COLAB_FORCE_EXEC_RC_ZERO", "1") == "1":
                return 0
            return 1

        try:
            if step == "run" and sleep_s > 0:
                time.sleep(sleep_s)
                sys.stderr.write(f"Execution timed out after {timeout_s}s\n")
                return 124
            kernel_code = (
                "import json, os, pathlib, shutil, subprocess, sys\n"
                "def _fake_disk_usage(p):\n"
                "    ps = os.path.abspath(os.fspath(p))\n"
                "    vm_root = os.path.abspath(os.environ['RDLAB_VM_CONTENT_DIR'])\n"
                "    dflt = int(os.environ.get('FAKE_COLAB_DISK_FREE_BYTES', '120000000000'))\n"
                "    free_b = int(os.environ.get('FAKE_COLAB_CONTENT_DISK_FREE_BYTES', str(dflt))) if ps == vm_root else int(os.environ.get('FAKE_COLAB_HF_DISK_FREE_BYTES', str(dflt)))\n"
                "    return shutil._ntuple_diskusage(max(free_b, 200000000000), 1000000000, free_b)\n"
                "shutil.disk_usage = _fake_disk_usage\n"
                "_real_which = shutil.which\n"
                "def _fake_which(cmd, *a, **kw):\n"
                "    if cmd == 'uv':\n"
                "        if os.environ.get('FAKE_COLAB_UV_ABSENT') == '1':\n"
                "            return None\n"
                "        return '/usr/local/bin/uv'\n"
                "    return _real_which(cmd, *a, **kw)\n"
                "shutil.which = _fake_which\n"
                "_real_subproc_run = subprocess.run\n"
                "def _fake_subproc_run(cmd, *a, **kw):\n"
                "    cmd_list = [str(x) for x in cmd] if isinstance(cmd, (list, tuple)) else [str(cmd)]\n"
                "    call_env = kw.get('env') or os.environ\n"
                "    st_dir = os.environ.get('FAKE_COLAB_STATE_DIR', '')\n"
                "    if st_dir:\n"
                "        rec_p = pathlib.Path(st_dir) / 'vm_subprocesses.jsonl'\n"
                "        rec = {\n"
                "            'step': os.environ.get('RDLAB_STEP', ''),\n"
                "            'cmd': cmd_list,\n"
                "            'env_keys': sorted(list(call_env.keys())),\n"
                "            'uv_cache_dir': call_env.get('UV_CACHE_DIR'),\n"
                "            'uv_python_install_dir': call_env.get('UV_PYTHON_INSTALL_DIR'),\n"
                "        }\n"
                "        with rec_p.open('a', encoding='utf-8', newline='\\n') as fh:\n"
                "            fh.write(json.dumps(rec) + '\\n')\n"
                "    plant = os.environ.get('FAKE_COLAB_PROBE_PLANT_STDOUT', '')\n"
                "    plant_sfx = (' ' + plant) if plant else ''\n"
                "    if cmd_list[:2] == ['uv', '--version']:\n"
                "        if os.environ.get('FAKE_COLAB_UV_ABSENT') == '1':\n"
                "            raise FileNotFoundError(2, 'No such file or directory', 'uv')\n"
                "        return subprocess.CompletedProcess(cmd_list, 0, stdout='uv 0.11.18' + plant_sfx + '\\n', stderr='')\n"
                "    if cmd_list[:3] == ['uv', 'python', 'find']:\n"
                "        if os.environ.get('FAKE_COLAB_UV_NO_PYTHON_315') == '1':\n"
                "            err_msg = os.environ.get('FAKE_COLAB_UV_STDERR', 'error: No interpreter found for Python 3.15 in managed installations or search path\\n')\n"
                "            return subprocess.CompletedProcess(cmd_list, 2, stdout='', stderr=err_msg)\n"
                "        return subprocess.CompletedProcess(cmd_list, 0, stdout='/usr/local/share/uv/python/cpython-3.15.0-linux-x86_64-gnu/bin/python3.15' + plant_sfx + '\\n', stderr='')\n"
                "    if cmd_list[:2] == ['uv', 'venv']:\n"
                "        if os.environ.get('FAKE_COLAB_FAIL_UV_VENV') == '1':\n"
                "            err_msg = os.environ.get('FAKE_COLAB_UV_STDERR', 'error: Failed to create virtualenv\\n')\n"
                "            out_msg = os.environ.get('FAKE_COLAB_UV_STDOUT', 'Creating virtual environment\\n')\n"
                "            return subprocess.CompletedProcess(cmd_list, 2, stdout=out_msg, stderr=err_msg)\n"
                "    if os.environ.get('RDLAB_STEP') == 'probe':\n"
                "        if cmd_list[:4] == ['uv', 'python', 'install', '--help']:\n"
                "            if os.environ.get('FAKE_COLAB_UV_SUPPORTS_DRY_RUN') == '1':\n"
                "                return subprocess.CompletedProcess(cmd_list, 0, stdout='Usage: uv python install --dry-run' + plant_sfx + '\\n', stderr='')\n"
                "            return subprocess.CompletedProcess(cmd_list, 0, stdout='Usage: uv python install [OPTIONS] [TARGETS]...' + plant_sfx + '\\n', stderr='')\n"
                "        if cmd_list[:3] == ['uv', 'python', 'list'] and '--only-downloads' in cmd_list:\n"
                "            dl_url = 'https://releases.astral.sh/github/python-build-standalone/releases/download/20260510/cpython-3.15.0b1%2B20260510-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz'\n"
                "            if plant:\n"
                "                dl_url = dl_url + '?' + plant\n"
                "            payload = json.dumps([{'key': 'cpython-3.15.0b1-linux-x86_64-gnu', 'version': '3.15.0b1', 'url': dl_url}])\n"
                "            return subprocess.CompletedProcess(cmd_list, 0, stdout=payload + '\\n', stderr='')\n"
                "        if cmd_list[:3] == ['uv', 'python', 'list']:\n"
                "            return subprocess.CompletedProcess(cmd_list, 0, stdout='cpython-3.12.13-linux-x86_64-gnu /usr/bin/python3.12' + plant_sfx + '\\n', stderr='')\n"
                "        if cmd_list[:3] == ['uv', 'python', 'install']:\n"
                "            inst_dir = call_env.get('UV_PYTHON_INSTALL_DIR', '/tmp/probe-py')\n"
                "            if '--dry-run' not in cmd_list:\n"
                "                p_dir = pathlib.Path(inst_dir) / 'cpython-3.15.0b1' / 'include' / 'python3.15'\n"
                "                p_dir.mkdir(parents=True, exist_ok=True)\n"
                "                (p_dir / 'Python.h').write_text('/* Python.h */\\n', encoding='utf-8')\n"
                "            return subprocess.CompletedProcess(cmd_list, 0, stdout='Installed Python 3.15.0b1 in 0.42s' + plant_sfx + '\\n', stderr='Downloading https://releases.astral.sh/cpython-3.15.tar.gz' + plant_sfx + '\\n')\n"
                "        if cmd_list[0] == 'curl' and '-I' in cmd_list:\n"
                "            return subprocess.CompletedProcess(cmd_list, 0, stdout='HTTP/2 200\\r\\ncontent-length: 32505856\\r\\n' + plant_sfx + '\\n', stderr='')\n"
                "        if cmd_list[0] == 'curl':\n"
                "            return subprocess.CompletedProcess(cmd_list, 0, stdout='206 5242880 0.125000' + plant_sfx + '\\n', stderr='')\n"
                "        if cmd_list == ['which', 'uv']:\n"
                "            return subprocess.CompletedProcess(cmd_list, 0, stdout='/usr/local/bin/uv' + plant_sfx + '\\n', stderr='')\n"
                "        if cmd_list[:4] == ['python3', '-m', 'pip', 'show']:\n"
                "            return subprocess.CompletedProcess(cmd_list, 0, stdout='Name: uv\\nVersion: 0.11.18\\nLocation: /usr/local/lib/python3.12/dist-packages' + plant_sfx + '\\n', stderr='')\n"
                "        if cmd_list[0].startswith('python3') and cmd_list[1:] == ['--version']:\n"
                "            if cmd_list[0] in ('python3', 'python3.12'):\n"
                "                return subprocess.CompletedProcess(cmd_list, 0, stdout='Python 3.12.13' + plant_sfx + '\\n', stderr='')\n"
                "            return subprocess.CompletedProcess(cmd_list, 127, stdout='', stderr='not found' + plant_sfx + '\\n')\n"
                "        if cmd_list[0].startswith('python3') and len(cmd_list) >= 2 and cmd_list[1] == '-c':\n"
                "            return subprocess.CompletedProcess(cmd_list, 0, stdout=json.dumps({'include_dir': '/usr/include/python3.12', 'python_h': '/usr/include/python3.12/Python.h', 'exists': True}) + plant_sfx + '\\n', stderr='')\n"
                "        return subprocess.CompletedProcess(cmd_list, 0, stdout='ok' + plant_sfx + '\\n', stderr='')\n"
                "    return _real_subproc_run(cmd, *a, **kw)\n"
                "subprocess.run = _fake_subproc_run\n"
                "sys.argv = ['colab_kernel_launcher.py', '-f', '/tmp/kernel-fake.json']\n"
                f"src = pathlib.Path({str(py_path.resolve())!r}).read_text(encoding='utf-8')\n"
                "ns = {'__name__': '__main__', '_PRIOR_EXEC_COUNT': "
                f"{len(prior_cells)!r}"
                "}\n"
                "exec(compile(src, '<cell>', 'exec'), ns)\n"
                "rc = ns.get('CELL_EXIT_CODE', 0)\n"
                "raise SystemExit(int(rc) if isinstance(rc, int) else 0)\n"
            )
            proc = subprocess.run(
                [sys.executable, "-c", kernel_code],
                stdin=subprocess.DEVNULL,
                env=exec_env,
                cwd=str(vm_fs),
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"Execution timed out after {timeout_s}s\n")
            return 124

        prior_cells.append(
            {
                "step": step,
                "py_file": str(py_path),
                "env": dict(persisted_env),
                "rc": int(proc.returncode),
            }
        )
        session_cells_file.write_text(
            json.dumps(prior_cells, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        status_env = exec_env.get("RDLAB_STATUS_FILE")
        if os.environ.get("FAKE_COLAB_DELETE_STATUS_FILE") == "1":
            if status_env and Path(status_env).is_file():
                Path(status_env).unlink()
        elif os.environ.get("FAKE_COLAB_OVERRIDE_STATUS_FAILED") == "1" and step == "run":
            if status_env:
                Path(status_env).parent.mkdir(parents=True, exist_ok=True)
                Path(status_env).write_text(
                    json.dumps(
                        {
                            "step": "run",
                            "rc": 2,
                            "status": "refused",
                            "failure_class": "FORCED_STATUS_OVERRIDE",
                            "reason": "FORCED_STATUS_OVERRIDE: status.json reports failure while colab exec returns 0",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
        if os.environ.get("FAKE_COLAB_FORCE_EXEC_RC_ZERO") == "1":
            return 0
        return proc.returncode

    if subcmd == "stop":
        if positionals:
            sys.stderr.write(
                f"Got unexpected extra arguments ({' '.join(positionals)})\n"
            )
            return 2
        sid = _get_opt("-s", "--session")
        if not sid:
            sys.stderr.write("colab stop requires -s/--session\n")
            return 2
        stopped_file = state_dir / "stopped_sessions.jsonl"
        with stopped_file.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"session_id": sid, "ts": time.time()}) + "\n")
        lag_reads = int(os.environ.get("FAKE_COLAB_STOP_LAG_READS", "0"))
        if os.environ.get("FAKE_COLAB_ORPHAN_ON_STOP") == "1":
            next_stop_sessions: list[dict[str, Any]] = []
            for s in sessions:
                if s.get("session_id") == sid:
                    orphan_entry: dict[str, Any] = {
                        "session_id": "?",
                        "gpu": s.get("gpu", "CPU"),
                    }
                    if lag_reads > 0:
                        orphan_entry["remaining_reads"] = lag_reads
                    next_stop_sessions.append(orphan_entry)
                else:
                    next_stop_sessions.append(s)
            sessions_file.write_text(
                json.dumps(next_stop_sessions), encoding="utf-8", newline="\n"
            )
        elif lag_reads > 0:
            next_lag_sessions: list[dict[str, Any]] = []
            for s in sessions:
                if s.get("session_id") == sid:
                    lag_entry = dict(s)
                    lag_entry["remaining_reads"] = lag_reads
                    next_lag_sessions.append(lag_entry)
                else:
                    next_lag_sessions.append(s)
            sessions_file.write_text(
                json.dumps(next_lag_sessions), encoding="utf-8", newline="\n"
            )
        elif os.environ.get("FAKE_COLAB_KEEP_SESSION_ON_STOP") != "1":
            sessions = [s for s in sessions if s.get("session_id") != sid]
            sessions_file.write_text(
                json.dumps(sessions), encoding="utf-8", newline="\n"
            )
            for state_f in (
                state_dir / f"session_env_{sid}.json",
                state_dir / f"session_cells_{sid}.json",
            ):
                if state_f.is_file():
                    state_f.unlink()
        print(f"Stopped session [{sid}]")
        return 0

    if subcmd == "status":
        sid = _get_opt("-s", "--session")
        print(f"Session [{sid or '?'}] Status: IDLE")
        return 0

    if subcmd == "usage":
        print(
            "Compute units balance: 100.0 | Usage rate: 0.08/hr | Active assignments: 1"
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
