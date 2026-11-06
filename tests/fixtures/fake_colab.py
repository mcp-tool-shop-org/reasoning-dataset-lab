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

import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.parse
import zipfile
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
    "[{sid}] m-s-fake-{sid} | Hardware: {gpu} | Shape: Standard | Variant: DEFAULT"
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


DEFAULT_FAKE_UV_PYTHON_315_PATH = (
    "/usr/local/share/uv/python/cpython-3.15.1-linux-x86_64-gnu/bin/python3.15"
)
DEFAULT_FAKE_UV_MISSING_315_STDERR = (
    "error: No interpreter found for Python 3.15 in managed installations or search path"
)
HERMETIC_SYSTEM_PATH = "/usr/bin:/bin"


def build_fixture_python_archive(
    archive_path: Path,
    *,
    python_version: str = "3.15.1",
    build: str = "20261021",
    build_tag: str | None = None,
    extra_bytes: int = 0,
    extra_payload: bytes = b"",
) -> tuple[Path, str, int]:
    """Build a small fixture `cpython-3.15.1+20261021` `.tar.gz` archive and return `(path, sha256, bytes)`."""
    effective_build = build_tag if build_tag is not None else build
    archive_path = Path(archive_path)
    if archive_path.is_dir() or not archive_path.name.endswith(".tar.gz"):
        archive_path = (
            archive_path
            / f"cpython-{python_version}+{effective_build}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    py_script = (
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--version" || "$1" == "-V" ]]; then\n'
        f'  echo "Python {python_version}"\n'
        "  exit 0\n"
        "fi\n"
        'exec python3 "$@"\n'
    ).encode("utf-8")
    python_h = b"/* Python.h fixture */\n"
    build_json = (
        json.dumps({"version": python_version, "release": effective_build}, indent=2)
        + "\n"
    ).encode("utf-8")

    with tarfile.open(archive_path, "w:gz") as tf:
        for arcname, payload, mode in (
            ("python/bin/python3.15", py_script, 0o755),
            ("python/bin/python3", py_script, 0o755),
            ("python/include/python3.15/Python.h", python_h, 0o644),
            ("python/BUILD.json", build_json, 0o644),
        ):
            info = tarfile.TarInfo(name=arcname)
            info.size = len(payload)
            info.mode = mode
            info.mtime = 1762041600
            tf.addfile(info, io.BytesIO(payload))
        if extra_bytes > 0 or extra_payload:
            pad_data = (b"P" * extra_bytes) + extra_payload
            pad_info = tarfile.TarInfo(name="python/lib/libpython3.15.so.fixture")
            pad_info.size = len(pad_data)
            pad_info.mode = 0o644
            pad_info.mtime = 1762041600
            tf.addfile(pad_info, io.BytesIO(pad_data))

    raw = archive_path.read_bytes()
    return archive_path, hashlib.sha256(raw).hexdigest(), len(raw)


PINNED_UV_WHEEL_EXACT_SIZE_BYTES = 17_211_825
_CACHED_DEFAULT_UV_WHEEL: tuple[Path, str, int] | None = None


def build_fixture_uv_wheel(
    wheel_path: Path,
    *,
    uv_binary_version: str = "0.13.5",
    target_size: int = PINNED_UV_WHEEL_EXACT_SIZE_BYTES,
    omit_uv_member: bool = False,
    omit_uvx_member: bool = False,
    extra_member_name: str | None = None,
    extra_member_is_symlink: bool = False,
    tamper_after_build: bool = False,
) -> tuple[Path, str, int]:
    """Build a 12-member fixture `uv-0.13.5` `.whl` of exact size `target_size` (`17,211,825` bytes) and return `(path, sha256, bytes)`."""
    wheel_path = Path(wheel_path)
    if wheel_path.is_dir() or not wheel_path.name.endswith(".whl"):
        wheel_path = (
            wheel_path
            / "uv-0.13.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        )
    wheel_path.parent.mkdir(parents=True, exist_ok=True)

    fake_colab_py_posix = _to_posix(Path(__file__).resolve())
    uv_script = (
        "#!/usr/bin/env bash\n"
        'export FAKE_COLAB_UV_INVOKED_AS="$0"\n'
        'export FAKE_COLAB_UV_SOURCE="pinned-wheel"\n'
        f'export FAKE_COLAB_PINNED_UV_BINARY_VERSION="uv {uv_binary_version}"\n'
        f"exec python3 '{fake_colab_py_posix}' --fake-uv \"$@\"\n"
    ).encode("utf-8")
    uvx_script = (
        "#!/usr/bin/env bash\n"
        f'echo "uvx {uv_binary_version}"\n'
    ).encode("utf-8")

    members: list[tuple[str, bytes, int]] = [
        ("uv/__init__.py", b"__all__ = ['find_uv_bin']\n", 0o644),
        ("uv/__main__.py", b"import sys\n", 0o644),
        ("uv/_find_uv.py", b"def find_uv_bin() -> str: return 'uv'\n", 0o644),
        ("uv/py.typed", b"", 0o644),
    ]
    if not omit_uv_member:
        members.append(("uv-0.13.5.data/scripts/uv", uv_script, 0o755))
    if not omit_uvx_member:
        members.append(("uv-0.13.5.data/scripts/uvx", uvx_script, 0o755))
    members.extend(
        [
            ("uv-0.13.5.dist-info/METADATA", b"Metadata-Version: 2.3\nName: uv\nVersion: 0.13.5\n", 0o644),
            ("uv-0.13.5.dist-info/WHEEL", b"Wheel-Version: 1.0\nGenerator: maturin\nRoot-Is-Purelib: false\nTag: py3-none-manylinux_2_17_x86_64\n", 0o644),
            ("uv-0.13.5.dist-info/entry_points.txt", b"[console_scripts]\nuv = uv:main\nuvx = uv:main\n", 0o644),
            ("uv-0.13.5.dist-info/licenses/LICENSE-APACHE", b"Apache-2.0\n", 0o644),
            ("uv-0.13.5.dist-info/licenses/LICENSE-MIT", b"MIT\n", 0o644),
        ]
    )
    if extra_member_name is not None:
        ext_mode = 0o120777 if extra_member_is_symlink else 0o100644
        members.append((extra_member_name, b"/tmp/target", ext_mode))

    def _write_zip(buf: io.BytesIO | Path, record_bytes: bytes) -> None:
        with zipfile.ZipFile(
            buf, "w", compression=zipfile.ZIP_STORED, allowZip64=False
        ) as zf:
            for arcname, payload, mode in members:
                zi = zipfile.ZipInfo(filename=arcname, date_time=(2026, 10, 21, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_STORED
                full_mode = mode if (mode & 0o170000) else (0o100000 | mode)
                zi.external_attr = (full_mode & 0xFFFF) << 16
                zf.writestr(zi, payload)
            rec_zi = zipfile.ZipInfo(
                filename="uv-0.13.5.dist-info/RECORD",
                date_time=(2026, 10, 21, 0, 0, 0),
            )
            rec_zi.compress_type = zipfile.ZIP_STORED
            rec_zi.external_attr = (0o100644 & 0xFFFF) << 16
            zf.writestr(rec_zi, record_bytes)

    probe_bio = io.BytesIO()
    _write_zip(probe_bio, b"")
    base_len = len(probe_bio.getvalue())
    pad_len = max(0, int(target_size) - base_len)
    record_payload = (b"R" * max(0, pad_len - 1)) + (b"\n" if pad_len > 0 else b"")
    _write_zip(wheel_path, record_payload)

    raw = bytearray(wheel_path.read_bytes())
    if tamper_after_build and len(raw) > 64:
        raw[-64] ^= 0x01
        wheel_path.write_bytes(bytes(raw))
    final_bytes = bytes(raw)
    return wheel_path, hashlib.sha256(final_bytes).hexdigest(), len(final_bytes)


def _get_default_fixture_uv_wheel() -> tuple[Path, str, int]:
    """Return a cached valid 12-member `17,211,825`-byte fixture wheel in `/tmp`."""
    global _CACHED_DEFAULT_UV_WHEEL
    if _CACHED_DEFAULT_UV_WHEEL is not None and _CACHED_DEFAULT_UV_WHEEL[0].is_file():
        return _CACHED_DEFAULT_UV_WHEEL
    cache_path = Path(f"/tmp/rdlab-fixture-uv-0.13.5-{os.getpid()}.whl")
    _CACHED_DEFAULT_UV_WHEEL = build_fixture_uv_wheel(cache_path)
    return _CACHED_DEFAULT_UV_WHEEL


def setup_fake_colab_environment(
    base_dir: Path,
    *,
    neighbour_sessions: list[dict[str, str]] | None = None,
    uv_python_315_found: bool = True,
    uv_python_315_path: str = DEFAULT_FAKE_UV_PYTHON_315_PATH,
    uv_python_315_stderr: str = DEFAULT_FAKE_UV_MISSING_315_STDERR,
    uv_version: str = "uv 0.13.5",
) -> dict[str, Path]:
    """Create a hermetic fake `colab` + fake `uv` + fake `curl` bin directory and sentinel real-binary traps in `base_dir`."""
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

    default_wheel_path, default_wheel_sha, _ = _get_default_fixture_uv_wheel()
    uv_state_file = state_dir / "uv_state.json"
    uv_state_file.write_text(
        json.dumps(
            {
                "version": uv_version,
                "python_315_found": bool(uv_python_315_found),
                "python_315_path": uv_python_315_path,
                "python_315_stderr": uv_python_315_stderr,
                "uv_wheel_path": str(default_wheel_path),
                "expected_uv_wheel_sha256": default_wheel_sha,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    sentinel_called = state_dir / "REAL_COLAB_CALLED"
    sentinel_script = sentinel_bin / "colab"
    sentinel_script.write_text(
        "#!/usr/bin/env bash\n"
        f"touch '{_to_posix(sentinel_called)}'\n"
        "echo 'SENTINEL_REAL_BINARY_REACHED: colab' >&2\n"
        "exit 99\n",
        encoding="utf-8",
        newline="\n",
    )
    sentinel_script.chmod(sentinel_script.stat().st_mode | stat.S_IEXEC)

    sentinel_uv_called = state_dir / "REAL_UV_CALLED"
    sentinel_uv_script = sentinel_bin / "uv"
    sentinel_uv_script.write_text(
        "#!/usr/bin/env bash\n"
        f"touch '{_to_posix(sentinel_uv_called)}'\n"
        "echo 'SENTINEL_REAL_BINARY_REACHED: uv' >&2\n"
        "exit 99\n",
        encoding="utf-8",
        newline="\n",
    )
    sentinel_uv_script.chmod(sentinel_uv_script.stat().st_mode | stat.S_IEXEC)

    sentinel_curl_called = state_dir / "REAL_CURL_CALLED"
    sentinel_curl_script = sentinel_bin / "curl"
    sentinel_curl_script.write_text(
        "#!/usr/bin/env bash\n"
        f"touch '{_to_posix(sentinel_curl_called)}'\n"
        "echo 'SENTINEL_REAL_BINARY_REACHED: curl' >&2\n"
        "exit 99\n",
        encoding="utf-8",
        newline="\n",
    )
    sentinel_curl_script.chmod(sentinel_curl_script.stat().st_mode | stat.S_IEXEC)

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

    fake_uv_sh = fake_bin / "uv"
    fake_uv_sh.write_text(
        "#!/usr/bin/env bash\n"
        f"export FAKE_COLAB_STATE_DIR=\"${{FAKE_COLAB_STATE_DIR:-{_to_posix(state_dir)}}}\"\n"
        'export FAKE_COLAB_UV_INVOKED_AS="${FAKE_COLAB_UV_INVOKED_AS:-$0}"\n'
        'export FAKE_COLAB_UV_SOURCE="${FAKE_COLAB_UV_SOURCE:-image}"\n'
        f"exec python3 '{fake_colab_py_posix}' --fake-uv \"$@\"\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_uv_sh.chmod(fake_uv_sh.stat().st_mode | stat.S_IEXEC)

    fake_curl_sh = fake_bin / "curl"
    fake_curl_sh.write_text(
        "#!/usr/bin/env bash\n"
        f"export FAKE_COLAB_STATE_DIR=\"${{FAKE_COLAB_STATE_DIR:-{_to_posix(state_dir)}}}\"\n"
        f"exec python3 '{fake_colab_py_posix}' --fake-curl \"$@\"\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_curl_sh.chmod(fake_curl_sh.stat().st_mode | stat.S_IEXEC)

    fake_nvidia_smi_sh = fake_bin / "nvidia-smi"
    fake_nvidia_smi_sh.write_text(
        "#!/usr/bin/env bash\n"
        f"export FAKE_COLAB_STATE_DIR=\"${{FAKE_COLAB_STATE_DIR:-{_to_posix(state_dir)}}}\"\n"
        'mode="${FAKE_COLAB_NVIDIA_SMI_MODE:-ok}"\n'
        'if [[ "$mode" == "fail" ]]; then\n'
        '  echo "NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver." >&2\n'
        '  exit 9\n'
        'elif [[ "$mode" == "empty" ]]; then\n'
        '  exit 0\n'
        'elif [[ "$mode" == "non_int" ]]; then\n'
        '  echo "N/A"\n'
        '  exit 0\n'
        'fi\n'
        'for a in "$@"; do if [[ "$a" == "--query-gpu=name" ]]; then\n'
        f"  exec python3 '{fake_colab_py_posix}' --fake-nvidia-smi-name\n"
        'fi; done\n'
        'if [[ "$*" == *"--query-gpu=memory.used"* ]]; then\n'
        '  echo "${FAKE_COLAB_NVIDIA_SMI_MIB:-1024}"\n'
        '  exit 0\n'
        'fi\n'
        'echo "NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4"\n'
        'exit 0\n',
        encoding="utf-8",
        newline="\n",
    )
    fake_nvidia_smi_sh.chmod(fake_nvidia_smi_sh.stat().st_mode | stat.S_IEXEC)

    return {
        "fake_bin": fake_bin,
        "sentinel_bin": sentinel_bin,
        "state_dir": state_dir,
        "colab_locks_dir": colab_locks_dir,
        "sentinel_called": sentinel_called,
        "sentinel_uv_called": sentinel_uv_called,
        "sentinel_curl_called": sentinel_curl_called,
        "uv_state_file": uv_state_file,
        "uv_calls_log": state_dir / "uv_calls.jsonl",
        "curl_calls_log": state_dir / "curl_calls.jsonl",
        "calls_log": state_dir / "calls.jsonl",
        "file_access_log": state_dir / "file_access.jsonl",
        "sessions_file": sessions_file,
        "stopped_file": state_dir / "stopped_sessions.jsonl",
    }


def build_hermetic_path(env_paths: dict[str, Path]) -> str:
    """Return a hermetic PATH containing `fake_bin`, `sentinel_bin`, and minimal system directories (`/usr/bin:/bin`)."""
    return (
        f"{_to_posix(env_paths['fake_bin'])}:"
        f"{_to_posix(env_paths['sentinel_bin'])}:"
        f"{HERMETIC_SYSTEM_PATH}"
    )


def set_fake_uv_python_315_state(
    state_dir_or_env: Path | dict[str, Path],
    *,
    found: bool,
    python_path: str = DEFAULT_FAKE_UV_PYTHON_315_PATH,
    stderr_message: str = DEFAULT_FAKE_UV_MISSING_315_STDERR,
    uv_version: str = "uv 0.13.5",
    fixture_archive_path: Path | str | None = None,
    expected_archive_sha256: str | None = None,
    install_fail: bool = False,
    install_rc: int = 1,
    install_stderr: str = "error: Failed to install cpython-3.15.1-linux-x86_64-gnu",
    install_sleep_seconds: float = 0.0,
    curl_fail: bool = False,
    curl_rc: int = 22,
    curl_stderr: str = "curl: (22) The requested URL returned error: 500",
    archive_size_bytes: int | None = None,
    curl_ignore_max_filesize: bool = False,
    uv_wheel_path: Path | str | None = None,
    expected_uv_wheel_sha256: str | None = None,
    uv_wheel_download_fail: bool = False,
    uv_wheel_download_error: str = "HTTP 503 Service Unavailable",
    uv_wheel_download_http_code: int = 503,
    uv_wheel_short_read_bytes: int | None = None,
    uv_wheel_content_length: int | None = None,
    pinned_uv_binary_version: str | None = None,
) -> Path:
    """Update the fake `uv` + `curl` state file controlling find-or-install behavior."""
    state_dir = (
        state_dir_or_env["state_dir"]
        if isinstance(state_dir_or_env, dict)
        else state_dir_or_env
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    default_wheel_path, default_wheel_sha, _ = _get_default_fixture_uv_wheel()
    eff_wheel_path = (
        str(uv_wheel_path) if uv_wheel_path is not None else str(default_wheel_path)
    )
    eff_wheel_sha = (
        expected_uv_wheel_sha256
        if expected_uv_wheel_sha256 is not None
        else default_wheel_sha
    )
    uv_state_file = state_dir / "uv_state.json"
    state_payload: dict[str, Any] = {
        "version": uv_version,
        "python_315_found": bool(found),
        "python_315_path": python_path,
        "python_315_stderr": stderr_message,
        "fixture_archive_path": (
            str(fixture_archive_path) if fixture_archive_path is not None else None
        ),
        "expected_archive_sha256": expected_archive_sha256,
        "install_fail": bool(install_fail),
        "install_rc": int(install_rc),
        "install_stderr": stderr_message if install_fail and not install_stderr else install_stderr,
        "install_sleep_seconds": float(install_sleep_seconds),
        "curl_fail": bool(curl_fail),
        "curl_rc": int(curl_rc),
        "curl_stderr": curl_stderr,
        "archive_size_bytes": archive_size_bytes,
        "curl_ignore_max_filesize": bool(curl_ignore_max_filesize),
        "uv_wheel_path": eff_wheel_path,
        "expected_uv_wheel_sha256": eff_wheel_sha,
        "uv_wheel_download_fail": bool(uv_wheel_download_fail),
        "uv_wheel_download_error": uv_wheel_download_error,
        "uv_wheel_download_http_code": int(uv_wheel_download_http_code),
        "uv_wheel_short_read_bytes": uv_wheel_short_read_bytes,
        "uv_wheel_content_length": uv_wheel_content_length,
        "pinned_uv_binary_version": pinned_uv_binary_version,
    }
    uv_state_file.write_text(
        json.dumps(state_payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return uv_state_file


def _load_fake_uv_state(state_dir: Path) -> dict[str, Any]:
    uv_state: dict[str, Any] = {
        "version": "uv 0.13.5",
        "python_315_found": True,
        "python_315_path": DEFAULT_FAKE_UV_PYTHON_315_PATH,
        "python_315_stderr": DEFAULT_FAKE_UV_MISSING_315_STDERR,
    }
    uv_state_file = state_dir / "uv_state.json"
    if uv_state_file.is_file():
        try:
            loaded = json.loads(uv_state_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                uv_state.update(loaded)
        except Exception:
            pass
    return uv_state


def run_fake_curl(argv: list[str]) -> int:
    """Execute the standalone fake `curl` CLI placed in `fake_bin/curl`."""
    state_dir_raw = os.environ.get("FAKE_COLAB_STATE_DIR")
    state_dir = (
        Path(state_dir_raw) if state_dir_raw else Path("/tmp/rdlab-fake-colab-state")
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    uv_state = _load_fake_uv_state(state_dir)

    curl_calls_log = state_dir / "curl_calls.jsonl"
    with curl_calls_log.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"args": list(argv), "ts": time.time()}) + "\n")

    out_file: str | None = None
    max_filesize: int | None = None
    url_arg: str = ""
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "-o" and i + 1 < len(argv):
            out_file = argv[i + 1]
            i += 2
            continue
        if tok == "--max-filesize" and i + 1 < len(argv):
            try:
                max_filesize = int(argv[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if tok in ("-r", "-w", "--max-time") and i + 1 < len(argv):
            i += 2
            continue
        if not tok.startswith("-"):
            url_arg = tok
        i += 1

    if "-I" in argv:
        sys.stdout.write("HTTP/2 200\r\ncontent-length: 32505856\r\n\r\n")
        return 0

    if out_file is None or out_file == "-":
        total_to_stream = (
            50 * 1024 * 1024
            if (
                "50mb" in url_arg.lower()
                or os.environ.get("FAKE_COLAB_CURL_OFFER_50MB") == "1"
            )
            else min(max_filesize or 4096, 5_242_880)
        )
        chunk = b"A" * 65536
        remaining = total_to_stream
        try:
            while remaining > 0:
                n = min(len(chunk), remaining)
                sys.stdout.buffer.write(chunk[:n])
                sys.stdout.buffer.flush()
                remaining -= n
            return 0
        except (BrokenPipeError, ConnectionResetError, OSError):
            return 0

    if uv_state.get("curl_fail"):
        err_msg = str(
            uv_state.get("curl_stderr")
            or "curl: (22) The requested URL returned error: 500"
        )
        if not err_msg.endswith("\n"):
            err_msg += "\n"
        sys.stderr.write(err_msg)
        return int(uv_state.get("curl_rc") or 22)

    sim_size = uv_state.get("archive_size_bytes")
    if isinstance(sim_size, int) and sim_size > 0:
        if (
            max_filesize is not None
            and sim_size > max_filesize
            and not uv_state.get("curl_ignore_max_filesize")
        ):
            sys.stderr.write("curl: (63) Maximum file size exceeded\n")
            return 63
        if out_file and out_file != "-":
            out_p = Path(out_file)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with out_p.open("wb") as fh:
                fh.truncate(sim_size)
            return 0

    fixture_archive = uv_state.get("fixture_archive_path")
    if fixture_archive and Path(str(fixture_archive)).is_file():
        if out_file and out_file != "-":
            out_p = Path(out_file)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(fixture_archive), out_p)
            return 0

    err_fallback = os.environ.get(
        "FAKE_COLAB_UV_STDERR",
        str(uv_state.get("python_315_stderr") or "curl: (22) fixture archive not configured"),
    )
    if not err_fallback.endswith("\n"):
        err_fallback += "\n"
    sys.stderr.write(err_fallback)
    return 22


def _parse_fake_lock_entries(lock_path: Path) -> list[tuple[str, str, list[str], list[str]]]:
    """Parse `(name, version, hashes, distributions)` from a lock/requirements file for `run_fake_uv`."""
    entries: list[tuple[str, str, list[str], list[str]]] = []
    cur_name: str | None = None
    cur_ver: str | None = None
    cur_hashes: list[str] = []
    cur_dists: list[str] = []
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m_pkg = re.match(r"^([a-zA-Z0-9_.-]+)==([^\s\\]+)", line)
        if m_pkg:
            if cur_name is not None and cur_ver is not None:
                entries.append((cur_name, cur_ver, cur_hashes, cur_dists))
            cur_name = m_pkg.group(1).lower()
            cur_ver = m_pkg.group(2)
            cur_hashes = [
                hm.group(1) for hm in re.finditer(r"--hash=sha256:([0-9a-f]{64})", line)
            ]
            cur_dists = []
            continue
        if cur_name is not None:
            for hm in re.finditer(r"--hash=sha256:([0-9a-f]{64})", line):
                cur_hashes.append(hm.group(1))
            if line.startswith("# distribution:"):
                cur_dists.append(line.split("# distribution:", 1)[1].strip())
    if cur_name is not None and cur_ver is not None:
        entries.append((cur_name, cur_ver, cur_hashes, cur_dists))
    return entries


def run_fake_uv(argv: list[str]) -> int:
    """Execute the standalone fake `uv` CLI placed in `fake_bin/uv`."""
    state_dir_raw = os.environ.get("FAKE_COLAB_STATE_DIR")
    state_dir = (
        Path(state_dir_raw) if state_dir_raw else Path("/tmp/rdlab-fake-colab-state")
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    uv_state = _load_fake_uv_state(state_dir)

    invoked_as = os.environ.get("FAKE_COLAB_UV_INVOKED_AS", "uv")
    # The pinned-uv dir is /tmp/uv-pinned, or a per-test dir (FAKE_COLAB_PINNED_UV_DIR) so test workers never share it.
    pinned_prefix = os.environ.get("FAKE_COLAB_PINNED_UV_DIR", "").strip() or "/tmp/uv-pinned"
    uv_source = os.environ.get("FAKE_COLAB_UV_SOURCE") or (
        "pinned-wheel" if pinned_prefix in invoked_as else "image"
    )
    if (
        invoked_as.startswith(pinned_prefix)
        and state_dir_raw
        and not (state_dir / ".pinned_uv_installed").is_file()
        and not (state_dir / ".pinned_uv_downloading").is_file()
    ):
        non_pinned_path = os.pathsep.join(
            p
            for p in os.environ.get("PATH", "").split(os.pathsep)
            if p and p != "/tmp/uv-pinned"
        )
        fallback_uv = shutil.which("uv", path=non_pinned_path)
        invoked_as = fallback_uv or "uv"
        uv_source = "image"

    uv_calls_log = state_dir / "uv_calls.jsonl"
    with uv_calls_log.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(
            json.dumps(
                {
                    "args": list(argv),
                    "uv_bin": invoked_as,
                    "uv_source": uv_source,
                    "path": os.environ.get("PATH"),
                    "uv_system_python": os.environ.get("UV_SYSTEM_PYTHON"),
                    "ts": time.time(),
                }
            )
            + "\n"
        )

    plant = os.environ.get("FAKE_COLAB_PROBE_PLANT_STDOUT", "")
    plant_sfx = f" {plant}" if plant else ""

    if argv == ["--version"]:
        if uv_source == "pinned-wheel":
            state_pver = uv_state.get("pinned_uv_binary_version")
            if state_pver and invoked_as.startswith("/tmp/uv-pinned"):
                ver = str(state_pver)
            else:
                ver = os.environ.get("FAKE_COLAB_PINNED_UV_BINARY_VERSION", "uv 0.13.5")
            if state_dir_raw and invoked_as.startswith("/tmp/uv-pinned"):
                try:
                    if ver.startswith("uv 0.13.5"):
                        (state_dir / ".pinned_uv_installed").write_text(
                            "1\n", encoding="utf-8"
                        )
                    else:
                        (state_dir / ".pinned_uv_downloading").unlink(missing_ok=True)
                        (state_dir / ".pinned_uv_installed").unlink(missing_ok=True)
                except OSError:
                    pass
        else:
            ver = os.environ.get(
                "FAKE_COLAB_UV_VERSION", str(uv_state.get("version") or "uv 0.13.5")
            )
        sys.stdout.write(f"{ver}{plant_sfx}\n")
        return 0

    if argv in (
        ["python", "find", "3.15"],
        ["python", "find", "--no-python-downloads", "3.15"],
        ["python", "find", "--no-python-downloads", "cpython-3.15.1-linux-x86_64-gnu"],
        [
            "python",
            "find",
            "--no-python-downloads",
            "--managed-python",
            "cpython-3.15.1-linux-x86_64-gnu",
        ],
    ):
        inst_dir_env = os.environ.get("UV_PYTHON_INSTALL_DIR", "").strip()
        if inst_dir_env and Path(inst_dir_env).is_dir():
            installed_bins = [
                b
                for b in sorted(Path(inst_dir_env).glob("cpython-3.15*/bin/python3.15"))
                if (b.parent.parent / ".fake_uv_installed").is_file()
            ]
            if installed_bins:
                sys.stdout.write(f"{installed_bins[0]}{plant_sfx}\n")
                return 0

        no_315_env = os.environ.get("FAKE_COLAB_UV_NO_PYTHON_315")
        if no_315_env is not None:
            found = no_315_env != "1"
        else:
            found = bool(uv_state.get("python_315_found", True))
        if not found:
            err_msg = os.environ.get(
                "FAKE_COLAB_UV_STDERR",
                str(
                    uv_state.get("python_315_stderr")
                    or DEFAULT_FAKE_UV_MISSING_315_STDERR
                ),
            )
            if not err_msg.endswith("\n"):
                err_msg += "\n"
            sys.stderr.write(err_msg)
            return 2
        py_path = os.environ.get(
            "FAKE_COLAB_UV_PYTHON_315_PATH",
            str(uv_state.get("python_315_path") or DEFAULT_FAKE_UV_PYTHON_315_PATH),
        )
        sys.stdout.write(f"{py_path}{plant_sfx}\n")
        return 0

    if argv[:2] == ["python", "install"] and "--help" not in argv:
        sleep_s = float(uv_state.get("install_sleep_seconds") or 0.0)
        if sleep_s > 0:
            time.sleep(sleep_s)

        if uv_state.get("install_fail") or os.environ.get("FAKE_COLAB_UV_INSTALL_FAIL") == "1":
            err_msg = str(
                uv_state.get("install_stderr")
                or "error: Failed to install cpython-3.15.1-linux-x86_64-gnu"
            )
            if not err_msg.endswith("\n"):
                err_msg += "\n"
            sys.stderr.write(err_msg)
            return int(uv_state.get("install_rc") or 1)

        inst_dir_str = os.environ.get("UV_PYTHON_INSTALL_DIR", "")
        mirror_url = os.environ.get("UV_PYTHON_INSTALL_MIRROR", "")
        idx = 2
        while idx < len(argv):
            if argv[idx] == "--install-dir" and idx + 1 < len(argv):
                inst_dir_str = argv[idx + 1]
                idx += 2
                continue
            if argv[idx] == "--mirror" and idx + 1 < len(argv):
                mirror_url = argv[idx + 1]
                idx += 2
                continue
            idx += 1

        if not mirror_url.startswith("file://"):
            sys.stderr.write(
                f"fake uv: expected file:// mirror URL, got {mirror_url!r}\n"
            )
            return 2

        parsed_uri = urllib.parse.urlparse(mirror_url)
        mirror_path = Path(urllib.parse.unquote(parsed_uri.path))
        build_tag = os.environ.get("UV_PYTHON_CPYTHON_BUILD", "20261021")
        archive_file = (
            mirror_path
            / build_tag
            / "cpython-3.15.1+20261021-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
        )
        if not archive_file.is_file():
            sys.stderr.write(
                f"error: Local mirror archive not found at {archive_file}\n"
            )
            return 2

        target_root = Path(inst_dir_str) / "cpython-3.15.1-linux-x86_64-gnu"
        if target_root.exists():
            shutil.rmtree(target_root)
        target_root.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_file, "r:gz") as tf:
            for member in tf.getmembers():
                rel_name = member.name
                if rel_name.startswith("python/"):
                    rel_name = rel_name[len("python/") :]
                if not rel_name or rel_name == "python":
                    continue
                dest = target_root / rel_name
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    extracted = tf.extractfile(member)
                    if extracted is not None:
                        dest.write_bytes(extracted.read())
                    if member.mode & 0o111:
                        dest.chmod(dest.stat().st_mode | stat.S_IEXEC)

        build_json_p = target_root / "BUILD.json"
        release_val = build_tag
        if build_json_p.is_file():
            try:
                bdoc = json.loads(build_json_p.read_text(encoding="utf-8"))
                release_val = str(bdoc.get("release") or build_tag)
            except Exception:
                pass
        (target_root / "BUILD").write_text(
            f"{release_val}\n", encoding="utf-8", newline="\n"
        )
        (target_root / ".fake_uv_installed").write_text(
            "1\n", encoding="utf-8", newline="\n"
        )
        sys.stderr.write(f"Installed Python 3.15.1 in 0.05s{plant_sfx}\n")
        return 0

    if argv[:1] == ["venv"] and "--python" in argv:
        if os.environ.get("FAKE_COLAB_FAIL_UV_VENV") == "1" or uv_state.get("venv_fail"):
            err_msg = os.environ.get(
                "FAKE_COLAB_UV_STDERR", "error: Failed to create virtualenv\n"
            )
            out_msg = os.environ.get(
                "FAKE_COLAB_UV_STDOUT", "Creating virtual environment\n"
            )
            sys.stdout.write(out_msg)
            sys.stderr.write(err_msg)
            return 2

        py_interp = ""
        venv_dir_str = ""
        has_clear = os.environ.get("UV_VENV_CLEAR", "").lower() in ("1", "true")
        has_system = os.environ.get("UV_SYSTEM_PYTHON", "").lower() in ("1", "true")
        idx = 1
        while idx < len(argv):
            tok = argv[idx]
            if tok == "--python" and idx + 1 < len(argv):
                py_interp = argv[idx + 1]
                idx += 2
                continue
            if tok == "--clear":
                has_clear = True
                idx += 1
                continue
            if tok == "--system":
                has_system = True
                idx += 1
                continue
            if not tok.startswith("-"):
                venv_dir_str = tok
            idx += 1

        if not py_interp or not venv_dir_str:
            sys.stderr.write(f"fake uv: unexpected arguments: {argv!r}\n")
            return 98

        if has_system:
            sys.stderr.write(
                "warning: The `--system` flag has no effect, `uv venv` always ignores virtual environments "
                "when finding a Python interpreter; did you mean `--no-managed-python`?\n"
            )

        venv_path = Path(venv_dir_str)
        sys.stderr.write(
            f"Using CPython 3.15.1 interpreter at: {py_interp}\n"
            f"Creating virtual environment at: {venv_path}\n"
        )
        if venv_path.is_file() or venv_path.is_symlink():
            sys.stderr.write(
                f"error: Failed to create virtual environment\n"
                f"  Caused by: File exists at `{venv_path}`\n"
            )
            return 2

        if (
            venv_path.is_dir()
            and ((venv_path / "pyvenv.cfg").is_file() or any(venv_path.iterdir()))
            and not has_clear
        ):
            sys.stderr.write(
                f"error: Failed to create virtual environment\n"
                f"  Caused by: A virtual environment already exists at: {venv_path}\n\n"
                f"  hint: Use the `--clear` flag or set `UV_VENV_CLEAR=1` to replace the existing virtual environment\n"
            )
            return 2

        if venv_path.is_dir() and has_clear:
            shutil.rmtree(venv_path)

        bin_dir = venv_path / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        (venv_path / "pyvenv.cfg").write_text(
            f"home = {Path(py_interp).parent}\n"
            "implementation = CPython\n"
            "uv = 0.13.5\n"
            "version_info = 3.15.1\n"
            "include-system-site-packages = false\n",
            encoding="utf-8",
            newline="\n",
        )
        is_d3_venv = venv_path.name == "venv-d3"
        hf_hub_ver = "1.33.0" if is_d3_venv else "2.4.0"
        hf_xet_ver = "1.8.0"
        site_pkgs = venv_path / "site-packages"
        hf_pkg_dir = site_pkgs / "huggingface_hub"
        xet_pkg_dir = site_pkgs / "hf_xet"
        torch_pkg_dir = site_pkgs / "torch"
        hf_pkg_dir.mkdir(parents=True, exist_ok=True)
        xet_pkg_dir.mkdir(parents=True, exist_ok=True)
        torch_pkg_dir.mkdir(parents=True, exist_ok=True)
        if not (hf_pkg_dir / "__init__.py").is_file():
            (hf_pkg_dir / "__init__.py").write_text(
                "import json, os, pathlib, time\n"
                f"__version__ = {hf_hub_ver!r}\n"
                "_IS_FAKE_HF_HUB = bool(\n"
                "    os.environ.get('FAKE_COLAB_CUT_MID_JUDGE')\n"
                "    or os.environ.get('FAKE_COLAB_VM_STEP_DELAY_SECONDS')\n"
                "    or os.environ.get('FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD') == '1'\n"
                "    or os.environ.get('FAKE_COLAB_HF_FAIL_REPO')\n"
                "    or os.environ.get('VM_DRYRUN') != '1'\n"
                ")\n"
                "def snapshot_download(repo_id, revision=None, cache_dir=None, **kw):\n"
                "    st_dir = os.environ.get('FAKE_COLAB_STATE_DIR', '')\n"
                "    if st_dir:\n"
                "        hf_log = pathlib.Path(st_dir) / 'hf_calls.jsonl'\n"
                "        hf_log.parent.mkdir(parents=True, exist_ok=True)\n"
                "        with hf_log.open('a', encoding='utf-8', newline='\\n') as _hff:\n"
                "            _hff.write(json.dumps({'repo_id': str(repo_id), 'revision': str(revision or ''), 'cache_dir': str(cache_dir or ''), 'max_workers': kw.get('max_workers'), 'hf_xet_num_concurrent_range_gets': os.environ.get('HF_XET_NUM_CONCURRENT_RANGE_GETS'), 'huggingface_hub_version': __version__, 'ts': time.time()}) + '\\n')\n"
                "    delay_s = float(os.environ.get('FAKE_COLAB_VM_STEP_DELAY_SECONDS', '0') or '0')\n"
                "    if delay_s > 0:\n"
                "        time.sleep(delay_s)\n"
                "    fail_repo = os.environ.get('FAKE_COLAB_HF_FAIL_REPO', '').strip()\n"
                "    if fail_repo and (fail_repo == repo_id or fail_repo in str(repo_id)):\n"
                "        raise RuntimeError(f'Revision Not Found for {repo_id}@{revision}')\n"
                "    cut_target = os.environ.get('FAKE_COLAB_CUT_MID_JUDGE', '').strip()\n"
                "    if cut_target and (cut_target == '1' or cut_target == repo_id or cut_target in str(repo_id)):\n"
                "        if os.environ.get('FAKE_COLAB_CUT_MID_JUDGE_HARD') == '1':\n"
                "            os._exit(137)\n"
                "        raise RuntimeError(f'Simulated cut mid-judge for {repo_id}')\n"
                "    owner, repo = str(repo_id).split('/', 1)\n"
                "    base = pathlib.Path(cache_dir) if cache_dir else (pathlib.Path(os.environ.get('HF_HOME', '.')) / 'hub')\n"
                "    snap_dir = base / f'models--{owner}--{repo}' / 'snapshots' / str(revision or 'main')\n"
                "    return str(snap_dir)\n",
                encoding="utf-8",
                newline="\n",
            )
        if not (xet_pkg_dir / "__init__.py").is_file():
            (xet_pkg_dir / "__init__.py").write_text(
                f"__version__ = {hf_xet_ver!r}\n",
                encoding="utf-8",
                newline="\n",
            )
        if not (torch_pkg_dir / "__init__.py").is_file():
            (torch_pkg_dir / "__init__.py").write_text(
                "import os, time, types\n"
                "__version__ = '2.14.1+cu132'\n"
                "float16 = 'float16'\n"
                "bfloat16 = 'bfloat16'\n"
                "_DETERMINISTIC_ALGORITHMS = True\n"
                "_DETERMINISTIC_WARN_ONLY = False\n"
                "def use_deterministic_algorithms(mode, *, warn_only=False):\n"
                "    global _DETERMINISTIC_ALGORITHMS, _DETERMINISTIC_WARN_ONLY\n"
                "    _DETERMINISTIC_ALGORITHMS = bool(mode)\n"
                "    _DETERMINISTIC_WARN_ONLY = bool(warn_only)\n"
                "def are_deterministic_algorithms_enabled():\n"
                "    return _DETERMINISTIC_ALGORITHMS\n"
                "def is_deterministic_algorithms_warn_only_enabled():\n"
                "    return _DETERMINISTIC_WARN_ONLY\n"
                "def manual_seed(_seed):\n"
                "    return None\n"
                "class _NullCtx:\n"
                "    def __enter__(self):\n"
                "        return None\n"
                "    def __exit__(self, *_a):\n"
                "        return None\n"
                "def no_grad():\n"
                "    return _NullCtx()\n"
                "def inference_mode():\n"
                "    return _NullCtx()\n"
                "class backends:\n"
                "    class cuda:\n"
                "        class matmul:\n"
                "            allow_tf32 = False\n"
                "        _flash = False\n"
                "        _mem_eff = False\n"
                "        _math = True\n"
                "        @classmethod\n"
                "        def enable_flash_sdp(cls, val):\n"
                "            cls._flash = bool(val)\n"
                "        @classmethod\n"
                "        def enable_mem_efficient_sdp(cls, val):\n"
                "            cls._mem_eff = bool(val)\n"
                "        @classmethod\n"
                "        def enable_math_sdp(cls, val):\n"
                "            cls._math = bool(val)\n"
                "        @classmethod\n"
                "        def flash_sdp_enabled(cls):\n"
                "            return cls._flash\n"
                "        @classmethod\n"
                "        def mem_efficient_sdp_enabled(cls):\n"
                "            return cls._mem_eff\n"
                "        @classmethod\n"
                "        def math_sdp_enabled(cls):\n"
                "            return cls._math\n"
                "    class cudnn:\n"
                "        allow_tf32 = False\n"
                "        benchmark = False\n"
                "        deterministic = True\n"
                "class version:\n"
                "    cuda = '13.2'\n"
                "def _active_mode():\n"
                "    target_venv = os.environ.get('FAKE_COLAB_GPU_GATE_VENV', '').strip()\n"
                "    this_venv = os.environ.get('RDLAB_ACTIVE_VENV_NAME', '').strip()\n"
                "    if target_venv and this_venv and target_venv != this_venv:\n"
                "        return 'ok'\n"
                "    raw = os.environ.get('FAKE_COLAB_GPU_GATE_MODE', '').strip()\n"
                "    if raw:\n"
                "        return raw\n"
                "    if os.environ.get('RDLAB_MODE', '').strip() == 'stub':\n"
                "        return 'not_available'\n"
                "    return 'ok'\n"
                "class _FakeTensor:\n"
                "    def __init__(self, shape, dtype=None, device=None, finite=True, vals=None):\n"
                "        self.shape = tuple(shape)\n"
                "        self.dtype = dtype\n"
                "        self.device = device\n"
                "        self._finite = finite\n"
                "        self.vals = vals if vals is not None else [[1, 3]]\n"
                "    def all(self):\n"
                "        return self._finite\n"
                "    def to(self, _dev):\n"
                "        return self\n"
                "    def float(self):\n"
                "        return self\n"
                "    def tolist(self):\n"
                "        return self.vals\n"
                "    def item(self):\n"
                "        return float(self.vals[0]) if isinstance(self.vals, list) and self.vals else 0.0\n"
                "    def __getitem__(self, idx):\n"
                "        if isinstance(idx, tuple):\n"
                "            r, sl = idx\n"
                "            row_v = self.vals[r] if isinstance(self.vals, list) and isinstance(self.vals[0], list) else self.vals\n"
                "            sub = row_v[sl]\n"
                "            return _FakeTensor((len(sub),), vals=sub)\n"
                "        v = self.vals[idx]\n"
                "        if isinstance(v, (int, float)):\n"
                "            return _FakeTensor((1,), vals=[v])\n"
                "        return _FakeTensor((len(v),), vals=v)\n"
                "def log_softmax(t, dim=-1):\n"
                "    return t\n"
                "def topk(t, k=5):\n"
                "    arr = t.vals if isinstance(t.vals, list) else [-0.1, -2.2, -8.0, -8.0, -8.0]\n"
                "    use_k = min(int(k), len(arr))\n"
                "    return (_FakeTensor((use_k,), vals=arr[:use_k]), _FakeTensor((use_k,), vals=list(range(use_k))))\n"
                "def ones(shape, dtype=None, device=None):\n"
                "    return _FakeTensor(shape, dtype=dtype, device=device, finite=True)\n"
                "def matmul(a, b):\n"
                "    mode = _active_mode()\n"
                "    if mode in ('sleep', 'timeout'):\n"
                "        delay = float(os.environ.get('FAKE_COLAB_GPU_GATE_SLEEP_SECONDS', '5.0') or '5.0')\n"
                "        time.sleep(delay)\n"
                "    if mode == 'exception':\n"
                "        secret = os.environ.get('FAKE_COLAB_GPU_GATE_SECRET', '')\n"
                "        raise RuntimeError(f'CUDA error: device-side assert triggered {secret}'.strip())\n"
                "    finite = (mode != 'non_finite')\n"
                "    return _FakeTensor((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device, finite=finite)\n"
                "def isfinite(t):\n"
                "    return _FakeTensor(t.shape, dtype='bool', device=t.device, finite=getattr(t, '_finite', True))\n"
                "class _FakeCudart:\n"
                "    def cudaRuntimeGetVersion(self):\n"
                "        return 13020\n"
                "class device:\n"
                "    def __init__(self, spec):\n"
                "        if isinstance(spec, device):\n"
                "            self.type = spec.type\n"
                "            self.index = spec.index\n"
                "            return\n"
                "        s = str(spec)\n"
                "        if ':' in s:\n"
                "            t, idx_s = s.split(':', 1)\n"
                "            self.type = t\n"
                "            self.index = int(idx_s)\n"
                "        else:\n"
                "            self.type = s\n"
                "            self.index = None\n"
                "    def __repr__(self):\n"
                "        if self.index is not None:\n"
                "            return f\"device(type='{self.type}', index={self.index})\"\n"
                "        return f\"device(type='{self.type}')\"\n"
                "    def __str__(self):\n"
                "        return f'{self.type}:{self.index}' if self.index is not None else self.type\n"
                "class _FakeCudaUtils:\n"
                "    @staticmethod\n"
                "    def _get_device_index(dev, optional=False, allow_cpu=False):\n"
                "        if isinstance(dev, int):\n"
                "            return dev\n"
                "        d = dev if isinstance(dev, device) else device(dev)\n"
                "        if d.type == 'cpu':\n"
                "            if allow_cpu:\n"
                "                return -1\n"
                "            raise ValueError(f'Expected a cuda device, but got: {dev}')\n"
                "        if d.type != 'cuda':\n"
                "            raise ValueError(f'Expected a cuda device, but got: {dev}')\n"
                "        if d.index is None:\n"
                "            if optional:\n"
                "                return 0\n"
                "            raise ValueError(f'Expected a torch.device with a specified index or an integer, but got:{d}')\n"
                "        return int(d.index)\n"
                "class cuda:\n"
                "    _utils = _FakeCudaUtils\n"
                "    @staticmethod\n"
                "    def is_initialization_complete():\n"
                "        return False\n"
                "    @staticmethod\n"
                "    def is_available():\n"
                "        return _active_mode() != 'not_available'\n"
                "    @staticmethod\n"
                "    def device_count():\n"
                "        return 0 if _active_mode() == 'not_available' else 1\n"
                "    @staticmethod\n"
                "    def get_device_name(idx=0):\n"
                "        if _active_mode() == 'not_available':\n"
                "            raise RuntimeError('Torch not compiled with CUDA enabled')\n"
                "        return 'NVIDIA A100-SXM4-40GB'\n"
                "    @staticmethod\n"
                "    def mem_get_info(idx=0):\n"
                "        if _active_mode() == 'not_available':\n"
                "            raise RuntimeError('CUDA not available')\n"
                "        return (41875931136, 42949672960)\n"
                "    @staticmethod\n"
                "    def synchronize():\n"
                "        if _active_mode() == 'not_available':\n"
                "            raise RuntimeError('CUDA not available')\n"
                "    @staticmethod\n"
                "    def cudart():\n"
                "        return _FakeCudart()\n",
                encoding="utf-8",
                newline="\n",
            )
        tf_ver = "5.17.0" if is_d3_venv else "5.19.0"
        tf_pkg_dir = site_pkgs / "transformers"
        tf_pkg_dir.mkdir(parents=True, exist_ok=True)
        if not (tf_pkg_dir / "__init__.py").is_file():
            (tf_pkg_dir / "__init__.py").write_text(
                "import hashlib, types, torch\n"
                f"__version__ = {tf_ver!r}\n"
                "_VOCAB = {'<unk>': 0, '<s>': 1, '</s>': 2, 'yes': 3, 'no': 4, '<feedback>ok</feedback><score>Pass</score>': 5}\n"
                "class AutoTokenizer:\n"
                "    pad_token_id = 0\n"
                "    eos_token_id = 2\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, *_a, **_kw):\n"
                "        return cls()\n"
                "    def get_vocab(self):\n"
                "        return dict(_VOCAB)\n"
                "    def __call__(self, text, **_kw):\n"
                "        h = int.from_bytes(hashlib.sha256(str(text).encode('utf-8')).digest()[:2], 'big') % 2\n"
                "        ids = [[1, 3 + h]]\n"
                "        return {'input_ids': torch._FakeTensor((1, 2), vals=ids), 'attention_mask': torch._FakeTensor((1, 2), vals=[[1, 1]])}\n"
                "    def convert_ids_to_tokens(self, ids):\n"
                "        inv = {v: k for k, v in _VOCAB.items()}\n"
                "        return inv.get(ids, '<unk>') if isinstance(ids, int) else [inv.get(int(i), '<unk>') for i in ids]\n"
                "    def convert_tokens_to_string(self, toks):\n"
                "        return ''.join(toks)\n"
                "    def decode(self, ids, **_kw):\n"
                "        return ' '.join(self.convert_ids_to_tokens(ids))\n"
                "class AutoModelForCausalLM:\n"
                "    generation_config = types.SimpleNamespace(eos_token_id=[2])\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, *_a, **_kw):\n"
                "        return cls()\n"
                "    def to(self, _dev):\n"
                "        return self\n"
                "    def eval(self):\n"
                "        return self\n"
                "    def generate(self, **kw):\n"
                "        in_ids = kw['input_ids'].tolist()[0]\n"
                "        out_tok = 3 if (sum(in_ids) % 2 == 0) else 4\n"
                "        seq = [*in_ids, out_tok]\n"
                "        logits = [-8.0, -8.0, -8.0, -0.15, -2.05, -8.0] if out_tok == 3 else [-8.0, -8.0, -8.0, -2.05, -0.15, -8.0]\n"
                "        streamer = kw.get('streamer')\n"
                "        if streamer is not None and hasattr(streamer, 'put'):\n"
                "            streamer.put(torch._FakeTensor((1, 1), vals=[[out_tok]]))\n"
                "        if streamer is not None and hasattr(streamer, 'end'):\n"
                "            streamer.end()\n"
                "        return types.SimpleNamespace(\n"
                "            sequences=torch._FakeTensor((1, len(seq)), vals=[seq]),\n"
                "            scores=[torch._FakeTensor((1, len(logits)), vals=[logits])],\n"
                "        )\n"
                "class AutoModel:\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, *_a, **_kw):\n"
                "        return cls()\n"
                "    def system_one(self, *, state, questions):\n"
                "        return {'answers': {'supported': {'noul': 0.85}}}\n",
                encoding="utf-8",
                newline="\n",
            )
        site_pkgs_posix = site_pkgs.as_posix()
        venv_py = bin_dir / "python"
        venv_py.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "$1" == "--version" || "$1" == "-V" ]]; then\n'
            '  echo "Python 3.15.1"\n'
            "  exit 0\n"
            "fi\n"
            f'export RDLAB_ACTIVE_VENV_NAME="{venv_path.name}"\n'
            f'export PYTHONPATH="{site_pkgs_posix}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
            'if [[ -n "${FAKE_COLAB_FAIL_JUDGE_CHILD:-}" && "$*" == *"bakeoff_judge"* ]]; then\n'
            '  _prev=""\n'
            '  _jid=""\n'
            '  for _arg in "$@"; do\n'
            '    if [[ "$_prev" == "--judge" ]]; then\n'
            '      _jid="$_arg"\n'
            "      break\n"
            "    fi\n"
            '    _prev="$_arg"\n'
            "  done\n"
            '  _target="${FAKE_COLAB_FAIL_JUDGE_CHILD%%:*}"\n'
            '  _rest="${FAKE_COLAB_FAIL_JUDGE_CHILD#*:}"\n'
            '  if [[ -n "$_jid" && "$_jid" == "$_target" ]]; then\n'
            '    _mode="${_rest%%:*}"\n'
            '    _msg="${_rest#*:}"\n'
            '    if [[ "$_mode" == "sigkill" ]]; then\n'
            "      kill -9 $$\n"
            '    elif [[ "$_mode" == "exit137" ]]; then\n'
            "      exit 137\n"
            '    elif [[ "$_mode" == "exit1" ]]; then\n'
            '      echo "${_msg:-Simulated child failure}" >&2\n'
            "      exit 1\n"
            "    fi\n"
            "  fi\n"
            "fi\n"
            'if [[ -n "${FAKE_COLAB_FAIL_DOWNLOAD_CHILD:-}" && "$*" == *"judge_download"* ]]; then\n'
            '  _prev=""\n'
            '  _jid=""\n'
            '  _done_f=""\n'
            '  for _arg in "$@"; do\n'
            '    if [[ "$_prev" == "--judge" ]]; then\n'
            '      _jid="$_arg"\n'
            '    elif [[ "$_prev" == "--done-file" ]]; then\n'
            '      _done_f="$_arg"\n'
            "    fi\n"
            '    _prev="$_arg"\n'
            "  done\n"
            '  _target="${FAKE_COLAB_FAIL_DOWNLOAD_CHILD%%:*}"\n'
            '  _rest="${FAKE_COLAB_FAIL_DOWNLOAD_CHILD#*:}"\n'
            '  if [[ -n "$_jid" && "$_jid" == "$_target" ]]; then\n'
            '    _mode="${_rest%%:*}"\n'
            '    _msg="${_rest#*:}"\n'
            '    if [[ "$_mode" == "sigkill" ]]; then\n'
            "      kill -9 $$\n"
            '    elif [[ "$_mode" == "exit137" ]]; then\n'
            "      exit 137\n"
            '    elif [[ "$_mode" == "exit1" ]]; then\n'
            '      echo "${_msg:-Simulated download child failure}" >&2\n'
            "      exit 1\n"
            '    elif [[ "$_mode" == "missing_done" ]]; then\n'
            "      exit 0\n"
            '    elif [[ "$_mode" == "malformed_done" ]]; then\n'
            '      mkdir -p "$(dirname "$_done_f")"\n'
            '      printf "{malformed-download-done\\n" > "$_done_f"\n'
            "      exit 0\n"
            "    fi\n"
            "  fi\n"
            "fi\n"
            'if [[ "$*" == *"bakeoff_judge"* && "$*" != *"--stub"* && "$*" != *"--fake-transport"* ]]; then\n'
            f'  exec "{sys.executable}" "$@" --stub\n'
            "fi\n"
            f'exec "{sys.executable}" "$@"\n',
            encoding="utf-8",
            newline="\n",
        )
        venv_py.chmod(venv_py.stat().st_mode | stat.S_IEXEC)
        (venv_path / ".installed_packages.json").write_text(
            "{}\n", encoding="utf-8", newline="\n"
        )
        return 0

    if argv[:2] == ["pip", "install"]:
        venv_python = ""
        req_file = ""
        require_hashes = False
        no_binary_pkgs: set[str] = set()
        only_binary_all = False
        idx = 2
        while idx < len(argv):
            tok = argv[idx]
            if tok == "--python" and idx + 1 < len(argv):
                venv_python = argv[idx + 1]
                idx += 2
                continue
            if tok in ("-r", "--requirement") and idx + 1 < len(argv):
                req_file = argv[idx + 1]
                idx += 2
                continue
            if tok == "--require-hashes":
                require_hashes = True
                idx += 1
                continue
            if tok == "--only-binary" and idx + 1 < len(argv):
                if argv[idx + 1] == ":all:":
                    only_binary_all = True
                idx += 2
                continue
            if tok == "--no-binary" and idx + 1 < len(argv):
                no_binary_pkgs.add(argv[idx + 1].lower())
                idx += 2
                continue
            idx += 1

        if not venv_python or not req_file:
            sys.stderr.write(f"fake uv: unexpected arguments: {argv!r}\n")
            return 98

        venv_py_path = Path(venv_python)
        if not venv_py_path.is_file():
            sys.stderr.write(
                f"error: Failed to inspect Python interpreter from `--python` at `{venv_py_path}`\n"
            )
            return 2

        req_path = Path(req_file)
        if not req_path.is_file():
            sys.stderr.write(f"error: File not found: `{req_path}`\n")
            return 2

        entries = _parse_fake_lock_entries(req_path)
        expected_pyyaml_sha = (
            "d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f"
        )
        sim_mismatch = (
            bool(uv_state.get("simulate_hash_mismatch"))
            or os.environ.get("FAKE_COLAB_UV_HASH_MISMATCH") == "1"
        )
        for name, ver, hashes, dists in entries:
            if require_hashes and not hashes:
                sys.stderr.write(
                    f"error: In `--require-hashes` mode, all requirements must have their versions pinned with `==` and a `--hash` specified: `{name}=={ver}`\n"
                )
                return 2
            if require_hashes and (
                sim_mismatch
                or (name == "pyyaml" and hashes != [expected_pyyaml_sha])
            ):
                exp_h = hashes[0] if hashes else expected_pyyaml_sha
                sys.stderr.write(
                    f"error: Hash mismatch for `{name}=={ver}`\n\n"
                    f"Expected:\n    sha256:{exp_h}\n\n"
                    f"Computed:\n    sha256:{'0' * 64}\n"
                )
                return 2
            if only_binary_all and name not in no_binary_pkgs:
                if any(not d.endswith(".whl") for d in dists):
                    sys.stderr.write(
                        f"error: Cannot build source distribution for `{name}=={ver}` under `--only-binary :all:`\n"
                    )
                    return 2

        venv_root = venv_py_path.parent.parent
        pkg_db_path = venv_root / ".installed_packages.json"
        installed_db: dict[str, Any] = {}
        if pkg_db_path.is_file():
            try:
                installed_db = json.loads(pkg_db_path.read_text(encoding="utf-8"))
            except Exception:
                installed_db = {}

        wheel_cache_path = state_dir / "uv_wheel_cache.json"
        wheel_cache: dict[str, Any] = {}
        if wheel_cache_path.is_file():
            try:
                wheel_cache = json.loads(wheel_cache_path.read_text(encoding="utf-8"))
            except Exception:
                wheel_cache = {}

        for name, ver, hashes, dists in entries:
            is_sdist = any(d.endswith(".tar.gz") for d in dists)
            installed_db[name] = {
                "version": ver,
                "hashes": hashes,
                "distributions": dists,
                "source": "sdist_built" if is_sdist else "wheel",
                "installed_by": "pip_install",
                "kept_by_sync": False,
            }
            if is_sdist:
                wheel_cache[f"{name}=={ver}"] = {
                    "hashes": hashes,
                    "distributions": dists,
                    "built_from_sdist": True,
                }

        pkg_db_path.write_text(
            json.dumps(installed_db, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        wheel_cache_path.write_text(
            json.dumps(wheel_cache, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        sys.stderr.write(f"Installed {len(entries)} package(s)\n")
        return 0

    if argv[:2] == ["pip", "sync"]:
        venv_python = ""
        lock_file = ""
        require_hashes = False
        no_binary_pkgs: set[str] = set()
        only_binary_all = False
        extra_index_urls: list[str] = []
        index_strategy = ""
        idx = 2
        while idx < len(argv):
            tok = argv[idx]
            if tok == "--python" and idx + 1 < len(argv):
                venv_python = argv[idx + 1]
                idx += 2
                continue
            if tok == "--require-hashes":
                require_hashes = True
                idx += 1
                continue
            if tok == "--only-binary" and idx + 1 < len(argv):
                if argv[idx + 1] == ":all:":
                    only_binary_all = True
                idx += 2
                continue
            if tok == "--no-binary" and idx + 1 < len(argv):
                no_binary_pkgs.add(argv[idx + 1].lower())
                idx += 2
                continue
            if tok == "--extra-index-url" and idx + 1 < len(argv):
                extra_index_urls.append(argv[idx + 1])
                idx += 2
                continue
            if tok == "--index-strategy" and idx + 1 < len(argv):
                index_strategy = argv[idx + 1]
                idx += 2
                continue
            if not tok.startswith("-"):
                lock_file = tok
            idx += 1

        if not venv_python or not lock_file:
            sys.stderr.write(f"fake uv: unexpected arguments: {argv!r}\n")
            return 98

        venv_py_path = Path(venv_python)
        if not venv_py_path.is_file():
            sys.stderr.write(
                f"error: Failed to inspect Python interpreter from `--python` at `{venv_py_path}`\n"
            )
            return 2

        lock_path = Path(lock_file)
        if not lock_path.is_file():
            sys.stderr.write(f"error: File not found: `{lock_path}`\n")
            return 2

        entries = _parse_fake_lock_entries(lock_path)
        expected_pyyaml_sha = (
            "d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f"
        )
        sim_mismatch = (
            bool(uv_state.get("simulate_hash_mismatch"))
            or os.environ.get("FAKE_COLAB_UV_HASH_MISMATCH") == "1"
        )
        for name, ver, hashes, dists in entries:
            if require_hashes and not hashes:
                sys.stderr.write(
                    f"error: In `--require-hashes` mode, all requirements must have their versions pinned with `==` and a `--hash` specified: `{name}=={ver}`\n"
                )
                return 2
            if require_hashes and (
                sim_mismatch
                or (name == "pyyaml" and hashes != [expected_pyyaml_sha])
            ):
                exp_h = hashes[0] if hashes else expected_pyyaml_sha
                sys.stderr.write(
                    f"error: Hash mismatch for `{name}=={ver}`\n\n"
                    f"Expected:\n    sha256:{exp_h}\n\n"
                    f"Computed:\n    sha256:{'0' * 64}\n"
                )
                return 2
            if "+cu132" in ver and (
                "https://download.pytorch.org/whl/cu132" not in extra_index_urls
                or index_strategy != "unsafe-best-match"
            ):
                sys.stderr.write(
                    f"error: No solution found when resolving: `{name}=={ver}` was not found on the package index\n"
                )
                return 2
            if only_binary_all and name not in no_binary_pkgs:
                if any(not d.endswith(".whl") for d in dists):
                    sys.stderr.write(
                        f"error: Cannot build source distribution for `{name}=={ver}` under `--only-binary :all:`\n"
                    )
                    return 2

        venv_root = venv_py_path.parent.parent
        pkg_db_path = venv_root / ".installed_packages.json"
        existing_db: dict[str, Any] = {}
        if pkg_db_path.is_file():
            try:
                existing_db = json.loads(pkg_db_path.read_text(encoding="utf-8"))
            except Exception:
                existing_db = {}

        wheel_cache_path = state_dir / "uv_wheel_cache.json"
        wheel_cache: dict[str, Any] = {}
        if wheel_cache_path.is_file():
            try:
                wheel_cache = json.loads(wheel_cache_path.read_text(encoding="utf-8"))
            except Exception:
                wheel_cache = {}

        new_db: dict[str, Any] = {}
        for name, ver, hashes, dists in entries:
            prior = existing_db.get(name)
            if isinstance(prior, dict) and prior.get("version") == ver:
                kept = dict(prior)
                kept["kept_by_sync"] = True
                new_db[name] = kept
            else:
                is_sdist = any(d.endswith(".tar.gz") for d in dists)
                from_cache = f"{name}=={ver}" in wheel_cache
                new_db[name] = {
                    "version": ver,
                    "hashes": hashes,
                    "distributions": dists,
                    "source": (
                        "cached_sdist_wheel"
                        if (is_sdist and from_cache)
                        else ("sdist_built" if is_sdist else "wheel")
                    ),
                    "installed_by": "pip_sync",
                    "kept_by_sync": False,
                }

        pkg_db_path.write_text(
            json.dumps(new_db, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        sys.stderr.write(f"Synced {len(entries)} package(s)\n")
        return 0

    sys.stderr.write(f"fake uv: unexpected arguments: {argv!r}\n")
    return 98


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


# `nvidia-smi --query-gpu=name` per session hardware, as the real driver names these cards.
FAKE_GPU_NAMES: dict[str, str] = {
    "A100": "NVIDIA A100-SXM4-40GB",
    "A100-80": "NVIDIA A100-SXM4-80GB",
    "L4": "NVIDIA L4",
    "T4": "Tesla T4",
    "H100": "NVIDIA H100 80GB HBM3",
    "G4": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
}


def run_fake_nvidia_smi_name() -> int:
    """Fake `nvidia-smi --query-gpu=name --format=csv,noheader`: the name of the card the most recently created session
    holds (`sessions.json`), or `<state_dir>/nvidia_smi_gpu_name.txt` when a test writes one (a VM whose card differs
    from what the session reported). A CPU session has no driver: exit 9 like the real binary."""
    state_dir = Path(os.environ.get("FAKE_COLAB_STATE_DIR") or "/tmp/rdlab-fake-colab-state")
    override = state_dir / "nvidia_smi_gpu_name.txt"
    if override.is_file():
        print(override.read_text(encoding="utf-8").strip())
        return 0
    try:
        sessions = json.loads((state_dir / "sessions.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        sessions = []
    gpu = str(sessions[-1].get("gpu", "CPU")) if sessions else "CPU"
    name = FAKE_GPU_NAMES.get(gpu)
    if name is None:
        sys.stderr.write("NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver.\n")
        return 9
    print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args[:1] == ["--fake-uv"]:
        return run_fake_uv(raw_args[1:])
    if raw_args[:1] == ["--fake-curl"]:
        return run_fake_curl(raw_args[1:])
    if raw_args[:1] == ["--fake-nvidia-smi-name"]:
        return run_fake_nvidia_smi_name()

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

    active_sid = _get_opt("-s", "--session")
    if active_sid and os.environ.get("FAKE_COLAB_ISOLATE_SESSION_FS") == "1":
        safe_sid = re.sub(r"[^A-Za-z0-9_.-]+", "_", active_sid)
        vm_fs = state_dir / f"vm_fs_{safe_sid}"
        vm_fs.mkdir(parents=True, exist_ok=True)

    sessions_lock_path = state_dir / "sessions.lock"
    sessions_lock_fh = sessions_lock_path.open("a+", encoding="utf-8")
    try:
        import fcntl

        fcntl.flock(sessions_lock_fh.fileno(), fcntl.LOCK_EX)
    except Exception:
        pass

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
        "request_timeout": os.environ.get("REQUEST_TIMEOUT"),
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
        # The service may give a different card than requested (e.g. an A100 80 GB for `--gpu A100`): a test writes
        # the hardware the new session reports into `<state_dir>/new_session_hardware.txt`.
        hw_override = state_dir / "new_session_hardware.txt"
        if hw_override.is_file() and hw_override.read_text(encoding="utf-8").strip():
            gpu = hw_override.read_text(encoding="utf-8").strip()
        sessions.append({"session_id": sid, "gpu": gpu})
        sessions_file.write_text(json.dumps(sessions), encoding="utf-8", newline="\n")
        print(f"Created session [{sid}] ({gpu})")
        return 0

    try:
        import fcntl as _fcntl

        _fcntl.flock(sessions_lock_fh.fileno(), _fcntl.LOCK_UN)
    except Exception:
        pass

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
        if (
            dst == "/content/device.py"
            and os.environ.get("FAKE_COLAB_CORRUPT_DEVICE_PY") == "1"
        ):
            vm_dst.write_text(
                vm_dst.read_text(encoding="utf-8") + "# tampered on VM\n",
                encoding="utf-8",
                newline="\n",
            )
        keep_uploads_dir = os.environ.get("FAKE_COLAB_KEEP_UPLOADS_DIR", "").strip()
        if keep_uploads_dir:
            # Test-only: keep a copy of each uploaded file so a test can inspect exactly what was staged.
            Path(keep_uploads_dir).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, Path(keep_uploads_dir) / Path(dst).name)
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
        fail_periodic_spec = os.environ.get("FAKE_COLAB_FAIL_PERIODIC_DOWNLOAD", "").strip()
        if fail_periodic_spec and dst_path.name.startswith("progress.jsonl.tmp"):
            attempts_file = state_dir / "periodic_download_attempts.txt"
            try:
                prev_attempts = int(attempts_file.read_text(encoding="utf-8").strip())
            except Exception:
                prev_attempts = 0
            attempt = prev_attempts + 1
            attempts_file.write_text(f"{attempt}\n", encoding="utf-8", newline="\n")
            should_fail = fail_periodic_spec == "all" or (
                fail_periodic_spec.isdigit() and attempt <= int(fail_periodic_spec)
            )
            if should_fail:
                sys.stderr.write(
                    f"[colab] Download failed: transient proxy error "
                    f"colab-runtime-proxy-token=FAKE_PERIODIC_PROXY_SECRET "
                    f"hf_FakePeriodicSecret99 for {src} (attempt {attempt})\n"
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

        exec_env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("RDLAB_")
            and not k.startswith("HF_TOKEN")
            and k != "HUGGING_FACE_HUB_TOKEN"
            and k != "HF_HOME"
        }

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

        if os.environ.get("FAKE_COLAB_REAL_MODE_SHIM", "").strip():
            # Real-mode shim (test-only): the VM has its own weight cache inside the fake VM root, standing in for the
            # weights a real VM downloads. It is seeded from FAKE_COLAB_VM_HF_SEED (the test's synthetic snapshots) by
            # this CLI process, never read by the cell from the launcher's local HF_HOME, whose forwarded value is
            # replaced here.
            vm_hf_home = vm_fs / "hf_home"
            hf_seed = os.environ.get("FAKE_COLAB_VM_HF_SEED", "").strip()
            if hf_seed and not vm_hf_home.exists() and Path(hf_seed).is_dir():
                shutil.copytree(hf_seed, vm_hf_home)
            vm_hf_home.mkdir(parents=True, exist_ok=True)
            exec_env["HF_HOME"] = str(vm_hf_home)
            extra_raw = os.environ.get("FAKE_COLAB_EXEC_ENV_EXTRA", "").strip()
            if extra_raw and exec_env.get("RDLAB_STEP", "run") == "run":
                # As if the launcher had forwarded these with --env (used to prove host paths are unreadable).
                for ek, ev in json.loads(extra_raw).items():
                    exec_env[str(ek)] = str(ev)

        (vm_fs / ".cache").mkdir(parents=True, exist_ok=True)
        sleep_s = float(os.environ.get("FAKE_COLAB_EXEC_SLEEP", "0"))
        step = exec_env.get("RDLAB_STEP", "run")
        if step == "alive":
            alive_log = state_dir / "alive_attempts.jsonl"
            attempt_idx = 1
            if alive_log.is_file():
                attempt_idx = (
                    len(
                        [
                            ln
                            for ln in alive_log.read_text(encoding="utf-8").splitlines()
                            if ln.strip()
                        ]
                    )
                    + 1
                )
            fail_count = int(os.environ.get("FAKE_COLAB_ALIVE_FAIL_COUNT", "0") or "0")
            failed_this_try = attempt_idx <= fail_count
            with alive_log.open("a", encoding="utf-8", newline="\n") as afh:
                afh.write(
                    json.dumps(
                        {
                            "session_id": sid,
                            "attempt": attempt_idx,
                            "failed": failed_this_try,
                            "ts": time.time(),
                        }
                    )
                    + "\n"
                )
            if failed_this_try:
                sys.stderr.write("Connection was lost\n")
                return 1
        if step == "gate" and os.environ.get("FAKE_COLAB_GATE_CONNECTION_LOST") == "1":
            sys.stderr.write("Connection was lost\n")
            return 1
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
                run_dir_sleep = vm_fs / "rdlab_run"
                run_dir_sleep.mkdir(parents=True, exist_ok=True)
                plant_leak = os.environ.get("FAKE_COLAB_PROGRESS_PLANT_LEAK", "").strip()
                sleep_prog_row: dict[str, Any] = {
                    "kind": "step_start",
                    "step": "unpack",
                    "ts": "2026-11-02T00:00:00Z",
                    "start": "2026-11-02T00:00:00Z",
                    "mem_available_bytes": 53687091200,
                    "cgroup_memory_current": 1073741824,
                    "cgroup_memory_max": None,
                }
                if plant_leak:
                    sleep_prog_row["detail"] = plant_leak
                (run_dir_sleep / "progress.jsonl").write_text(
                    json.dumps(sleep_prog_row) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                if os.environ.get("FAKE_COLAB_WRITE_PARTIAL_RUN_BEFORE_SLEEP") == "1":
                    partial_prog_rows = [
                        sleep_prog_row,
                        {
                            "kind": "judge_pass_done",
                            "judge_id": "nemotron-3-nano-4b",
                            "status": "ok",
                            "calls_count": 2,
                            "ts": "2026-11-02T00:00:05Z",
                        },
                    ]
                    (run_dir_sleep / "progress.jsonl").write_text(
                        "\n".join(json.dumps(r) for r in partial_prog_rows) + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                    logs_dir_sleep = run_dir_sleep / "logs"
                    logs_dir_sleep.mkdir(parents=True, exist_ok=True)
                    (logs_dir_sleep / "vm.log").write_text(
                        "2026-11-02T00:00:05Z [vm] completed judge=nemotron-3-nano-4b before cap stop\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                    jdone_sleep = run_dir_sleep / "judge_done"
                    jdone_sleep.mkdir(parents=True, exist_ok=True)
                    (jdone_sleep / "nemotron-3-nano-4b.json").write_text(
                        json.dumps(
                            {
                                "judge_id": "nemotron-3-nano-4b",
                                "status": "ok",
                                "device": "cuda:0",
                                "calls_count": 2,
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                    rows_sleep = run_dir_sleep / "rows"
                    rows_sleep.mkdir(parents=True, exist_ok=True)
                    (rows_sleep / "nemotron-3-nano-4b.jsonl").write_text(
                        json.dumps(
                            {
                                "id": "item-0000",
                                "judge_id": "nemotron-3-nano-4b",
                                "state": "parsed",
                                "verdict": "yes",
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                if os.environ.get("FAKE_COLAB_WRITE_RESULT_TAR_BEFORE_SLEEP") == "1":
                    import tarfile as _tarfile

                    sched_f = run_dir_sleep / "schedule.json"
                    sched_f.write_text(
                        json.dumps({"items_count": 2, "judges": []}) + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                    rec_f = run_dir_sleep / "vm_receipt.json"
                    rec_f.write_text(
                        json.dumps({"status": "running", "step": "run"}) + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                    with _tarfile.open(vm_fs / "result.tar.gz", "w:gz") as _tf:
                        _tf.add(sched_f, arcname="schedule.json", recursive=False)
                        _tf.add(rec_f, arcname="vm_receipt.json", recursive=False)
                    (vm_fs / "status.json").write_text(
                        json.dumps({"step": "run", "rc": 0, "status": "ok", "reason": "partial_before_cap"}) + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                time.sleep(sleep_s)
                sys.stderr.write(f"Execution timed out after {timeout_s}s\n")
                return 124
            kernel_code = (
                "import json, os, pathlib, shutil, subprocess, sys, time, types\n"
                "_RDLAB_VM_ROOT = os.path.abspath(os.environ['RDLAB_VM_CONTENT_DIR'])\n"
                "_REAL_MODE_SHIM = os.environ.get('FAKE_COLAB_REAL_MODE_SHIM', '').strip()\n"
                "_SHIM_STATE_DIR = os.environ.get('FAKE_COLAB_STATE_DIR', '')\n"
                "_use_fake_hf = bool(\n"
                "    os.environ.get('FAKE_COLAB_CUT_MID_JUDGE')\n"
                "    or os.environ.get('FAKE_COLAB_VM_STEP_DELAY_SECONDS')\n"
                "    or os.environ.get('FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD') == '1'\n"
                "    or os.environ.get('VM_DRYRUN') != '1'\n"
                ")\n"
                "if _use_fake_hf:\n"
                "    _fake_hf = types.ModuleType('huggingface_hub')\n"
                "    _fake_hf._IS_FAKE_HF_HUB = bool(\n"
                "        os.environ.get('FAKE_COLAB_CUT_MID_JUDGE')\n"
                "        or os.environ.get('FAKE_COLAB_VM_STEP_DELAY_SECONDS')\n"
                "        or os.environ.get('FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD') == '1'\n"
                "    )\n"
                "    def _fake_snapshot_download(repo_id, revision=None, cache_dir=None, **kw):\n"
                "        st_dir = os.environ.get('FAKE_COLAB_STATE_DIR', '')\n"
                "        if st_dir:\n"
                "            hf_log = pathlib.Path(st_dir) / 'hf_calls.jsonl'\n"
                "            with hf_log.open('a', encoding='utf-8', newline='\\n') as _hff:\n"
                "                _hff.write(json.dumps({'repo_id': str(repo_id), 'revision': str(revision or ''), 'cache_dir': str(cache_dir or ''), 'max_workers': kw.get('max_workers'), 'hf_xet_num_concurrent_range_gets': os.environ.get('HF_XET_NUM_CONCURRENT_RANGE_GETS'), 'ts': time.time()}) + '\\n')\n"
                "        delay_s = float(os.environ.get('FAKE_COLAB_VM_STEP_DELAY_SECONDS', '0') or '0')\n"
                "        if delay_s > 0:\n"
                "            time.sleep(delay_s)\n"
                "        fail_repo = os.environ.get('FAKE_COLAB_HF_FAIL_REPO', '').strip()\n"
                "        if fail_repo and (fail_repo == repo_id or fail_repo in str(repo_id)):\n"
                "            raise RuntimeError(f'Revision Not Found for {repo_id}@{revision}')\n"
                "        cut_target = os.environ.get('FAKE_COLAB_CUT_MID_JUDGE', '').strip()\n"
                "        if cut_target and (cut_target == repo_id or cut_target in str(repo_id)):\n"
                "            if os.environ.get('FAKE_COLAB_CUT_MID_JUDGE_HARD') == '1':\n"
                "                os._exit(137)\n"
                "            raise RuntimeError(f'Simulated cut mid-judge for {repo_id}')\n"
                "        owner, repo = str(repo_id).split('/', 1)\n"
                "        base = pathlib.Path(cache_dir) if cache_dir else (pathlib.Path(os.environ.get('HF_HOME', '.')) / 'hub')\n"
                "        snap_dir = base / f'models--{owner}--{repo}' / 'snapshots' / str(revision or 'main')\n"
                "        return str(snap_dir)\n"
                "    _fake_hf.snapshot_download = _fake_snapshot_download\n"
                "    sys.modules['huggingface_hub'] = _fake_hf\n"
                "def _fake_disk_usage(p):\n"
                "    ps = os.path.abspath(os.fspath(p))\n"
                "    vm_root = _RDLAB_VM_ROOT\n"
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
                "        if os.environ.get('FAKE_COLAB_UV_ABSENT') == '1' and '/tmp/uv-pinned' not in call_env.get('PATH', '').split(os.pathsep):\n"
                "            raise FileNotFoundError(2, 'No such file or directory', 'uv')\n"
                "        return _real_subproc_run(['uv', '--version'], *a, **kw)\n"
                "    if cmd_list[:2] == ['uv', 'venv']:\n"
                "        if os.environ.get('FAKE_COLAB_FAIL_UV_VENV') == '1':\n"
                "            err_msg = os.environ.get('FAKE_COLAB_UV_STDERR', 'error: Failed to create virtualenv\\n')\n"
                "            out_msg = os.environ.get('FAKE_COLAB_UV_STDOUT', 'Creating virtual environment\\n')\n"
                "            return subprocess.CompletedProcess(cmd_list, 2, stdout=out_msg, stderr=err_msg)\n"
                "    if os.environ.get('RDLAB_STEP') == 'probe':\n"
                "        if cmd_list[:3] == ['uv', 'python', 'find']:\n"
                "            if os.environ.get('FAKE_COLAB_UV_NO_PYTHON_315') == '1':\n"
                "                err_msg = os.environ.get('FAKE_COLAB_UV_STDERR', 'error: No interpreter found for Python 3.15 in managed installations or search path\\n')\n"
                "                return subprocess.CompletedProcess(cmd_list, 2, stdout='', stderr=err_msg)\n"
                "            return subprocess.CompletedProcess(cmd_list, 0, stdout='/usr/local/share/uv/python/cpython-3.15.1-linux-x86_64-gnu/bin/python3.15' + plant_sfx + '\\n', stderr='')\n"
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
                "            return subprocess.CompletedProcess(cmd_list, 0, stdout='Name: uv\\nVersion: 0.13.5\\nLocation: /usr/local/lib/python3.12/dist-packages' + plant_sfx + '\\n', stderr='')\n"
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
                "_st_uv_p = pathlib.Path(os.environ.get('FAKE_COLAB_STATE_DIR', '')) / 'uv_state.json'\n"
                "if _st_uv_p.is_file():\n"
                "    _uv_st = json.loads(_st_uv_p.read_text(encoding='utf-8'))\n"
                "    if _uv_st.get('expected_archive_sha256'):\n"
                "        src = src.replace('01d901e09ca673c99a6c9bd9ec2cfbd58dbe8813d07514c36cb73cfd20b18bd8', str(_uv_st['expected_archive_sha256']))\n"
                "    if _uv_st.get('expected_uv_wheel_sha256'):\n"
                "        src = src.replace('fa8d26e08e34c29c0c4e820cf3181757086d547240b4f4d31ff1d8bd810ca946', str(_uv_st['expected_uv_wheel_sha256']))\n"
                "ns = {'__name__': '__main__', '_PRIOR_EXEC_COUNT': "
                f"{len(prior_cells)!r}"
                "}\n"
                "if _REAL_MODE_SHIM and os.environ.get('RDLAB_STEP') == 'run':\n"
                "    # Real-mode cell (FAKE_COLAB_REAL_MODE_SHIM, test-only): the VM process sees no test hook, as on a\n"
                "    # real Colab VM; the shim replaces only the GPU/network/venv steps and the judge child runner, and\n"
                "    # its host-path guard makes the test's temp root and the repo checkout unreadable from this cell.\n"
                "    _deny_roots = [r for r in os.environ.get('FAKE_COLAB_VM_DENY_ROOTS', '').split(os.pathsep) if r.strip()]\n"
                "    _deny_roots.append(str(pathlib.Path(_REAL_MODE_SHIM).resolve().parents[2]))\n"
                "    for _hook in [k for k in os.environ if k.startswith('FAKE_COLAB_') or k in ('RDLAB_VM_CONTENT_DIR', 'RDLAB_CGROUP_DIR', 'RDLAB_CGROUP_MEMORY_EVENTS_PATH', 'RDLAB_PROC_MEMINFO_PATH')]:\n"
                "        os.environ.pop(_hook, None)\n"
                "    ns['__name__'] = 'rdlab_vm_cell'\n"
                "    exec(compile(src, '<cell>', 'exec'), ns)\n"
                "    import runpy\n"
                "    _shim = runpy.run_path(_REAL_MODE_SHIM)\n"
                "    _shim['install'](ns, state_dir=_SHIM_STATE_DIR, vm_root=_RDLAB_VM_ROOT)\n"
                "    _shim['install_host_path_guard'](deny_roots=_deny_roots, allow_roots=[_RDLAB_VM_ROOT, _SHIM_STATE_DIR, sys.prefix, sys.base_prefix, sys.exec_prefix], state_dir=_SHIM_STATE_DIR)\n"
                "    ns['run_kernel_or_cli_entry']()\n"
                "else:\n"
                "    exec(compile(src, '<cell>', 'exec'), ns)\n"
                "rc = ns.get('CELL_EXIT_CODE', 0)\n"
                "raise SystemExit(int(rc) if isinstance(rc, int) else 0)\n"
            )
            if step == "run" and os.environ.get("FAKE_COLAB_LOST_STREAM") == "1":
                prior_cells.append(
                    {
                        "step": step,
                        "py_file": str(py_path),
                        "env": dict(persisted_env),
                        "rc": 0,
                        "lost_stream": True,
                    }
                )
                session_cells_file.write_text(
                    json.dumps(prior_cells, indent=2) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                print('{"kind":"stream_started","step":"run"}', flush=True)
                subprocess.Popen(
                    [sys.executable, "-c", kernel_code],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=exec_env,
                    cwd=str(vm_fs),
                    start_new_session=True,
                )
                time.sleep(timeout_s)
                sys.stderr.write(f"Execution timed out after {timeout_s}s (lost stream)\n")
                return 1
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
        try:
            import fcntl as _fcntl

            _fcntl.flock(sessions_lock_fh.fileno(), _fcntl.LOCK_EX)
        except Exception:
            pass
        sessions = json.loads(sessions_file.read_text(encoding="utf-8"))
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
        if os.environ.get("FAKE_COLAB_USAGE_FAIL") == "1":
            sys.stderr.write("colab usage failed\n")
            return 1
        raw_override = os.environ.get("FAKE_COLAB_USAGE_OUTPUT")
        if raw_override is not None:
            print(raw_override)
            return 0
        rate_str = os.environ.get("FAKE_COLAB_USAGE_RATE", "").strip()
        if not rate_str:
            has_a100 = any(str(s.get("gpu", "")).upper() == "A100" for s in sessions)
            rate_str = "5.30" if has_a100 else "0.08"
        print(
            f"Compute units balance: 100.0 | Usage rate: {rate_str}/h | Active assignments: 1"
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
