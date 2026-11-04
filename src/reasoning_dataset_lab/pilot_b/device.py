"""Package-safe judge device normaliser (pure Python standard library)."""

from __future__ import annotations

import json
import re
from typing import Any

_CUDA_INDEX_DEVICE_RE: re.Pattern[str] = re.compile(r"cuda:(?:0|[1-9][0-9]*)")

_PROXY_TOKEN_RE = re.compile(r"(colab-runtime-proxy-token=)[^ \"'&;,)\]}]+")
_BEARER_TOKEN_RE = re.compile(
    r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE
)
_OAUTH_YA29_RE = re.compile(r"ya29\.[A-Za-z0-9._-]+")
_OAUTH_REFRESH_RE = re.compile(r"1//[A-Za-z0-9._-]+")
_OAUTH_KV_RE = re.compile(
    r"((?:access_token|refresh_token|id_token|client_secret)\s*[=:]\s*[\"']?)[A-Za-z0-9._~+/=-]+",
    re.IGNORECASE,
)
_JWT_RE = re.compile(
    r"eyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}"
)
_HF_TOKEN_RE = re.compile(
    r"\bhf_(?!(?:home|cache|low|match)\b)[A-Za-z0-9]{6,}\b"
)
_HF_ENV_RE = re.compile(
    r"((?:HF_TOKEN(?:_[A-Za-z0-9_]+)?|HUGGING_FACE_HUB_TOKEN)=)[^ \"'\n;]+"
)


def scrub_device_text(text: str) -> str:
    """Scrub secret token patterns from a device string representation."""
    out = _PROXY_TOKEN_RE.sub(r"\1[REDACTED]", text)
    out = _BEARER_TOKEN_RE.sub(r"\1[REDACTED]", out)
    out = _OAUTH_YA29_RE.sub("[REDACTED_OAUTH_TOKEN]", out)
    out = _OAUTH_REFRESH_RE.sub("[REDACTED_OAUTH_TOKEN]", out)
    out = _OAUTH_KV_RE.sub(r"\1[REDACTED]", out)
    out = _JWT_RE.sub("[REDACTED_JWT]", out)
    out = _HF_TOKEN_RE.sub("hf_[REDACTED]", out)
    out = _HF_ENV_RE.sub(r"\1[REDACTED]", out)
    return out


class DeviceRefusalError(RuntimeError):
    """Standard-library structured refusal error carrying {code, message, hint, exit_code}."""

    def __init__(
        self,
        code: str,
        message: str,
        hint: str = "Inspect VM logs and refusal details before retrying.",
        exit_code: int = 1,
    ) -> None:
        RuntimeError.__init__(self, f"[{code}] {message} (hint: {hint})")
        self.code = code
        self.failure_class = code
        self.message = message
        self.detail = message
        self.hint = hint
        self.exit_code = exit_code

    def __str__(self) -> str:
        return f"[{self.code}] {self.message} (hint: {self.hint})"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def normalize_judge_device(device: Any) -> str:
    """Normalise judge device strings: `'cuda' -> 'cuda:0'`, `'cuda:N' -> 'cuda:N'`, `'cpu' -> 'cpu'`.

    Any other value (including `'cuda:'`, `'cuda:-1'`, `'gpu'`, `'cuda:0 '`, `' cuda:0'`, `'cuda:0\\n'`, `''`, or non-str)
    is refused with `BAKEOFF_JUDGE_DEVICE_INVALID` and a scrubbed message.
    """
    if isinstance(device, str) and not isinstance(device, bool):
        if device == "cuda":
            return "cuda:0"
        if device == "cpu" or _CUDA_INDEX_DEVICE_RE.fullmatch(device):
            return device
    raw_repr = scrub_device_text(repr(device))
    raise DeviceRefusalError(
        "BAKEOFF_JUDGE_DEVICE_INVALID",
        f"Invalid judge device {raw_repr}; expected 'cuda', "
        "'cuda:N' (non-negative integer N), or 'cpu'",
        "Pass --device cuda:0 (or --device cuda, which normalises to cuda:0) or --device cpu.",
    )
