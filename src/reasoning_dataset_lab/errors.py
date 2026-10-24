"""Structured errors and exit codes for reasoning-dataset-lab.

Exit codes:
- 0: clean
- 1: refused (validation or policy refusal)
- 2: usage (CLI invocation or file I/O error)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

EXIT_CLEAN = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2


@dataclass(frozen=True)
class RdlabError(Exception):
    """Structured error carrying {code, message, hint} and a CLI exit code."""

    code: str
    message: str
    hint: str
    exit_code: int = EXIT_REFUSED

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


def refused_error(code: str, message: str, hint: str) -> RdlabError:
    """Create a validation/policy refusal error (exit code 1)."""
    return RdlabError(code=code, message=message, hint=hint, exit_code=EXIT_REFUSED)


def usage_error(code: str, message: str, hint: str) -> RdlabError:
    """Create a CLI usage or file error (exit code 2)."""
    return RdlabError(code=code, message=message, hint=hint, exit_code=EXIT_USAGE)
