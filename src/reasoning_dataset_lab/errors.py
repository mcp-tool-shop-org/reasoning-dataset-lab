"""Structured errors and exit codes for reasoning-dataset-lab.

Exit codes:
- 0: clean
- 1: refused (validation or policy refusal)
- 2: usage (CLI invocation or file I/O error)
"""

from __future__ import annotations

from reasoning_dataset_lab.pilot_b.device import DeviceRefusalError

EXIT_CLEAN = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2

RdlabError = DeviceRefusalError


def refused_error(code: str, message: str, hint: str) -> RdlabError:
    """Create a validation/policy refusal error (exit code 1)."""
    return RdlabError(code=code, message=message, hint=hint, exit_code=EXIT_REFUSED)


def usage_error(code: str, message: str, hint: str) -> RdlabError:
    """Create a CLI usage or file error (exit code 2)."""
    return RdlabError(code=code, message=message, hint=hint, exit_code=EXIT_USAGE)
