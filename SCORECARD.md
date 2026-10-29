# Scorecard

**Repo:** reasoning-dataset-lab
**Date:** 2026-10-29
**Type tags:** `[pypi]` `[cli]`

## Pre-Remediation Assessment

| Category | Score | Notes |
|----------|-------|-------|
| A. Security | 5/10 | SECURITY.md existed but listed only 0.1.x and had no data or threat-model section |
| B. Error handling | 8/10 | Structured `{code, message, hint}` errors on stderr already |
| C. Operator docs | 3/10 | The README said "0.1.1 scaffold" while the repo was at 0.3.0, and was a wall of Pilot B flags; no handbook |
| D. Shipping hygiene | 3/10 | Two git-URL dependencies PyPI refuses, no release workflow, no dependency scan, no PyPI metadata |
| E. Identity (soft) | 0/10 | No logo, translations, landing page, topics or homepage |
| **Overall** | **19/50** | |

## Key Gaps

1. Not publishable to PyPI: the dependencies were git URLs, and role-set and aspire-si-gates were not on PyPI.
2. A README that described a scaffold that no longer exists.
3. No public surface beyond the README.

## Post-Remediation

Hard gates A to D pass (`shipcheck audit`: 0 unchecked, items that do not apply are marked `SKIP:` with the reason in
SHIP_GATE.md; `security-docs`, `manifest --expect 0.4.0`, `ci` and `deps` gates pass). Section E is done: brand logo,
README translations, landing page and handbook, repository metadata.

| Category | Score |
|----------|-------|
| A. Security | 10/10 |
| B. Error handling | 9/10 (exit code 3 for partial success does not exist; documented as a skip) |
| C. Operator docs | 10/10 |
| D. Shipping hygiene | 10/10 |
| E. Identity (soft) | 10/10 |
| **Overall** | **49/50** |

Coverage of this repository's own tests is 95% with the floor at 90 (target met).
