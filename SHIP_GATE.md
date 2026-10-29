# Ship Gate

> No repo is "done" until every applicable line is checked.
> Copy this into your repo root. Check items off per-release.

**Tags:** `[all]` every repo · `[npm]` `[pypi]` `[vsix]` `[desktop]` `[container]` published artifacts · `[mcp]` MCP servers · `[cli]` CLI tools

---

## A. Security Baseline

- [x] `[all]` SECURITY.md exists (report email, supported versions, response timeline) — executed by `npx @mcptoolshop/shipcheck security-docs` (A1: present + reporting contact, not an empty stub) (2026-10-29; SECURITY.md: supported versions, contact, timeline, data and network, threat model)
- [x] `[all]` README includes threat model paragraph (data touched, data NOT touched, permissions required) — executed by `npx @mcptoolshop/shipcheck security-docs` (A2: trust/threat-model section present + non-empty; *quality* is not machine-checkable) (2026-10-29; README "Security and data"; SECURITY.md threat model)
- [x] `[all]` No secrets, tokens, or credentials in source or diagnostics output — executed by `npx @mcptoolshop/shipcheck secrets` (scans every publishable tarball; matches redacted; not a manual attestation) (2026-10-29; `shipcheck secrets` finds no npm package; the sdist and wheel were identity-scanned before the tag)
- [x] `[all]` No telemetry by default — state it explicitly even if obvious (2026-10-29; no telemetry; the only network use is Ollama on loopback)

### Default safety posture

- [ ] SKIP: a command-line tool that does not delete, restart or kill anything and writes only to the output directory you name. `[cli|mcp|desktop]` Dangerous actions (kill, delete, restart) require explicit `--allow-*` flag
- [ ] SKIP: a command-line tool that does not delete, restart or kill anything and writes only to the output directory you name. `[cli|mcp|desktop]` File operations constrained to known directories
- [ ] SKIP: not an MCP server. `[mcp]` Network egress off by default
- [ ] SKIP: not an MCP server. `[mcp]` Stack traces never exposed — structured error results only

## B. Error Handling

- [x] `[all]` Errors follow the Structured Error Shape: `code`, `message`, `hint`, `cause?`, `retryable?` (2026-10-29; refusals are `{code, message, hint}` JSON on stderr)
- [ ] SKIP: the exit codes are 0 ok, 1 refused (a structured error on stderr), 2 usage or file error. There is no partial-success code 3 because a run either seals or refuses. `[cli]` Exit codes: 0 ok · 1 user error · 2 runtime error · 3 partial success
- [x] `[cli]` No raw stack traces without `--debug` (2026-10-29; a missing map and a missing licence registry each print one `{code, message, hint}` line and exit 2; there is no `--debug` flag)
- [ ] SKIP: not an MCP server. `[mcp]` Tool errors return structured results — server never crashes on bad input
- [ ] SKIP: not an MCP server. `[mcp]` State/config corruption degrades gracefully (stale data over crash)
- [ ] SKIP: not a desktop app. `[desktop]` Errors shown as user-friendly messages — no raw exceptions in UI
- [ ] SKIP: not a VS Code extension. `[vscode]` Errors surface via VS Code notification API — no silent failures

## C. Operator Docs

- [x] `[all]` README is current: what it does, install, usage, supported platforms + runtime versions (2026-10-29; README rewritten for 0.4.0)
- [x] `[all]` CHANGELOG.md (Keep a Changelog format) (2026-10-29; CHANGELOG.md, Keep a Changelog; [0.4.0] added)
- [x] `[all]` LICENSE file present and repo states support status (2026-10-29; MIT; support status in SECURITY.md)
- [x] `[cli]` `--help` output accurate for all commands and flags (2026-10-29; every command in the handbook reference answers `--help`: map check, project fingerprint, models, variety, pilot-b, pilot-b calibrate, certify-values, compare-arms)
- [ ] SKIP: a command-line tool that does not delete, restart or kill anything and writes only to the output directory you name. `[cli|mcp|desktop]` Logging levels defined: silent / normal / verbose / debug — secrets redacted at all levels
- [ ] SKIP: not an MCP server. `[mcp]` All tools documented with description + parameters
- [ ] SKIP: covered by the handbook. `[complex]` HANDBOOK.md: daily ops, warn/critical response, recovery procedures

## D. Shipping Hygiene

- [x] `[all]` `verify` script exists (test + build + smoke in one command) (2026-10-29; CI runs pytest with the 90% floor, `rdlab map check` on the example map, and the atlas check; `scripts/verify.sh` runs the same locally)
- [x] `[all]` Version in manifest matches git tag — executed by `npx @mcptoolshop/shipcheck manifest` (D2: manifest version not behind the newest released tag; `--expect <ver>` for a strict release-time match) (2026-10-29; 0.4.0; `shipcheck manifest --expect 0.4.0` passes)
- [x] `[all]` Dependency scanning runs in CI (ecosystem-appropriate) — executed by `npx @mcptoolshop/shipcheck ci` (D3: a recognized scanner is *configured* in CI, or dependabot is present) (2026-10-29; `npm audit` on the site tree in ci.yml; the Python tree is pinned exact in uv.lock)
- [x] `[all]` No known high/critical vulnerabilities in any dependency tree, and Dependabot alerts are enabled — executed by `npx @mcptoolshop/shipcheck deps` (the OUTCOME: audits **every** tree incl. subtrees, not just the root; `ci` only proves a scanner is configured) (2026-10-29; `shipcheck deps`: no vulnerabilities at or above high)
- [ ] SKIP: the org restricts the auto-PR bot; dependencies are pinned exact and the site tree is audited in CI. `[all]` Automated dependency **update** mechanism exists <!-- soft/optional: the org rule restricts the auto-PR bot (CI minutes), NOT alerts. The security outcome is enforced by `shipcheck deps`; the update bot is optional. -->
- [ ] SKIP: not an npm package. `[npm]` Published via OIDC trusted publishing with `--provenance` — executed by `npx @mcptoolshop/shipcheck ci` (config/intent; `--registry <pkg>` also confirms the attestation on npm)
- [ ] SKIP: not an npm package. `[npm]` **Every publishable package** passes `npx @mcptoolshop/shipcheck pack` — `npm pack --dry-run` on each workspace package includes README.md + LICENSE and all `files[]` entries resolve (executed check, not a manual attestation; in a monorepo it verifies all packages, not just the root)
- [x] `[npm]` `engines.node` set · `[pypi]` `python_requires` set — executed by `npx @mcptoolshop/shipcheck manifest` (D6, checked per publishable package) (2026-10-29; requires-python >=3.13)
- [x] `[npm]` Lockfile committed · `[pypi]` Clean wheel + sdist build — lockfile executed by `npx @mcptoolshop/shipcheck manifest` (D7); the pypi wheel/sdist build is not yet executed (2026-10-29; `uv build` builds both; the wheel installs in a clean venv and runs `rdlab --version`)
- [ ] SKIP: not a VS Code extension. `[vsix]` `vsce package` produces clean .vsix with correct metadata
- [ ] SKIP: not a desktop app. `[desktop]` Installer/package builds and runs on stated platforms

## E. Identity (soft gate — does not block ship)

- [x] `[all]` Logo in README header (2026-10-29; brand logo, brand repo logos/reasoning-dataset-lab)
- [x] `[all]` Translations (polyglot-mcp, 8 languages) (2026-10-29; 7 READMEs by TranslateGemma 27B)
- [x] `[org]` Landing page (@mcptoolshop/site-theme) (2026-10-29; https://mcp-tool-shop-org.github.io/reasoning-dataset-lab/ and /handbook/ return 200)
- [x] `[all]` GitHub repo metadata: description, homepage, topics (2026-10-29; description, homepage and topics set)

---

## Gate Rules

**Hard gate (A–D):** Must pass before any version is tagged or published.
If a section doesn't apply, mark `SKIP:` with justification — don't leave it unchecked.

**Soft gate (E):** Should be done. Product ships without it, but isn't "whole."

**Executed vs attested.** `shipcheck audit` only *counts these checkboxes* — it does not read your repo, so a box can be green while the fact is false. The lines that say **"executed by `npx @mcptoolshop/shipcheck <gate>`"** are backed by a command that reads the real artifact and exits 1 on the real defect. Run those gates (they are wired into shipcheck's own `verify`); don't just tick their boxes. Executed today: **A1/A2** (`security-docs`), **A3** (`secrets`), **D2/D6/D7** (`manifest`), **D3-config + OIDC/provenance** (`ci`), **real vulnerabilities + alerting** (`deps`), **D5** (`pack`), plus front-door (`front-door`) and dogfood freshness (`dogfood`). Every other line is still an attestation you are vouching for. Note the two dependency layers: `ci` proves a scanner is *configured*; `deps` proves there are *no known vulnerabilities* — a repo can pass the first while failing the second.

**Checking off:**
```
- [x] `[all]` SECURITY.md exists (2026-02-27)
```

**Skipping:**
```
- [ ] `[pypi]` SKIP: not a Python project
```
