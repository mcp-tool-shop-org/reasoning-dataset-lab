---
title: Reference
description: The rdlab commands and their exit codes.
sidebar:
  order: 5
---

## Commands

| Command | What it does | Needs a model |
|---|---|---|
| `rdlab map check <map.yaml>` | Validates a dataset map and prints its canonical SHA-256. | No |
| `rdlab project fingerprint <dir> --map <map.yaml>` | Hashes the five project files, the map and `canon/`. | No |
| `rdlab models` | Lists the models a Pilot B build pins. | No |
| `rdlab models --verify-live --licence-registry <path>` | Checks the live Ollama manifests and licence digests against a licence registry. | Ollama on loopback |
| `rdlab variety` | Computes the deterministic variety report for a Pilot B build. | No |
| `rdlab pilot-b --stub N` | A calibration run of N triads on the real models; seals nothing. | Yes |
| `rdlab pilot-b calibrate` | Measures total `eval_count` for one operation; writes `calibration.json`. | Yes |
| `rdlab pilot-b` | Builds the Pilot B critic test set. | Yes |
| `rdlab pilot-b certify-values` | Certifies the generator's value sets with two blind readers. | Yes |
| `rdlab pilot-b compare-arms` | Compares two certification runs. | No |
| `rdlab --version` | Prints the version. | No |

Run `rdlab <command> --help` for every flag.

## Exit codes

- `0`: success.
- `1`: refused. A structured `{code, message, hint}` JSON error goes to stderr.
- `2`: a usage or file error, in the same structured form.

## Licence registry

`rdlab pilot-b`, `rdlab pilot-b calibrate` and `rdlab models --verify-live` require `--licence-registry <path>` pointing at
a `licences/registry.json` file. Omitting it refuses the run with `PILOT_B_LICENCE_REGISTRY_REQUIRED`. The build report
records the registry path, its SHA-256 and the read date of each licence entry the model chain used.

## What a build records

`pilot_b_report.json` and `provenance.jsonl` record the registry, each adapter's class, `num_ctx` and `max_num_ctx`, the
`num_predict` values and where each came from, and for every call the model, its digest, the prompt's SHA-256, the token
count, `thinking_words`, and the wall time.
