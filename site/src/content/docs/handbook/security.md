---
title: Security and data
description: What reasoning-dataset-lab reads and writes, and how to report a problem.
sidebar:
  order: 6
---

## What it touches

- **Reads:** the map, the project files and the canon you point it at.
- **Writes:** the output directory you give it.
- **Network:** only Ollama on loopback (`127.0.0.1`), and only for `pilot-b` and `models --verify-live`. Any other host is
  refused before a request is sent. A model that is missing or whose digest does not match is refused; `rdlab` never
  pulls a model.
- **Telemetry:** none.

## Threat model

`rdlab` is a local command line tool. It does not listen on a port. The risks are a crafted map or project file, and a
model's output being trusted as if it were a key. The first is handled by validating before use. The second is handled by
design: keys are built and proved by code, and model output is only ever the wording, checked by blind readers.

## Reporting

Report a vulnerability privately through GitHub's security advisories on the repository, or by email to the address in
`SECURITY.md`. Do not open a public issue. A report is acknowledged within 48 hours, assessed within 7 days, and a fix is
aimed for within 30 days.
