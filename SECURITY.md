# Security

## Reporting a Vulnerability

Report a vulnerability privately through GitHub's security advisories on this
repository, or by email to `64996768+mcp-tool-shop@users.noreply.github.com`.
Please do not open a public issue for a security vulnerability.

We aim to acknowledge a report within 48 hours, assess its impact within 7 days,
and ship a fix within 30 days.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.4.x   | Yes, as an alpha |
| 0.3.x and earlier | No |

## Scope

`reasoning-dataset-lab` (`rdlab`) is a local CLI tool for validating dataset maps,
loading project configurations, and computing deterministic configuration fingerprints.
It is not a network server, does not listen on any port, and does not collect telemetry.

## Data and network

- **Reads:** the map, project files and canon you point it at.
- **Writes:** the output directory you give it.
- **Network:** only Ollama on loopback (`127.0.0.1`), and only for `pilot-b` and `models --verify-live`. Any other host is refused before a request is sent. It never pulls a model.

## Threat model

The risks are a crafted map or project file, and a model's output being trusted as if it were an answer key. The first is
handled by validating before use. The second is handled by design: keys are built and proved by code, and model output is
only ever the wording, checked by blind readers.
