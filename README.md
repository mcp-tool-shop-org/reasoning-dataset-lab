<p align="center">
  <a href="README.md">English</a> | <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/reasoning-dataset-lab/readme.png" alt="reasoning-dataset-lab" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/reasoning-dataset-lab/"><img src="https://img.shields.io/pypi/v/reasoning-dataset-lab" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/reasoning-dataset-lab/"><img src="https://img.shields.io/badge/docs-landing_page-blue" alt="Landing page"></a>
</p>

**A map-first lab for datasets of puzzles, quizzes and verification items, where code proves the answer key before any
model words the item.** `rdlab` is the command line. It plans a dataset as a pre-registered map, builds items whose keys
are constructed and proven, asks a local model only to put them into words, and records the provenance of every item.

A key that a model chose is an opinion. A key that code built from a hidden state, and proved by a solver, is a fact
about the item. `rdlab` is built so the key never depends on a judge.

## Install

```bash
pip install reasoning-dataset-lab
```

Python 3.13 or newer (3.15 is the main target; CI tests 3.13, 3.14 and 3.15). It depends on
[aspire-si-gates](https://pypi.org/project/aspire-si-gates/) and [role-set](https://pypi.org/project/role-set/).

## What you can do without any model

```bash
rdlab map check examples/t10-pilot-a.map.yaml
rdlab project fingerprint . --map examples/t10-pilot-a.map.yaml
rdlab --help
```

- `rdlab map check` validates a dataset map (its axes, every cell's target against the generator's capacity, share
  floors and ceilings, variety bounds, acceptance criteria) and prints the map's canonical SHA-256. Exit 0 is a valid
  map. Exit 1 is a refusal, as a `{code, message, hint}` JSON object on stderr. Exit 2 is a usage or file error.
- `rdlab project fingerprint` hashes the five project files, the map and `canon/` in a fixed order, so a build can say
  exactly what it was built from.
- `rdlab models` lists the models a Pilot B build pins, and `rdlab variety` computes the deterministic variety report
  for a build.

## What needs a GPU and local models

`rdlab pilot-b` builds the Pilot B critic test set: triads of supported, unsupported and cannot-tell claims made from
typed fact tables. It runs real local models through Ollama on loopback (a writer and two blind readers from different
families), needs a GPU, and needs a licence registry file that says each model may be used for evaluation. It never pulls
a model for you and never calls a cloud endpoint. The full flags, the memory plan and the `num_predict` settings are in
the [handbook](https://mcp-tool-shop-org.github.io/reasoning-dataset-lab/handbook/).

## The map-first principle

1. **Prove first, word second.** Code constructs a formal state and proves the key before a model writes any prose. A
   kind of item that cannot be proved by construction is marked `judged` and enters only through a multi-family
   unanimity gate.
2. **Pre-registered cells and capacity.** `map.yaml` declares every intended cell, its target, the generator's capacity
   for it, share floors and ceilings, and the variety bounds, before any item exists. A cell the generator cannot yet
   build stays in the map with `capacity: 0` and a stated reason, so a coverage gap is visible and the map is never
   shrunk to hide it.
3. **One fingerprint.** A project binds five files, the map and its canon, and a deterministic hash says which.

## What has been measured

The generator's value sets (the lists a claim's values are swapped within) are certified before use. In the v3 run, 44
of 50 sets certified, against 26 before the sets were repaired. Here "certified" means two blind readers from different
families called every ordered pair a contradiction, and an independent blind key had already agreed. It is evidence, not
proof: model readers were part of it. The wording of a rendered item can still differ from its key, and a person's
hand-read is the check for that. Nothing `rdlab` writes is training data until a person has checked it. The run and its
files are in [certification/v3](certification/v3).

## Security and data

- **Reads:** the map, project files and canon you point it at.
- **Writes:** the output directory you give it.
- **Network:** only Ollama on loopback (`127.0.0.1`), and only for `pilot-b` and `models --verify-live`. Anything else
  is refused. No telemetry.
- **Permissions:** none beyond those files and that local port.

Report a vulnerability privately; see [SECURITY.md](SECURITY.md).

## Status

Version 0.4.0, the first PyPI release. It is alpha: interfaces and data shapes can change without notice. Coverage of
this repository's own tests is 95%, with the floor at 90.

## License

MIT, © mcp-tool-shop. See [LICENSE](LICENSE).

---

Built by [MCP Tool Shop](https://mcp-tool-shop.github.io/)
