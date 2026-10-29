---
title: Getting started
description: Install reasoning-dataset-lab and validate a dataset map with no model.
sidebar:
  order: 1
---

## Install

```bash
pip install reasoning-dataset-lab
```

Python 3.13 or newer. The package depends on [aspire-si-gates](https://pypi.org/project/aspire-si-gates/) and
[role-set](https://pypi.org/project/role-set/), and installs both.

## Validate a map

The repository ships an example map, `examples/t10-pilot-a.map.yaml`. From a clone:

```bash
rdlab map check examples/t10-pilot-a.map.yaml
```

It checks the map's schema, every cell's target against the generator's capacity, the share rules, the variety caps
and the acceptance criteria, then prints the canonical SHA-256 of the map. Exit code 0 means the map is valid. Exit 1
means it was refused, and the reason is printed to stderr as `{code, message, hint}`. Exit 2 is a usage or file error.

## Fingerprint a project

```bash
rdlab project fingerprint . --map examples/t10-pilot-a.map.yaml
```

This hashes the five project files (`project.json`, `constitution.json`, `terminology.json`, `lanes.json`,
`rubric.json`), the map and `canon/`, in a fixed order, so a build can state exactly what it was built from.

## What needs a model

Everything above runs with no model and no network. `rdlab pilot-b`, `rdlab pilot-b calibrate` and
`rdlab models --verify-live` call real local models through Ollama on loopback. They need a GPU, the pinned models
pulled by hand, and a licence registry file. See [Pilot B builds](../pilot-b/).

## Run the tests

From a clone:

```bash
python -m pytest -q
```
