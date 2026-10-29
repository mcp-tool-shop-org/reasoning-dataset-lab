---
title: The map-first principle
description: Pre-registered maps, proven keys, and one fingerprint for a whole project.
sidebar:
  order: 3
---

Every dataset begins as a pre-registered `map.yaml` before any item is generated.

## Prove first, word second

Code constructs a formal state and proves the key before any language model renders prose. For a puzzle, the builder
samples a hidden solution, adds clues that are true in it until the answer is fixed, and a solver checks the result. A
model is then asked only to word the item, and blind readers check that the wording still says what the state says.

A kind of item that cannot be proved by construction is marked `judged`. It enters only through a multi-family
unanimity gate, and it is never treated as proved.

## Pre-registered cells and capacity

`map.yaml` declares the dataset's axes, every intended cell with its `target` and the generator's `capacity` for it,
axis share floors and ceilings, variety bounds, and the acceptance thresholds, all before any item exists. A cell the
generator cannot yet build stays in the map with `capacity: 0` and an explicit `reason`. A coverage gap is therefore
visible, and the map is never shrunk to hide it.

## One fingerprint

A project binds `project.json`, `constitution.json`, `terminology.json`, `lanes.json` (derived from the map's `kind`
axis) and `rubric.json` together with `map.yaml` and `canon/`. A single deterministic SHA-256 says exactly which
configuration a build used.

## Where the gates come from

The item checks are the Stage 1 data gates in [aspire-si-gates](https://pypi.org/project/aspire-si-gates/), and the
puzzle generator and its solver come from [role-set](https://pypi.org/project/role-set/). `rdlab` composes them into
frozen parts with per-item provenance.
