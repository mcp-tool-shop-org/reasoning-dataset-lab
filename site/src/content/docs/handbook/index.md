---
title: reasoning-dataset-lab Handbook
description: A map-first lab for datasets of puzzles, quizzes and verification items, where code proves the answer key.
sidebar:
  order: 0
---

**reasoning-dataset-lab** (`rdlab`) builds datasets of puzzles, quizzes, verification items and reasoning items from
pre-registered maps, bound canon and frozen parts, with provenance on every item. Its one rule is that the answer key
comes from code, not from a model: code builds the item from a hidden formal state and proves the key, and a model is
asked only to put the item into words.

## Where to go

- [Getting started](./getting-started/): install it and validate a map with no model at all.
- [The map-first principle](./map-first/): the map, cells and capacity, and the project fingerprint.
- [Pilot B builds](./pilot-b/): the part that runs real local models: the memory plan, the models, calibration.
- [Certification](./certification/): how the generator's value sets are certified, and what the v3 run found.
- [Reference](./reference/): the commands, exit codes and error codes.
- [Security and data](./security/): what it reads and writes, and how to report a problem.

## Status

Alpha. The first PyPI release is 0.4.0. Interfaces and data shapes can change without notice. Nothing `rdlab` writes is
training data until a person has checked it.
