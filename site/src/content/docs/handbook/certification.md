---
title: Certification
description: How the generator's value sets are certified, and what the v3 run found.
sidebar:
  order: 4
---

The Pilot B generator makes an unsupported claim by swapping a true value for another value from the same set. That is
a contradiction only if the two values cannot both be true, so every set has to be checked: its values must sit on one
axis and exclude each other. `rdlab pilot-b certify-values` does the check, and the files from each run are kept in the
repository under `certification/`.

## What "certified" means

Here a set is certified when two blind readers from different families called every ordered pair of its values a
contradiction (20 pairs for 5 values), and the set had already keyed 20 of 20 "contradiction" in an independent blind
second key before any reader saw it. A pair that stops on length, or that a reader cannot decide, is not certified: it goes
to a review queue with its reason.

It is evidence, not proof. Model readers were part of it, and the second key was a model's. The wording of a rendered
item can still differ from its key, and a person's hand-read is the check for that.

## The v3 run (2026-10-28)

- 50 sets, **44 certified and 6 queued**. The same check on the unrepaired sets (run r3) certified 26 and queued 24.
- 2,000 reader calls, none failed.
- Nemotron length stops: 2 of 1,000 pairs, against 25 in r3. The sets and the setting changed together, so this is
  suggestive only. A separate arm ran Nemotron at a "medium" effort setting on r3's exact pairs and gave the same
  verdict on 95.5% of the 954 pairs that got a verdict at both settings.
- Of the 11 failed pairs, read one by one, 6 were pair faults the blind key had missed (the readers were right to
  refuse them) and 5 were reader errors, split between the two readers.
- A read of 10 of the 44 certified sets found 9 clean and 1 with a mild wording note. The same read of the old sets found
  5 of 10 clean.

The manifest, the review queue, the control-pair results and the findings are in `certification/v3/`.
