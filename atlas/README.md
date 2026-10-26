# reasoning-dataset-lab: how it works

Mapped at 2026-10-26 from commit ab7c07c by Atlas 1.24.0.

## What this is

reasoning-dataset-lab validates dataset maps, loads the five-file project configuration, and computes deterministic configuration fingerprints. (written by a person)

5 parts, mostly Python (33 files). Work enters through 2 doors; the busiest is ci, which reaches 2 parts. People run rdlab.

## What changed since 2026-10-26 (ee62fb1)

Nothing structural changed since 2026-10-26; 6 files changed content.

## What comes in

1. **ci.** On a pull request; on a push to main touching 8 paths; or by hand. Runs tests/.
2. **rdlab** (a command people run). Runs src/reasoning_dataset_lab/cli.py.

## What happens through ci

1. The workflow runs tests/ in tests.
2. That reaches src (17 files).
3. It uploads coverage to Codecov.

## Who reads the results

ci writes nothing this map can see.

## The other doors

**rdlab** (a command people run) runs src/reasoning_dataset_lab/cli.py.

## What breaks what

- **src** is imported only from tests, by 1 part (tests), and sits on the path of 2 doors.

## What tends to change together

No two source files changed together often enough to name.

Window: 180 days; a pair counts from 3 shared commits, since the window holds fewer than 30 qualifying commits.

## What no test touches

Every code part is imported by at least one test.

## Written but never read

No place this map can see is written, so none goes unread.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

Nothing in this repository writes to a tracked place this map can see.

## Hand-authored

People write .github/, examples/ and the repository root. Nothing in this repository writes to them.

## Where to start

src/reasoning_dataset_lab/cli.py → src/reasoning_dataset_lab/errors.py → src/reasoning_dataset_lab/pilot_b/models_and_build.py → src/reasoning_dataset_lab/map.py → src/reasoning_dataset_lab/config.py → src/reasoning_dataset_lab/pilot_b/variety.py

Read those in order to follow one run of rdlab end to end. This path follows rdlab (a command people run) from its entry, since ci runs only tests.

## What this map cannot see

- 6 import sites name a declared dependency that shares its name with a local module (role_set); they are read as the dependency, which is not in this repository.
- 7 reads use paths built at run time and are not named here.
- 11 writes and 33 reads go to a path their caller passes, not to this repository.
- 1 file belongs to no part: docs/calibration-2026-10-25.md.
- Statistics confidence is low: fewer than 30 qualifying commits in the window, and fewer than 25 source files reach 10 revisions.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
