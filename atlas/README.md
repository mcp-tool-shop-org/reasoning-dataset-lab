# reasoning-dataset-lab: how it works

Mapped at 2026-10-29 from commit 0aabe5b by Atlas 1.24.0.

## What this is

reasoning-dataset-lab plans a dataset as a pre-registered map, builds items whose answer keys are constructed and proved by code, asks a local model only for the wording, and records the provenance of every item. (written by a person)

9 parts, mostly Python (39 files), CSS (2), TypeScript (2), Astro (1), JavaScript (1) and shell (1). Work enters through 4 doors; the busiest is ci, which reaches 2 parts. It publishes to PyPI. It deploys a site to GitHub Pages. People run rdlab.

## What changed since 2026-10-29 (17a9b8a)

- ci's push trigger now also names `docs/**` and `site/src/content/docs/handbook/pilot-b.md`.
- Release no longer starts by hand.
- 6 files changed content, across 3 parts.

## What comes in

1. **ci.** On a pull request; on a push to main touching 11 paths; or by hand. Runs tests/.
2. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
3. **Release.** When a release is published. Checks src/reasoning_dataset_lab/.
4. **rdlab** (a command people run). Runs src/reasoning_dataset_lab/cli.py.

## What happens through ci

1. The workflow runs tests/ in tests.
2. That reaches src (18 files).
3. It uploads coverage to Codecov.

## Who reads the results

ci writes nothing this map can see.

## The other doors

**Deploy site to GitHub Pages** runs site/astro.config.mjs and site/src/, and deploys the site.

**Release** checks src/reasoning_dataset_lab/, publishes to PyPI, and uploads dist/*.tar.gz and dist/*.whl to the release.

**rdlab** (a command people run) runs src/reasoning_dataset_lab/cli.py.

## What breaks what

- **src** is imported only from tests, by 1 part (tests), and sits on the path of 3 doors.

scripts holds only shell files, which this map does not read, so what uses it cannot be seen.

## What tends to change together

- **src/reasoning_dataset_lab/pilot_b/value_certification.py** and **tests/test_pilot_b_value_certification.py** changed together in 8 of 8 commits, and the tests part imports the src part.
- **src/reasoning_dataset_lab/pilot_b/rendering.py** and **tests/test_pilot_b_part_b.py** changed together in 5 of 6 commits, and the tests part imports the src part.
- **src/reasoning_dataset_lab/pilot_b/models_and_build.py** and **tests/test_pilot_b_ollama_adapters.py** changed together in 9 of 11 commits, and the tests part imports the src part.
- **src/reasoning_dataset_lab/pilot_b/models_and_build.py** and **src/reasoning_dataset_lab/pilot_b/ollama.py** changed together in 9 of 14 commits, inside the src part.
- **src/reasoning_dataset_lab/pilot_b/ollama.py** and **tests/test_pilot_b_ollama_adapters.py** changed together in 8 of 13 commits, and the tests part imports the src part.

Confidence is low: fewer than 30 qualifying commits in the window, and fewer than 25 source files reach 10 revisions.

Window: 180 days; a pair counts from 3 shared commits, since the window holds fewer than 30 qualifying commits.

## What no test touches

Every code part this map reads is imported by at least one test.

scripts holds only shell files, which this map does not read, so whether a test touches it cannot be seen.

## Written but never read

No place this map can see is written, so none goes unread.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

Nothing in this repository writes to a tracked place this map can see.

## Hand-authored

People write .github/, certification/, docs/, examples/, the repository root and site/. Nothing in this repository writes to them.

## Where to start

src/reasoning_dataset_lab/cli.py → src/reasoning_dataset_lab/errors.py → src/reasoning_dataset_lab/pilot_b/fact_tables.py → src/reasoning_dataset_lab/pilot_b/__init__.py → src/reasoning_dataset_lab/pilot_b/models_and_build.py → src/reasoning_dataset_lab/map.py

Read those in order to follow one run of rdlab end to end. This path follows rdlab (a command people run) from its entry, since ci runs only tests.

## What this map cannot see

- 8 import sites name a declared dependency that shares its name with a local module (role_set); they are read as the dependency, which is not in this repository.
- 7 reads use paths built at run time and are not named here.
- 15 writes and 42 reads go to a path their caller passes, not to this repository.
- Statistics confidence is low: fewer than 30 qualifying commits in the window, and fewer than 25 source files reach 10 revisions.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
