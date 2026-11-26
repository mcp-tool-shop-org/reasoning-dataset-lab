# reasoning-dataset-lab: how it works

Mapped at 2026-11-26 from commit fc488c7 by Atlas 1.26.0.

## What this is

reasoning-dataset-lab plans a dataset as a pre-registered map, builds items whose answer keys are constructed and proved by code, asks a local model only for the wording, and records the provenance of every item. (written by a person)

9 parts, mostly Python (109 files), shell (4), CSS (2), JavaScript (2), TypeScript (2) and Astro (1). Work enters through 4 doors; the busiest is ci, which reaches 3 parts. It publishes to PyPI. It deploys a site to GitHub Pages. People run rdlab.

## What changed since 2026-11-26 (f42f16e)

Nothing structural changed since 2026-11-26; 1 file changed content.

## What comes in

1. **ci.** On a pull request; on a push to main touching 13 paths; or by hand. Runs scripts/check-audit-allowlist.mjs and tests/.
2. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
3. **Release.** When a release is published. Checks src/reasoning_dataset_lab/.
4. **rdlab** (a command people run). Runs src/reasoning_dataset_lab/cli.py.

## What happens through ci

1. The workflow runs scripts/check-audit-allowlist.mjs in scripts and tests/ in tests.
2. That reaches src (30 files).
3. It writes to src/reasoning_dataset_lab/pilot_b/canary_config.json.
4. It also writes to site/audit-report.json, which is not tracked.
5. It uploads coverage to Codecov.

## Who reads the results

- **src/reasoning_dataset_lab/pilot_b/canary_config.json** is read by src/reasoning_dataset_lab/pilot_b/lockcheck_anchors.py.

## The other doors

**Deploy site to GitHub Pages** runs site/astro.config.mjs and site/src/, and deploys the site.

**Release** checks src/reasoning_dataset_lab/, publishes to PyPI, and uploads dist/*.tar.gz and dist/*.whl to the release.

**rdlab** (a command people run) runs src/reasoning_dataset_lab/cli.py and writes to src/reasoning_dataset_lab/pilot_b/canary_config.json.

## What breaks what

- **src** is imported by 1 part (scripts), and by 1 more only from tests; it sits on the path of 3 doors.
- **scripts** is imported only from tests, by 1 part (tests), and sits on the path of 1 door.

## What tends to change together

- **src/reasoning_dataset_lab/pilot_b/value_certification.py** and **tests/test_pilot_b_value_certification.py** changed together in 8 of 8 commits, and the tests part imports the src part.
- **src/reasoning_dataset_lab/pilot_b/models_and_build.py** and **tests/test_pilot_b_ollama_adapters.py** changed together in 10 of 12 commits, and the tests part imports the src part.
- **src/reasoning_dataset_lab/pilot_b/fact_tables.py** and **tests/test_pilot_b_variety.py** changed together in 5 of 7 commits, and the tests part imports the src part.
- **scripts/colab/vm_judges.py** and **tests/test_colab_judge_launcher.py** changed together in 44 of 73 commits, and the tests part imports the scripts part.
- **src/reasoning_dataset_lab/pilot_b/rendering.py** and **tests/test_pilot_b_part_b.py** changed together in 5 of 9 commits, and the tests part imports the src part.

1 file changed together with its own test, as expected.

Confidence is low: fewer than 25 source files reach 10 revisions in the window.

Window: 180 days; a pair counts from 3 shared commits, since 15 source files reach 10 revisions; the floor rises to 10 when 25 do.

## What no test touches

Every code part is imported by at least one test.

## Written but never read

Every written place has a reader.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

- **src/reasoning_dataset_lab/pilot_b/canary_config.json** has a block written by src/reasoning_dataset_lab/pilot_b/bakeoff_judge.py.

## Hand-authored

People write .github/, certification/, docs/, examples/, the repository root and site/; 14 writes with paths built at run time may land here.

## Where to start

src/reasoning_dataset_lab/cli.py → src/reasoning_dataset_lab/pilot_b/bakeoff_judge.py → src/reasoning_dataset_lab/pilot_b/canary_config.json → src/reasoning_dataset_lab/pilot_b/lockcheck_anchors.py

Read those in order to follow one run of rdlab end to end. This path follows rdlab (a command people run) from its entry, since ci runs only tests and scripts that import no code here.

## What this map cannot see

- 8 import sites name a declared dependency that shares its name with a local module (role_set); they are read as the dependency, which is not in this repository.
- 20 imports could not be resolved: `scripts/colab/vm_judges.py` imports a path built at run time; `scripts/colab/vm_judges.py` imports `device`, which is no module on its import path and no declared dependency, twice; `tests/test_bakeoff_judge_d3_transport.py` imports a path built at run time; and 16 more.
- 14 writes and 43 reads use paths built at run time and are not named here.
- 1 write goes to places this repository does not track, so it is not listed as generated.
- 98 writes and 240 reads go to a path their caller passes, not to this repository.
- 12 reads go to the directory the command is run in, not to this repository.
- 1 read goes to the directory the command is run in (memory.kmem.usage_in_bytes) or a path its caller passes, not to this repository.
- 1 read goes to the home directory (.config/, AppData/ and Library/) or a path its caller passes, not to this repository.
- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
