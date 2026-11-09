# reasoning-dataset-lab: how it works

Mapped at 2026-11-09 from commit 664d640 by Atlas 1.24.0.

## What this is

reasoning-dataset-lab plans a dataset as a pre-registered map, builds items whose answer keys are constructed and proved by code, asks a local model only for the wording, and records the provenance of every item. (written by a person)

9 parts, mostly Python (80 files), shell (3), CSS (2), TypeScript (2), Astro (1) and JavaScript (1). Work enters through 4 doors; the busiest is ci, which reaches 3 parts. It publishes to PyPI. It deploys a site to GitHub Pages. People run rdlab.

## What changed since 2026-11-08 (9d1650e)

- scripts/colab/ is now also read by scripts/colab/promote_goldens.py.
- scripts/colab/boot_gatekeeper_goldens.json is now read by scripts/colab/promote_goldens.py.
- 8 files changed content, across 3 parts.

## What comes in

1. **ci.** On a pull request; on a push to main touching 11 paths; or by hand. Runs tests/.
2. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
3. **Release.** When a release is published. Checks src/reasoning_dataset_lab/.
4. **rdlab** (a command people run). Runs src/reasoning_dataset_lab/cli.py.

## What happens through ci

1. The workflow runs tests/ in tests.
2. That reaches scripts (5 files) and src (27 files).
3. It writes to src/reasoning_dataset_lab/pilot_b/canary_config.json.
4. It uploads coverage to Codecov.

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
- **scripts/colab/vm_judges.py** and **tests/test_colab_judge_launcher.py** changed together in 24 of 26 commits, and the tests part imports the scripts part.
- **src/reasoning_dataset_lab/pilot_b/models_and_build.py** and **tests/test_pilot_b_ollama_adapters.py** changed together in 9 of 11 commits, and the tests part imports the src part.
- **tests/fixtures/fake_colab.py** and **tests/test_colab_judge_launcher.py** changed together in 19 of 26 commits, inside the tests part.
- **scripts/colab/vm_judges.py** and **tests/fixtures/fake_colab.py** changed together in 18 of 25 commits, and the tests part imports the scripts part.

1 file changed together with its own test, as expected.

Confidence is low: fewer than 25 source files reach 10 revisions in the window.

Window: 180 days; a pair counts from 3 shared commits, since 8 source files reach 10 revisions; the floor rises to 10 when 25 do.

## What no test touches

Every code part is imported by at least one test.

## Written but never read

Every written place has a reader.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

- **src/reasoning_dataset_lab/pilot_b/canary_config.json** has a block written by src/reasoning_dataset_lab/pilot_b/bakeoff_judge.py.

## Hand-authored

People write .github/, certification/, docs/, examples/, the repository root and site/; 13 writes with paths built at run time may land here.

## Where to start

src/reasoning_dataset_lab/cli.py → src/reasoning_dataset_lab/pilot_b/bakeoff_judge.py → src/reasoning_dataset_lab/pilot_b/canary_config.json → src/reasoning_dataset_lab/pilot_b/lockcheck_anchors.py

Read those in order to follow one run of rdlab end to end. This path follows rdlab (a command people run) from its entry, since ci runs only tests.

## What this map cannot see

- 8 import sites name a declared dependency that shares its name with a local module (role_set); they are read as the dependency, which is not in this repository.
- 16 imports could not be resolved: `scripts/colab/vm_judges.py` imports `device`, which is no module on its import path and no declared dependency, twice; `tests/test_bakeoff_judge_d3_transport.py` imports a path built at run time; `tests/test_counted_pin_and_no_capacity.py` imports `promote_goldens`, which is no module on its import path and no declared dependency; and 12 more.
- 13 writes and 33 reads use paths built at run time and are not named here.
- 80 writes and 180 reads go to a path their caller passes, not to this repository.
- 11 reads go to the directory the command is run in, not to this repository.
- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
