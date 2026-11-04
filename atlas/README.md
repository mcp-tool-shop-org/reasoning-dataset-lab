# reasoning-dataset-lab: how it works

Mapped at 2026-11-04 from commit c24372a by Atlas 1.23.0.

## What this is

reasoning-dataset-lab plans a dataset as a pre-registered map, builds items whose answer keys are constructed and proved by code, asks a local model only for the wording, and records the provenance of every item. (written by a person)

9 parts, mostly Python (68 files), shell (3), CSS (2), TypeScript (2), Astro (1) and JavaScript (1). Work enters through 4 doors; the busiest is ci, which reaches 3 parts. It publishes to PyPI. It deploys a site to GitHub Pages. People run rdlab.

## What changed since 2026-11-04 (7955c81)

Nothing structural changed since 2026-11-04; 2 files added and 6 changed content.

## What comes in

1. **ci.** On a pull request; on a push to main touching 11 paths; or by hand. Runs tests/.
2. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
3. **Release.** When a release is published. Checks src/reasoning_dataset_lab/.
4. **rdlab** (a command people run). Runs src/reasoning_dataset_lab/cli.py.

## What happens through ci

1. The workflow runs tests/ in tests.
2. That reaches scripts (3 files) and src (23 files).
3. It writes to src/reasoning_dataset_lab/pilot_b/canary_config.json.
4. It uploads coverage to Codecov.

## Who reads the results

Only ci itself reads what it writes.

## The other doors

**Deploy site to GitHub Pages** runs site/astro.config.mjs and site/src/, and deploys the site.

**Release** checks src/reasoning_dataset_lab/, publishes to PyPI, and uploads dist/*.tar.gz and dist/*.whl to the release.

**rdlab** (a command people run) runs src/reasoning_dataset_lab/cli.py and writes to src/reasoning_dataset_lab/pilot_b/canary_config.json.

## What breaks what

- **src** is imported by 1 part (scripts), and by 1 more only from tests; it sits on the path of 3 doors.
- **scripts** is imported only from tests, by 1 part (tests), and sits on the path of 1 door.

## What tends to change together

- **src/reasoning_dataset_lab/pilot_b/value_certification.py** and **tests/test_pilot_b_value_certification.py** changed together in 8 of 8 commits, and the tests part imports the src part.
- **scripts/colab/vm_judges.py** and **tests/test_colab_judge_launcher.py** changed together in 14 of 15 commits, and the tests part imports the scripts part.
- **scripts/colab/vm_judges.py** and **tests/fixtures/fake_colab.py** changed together in 12 of 14 commits, and the tests part imports the scripts part.
- **src/reasoning_dataset_lab/pilot_b/models_and_build.py** and **tests/test_pilot_b_ollama_adapters.py** changed together in 9 of 11 commits, and the tests part imports the src part.
- **tests/fixtures/fake_colab.py** and **tests/test_colab_judge_launcher.py** changed together in 12 of 15 commits, inside the tests part.

1 file changed together with its own test, as expected.

Confidence is low: fewer than 25 source files reach 10 revisions in the window.

Window: 180 days; a pair counts from 3 shared commits, since 7 source files reach 10 revisions; the floor rises to 10 when 25 do.

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

src/reasoning_dataset_lab/cli.py → src/reasoning_dataset_lab/errors.py → src/reasoning_dataset_lab/pilot_b/fact_tables.py → src/reasoning_dataset_lab/pilot_b/__init__.py → src/reasoning_dataset_lab/pilot_b/models_and_build.py

Read those in order to follow one run of rdlab end to end. This path follows rdlab (a command people run) from its entry, since ci runs only tests.

## What this map cannot see

- 8 import sites name a declared dependency that shares its name with a local module (role_set); they are read as the dependency, which is not in this repository.
- 3 imports could not be resolved: `scripts/colab/vm_judges.py` imports `device`, which is no module on its import path and no declared dependency, twice; `tests/test_bakeoff_judge_d3_transport.py` imports a path built at run time.
- 13 writes and 28 reads use paths built at run time and are not named here.
- 58 writes and 127 reads go to a path their caller passes, not to this repository.
- 5 reads go to the directory the command is run in, not to this repository.
- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
