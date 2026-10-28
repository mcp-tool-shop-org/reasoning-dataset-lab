# Certification run v3 (2026-10-28)

**44 of 50 sets certified. Here "certified" means both readers called every pair of the set a contradiction, and the set had already keyed 20 of 20 "contradiction" in an independent blind second key before any reader saw it.** Reader agreement alone was a weak filter in r3 (see `certification/r3/`); v3 adds the blind key first. Six sets are queued, with 11 failed pairs read one by one in `FINDINGS.md`.

**The medium arm.** "Medium" is an effort hint the model's template knows, not a cap. On r3's exact pairs, Nemotron at `think: "medium"` hit 22 length stops against 25 at full thinking, and gave the same verdict on 95.5% of the 954 pairs that got a verdict at both settings. Medium is not the fix for length stops; clean sets are (v3: 2 stops in 1,000 pairs).

| File | What it is |
|---|---|
| `manifest-v3.json`, `review_queue-v3.json` | the v3 run: 44 certified sets, 6 queued |
| `control-mistral-v3.json`, `control-nemotron-v3.json` | the control-pair results for each reader |
| `FINDINGS.md` | the run, the 11 failed pairs, a sample read of the 44, and the medium-arm comparison |
| `medium-arm/` | Nemotron at medium on r3's table: manifest, review queue, control, and `COMPARE-r3-nemotron-medium.md` |
| `second-keying/v3`, `v3b`, `v3c` | the blind second-keying inputs (`keying-items-*`) and Argon's keys (`argon-keys-*`) |

sha256: `manifest-v3.json` `62cbd66c…`, `review_queue-v3.json` `a6d073a5…`, `medium-arm/manifest-r3-nemotron-medium.json` `fcd6e0d5…`, `medium-arm/review_queue-r3-nemotron-medium.json` `840804bd…`, `medium-arm/COMPARE-r3-nemotron-medium.md` `9aa6bcb8…`. Full digests are in `FINDINGS.md` and can be recomputed from the files.
