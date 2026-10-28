# Certification run r3: findings (R&D, 2026-10-28)

**Grant:** Publisher, 5090, cap 28,000 MiB card total. r3 (06:30-06:31Z) ran the two control phases; its phase C was
refused for a missing `--licence-registry` (my miss, rc=2 before any load). r3c (06:33:35-07:17:10Z) ran phase C only.
rdlab 6310db9, licence registry rnd fb27d76 (596e7b3c…). Readers in order: mistral-small3.2:24b (reader_3, think off),
nemotron-3.5-lightning:30b (think on). One model resident at a time; `ollama ps` empty after; card 1,879 MiB at the end.

## Gates (READER3-GATE-2026-10-27, fixed before the run)

| Reader | Phase | Pass | Compatible | Stop | Tokens | Gate |
|---|---|---|---|---|---|---|
| Mistral | r3 A | 19/20 | 19/20 | 20/20 | 1-5 | PASS |
| Nemotron | r3 B | 20/20 | 20/20 | 20/20 | n/a | PASS |
| Mistral | in-code control inside r3c | 19/20 | | | | passed (>= 16) |
| Nemotron | in-code control inside r3c | 20/20 | | | | passed |

## Certification

- 50 sets asked, **26 certified, 24 to the review queue**; 2,000 calls, 0 failed calls; `complete: true`, nothing
  `not_reached`, no stop. 44 min for both readers (Mistral's 1,000 calls in about 3 min).
- VRAM: peak **27,579 MiB** card total (Nemotron), under the 28,000 cap; no HALT.
- Outputs: `manifest-r3.json` (sha256 77bfd2d51bf907060b8f8894691ed3e9dfdbb900512505b0ec7b3cacf5f00080),
  `review_queue-r3.json` (d6a7aa10ffbab5663d22b269940c773296875b07d7bcc023b07661e215f8c9c1, JSONL, one line a set).

## Why 24 sets were queued (121 failed pairs; a set is admitted only if both readers call every pair a contradiction)

| Mistral | Nemotron | Pairs |
|---|---|---|
| incompatible | compatible | 40 |
| compatible | compatible | 27 |
| incompatible | **length** (hit num_predict 3072) | 18 |
| compatible | incompatible | 15 |
| incompatible | unsure | 6 |
| unsure | compatible | 6 |
| compatible | length | 4 |
| unsure | length | 3 |
| unsure | incompatible | 2 |

- All 121 failed pairs, in four groups: **55** where one reader said contradiction and the other compatible (40 + 15);
  **27** both readers call compatible (likely real weak pairs in the sets); **25 Nemotron length stops** (its thinking
  ran past 3,072 tokens; Mistral 18 incompatible, 4 compatible, 3 unsure), which count as failures, not verdicts; and
  **14** where a reader answered "unsure" (6 + 6 + 2). (Corrected 2026-10-28: an earlier line said "61 split", which
  counted the 6 incompatible/unsure pairs as split but not the 2 unsure/incompatible ones; the Editor caught it.)
- Failed pairs per queued set: 1 to 14 (median about 4). 10 domains; compressors and pumps 4 sets each.
- Not a reader-quality verdict: no gold labels here. What it means for the sets goes to the Book Keeper's review.

## R&D's own read of a sample against the keys (2026-10-28; my reading, not verified by an outside model)

Sample drawn with `random.Random(20261028)`: 10 of the 26 certified sets (all 20 pairs each, every one "incompatible"
from both readers) and 10 of the 27 pairs both readers called "compatible". The key of a set is its five values,
meant to be mutually exclusive for that attribute, so every ordered pair should be a contradiction.

### Certified sets: 5 clean, 3 hold a pair that is not a strict contradiction, 2 with a mild wording issue

| Set | Read |
|---|---|
| boilers / burner state: prepurge, lightoff, firing, postpurge, lockout | clean: distinct sequence states |
| transformers / vector group: Dyn11, Yyn0, YNd1, Dd0, Yzn11 | clean: one winding connection per unit |
| chillers / circuit state: idle, startup, chilling, defrosting, pumpdown | clean |
| reactors / coolant type: water, sodium, helium, CO2, lead-bismuth | clean (primary coolant) |
| chillers / refrigerant: ammonia, propane, CO2, isobutane, R134a | clean per circuit; a cascade plant runs CO2 and ammonia in two circuits, so the attribute needs "per circuit" |
| vessels / berth status: underway, anchored, alongside, adrift, aground | **not strict:** under COLREGs a vessel adrift is "underway" (underway means not at anchor, made fast or aground), so underway/adrift both ways are not contradictions |
| turbines / cooling medium: helium, nitrogen, hydrogen, water, oil | **not strict:** a large generator is commonly hydrogen-cooled with water-cooled stator windings, so hydrogen/water can both hold |
| transformers / bushing material: porcelain, oilpaper, epoxy, siliconerubber, glass | **not strict:** mixes insulation (oil-paper, epoxy resin) with housing (porcelain, silicone rubber); an oil-impregnated-paper bushing usually has a porcelain or silicone-rubber housing |
| chillers / brazing filler: silver, nickel, aluminum, copper, gold | mild: silver brazing fillers are usually Ag-Cu-Zn alloys, so "silver" vs "copper" depends on reading the labels as filler families |
| relays / mount style: rack, rail, wall, pole, floor | mild: a DIN-rail device inside a wall-mounted enclosure is both "rail" and "wall" |

**What it shows:** two readers agreeing on "incompatible" does not make a pair a contradiction. In 3 of 10 sampled
certified sets, both readers passed at least one pair that a domain reading rejects. They share the same blind spot:
a label read as a single category when the field uses it on two axes (insulation vs housing, coolant per circuit, the
legal sense of "underway"). Certification as it stands admits these.

### The 27 both-compatible pairs: mostly set-design problems, not reader errors

| Pair (set) | Read |
|---|---|
| highest vs higher (tap position) | **wording:** "higher" is relative; the readers are right |
| cold vs drained (reactor loop state) | **mixed axes** (temperature vs inventory); a drained loop can be cold; readers right |
| clutch vs flexible; fluid vs flexible (drive coupling) | **mixed axes / overlapping classes:** fluid and some clutch couplings are flexible in the misalignment sense; readers defensible |
| sonic vs retractable (sootblower type) | **mixed axes:** mechanism (sonic, rake) vs motion or mounting (retractable, rotary, wall) |
| electronic vs automatic (expansion valve) | **wording:** "automatic expansion valve" is a specific type, but an electronic valve is automatic in plain English |
| amorphous vs silicon; silicon vs nanocrystalline (core steel) | **wording:** amorphous (Fe-Si-B) and nanocrystalline (Fe-Si-B-Nb-Cu) alloys contain silicon; "silicon" needs to read "grain-oriented silicon steel" |
| withdrawn vs disconnected (breaker position) | **wording:** distinct positions in switchgear terms, but a withdrawn breaker is disconnected in plain English |
| oxygenated vs caustic (feedwater program) | the one pair where I'd call it a contradiction and both readers missed it: oxygenated treatment and caustic treatment are distinct programs |

**What it shows:** 9 of the 10 sampled are faults in the set (relative terms, two axes in one attribute, a class label
that contains a member), so the readers were right to refuse them. 1 of 10 looks like a real reader miss.

### Consequences (proposals; nothing changed)

- The certified 26 need a human or third-model pass before they're used as keys. The sample's 3 of 10 suggests
  several more carry a non-contradiction pair.
- The set generator needs three rules: one axis per attribute; no relative terms; no label that is a class containing
  another value. Each would have caught most of the faults above.
- Readers agreeing is a filter for clear failures, not a proof. A pair both readers pass still needs a check that
  doesn't share their blind spot (a domain reference, or a reader from a third family).

## Second keying by Argon (card la, 2026-10-28; blind to the readers' verdicts and to the read above)

Input: 547 pairs (all 520 from the 26 certified sets, and the 27 both-compatible pairs), shuffled with seed 20261028,
no verdicts or group labels (`second-keying-2026-10-28/keying-items.jsonl`, 9a703ca1…). Argon keyed each pair
`contradiction`, `arguable` or `compatible` with a reason, in the certifier's Statement 1 / Statement 2 frame. Output
`argon-keys.jsonl` (547 lines, every id once, input order). Reply: `replies/2026-10-28la-second-keying.md`.

| Group | contradiction | arguable | compatible |
|---|---|---|---|
| 520 pairs of the 26 certified sets | 272 | 216 | 32 |
| 27 pairs both readers called compatible | 2 | 19 | 6 |

- **Only 3 of the 26 certified sets are contradictions on all 20 pairs:** reactors / coolant type, vessels / registry
  port, vessels / ice class. The worst: boilers / tube alloy (0 of 20), reactors / shielding material (0 of 20).
- **It agrees with the human sample read on every fault that read named:** berth status underway/adrift (COLREGs Rule
  3(i)), cooling medium hydrogen/water ("compatible": combined hydrogen-water generator cooling), bushing material
  (6 compatible, 10 arguable). It is stricter than that read on sets the read called clean, for reasons that hold up:
  a three-winding transformer carries two vector-group designations (8 of 20 arguable), and a burner's lightoff and
  firing overlap during the ignition trial (2 of 20).
- **The 27 both-compatible pairs:** Argon agrees with the readers on 25. It calls 2 contradictions:
  tap position higher → nominal, and feedwater program oxygenated → chelant. (The human read had called a different
  pair, oxygenated vs caustic, the likely reader miss; Argon keys that one arguable.)
- **What it means:** reader agreement certified 26 sets, but by an independent key 23 of them hold at least one pair a
  careful domain reader could defend as compatible. Two readers agreeing is a weak filter for this task; the sets need
  the repair (one axis per attribute, no relative terms, no class labels containing another value, no co-occurring
  values) before any certification means "every pair is a contradiction".
- "Arguable" is Argon's judgement, not a measurement; some of its calls are strict (a two-winding unit is the obvious
  reading of "the vector group of the transformer"). The repair should decide per attribute which readings to exclude
  by wording, not by trusting either key alone.
