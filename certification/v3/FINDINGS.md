# Certification run v3: findings (R&D, 2026-10-28)

**What "certified" means here:** both blind readers called every one of a set's 20 ordered pairs a contradiction, **and**
the set had already keyed 20/20 "contradiction" in an independent blind second key (Argon, cards lg, lp, lw). Reader
agreement alone was shown in r3 to be a weak filter (FINDINGS-r3); v3 adds the blind key before any reader sees a set.

**Grant:** Publisher, 5090, cap 28,000 MiB card total. rdlab main `a5d507c6` (PR 21, the repaired sets; PR 23, Nemotron's
pin to `think: "medium"`, an effort hint the model's template knows, not a cap). Readers: Mistral Small 3.2 24B
(`think: false`) then Nemotron 3.5 Lightning 30B (`think: "medium"`), as recorded in the manifest's reader records. v3
control pairs (`CONTROL_PAIRS_SHA256` 3edf79bd…), prompt sha unchanged from r3 (6a65de77…). Licence registry 596e7b3c….

## Gates

| Reader | Pass | Compatible | Stop | Tokens | Gate |
|---|---|---|---|---|---|
| Mistral (A) | 20/20 | 20/20 | 20/20 | 1-5 | PASS |
| Nemotron, medium (B) | 20/20 | 20/20 | 20/20 | n/a | PASS |
| In-code controls inside certify-values | Mistral 20/20, Nemotron 20/20 | | | | passed |

## Result

- 50 sets, **44 certified, 6 queued** (r3 on the old sets: 26 and 24). 2,000 calls, 0 failed; 11:42:36-12:09:45Z (27 min).
- Peak 26,818 MiB card total (r3: 27,579), no HALT. Card free at the end (1,881 MiB, nothing loaded).
- Outputs: `manifest-v3.json` (62cbd66c…), `review_queue-v3.json` (a6d073a5…).
- **Nemotron length stops: 2 of 1,000 pairs** (r3: 25). This compares different sets and a different setting at once, so it
  is suggestive only; the arm below runs Nemotron at medium on r3's exact pairs for the clean comparison.

## The 11 failed pairs, read one by one (R&D's reading; not verified by an outside model)

| Set | Pair | Mistral / Nemotron | Read |
|---|---|---|---|
| turbines / turbine operating mode | accelerating → online, online → accelerating | unsure, incompatible / compatible, compatible | **wording:** "accelerating" also reads as a load ramp while online; Nemotron's call is defensible. Should say run-up to synchronous speed |
| compressors / compressor operating state | accelerating ↔ loaded (×2) | incompatible / compatible | **set fault:** a variable-speed compressor accelerates while loaded; Nemotron right |
| chillers / metering device type | electronic ↔ constantpressure (×2) | compatible / length, unsure | **set fault:** an electronic valve can run a constant-pressure mode; either call defensible |
| transformers / tap step position | minus1 → nominal, plus1 → nominal | incompatible / unsure, length | **reader (Nemotron):** a tap changer holds one position; Mistral right |
| transformers / unblended main core alloy | siliconsteel → ferrite | compatible / incompatible | **reader (Mistral):** one main core material; Nemotron right |
| chillers / metering device type | float → constantpressure | incompatible / unsure | **reader (Nemotron):** distinct valve types; Mistral right |
| chillers / sole condenser heat rejection loop | oncethroughwell → coolingtower | compatible / incompatible | **reader (Mistral):** one sole loop; Nemotron right |

**So:** 6 of the 11 are pair faults the blind key didn't catch (two "accelerating" pairs in operating-mode sets and the
electronic/constant-pressure pair) — the readers were right to refuse them; 5 are reader errors, split between the two
(Mistral 2, Nemotron 3). The 6 queued sets are the right ones to queue.

## A sample of the 44 certified sets (seed 20261028, 10 sets; R&D's reading)

| Set | Read |
|---|---|
| pumps / pump operating mode: standby, starting, delivering, coastingdown, stoppedlockedout | clean |
| boilers / coal firing system: pulverizedburner, travelingstoker, fluidizedbed, cyclonebarrel, underfeedretort | clean for one firing system; mild: some boilers are dual-fired, so "primary" or "sole" in the attribute would close it |
| chillers / leaving chilled-water setpoint step: 4, 6, 8, 10, 12 °C | clean |
| turbines / unblended base-oil grade: mineral, polyalphaolefin, ester, polyglycol, silicone | clean ("unblended") |
| pumps / sole mechanical shaft coupling: rigidflange, elastomericjaw, metallicdisc, gearcoupling, gridcoupling | clean ("sole") |
| vessels / sole propulsion power plant: diesel, steam, fullsailrig, gasturbine, fuelcell | clean ("sole" excludes hybrids) |
| pumps / sole secondary seal O-ring elastomer: viton, epdm, nitrile, neoprene, kalrez | clean (FKM, EPDM, NBR, CR, FFKM) |
| boilers / burner sequence step: coldstandby, prepurge, mainflame, postpurge, lockout | clean (a sequence register) |
| boilers / sole sootblower cleaning mechanism: steamjet, compressedair, waterlance, sonichorn, pulsedetonation | clean ("sole") |
| vessels / bridge telegraph order step: full/half astern, stop, half/full ahead | clean |

9 of 10 clean, 1 with a mild wording note — against 5 of 10 clean in the r3 sample.

## For the next table version

- The two operating-mode sets with "accelerating" and the metering-device set: reword ("running up to synchronous speed",
  "fixed-speed …") or replace; the blind key passed them, so the generator rule needs a line on modes that a variable-speed
  machine can hold together.
- The coal firing attribute: add "primary".

## The medium arm on the r3 table: the clean comparison

**"Medium" is an effort hint the model's template knows, not a cap.** This arm ran Nemotron alone at `think: "medium"` on
r3's exact table, pairs, v2 controls and prompt (rdlab `7dc3f222`), as an addendum to `manifest-r3.json`
(`parent_manifest_sha256` 77bfd2d5…, `setting_tier` "nemotron:think=medium"; r3's files untouched). Control at medium
20/20 pass, 20/20 compatible, 20/20 stop (GATE PASS). 1,000 calls, 1 failed (inside the 2% error rate); 12:11:09-12:47:52Z
(36.7 min; Nemotron's half of r3 at full thinking took about 40 min). Peak 26,816 MiB, no HALT; card free after.
Outputs: `manifest-r3-nemotron-medium.json` (fcd6e0d5…), `review_queue-r3-nemotron-medium.json` (840804bd…), and
`COMPARE-r3-nemotron-medium.md` (9aa6bcb8…).

| | Full thinking (r3) | Medium (arm) |
|---|---|---|
| Length stops (reported apart from verdicts) | 25 | **22** |
| Pairs with a verdict at both settings | 954 | |
| Same verdict, over those 954 pairs | **911 (95.5%)** | |
| Same verdict, over all 1,000 pairs | 911 (91.1%) | |
| Sets whose Nemotron half would pass | 29 | 30 |

Cross-table (rows full thinking, columns medium): incompatible→incompatible 862, compatible→compatible 49; verdicts that
flipped: incompatible→compatible 20, compatible→incompatible 15, and 8 more with an "unsure". Length stops: of r3's 25,
only 2 stopped again at medium; 23 now give a verdict, and 20 other pairs newly stop.

**What it shows:**
- **Medium does not cure length stops on these pairs:** 22 against 25, and mostly *different* pairs. v3's low count (2 of
  1,000) came from the cleaner sets, not from the setting. The earlier reading that medium "nearly stops" them was wrong.
- **Medium gives the same verdict as full thinking on 95.5% of pairs that get one at both settings**, and about 8% less
  wall time. Where the 35 flipped verdicts sit (which sets, which kinds of pair) hasn't been read yet.
- So the pin change is cheap and nearly neutral on verdicts, but it isn't the fix for length stops; clean sets are.
