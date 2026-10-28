# Certification run r3 (2026-10-28)

**Certified = reader agreement only, and it is a weak filter. The 26 certified sets need repair and a second pass before anyone uses them as keys.**
By an independent key (Argon's second keying, blind to the readers' verdicts), only 3 of the 26 certified sets are contradictions on all 20 pairs
(reactors / coolant type, vessels / registry port, vessels / ice class). Of the 520 certified pairs, 272 are strict contradictions, 216 arguable and 32 compatible.
23 of the 26 hold at least one pair a careful domain reader could defend as compatible. The sets are being repaired before any reuse.

| File | What it is |
|---|---|
| `manifest-r3.json` | 26 certified sets, both controls, the prompt and control-pair shas, `licence_registry_sha256` |
| `review_queue-r3.json` | JSONL, one line a set: 24 queued sets, 121 failed pairs |
| `FINDINGS.md` | the run, the gates, a sample read by R&D, and the second keying |
| `second-keying/keying-items.jsonl` | the 547 pairs given to Argon, shuffled, with no verdicts or group labels |
| `second-keying/argon-keys.jsonl` | Argon's key for each pair: `contradiction`, `arguable` or `compatible`, with a reason |

Unresolved: the 25 Nemotron length stops (thinking past `num_predict` 3072) are not verdicts. They stay queued. A run at a higher budget would go in a separate manifest with a `budget_tier`, never pooled with this one.

sha256: `manifest-r3.json` `77bfd2d51bf907060b8f8894691ed3e9dfdbb900512505b0ec7b3cacf5f00080`, `review_queue-r3.json` `d6a7aa10ffbab5663d22b269940c773296875b07d7bcc023b07661e215f8c9c1`, `keying-items.jsonl` `9a703ca13858e5eff918c4135b47fa47a1d7615c67e3adf587261bc66841bdc7`.
