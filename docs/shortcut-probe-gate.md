# Shortcut Probe Gate Specification (`triad_clustered_wilson_holm` v1.0)

This document specifies the statistical decision rule, pre-registered null false-fail and leak-detection benchmarks, minimum detectable effect (MDE) table, and governance conditions for the judge bake-off shortcut probe (`reasoning_dataset_lab.pilot_b.shortcut_probe`).

---

## 1. Gate Rules

### 1.1 Gate 1 — Out-of-Fold Shortcut Probe (`triad_clustered_wilson_holm`, Version `1.0`)

Each bake-off part consists of $N_{\text{triads}}$ contrast triads ($N_{\text{items}} = 3 N_{\text{triads}}$ items), where each triad contains one `supported`, one `unsupported`, and one `cannot_tell` claim built over the same material table.

The probe evaluates:
- **10 main feature families** (`gated = True`) on both **3-class** (`supported`, `unsupported`, `cannot_tell`; base rate $p_0 = 1/3$) and **binary** (`supported` vs. `not_supported`; base rate $p_0 = 2/3$) tasks via 5-fold triad-grouped out-of-fold (OOF) cross-validation:
  - `(a)` `a_claim_lexical_ngrams`: domain-masked claim surface lexical n-grams;
  - `(b)` `b_target_material_line`: sealed `target_line` holding the queried fact;
  - `(c)` `c_formatting_and_overlap`: numeric/unit vs. string formatting and normalised claim-material token overlap;
  - `claim_all_tokens_in_material`: normalised claim value and all-claim-token presence in the material;
  - `(d)` `d_closure_phrasing`: closure tag and header phrasing;
  - `(e)` `e_lengths_and_triad_rank`: material length, claim/material length ratio, and within-triad claim-length rank;
  - `(f)` `f_item_index_mod3`: item index modulo 3;
  - `(g)` `g_attribute_role`: attribute role (`number`, `interval`, `state`, `spec`, `open`, `date`);
  - `(h)` `h_attribute_name`: attribute name;
  - `(i)` `i_claim_length`: claim word and character length.
- **2 entity-value proximity probes** (`same_line_cooccurrence` and `min_entity_value_token_distance`):
  - 3-class and binary OOF accuracies are reported diagnostically (`gated = False`), because `supported` items co-occur with their entity and value on the same line by definition of a true table statement;
  - **`within_not_supported`** (`unsupported` vs. `cannot_tell`; base rate $p_0 = 1/2$ across $N_{\text{triads}}$ triads / $2 N_{\text{triads}}$ items) is **gated** (`within_not_supported_gated = True`).

#### Decision Rule (Option 1: Holm-Adjusted Triad-Clustered Wilson Score Interval)

1. **Degenerate probes:** Any probe-task whose feature has zero variance across all items (for example, `claim_all_tokens_in_material` on a repaired part where $100\%$ of claim values and tokens appear in the material) is marked `degenerate = True`, reported with separate `gd_accuracy` and `centroid_accuracy` (without a `max()` floor), excluded from the multiplicity family, and passes (`passed = True`).
2. **Non-degenerate family ($M$ gated probe-tasks):** Let $k \in \{1, \dots, M\}$ index all non-degenerate gated probe-tasks across the 10 main probes (3-class and binary) and the 2 proximity probes (`within_not_supported`). On a standard repaired part where `claim_all_tokens_in_material` (2 tasks) and `same_line_cooccurrence.within_not_supported` (1 task) are zero-variance, $M = 19$, or $M = 18$ when both `claim_all_tokens_in_material` tasks and one proximity task are degenerate (up to $M = 22$ if every gated task has non-zero variance).
3. **Triad-clustered score statistic and Wilson interval:** Because the 3 items within a triad share a material table and are assigned a joint permutation of `("supported", "unsupported", "cannot_tell")`, the independent sampling unit is the **triad** ($n = N_{\text{triads}}$). For task $k$ with OOF accuracy $\hat{p}_k$ and base rate $p_{0, k} \in \{1/3, 2/3, 1/2\}$:
   $$\text{SE}_{\text{triad}, k} = \sqrt{\frac{p_{0, k}(1 - p_{0, k})}{N_{\text{triads}}}}, \qquad z_k = \frac{\lvert \hat{p}_k - p_{0, k}\rvert}{\text{SE}_{\text{triad}, k}}, \qquad p_{\text{raw}, k} = 2\bigl(1 - \Phi(z_k)\bigr).$$
4. **Holm (1979) step-down multiplicity correction at family-wise $\alpha = 0.05$:**
   Order the $M$ non-degenerate gated probe-tasks by decreasing $z_{(1)} \ge z_{(2)} \ge \dots \ge z_{(M)}$ (increasing $p_{\text{raw}, (1)} \le \dots \le p_{\text{raw}, (M)}$). At step $r \in \{1, \dots, M\}$:
   $$\alpha_{(r)} = \frac{\alpha}{M - r + 1}, \qquad z_{\text{crit}, (r)} = \Phi^{-1}\!\left(1 - \frac{\alpha_{(r)}}{2}\right), \qquad p_{\text{Holm}, (r)} = \max_{j \le r} \min\!\bigl(1,\; (M - j + 1)\, p_{\text{raw}, (j)}\bigr).$$
   The Holm-adjusted triad-clustered Wilson score interval $[L_{(r)}, U_{(r)}]$ is computed at critical value $z = z_{\text{crit}, (r)}$ with $n = N_{\text{triads}}$ (Wilson 1927):
   $$\frac{\hat{p}_{(r)} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}_{(r)}(1 - \hat{p}_{(r)})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}.$$
   By the exact algebraic duality of Wilson (1927), $p_{0, (r)} \in [L_{(r)}, U_{(r)}] \iff z_{(r)} \le z_{\text{crit}, (r)} \iff p_{\text{raw}, (r)} \ge \alpha_{(r)}$.
   Under Holm's sequentially rejective rule, if step $r$ fails ($p_{0, (r)} \notin [L_{(r)}, U_{(r)}]$) and all preceding steps $1, \dots, r - 1$ also failed, task $(r)$ is marked `passed = False`; once the first ordered task passes, it and all remaining tasks with smaller $z$ pass.
5. **Removal of the flat $\pm 0.05$ margin:** The former flat point-estimate cap (`abs(oof - base) <= 0.05`) is dropped. Each probe-task reports `margin_over_base`, `abs_margin_over_base`, `z_score`, `raw_p`, `holm_adjusted_p`, `holm_alpha`, `holm_cluster_ci_low`, `holm_cluster_ci_high`, `in_holm_cluster_ci`, and `passed`.

### 1.2 Gate 2 — Attribute Role and Closure Class Shares (Unchanged)

Every attribute role (`number`, `interval`, `state`, `spec`, `open`, `date`) and every closure (`closed`, `open`) must have its 3-class shares (`supported`, `unsupported`, `cannot_tell`) within $[0.25, 0.42]$.

---

## 2. Pre-Registered Benchmarks (from Card `sh`)

### 2.1 Null False-Fail Rate Across Candidate Rules (200 Within-Triad Permutations, Seed `20261030`)

Under the null (permuting labels within each triad), the legacy flat $\pm 0.05$ margin failed $80.0\%\text{–}98.5\%$ of no-leak parts at $N = 34\text{–}40$ triads because $\pm 0.05$ is only $0.62\text{–}0.67\,\text{SE}_{\text{triad}}$.

| Gate 1 Rule Option | Null False-Fail at $N=40$ (Hybrid / Full 3C) | Null False-Fail at $N=34$ (Hybrid / Full 3C) | Leak Fixtures Caught (out of 12) | Smallest Detectable Leak at $N=34$ (3-Class / Binary / WNS) | Smallest Detectable Leak at $N=40$ (3-Class / Binary / WNS) | Smallest Detectable Leak at $N=430$ (3-Class / Binary / WNS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy Flat Margin** (`\|oof-base\| <= 0.05` & `cl_ci95`) | `80.0%` / `98.5%` | `91.0%` / `98.5%` | `12 / 12` (`100%`) | `+0.0588` / `+0.0588` / `+0.0588` | `+0.0583` / `+0.0583` / `+0.0625` | `+0.0450` / `+0.0450` / `+0.0477` |
| **Option 0: Unadjusted Cluster CI95 only** ($z = 1.9600$) | `4.5%` / `10.0%` | `5.0%` / `12.0%` | `12 / 12` (`100%`) | `+0.1585` / `+0.1585` / `+0.1681` | `+0.1461` / `+0.1461` / `+0.1549` | `+0.0446` / `+0.0446` / `+0.0473` |
| **Option 1 (Adopted): Cluster CI + Holm** ($\alpha = 0.05$, $z_1 = 2.99\text{–}3.02$) | **`0.0%` / `0.0%`** | **`0.5%` / `0.5%`** | **`12 / 12` (`100%`)** | **`+0.2418` / `+0.2418` / `+0.2565`** | **`+0.2230` / `+0.2230` / `+0.2365`** | **`+0.0680` / `+0.0680` / `+0.0721`** |
| **Option 2a: SE Margin $2.50 \cdot \text{SE}_{\text{triad}}$** ($z = 2.5000$) | `1.0%` / `3.0%` | `1.0%` / `1.5%` | `12 / 12` (`100%`) | `+0.2021` / `+0.2021` / `+0.2144` | `+0.1863` / `+0.1863` / `+0.1976` | `+0.0568` / `+0.0568` / `+0.0603` |
| **Option 2b: SE Margin $2.80 \cdot \text{SE}_{\text{triad}}$** ($z = 2.8000$) | `0.0%` / `0.0%` | `0.5%` / `0.5%` | `12 / 12` (`100%`) | `+0.2264` / `+0.2264` / `+0.2401` | `+0.2087` / `+0.2087` / `+0.2214` | `+0.0637` / `+0.0637` / `+0.0675` |

### 2.2 Per-Fixture Detection Breakdown across All 12 Leak Fixtures ($N = 34$ Triads / $102$ Items)

All 12 leak fixtures fail under Gate 1 (Option 1, $\alpha = 0.05$):

| # | Leak Fixture | Target Probe Failing | 3-Class OOF (Margin, $z_{\text{triad}}$) | Binary / WNS OOF (Margin, $z_{\text{triad}}$) | Caught by Option 1 (Holm Cluster CI, $\alpha=0.05$)? |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `fixture_a_claim_lexical_ngrams` | `a_claim_lexical_ngrams` | `1.0000` (`+0.6667`, $z=8.25$) | `1.0000` (`+0.3333`, $z=4.12$) | **YES (`100%`)** |
| 2 | `fixture_b_target_material_line` | `b_target_material_line` | `1.0000` (`+0.6667`, $z=8.25$) | `1.0000` (`+0.3333`, $z=4.12$) | **YES (`100%`)** |
| 3 | `fixture_c_formatting_and_overlap` | `c_formatting_and_overlap` | `1.0000` (`+0.6667`, $z=8.25$) | `1.0000` (`+0.3333`, $z=4.12$) | **YES (`100%`)** |
| 4 | `fixture_d_closure_phrasing` | `d_closure_phrasing` | `1.0000` (`+0.6667`, $z=8.25$) | `1.0000` (`+0.3333`, $z=4.12$) | **YES (`100%`)** |
| 5 | `fixture_e_lengths_and_triad_rank` | `e_lengths_and_triad_rank` | `1.0000` (`+0.6667`, $z=8.25$) | `1.0000` (`+0.3333`, $z=4.12$) | **YES (`100%`)** |
| 6 | `fixture_f_item_index_mod3` | `f_item_index_mod3` | `1.0000` (`+0.6667`, $z=8.25$) | `1.0000` (`+0.3333`, $z=4.12$) | **YES (`100%`)** |
| 7 | `fixture_g_attribute_role` | `g_attribute_role` | `0.6667` (`+0.3333`, $z=4.12$) | `1.0000` (`+0.3333`, $z=4.12$) | **YES (`100%`)** |
| 8 | `fixture_h_attribute_name` | `h_attribute_name` | `0.6667` (`+0.3333`, $z=4.12$) | `1.0000` (`+0.3333`, $z=4.12$) | **YES (`100%`)** |
| 9 | `fixture_i_claim_length` | `i_claim_length` | `1.0000` (`+0.6667`, $z=8.25$) | `1.0000` (`+0.3333`, $z=4.12$) | **YES (`100%`)** |
| 10 | `fixture_old_construction_overlap` | `claim_all_tokens_in_material` & `c_formatting_and_overlap` | `0.6667` (`+0.3333`, $z=4.12$) | `1.0000` (`+0.3333`, $z=4.12$) | **YES (`100%`)** |
| 11 | `fixture_proximity_within_not_supported` | `same_line_cooccurrence.within_not_supported` | *(ungated)* | `1.0000` (`+0.5000`, $z=5.83$) | **YES (`100%`)** |
| 12 | `fixture_pilot_40_overlap_leak` (34 triads) | `claim_all_tokens_in_material` | `0.5980` (`+0.2647`, $z=3.27$) | `0.9216` (`+0.2549`, $z=3.15$) | **YES (`100%`)** |

### 2.3 Minimum Detectable Single-Probe Margin (MDE) Table at `34`, `40`, `106`, and `430` Triads

At step 1 of the Holm procedure with family size $M = 18$ ($\alpha_1 = 0.05 / 18 = 0.002778$, critical value $z_{\text{crit}, 1} = 2.9913$), the minimum detectable single-probe margin is:
$$\text{MDE}(N_{\text{triads}}) = z_{\text{crit}, 1} \sqrt{\frac{p_0(1 - p_0)}{N_{\text{triads}}}}.$$

| Triad Count ($N_{\text{triads}}$) | Item Count ($N_{\text{items}}$) | $\text{SE}_{\text{triad}}$ (3-Class / Binary, $p_0 \in \{1/3, 2/3\}$) | MDE: 3-Class & Binary Margin ($M=18$, $z_1=2.9913$ / $M=19$, $z_1=3.0077$) | MDE: `within_not_supported` Margin ($p_0=1/2$, $M=18$ / $M=19$) | Interpretation / Milestone |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`34`** | `102` | `0.0808` | **`+0.2418` / `+0.2432` (~24 points)** | **`+0.2565` / `+0.2579` (~26 points)** | Pilot screen size (`part-pilot-40`); catches direct overlap (`+0.2647`, $z=3.27$) but not ~14-point single-probe leaks |
| **`40`** | `120` | `0.0745` | **`+0.2230` / `+0.2242` (~22 points)** | **`+0.2365` / `+0.2378` (~24 points)** | Planned 40-triad pilot batch |
| **`102–103`** *(exact 14-pt)* | `306–309` | `0.0467` | **`+0.1396` ($N=102, M=18$) / `+0.1399` ($N=103, M=19$)** | **`+0.1481` / `+0.1484`** | Exact triad count required to detect a **14-point** (`+0.1400`) 3-class/binary leak (`102` at $M=18$, `103` at $M=19$) |
| **`106–107`** *(+13.73%)* | `318–321` | `0.0458` | **`+0.1370` ($N=106, M=18$) / `+0.1377` ($N=106, M=19$)** | **`+0.1453` / `+0.1460`** | Triad count required to detect a `+0.1373` leak (e.g. `c_formatting_and_overlap` binary alone without `claim_all_tokens_in_material`) |
| **`406–411`** *(exact 7-pt)* | `1,218–1,233` | `0.0234` | **`+0.0700` ($N=406, M=18$) / `+0.0700` ($N=411, M=19$)** | **`+0.0742` / `+0.0742`** | Exact triad count required to detect a **7-point** (`+0.0700`) 3-class/binary leak (`406` at $M=18$, `411` at $M=19$) |
| **`430`** | `1,290` | `0.0227` | **`+0.0680` / `+0.0684` (~6.8 points)** | **`+0.0721` / `+0.0725` (~7.2 points)** | Full bake-off part size |

---

## 3. Mandatory Governance and Reporting Conditions

Per `GATE-RECALIBRATION-CONDITIONS.md` and Card `so` requirements (1)–(2):

1. **Mandatory report header and plain-language disclosure (Condition 1 & 2):**
   - Every probe report header (`report.header`) computes and carries `n_triads`, `n_items`, `gate_name`, `gate_version`, `alpha`, `family_size`, `min_detectable_single_probe_margin`, `min_detectable_within_not_supported_margin`, `triads_needed_for_14pt_leak` (`102` at $M=18$, `103` at $M=19$), `triads_needed_for_7pt_leak` (`406` at $M=18$, `411` at $M=19$), and `pass_meaning_note`.
   - At 34 triads, the report states verbatim:
     > "A pass means no single-probe leak above about 24 points at 34 triads. It is a coarse screen, not a clean bill."
   - **No report may say "no leak" from a pilot pass alone.** A 34–40 triad pilot pass is a coarse screen against large single-probe leaks ($\ge 22\text{–}24$ percentage points), not proof of the absence of smaller leaks.
2. **Pre-registered calibration and no retroactive re-grading:**
   - The gate rule (Option 1: Holm-adjusted triad-clustered Wilson interval at $\alpha = 0.05$) is pre-registered from the null false-fail distribution and the 12 leak fixtures before running on any new real-writer pilot output, and is never tuned on pilot outcomes.
   - Historical `part-pilot-40` is **not** re-graded and remains failed on its record (and its 34-triad overlap leak is also caught directly under Option 1 with $z = 3.27 > 2.991$).

---

## 4. References

- **Wilson (1927):** E. B. Wilson (1927), *"Probable Inference, the Law of Succession, and Statistical Inference,"* **Journal of the American Statistical Association**, 22(158), 209–212. DOI: `10.1080/01621459.1927.10502953`, URL: https://doi.org/10.1080/01621459.1927.10502953.
- **Holm (1979):** S. Holm (1979), *"A Simple Sequentially Rejective Multiple Test Procedure,"* **Scandinavian Journal of Statistics**, 6(2), 65–70. JSTOR: `4615733`, URL: https://www.jstor.org/stable/4615733.
