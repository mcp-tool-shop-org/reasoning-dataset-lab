# Pilot B Calibration Record (2026-10-25)

**Date:** `2026-10-25`  
**Status:** Provisional until the next stub's per-call `p95` of `eval_count`.

This document records the empirical calibration measurements that the Pilot B provisional `num_predict` and `max_num_ctx` defaults rest on.

## Pinned Models & Manifest Digests

| Role | Ollama Tag | Manifest Digest (`sha256`) | Short Digest |
| :--- | :--- | :--- | :--- |
| `writer` (`render_material`, `word_claim`) | `qwen3.6:35b` | `a7eb95c53bcf96b4bdd008d0fab4a5dac88047d9c1a7a9ab88ed453423fbd87c` | `a7eb95c5` |
| `reader_1` (`read_claim`) | `nemotron-3.5-lightning:30b` | `e6905fffe2e2d9bbe325a9a2d2e074e3bd32683721bc6dfba409dede2f17f4dc` | `e6905fff` |
| `reader_2` (`read_claim`) | `granite4.2:8b` | `f586c02fdecdf151b656207c339aa003997345774a41768bac1fd6d2fb85913b` | `f586c02f` |

## Method

- Ollama's `/api/chat` response returns `eval_count` (total generated completion tokens: thinking + answer combined) and `message.thinking` text, but returns **no separate thinking token count**.
- The adapter records `thinking_words` (`len(message.thinking.split())`) as words for information only; nothing gates on `thinking_words`.
- `options.num_predict` is the only enforceable per-call token budget in Ollama, configured as one total `num_predict` per operation (`cli` > `config` > `default`).
- All calibration runs executed with `think: true` and `temperature: 0`, unloading each model after its run (`keep_alive: 0`).

## Measured Calibration Numbers

- **Render (`render_material`, writer `qwen3.6:35b`, digest `a7eb95c5`):**
  - `eval_count`: `2,095 / 2,306 / 3,462` (`n = 3`, min `2,095`, median `2,306`, p95 `3,462`, max `3,462`, `0` length stops at `num_predict 4,096`).
  - `thinking_words` (informational): `1,144 / 1,234 / 1,945`.
  - Peak VRAM: `25,229 MiB` at `num_ctx 8,192`.
  - Provisional defaults: `num_predict = 4,096`, `max_num_ctx = 8,192`.
- **Nemotron read (`read_claim` on `reader_1`, `nemotron-3.5-lightning:30b`, digest `e6905fff`):**
  - `eval_count`: median `1,138.5`, `p95` `3,086`, `max` `3,434` (`n = 20`, min `384`), `0` length stops at `num_predict 8,192`; plus one earlier call cut off at `4,096` (`n = 9`, `1` length stop at `4,096`, `8` stops with max `2,213`).
  - `thinking_words` (informational): `283` to `2,598` (`n = 20` at `num_predict 8,192`, median `862.5`, p95 `2,286`, max `2,598`).
  - Peak VRAM: `27,355 MiB` (`27.4 GB`) at `num_ctx 9,216` (runs under the recorded exception to the 26 GB plan rule: alone on the card, halt at `30,000 MiB`).
  - Provisional defaults: `num_predict = 6,144`, `max_num_ctx = 7,168`.
- **Granite read (`read_claim` on `reader_2`, `granite4.2:8b`, digest `f586c02f`):**
  - `eval_count`: max about `1,239` (`n = 9`, `0` length stops at `num_predict 4,096`).
  - `thinking_words` (informational): min `213`, median `330`, p95 `919`, max `919` (`n = 9`).
  - Peak VRAM: `8,825 MiB` (model release / process peak; steady-state `6,467 MiB` at `num_ctx 8,192`).
  - Provisional defaults: `num_predict = 2,048`, `max_num_ctx = 8,192`.
- **Word claim (`word_claim`, writer `qwen3.6:35b`, digest `a7eb95c5`):**
  - Keeps the current total `num_predict = 352` until measured.

## Provisional Default Summary

All defaults below are **provisional until the next stub's per-call p95** of `eval_count`:

| Operation / Role | Model (`digest`) | Default `num_predict` | Default `max_num_ctx` |
| :--- | :--- | ---: | ---: |
| `render_material` (`writer`) | `qwen3.6:35b` (`a7eb95c5`) | `4,096` | `8,192` |
| `word_claim` (`writer`) | `qwen3.6:35b` (`a7eb95c5`) | `352` (until measured) | `8,192` |
| `read_claim` (`reader_1`) | `nemotron-3.5-lightning:30b` (`e6905fff`) | `6,144` | `7,168` |
| `read_claim` (`reader_2`) | `granite4.2:8b` (`f586c02f`) | `2,048` | `8,192` |
