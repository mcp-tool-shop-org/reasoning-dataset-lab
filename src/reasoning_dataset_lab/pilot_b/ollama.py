"""Part A: Real loopback Ollama writer and blind-reader adapters for Pilot B (Card gk, Card gn, Card gp, Card gr, Card gs, Card gw).

Relies on the official Ollama HTTP API (`https://github.com/ollama/ollama/blob/main/docs/api.md`):
- `GET /api/tags` (`https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models`):
  reads `models[].name`, `models[].model`, and `models[].digest` to verify that the local model's
  digest equals the pinned `manifest_digest` before the first chat call. Never pulls (`/api/pull`
  is never called).
- `POST /api/chat` (`https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion`):
  sends `model` (pinned `ollama_tag`), `messages`, top-level `think` (`docs/api.md:499`), `stream: false`,
  and `options: {"temperature": 0, "num_predict": <configured_total_num_predict>, "num_ctx": <computed_num_ctx>}`
  (`docs/api.md:393`, `:513`); reads `message.content`, `message.thinking` (`docs/api.md:505`),
  `done_reason` (`docs/api.md:658`), and `eval_count`.
- `POST /api/chat` with `keep_alive: 0` (`https://github.com/ollama/ollama/blob/main/docs/api.md#unload-a-model`):
  unloads the model from VRAM after each wave.

Publisher rulings (Card gs & Card gw, 2026-10-25):
1. **One total `num_predict` per operation**, as configuration (`cli` > `config` > `default`), with the
   value and its source recorded. No separate thinking cap and no answer budget as controls.
2. **Thinking is recorded as `thinking_words`** (word count of `message.thinking`), for information only;
   nothing gates on it.
3. **Provisional defaults** (provisional until the full build's p95 of `eval_count`; see `docs/calibration-2026-10-25.md`):
   - `render_material` (writer, `qwen3.6:35b`): `num_predict = 8,192`, `max_num_ctx = 9,216`.
     Source: `docs/calibration-2026-10-25.md` — render `eval_count`
     `2,095 / 2,306 / 3,462` (`n = 3`, peak VRAM `25,229 MiB`); 10-triad stub p95 `3,515`, max `3,515` (`n = 9`), peak VRAM `25,157 MiB`;
     Qwen VRAM check 2026-10-25: peak 25,445 MiB at num_ctx 9,216 (render) and 25,399 MiB at 7,168 (word); halt line 26,000 MiB.
   - `word_claim` (writer, `qwen3.6:35b`): `num_predict = 6,144`, `max_num_ctx = 9,216`.
     Source: `docs/calibration-2026-10-25.md` — word_claim calibration `eval_count` min `1,101`, median `1,300`, p95 `3,743`, max `3,743` (`n = 15`, `0` length stops, peak VRAM `25,221 MiB`), initial default `4,255` (`p95 3,743` + calibrate margin `512`); 10-triad stub p95 `3,083`, max `3,743` (`n = 27`); raised to `6,144` (Qwen VRAM check 2026-10-25: peak 25,445 MiB at num_ctx 9,216 (render) and 25,399 MiB at 7,168 (word); halt line 26,000 MiB).
   - `read_claim` on `reader_1` (`nemotron-3.5-lightning:30b`): `num_predict = 6,144`, `max_num_ctx = 7,168`.
     Source: `docs/calibration-2026-10-25.md` — Nemotron `eval_count`
     p95 `3,086` and max `3,434` (`n = 20`), plus one earlier call cut off at `4,096` (`n = 9`); 10-triad stub p95 `2,160`, max `2,186` (`n = 27`). Runs under
     the Publisher's recorded exception to the 26 GB plan rule (alone on the card, halt at `30,000 MiB`;
     measured peak `27,355 MiB` at `num_ctx 9,216`, stub peak `27,185 MiB`).
   - `read_claim` on `reader_2` (`granite4.2:8b`): `num_predict = 2,560`, `max_num_ctx = 8,192`.
     Source: `docs/calibration-2026-10-25.md` — Granite calibration `eval_count` min `154`, median `340`, max `1,239` (`n = 9`, peak VRAM `8,825 MiB`); 10-triad stub p95 `1,251`, max `1,877` (`n = 27`, peak VRAM `8,242 MiB`); raised from `2,048` to `2,560` because its max of `1,877` was `92%` of `2,048`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import urlparse

from reasoning_dataset_lab.errors import refused_error, usage_error
from reasoning_dataset_lab.pilot_b.fact_tables import (
    FactTable,
    ProvedClaim,
)
from reasoning_dataset_lab.pilot_b.rendering import (
    StyleIssue,
    check_rendered_material,
    check_worded_claim,
    style_lint,
)
from reasoning_dataset_lab.pilot_b.unanimity import (
    ALLOWED_READER_VERDICTS,
    BlindReadPrompt,
    BlindReadResult,
    ModelIdentity,
    ReaderVerdict,
    ThinkSetting,
)

if TYPE_CHECKING:
    from reasoning_dataset_lab.pilot_b.models_and_build import PinnedModelSpec

DEFAULT_OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
LOOPBACK_HOST: str = "127.0.0.1"

OllamaOperation = Literal["render_material", "word_claim", "read_claim"]
PILOT_B_OPERATIONS: tuple[OllamaOperation, ...] = (
    "render_material",
    "word_claim",
    "read_claim",
)
NumPredictOperationKey = Literal[
    "render_material",
    "word_claim",
    "read_claim_reader_1",
    "read_claim_reader_2",
]
NUM_PREDICT_OPERATION_KEYS: tuple[NumPredictOperationKey, ...] = (
    "render_material",
    "word_claim",
    "read_claim_reader_1",
    "read_claim_reader_2",
)
NumPredictSource = Literal["cli", "config", "default"]
OllamaHttpTransport = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]

# Recorded total num_predict defaults per operation (Card gw Part A & Card he Part D — provisional until the full build's p95 of eval_count; see docs/calibration-2026-10-25.md):
# - render_material (qwen3.6:35b): 8,192 (eval_count 2,095 / 2,306 / 3,462, n = 3; stub p95 3,515, max 3,515, n = 9; Qwen VRAM check 2026-10-25: peak 25,445 MiB at num_ctx 9,216 (render) and 25,399 MiB at 7,168 (word); halt line 26,000 MiB; docs/calibration-2026-10-25.md)
# - word_claim (qwen3.6:35b): 6,144 (eval_count min 1,101, median 1,300, p95 3,743, max 3,743, n = 15; stub p95 3,083, max 3,743, n = 27; Qwen VRAM check 2026-10-25: peak 25,445 MiB at num_ctx 9,216 (render) and 25,399 MiB at 7,168 (word); halt line 26,000 MiB; docs/calibration-2026-10-25.md)
# - read_claim_reader_1 (nemotron-3.5-lightning:30b): 6,144 (eval_count p95 3,086, max 3,434, n = 20; stub p95 2,160, max 2,186, n = 27; docs/calibration-2026-10-25.md)
# - read_claim_reader_2 (granite4.2:8b): 2,560 (raised from 2,048 to 2,560 because stub max of 1,877 was 92% of 2,048; stub p95 1,251, max 1,877, n = 27; docs/calibration-2026-10-25.md)
DOCUMENTED_DEFAULT_RENDER_NUM_PREDICT: int = 8192
DOCUMENTED_DEFAULT_WORD_NUM_PREDICT: int = 6144
DOCUMENTED_DEFAULT_READ_READER_1_NUM_PREDICT: int = 6144
DOCUMENTED_DEFAULT_READ_READER_2_NUM_PREDICT: int = 2560
DOCUMENTED_DEFAULT_READ_NUM_PREDICT: int = DOCUMENTED_DEFAULT_READ_READER_1_NUM_PREDICT
DOCUMENTED_DEFAULT_NUM_PREDICT: int = DOCUMENTED_DEFAULT_RENDER_NUM_PREDICT

DOCUMENTED_DEFAULT_NUM_PREDICT_BY_OPERATION: dict[NumPredictOperationKey, int] = {
    "render_material": DOCUMENTED_DEFAULT_RENDER_NUM_PREDICT,
    "word_claim": DOCUMENTED_DEFAULT_WORD_NUM_PREDICT,
    "read_claim_reader_1": DOCUMENTED_DEFAULT_READ_READER_1_NUM_PREDICT,
    "read_claim_reader_2": DOCUMENTED_DEFAULT_READ_READER_2_NUM_PREDICT,
}

# Explicit num_ctx calculation and per-role ceilings (Card gp Part A, Card gs Ruling 3 & Card he Part D):
# options.num_ctx = round_up(prompt_tokens_estimate + num_predict + margin, 1024)
PROMPT_TOKEN_ESTIMATE_RULE: str = "ceil(utf8_bytes / 3)"
NUM_CTX_MARGIN_TOKENS: int = 256
NUM_CTX_STEP_TOKENS: int = 1024
DEFAULT_RENDER_MAX_NUM_CTX: int = 9216
DEFAULT_READER_1_MAX_NUM_CTX: int = 7168
DEFAULT_READER_2_MAX_NUM_CTX: int = 8192
# reader_3 (mistral-small3.2:24b) is used by the certification only, which runs at num_ctx 4096; its cap is not raised past reader_1's.
DEFAULT_READER_3_MAX_NUM_CTX: int = 7168

# The reader role of each reader family. One map: the adapter's role fallback and the certification's reader builder both use it.
READER_ROLE_BY_FAMILY: dict[str, str] = {
    "nemotron": "reader_1",
    "granite": "reader_2",
    "mistral": "reader_3",
}

# Per-role max_num_ctx defaults:
# - writer (qwen3.6:35b, weights 22.6 GB): 9216 (Qwen VRAM check 2026-10-25: peak 25,445 MiB at num_ctx 9,216 (render) and 25,399 MiB at 7,168 (word); halt line 26,000 MiB)
# - reader_1 (nemotron-3.5-lightning:30b, weights 25.4 GB): 7168 (Publisher's recorded exception to the 26 GB rule:
#   runs alone on the card, halt at 30,000 MiB; measured peak 27,355 MiB at num_ctx 9,216; stub peak 27,185 MiB)
# - reader_2 (granite4.2:8b, weights 5.3 GB): 8192 (measured peak 8,825 MiB at num_ctx 5,120; stub peak 8,242 MiB)
DEFAULT_ROLE_MAX_NUM_CTX: dict[str, int] = {
    "writer": DEFAULT_RENDER_MAX_NUM_CTX,
    "reader_1": DEFAULT_READER_1_MAX_NUM_CTX,
    "reader_2": DEFAULT_READER_2_MAX_NUM_CTX,
    "reader_3": DEFAULT_READER_3_MAX_NUM_CTX,
}


def estimate_prompt_tokens(messages: Sequence[Mapping[str, str]]) -> int:
    """Conservatively estimate prompt tokens using the named rule `ceil(utf8_bytes / 3)`."""
    utf8_bytes = sum(
        len(str(m.get("role", "")).encode("utf-8"))
        + len(str(m.get("content", "")).encode("utf-8"))
        for m in messages
    )
    return max(1, (utf8_bytes + 2) // 3)


def round_up_num_ctx(needed_tokens: int, step: int = NUM_CTX_STEP_TOKENS) -> int:
    """Round `needed_tokens` up to the next multiple of `step` (default 1024)."""
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    return max(step, ((max(1, needed_tokens) + step - 1) // step) * step)


def compute_num_ctx(
    prompt_tokens_estimate: int,
    num_predict: int,
    margin: int = NUM_CTX_MARGIN_TOKENS,
    step: int = NUM_CTX_STEP_TOKENS,
) -> int:
    """Compute `options.num_ctx = round_up(prompt_tokens_estimate + num_predict + margin, 1024)`."""
    return round_up_num_ctx(prompt_tokens_estimate + num_predict + margin, step=step)


ENV_NUM_PREDICT_GLOBAL: str = "RDLAB_NUM_PREDICT"
ENV_NUM_PREDICT_BY_KEY: dict[str, tuple[str, ...]] = {
    "render_material": (
        "RDLAB_RENDER_NUM_PREDICT",
        "RDLAB_NUM_PREDICT_RENDER_MATERIAL",
    ),
    "word_claim": (
        "RDLAB_WORD_NUM_PREDICT",
        "RDLAB_NUM_PREDICT_WORD_CLAIM",
    ),
    "read_claim": (
        "RDLAB_READ_NUM_PREDICT",
        "RDLAB_NUM_PREDICT_READ_CLAIM",
    ),
    "read_claim_reader_1": (
        "RDLAB_READER_1_NUM_PREDICT",
        "RDLAB_NUM_PREDICT_READER_1",
        "RDLAB_READ_READER_1_NUM_PREDICT",
    ),
    "read_claim_reader_2": (
        "RDLAB_READER_2_NUM_PREDICT",
        "RDLAB_NUM_PREDICT_READER_2",
        "RDLAB_READ_READER_2_NUM_PREDICT",
    ),
}


def _validate_positive_num_predict(raw_val: object, context_label: str) -> int:
    if isinstance(raw_val, bool):
        raise refused_error(
            "PILOT_B_INVALID_NUM_PREDICT",
            f"Invalid num_predict for {context_label}: boolean {raw_val!r} is not an integer.",
            "Pass a positive integer num_predict >= 1.",
        )
    if isinstance(raw_val, str):
        try:
            val = int(raw_val.strip())
        except ValueError as exc:
            raise refused_error(
                "PILOT_B_INVALID_NUM_PREDICT",
                f"Invalid num_predict for {context_label}: {raw_val!r} is not an integer.",
                "Pass a positive integer num_predict >= 1.",
            ) from exc
    elif isinstance(raw_val, int):
        val = raw_val
    else:
        raise refused_error(
            "PILOT_B_INVALID_NUM_PREDICT",
            f"Invalid num_predict for {context_label}: {raw_val!r} must be an integer >= 1.",
            "Pass a positive integer num_predict >= 1.",
        )
    if val < 1:
        raise refused_error(
            "PILOT_B_INVALID_NUM_PREDICT",
            f"num_predict must be >= 1 for {context_label}, got {val}.",
            "Pass a positive integer num_predict.",
        )
    return val


@dataclass(frozen=True)
class NumPredictConfig:
    """Configured per-operation total `options.num_predict` budget and provenance source (`cli` / `config` / `default`)."""

    render_material: int = DOCUMENTED_DEFAULT_RENDER_NUM_PREDICT
    word_claim: int = DOCUMENTED_DEFAULT_WORD_NUM_PREDICT
    read_claim_reader_1: int = DOCUMENTED_DEFAULT_READ_READER_1_NUM_PREDICT
    read_claim_reader_2: int = DOCUMENTED_DEFAULT_READ_READER_2_NUM_PREDICT
    source: NumPredictSource = "default"
    render_material_source: NumPredictSource = "default"
    word_claim_source: NumPredictSource = "default"
    read_claim_reader_1_source: NumPredictSource = "default"
    read_claim_reader_2_source: NumPredictSource = "default"

    def __post_init__(self) -> None:
        _validate_positive_num_predict(self.render_material, "render_material")
        _validate_positive_num_predict(self.word_claim, "word_claim")
        _validate_positive_num_predict(self.read_claim_reader_1, "read_claim_reader_1")
        _validate_positive_num_predict(self.read_claim_reader_2, "read_claim_reader_2")

    def num_predict_for(
        self,
        operation: OllamaOperation,
        *,
        role: str | None = None,
    ) -> int:
        if operation == "render_material":
            return self.render_material
        if operation == "word_claim":
            return self.word_claim
        if role == "reader_2":
            return self.read_claim_reader_2
        return self.read_claim_reader_1

    def source_for(
        self,
        operation: OllamaOperation,
        *,
        role: str | None = None,
    ) -> NumPredictSource:
        if operation == "render_material":
            return self.render_material_source
        if operation == "word_claim":
            return self.word_claim_source
        if role == "reader_2":
            return self.read_claim_reader_2_source
        return self.read_claim_reader_1_source

    @property
    def effective_num_predict(self) -> int:
        if (
            self.render_material
            == self.word_claim
            == self.read_claim_reader_1
            == self.read_claim_reader_2
        ):
            return self.render_material
        return max(
            self.render_material,
            self.word_claim,
            self.read_claim_reader_1,
            self.read_claim_reader_2,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.effective_num_predict,
            "source": self.source,
            "documented_defaults_by_operation": dict(
                DOCUMENTED_DEFAULT_NUM_PREDICT_BY_OPERATION
            ),
            "by_operation": {
                "render_material": {
                    "value": self.render_material,
                    "source": self.render_material_source,
                },
                "word_claim": {
                    "value": self.word_claim,
                    "source": self.word_claim_source,
                },
                "read_claim_reader_1": {
                    "value": self.read_claim_reader_1,
                    "source": self.read_claim_reader_1_source,
                },
                "read_claim_reader_2": {
                    "value": self.read_claim_reader_2,
                    "source": self.read_claim_reader_2_source,
                },
            },
        }


def _extract_num_predict_block_from_config_dict(data: Mapping[str, Any]) -> Any:
    if "pilot_b" in data and isinstance(data["pilot_b"], Mapping):
        pb = data["pilot_b"]
        if "num_predict" in pb:
            return pb["num_predict"]
    if "num_predict" in data:
        return data["num_predict"]
    return None


def _populate_op_dict_from_mapping(
    raw_map: Mapping[str, Any],
    label_prefix: str,
) -> tuple[int | None, dict[NumPredictOperationKey, int]]:
    global_val: int | None = None
    per_op: dict[NumPredictOperationKey, int] = {}
    if "default" in raw_map and raw_map["default"] is not None:
        global_val = _validate_positive_num_predict(
            raw_map["default"], f"{label_prefix}.default"
        )
    if "render_material" in raw_map and raw_map["render_material"] is not None:
        per_op["render_material"] = _validate_positive_num_predict(
            raw_map["render_material"], f"{label_prefix}.render_material"
        )
    if "render" in raw_map and raw_map["render"] is not None and "render_material" not in per_op:
        per_op["render_material"] = _validate_positive_num_predict(
            raw_map["render"], f"{label_prefix}.render"
        )
    if "word_claim" in raw_map and raw_map["word_claim"] is not None:
        per_op["word_claim"] = _validate_positive_num_predict(
            raw_map["word_claim"], f"{label_prefix}.word_claim"
        )
    if "word" in raw_map and raw_map["word"] is not None and "word_claim" not in per_op:
        per_op["word_claim"] = _validate_positive_num_predict(
            raw_map["word"], f"{label_prefix}.word"
        )
    # Generic read_claim sets both reader_1 and reader_2 unless overridden per reader
    for read_alias in ("read_claim", "read"):
        if read_alias in raw_map and raw_map[read_alias] is not None:
            read_val = _validate_positive_num_predict(
                raw_map[read_alias], f"{label_prefix}.{read_alias}"
            )
            per_op["read_claim_reader_1"] = read_val
            per_op["read_claim_reader_2"] = read_val
    for r1_alias in ("read_claim_reader_1", "reader_1", "read_reader_1"):
        if r1_alias in raw_map and raw_map[r1_alias] is not None:
            per_op["read_claim_reader_1"] = _validate_positive_num_predict(
                raw_map[r1_alias], f"{label_prefix}.{r1_alias}"
            )
    for r2_alias in ("read_claim_reader_2", "reader_2", "read_reader_2"):
        if r2_alias in raw_map and raw_map[r2_alias] is not None:
            per_op["read_claim_reader_2"] = _validate_positive_num_predict(
                raw_map[r2_alias], f"{label_prefix}.{r2_alias}"
            )
    return global_val, per_op


def resolve_num_predict_config(
    *,
    cli_num_predict: int | Mapping[str, int] | None = None,
    cli_by_operation: Mapping[str, int | None] | None = None,
    config_path: Path | None = None,
    config_data: Mapping[str, Any] | int | None = None,
    env: Mapping[str, str] | None = None,
    use_environ: bool = True,
    default_num_predict: int | Mapping[str, int] | None = None,
) -> NumPredictConfig:
    """Resolve `NumPredictConfig` with precedence `cli` > `config` (file/env) > `default`."""
    default_by_op: dict[NumPredictOperationKey, int] = dict(
        DOCUMENTED_DEFAULT_NUM_PREDICT_BY_OPERATION
    )
    if default_num_predict is not None:
        if isinstance(default_num_predict, Mapping):
            def_glob, def_ops = _populate_op_dict_from_mapping(
                default_num_predict, "default"
            )
            if def_glob is not None:
                for k in NUM_PREDICT_OPERATION_KEYS:
                    default_by_op[k] = def_glob
            default_by_op.update(def_ops)
        else:
            validated_def = _validate_positive_num_predict(
                default_num_predict, "default"
            )
            for k in NUM_PREDICT_OPERATION_KEYS:
                default_by_op[k] = validated_def

    active_env: Mapping[str, str] = (
        env if env is not None else (os.environ if use_environ else {})
    )

    loaded_cfg_block: Any = None
    if config_path is not None:
        if not config_path.exists() or not config_path.is_file():
            raise usage_error(
                "PILOT_B_CONFIG_NOT_FOUND",
                f"Pilot B config file not found: {config_path}",
                "Pass an existing JSON configuration file to --config.",
            )
        try:
            raw_cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise refused_error(
                "PILOT_B_INVALID_CONFIG_JSON",
                f"Invalid JSON in Pilot B config file {config_path}: {exc}",
                "Ensure the config file is valid UTF-8 JSON.",
            ) from exc
        if not isinstance(raw_cfg, Mapping):
            raise refused_error(
                "PILOT_B_INVALID_CONFIG_SHAPE",
                f"Pilot B config file {config_path} must be a JSON object.",
                "Format the config file as a JSON object.",
            )
        loaded_cfg_block = _extract_num_predict_block_from_config_dict(raw_cfg)
        if loaded_cfg_block is None and set(raw_cfg.keys()) <= {
            "default",
            "render_material",
            "render",
            "word_claim",
            "word",
            "read_claim",
            "read",
            "read_claim_reader_1",
            "read_claim_reader_2",
            "reader_1",
            "reader_2",
        }:
            loaded_cfg_block = raw_cfg

    if config_data is not None:
        if isinstance(config_data, Mapping):
            extracted = _extract_num_predict_block_from_config_dict(config_data)
            loaded_cfg_block = (
                extracted if extracted is not None else config_data
            )
        else:
            loaded_cfg_block = config_data

    cfg_global: int | None = None
    cfg_per_op: dict[NumPredictOperationKey, int] = {}
    if loaded_cfg_block is not None:
        if isinstance(loaded_cfg_block, Mapping):
            cfg_global, cfg_per_op = _populate_op_dict_from_mapping(
                loaded_cfg_block, "config"
            )
        else:
            cfg_global = _validate_positive_num_predict(loaded_cfg_block, "config")

    env_global_raw = active_env.get(ENV_NUM_PREDICT_GLOBAL, "").strip()
    env_global = (
        _validate_positive_num_predict(env_global_raw, ENV_NUM_PREDICT_GLOBAL)
        if env_global_raw
        else None
    )
    env_per_op: dict[NumPredictOperationKey, int] = {}
    for env_name in ENV_NUM_PREDICT_BY_KEY["render_material"]:
        raw_v = active_env.get(env_name, "").strip()
        if raw_v:
            env_per_op["render_material"] = _validate_positive_num_predict(
                raw_v, env_name
            )
            break
    for env_name in ENV_NUM_PREDICT_BY_KEY["word_claim"]:
        raw_v = active_env.get(env_name, "").strip()
        if raw_v:
            env_per_op["word_claim"] = _validate_positive_num_predict(
                raw_v, env_name
            )
            break
    for env_name in ENV_NUM_PREDICT_BY_KEY["read_claim"]:
        raw_v = active_env.get(env_name, "").strip()
        if raw_v:
            v = _validate_positive_num_predict(raw_v, env_name)
            env_per_op["read_claim_reader_1"] = v
            env_per_op["read_claim_reader_2"] = v
            break
    for env_name in ENV_NUM_PREDICT_BY_KEY["read_claim_reader_1"]:
        raw_v = active_env.get(env_name, "").strip()
        if raw_v:
            env_per_op["read_claim_reader_1"] = _validate_positive_num_predict(
                raw_v, env_name
            )
            break
    for env_name in ENV_NUM_PREDICT_BY_KEY["read_claim_reader_2"]:
        raw_v = active_env.get(env_name, "").strip()
        if raw_v:
            env_per_op["read_claim_reader_2"] = _validate_positive_num_predict(
                raw_v, env_name
            )
            break

    cli_global: int | None = None
    cli_per_op: dict[NumPredictOperationKey, int] = {}
    if cli_num_predict is not None:
        if isinstance(cli_num_predict, Mapping):
            cli_global, cli_per_op = _populate_op_dict_from_mapping(
                cli_num_predict, "cli"
            )
        else:
            cli_global = _validate_positive_num_predict(cli_num_predict, "cli")

    if cli_by_operation is not None:
        _, extra_cli_ops = _populate_op_dict_from_mapping(cli_by_operation, "cli")
        cli_per_op.update(extra_cli_ops)

    vals: dict[NumPredictOperationKey, int] = {}
    sources: dict[NumPredictOperationKey, NumPredictSource] = {}
    for k in NUM_PREDICT_OPERATION_KEYS:
        if k in cli_per_op:
            vals[k] = cli_per_op[k]
            sources[k] = "cli"
        elif cli_global is not None:
            vals[k] = cli_global
            sources[k] = "cli"
        elif k in cfg_per_op:
            vals[k] = cfg_per_op[k]
            sources[k] = "config"
        elif k in env_per_op:
            vals[k] = env_per_op[k]
            sources[k] = "config"
        elif cfg_global is not None:
            vals[k] = cfg_global
            sources[k] = "config"
        elif env_global is not None:
            vals[k] = env_global
            sources[k] = "config"
        else:
            vals[k] = default_by_op[k]
            sources[k] = "default"

    unique_sources = set(sources.values())
    if len(unique_sources) == 1:
        overall_source: NumPredictSource = next(iter(unique_sources))
    elif "cli" in unique_sources:
        overall_source = "cli"
    elif "config" in unique_sources:
        overall_source = "config"
    else:
        overall_source = "default"

    return NumPredictConfig(
        render_material=vals["render_material"],
        word_claim=vals["word_claim"],
        read_claim_reader_1=vals["read_claim_reader_1"],
        read_claim_reader_2=vals["read_claim_reader_2"],
        source=overall_source,
        render_material_source=sources["render_material"],
        word_claim_source=sources["word_claim"],
        read_claim_reader_1_source=sources["read_claim_reader_1"],
        read_claim_reader_2_source=sources["read_claim_reader_2"],
    )


def coerce_num_predict_config(
    num_predict: int | NumPredictConfig | Mapping[str, Any] | None = None,
    *,
    source: NumPredictSource | None = None,
    config_path: Path | None = None,
) -> NumPredictConfig:
    """Coerce an optional int, mapping, or `NumPredictConfig` into a validated `NumPredictConfig`."""
    if isinstance(num_predict, NumPredictConfig):
        return num_predict
    if num_predict is None:
        return resolve_num_predict_config(config_path=config_path)
    if isinstance(num_predict, Mapping):
        if source == "cli":
            return resolve_num_predict_config(
                cli_num_predict=num_predict, config_path=config_path
            )
        return resolve_num_predict_config(
            config_data=num_predict, config_path=config_path
        )
    val = _validate_positive_num_predict(num_predict, "num_predict")
    if source == "cli":
        return resolve_num_predict_config(cli_num_predict=val, config_path=config_path)
    if source == "config":
        return resolve_num_predict_config(config_data=val, config_path=config_path)
    if source == "default":
        return NumPredictConfig(
            render_material=val,
            word_claim=val,
            read_claim_reader_1=val,
            read_claim_reader_2=val,
            source="default",
            render_material_source="default",
            word_claim_source="default",
            read_claim_reader_1_source="default",
            read_claim_reader_2_source="default",
        )
    return resolve_num_predict_config(config_data=val, config_path=config_path)


def require_loopback_ollama_url(url: str) -> str:
    """Enforce interlock #140: only plain http://127.0.0.1:<port> on the loopback is permitted."""
    raw = url.strip() if isinstance(url, str) else ""
    parsed = urlparse(raw)
    if (
        parsed.scheme != "http"
        or parsed.hostname != LOOPBACK_HOST
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise refused_error(
            "PILOT_B_NON_LOOPBACK_URL",
            f"REFUSED: Ollama endpoint must be plain http://{LOOPBACK_HOST}:<port> on the loopback, got {url!r}.",
            f"Use {DEFAULT_OLLAMA_BASE_URL} (or a loopback tunnel port on {LOOPBACK_HOST}).",
        )
    return raw.rstrip("/")


def default_ollama_http_transport(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Default loopback-only HTTP transport for Ollama `/api/tags` and `/api/chat`."""
    require_loopback_ollama_url(url)
    data = (
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        if body is not None
        else None
    )
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310 - loopback verified above
            raw_bytes = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        raise refused_error(
            "PILOT_B_OLLAMA_HTTP_ERROR",
            f"Loopback Ollama request failed ({method} {url}): {exc}",
            "Ensure local Ollama is running on 127.0.0.1:11434.",
        ) from exc

    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise refused_error(
            "PILOT_B_OLLAMA_INVALID_JSON",
            f"Loopback Ollama returned invalid JSON from {method} {url}: {exc}",
            "Check that the loopback endpoint is an Ollama server.",
        ) from exc

    if not isinstance(parsed, dict):
        raise refused_error(
            "PILOT_B_OLLAMA_INVALID_JSON",
            f"Loopback Ollama returned non-object JSON from {method} {url}.",
            "Check that the loopback endpoint is an Ollama server.",
        )
    return parsed


@dataclass(frozen=True)
class OllamaCallRecord:
    """Per-call audit record for an Ollama writer or blind-reader invocation (Card gk, Card gn, Card gp, Card gs)."""

    model: str
    digest: str
    operation: OllamaOperation
    target_id: str
    prompt_sha256: str
    eval_tokens: int
    thinking_words: int
    num_predict: int
    wall_time_s: float
    think: ThinkSetting = True
    num_predict_source: NumPredictSource = "default"
    num_ctx: int = 0
    max_num_ctx: int = 0
    prompt_tokens_estimate: int = 0
    done_reason: str = "stop"
    prompt_eval_count: int = 0
    eval_count: int = 0

    def __post_init__(self) -> None:
        if self.eval_count == 0 and self.eval_tokens != 0:
            object.__setattr__(self, "eval_count", self.eval_tokens)

    def to_dict(self, *, include_target_id: bool = True) -> dict[str, object]:
        out: dict[str, object] = {
            "model": self.model,
            "digest": self.digest,
            "operation": self.operation,
            "prompt_sha256": self.prompt_sha256,
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
            "think": self.think,
            "num_predict": self.num_predict,
            "num_predict_source": self.num_predict_source,
            "num_ctx": self.num_ctx,
            "max_num_ctx": self.max_num_ctx,
            "done_reason": self.done_reason,
            "eval_tokens": self.eval_tokens,
            "thinking_words": self.thinking_words,
            "wall_time_s": round(self.wall_time_s, 6),
        }
        if include_target_id:
            out["target_id"] = self.target_id
        return out


@dataclass(frozen=True)
class OllamaUnloadRecord:
    """Record of an Ollama `keep_alive: 0` model unload request after a wave."""

    model: str
    digest: str
    keep_alive: int
    wall_time_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "digest": self.digest,
            "keep_alive": self.keep_alive,
            "wall_time_s": round(self.wall_time_s, 6),
        }


def _resolve_model_identity(model: ModelIdentity | PinnedModelSpec) -> ModelIdentity:
    if isinstance(model, ModelIdentity):
        return model
    return model.to_model_identity()


def _canonical_prompt_sha256(messages: list[dict[str, str]]) -> str:
    raw = json.dumps(
        messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def verify_local_ollama_model_digest(
    *,
    base_url: str,
    ollama_tag: str,
    expected_digest: str,
    http_transport: OllamaHttpTransport,
) -> str:
    """Verify via GET /api/tags that `ollama_tag` is present locally and matches `expected_digest`. Never pulls."""
    clean_base = require_loopback_ollama_url(base_url)
    tags_url = f"{clean_base}/api/tags"
    resp = http_transport("GET", tags_url, None)
    models = resp.get("models")
    if not isinstance(models, list):
        models = []

    candidates = {ollama_tag, f"{ollama_tag}:latest"}
    entry = next(
        (
            m
            for m in models
            if isinstance(m, dict)
            and (m.get("name") in candidates or m.get("model") in candidates)
        ),
        None,
    )
    if entry is None:
        raise refused_error(
            "PILOT_B_OLLAMA_MODEL_MISSING",
            f"Pinned model '{ollama_tag}' is not present in local Ollama at {tags_url}. Auto-pull is forbidden.",
            f"Run 'ollama pull {ollama_tag}' before running Pilot B.",
        )

    got_digest = str(entry.get("digest") or "").strip().lower().removeprefix("sha256:")
    norm_expected = expected_digest.strip().lower().removeprefix("sha256:")
    if got_digest != norm_expected:
        raise refused_error(
            "PILOT_B_OLLAMA_DIGEST_MISMATCH",
            f"Local Ollama model '{ollama_tag}' has digest '{got_digest}', which differs from pinned manifest_digest '{norm_expected}'.",
            f"Run 'ollama pull {ollama_tag}' and verify that its digest matches '{norm_expected}'.",
        )
    return norm_expected


KEY_BLIND_PHRASING_CUES: tuple[str, ...] = (
    "Open with a definite article phrase naming the attribute and entity.",
    "Begin with the entity name as the grammatical subject with an active verb (such as has, carries, or operates with).",
    "Open with a fronted prepositional phrase ('For ..., the ...') before stating the attribute and value.",
    "Use a possessive opening on the entity name or link the attribute and entity with 'for' rather than 'of'.",
)


def derive_key_blind_phrasing_cue(claim_id: str) -> str:
    """Derive a deterministic phrasing cue solely from the opaque hash in `claim_id` (never from `claim.key`)."""
    suffix = claim_id.rsplit("-c-", 1)[-1].strip()
    try:
        hash_int = int(suffix, 16)
    except ValueError:
        hash_int = int(hashlib.sha256(claim_id.encode("utf-8")).hexdigest()[:8], 16)
    return KEY_BLIND_PHRASING_CUES[hash_int % len(KEY_BLIND_PHRASING_CUES)]


def build_writer_render_messages(table: FactTable) -> list[dict[str, str]]:
    """Build the `/api/chat` messages payload for `render_material`."""
    table_json = json.dumps(
        table.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the Pilot B writer. Render each fact in the formal FactTable into one numbered "
                "sentence per line ('1. ...', '2. ...') in natural, grammatical English with varied, "
                "natural phrasing across lines, preserving exact entities, attributes, numbers, dates, "
                "intervals, polarity, and quantifier direction. Introduce no extra numbers or proper "
                "nouns. Keep subject-verb agreement for plural attributes, never use 'equals' or "
                "'is equal to' for an exact value, never write bare entity-attribute-verb without a "
                "connecting article or preposition, and do not repeat a unit-noun (never '36 holds' "
                "after 'cargo holds'). "
                "State an interval as 'between <lo> and <hi>'. For an exact fact, link the attribute to its "
                "value with a direct verb (is, operates in, uses, has), never a hedged link such as "
                "'characterized by' or 'associated with'."
            ),
        },
        {
            "role": "user",
            "content": f"Render the following formal FactTable into numbered lines.\nFACTS_JSON:\n{table_json}",
        },
    ]


def normalise_material(text: str) -> str:
    """Regular whitespace in a rendered material: one space between words, ``\n`` line ends, no trailing blank lines.

    Applied to the writer's output before it is checked and before anything reads it, so an irregularity of the
    model's spacing never reaches a reader or the sealed part (which accepts only regular whitespace).
    """

    lines = [" ".join(line.split()) for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def normalise_claim(text: str) -> str:
    """Regular whitespace in a worded claim: one space between words, nothing at either end."""

    return " ".join(text.split())


def build_writer_word_messages(claim: ProvedClaim) -> list[dict[str, str]]:
    """Build the `/api/chat` messages payload for `word_claim`."""
    claim_spec = {
        "claim_id": claim.claim_id,
        "table_id": claim.table_id,
        "entity": claim.entity,
        "attribute": claim.attribute,
        "fact_type": claim.fact_type,
        "claimed_value": (
            [int(claim.claimed_value[0]), int(claim.claimed_value[1])]
            if isinstance(claim.claimed_value, tuple)
            else claim.claimed_value
        ),
        "unit": claim.unit,
        "op": claim.op,
    }
    claim_json = json.dumps(
        claim_spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    phrasing_cue = derive_key_blind_phrasing_cue(claim.claim_id)
    return [
        {
            "role": "system",
            "content": (
                "You are the Pilot B writer. Word the formal claim into a single declarative sentence in "
                "natural, grammatical English with varied, natural phrasing, preserving its exact entity, "
                "attribute, value, unit, polarity, and quantifier direction. Introduce no unlisted numbers "
                "or proper nouns. Keep subject-verb agreement for plural attributes, never use 'equals' or "
                "'is equal to' for an exact value, never write bare entity-attribute-verb without a "
                "connecting article or preposition, and do not repeat a unit-noun (never '36 holds' "
                "after 'cargo holds'). "
                "State an interval as 'between <lo> and <hi>'. For an exact fact, link the attribute to its "
                "value with a direct verb (is, operates in, uses, has), never a hedged link such as "
                "'characterized by' or 'associated with'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Word the following formal ProvedClaim into one sentence.\n"
                f"PHRASING_CUE: {phrasing_cue}\n"
                f"CLAIM_JSON:\n{claim_json}"
            ),
        },
    ]


def build_reader_read_messages(prompt: BlindReadPrompt) -> list[dict[str, str]]:
    """Build the `/api/chat` messages payload for `read_claim`."""
    return [
        {
            "role": "system",
            "content": (
                "You are a blind verification reader. Given only the Material and a single Claim, "
                "reply with exactly one token: supported, unsupported, cannot_tell, or ambiguous."
            ),
        },
        {
            "role": "user",
            "content": prompt.user_message,
        },
    ]


def format_reference_material(table: FactTable) -> str:
    """Format a `FactTable` into numbered lines for `read_claim` calibration prompts."""
    lines: list[str] = []
    for idx, f in enumerate(table.facts, start=1):
        unit_suffix = f" {f.unit}" if f.unit else ""
        if f.op == "between" and isinstance(f.value, tuple):
            phrase = f"is between {f.value[0]} and {f.value[1]}{unit_suffix}"
        elif f.op == "at_least":
            phrase = f"is at least {f.value}{unit_suffix}"
        elif f.op == "at_most":
            phrase = f"is at most {f.value}{unit_suffix}"
        elif f.op == "not_equal":
            phrase = f"is not {f.value}{unit_suffix}"
        else:
            phrase = f"is {f.value}{unit_suffix}"
        lines.append(f"{idx}. {f.entity} {f.attribute} {phrase}.")
    return "\n".join(lines)


def format_reference_claim(claim: ProvedClaim) -> str:
    """Format a `ProvedClaim` into one sentence for `read_claim` calibration prompts."""
    unit_suffix = f" {claim.unit}" if claim.unit else ""
    if claim.op == "between" and isinstance(claim.claimed_value, tuple):
        phrase = f"is between {claim.claimed_value[0]} and {claim.claimed_value[1]}{unit_suffix}"
    elif claim.op == "at_least":
        phrase = f"is at least {claim.claimed_value}{unit_suffix}"
    elif claim.op == "at_most":
        phrase = f"is at most {claim.claimed_value}{unit_suffix}"
    elif claim.op == "not_equal":
        phrase = f"is not {claim.claimed_value}{unit_suffix}"
    else:
        phrase = f"is {claim.claimed_value}{unit_suffix}"
    return f"{claim.entity} {claim.attribute} {phrase}."


def _extract_token_counts(
    resp: dict[str, Any],
    content: str,
    thinking_text: str,
) -> tuple[int, int]:
    """Extract `(eval_tokens, thinking_words)` from an Ollama `/api/chat` response.

    `eval_count` is Ollama's real generated token count; `thinking_words` is the word count
    of `message.thinking` (`len(thinking_text.split())`), recorded for information only.
    """
    raw_eval = resp.get("eval_count")
    if isinstance(raw_eval, int) and not isinstance(raw_eval, bool) and raw_eval >= 0:
        eval_tokens = raw_eval
    else:
        eval_tokens = len(content.split())

    raw_think_words = resp.get("thinking_words")
    if (
        isinstance(raw_think_words, int)
        and not isinstance(raw_think_words, bool)
        and raw_think_words >= 0
    ):
        thinking_words = raw_think_words
    else:
        thinking_words = len(thinking_text.split()) if thinking_text.strip() else 0

    return eval_tokens, thinking_words


def _extract_prompt_eval_count(resp: Mapping[str, Any]) -> int:
    """Extract Ollama's real `prompt_eval_count` from an `/api/chat` response."""
    raw_prompt_eval = resp.get("prompt_eval_count")
    if (
        isinstance(raw_prompt_eval, int)
        and not isinstance(raw_prompt_eval, bool)
        and raw_prompt_eval >= 0
    ):
        return raw_prompt_eval
    return 0


class OllamaWriterAdapter:
    """Real loopback Ollama writer adapter implementing `PilotBWriterAdapter` (Card gk Part A, Card gp, Card gs)."""

    is_fake: ClassVar[bool] = False

    def __init__(
        self,
        model: ModelIdentity | PinnedModelSpec,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        num_predict: int | NumPredictConfig | Mapping[str, Any] | None = None,
        num_predict_source: NumPredictSource | None = None,
        max_num_ctx: int | None = None,
        http_transport: OllamaHttpTransport | None = None,
    ) -> None:
        self._num_predict_config = coerce_num_predict_config(
            num_predict, source=num_predict_source
        )
        self._base_url = require_loopback_ollama_url(base_url)
        self._model = _resolve_model_identity(model)
        resolved_max_ctx = (
            max_num_ctx
            if max_num_ctx is not None
            else (self._model.max_num_ctx or DEFAULT_ROLE_MAX_NUM_CTX["writer"])
        )
        if resolved_max_ctx < NUM_CTX_STEP_TOKENS:
            raise refused_error(
                "PILOT_B_INVALID_MAX_NUM_CTX",
                f"max_num_ctx must be >= {NUM_CTX_STEP_TOKENS}, got {resolved_max_ctx}.",
                f"Configure a positive max_num_ctx >= {NUM_CTX_STEP_TOKENS}.",
            )
        self._max_num_ctx = resolved_max_ctx
        self._think: ThinkSetting = self._model.think
        self._num_predict = self._num_predict_config.effective_num_predict
        self._transport: OllamaHttpTransport = (
            http_transport if http_transport is not None else default_ollama_http_transport
        )
        self._verified_local_digest = False
        self._call_records: list[OllamaCallRecord] = []
        self._unload_records: list[OllamaUnloadRecord] = []
        self._last_raw_output: str = ""

    @property
    def model(self) -> ModelIdentity:
        return self._model

    @property
    def ollama_tag(self) -> str:
        return self._model.name

    @property
    def manifest_digest(self) -> str:
        return self._model.digest

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def think(self) -> ThinkSetting:
        return self._think

    @property
    def num_predict(self) -> int:
        return self._num_predict

    @property
    def num_predict_config(self) -> NumPredictConfig:
        return self._num_predict_config

    @property
    def max_num_ctx(self) -> int:
        return self._max_num_ctx

    @property
    def http_transport(self) -> OllamaHttpTransport:
        return self._transport

    @property
    def call_records(self) -> tuple[OllamaCallRecord, ...]:
        return tuple(self._call_records)

    @property
    def last_call_record(self) -> OllamaCallRecord | None:
        return self._call_records[-1] if self._call_records else None

    @property
    def unload_records(self) -> tuple[OllamaUnloadRecord, ...]:
        return tuple(self._unload_records)

    @property
    def last_raw_output(self) -> str:
        return self._last_raw_output

    def _ensure_verified(self) -> None:
        if not self._verified_local_digest:
            verify_local_ollama_model_digest(
                base_url=self._base_url,
                ollama_tag=self.ollama_tag,
                expected_digest=self.manifest_digest,
                http_transport=self._transport,
            )
            self._verified_local_digest = True

    def _chat(
        self,
        *,
        operation: OllamaOperation,
        target_id: str,
        messages: list[dict[str, str]],
    ) -> str:
        self._ensure_verified()
        prompt_sha = _canonical_prompt_sha256(messages)
        prompt_tokens_estimate = estimate_prompt_tokens(messages)
        op_num_predict = self._num_predict_config.num_predict_for(operation)
        op_source = self._num_predict_config.source_for(operation)
        num_ctx = compute_num_ctx(
            prompt_tokens_estimate,
            op_num_predict,
            margin=NUM_CTX_MARGIN_TOKENS,
            step=NUM_CTX_STEP_TOKENS,
        )
        if num_ctx > self._max_num_ctx:
            self._call_records.append(
                OllamaCallRecord(
                    model=self.ollama_tag,
                    digest=self.manifest_digest,
                    operation=operation,
                    target_id=target_id,
                    prompt_sha256=prompt_sha,
                    eval_tokens=0,
                    thinking_words=0,
                    num_predict=op_num_predict,
                    num_predict_source=op_source,
                    wall_time_s=0.0,
                    think=self._think,
                    num_ctx=num_ctx,
                    max_num_ctx=self._max_num_ctx,
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    done_reason="max_num_ctx_exceeded",
                )
            )
            self._last_raw_output = ""
            raise refused_error(
                "PILOT_B_NUM_CTX_EXCEEDS_MAX",
                f"Writer call '{target_id}' ({operation}) requires num_ctx={num_ctx} "
                f"(prompt_tokens_estimate={prompt_tokens_estimate} + num_predict={op_num_predict} + "
                f"margin={NUM_CTX_MARGIN_TOKENS}), which exceeds max_num_ctx={self._max_num_ctx} "
                f"for '{self.ollama_tag}'.",
                "Reduce num_predict or configure a higher max_num_ctx if VRAM permits.",
            )

        chat_url = f"{self._base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.ollama_tag,
            "messages": messages,
            "think": self._think,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": op_num_predict,
                "num_ctx": num_ctx,
            },
        }
        t0 = time.perf_counter()
        resp = self._transport("POST", chat_url, payload)
        wall_s = max(0.0, time.perf_counter() - t0)

        msg = resp.get("message")
        if not isinstance(msg, dict):
            msg = {}
        content = str(msg.get("content") or "").strip()
        thinking_text = str(msg.get("thinking") or "")
        done_reason = str(resp.get("done_reason") or "stop").strip()
        eval_tokens, thinking_words = _extract_token_counts(resp, content, thinking_text)
        prompt_eval_count = _extract_prompt_eval_count(resp)

        record = OllamaCallRecord(
            model=self.ollama_tag,
            digest=self.manifest_digest,
            operation=operation,
            target_id=target_id,
            prompt_sha256=prompt_sha,
            eval_tokens=eval_tokens,
            thinking_words=thinking_words,
            num_predict=op_num_predict,
            num_predict_source=op_source,
            wall_time_s=wall_s,
            think=self._think,
            num_ctx=num_ctx,
            max_num_ctx=self._max_num_ctx,
            prompt_tokens_estimate=prompt_tokens_estimate,
            done_reason=done_reason,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_tokens,
        )
        self._call_records.append(record)
        self._last_raw_output = content

        if done_reason == "length":
            raise refused_error(
                "PILOT_B_OLLAMA_LENGTH_STOP",
                f"Writer call '{target_id}' ({operation}) stopped with done_reason='length' "
                f"(num_predict={op_num_predict}).",
                "Increase num_predict for the operation.",
            )
        return content

    def render_material(self, table: FactTable) -> str:
        """Render `table` into numbered prose via `POST /api/chat` and run `check_rendered_material`."""
        messages = build_writer_render_messages(table)
        rendered = self._chat(
            operation="render_material",
            target_id=table.table_id,
            messages=messages,
        )
        rendered = normalise_material(rendered)
        check_rendered_material(table, rendered)
        return rendered

    def word_claim(self, table: FactTable, claim: ProvedClaim) -> str:
        """Word `claim` into a single sentence via `POST /api/chat` and run `check_worded_claim`."""
        messages = build_writer_word_messages(claim)
        worded = self._chat(
            operation="word_claim",
            target_id=claim.claim_id,
            messages=messages,
        )
        worded = normalise_claim(worded)
        check_worded_claim(table, claim, worded)
        return worded

    def unload_model(self) -> OllamaUnloadRecord:
        """Unload the writer model from VRAM after a wave using `keep_alive: 0`."""
        chat_url = f"{self._base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.ollama_tag,
            "messages": [],
            "keep_alive": 0,
            "stream": False,
        }
        t0 = time.perf_counter()
        self._transport("POST", chat_url, payload)
        wall_s = max(0.0, time.perf_counter() - t0)
        rec = OllamaUnloadRecord(
            model=self.ollama_tag,
            digest=self.manifest_digest,
            keep_alive=0,
            wall_time_s=wall_s,
        )
        self._unload_records.append(rec)
        return rec


def parse_reader_verdict(raw_text: str) -> ReaderVerdict:
    """Strictly parse a reader answer into `supported`, `unsupported`, `cannot_tell`, or `ambiguous`.

    Anything outside those exact four labels (or a JSON object `{"verdict": "<label>"}` with no
    conflicting text) counts as `ambiguous`.
    """
    cleaned = raw_text.strip()
    if not cleaned:
        return "ambiguous"
    lowered = cleaned.lower()
    if lowered in ALLOWED_READER_VERDICTS:
        return lowered  # type: ignore[return-value]
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return "ambiguous"
        if isinstance(parsed, dict) and set(parsed.keys()) <= {"verdict", "reasoning"}:
            v = parsed.get("verdict")
            if isinstance(v, str) and v.strip().lower() in ALLOWED_READER_VERDICTS:
                return v.strip().lower()  # type: ignore[return-value]
    return "ambiguous"


class OllamaBlindReaderAdapter:
    """Real loopback Ollama blind-reader adapter implementing `BlindReaderAdapter` (Card gk Part B, Card gp, Card gs)."""

    is_fake: ClassVar[bool] = False

    def __init__(
        self,
        model: ModelIdentity | PinnedModelSpec,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        num_predict: int | NumPredictConfig | Mapping[str, Any] | None = None,
        num_predict_source: NumPredictSource | None = None,
        role: Literal["reader_1", "reader_2", "reader_3"] | None = None,
        max_num_ctx: int | None = None,
        think_overridden: bool = False,
        http_transport: OllamaHttpTransport | None = None,
    ) -> None:
        self._num_predict_config = coerce_num_predict_config(
            num_predict, source=num_predict_source
        )
        self._base_url = require_loopback_ollama_url(base_url)
        self._model = _resolve_model_identity(model)
        if role is None:
            mapped = READER_ROLE_BY_FAMILY.get(self._model.normalized_family)
            if mapped is None:
                raise refused_error(
                    "PILOT_B_UNKNOWN_READER_FAMILY",
                    f"No reader role for the model family '{self._model.normalized_family}'.",
                    f"Pass role explicitly, or use a family from: {', '.join(sorted(READER_ROLE_BY_FAMILY))}.",
                )
            role = mapped  # type: ignore[assignment]
        resolved_role: Literal["reader_1", "reader_2", "reader_3"] = role  # type: ignore[assignment]
        self._role = resolved_role
        default_role_max = DEFAULT_ROLE_MAX_NUM_CTX[resolved_role]
        resolved_max_ctx = (
            max_num_ctx
            if max_num_ctx is not None
            else (self._model.max_num_ctx or default_role_max)
        )
        if resolved_max_ctx < NUM_CTX_STEP_TOKENS:
            raise refused_error(
                "PILOT_B_INVALID_MAX_NUM_CTX",
                f"max_num_ctx must be >= {NUM_CTX_STEP_TOKENS}, got {resolved_max_ctx}.",
                f"Configure a positive max_num_ctx >= {NUM_CTX_STEP_TOKENS}.",
            )
        self._max_num_ctx = resolved_max_ctx
        self._think: ThinkSetting = self._model.think
        self._think_overridden = bool(think_overridden)
        self._num_predict = self._num_predict_config.num_predict_for(
            "read_claim", role=self._role
        )
        self._transport: OllamaHttpTransport = (
            http_transport if http_transport is not None else default_ollama_http_transport
        )
        self._verified_local_digest = False
        self._call_records: list[OllamaCallRecord] = []
        self._unload_records: list[OllamaUnloadRecord] = []

    @property
    def model(self) -> ModelIdentity:
        return self._model

    @property
    def role(self) -> Literal["reader_1", "reader_2", "reader_3"]:
        return self._role

    @property
    def ollama_tag(self) -> str:
        return self._model.name

    @property
    def manifest_digest(self) -> str:
        return self._model.digest

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def think(self) -> ThinkSetting:
        return self._think

    @property
    def think_overridden(self) -> bool:
        return self._think_overridden

    @property
    def num_predict(self) -> int:
        return self._num_predict

    @property
    def num_predict_config(self) -> NumPredictConfig:
        return self._num_predict_config

    @property
    def max_num_ctx(self) -> int:
        return self._max_num_ctx

    @property
    def http_transport(self) -> OllamaHttpTransport:
        return self._transport

    @property
    def call_records(self) -> tuple[OllamaCallRecord, ...]:
        return tuple(self._call_records)

    @property
    def last_call_record(self) -> OllamaCallRecord | None:
        return self._call_records[-1] if self._call_records else None

    @property
    def unload_records(self) -> tuple[OllamaUnloadRecord, ...]:
        return tuple(self._unload_records)

    def _ensure_verified(self) -> None:
        if not self._verified_local_digest:
            verify_local_ollama_model_digest(
                base_url=self._base_url,
                ollama_tag=self.ollama_tag,
                expected_digest=self.manifest_digest,
                http_transport=self._transport,
            )
            self._verified_local_digest = True

    def _chat(
        self,
        *,
        target_id: str,
        messages: list[dict[str, str]],
    ) -> tuple[str, OllamaCallRecord]:
        self._ensure_verified()
        prompt_sha = _canonical_prompt_sha256(messages)
        prompt_tokens_estimate = estimate_prompt_tokens(messages)
        op_num_predict = self._num_predict_config.num_predict_for(
            "read_claim", role=self._role
        )
        op_source = self._num_predict_config.source_for("read_claim", role=self._role)
        num_ctx = compute_num_ctx(
            prompt_tokens_estimate,
            op_num_predict,
            margin=NUM_CTX_MARGIN_TOKENS,
            step=NUM_CTX_STEP_TOKENS,
        )
        if num_ctx > self._max_num_ctx:
            record = OllamaCallRecord(
                model=self.ollama_tag,
                digest=self.manifest_digest,
                operation="read_claim",
                target_id=target_id,
                prompt_sha256=prompt_sha,
                eval_tokens=0,
                thinking_words=0,
                num_predict=op_num_predict,
                num_predict_source=op_source,
                wall_time_s=0.0,
                think=self._think,
                num_ctx=num_ctx,
                max_num_ctx=self._max_num_ctx,
                prompt_tokens_estimate=prompt_tokens_estimate,
                done_reason="max_num_ctx_exceeded",
            )
            self._call_records.append(record)
            raise refused_error(
                "PILOT_B_NUM_CTX_EXCEEDS_MAX",
                f"Reader call '{target_id}' (read_claim, {self._role}) requires num_ctx={num_ctx} "
                f"(prompt_tokens_estimate={prompt_tokens_estimate} + num_predict={op_num_predict} + "
                f"margin={NUM_CTX_MARGIN_TOKENS}), which exceeds max_num_ctx={self._max_num_ctx} "
                f"for '{self.ollama_tag}'.",
                "Reduce num_predict or configure a higher max_num_ctx if VRAM permits.",
            )

        chat_url = f"{self._base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.ollama_tag,
            "messages": messages,
            "think": self._think,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": op_num_predict,
                "num_ctx": num_ctx,
            },
        }
        t0 = time.perf_counter()
        resp = self._transport("POST", chat_url, payload)
        wall_s = max(0.0, time.perf_counter() - t0)

        msg = resp.get("message")
        if not isinstance(msg, dict):
            msg = {}
        raw_text = str(msg.get("content") or "").strip()
        thinking_text = str(msg.get("thinking") or "")
        done_reason = str(resp.get("done_reason") or "stop").strip()
        eval_tokens, thinking_words = _extract_token_counts(resp, raw_text, thinking_text)
        prompt_eval_count = _extract_prompt_eval_count(resp)

        record = OllamaCallRecord(
            model=self.ollama_tag,
            digest=self.manifest_digest,
            operation="read_claim",
            target_id=target_id,
            prompt_sha256=prompt_sha,
            eval_tokens=eval_tokens,
            thinking_words=thinking_words,
            num_predict=op_num_predict,
            num_predict_source=op_source,
            wall_time_s=wall_s,
            think=self._think,
            num_ctx=num_ctx,
            max_num_ctx=self._max_num_ctx,
            prompt_tokens_estimate=prompt_tokens_estimate,
            done_reason=done_reason,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_tokens,
        )
        self._call_records.append(record)
        return raw_text, record

    def chat_raw(
        self, *, target_id: str, messages: list[dict[str, str]]
    ) -> tuple[str, OllamaCallRecord]:
        """One chat call with caller-built messages (the value-set certifier's own prompt), same pins and settings."""
        return self._chat(target_id=target_id, messages=messages)

    def read_claim(self, prompt: BlindReadPrompt) -> BlindReadResult:
        """Send only `prompt` (material + one claim, never the key) to `POST /api/chat` and parse the verdict."""
        messages = build_reader_read_messages(prompt)
        raw_text, record = self._chat(
            target_id=prompt.claim_id,
            messages=messages,
        )
        verdict: ReaderVerdict = (
            "ambiguous" if record.done_reason == "length" else parse_reader_verdict(raw_text)
        )
        return BlindReadResult(
            claim_id=prompt.claim_id,
            reader=self._model,
            verdict=verdict,
            thinking_words=record.thinking_words,
            completion_tokens=record.eval_tokens,
            seconds=record.wall_time_s,
            raw_text=raw_text,
            prompt_sha256=record.prompt_sha256,
            done_reason=record.done_reason,
            num_ctx=record.num_ctx,
        )

    def unload_model(self) -> OllamaUnloadRecord:
        """Unload the reader model from VRAM after a wave using `keep_alive: 0`."""
        chat_url = f"{self._base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.ollama_tag,
            "messages": [],
            "keep_alive": 0,
            "stream": False,
        }
        t0 = time.perf_counter()
        self._transport("POST", chat_url, payload)
        wall_s = max(0.0, time.perf_counter() - t0)
        rec = OllamaUnloadRecord(
            model=self.ollama_tag,
            digest=self.manifest_digest,
            keep_alive=0,
            wall_time_s=wall_s,
        )
        self._unload_records.append(rec)
        return rec


CALIBRATION_NUM_PREDICT_MARGIN_TOKENS: int = 512
DEFAULT_CALIBRATION_TABLES: int = 3
DEFAULT_CALIBRATION_NUM_PREDICT: int = 4096
DEFAULT_CALIBRATION_NUM_CTX: int = 8192


def run_ollama_calibration_calls(
    *,
    model: ModelIdentity | PinnedModelSpec,
    operation: OllamaOperation,
    call_items: Sequence[tuple[str, list[dict[str, str]]]],
    num_predict: int = DEFAULT_CALIBRATION_NUM_PREDICT,
    num_ctx: int = DEFAULT_CALIBRATION_NUM_CTX,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    think: Any = None,
    http_transport: OllamaHttpTransport | None = None,
) -> tuple[list[dict[str, Any]], OllamaUnloadRecord]:
    """Run calibration calls for `operation` with the model's pinned `think` setting and unload with `keep_alive: 0`.

    `think=None` takes the pin's setting (a thinking model calibrates with thinking on; a direct-answer model such as
    reader_3 with it off). An explicit value must equal the pin's: thinking is never switched off for a model that
    thinks, and never switched on for one that cannot.
    """
    pinned_think = model.think
    if think is None:
        think = pinned_think
    if type(think) is not type(pinned_think) or think != pinned_think:
        if pinned_think is not False and think is False:
            raise refused_error(
                "PILOT_B_CALIBRATE_THINK_OFF_FORBIDDEN",
                f"Calibration requires thinking enabled (pinned think={pinned_think!r}), got {think!r}.",
                "Run calibration with thinking enabled.",
            )
        raise refused_error(
            "PILOT_B_CALIBRATE_THINK_MISMATCH",
            f"Calibration got think={think!r}, but the pin for {model.name if isinstance(model, ModelIdentity) else model.ollama_tag} says think={pinned_think!r}.",
            "Leave think unset to use the pin's setting.",
        )
    if operation not in PILOT_B_OPERATIONS:
        raise usage_error(
            "PILOT_B_INVALID_CALIBRATE_OPERATION",
            f"Unsupported calibration operation {operation!r}; must be one of {list(PILOT_B_OPERATIONS)}.",
            "Pass --operation render_material|word_claim|read_claim.",
        )
    if num_predict < 1:
        raise usage_error(
            "PILOT_B_INVALID_CALIBRATE_NUM_PREDICT",
            f"--num-predict must be >= 1, got {num_predict}.",
            "Pass a positive integer --num-predict (default: 4096).",
        )
    if num_ctx < NUM_CTX_STEP_TOKENS:
        raise usage_error(
            "PILOT_B_INVALID_CALIBRATE_NUM_CTX",
            f"--num-ctx must be >= {NUM_CTX_STEP_TOKENS}, got {num_ctx}.",
            f"Pass --num-ctx >= {NUM_CTX_STEP_TOKENS} (default: 8192).",
        )

    norm_url = require_loopback_ollama_url(base_url)
    resolved_model = _resolve_model_identity(model)
    transport = (
        http_transport if http_transport is not None else default_ollama_http_transport
    )

    verify_local_ollama_model_digest(
        base_url=norm_url,
        ollama_tag=resolved_model.name,
        expected_digest=resolved_model.digest,
        http_transport=transport,
    )

    chat_url = f"{norm_url}/api/chat"
    rows: list[dict[str, Any]] = []
    try:
        for target_id, messages in call_items:
            prompt_sha = _canonical_prompt_sha256(messages)
            prompt_tokens_est = estimate_prompt_tokens(messages)
            payload: dict[str, Any] = {
                "model": resolved_model.name,
                "messages": messages,
                "think": think,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": int(num_predict),
                    "num_ctx": int(num_ctx),
                },
            }
            t0 = time.perf_counter()
            resp = transport("POST", chat_url, payload)
            wall_s = round(max(0.0, time.perf_counter() - t0), 6)

            msg = resp.get("message")
            if not isinstance(msg, dict):
                msg = {}
            content = str(msg.get("content") or "").strip()
            thinking_text = str(msg.get("thinking") or "")
            done_reason = str(resp.get("done_reason") or "stop").strip()
            eval_tokens, thinking_words = _extract_token_counts(
                resp, content, thinking_text
            )
            rows.append(
                {
                    "target_id": target_id,
                    "operation": operation,
                    "model": resolved_model.name,
                    "digest": resolved_model.digest,
                    "prompt_sha256": prompt_sha,
                    "prompt_tokens_estimate": prompt_tokens_est,
                    "think": think,
                    "eval_count": eval_tokens,
                    "eval_tokens": eval_tokens,
                    "thinking_words": thinking_words,
                    "done_reason": done_reason,
                    "wall_time_s": wall_s,
                    "num_ctx": int(num_ctx),
                    "num_predict": int(num_predict),
                }
            )
    finally:
        unload_payload: dict[str, Any] = {
            "model": resolved_model.name,
            "messages": [],
            "keep_alive": 0,
            "stream": False,
        }
        t_u = time.perf_counter()
        transport("POST", chat_url, unload_payload)
        unload_rec = OllamaUnloadRecord(
            model=resolved_model.name,
            digest=resolved_model.digest,
            keep_alive=0,
            wall_time_s=max(0.0, time.perf_counter() - t_u),
        )

    return rows, unload_rec
