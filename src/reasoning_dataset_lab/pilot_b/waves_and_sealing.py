"""Part D: Waves, num_predict configuration, and sealing (DESIGN.md ruling 5a, ruling 8b, §2.1, §7, §11 B.4).

Runs go in single-model waves (render, word, reader_1, reader_2), each model loaded once,
never interleaved per triad. Configures a total num_predict per operation and records its value
and source, plus informational thinking_words. Keys go only to keys.sealed.jsonl, and the freeze
refuses any key-bearing field name or per-item key-derived value anywhere else in the part.
Items are shuffled with a recorded seed so a triad's position cannot hint at its key, and
provenance carries judgment outcomes only as aggregates (§2.1).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import reasoning_dataset_lab.pilot_b.ollama as ollama_mod
from reasoning_dataset_lab.pilot_b import schema_check
from reasoning_dataset_lab.errors import RdlabError, refused_error
from reasoning_dataset_lab.pilot_b.fact_tables import (
    ProvedTriad,
    VerdictKey,
    verify_triad_against_table,
)
from reasoning_dataset_lab.pilot_b.ollama import (
    DEFAULT_ROLE_MAX_NUM_CTX,
    DOCUMENTED_DEFAULT_NUM_PREDICT,
    NUM_CTX_MARGIN_TOKENS,
    NUM_CTX_STEP_TOKENS,
    PROMPT_TOKEN_ESTIMATE_RULE,
    NumPredictConfig,
    NumPredictSource,
    OllamaBlindReaderAdapter,
    OllamaOperation,
    OllamaWriterAdapter,
    coerce_num_predict_config,
)
from reasoning_dataset_lab.pilot_b.rendering import (
    PilotBWriterAdapter,
    WordedTriad,
    check_rendered_material,
    check_worded_claim,
)
from reasoning_dataset_lab.pilot_b.unanimity import (
    DEFAULT_MIN_PASS_RATE_PER_CLASS,
    VERDICT_CLASSES,
    BlindReaderAdapter,
    BlindReadPrompt,
    BlindReadResult,
    ClaimGateEvaluation,
    ModelIdentity,
    TriadGateEvaluation,
    append_to_review_queue,
    build_blind_read_prompt,
    DEFAULT_MAX_RENDER_FAILURE_RATE,
    check_min_pass_rate_per_class,
    render_failure_rate,
    evaluate_triad_unanimity,
    validate_chain_families,
)

WaveName = Literal["render", "word", "reader_1", "reader_2"]
EXPECTED_WAVE_ORDER: tuple[WaveName, ...] = ("render", "word", "reader_1", "reader_2")
# Documented default only (Card gs: total num_predict per operation is configuration, never a constant source of truth)
DEFAULT_NUM_PREDICT: int = DOCUMENTED_DEFAULT_NUM_PREDICT
WAVE_TO_OPERATION: dict[WaveName, OllamaOperation] = {
    "render": "render_material",
    "word": "word_claim",
    "reader_1": "read_claim",
    "reader_2": "read_claim",
}

FORBIDDEN_SEALED_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "key",
        "verdict",
        "label",
        "answer",
        "expected_verdict",
        "ground_truth",
        "gold_label",
        "key_contradicted",
        "target_verdict",
        "shuffle_seed",
        "table_secret",
        "commitment_key",
    }
)

SEALING_STORE_NAME = "sealing.sealed.json"
CALL_METRICS_NAME = "call_metrics.sealed.jsonl"
RENDER_INPUTS_NAME = "render_inputs.sealed.json"
REPORT_NAME = "report.sealed.json"

# Per-call measurements that can vary with an item: they live in the sealed store, and the public part carries
# whole-build distributions only.
PER_ITEM_METRIC_FIELDS: tuple[str, ...] = (
    "prompt_tokens_estimate",
    "num_ctx",
    "eval_tokens",
    "thinking_words",
    "wall_time_s",
    "done_reason",
)


@dataclass(frozen=True)
class WaveCallEntry:
    """Single model invocation recorded during a wave."""

    wave_name: WaveName
    model_name: str
    target_id: str
    thinking_words: int
    completion_tokens: int
    seconds: float
    num_ctx: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "wave_name": self.wave_name,
            "model_name": self.model_name,
            "target_id": self.target_id,
            "thinking_words": self.thinking_words,
            "completion_tokens": self.completion_tokens,
            "seconds": self.seconds,
            "num_ctx": self.num_ctx,
        }


@dataclass(frozen=True)
class WaveMetric:
    """Per-wave aggregate timing and token statistics (ruling 8b, Part E)."""

    wave_name: WaveName
    model_name: str
    model_family: str
    model_digest: str
    calls: int
    num_predict: int
    thinking_words_total: int
    max_thinking_words_observed: int
    completion_tokens_total: int
    seconds_total: float
    adapter: str = ""
    unloaded: bool = False
    max_num_ctx: int = 0
    max_num_ctx_observed: int = 0
    num_predict_source: NumPredictSource = "default"

    def to_dict(self) -> dict[str, object]:
        return {
            "wave_name": self.wave_name,
            "adapter": self.adapter,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "model_digest": self.model_digest,
            "calls": self.calls,
            "num_predict": self.num_predict,
            "num_predict_source": self.num_predict_source,
            "thinking_words_total": self.thinking_words_total,
            "max_thinking_words_observed": self.max_thinking_words_observed,
            "completion_tokens_total": self.completion_tokens_total,
            "eval_tokens_total": self.completion_tokens_total,
            "seconds_total": round(self.seconds_total, 4),
            "wall_time_s_total": round(self.seconds_total, 4),
            "unloaded": self.unloaded,
            "max_num_ctx": self.max_num_ctx,
            "max_num_ctx_observed": self.max_num_ctx_observed,
        }


@dataclass(frozen=True)
class PilotBWaveRunResult:
    """Complete result of executing the 4 single-model waves over a batch of ProvedTriads."""

    writer_model: ModelIdentity
    reader_1_model: ModelIdentity
    reader_2_model: ModelIdentity
    num_predict: int
    wave_metrics: tuple[WaveMetric, ...]
    call_log: tuple[WaveCallEntry, ...]
    kept_triads: tuple[TriadGateEvaluation, ...]
    failed_triads: tuple[TriadGateEvaluation, ...]
    pass_rate_per_class: dict[VerdictKey, float]
    triad_pass_rate: float
    writer_adapter_class: str = "OllamaWriterAdapter"
    reader_1_adapter_class: str = "OllamaBlindReaderAdapter"
    reader_2_adapter_class: str = "OllamaBlindReaderAdapter"
    has_fake_adapter: bool = False
    render_failure_rate: float = 0.0
    max_render_failure_rate: float = DEFAULT_MAX_RENDER_FAILURE_RATE
    writer_adapter: object = None
    reader_1_adapter: object = None
    reader_2_adapter: object = None
    writer_call_records: Mapping[str, dict[str, object]] | None = None
    reader_1_call_records: Mapping[str, dict[str, object]] | None = None
    reader_2_call_records: Mapping[str, dict[str, object]] | None = None
    num_predict_source: NumPredictSource = "default"
    num_predict_config: NumPredictConfig | None = None
    writer_only: bool = False
    triad_metadata: Mapping[str, Mapping[str, Any]] | None = None
    split_counts: Mapping[str, Any] | None = None


class WaveCallTracker:
    """Enforces strict non-interleaved wave execution and records per-operation num_predict and thinking_words."""

    def __init__(
        self,
        num_predict: int | NumPredictConfig | Mapping[str, Any] | None = None,
        *,
        num_predict_source: NumPredictSource | None = None,
    ) -> None:
        self._num_predict_config = coerce_num_predict_config(
            num_predict, source=num_predict_source
        )
        self._num_predict = self._num_predict_config.effective_num_predict
        self._entries: list[WaveCallEntry] = []
        self._seen_waves: list[WaveName] = []

    @property
    def num_predict(self) -> int:
        return self._num_predict

    @property
    def num_predict_config(self) -> NumPredictConfig:
        return self._num_predict_config

    @property
    def entries(self) -> tuple[WaveCallEntry, ...]:
        return tuple(self._entries)

    def record(
        self,
        *,
        wave_name: WaveName,
        model_name: str,
        target_id: str,
        thinking_words: int,
        completion_tokens: int,
        seconds: float,
        num_ctx: int = 0,
    ) -> None:
        if self._seen_waves and self._seen_waves[-1] != wave_name:
            if wave_name in self._seen_waves:
                raise refused_error(
                    "PILOT_B_INTERLEAVED_WAVE_CALL",
                    f"Wave '{wave_name}' was resumed after {', '.join(self._seen_waves)}. "
                    "Calls must run in single-model waves, never interleaved per triad.",
                    "Execute all calls of each wave before starting the next wave.",
                )
            prev_idx = EXPECTED_WAVE_ORDER.index(self._seen_waves[-1])
            curr_idx = EXPECTED_WAVE_ORDER.index(wave_name)
            if curr_idx != prev_idx + 1:
                raise refused_error(
                    "PILOT_B_WAVE_ORDER_VIOLATION",
                    f"Wave '{wave_name}' started out of order after '{self._seen_waves[-1]}'. "
                    f"Expected order: {list(EXPECTED_WAVE_ORDER)}.",
                    "Run waves strictly in order: render -> word -> reader_1 -> reader_2.",
                )
        if not self._seen_waves or self._seen_waves[-1] != wave_name:
            self._seen_waves.append(wave_name)

        self._entries.append(
            WaveCallEntry(
                wave_name=wave_name,
                model_name=model_name,
                target_id=target_id,
                thinking_words=thinking_words,
                completion_tokens=completion_tokens,
                seconds=seconds,
                num_ctx=num_ctx,
            )
        )

    def summarize_wave(
        self,
        wave_name: WaveName,
        model: ModelIdentity,
        *,
        adapter: str = "",
        unloaded: bool = False,
        max_num_ctx: int = 0,
    ) -> WaveMetric:
        wave_calls = [e for e in self._entries if e.wave_name == wave_name]
        resolved_max_ctx = max_num_ctx if max_num_ctx > 0 else model.max_num_ctx
        op = WAVE_TO_OPERATION[wave_name]
        role = (
            "reader_2"
            if wave_name == "reader_2"
            else ("reader_1" if wave_name == "reader_1" else None)
        )
        return WaveMetric(
            wave_name=wave_name,
            model_name=model.name,
            model_family=model.normalized_family,
            model_digest=model.digest,
            calls=len(wave_calls),
            num_predict=self._num_predict_config.num_predict_for(op, role=role),
            num_predict_source=self._num_predict_config.source_for(op, role=role),
            thinking_words_total=sum(e.thinking_words for e in wave_calls),
            max_thinking_words_observed=max(
                (e.thinking_words for e in wave_calls), default=0
            ),
            completion_tokens_total=sum(e.completion_tokens for e in wave_calls),
            seconds_total=round(sum(e.seconds for e in wave_calls), 4),
            adapter=adapter,
            unloaded=unloaded,
            max_num_ctx=resolved_max_ctx,
            max_num_ctx_observed=max((e.num_ctx for e in wave_calls), default=0),
        )


def run_pilot_b_waves(
    *,
    triads: Sequence[ProvedTriad],
    writer_model: ModelIdentity,
    writer_adapter: PilotBWriterAdapter,
    reader_1: BlindReaderAdapter,
    reader_2: BlindReaderAdapter,
    review_queue_path: Path,
    num_predict: int | NumPredictConfig | Mapping[str, Any] | None = None,
    num_predict_source: NumPredictSource | None = None,
    min_pass_rate_per_class: float = DEFAULT_MIN_PASS_RATE_PER_CLASS,
    max_render_failure_rate: float = DEFAULT_MAX_RENDER_FAILURE_RATE,
) -> PilotBWaveRunResult:
    """Execute Pilot B in 4 strict single-model waves: render -> word -> reader_1 -> reader_2."""
    validate_chain_families(writer_model, reader_1.model, reader_2.model)
    for t in triads:
        verify_triad_against_table(t.table, t)

    if num_predict is not None:
        np_config = coerce_num_predict_config(
            num_predict, source=num_predict_source
        )
    elif isinstance(getattr(writer_adapter, "num_predict_config", None), NumPredictConfig):
        np_config = getattr(writer_adapter, "num_predict_config")
    else:
        np_config = coerce_num_predict_config(None, source=num_predict_source)
    effective_num_predict = np_config.effective_num_predict
    tracker = WaveCallTracker(
        num_predict=np_config,
    )

    # Wave 1: render material for all tables (writer model)
    rendered_materials: dict[str, str] = {}
    render_failed_evals: list[TriadGateEvaluation] = []

    def _build_render_failed_eval(
        triad_obj: ProvedTriad,
        material_text: str,
        partial_claims: dict[VerdictKey, str],
        reason: str,
    ) -> TriadGateEvaluation:
        dummy_r1 = BlindReadResult(
            claim_id="",
            reader=reader_1.model,
            verdict="ambiguous",
            thinking_words=0,
            completion_tokens=0,
            seconds=0.0,
        )
        dummy_r2 = BlindReadResult(
            claim_id="",
            reader=reader_2.model,
            verdict="ambiguous",
            thinking_words=0,
            completion_tokens=0,
            seconds=0.0,
        )
        c_evals: dict[VerdictKey, ClaimGateEvaluation] = {}
        for k in VERDICT_CLASSES:
            cid = triad_obj.claims[k].claim_id
            c_evals[k] = ClaimGateEvaluation(
                claim_id=cid,
                key=k,
                read_1=BlindReadResult(
                    claim_id=cid,
                    reader=dummy_r1.reader,
                    verdict="ambiguous",
                    thinking_words=0,
                    completion_tokens=0,
                    seconds=0.0,
                ),
                read_2=BlindReadResult(
                    claim_id=cid,
                    reader=dummy_r2.reader,
                    verdict="ambiguous",
                    thinking_words=0,
                    completion_tokens=0,
                    seconds=0.0,
                ),
                agrees_1=False,
                agrees_2=False,
                passed=False,
                failure_reason=reason,
            )
        return TriadGateEvaluation(
            worded=WordedTriad(
                triad=triad_obj,
                material=material_text,
                worded_claims={
                    k: partial_claims.get(k, "") for k in VERDICT_CLASSES
                },
            ),
            claim_evals=c_evals,
            passed=False,
            failure_reasons=(reason,),
            render_failed=True,
        )

    for t in triads:
        try:
            mat = writer_adapter.render_material(t.table)
            check_rendered_material(t.table, mat)
        except RdlabError as err:
            if not (
                err.code.startswith("PILOT_B_RENDER_")
                or err.code in ("PILOT_B_OLLAMA_LENGTH_STOP", "PILOT_B_NUM_CTX_EXCEEDS_MAX")
            ):
                raise
            last_rec = getattr(writer_adapter, "last_call_record", None)
            raw_out = str(getattr(writer_adapter, "last_raw_output", ""))
            tracker.record(
                wave_name="render",
                model_name=writer_model.name,
                target_id=t.table.table_id,
                thinking_words=(
                    last_rec.thinking_words
                    if last_rec is not None
                    else min(np_config.num_predict_for("render_material"), 32)
                ),
                completion_tokens=(
                    last_rec.eval_tokens
                    if last_rec is not None
                    else max(16, len(raw_out.split()))
                ),
                seconds=last_rec.wall_time_s if last_rec is not None else 0.02,
                num_ctx=last_rec.num_ctx if last_rec is not None else 0,
            )
            if err.code == "PILOT_B_OLLAMA_LENGTH_STOP":
                reason_prefix = "length_stop"
            elif err.code == "PILOT_B_NUM_CTX_EXCEEDS_MAX":
                reason_prefix = "num_ctx_exceeded"
            else:
                reason_prefix = "rendering_check"
            render_failed_evals.append(
                _build_render_failed_eval(
                    t,
                    raw_out,
                    {},
                    f"{reason_prefix} ({err.code}): {err.message}",
                )
            )
            continue

        last_rec = getattr(writer_adapter, "last_call_record", None)
        est_thinking = (
            last_rec.thinking_words
            if last_rec is not None
            else min(np_config.num_predict_for("render_material"), 32)
        )
        est_completion = (
            last_rec.eval_tokens
            if last_rec is not None
            else max(16, len(mat.split()))
        )
        est_seconds = last_rec.wall_time_s if last_rec is not None else 0.02
        est_num_ctx = last_rec.num_ctx if last_rec is not None else 0
        tracker.record(
            wave_name="render",
            model_name=writer_model.name,
            target_id=t.table.table_id,
            thinking_words=est_thinking,
            completion_tokens=est_completion,
            seconds=est_seconds,
            num_ctx=est_num_ctx,
        )
        rendered_materials[t.triad_id] = mat

    render_unloaded = False
    if callable(getattr(writer_adapter, "unload_model", None)):
        writer_adapter.unload_model()  # type: ignore[attr-defined]
        render_unloaded = True

    # Wave 2: word all claims across all triads that passed Wave 1 (writer model)
    worded_triads: list[WordedTriad] = []
    for t in triads:
        if t.triad_id not in rendered_materials:
            continue
        w_claims: dict[VerdictKey, str] = {}
        word_failed_reason: str | None = None
        for key in VERDICT_CLASSES:
            claim = t.claims[key]
            try:
                w_text = writer_adapter.word_claim(t.table, claim)
                check_worded_claim(t.table, claim, w_text)
            except RdlabError as err:
                if not (
                    err.code.startswith("PILOT_B_RENDER_")
                    or err.code in ("PILOT_B_OLLAMA_LENGTH_STOP", "PILOT_B_NUM_CTX_EXCEEDS_MAX")
                ):
                    raise
                last_rec = getattr(writer_adapter, "last_call_record", None)
                raw_out = str(getattr(writer_adapter, "last_raw_output", ""))
                tracker.record(
                    wave_name="word",
                    model_name=writer_model.name,
                    target_id=claim.claim_id,
                    thinking_words=(
                        last_rec.thinking_words
                        if last_rec is not None
                        else min(np_config.num_predict_for("word_claim"), 24)
                    ),
                    completion_tokens=(
                        last_rec.eval_tokens
                        if last_rec is not None
                        else max(8, len(raw_out.split()))
                    ),
                    seconds=last_rec.wall_time_s if last_rec is not None else 0.01,
                    num_ctx=last_rec.num_ctx if last_rec is not None else 0,
                )
                w_claims[key] = raw_out
                if err.code == "PILOT_B_OLLAMA_LENGTH_STOP":
                    reason_prefix = "length_stop"
                elif err.code == "PILOT_B_NUM_CTX_EXCEEDS_MAX":
                    reason_prefix = "num_ctx_exceeded"
                else:
                    reason_prefix = "wording_check"
                word_failed_reason = f"{reason_prefix} ({err.code}): {err.message}"
                break

            last_rec = getattr(writer_adapter, "last_call_record", None)
            est_thinking = (
                last_rec.thinking_words
                if last_rec is not None
                else min(np_config.num_predict_for("word_claim"), 24)
            )
            est_completion = (
                last_rec.eval_tokens
                if last_rec is not None
                else max(8, len(w_text.split()))
            )
            est_seconds = last_rec.wall_time_s if last_rec is not None else 0.01
            est_num_ctx = last_rec.num_ctx if last_rec is not None else 0
            tracker.record(
                wave_name="word",
                model_name=writer_model.name,
                target_id=claim.claim_id,
                thinking_words=est_thinking,
                completion_tokens=est_completion,
                seconds=est_seconds,
                num_ctx=est_num_ctx,
            )
            w_claims[key] = w_text

        if word_failed_reason is not None:
            render_failed_evals.append(
                _build_render_failed_eval(
                    t,
                    rendered_materials[t.triad_id],
                    w_claims,
                    word_failed_reason,
                )
            )
            continue

        worded_triads.append(
            WordedTriad(
                triad=t,
                material=rendered_materials[t.triad_id],
                worded_claims=w_claims,
            )
        )

    word_unloaded = False
    if callable(getattr(writer_adapter, "unload_model", None)):
        writer_adapter.unload_model()  # type: ignore[attr-defined]
        word_unloaded = True

    # Prepare blind-read prompts for all triads (key-free)
    prompts_by_triad: list[dict[VerdictKey, BlindReadPrompt]] = []
    for wt in worded_triads:
        t_prompts: dict[VerdictKey, BlindReadPrompt] = {}
        for key in VERDICT_CLASSES:
            claim = wt.triad.claims[key]
            t_prompts[key] = build_blind_read_prompt(
                triad_id=wt.triad.triad_id,
                claim_id=claim.claim_id,
                closure=wt.triad.table.closure,
                material=wt.material,
                claim_text=wt.worded_claims[key],
            )
        prompts_by_triad.append(t_prompts)

    # Wave 3: reader_1 over all claims
    reads_1_list: list[dict[VerdictKey, BlindReadResult]] = []
    for t_prompts in prompts_by_triad:
        r1_map: dict[VerdictKey, BlindReadResult] = {}
        for key in VERDICT_CLASSES:
            try:
                res1 = reader_1.read_claim(t_prompts[key])
            except RdlabError as err:
                if err.code != "PILOT_B_NUM_CTX_EXCEEDS_MAX":
                    raise
                last_r1 = getattr(reader_1, "last_call_record", None)
                res1 = BlindReadResult(
                    claim_id=t_prompts[key].claim_id,
                    reader=reader_1.model,
                    verdict="ambiguous",
                    thinking_words=0,
                    completion_tokens=0,
                    seconds=0.0,
                    raw_text="",
                    prompt_sha256=last_r1.prompt_sha256 if last_r1 is not None else "",
                    done_reason="max_num_ctx_exceeded",
                    num_ctx=last_r1.num_ctx if last_r1 is not None else 0,
                )
            tracker.record(
                wave_name="reader_1",
                model_name=reader_1.model.name,
                target_id=t_prompts[key].claim_id,
                thinking_words=res1.thinking_words,
                completion_tokens=res1.completion_tokens,
                seconds=res1.seconds,
                num_ctx=res1.num_ctx,
            )
            r1_map[key] = res1
        reads_1_list.append(r1_map)

    r1_unloaded = False
    if callable(getattr(reader_1, "unload_model", None)):
        reader_1.unload_model()  # type: ignore[attr-defined]
        r1_unloaded = True

    # Wave 4: reader_2 over all claims
    reads_2_list: list[dict[VerdictKey, BlindReadResult]] = []
    for t_prompts in prompts_by_triad:
        r2_map: dict[VerdictKey, BlindReadResult] = {}
        for key in VERDICT_CLASSES:
            try:
                res2 = reader_2.read_claim(t_prompts[key])
            except RdlabError as err:
                if err.code != "PILOT_B_NUM_CTX_EXCEEDS_MAX":
                    raise
                last_r2 = getattr(reader_2, "last_call_record", None)
                res2 = BlindReadResult(
                    claim_id=t_prompts[key].claim_id,
                    reader=reader_2.model,
                    verdict="ambiguous",
                    thinking_words=0,
                    completion_tokens=0,
                    seconds=0.0,
                    raw_text="",
                    prompt_sha256=last_r2.prompt_sha256 if last_r2 is not None else "",
                    done_reason="max_num_ctx_exceeded",
                    num_ctx=last_r2.num_ctx if last_r2 is not None else 0,
                )
            tracker.record(
                wave_name="reader_2",
                model_name=reader_2.model.name,
                target_id=t_prompts[key].claim_id,
                thinking_words=res2.thinking_words,
                completion_tokens=res2.completion_tokens,
                seconds=res2.seconds,
                num_ctx=res2.num_ctx,
            )
            r2_map[key] = res2
        reads_2_list.append(r2_map)

    r2_unloaded = False
    if callable(getattr(reader_2, "unload_model", None)):
        reader_2.unload_model()  # type: ignore[attr-defined]
        r2_unloaded = True

    # Evaluate unanimity per triad
    evaluations: list[TriadGateEvaluation] = list(render_failed_evals)
    kept: list[TriadGateEvaluation] = []
    failed: list[TriadGateEvaluation] = list(render_failed_evals)
    for wt, r1_map, r2_map in zip(worded_triads, reads_1_list, reads_2_list):
        ev = evaluate_triad_unanimity(wt, r1_map, r2_map)
        evaluations.append(ev)
        if ev.passed:
            kept.append(ev)
        else:
            failed.append(ev)

    append_to_review_queue(review_queue_path, failed)
    rates = check_min_pass_rate_per_class(
        evaluations,
        min_pass_rate_per_class=min_pass_rate_per_class,
        max_render_failure_rate=max_render_failure_rate,
    )
    triad_rate = round(len(kept) / len(evaluations), 4) if evaluations else 0.0

    w_cls = type(writer_adapter).__name__
    r1_cls = type(reader_1).__name__
    r2_cls = type(reader_2).__name__
    has_fake = not (
        type(writer_adapter) is OllamaWriterAdapter
        and type(reader_1) is OllamaBlindReaderAdapter
        and type(reader_2) is OllamaBlindReaderAdapter
        and writer_adapter.http_transport is ollama_mod.default_ollama_http_transport
        and reader_1.http_transport is ollama_mod.default_ollama_http_transport
        and reader_2.http_transport is ollama_mod.default_ollama_http_transport
    )

    w_call_map: dict[str, dict[str, object]] = {
        rec.target_id: rec.to_dict(include_target_id=False)
        for rec in getattr(writer_adapter, "call_records", ())
    }
    r1_call_map: dict[str, dict[str, object]] = {
        rec.target_id: rec.to_dict(include_target_id=False)
        for rec in getattr(reader_1, "call_records", ())
    }
    r2_call_map: dict[str, dict[str, object]] = {
        rec.target_id: rec.to_dict(include_target_id=False)
        for rec in getattr(reader_2, "call_records", ())
    }

    w_max_ctx = int(getattr(writer_adapter, "max_num_ctx", writer_model.max_num_ctx))
    r1_max_ctx = int(getattr(reader_1, "max_num_ctx", reader_1.model.max_num_ctx))
    r2_max_ctx = int(getattr(reader_2, "max_num_ctx", reader_2.model.max_num_ctx))

    wave_metrics = (
        tracker.summarize_wave(
            "render",
            writer_model,
            adapter=w_cls,
            unloaded=render_unloaded,
            max_num_ctx=w_max_ctx,
        ),
        tracker.summarize_wave(
            "word",
            writer_model,
            adapter=w_cls,
            unloaded=word_unloaded,
            max_num_ctx=w_max_ctx,
        ),
        tracker.summarize_wave(
            "reader_1",
            reader_1.model,
            adapter=r1_cls,
            unloaded=r1_unloaded,
            max_num_ctx=r1_max_ctx,
        ),
        tracker.summarize_wave(
            "reader_2",
            reader_2.model,
            adapter=r2_cls,
            unloaded=r2_unloaded,
            max_num_ctx=r2_max_ctx,
        ),
    )

    return PilotBWaveRunResult(
        writer_model=writer_model,
        reader_1_model=reader_1.model,
        reader_2_model=reader_2.model,
        num_predict=effective_num_predict,
        wave_metrics=wave_metrics,
        call_log=tracker.entries,
        kept_triads=tuple(kept),
        failed_triads=tuple(failed),
        pass_rate_per_class=rates,
        triad_pass_rate=triad_rate,
        render_failure_rate=round(render_failure_rate(evaluations), 4),
        max_render_failure_rate=max_render_failure_rate,
        writer_adapter_class=w_cls,
        reader_1_adapter_class=r1_cls,
        reader_2_adapter_class=r2_cls,
        has_fake_adapter=has_fake,
        writer_adapter=writer_adapter,
        reader_1_adapter=reader_1,
        reader_2_adapter=reader_2,
        writer_call_records=w_call_map,
        reader_1_call_records=r1_call_map,
        reader_2_call_records=r2_call_map,
        num_predict_source=np_config.source,
        num_predict_config=np_config,
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── the one renderer ────────────────────────────────────────────────────────────────────────────────────────────
#
# Every public file of a sealed part (and the build report) is the output of the functions below, and verify
# re-renders each of them from the sealed store and requires a byte-for-byte match. The rules are properties of
# this renderer: sorted keys, fixed separators, one escaping policy (ASCII, \uXXXX in lower case), "\n" only, no
# trailing whitespace, no -0 and no NaN, and floats from one formatter (Python's shortest repr).


def _normalise(obj: Any) -> Any:
    """The JSON form of a value: plain dicts and lists, ``0.0`` for ``-0.0``, and no NaN or infinity."""

    if isinstance(obj, Mapping):
        return {str(k): _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise refused_error(
                "PILOT_B_SEALED_RENDER_MISMATCH", "A non-finite number cannot be rendered.", "Remove it from the part."
            )
        return 0.0 if obj == 0 else obj
    return obj


def _canonical_json_line(obj: object) -> str:
    return json.dumps(_normalise(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _render_json_doc(obj: object) -> bytes:
    return (json.dumps(_normalise(obj), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode(
        "ascii"
    )


def _render_lines(lines: Sequence[str]) -> bytes:
    return ("\n".join(lines) + ("\n" if lines else "")).encode("ascii")


MIN_SHUFFLE_SEED_BITS = 64
MIN_TABLE_SECRET_BITS = 64


def _triad_handle(key: bytes, triad_id: str) -> str:
    """The public name of a triad: an opaque keyed handle, so the public part carries no generation seed."""

    return "triad-" + _commit(key, "triad", triad_id)[:16]


def _commit(key: bytes, kind: str, value: str) -> str:
    """A keyed commitment to a hash, so the public provenance carries no plain digest of per-item material."""

    return hmac.new(key, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()


_MODEL_FIELDS: tuple[str, ...] = (
    "name",
    "normalized_family",
    "digest",
    "licence_id",
    "licence_read_date",
    "bundled_licence_id",
    "bundled_licence_read_date",
    "licence_layer_digest",
    "source_match",
    "source_match_reason",
)


def _render_inputs(
    *,
    part_id: str,
    map_version: int,
    map_sha256: str,
    wave_result: PilotBWaveRunResult,
    licence_registry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Everything the public files of a part are rendered from. It is kept in the sealed store."""

    w_calls = dict(wave_result.writer_call_records or {})
    r1_calls = dict(wave_result.reader_1_call_records or {})
    r2_calls = dict(wave_result.reader_2_call_records or {})
    writer_only = bool(wave_result.writer_only)
    triad_meta_map = dict(wave_result.triad_metadata or {})

    default_read_dates: dict[str, str] = {}
    models_for_dates = (
        (wave_result.writer_model,)
        if writer_only
        else (wave_result.writer_model, wave_result.reader_1_model, wave_result.reader_2_model)
    )
    for m in models_for_dates:
        if m.licence_id:
            default_read_dates.setdefault(m.licence_id, m.licence_read_date)
        if m.bundled_licence_id:
            default_read_dates.setdefault(m.bundled_licence_id, m.bundled_licence_read_date)
    reg_meta: dict[str, Any] = (
        dict(licence_registry)
        if licence_registry is not None
        else {"path": "", "sha256": "0" * 64, "read_dates": default_read_dates}
    )

    kept_targets: set[str] = set()
    for ev in wave_result.kept_triads:
        kept_targets.add(ev.worded.triad.table.table_id)
        kept_targets.update(claim.claim_id for claim in ev.worded.triad.claims.values())

    entries: list[dict[str, Any]] = []
    for ev in wave_result.kept_triads:
        wt = ev.worded
        table = wt.triad.table
        t_meta = dict(triad_meta_map.get(wt.triad.triad_id, {}))
        for key in VERDICT_CLASSES:
            claim = wt.triad.claims[key]
            claim_ev = ev.claim_evals[key]
            w_recs: list[dict[str, object]] = []
            if table.table_id in w_calls:
                w_recs.append(dict(w_calls[table.table_id]))
            if claim.claim_id in w_calls:
                w_recs.append(dict(w_calls[claim.claim_id]))
            r_recs: list[dict[str, object]] = []
            if not writer_only:
                if claim.claim_id in r1_calls:
                    r_recs.append(dict(r1_calls[claim.claim_id]))
                if claim.claim_id in r2_calls:
                    r_recs.append(dict(r2_calls[claim.claim_id]))
            entry_dict: dict[str, Any] = {
                "triad_id": wt.triad.triad_id,
                "table_id": table.table_id,
                "domain": table.domain,
                "closure": table.closure,
                "material": wt.material,
                "claim_text": wt.worded_claims[key],
                "key": key,
                "proof": claim.proof.to_dict(),
                "read_1": claim_ev.read_1.to_dict(),
                "read_2": claim_ev.read_2.to_dict(),
                "writer_call_records": w_recs,
                "reader_call_records": r_recs,
            }
            if writer_only:
                default_t_line = (
                    table.get_fact(claim.fact_id).line
                    if claim.fact_id is not None
                    else next(f.line for f in table.facts if f.entity == claim.entity)
                )
                entry_dict["value_set_id"] = str(t_meta.get("value_set_id", ""))
                entry_dict["value_set_sha256"] = str(t_meta.get("value_set_sha256", ""))
                entry_dict["split"] = str(t_meta.get("split", "test"))
                entry_dict["target_line"] = int(t_meta.get(f"target_line_{key}", default_t_line))
            entries.append(entry_dict)

    def role(adapter_class: str, model: Any, adapter: Any) -> dict[str, Any]:
        return {
            "adapter_class": adapter_class,
            "model": {name: getattr(model, name) for name in _MODEL_FIELDS},
            "max_num_ctx": int(getattr(adapter, "max_num_ctx", model.max_num_ctx)),
        }

    inputs: dict[str, Any] = {
        "part_id": part_id,
        "map_version": map_version,
        "map_sha256": map_sha256,
        "registry": reg_meta,
        "writer": role(wave_result.writer_adapter_class, wave_result.writer_model, wave_result.writer_adapter),
        "readers": (
            []
            if writer_only
            else [
                role(wave_result.reader_1_adapter_class, wave_result.reader_1_model, wave_result.reader_1_adapter),
                role(wave_result.reader_2_adapter_class, wave_result.reader_2_model, wave_result.reader_2_adapter),
            ]
        ),
        "num_predict": wave_result.num_predict,
        "num_predict_source": wave_result.num_predict_source,
        "num_predict_config": (
            wave_result.num_predict_config.to_dict()
            if wave_result.num_predict_config is not None
            else {
                "default_num_predict": wave_result.num_predict,
                "source": wave_result.num_predict_source,
                "by_operation": {},
                "sources_by_operation": {},
            }
        ),
        "waves": [w.to_dict() for w in wave_result.wave_metrics],
        "pass_rate_per_class": dict(wave_result.pass_rate_per_class),
        "triad_pass_rate": wave_result.triad_pass_rate,
        "triads_kept": len(wave_result.kept_triads),
        "triads_failed": len(wave_result.failed_triads),
        "call_distributions": _call_distributions(
            {"writer": w_calls}
            if writer_only
            else {"writer": w_calls, "reader_1": r1_calls, "reader_2": r2_calls},
            kept_targets,
        ),
        "entries": entries,
    }
    if writer_only and wave_result.split_counts is not None:
        inputs["split_counts"] = dict(wave_result.split_counts)
    inputs = _normalise(inputs)
    _check_render_inputs(inputs)
    return inputs


# ── the inputs the renderer reads are checked too ────────────────────────────────────────────────────────────
#
# The re-render in verify is only as honest as the inputs it renders from. These checks hold the inputs to the shape
# and the value domains of an honest build, at write time and again at verify time, so a poisoned wave result or a
# rewritten sealed store cannot make the honest render itself carry a signal.

_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})?$")  # a read date, or empty when none was recorded
_OPERATIONS = ("read_claim_reader_1", "read_claim_reader_2", "render_material", "word_claim")
_INPUT_KEYS = frozenset(
    {
        "part_id",
        "map_version",
        "map_sha256",
        "registry",
        "writer",
        "readers",
        "num_predict",
        "num_predict_source",
        "num_predict_config",
        "waves",
        "pass_rate_per_class",
        "triad_pass_rate",
        "triads_kept",
        "triads_failed",
        "call_distributions",
        "entries",
    }
)
_ENTRY_KEYS = frozenset(
    {
        "triad_id",
        "table_id",
        "domain",
        "closure",
        "material",
        "claim_text",
        "key",
        "proof",
        "read_1",
        "read_2",
        "writer_call_records",
        "reader_call_records",
    }
)
_WAVE_ORDER = ("render", "word", "reader_1", "reader_2")
_CALL_OPERATIONS = {"writer": {"render_material", "word_claim"}, "reader_1": {"read_claim"}, "reader_2": {"read_claim"}}
_DIST_KEYS = {"n", "min", "median", "p95", "max"}


def _invalid_inputs(detail: str) -> Any:
    return refused_error(
        "PILOT_B_SEALED_INPUT_INVALID",
        f"The render inputs are not an honest build's: {detail}",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_rate(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


def _check_render_inputs(inputs: Mapping[str, Any]) -> None:
    """Hold the render inputs to an honest build's shape and value domains."""

    from reasoning_dataset_lab.pilot_b.fact_tables import (
        DOMAIN_CATALOGUE,
        PILOT_B_PROVER_NAME,
        PILOT_B_PROVER_VERSION,
    )
    from reasoning_dataset_lab.pilot_b.ollama import _canonical_prompt_sha256, build_reader_read_messages
    from reasoning_dataset_lab.pilot_b.unanimity import build_blind_read_prompt

    if not isinstance(inputs, Mapping):
        raise _invalid_inputs("the top-level fields are not the expected ones")
    is_bakeoff = (
        isinstance(inputs.get("part_id"), str)
        and str(inputs["part_id"]).startswith("bakeoff")
        and inputs.get("readers") == []
    )
    expected_top_keys = _INPUT_KEYS | ({"split_counts"} if is_bakeoff and "split_counts" in inputs else set())
    if set(inputs) != expected_top_keys:
        raise _invalid_inputs("the top-level fields are not the expected ones")
    if not isinstance(inputs["part_id"], str) or not inputs["part_id"]:
        raise _invalid_inputs("part_id")
    if not _is_int(inputs["map_version"]) or inputs["map_version"] < 1 or not isinstance(inputs["map_sha256"], str):
        raise _invalid_inputs("the map version or digest")
    registry = inputs["registry"]
    if not isinstance(registry, Mapping) or set(registry) != {"path", "read_dates", "sha256"}:
        raise _invalid_inputs("the licence registry fields")
    dates = registry["read_dates"]
    if not isinstance(dates, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) and _DATE.match(v) for k, v in dates.items()
    ):
        raise _invalid_inputs("the licence read dates are not dates")
    if not isinstance(inputs.get("readers"), list):
        raise _invalid_inputs("readers is not a list")
    licence_ids = {
        role["model"][field]
        for role in (inputs["writer"], *inputs["readers"])
        if isinstance(role, Mapping) and isinstance(role.get("model"), Mapping)
        for field in ("licence_id", "bundled_licence_id")
        if role["model"].get(field)
    }
    if not set(dates) <= licence_ids:
        raise _invalid_inputs("the licence read dates name something other than the models' licences")
    for role in (inputs["writer"], *inputs["readers"]):
        if not isinstance(role, Mapping) or set(role) != {"adapter_class", "model", "max_num_ctx"}:
            raise _invalid_inputs("a model role's fields")
        if set(role["model"]) != set(_MODEL_FIELDS) or not _is_int(role["max_num_ctx"]) or role["max_num_ctx"] < 1:
            raise _invalid_inputs("a model's fields")
        if not all(v is None or isinstance(v, str) for v in role["model"].values()):
            raise _invalid_inputs("a model field is not text")
    if not is_bakeoff and len(inputs["readers"]) != 2:
        raise _invalid_inputs("there are not exactly two readers")
    if not _is_int(inputs["num_predict"]) or inputs["num_predict"] < 1 or not isinstance(inputs["num_predict_source"], str):
        raise _invalid_inputs("num_predict")
    config = inputs["num_predict_config"]
    by_op = config.get("by_operation") if isinstance(config, Mapping) else None
    if not isinstance(by_op, Mapping) or not set(by_op) <= set(_OPERATIONS):
        raise _invalid_inputs("num_predict_config operations")
    waves = inputs["waves"]
    expected_waves = ("render", "word") if is_bakeoff else _WAVE_ORDER
    if not isinstance(waves, list) or [w.get("wave_name") for w in waves if isinstance(w, Mapping)] != list(expected_waves):
        raise _invalid_inputs("the waves are not the expected waves in order")
    rates = inputs["pass_rate_per_class"]
    if not isinstance(rates, Mapping) or set(rates) != set(VERDICT_CLASSES) or not all(_is_rate(v) for v in rates.values()):
        raise _invalid_inputs("the pass rates")
    if not _is_rate(inputs["triad_pass_rate"]):
        raise _invalid_inputs("the triad pass rate")
    entries = inputs["entries"]
    if not isinstance(entries, list) or len(entries) % 3 != 0:
        raise _invalid_inputs("the entries are not whole triads")
    if inputs["triads_kept"] != len(entries) // 3 or not _is_int(inputs["triads_failed"]) or inputs["triads_failed"] < 0:
        raise _invalid_inputs("the triad counts")
    distributions = inputs["call_distributions"]
    expected_dist_roles = {"writer"} if is_bakeoff else set(_CALL_OPERATIONS)
    if not isinstance(distributions, Mapping) or set(distributions) != expected_dist_roles:
        raise _invalid_inputs("the call distributions' roles")
    for role_name, per_op in distributions.items():
        if not isinstance(per_op, Mapping) or not set(per_op) <= _CALL_OPERATIONS[role_name]:
            raise _invalid_inputs("the call distributions' operations")
        for metrics in per_op.values():
            if not isinstance(metrics, Mapping) or not set(metrics) <= set(PER_ITEM_METRIC_FIELDS):
                raise _invalid_inputs("the call distributions' metrics")
            if not all(isinstance(d, Mapping) and set(d) <= _DIST_KEYS for d in metrics.values()):
                raise _invalid_inputs("a call distribution")

    catalogue = {str(d["domain"]) for d in DOMAIN_CATALOGUE}
    seen_triads: set[str] = set()
    seen_tables: set[str] = set()
    seen_traces: set[str] = set()
    render_prompts: set[str] = set()
    word_prompts: set[str] = set()
    constant_calls: dict[tuple[str, int], tuple[Any, ...]] = {}
    prover_version: set[tuple[Any, Any]] = set()
    hex64 = re.compile(r"[0-9a-f]{64}")
    expected_entry_keys = (
        _ENTRY_KEYS | {"value_set_id", "value_set_sha256", "split", "target_line"}
        if is_bakeoff
        else _ENTRY_KEYS
    )
    for start in range(0, len(entries), 3):
        group = entries[start : start + 3]
        if not all(isinstance(e, Mapping) and set(e) == expected_entry_keys for e in group):
            raise _invalid_inputs("an entry's fields")
        if [e["key"] for e in group] != list(VERDICT_CLASSES):
            raise _invalid_inputs("a triad does not hold one entry per class in class order")
        common = {(e["triad_id"], e["table_id"], e["domain"], e["closure"], e["material"]) for e in group}
        if len(common) != 1:
            raise _invalid_inputs("the entries of a triad differ in their shared fields")
        triad_id, table_id, domain, closure, material = next(iter(common))
        if domain not in catalogue:
            raise _invalid_inputs("a triad's domain is not in the domain catalogue")
        if table_id in seen_tables:
            raise _invalid_inputs("two triads share a table")
        seen_tables.add(table_id)
        if len({e["claim_text"] for e in group}) != 3:
            raise _invalid_inputs("the three claims of a triad are not distinct")
        if "\n".join(" ".join(line.split()) for line in material.splitlines()) != material:
            raise _invalid_inputs("a triad's material has irregular whitespace")
        triad_render = {e["writer_call_records"][0].get("prompt_sha256") for e in group}
        if len(triad_render) != 1:
            raise _invalid_inputs("the table render prompt is not shared by the triad")
        render_prompt = next(iter(triad_render))
        if not isinstance(render_prompt, str) or not hex64.fullmatch(render_prompt) or render_prompt in render_prompts:
            raise _invalid_inputs("the table render prompt is not a digest that is unique to the triad")
        render_prompts.add(render_prompt)
        if triad_id in seen_triads:
            raise _invalid_inputs("a triad appears twice")
        seen_triads.add(triad_id)
        if closure not in ("open", "closed") or not all(isinstance(x, str) and x for x in (triad_id, domain, material)):
            raise _invalid_inputs("a triad's closure, domain or material")
        for entry in group:
            claim = entry["claim_text"]
            if not isinstance(claim, str) or not claim or " ".join(claim.split()) != claim:
                raise _invalid_inputs("a claim is empty or has irregular whitespace")
            proof = entry["proof"]
            if not isinstance(proof, Mapping) or proof.get("key") != entry["key"]:
                raise _invalid_inputs("a proof does not belong to its entry's class")
            if not isinstance(proof.get("trace_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", proof["trace_sha256"]):
                raise _invalid_inputs("a proof trace digest")
            prover_version.add((proof.get("prover"), proof.get("version")))
            if proof["trace_sha256"] in seen_traces:
                raise _invalid_inputs("two proofs share a trace digest")
            seen_traces.add(proof["trace_sha256"])
            word_prompt = entry["writer_call_records"][1].get("prompt_sha256")
            if not isinstance(word_prompt, str) or not hex64.fullmatch(word_prompt) or word_prompt in word_prompts:
                raise _invalid_inputs("a claim-wording prompt is not a digest that is unique to the entry")
            word_prompts.add(word_prompt)
            for slot, record in enumerate((*entry["writer_call_records"], *entry["reader_call_records"])):
                fields = tuple(
                    record.get(f) for f in ("model", "digest", "max_num_ctx", "num_predict", "num_predict_source", "think")
                )
                if constant_calls.setdefault((("w", "r")[slot >= 2], slot), fields) != fields:
                    raise _invalid_inputs("a call's model, digest or limits vary between entries")
            if [r.get("operation") for r in entry["writer_call_records"]] != ["render_material", "word_claim"]:
                raise _invalid_inputs("a claim's writer calls are not the table render and the claim wording")
            readers_calls = entry["reader_call_records"]
            if is_bakeoff:
                if readers_calls != []:
                    raise _invalid_inputs("a writer-only bakeoff entry must have no reader call records")
                if entry.get("split") not in ("dev", "test"):
                    raise _invalid_inputs("a bakeoff entry split must be 'dev' or 'test'")
                if not isinstance(entry.get("value_set_id"), str) or "/" not in entry["value_set_id"]:
                    raise _invalid_inputs("a bakeoff entry value_set_id is invalid")
                if not isinstance(entry.get("value_set_sha256"), str) or not hex64.fullmatch(entry["value_set_sha256"]):
                    raise _invalid_inputs("a bakeoff entry value_set_sha256 is invalid")
                if not _is_int(entry.get("target_line")) or not (1 <= entry["target_line"] <= 5):
                    raise _invalid_inputs("a bakeoff entry target_line must be 1..5")
            else:
                if [r.get("operation") for r in readers_calls] != ["read_claim", "read_claim"]:
                    raise _invalid_inputs("a claim's reader calls are not two reads")
                expected = _canonical_prompt_sha256(
                    build_reader_read_messages(
                        build_blind_read_prompt(
                            triad_id=triad_id, claim_id="", closure=closure, material=material, claim_text=claim
                        )
                    )
                )
                if any(r.get("prompt_sha256") != expected for r in readers_calls):
                    raise _invalid_inputs("a reader's prompt is not the blind prompt for the claim and material shown")
    if prover_version and prover_version != {(PILOT_B_PROVER_NAME, PILOT_B_PROVER_VERSION)}:
        raise _invalid_inputs("the prover or its version is not the one this code pins")
    if not is_bakeoff:
        for slot, role in ((2, inputs["readers"][0]), (3, inputs["readers"][1])):
            fields = constant_calls.get(("r", slot))
            if fields is not None and (
                fields[0] != role["model"]["name"]
                or fields[1] != role["model"]["digest"]
                or fields[2] != role["max_num_ctx"]
            ):
                raise _invalid_inputs("a reader's calls do not match the reader recorded")
    writer_fields = [constant_calls.get(("w", slot)) for slot in (0, 1)]
    for fields in writer_fields:
        if fields is not None and (
            fields[0] != inputs["writer"]["model"]["name"]
            or fields[1] != inputs["writer"]["model"]["digest"]
            or fields[2] != inputs["writer"]["max_num_ctx"]
        ):
            raise _invalid_inputs("the writer's calls do not match the writer recorded")


def _check_sealed_rows(inputs: Mapping[str, Any], keys_rows: Sequence[Mapping[str, Any]], metrics_rows: Sequence[Mapping[str, Any]]) -> None:
    """The sealed keys and call metrics line up with the inputs: ids in order, one key per class per triad."""

    part_id = inputs["part_id"]
    ids = [f"{part_id}-item-{i:03d}" for i in range(1, len(inputs["entries"]) + 1)]
    if [r.get("id") for r in keys_rows] != ids or [r.get("id") for r in metrics_rows] != ids:
        raise _invalid_inputs("the sealed rows are not the part's items in order")
    counts = {k: sum(1 for r in keys_rows if r.get("key") == k) for k in VERDICT_CLASSES}
    if set(counts.values()) != {inputs["triads_kept"]} and inputs["entries"]:
        raise _invalid_inputs("the sealed keys are not one per class per triad")


def _render_part(
    inputs: Mapping[str, Any],
    shuffle_seed: int,
    commitment_key: bytes,
    *,
    frozen: bool = False,
    test_only: bool | None = None,
) -> dict[str, bytes]:
    """Render every file of a part from its inputs, the seed and the key: the public files, the sealed keys and the
    sealed call metrics. The same call serves the writer, freeze and verify."""

    part_id = inputs["part_id"]
    entries = inputs["entries"]
    registry = inputs["registry"]
    writer = inputs["writer"]
    readers = inputs["readers"]
    is_bakeoff = len(readers) == 0 and str(part_id).startswith("bakeoff")

    # Shuffle items with the seed so triad position cannot hint at the key (§2.1)
    order = list(range(len(entries)))
    rng = random.Random(shuffle_seed)
    rng.shuffle(order)
    if is_bakeoff and len(order) >= 6:
        for _ in range(16):
            try:
                check_items_not_in_key_order([entries[i] for i in order])
                break
            except RdlabError:
                rng.shuffle(order)

    items_lines: list[str] = []
    prov_lines: list[str] = []
    key_lines: list[str] = []
    metric_lines: list[str] = []
    for out_idx, orig_idx in enumerate(order, start=1):
        entry = entries[orig_idx]
        item_id = f"{part_id}-item-{out_idx:03d}"
        user_turn = f"Material ({entry['closure']}):\n{entry['material']}\n\nClaim: {entry['claim_text']}"
        msg_obj = [{"role": "user", "content": user_turn}]
        msg_sha = _sha256_bytes(_canonical_json_line(msg_obj).encode("ascii"))
        handle = _triad_handle(commitment_key, entry["triad_id"])

        item_obj: dict[str, Any] = {
            "id": item_id,
            "triad_id": handle,
            "trait": "T10",
            "kind": "D",
            "domain": entry["domain"],
            "closure": entry["closure"],
            "proof_basis": "proved",
            "key_basis": "construction",
            "material": entry["material"],
            "claim": entry["claim_text"],
            "messages": msg_obj,
        }
        if "split" in entry:
            item_obj["split"] = entry["split"]
        items_lines.append(_canonical_json_line(item_obj))
        m = writer["model"]
        prov_obj: dict[str, Any] = {
            "id": item_id,
            "map": {
                "version": inputs["map_version"],
                "sha256": inputs["map_sha256"],
                "cell": f"T10/D/{entry['domain']}",
            },
            "licence_registry_path": str(registry.get("path", "")),
            "licence_registry_sha256": str(registry.get("sha256", "")),
            "licence_registry": registry,
            "proof_basis": "proved",
            "key_basis": "construction",
            "proof": {
                "prover": entry["proof"]["prover"],
                "version": entry["proof"]["version"],
                "trace_commitment": _commit(commitment_key, "trace", entry["proof"]["trace_sha256"]),
            },
            "writer": {
                "adapter": writer["adapter_class"],
                "model": m["name"],
                "family": m["normalized_family"],
                "digest": m["digest"],
                "licence_id": m["licence_id"],
                "licence_read_date": m["licence_read_date"],
                "bundled_licence_id": m["bundled_licence_id"],
                "bundled_licence_read_date": m["bundled_licence_read_date"],
                "licence_layer_digest": m["licence_layer_digest"],
                "source_match": m["source_match"],
                "source_match_reason": m["source_match_reason"],
                "messages_sha256": msg_sha,
                "num_predict": inputs["num_predict"],
                "num_predict_source": inputs["num_predict_source"],
                "num_predict_config": inputs["num_predict_config"],
                "max_num_ctx": writer["max_num_ctx"],
                "call_records": [
                    _public_call_record(commitment_key, rec, writer=True)
                    for rec in entry["writer_call_records"]
                ],
            },
            "checks": (
                [{"name": "pilot_b.rendering_check", "version": "0.3.0", "result": "pass"}]
                if is_bakeoff
                else [
                    {"name": "pilot_b.rendering_check", "version": "0.3.0", "result": "pass"},
                    {"name": "pilot_b.unanimity_gate", "version": "0.3.0", "result": "pass"},
                ]
            ),
            "judgments": (
                []
                if is_bakeoff
                else [
                    {
                        "kind": "blind_read_unanimity_aggregate",
                        "reader_adapters": [r["adapter_class"] for r in readers],
                        "readers": [f"model:{r['model']['name']}@{r['model']['digest']}" for r in readers],
                        "families": [r["model"]["normalized_family"] for r in readers],
                        "reader_licence_ids": [r["model"]["licence_id"] for r in readers],
                        "reader_licence_read_dates": [r["model"]["licence_read_date"] for r in readers],
                        "reader_bundled_licence_ids": [r["model"]["bundled_licence_id"] for r in readers],
                        "reader_bundled_licence_read_dates": [
                            r["model"]["bundled_licence_read_date"] for r in readers
                        ],
                        "reader_licence_layer_digests": [r["model"]["licence_layer_digest"] for r in readers],
                        "reader_source_matches": [r["model"]["source_match"] for r in readers],
                        "reader_source_match_reasons": [r["model"]["source_match_reason"] for r in readers],
                        "reader_max_num_ctx": [r["max_num_ctx"] for r in readers],
                        "call_records": [
                            _public_call_record(commitment_key, rec, writer=False)
                            for rec in entry["reader_call_records"]
                        ],
                        "agrees_all_readers": True,
                        "ambiguous_count": 0,
                    }
                ]
            ),
        }
        if "split" in entry:
            prov_obj["split"] = entry["split"]
        prov_lines.append(_canonical_json_line(prov_obj))
        if is_bakeoff:
            key_lines.append(
                _canonical_json_line(
                    {
                        "id": item_id,
                        "triad_id": entry["triad_id"],
                        "triad_handle": handle,
                        "key": entry["key"],
                        "class": entry["key"],
                        "key_source": "construction",
                        "value_set_id": entry["value_set_id"],
                        "value_set_sha256": entry["value_set_sha256"],
                        "closure": entry["closure"],
                        "target_line": entry["target_line"],
                        "split": entry["split"],
                        "proof": entry["proof"],
                        "provenance": prov_obj,
                        "blind_reads": [],
                    }
                )
            )
        else:
            key_lines.append(
                _canonical_json_line(
                    {
                        "id": item_id,
                        "triad_id": entry["triad_id"],
                        "triad_handle": handle,
                        "key": entry["key"],
                        "proof": entry["proof"],
                        "blind_reads": [entry["read_1"], entry["read_2"]],
                    }
                )
            )
        metric_lines.append(
            _canonical_json_line(
                {
                    "id": item_id,
                    "writer": _sealed_call_metrics(entry["writer_call_records"]),
                    "readers": _sealed_call_metrics(entry["reader_call_records"]),
                }
            )
        )

    items_bytes = _render_lines(items_lines)
    prov_bytes = _render_lines(prov_lines)
    keys_bytes = _render_lines(key_lines)
    is_test_only = (
        shuffle_seed.bit_length() < MIN_SHUFFLE_SEED_BITS
        if test_only is None
        else (bool(test_only) or shuffle_seed.bit_length() < MIN_SHUFFLE_SEED_BITS)
    )
    meta: dict[str, Any] = {
        "part_id": part_id,
        "split": "sealed",
        "frozen": False,
        "map_version": inputs["map_version"],
        "map_sha256": inputs["map_sha256"],
        "num_predict": inputs["num_predict"],
        "num_predict_source": inputs["num_predict_source"],
        "num_predict_config": inputs["num_predict_config"],
        "items_count": len(items_lines),
        "triads_kept": inputs["triads_kept"],
        "triads_failed": inputs["triads_failed"],
        "keys_sealed_sha256": _sha256_bytes(keys_bytes),
        "call_distributions": inputs["call_distributions"],
        "judgment_aggregates": {
            "pass_rate_per_class": inputs["pass_rate_per_class"],
            "triad_pass_rate": inputs["triad_pass_rate"],
        },
        "waves": inputs["waves"],
    }
    if "split_counts" in inputs:
        meta["split_counts"] = inputs["split_counts"]
    if is_test_only:
        meta["test_only"] = True
    if frozen and not is_test_only:
        # the seal covers the rendered items, provenance and the part's own content
        meta["frozen"] = True
        meta["frozen_mac"] = _part_mac(items_bytes, prov_bytes, meta, commitment_key)
    part_bytes = _render_json_doc(meta)
    checksums = [
        f"{_sha256_bytes(data)}  {name}"
        for name, data in sorted({"items.jsonl": items_bytes, "part.json": part_bytes, "provenance.jsonl": prov_bytes}.items())
    ]
    return {
        "items.jsonl": items_bytes,
        "provenance.jsonl": prov_bytes,
        "part.json": part_bytes,
        "checksums.txt": _render_lines(checksums),
        "keys.sealed.jsonl": keys_bytes,
        CALL_METRICS_NAME: _render_lines(metric_lines),
    }


def _part_mac(items_bytes: bytes, prov_bytes: bytes, meta: Mapping[str, Any], commitment_key: bytes) -> str:
    """The seal: an HMAC under the sealed commitment key over the rendered items, provenance and part content
    (without the seal and the frozen mark). Nothing a public file says about itself stands in for it."""

    content = {k: v for k, v in meta.items() if k not in ("frozen", "frozen_mac")}
    body = _canonical_json_line(
        {"items": _sha256_bytes(items_bytes), "provenance": _sha256_bytes(prov_bytes), "part": content}
    )
    return _commit(commitment_key, "frozen", body)


def write_sealed_part(
    *,
    part_dir: Path,
    keys_sealed_path: Path,
    part_id: str,
    map_version: int,
    map_sha256: str,
    wave_result: PilotBWaveRunResult,
    shuffle_seed: int | None = None,
    table_secret: int | None = None,
    _generated_table_secret: int | None = None,
    licence_registry: Mapping[str, Any] | None = None,
    allow_weak_test_seed: bool = False,
) -> dict[str, Any]:
    """Write a sealed Pilot B part directory with shuffled items, aggregate-only judgment provenance, and external keys.sealed.jsonl.

    The shuffle seed and table secret are drawn per build from ``secrets`` unless given (a reproducible test build),
    and they are kept only in the sealed store, in ``sealing.sealed.json`` beside the keys. They are never written to
    a public file of the part. The sealed store also holds every input the public files are rendered from
    (``render_inputs.sealed.json``): ``verify_sealed_part`` re-renders them and requires a byte-for-byte match.
    """
    w = wave_result.writer_adapter
    r1 = wave_result.reader_1_adapter
    r2 = wave_result.reader_2_adapter
    adapters_valid = (
        type(w) is OllamaWriterAdapter
        and w.http_transport is ollama_mod.default_ollama_http_transport
        and (
            bool(wave_result.writer_only)
            or (
                type(r1) is OllamaBlindReaderAdapter
                and type(r2) is OllamaBlindReaderAdapter
                and r1.http_transport is ollama_mod.default_ollama_http_transport
                and r2.http_transport is ollama_mod.default_ollama_http_transport
            )
        )
    )
    if wave_result.has_fake_adapter or not adapters_valid:
        raise refused_error(
            "PILOT_B_FAKE_ADAPTER_SEAL_REFUSED",
            f"Refused to seal Pilot B part '{part_id}' because adapters or transports are not allow-listed "
            f"(writer={wave_result.writer_adapter_class}, reader_1={wave_result.reader_1_adapter_class}, "
            f"reader_2={wave_result.reader_2_adapter_class}).",
            "Run Pilot B waves with exact OllamaWriterAdapter and OllamaBlindReaderAdapter instances using default_ollama_http_transport.",
        )

    if shuffle_seed is None:
        shuffle_seed = secrets.randbits(128) | (1 << 127)
    elif not isinstance(shuffle_seed, int) or isinstance(shuffle_seed, bool) or shuffle_seed <= 0:
        raise refused_error(
            "PILOT_B_WEAK_SHUFFLE_SEED",
            "A given shuffle seed must be a positive integer.",
            "Omit the seed for a fresh one per build, or pass a large positive one.",
        )
    elif shuffle_seed.bit_length() < MIN_SHUFFLE_SEED_BITS and not allow_weak_test_seed:
        raise refused_error(
            "PILOT_B_WEAK_SHUFFLE_SEED",
            f"A given shuffle seed must have at least {MIN_SHUFFLE_SEED_BITS} bits; a small seed can be searched for.",
            "Omit the seed for a fresh one per build, or pass a large one; small seeds are for tests only.",
        )

    explicit_table_secret = table_secret is not None
    if table_secret is None:
        resolved_table_secret = (
            _generated_table_secret
            if _generated_table_secret is not None
            else (secrets.randbits(128) | (1 << 127))
        )
    else:
        resolved_table_secret = table_secret

    if (
        not isinstance(resolved_table_secret, int)
        or isinstance(resolved_table_secret, bool)
        or resolved_table_secret <= 0
    ):
        raise refused_error(
            "PILOT_B_WEAK_TABLE_SECRET",
            "A given table secret must be a positive integer.",
            "Omit the table secret for a fresh one per build, or pass a positive integer with allow_weak_test_seed=True for tests.",
        )
    if (
        explicit_table_secret or resolved_table_secret.bit_length() < MIN_TABLE_SECRET_BITS
    ) and not allow_weak_test_seed:
        raise refused_error(
            "PILOT_B_WEAK_TABLE_SECRET",
            f"An explicit or small table secret requires allow_weak_test_seed=True (minimum {MIN_TABLE_SECRET_BITS} bits).",
            "Omit table_secret for a fresh per-build secret, or pass allow_weak_test_seed=True for test-only builds.",
        )

    is_test_only = (
        shuffle_seed.bit_length() < MIN_SHUFFLE_SEED_BITS
        or explicit_table_secret
        or resolved_table_secret.bit_length() < MIN_TABLE_SECRET_BITS
    )
    # The commitment key is its own random secret, never derived from the seed.
    commitment_key = secrets.token_bytes(32)

    part_dir.mkdir(parents=True, exist_ok=True)
    keys_sealed_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = _render_inputs(
        part_id=part_id,
        map_version=map_version,
        map_sha256=map_sha256,
        wave_result=wave_result,
        licence_registry=licence_registry,
    )
    rendered = _render_part(inputs, shuffle_seed, commitment_key, test_only=is_test_only)

    store_dir = keys_sealed_path.parent
    (store_dir / SEALING_STORE_NAME).write_bytes(
        _render_json_doc(
            {
                "part_id": part_id,
                "shuffle_seed": shuffle_seed,
                "table_secret": resolved_table_secret,
                "commitment_key": commitment_key.hex(),
                **({"test_only": True} if is_test_only else {}),
            }
        )
    )
    (store_dir / RENDER_INPUTS_NAME).write_bytes(_render_json_doc(inputs))
    keys_sealed_path.write_bytes(rendered["keys.sealed.jsonl"])
    (store_dir / CALL_METRICS_NAME).write_bytes(rendered[CALL_METRICS_NAME])
    for name in _PART_FILES:
        (part_dir / name).write_bytes(rendered[name])
    return json.loads(rendered["part.json"])


def write_public_report(report_path: Path, keys_sealed_path: Path, report: Mapping[str, Any]) -> None:
    """Write the build report: its content is kept in the sealed store, and the public file is its rendering."""

    keys_sealed_path.parent.mkdir(parents=True, exist_ok=True)
    (keys_sealed_path.parent / REPORT_NAME).write_bytes(_render_json_doc({"report": report}))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(_render_json_doc(report))


def _commit_prompt(key: bytes, record: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(record)
    if "prompt_sha256" in out:
        out["prompt_commitment"] = _commit(key, "prompt", str(out.pop("prompt_sha256")))
    return out


def _public_call_record(key: bytes, record: Mapping[str, Any], *, writer: bool) -> dict[str, Any]:
    """A call record as the public part carries it: no per-item measurement, and the writer prompt committed."""

    out = _commit_prompt(key, record) if writer else dict(record)
    for name in PER_ITEM_METRIC_FIELDS:
        out.pop(name, None)
    return out


def _sealed_call_metrics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "operation": r.get("operation"),
            **({"prompt_sha256": r["prompt_sha256"]} if "prompt_sha256" in r else {}),
            **{n: r[n] for n in PER_ITEM_METRIC_FIELDS if n in r},
        }
        for r in records
    ]


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}

    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {"n": len(ordered), "min": ordered[0], "median": at(0.5), "p95": at(0.95), "max": ordered[-1]}


def _call_distributions(
    by_role: Mapping[str, Mapping[str, Mapping[str, Any]]], kept_targets: set[str]
) -> dict[str, Any]:
    """Whole-build distributions of the per-call measurements, by role and operation, over the kept triads only,
    with no item or claim ids."""

    out: dict[str, Any] = {}
    for role, records in sorted(by_role.items()):
        by_operation: dict[str, dict[str, list[float]]] = {}
        for target, record in records.items():
            if target not in kept_targets:
                continue
            bucket = by_operation.setdefault(str(record.get("operation")), {n: [] for n in PER_ITEM_METRIC_FIELDS})
            for name in PER_ITEM_METRIC_FIELDS:
                value = record.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    bucket[name].append(float(value))
        out[role] = {
            op: {name: _distribution(values) for name, values in sorted(metrics.items()) if values}
            for op, metrics in sorted(by_operation.items())
        }
    return out


def _scan_object_for_forbidden_keys(obj: object, path_label: str) -> None:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            k_str = str(k)
            if k_str.lower() in FORBIDDEN_SEALED_FIELD_NAMES:
                raise refused_error(
                    "PILOT_B_SEALED_KEY_FIELD_LEAK",
                    f"Sealed part file '{path_label}' contains forbidden key-bearing field '{k_str}'.",
                    "Remove all key, verdict, label, and answer fields from public files in a sealed part.",
                )
            _scan_object_for_forbidden_keys(v, f"{path_label}.{k_str}")
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for idx, elem in enumerate(obj):
            _scan_object_for_forbidden_keys(elem, f"{path_label}[{idx}]")


# Provenance fields whose value is checked exactly against the sealed store (or the public items) in
# verify_sealed_part, so none of them needs, or gets, an exemption from the content check by name or cardinality.
_VERIFIED_PATTERNS: frozenset[str] = frozenset(
    {
        "proof.trace_commitment",
        "writer.call_records[].prompt_commitment",
        "writer.messages_sha256",
        "judgments[].call_records[].prompt_sha256",
    }
)
# Build-constant digests: exempt only while they take one value across the whole part.
_CONSTANT_DIGESTS: frozenset[str] = frozenset({"map.sha256", "licence_registry_sha256", "licence_registry.sha256"})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _flatten_scalar_fields(obj: Any, prefix: str = "") -> dict[str, str]:
    """Every scalar of a row by path, with list elements by index (``a[0].b``)."""

    out: dict[str, str] = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if not prefix and k in ("id", "triad_id", "claim", "material", "messages"):
                continue
            full_key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_scalar_fields(v, full_key))
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for idx, elem in enumerate(obj):
            out.update(_flatten_scalar_fields(elem, f"{prefix}[{idx}]"))
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out[prefix] = json.dumps(obj, sort_keys=True)
    return out


def _field_pattern(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _item_rows_in(obj: Any, item_to_key: Mapping[str, str]) -> list[dict[str, Any]]:
    """Rows keyed by a sealed item id anywhere inside a parsed JSON document (a report's sample items, say)."""

    found: list[dict[str, Any]] = []
    if isinstance(obj, Mapping):
        if isinstance(obj.get("id"), str) and obj["id"] in item_to_key:
            found.append(dict(obj))
        for value in obj.values():
            found.extend(_item_rows_in(value, item_to_key))
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for element in obj:
            found.extend(_item_rows_in(element, item_to_key))
    return found


def _digest_mismatch(file_name: str, row_id: str, field: str) -> Any:
    return refused_error(
        "PILOT_B_SEALED_DIGEST_MISMATCH",
        f"Field '{field}' of '{row_id}' in '{file_name}' is not the value the sealed store gives for it.",
        "Public digests and commitments must be exactly the ones write_sealed_part computes.",
    )


def _verify_provenance_digests(
    rows: Sequence[Mapping[str, Any]],
    *,
    commitment_key: bytes,
    keys_by_id: Mapping[str, Mapping[str, Any]],
    metrics_by_id: Mapping[str, Mapping[str, Any]],
    messages_by_id: Mapping[str, Any],
    key_ids: Sequence[str],
) -> None:
    """Check each digest and commitment in the public provenance against its ground truth."""

    if [str(r.get("id")) for r in rows] != list(key_ids):
        raise _digest_mismatch("provenance.jsonl", "*", "id")
    for row in rows:
        rid = str(row.get("id"))
        key_row = keys_by_id.get(rid)
        metric_row = metrics_by_id.get(rid)
        if key_row is None or metric_row is None:
            raise _digest_mismatch("provenance.jsonl", rid, "id")
        expected_trace = _commit(commitment_key, "trace", str((key_row.get("proof") or {}).get("trace_sha256")))
        if (row.get("proof") or {}).get("trace_commitment") != expected_trace:
            raise _digest_mismatch("provenance.jsonl", rid, "proof.trace_commitment")
        writer = row.get("writer") or {}
        writer_calls = writer.get("call_records") or []
        sealed_writer = metric_row.get("writer") or []
        if len(writer_calls) != len(sealed_writer):
            raise _digest_mismatch("provenance.jsonl", rid, "writer.call_records")
        for public_call, sealed_call in zip(writer_calls, sealed_writer, strict=True):
            expected = _commit(commitment_key, "prompt", str(sealed_call.get("prompt_sha256")))
            if public_call.get("prompt_commitment") != expected or "prompt_sha256" in public_call:
                raise _digest_mismatch("provenance.jsonl", rid, "writer.call_records[].prompt_commitment")
        expected_messages = _sha256_bytes(_canonical_json_line(messages_by_id.get(rid)).encode("utf-8"))
        if writer.get("messages_sha256") != expected_messages:
            raise _digest_mismatch("provenance.jsonl", rid, "writer.messages_sha256")
        judgments = row.get("judgments")
        sealed_readers = metric_row.get("readers") or []
        if not isinstance(judgments, Sequence) or isinstance(judgments, (str, bytes)):
            raise _digest_mismatch("provenance.jsonl", rid, "judgments")
        if len(sealed_readers) == 0 and len(judgments) == 0:
            continue
        if len(judgments) != 1:
            raise _digest_mismatch("provenance.jsonl", rid, "judgments")
        reader_calls = judgments[0].get("call_records") or []
        if len(reader_calls) != len(sealed_readers):
            raise _digest_mismatch("provenance.jsonl", rid, "judgments[].call_records")
        for public_call, sealed_call in zip(reader_calls, sealed_readers, strict=True):
            if public_call.get("prompt_sha256") != sealed_call.get("prompt_sha256"):
                raise _digest_mismatch("provenance.jsonl", rid, "judgments[].call_records[].prompt_sha256")


def _check_content_leak(
    rows: Sequence[Mapping[str, Any]],
    item_to_key: Mapping[str, str],
    file_name: str,
    *,
    exempt: frozenset[str],
) -> None:
    """Refuse a public field with any value that does not span every class. The same exact rule applies to every
    public file; no public file carries a per-item sample for it to tolerate."""

    joined = [r for r in rows if isinstance(r.get("id"), str) and r["id"] in item_to_key]
    if len(joined) < 3 or len(set(item_to_key.values())) < 2:
        return
    flattened = {str(r["id"]): _flatten_scalar_fields(r) for r in joined}
    all_classes = {item_to_key[rid] for rid in flattened}
    for fname in sorted({name for flat in flattened.values() for name in flat}):
        if _field_pattern(fname) in exempt:
            continue
        present = {item_to_key[rid] for rid, flat in flattened.items() if fname in flat}
        if present != all_classes:
            raise refused_error(
                "PILOT_B_SEALED_CONTENT_LEAK",
                f"Field '{fname}' in '{file_name}' is present on the items of some classes and absent on others.",
                "A field must be present on every item or on none.",
            )
        val_to_keys: dict[str, set[str]] = {}
        for rid, flat in flattened.items():
            val = flat.get(fname, "<missing>")
            val_to_keys.setdefault(val, set()).add(item_to_key[rid])
        # An honest field spans every class: constant, or shared by the three items of a triad. A value held by
        # a proper subset of the classes isolates a class or leaves one out.
        leaking = any(kset != all_classes for kset in val_to_keys.values())
        if leaking:
            raise refused_error(
                "PILOT_B_SEALED_CONTENT_LEAK",
                f"Field '{fname}' in '{file_name}' has per-item values that line up with the sealed key "
                "(ruling 5a content-sealing violation).",
                "Remove per-item key-derived fields from public files; report aggregates only.",
            )


_SEALING_STORE_KEYS: frozenset[str] = frozenset(
    {"part_id", "shuffle_seed", "table_secret", "commitment_key", "test_only"}
)


def _validated_store(keys_sealed_path: Path) -> dict[str, Any]:
    store_path = keys_sealed_path.parent / SEALING_STORE_NAME
    if not store_path.is_file():
        raise refused_error(
            "PILOT_B_SEALED_STORE_MISSING",
            f"The sealed store file '{SEALING_STORE_NAME}' was not found beside the keys.",
            "Write the part with write_sealed_part, which creates it, before verifying or freezing.",
        )
    store = json.loads(store_path.read_text(encoding="utf-8"))
    if not isinstance(store, Mapping) or not set(store) <= _SEALING_STORE_KEYS:
        raise refused_error(
            "PILOT_B_SEALED_STORE_INVALID",
            "The sealed store contains unexpected fields.",
            "Rebuild the part with write_sealed_part.",
        )
    seed = store.get("shuffle_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0:
        raise refused_error(
            "PILOT_B_WEAK_SHUFFLE_SEED",
            "The sealed store holds no positive integer shuffle seed.",
            "Rebuild the part with write_sealed_part.",
        )
    if seed.bit_length() < MIN_SHUFFLE_SEED_BITS and store.get("test_only") is not True:
        raise refused_error(
            "PILOT_B_WEAK_SHUFFLE_SEED",
            f"The sealed store holds a shuffle seed of fewer than {MIN_SHUFFLE_SEED_BITS} bits and is not marked test-only.",
            "Rebuild the part with a fresh seed.",
        )
    table_secret = store.get("table_secret")
    if not isinstance(table_secret, int) or isinstance(table_secret, bool) or table_secret <= 0:
        raise refused_error(
            "PILOT_B_WEAK_TABLE_SECRET",
            "The sealed store holds no positive integer table secret.",
            "Rebuild the part with write_sealed_part.",
        )
    if table_secret.bit_length() < MIN_TABLE_SECRET_BITS and store.get("test_only") is not True:
        raise refused_error(
            "PILOT_B_WEAK_TABLE_SECRET",
            f"The sealed store holds a table secret of fewer than {MIN_TABLE_SECRET_BITS} bits and is not marked test-only.",
            "Rebuild the part with a fresh table secret.",
        )
    commitment_key = store.get("commitment_key")
    if not isinstance(commitment_key, str) or not _HEX64.match(commitment_key):
        raise refused_error(
            "PILOT_B_SEALED_STORE_INVALID",
            "The sealed store holds no valid commitment key.",
            "Rebuild the part with write_sealed_part.",
        )
    return store


_PART_FILES: tuple[str, ...] = ("checksums.txt", "items.jsonl", "part.json", "provenance.jsonl")


def _schema_errors(label: str, errors: list[str]) -> None:
    if errors:
        raise refused_error(
            "PILOT_B_SEALED_SCHEMA_MISMATCH",
            f"Public file '{label}' does not match its schema: {errors[0]}"
            + (f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""),
            "A public file may carry only the fields, entries and values the writer produces.",
        )


def _check_part_shapes(part_dir: Path, extra_public_files: Sequence[Path], key_ids: Sequence[str]) -> None:
    """The allow-list: every public file must be one of the known files and match its shipped schema."""

    present = sorted(p.name for p in part_dir.iterdir())
    if present != list(_PART_FILES):
        raise refused_error(
            "PILOT_B_SEALED_SCHEMA_MISMATCH",
            f"A sealed part holds exactly {list(_PART_FILES)}; found {present}.",
            "Remove files that the writer does not produce.",
        )
    _schema_errors("part.json", schema_check.validate(json.loads((part_dir / "part.json").read_text("utf-8")), "part"))
    items = _jsonl_rows(part_dir / "items.jsonl")
    if [str(r.get("id")) for r in items] != list(key_ids):
        raise _digest_mismatch("items.jsonl", "*", "id")
    for number, row in enumerate(items, start=1):
        _schema_errors(f"items.jsonl:{number}", schema_check.validate(row, "item"))
    for number, row in enumerate(_jsonl_rows(part_dir / "provenance.jsonl"), start=1):
        _schema_errors(f"provenance.jsonl:{number}", schema_check.validate(row, "provenance"))
    expected = [
        f"{_sha256_bytes((part_dir / name).read_bytes())}  {name}"
        for name in _PART_FILES
        if name != "checksums.txt"
    ]
    if (part_dir / "checksums.txt").read_text(encoding="utf-8").splitlines() != sorted(expected, key=lambda x: x[66:]):
        raise refused_error(
            "PILOT_B_SEALED_SCHEMA_MISMATCH",
            "checksums.txt does not list exactly the digests of items.jsonl, part.json and provenance.jsonl.",
            "Rewrite the checksums from the files.",
        )
    for extra in extra_public_files:
        extra = Path(extra)
        if extra.name not in ("pilot_b_report.json", "bakeoff_part_report.json"):
            raise refused_error(
                "PILOT_B_SEALED_SCHEMA_MISMATCH",
                f"Public file '{extra.name}' is not a file of a sealed part or its report.",
                "Pass only the build report as an extra public file.",
            )
        _schema_errors(extra.name, schema_check.validate(json.loads(extra.read_text(encoding="utf-8")), "report"))


def check_items_not_in_key_order(keys_rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuse when items are written in deterministic triad key order or grouped contiguously by key class."""
    if len(keys_rows) < 6:
        return
    keys_seq = [str(r.get("key", r.get("class", ""))) for r in keys_rows]
    triads_seq = [str(r.get("triad_id", r.get("triad_handle", ""))) for r in keys_rows]
    has_triad_ids = any(bool(t) for t in triads_seq)
    n_triads = len(keys_seq) // 3
    triads_contiguous = (
        not has_triad_ids
        or (
            len(keys_seq) == 3 * n_triads
            and all(
                triads_seq[3 * t] == triads_seq[3 * t + 1] == triads_seq[3 * t + 2]
                for t in range(n_triads)
            )
        )
    )
    if len(keys_seq) == 3 * n_triads and (len(keys_seq) >= 12 or triads_contiguous):
        for perm in (
            ("supported", "unsupported", "cannot_tell"),
            ("supported", "cannot_tell", "unsupported"),
            ("unsupported", "supported", "cannot_tell"),
            ("unsupported", "cannot_tell", "supported"),
            ("cannot_tell", "supported", "unsupported"),
            ("cannot_tell", "unsupported", "supported"),
        ):
            if keys_seq == list(perm) * n_triads:
                raise refused_error(
                    "PILOT_B_ITEMS_IN_KEY_ORDER",
                    "Items are written in deterministic triad key order.",
                    "Shuffle items with a private seed before writing.",
                )
    transitions = sum(1 for i in range(1, len(keys_seq)) if keys_seq[i] != keys_seq[i - 1])
    if transitions <= 2 and len(set(keys_seq)) >= 2 and (len(keys_seq) >= 12 or not has_triad_ids):
        raise refused_error(
            "PILOT_B_ITEMS_IN_KEY_ORDER",
            "Items are grouped contiguously in key class order.",
            "Shuffle items with a private seed before writing.",
        )


def _verify_known_leaks(part_dir: Path, keys_sealed_path: Path, extra_public_files: Sequence[Path] = ()) -> None:
    """The early stage of verification: named refusals for the leaks we know of (a sealed value in a public file, a
    forbidden field, a field that lines up with the keys, a digest that is not the sealed store's). The re-render in
    ``_verify_render`` is the authority; these give a specific reason first."""
    if not part_dir.exists() or not part_dir.is_dir():
        raise refused_error(
            "PILOT_B_PART_DIR_NOT_FOUND",
            f"Sealed part directory not found: {part_dir}",
            "Write the sealed part before verifying or freezing it.",
        )
    if not keys_sealed_path.exists():
        raise refused_error(
            "PILOT_B_SEALED_KEYS_NOT_FOUND",
            f"Sealed keys file not found: {keys_sealed_path}",
            "Provide the path to keys.sealed.jsonl.",
        )

    # Ensure keys.sealed.jsonl is not inside the public part directory
    if (part_dir / "keys.sealed.jsonl").exists() or keys_sealed_path.resolve().parent == part_dir.resolve():
        raise refused_error(
            "PILOT_B_SEALED_KEYS_IN_PUBLIC_PART",
            f"keys.sealed.jsonl was found inside public part directory '{part_dir}'.",
            "Store keys.sealed.jsonl outside the public part directory.",
        )

    store = _validated_store(keys_sealed_path)
    metrics_path = keys_sealed_path.parent / CALL_METRICS_NAME
    if not metrics_path.is_file():
        raise refused_error(
            "PILOT_B_SEALED_STORE_MISSING",
            f"The sealed store file '{CALL_METRICS_NAME}' was not found beside the keys.",
            "Write the part with write_sealed_part, which creates it, before verifying or freezing.",
        )
    keys_bytes = keys_sealed_path.read_bytes()
    part_json_path = part_dir / "part.json"
    if part_json_path.is_file():
        declared = json.loads(part_json_path.read_text(encoding="utf-8")).get("keys_sealed_sha256")
        if declared is not None and declared != _sha256_bytes(keys_bytes):
            raise refused_error(
                "PILOT_B_SEALED_KEYS_MISMATCH",
                "part.json names a keys digest that does not match the sealed keys file.",
                "Rebuild the part, or restore the sealed keys file it was written with.",
            )

    keys_rows = _jsonl_rows(keys_sealed_path)
    check_items_not_in_key_order(keys_rows)
    metrics_rows = _jsonl_rows(metrics_path)
    key_ids = [str(r["id"]) for r in keys_rows]
    if [str(r["id"]) for r in metrics_rows] != key_ids:
        raise _digest_mismatch(CALL_METRICS_NAME, "*", "id")
    item_to_key = {str(r["id"]): str(r["key"]) for r in keys_rows}
    keys_by_id = {str(r["id"]): r for r in keys_rows}
    metrics_by_id = {str(r["id"]): r for r in metrics_rows}

    # Every value the sealed store holds, which no public file may carry (compared case-insensitively)
    seed = int(store["shuffle_seed"])
    table_secret = int(store["table_secret"])
    sealed_values: list[str] = [store["commitment_key"]]
    if seed >= 10**11:
        sealed_values += [str(seed), hex(seed)[2:]]
    if table_secret >= 10**11:
        sealed_values += [str(table_secret), hex(table_secret)[2:]]
    for key_row in keys_rows:
        sealed_values.append(str(key_row.get("triad_id", "")))
        trace = (key_row.get("proof") or {}).get("trace_sha256")
        if isinstance(trace, str):
            sealed_values.append(trace)
    for metric_row in metrics_rows:
        for call in metric_row.get("writer") or []:
            if isinstance(call.get("prompt_sha256"), str):
                sealed_values.append(call["prompt_sha256"])
    sealed_values = [v.lower() for v in dict.fromkeys(sealed_values) if v]

    public_files = [p for p in sorted(part_dir.iterdir()) if p.is_file()] + [Path(p) for p in extra_public_files]
    for public in public_files:
        text = public.read_text(encoding="utf-8").lower()
        if any(value in text for value in sealed_values):
            raise refused_error(
                "PILOT_B_SEALED_SEED_IN_PUBLIC_FILE",
                f"Sealed part file '{public.name}' contains a value that belongs to the sealed store.",
                "Keep the per-build seed, keys and commitment key only in the sealed store.",
            )

    items_path = part_dir / "items.jsonl"
    messages_by_id = {str(r["id"]): r.get("messages") for r in _jsonl_rows(items_path)} if items_path.is_file() else {}

    for file_path in public_files:
        if file_path.name == "checksums.txt":
            continue
        if file_path.suffix == ".json":
            parsed = json.loads(file_path.read_text(encoding="utf-8"))
            _scan_object_for_forbidden_keys(parsed, file_path.name)
            _check_content_leak(_item_rows_in(parsed, item_to_key), item_to_key, file_path.name, exempt=frozenset())
        elif file_path.suffix == ".jsonl":
            rows = _jsonl_rows(file_path)
            for line_no, row in enumerate(rows, start=1):
                _scan_object_for_forbidden_keys(row, f"{file_path.name}:{line_no}")
            exempt: frozenset[str] = frozenset()
            if file_path.name == "provenance.jsonl":
                _verify_provenance_digests(
                    rows,
                    commitment_key=bytes.fromhex(store["commitment_key"]),
                    keys_by_id=keys_by_id,
                    metrics_by_id=metrics_by_id,
                    messages_by_id=messages_by_id,
                    key_ids=key_ids,
                )
                flat_rows = [_flatten_scalar_fields(r) for r in rows]
                for pattern in _CONSTANT_DIGESTS:
                    seen = {
                        next((f[p] for p in f if _field_pattern(p) == pattern), "<missing>") for f in flat_rows
                    }
                    if len(seen) != 1 or "<missing>" in seen:
                        raise _digest_mismatch(file_path.name, "*", pattern)
                exempt = _VERIFIED_PATTERNS
            _check_content_leak(rows, item_to_key, file_path.name, exempt=exempt)

    part_meta = json.loads(part_json_path.read_text(encoding="utf-8"))
    if part_meta.get("frozen") is True:
        if store.get("test_only") is True:
            raise refused_error(
                "PILOT_B_SEALED_TEST_ONLY_FROZEN",
                "A part written with a test-only seed is marked frozen.",
                "A test-only part is never releasable; rebuild it with a fresh seed.",
            )
    _check_part_shapes(part_dir, extra_public_files, key_ids)


def _check_report_coverage(report: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    """The report's per-domain coverage must be the catalogue's domains with the counts the inputs give."""

    from reasoning_dataset_lab.pilot_b.fact_tables import DOMAIN_CATALOGUE

    catalogue = {str(d["domain"]) for d in DOMAIN_CATALOGUE}
    by_domain = (report.get("coverage") or {}).get("by_domain")
    if not isinstance(by_domain, Mapping) or not set(by_domain) <= catalogue:
        raise _render_mismatch("pilot_b_report.json", "its coverage names something other than the domain catalogue")
    built: dict[str, int] = {}
    for entry in inputs["entries"]:
        built[entry["domain"]] = built.get(entry["domain"], 0) + 1
    def whole(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    target_total = 0
    for domain, counts in by_domain.items():
        if not isinstance(counts, Mapping) or not whole(counts.get("built")) or not whole(counts.get("target")):
            raise _render_mismatch("pilot_b_report.json", f"its coverage for '{domain}' is not whole numbers")
        if counts["built"] != built.get(domain, 0) or counts["target"] < counts["built"] or counts["target"] % 3:
            raise _render_mismatch("pilot_b_report.json", f"its coverage for '{domain}' differs from the build")
        target_total += counts["target"]
    if target_total != 3 * (inputs["triads_kept"] + inputs["triads_failed"]):
        raise _render_mismatch("pilot_b_report.json", "its coverage targets do not add up to the triads attempted")
    if {d for d, n in built.items() if n} - set(by_domain):
        raise _render_mismatch("pilot_b_report.json", "its coverage leaves out a domain that was built")


def _render_mismatch(label: str, detail: str) -> Any:
    return refused_error(
        "PILOT_B_SEALED_RENDER_MISMATCH",
        f"Public file '{label}' is not the byte-exact rendering the sealed store gives: {detail}",
        "A part must be exactly what write_sealed_part renders; rebuild it, or restore the file.",
    )


def _verify_render(part_dir: Path, keys_sealed_path: Path, extra_public_files: Sequence[Path]) -> None:
    """The authority: re-render every public file from the sealed store and require a byte-for-byte match."""

    store = _validated_store(keys_sealed_path)
    store_dir = keys_sealed_path.parent
    inputs_path = store_dir / RENDER_INPUTS_NAME
    if not inputs_path.is_file():
        raise refused_error(
            "PILOT_B_SEALED_STORE_MISSING",
            f"The sealed store file '{RENDER_INPUTS_NAME}' was not found beside the keys.",
            "Write the part with write_sealed_part, which creates it, before verifying or freezing.",
        )
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    _check_render_inputs(inputs)
    _check_sealed_rows(inputs, _jsonl_rows(keys_sealed_path), _jsonl_rows(store_dir / CALL_METRICS_NAME))
    commitment_key = bytes.fromhex(store["commitment_key"])
    seed = int(store["shuffle_seed"])

    actual_meta = json.loads((part_dir / "part.json").read_text(encoding="utf-8"))
    frozen = actual_meta.get("frozen") is True
    if frozen:
        claimed = actual_meta.get("frozen_mac")
        expected = _part_mac(
            (part_dir / "items.jsonl").read_bytes(),
            (part_dir / "provenance.jsonl").read_bytes(),
            actual_meta,
            commitment_key,
        )
        if not isinstance(claimed, str) or not hmac.compare_digest(claimed, expected):
            raise refused_error(
                "PILOT_B_SEALED_FROZEN_MISMATCH",
                "part.json says frozen, but its seal does not match the part.",
                "Only freeze_sealed_part marks a part frozen; rebuild or re-freeze it.",
            )
    rendered = _render_part(
        inputs,
        seed,
        commitment_key,
        frozen=frozen,
        test_only=bool(store.get("test_only")),
    )
    for name in _PART_FILES:
        if (part_dir / name).read_bytes() != rendered[name]:
            raise _render_mismatch(name, "the bytes differ")
    if keys_sealed_path.read_bytes() != rendered["keys.sealed.jsonl"]:
        raise _render_mismatch("keys.sealed.jsonl", "the bytes differ")
    if (store_dir / CALL_METRICS_NAME).read_bytes() != rendered[CALL_METRICS_NAME]:
        raise _render_mismatch(CALL_METRICS_NAME, "the bytes differ")

    for extra in extra_public_files:
        extra = Path(extra)
        report_store = store_dir / REPORT_NAME
        if not report_store.is_file():
            raise _render_mismatch(extra.name, "the sealed store holds no report to render it from")
        report = json.loads(report_store.read_text(encoding="utf-8"))["report"]
        if extra.read_bytes() != _render_json_doc(report):
            raise _render_mismatch(extra.name, "the bytes differ")
        shared = {
            "part_id": actual_meta["part_id"],
            "items_total": actual_meta["items_count"],
            "triads_kept": actual_meta["triads_kept"],
            "triads_failed": actual_meta["triads_failed"],
            "triad_pass_rate": actual_meta["judgment_aggregates"]["triad_pass_rate"],
            "gate_pass_rate_per_class": actual_meta["judgment_aggregates"]["pass_rate_per_class"],
            "num_predict": actual_meta["num_predict"],
            "num_predict_source": actual_meta["num_predict_source"],
            "num_predict_config": actual_meta["num_predict_config"],
            "waves": actual_meta["waves"],
            "licence_registry": inputs["registry"],
        }
        for field, value in shared.items():
            if _normalise(report.get(field)) != _normalise(value):
                raise _render_mismatch(extra.name, f"'{field}' differs from part.json")
        _check_report_coverage(report, inputs)
        text = extra.read_text(encoding="utf-8")
        public_ids = [str(r["id"]) for r in _jsonl_rows(part_dir / "items.jsonl")] + [
            str(r["triad_id"]) for r in _jsonl_rows(part_dir / "items.jsonl")
        ]
        if any(i in text for i in public_ids):
            raise _render_mismatch(extra.name, "it names an item or a triad of the part")


def verify_sealed_part(
    part_dir: Path, keys_sealed_path: Path, extra_public_files: Sequence[Path] = ()
) -> None:
    """Verify a sealed part against the sealed store: every public file must be a byte-exact honest rendering.

    Threat model. Honest builds leak nothing about keys or positions (this is checked by probes in the tests).
    verify's job is to refuse any public file that is not a byte-exact honest rendering from the sealed store, so a
    tampered or buggy build cannot be frozen. Without the sealed store a part is unverifiable (``part_status``).

    Two stages. First, named refusals for the leaks we know of (``_verify_known_leaks``); a file that cannot even be
    read as the writer's output counts as a mismatch. Then ``_verify_render`` regenerates every public file (part.json,
    items.jsonl, provenance.jsonl, checksums.txt, and each extra file such as the report) from the sealed store and
    compares the bytes. Any signal an honest render does not produce fails there, including ones nobody listed.
    """
    try:
        _verify_known_leaks(part_dir, keys_sealed_path, extra_public_files)
    except RdlabError:
        raise
    except Exception as exc:
        raise _render_mismatch("a public file", f"it could not be read as the writer's output ({exc!r})") from exc
    try:
        _verify_render(part_dir, keys_sealed_path, extra_public_files)
    except RdlabError:
        raise
    except Exception as exc:
        raise _render_mismatch("a public file", f"it could not be re-rendered ({exc!r})") from exc


def freeze_sealed_part(
    part_dir: Path, keys_sealed_path: Path, extra_public_files: Sequence[Path] = ()
) -> dict[str, Any]:
    """Verify the part, then mark it frozen by rendering it again with its seal.

    A part written with a test-only seed is verified but never marked frozen: it is not releasable.
    """
    verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_sealed_path, extra_public_files=extra_public_files)
    store = _validated_store(keys_sealed_path)
    inputs = json.loads((keys_sealed_path.parent / RENDER_INPUTS_NAME).read_text(encoding="utf-8"))
    rendered = _render_part(
        inputs,
        int(store["shuffle_seed"]),
        bytes.fromhex(store["commitment_key"]),
        frozen=True,
        test_only=bool(store.get("test_only")),
    )
    for name in ("part.json", "checksums.txt"):
        (part_dir / name).write_bytes(rendered[name])
    return json.loads(rendered["part.json"])


def part_status(part_dir: Path, keys_sealed_path: Path | None = None) -> str:
    """``"frozen"``, ``"not_frozen"``, or ``"unverifiable"``.

    Without the sealed store nothing a public file says about itself can be trusted, so the answer is
    ``"unverifiable"``, never ``"frozen"``. With it, the part is verified and its seal recomputed.
    """

    if keys_sealed_path is None or not (Path(keys_sealed_path).parent / SEALING_STORE_NAME).is_file():
        return "unverifiable"
    verify_sealed_part(part_dir=part_dir, keys_sealed_path=Path(keys_sealed_path))
    meta = json.loads((part_dir / "part.json").read_text(encoding="utf-8"))
    return "frozen" if meta.get("frozen") is True else "not_frozen"
