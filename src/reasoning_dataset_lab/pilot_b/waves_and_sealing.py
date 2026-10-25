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
import math
import random
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import reasoning_dataset_lab.pilot_b.ollama as ollama_mod
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
    check_min_pass_rate_per_class,
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
        "commitment_key",
    }
)

SEALING_STORE_NAME = "sealing.sealed.json"
CALL_METRICS_NAME = "call_metrics.sealed.jsonl"

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
    writer_adapter: object = None
    reader_1_adapter: object = None
    reader_2_adapter: object = None
    writer_call_records: Mapping[str, dict[str, object]] | None = None
    reader_1_call_records: Mapping[str, dict[str, object]] | None = None
    reader_2_call_records: Mapping[str, dict[str, object]] | None = None
    num_predict_source: NumPredictSource = "default"
    num_predict_config: NumPredictConfig | None = None


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
        evaluations, min_pass_rate_per_class=min_pass_rate_per_class
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


def _canonical_json_line(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


MIN_SHUFFLE_SEED_BITS = 64


def _triad_handle(key: bytes, triad_id: str) -> str:
    """The public name of a triad: an opaque keyed handle, so the public part carries no generation seed."""

    return "triad-" + _commit(key, "triad", triad_id)[:16]


def _commit(key: bytes, kind: str, value: str) -> str:
    """A keyed commitment to a hash, so the public provenance carries no plain digest of per-item material."""

    return hmac.new(key, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()


def write_sealed_part(
    *,
    part_dir: Path,
    keys_sealed_path: Path,
    part_id: str,
    map_version: int,
    map_sha256: str,
    wave_result: PilotBWaveRunResult,
    shuffle_seed: int | None = None,
    licence_registry: Mapping[str, Any] | None = None,
    allow_weak_test_seed: bool = False,
) -> dict[str, Any]:
    """Write a sealed Pilot B part directory with shuffled items, aggregate-only judgment provenance, and external keys.sealed.jsonl.

    The shuffle seed is drawn per build from ``secrets`` unless one is given (a reproducible re-seal), and it is
    kept only in the sealed store, in ``sealing.sealed.json`` beside the keys. It is never written to a public
    file of the part. The hashes in the public provenance that could be matched against a rebuilt triad
    (the proof trace and the writer's claim prompt) are published as keyed commitments.
    """
    w = wave_result.writer_adapter
    r1 = wave_result.reader_1_adapter
    r2 = wave_result.reader_2_adapter
    if wave_result.has_fake_adapter or not (
        type(w) is OllamaWriterAdapter
        and type(r1) is OllamaBlindReaderAdapter
        and type(r2) is OllamaBlindReaderAdapter
        and w.http_transport is ollama_mod.default_ollama_http_transport
        and r1.http_transport is ollama_mod.default_ollama_http_transport
        and r2.http_transport is ollama_mod.default_ollama_http_transport
    ):
        raise refused_error(
            "PILOT_B_FAKE_ADAPTER_SEAL_REFUSED",
            f"Refused to seal Pilot B part '{part_id}' because adapters or transports are not allow-listed "
            f"(writer={wave_result.writer_adapter_class}, reader_1={wave_result.reader_1_adapter_class}, "
            f"reader_2={wave_result.reader_2_adapter_class}).",
            "Run Pilot B waves with exact OllamaWriterAdapter and OllamaBlindReaderAdapter instances using default_ollama_http_transport.",
        )

    if shuffle_seed is None:
        shuffle_seed = secrets.randbits(128)
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
    # The commitment key is its own random secret, never derived from the seed.
    commitment_key = secrets.token_bytes(32)

    part_dir.mkdir(parents=True, exist_ok=True)
    keys_sealed_path.parent.mkdir(parents=True, exist_ok=True)

    w_calls_by_target = dict(wave_result.writer_call_records or {})
    r1_calls_by_target = dict(wave_result.reader_1_call_records or {})
    r2_calls_by_target = dict(wave_result.reader_2_call_records or {})

    default_read_dates: dict[str, str] = {}
    for m in (
        wave_result.writer_model,
        wave_result.reader_1_model,
        wave_result.reader_2_model,
    ):
        if m.licence_id:
            default_read_dates.setdefault(m.licence_id, m.licence_read_date)
        if m.bundled_licence_id:
            default_read_dates.setdefault(
                m.bundled_licence_id, m.bundled_licence_read_date
            )

    reg_meta: dict[str, Any] = (
        dict(licence_registry)
        if licence_registry is not None
        else {
            "path": "",
            "sha256": "0" * 64,
            "read_dates": default_read_dates,
        }
    )

    kept_targets: set[str] = set()
    for ev in wave_result.kept_triads:
        kept_targets.add(ev.worded.triad.table.table_id)
        kept_targets.update(claim.claim_id for claim in ev.worded.triad.claims.values())

    # Collect raw unshuffled items from kept triads
    raw_entries: list[dict[str, Any]] = []
    for ev in wave_result.kept_triads:
        wt = ev.worded
        table = wt.triad.table
        for key in VERDICT_CLASSES:
            claim = wt.triad.claims[key]
            claim_ev = ev.claim_evals[key]
            w_recs: list[dict[str, object]] = []
            if table.table_id in w_calls_by_target:
                w_recs.append(dict(w_calls_by_target[table.table_id]))
            if claim.claim_id in w_calls_by_target:
                w_recs.append(dict(w_calls_by_target[claim.claim_id]))
            r_recs: list[dict[str, object]] = []
            if claim.claim_id in r1_calls_by_target:
                r_recs.append(dict(r1_calls_by_target[claim.claim_id]))
            if claim.claim_id in r2_calls_by_target:
                r_recs.append(dict(r2_calls_by_target[claim.claim_id]))
            raw_entries.append(
                {
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
            )

    # Shuffle items with recorded shuffle_seed so triad position cannot hint at the key (§2.1)
    rng = random.Random(shuffle_seed)
    shuffled_indices = list(range(len(raw_entries)))
    rng.shuffle(shuffled_indices)

    items_lines: list[str] = []
    prov_lines: list[str] = []
    sealed_key_lines: list[str] = []
    metric_lines: list[str] = []

    for out_idx, orig_idx in enumerate(shuffled_indices, start=1):
        entry = raw_entries[orig_idx]
        item_id = f"{part_id}-item-{out_idx:03d}"
        user_turn = (
            f"Material ({entry['closure']}):\n{entry['material']}\n\n"
            f"Claim: {entry['claim_text']}"
        )
        msg_obj = [{"role": "user", "content": user_turn}]
        msg_sha = _sha256_bytes(_canonical_json_line(msg_obj).encode("utf-8"))

        # Public items.jsonl carries NO key, verdict, label, or answer (§7, §11 B.4)
        item_row = {
            "id": item_id,
            "triad_id": _triad_handle(commitment_key, entry["triad_id"]),
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
        items_lines.append(_canonical_json_line(item_row))

        # Public provenance.jsonl carries judgment outcomes ONLY as constant aggregates (§2.1, ruling 5a)
        prov_row = {
            "id": item_id,
            "map": {
                "version": map_version,
                "sha256": map_sha256,
                "cell": f"T10/D/{entry['domain']}",
            },
            "licence_registry_path": str(reg_meta.get("path", "")),
            "licence_registry_sha256": str(reg_meta.get("sha256", "")),
            "licence_registry": reg_meta,
            "proof_basis": "proved",
            "key_basis": "construction",
            "proof": {
                "prover": entry["proof"]["prover"],
                "version": entry["proof"]["version"],
                "trace_commitment": _commit(commitment_key, "trace", entry["proof"]["trace_sha256"]),
            },
            "writer": {
                "adapter": wave_result.writer_adapter_class,
                "model": wave_result.writer_model.name,
                "family": wave_result.writer_model.normalized_family,
                "digest": wave_result.writer_model.digest,
                "licence_id": wave_result.writer_model.licence_id,
                "licence_read_date": wave_result.writer_model.licence_read_date,
                "bundled_licence_id": wave_result.writer_model.bundled_licence_id,
                "bundled_licence_read_date": wave_result.writer_model.bundled_licence_read_date,
                "licence_layer_digest": wave_result.writer_model.licence_layer_digest,
                "source_match": wave_result.writer_model.source_match,
                "source_match_reason": wave_result.writer_model.source_match_reason,
                "messages_sha256": msg_sha,
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
                "max_num_ctx": int(
                    getattr(
                        wave_result.writer_adapter,
                        "max_num_ctx",
                        wave_result.writer_model.max_num_ctx,
                    )
                ),
                "call_records": [
                    _public_call_record(commitment_key, rec, writer=True) for rec in entry["writer_call_records"]
                ],
            },
            "checks": [
                {
                    "name": "pilot_b.rendering_check",
                    "version": "0.3.0",
                    "result": "pass",
                },
                {
                    "name": "pilot_b.unanimity_gate",
                    "version": "0.3.0",
                    "result": "pass",
                },
            ],
            "judgments": [
                {
                    "kind": "blind_read_unanimity_aggregate",
                    "reader_adapters": [
                        wave_result.reader_1_adapter_class,
                        wave_result.reader_2_adapter_class,
                    ],
                    "readers": [
                        f"model:{wave_result.reader_1_model.name}@{wave_result.reader_1_model.digest}",
                        f"model:{wave_result.reader_2_model.name}@{wave_result.reader_2_model.digest}",
                    ],
                    "families": [
                        wave_result.reader_1_model.normalized_family,
                        wave_result.reader_2_model.normalized_family,
                    ],
                    "reader_licence_ids": [
                        wave_result.reader_1_model.licence_id,
                        wave_result.reader_2_model.licence_id,
                    ],
                    "reader_licence_read_dates": [
                        wave_result.reader_1_model.licence_read_date,
                        wave_result.reader_2_model.licence_read_date,
                    ],
                    "reader_bundled_licence_ids": [
                        wave_result.reader_1_model.bundled_licence_id,
                        wave_result.reader_2_model.bundled_licence_id,
                    ],
                    "reader_bundled_licence_read_dates": [
                        wave_result.reader_1_model.bundled_licence_read_date,
                        wave_result.reader_2_model.bundled_licence_read_date,
                    ],
                    "reader_licence_layer_digests": [
                        wave_result.reader_1_model.licence_layer_digest,
                        wave_result.reader_2_model.licence_layer_digest,
                    ],
                    "reader_source_matches": [
                        wave_result.reader_1_model.source_match,
                        wave_result.reader_2_model.source_match,
                    ],
                    "reader_source_match_reasons": [
                        wave_result.reader_1_model.source_match_reason,
                        wave_result.reader_2_model.source_match_reason,
                    ],
                    "reader_max_num_ctx": [
                        int(
                            getattr(
                                wave_result.reader_1_adapter,
                                "max_num_ctx",
                                wave_result.reader_1_model.max_num_ctx,
                            )
                        ),
                        int(
                            getattr(
                                wave_result.reader_2_adapter,
                                "max_num_ctx",
                                wave_result.reader_2_model.max_num_ctx,
                            )
                        ),
                    ],
                    "call_records": [
                        _public_call_record(commitment_key, rec, writer=False) for rec in entry["reader_call_records"]
                    ],
                    "agrees_all_readers": True,
                    "ambiguous_count": 0,
                }
            ],
        }
        prov_lines.append(_canonical_json_line(prov_row))

        # Sealed keys store holds the item_id -> key, proof method, and per-reader verdicts
        sealed_row = {
            "id": item_id,
            "triad_id": entry["triad_id"],
            "triad_handle": _triad_handle(commitment_key, entry["triad_id"]),
            "key": entry["key"],
            "proof": entry["proof"],
            "blind_reads": [entry["read_1"], entry["read_2"]],
        }
        sealed_key_lines.append(_canonical_json_line(sealed_row))
        metric_lines.append(
            _canonical_json_line(
                {
                    "id": item_id,
                    "writer": _sealed_call_metrics(entry["writer_call_records"]),
                    "readers": _sealed_call_metrics(entry["reader_call_records"]),
                }
            )
        )

    items_bytes = ("\n".join(items_lines) + ("\n" if items_lines else "")).encode("utf-8")
    prov_bytes = ("\n".join(prov_lines) + ("\n" if prov_lines else "")).encode("utf-8")
    sealed_bytes = (
        "\n".join(sealed_key_lines) + ("\n" if sealed_key_lines else "")
    ).encode("utf-8")

    (part_dir / "items.jsonl").write_bytes(items_bytes)
    (part_dir / "provenance.jsonl").write_bytes(prov_bytes)
    keys_sealed_path.write_bytes(sealed_bytes)
    (keys_sealed_path.parent / CALL_METRICS_NAME).write_text(
        "\n".join(metric_lines) + ("\n" if metric_lines else ""), encoding="utf-8"
    )
    (keys_sealed_path.parent / SEALING_STORE_NAME).write_text(
        json.dumps(
            {
                "part_id": part_id,
                "shuffle_seed": shuffle_seed,
                "commitment_key": commitment_key.hex(),
                **({"test_only": True} if shuffle_seed.bit_length() < MIN_SHUFFLE_SEED_BITS else {}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    keys_sealed_sha256 = _sha256_bytes(sealed_bytes)

    part_meta: dict[str, Any] = {
        "part_id": part_id,
        "split": "sealed",
        "frozen": False,
        "map_version": map_version,
        "map_sha256": map_sha256,
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
        "items_count": len(items_lines),
        "triads_kept": len(wave_result.kept_triads),
        "triads_failed": len(wave_result.failed_triads),
        "keys_sealed_sha256": keys_sealed_sha256,
        "call_distributions": _call_distributions(
            {"writer": w_calls_by_target, "reader_1": r1_calls_by_target, "reader_2": r2_calls_by_target},
            kept_targets,
        ),
        "judgment_aggregates": {
            "pass_rate_per_class": dict(wave_result.pass_rate_per_class),
            "triad_pass_rate": wave_result.triad_pass_rate,
        },
        "waves": [w.to_dict() for w in wave_result.wave_metrics],
    }
    part_json_bytes = (
        json.dumps(part_meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    (part_dir / "part.json").write_bytes(part_json_bytes)

    _write_checksums(part_dir)
    return part_meta


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


def _write_checksums(part_dir: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for filename in ("items.jsonl", "provenance.jsonl", "part.json"):
        fpath = part_dir / filename
        if fpath.exists():
            checksums[filename] = _sha256_bytes(fpath.read_bytes())
    lines = [f"{sha}  {name}" for name, sha in sorted(checksums.items())]
    (part_dir / "checksums.txt").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return checksums


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
) -> None:
    """Check each digest and commitment in the public provenance against its ground truth."""

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
        judgments = row.get("judgments") or []
        reader_calls = (judgments[0].get("call_records") if judgments else None) or []
        sealed_readers = metric_row.get("readers") or []
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
    complete: bool = True,
) -> None:
    """Refuse a public field whose values line up with the sealed keys.

    In a complete file (every item of the part, whose triad-level fields always span all three classes) a field is
    refused when every value maps to one class, or when a value shared by several items maps to one class (a field
    that isolates a class from the others). In a partial one, such as a report's random sample of items, a single-class
    value can happen by chance, so the evidence must be strong: the chance of the single-class values seen, taking
    each class as equally likely, below one in ten thousand.
    """

    joined = [r for r in rows if isinstance(r.get("id"), str) and r["id"] in item_to_key]
    if len(joined) < 3 or len(set(item_to_key.values())) < 2:
        return
    flattened = {str(r["id"]): _flatten_scalar_fields(r) for r in joined}
    for fname in sorted({name for flat in flattened.values() for name in flat}):
        if _field_pattern(fname) in exempt:
            continue
        val_to_keys: dict[str, set[str]] = {}
        val_counts: dict[str, int] = {}
        for rid, flat in flattened.items():
            val = flat.get(fname, "<missing>")
            val_to_keys.setdefault(val, set()).add(item_to_key[rid])
            val_counts[val] = val_counts.get(val, 0) + 1
        if len(val_to_keys) < 2:
            continue
        if complete:
            all_single = all(len(kset) == 1 for kset in val_to_keys.values())
            isolates = any(len(kset) == 1 and val_counts[val] >= 2 for val, kset in val_to_keys.items())
            leaking = all_single or isolates
        else:
            evidence = sum((val_counts[v] - 1) * math.log(3) for v, kset in val_to_keys.items() if len(kset) == 1)
            leaking = evidence >= math.log(10_000)
        if leaking:
            raise refused_error(
                "PILOT_B_SEALED_CONTENT_LEAK",
                f"Field '{fname}' in '{file_name}' has per-item values that line up with the sealed key "
                "(ruling 5a content-sealing violation).",
                "Remove per-item key-derived fields from public files; report aggregates only.",
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
    commitment_key = store.get("commitment_key")
    if not isinstance(commitment_key, str) or not _HEX64.match(commitment_key):
        raise refused_error(
            "PILOT_B_SEALED_STORE_INVALID",
            "The sealed store holds no valid commitment key.",
            "Rebuild the part with write_sealed_part.",
        )
    return store


def verify_sealed_part(
    part_dir: Path, keys_sealed_path: Path, extra_public_files: Sequence[Path] = ()
) -> None:
    """Verify that no public file (the part's files and any extra ones, such as the report) holds a sealed value, a
    forbidden field, or a per-item value that lines up with the keys, and that every public digest is the exact one
    the sealed store gives (ruling 5a)."""
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
    metrics_rows = _jsonl_rows(metrics_path)
    item_to_key = {str(r["id"]): str(r["key"]) for r in keys_rows}
    keys_by_id = {str(r["id"]): r for r in keys_rows}
    metrics_by_id = {str(r["id"]): r for r in metrics_rows}

    # Every value the sealed store holds, which no public file may carry (compared case-insensitively)
    seed = int(store["shuffle_seed"])
    sealed_values: list[str] = [store["commitment_key"]]
    if seed >= 10**11:
        sealed_values += [str(seed), hex(seed)[2:]]
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
            _check_content_leak(
                _item_rows_in(parsed, item_to_key), item_to_key, file_path.name, exempt=frozenset(), complete=False
            )
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
                )
                for pattern in _CONSTANT_DIGESTS:
                    values = {
                        _flatten_scalar_fields(r).get(p)
                        for r in rows
                        for p in _flatten_scalar_fields(r)
                        if _field_pattern(p) == pattern
                    }
                    if len(values) > 1:
                        raise _digest_mismatch(file_path.name, "*", pattern)
                exempt = _VERIFIED_PATTERNS | _CONSTANT_DIGESTS
            _check_content_leak(rows, item_to_key, file_path.name, exempt=exempt)


def freeze_sealed_part(
    part_dir: Path, keys_sealed_path: Path, extra_public_files: Sequence[Path] = ()
) -> dict[str, Any]:
    """Run all sealed-part checks, mark part.json frozen=True, and update checksums.txt.

    A part written with a test-only seed is verified but never marked frozen: it is not releasable.
    """
    verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_sealed_path, extra_public_files=extra_public_files)
    store = json.loads((keys_sealed_path.parent / SEALING_STORE_NAME).read_text(encoding="utf-8"))
    part_json_path = part_dir / "part.json"
    part_meta: dict[str, Any] = json.loads(part_json_path.read_text(encoding="utf-8"))
    if store.get("test_only") is True:
        part_meta["frozen"] = False
        part_meta["test_only"] = True
    else:
        part_meta["frozen"] = True
    part_json_path.write_bytes(
        (json.dumps(part_meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
    )
    _write_checksums(part_dir)
    return part_meta
