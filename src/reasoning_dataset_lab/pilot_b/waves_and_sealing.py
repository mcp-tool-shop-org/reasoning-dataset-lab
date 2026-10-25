"""Part D: Waves, caps and sealing (DESIGN.md ruling 5a, ruling 8b, §2.1, §7, §11 B.4).

Runs go in single-model waves (render, word, reader_1, reader_2), each model loaded once,
never interleaved per triad. Caps thinking tokens per call and records the cap. Keys go
only to keys.sealed.jsonl, and the freeze refuses any key-bearing field name or per-item
key-derived value anywhere else in the part. Items are shuffled with a recorded seed so
a triad's position cannot hint at its key, and provenance carries judgment outcomes only
as aggregates (§2.1).
"""

from __future__ import annotations

import hashlib
import json
import random
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
    NUM_CTX_MARGIN_TOKENS,
    NUM_CTX_STEP_TOKENS,
    PROMPT_TOKEN_ESTIMATE_RULE,
    OllamaBlindReaderAdapter,
    OllamaWriterAdapter,
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
DEFAULT_THINKING_TOKEN_CAP: int = 256

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
    }
)


@dataclass(frozen=True)
class WaveCallEntry:
    """Single model invocation recorded during a wave."""

    wave_name: WaveName
    model_name: str
    target_id: str
    thinking_tokens: int
    completion_tokens: int
    seconds: float
    num_ctx: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "wave_name": self.wave_name,
            "model_name": self.model_name,
            "target_id": self.target_id,
            "thinking_tokens": self.thinking_tokens,
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
    thinking_token_cap: int
    thinking_tokens_total: int
    max_thinking_tokens_observed: int
    completion_tokens_total: int
    seconds_total: float
    adapter: str = ""
    unloaded: bool = False
    max_num_ctx: int = 0
    max_num_ctx_observed: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "wave_name": self.wave_name,
            "adapter": self.adapter,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "model_digest": self.model_digest,
            "calls": self.calls,
            "thinking_token_cap": self.thinking_token_cap,
            "thinking_tokens_total": self.thinking_tokens_total,
            "max_thinking_tokens_observed": self.max_thinking_tokens_observed,
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
    thinking_token_cap: int
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


class WaveCallTracker:
    """Enforces strict non-interleaved wave execution and per-call thinking token caps."""

    def __init__(self, thinking_token_cap: int) -> None:
        if thinking_token_cap < 1:
            raise refused_error(
                "PILOT_B_INVALID_THINKING_CAP",
                f"thinking_token_cap must be >= 1, got {thinking_token_cap}.",
                "Set a positive integer thinking_token_cap.",
            )
        self._cap = thinking_token_cap
        self._entries: list[WaveCallEntry] = []
        self._seen_waves: list[WaveName] = []

    @property
    def thinking_token_cap(self) -> int:
        return self._cap

    @property
    def entries(self) -> tuple[WaveCallEntry, ...]:
        return tuple(self._entries)

    def record(
        self,
        *,
        wave_name: WaveName,
        model_name: str,
        target_id: str,
        thinking_tokens: int,
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
                thinking_tokens=thinking_tokens,
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
        return WaveMetric(
            wave_name=wave_name,
            model_name=model.name,
            model_family=model.normalized_family,
            model_digest=model.digest,
            calls=len(wave_calls),
            thinking_token_cap=self._cap,
            thinking_tokens_total=sum(e.thinking_tokens for e in wave_calls),
            max_thinking_tokens_observed=max(
                (e.thinking_tokens for e in wave_calls), default=0
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
    thinking_token_cap: int = DEFAULT_THINKING_TOKEN_CAP,
    min_pass_rate_per_class: float = DEFAULT_MIN_PASS_RATE_PER_CLASS,
) -> PilotBWaveRunResult:
    """Execute Pilot B in 4 strict single-model waves: render -> word -> reader_1 -> reader_2."""
    validate_chain_families(writer_model, reader_1.model, reader_2.model)
    for t in triads:
        verify_triad_against_table(t.table, t)

    tracker = WaveCallTracker(thinking_token_cap=thinking_token_cap)

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
            thinking_tokens=0,
            completion_tokens=0,
            seconds=0.0,
        )
        dummy_r2 = BlindReadResult(
            claim_id="",
            reader=reader_2.model,
            verdict="ambiguous",
            thinking_tokens=0,
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
                    thinking_tokens=0,
                    completion_tokens=0,
                    seconds=0.0,
                ),
                read_2=BlindReadResult(
                    claim_id=cid,
                    reader=dummy_r2.reader,
                    verdict="ambiguous",
                    thinking_tokens=0,
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
                thinking_tokens=(
                    last_rec.thinking_tokens
                    if last_rec is not None
                    else min(thinking_token_cap, 32)
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
            last_rec.thinking_tokens
            if last_rec is not None
            else min(thinking_token_cap, 32)
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
            thinking_tokens=est_thinking,
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
                    thinking_tokens=(
                        last_rec.thinking_tokens
                        if last_rec is not None
                        else min(thinking_token_cap, 24)
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
                last_rec.thinking_tokens
                if last_rec is not None
                else min(thinking_token_cap, 24)
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
                thinking_tokens=est_thinking,
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
                    thinking_tokens=0,
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
                thinking_tokens=res1.thinking_tokens,
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
                    thinking_tokens=0,
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
                thinking_tokens=res2.thinking_tokens,
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
            "render", writer_model, adapter=w_cls, unloaded=render_unloaded, max_num_ctx=w_max_ctx
        ),
        tracker.summarize_wave(
            "word", writer_model, adapter=w_cls, unloaded=word_unloaded, max_num_ctx=w_max_ctx
        ),
        tracker.summarize_wave(
            "reader_1", reader_1.model, adapter=r1_cls, unloaded=r1_unloaded, max_num_ctx=r1_max_ctx
        ),
        tracker.summarize_wave(
            "reader_2", reader_2.model, adapter=r2_cls, unloaded=r2_unloaded, max_num_ctx=r2_max_ctx
        ),
    )

    return PilotBWaveRunResult(
        writer_model=writer_model,
        reader_1_model=reader_1.model,
        reader_2_model=reader_2.model,
        thinking_token_cap=thinking_token_cap,
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
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_line(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_sealed_part(
    *,
    part_dir: Path,
    keys_sealed_path: Path,
    part_id: str,
    map_version: int,
    map_sha256: str,
    wave_result: PilotBWaveRunResult,
    shuffle_seed: int,
    licence_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a sealed Pilot B part directory with shuffled items, aggregate-only judgment provenance, and external keys.sealed.jsonl."""
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
            "triad_id": entry["triad_id"],
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
                "trace_sha256": entry["proof"]["trace_sha256"],
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
                "thinking_token_cap": wave_result.thinking_token_cap,
                "max_num_ctx": int(
                    getattr(
                        wave_result.writer_adapter,
                        "max_num_ctx",
                        wave_result.writer_model.max_num_ctx,
                    )
                ),
                "call_records": entry["writer_call_records"],
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
                    "call_records": entry["reader_call_records"],
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
            "key": entry["key"],
            "proof": entry["proof"],
            "blind_reads": [entry["read_1"], entry["read_2"]],
        }
        sealed_key_lines.append(_canonical_json_line(sealed_row))

    items_bytes = ("\n".join(items_lines) + ("\n" if items_lines else "")).encode("utf-8")
    prov_bytes = ("\n".join(prov_lines) + ("\n" if prov_lines else "")).encode("utf-8")
    sealed_bytes = (
        "\n".join(sealed_key_lines) + ("\n" if sealed_key_lines else "")
    ).encode("utf-8")

    (part_dir / "items.jsonl").write_bytes(items_bytes)
    (part_dir / "provenance.jsonl").write_bytes(prov_bytes)
    keys_sealed_path.write_bytes(sealed_bytes)

    keys_sealed_sha256 = _sha256_bytes(sealed_bytes)

    part_meta: dict[str, Any] = {
        "part_id": part_id,
        "split": "sealed",
        "frozen": False,
        "map_version": map_version,
        "map_sha256": map_sha256,
        "shuffle_seed": shuffle_seed,
        "thinking_token_cap": wave_result.thinking_token_cap,
        "items_count": len(items_lines),
        "triads_kept": len(wave_result.kept_triads),
        "triads_failed": len(wave_result.failed_triads),
        "keys_sealed_sha256": keys_sealed_sha256,
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


def _flatten_scalar_fields(
    obj: Mapping[str, Any], prefix: str = ""
) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in obj.items():
        if k in ("id", "triad_id", "claim", "material", "messages"):
            continue
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, Mapping):
            out.update(_flatten_scalar_fields(v, full_key))
        elif isinstance(v, (str, int, float, bool)) or v is None:
            if not full_key.endswith("sha256"):
                out[full_key] = json.dumps(v, sort_keys=True)
    return out


def verify_sealed_part(part_dir: Path, keys_sealed_path: Path) -> None:
    """Verify that no file in part_dir contains keys.sealed.jsonl, forbidden field names, or per-item key-predicting values (ruling 5a)."""
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

    # Load sealed keys map: item_id -> key
    item_to_key: dict[str, str] = {}
    for raw_line in keys_sealed_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        k_row = json.loads(raw_line)
        item_to_key[str(k_row["id"])] = str(k_row["key"])

    distinct_keys = set(item_to_key.values())

    # Inspect every file in part_dir
    for file_path in sorted(part_dir.iterdir()):
        if not file_path.is_file() or file_path.name == "checksums.txt":
            continue
        if file_path.suffix == ".json":
            parsed = json.loads(file_path.read_text(encoding="utf-8"))
            _scan_object_for_forbidden_keys(parsed, file_path.name)
        elif file_path.suffix == ".jsonl":
            rows: list[dict[str, Any]] = []
            for line_no, raw_line in enumerate(
                file_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                _scan_object_for_forbidden_keys(row, f"{file_path.name}:{line_no}")
                rows.append(row)

            # Content-based sealing check (ruling 5a): join rows with item_id against item_to_key
            joined_rows = [
                r for r in rows if isinstance(r.get("id"), str) and r["id"] in item_to_key
            ]
            if len(joined_rows) >= 3 and len(distinct_keys) >= 2:
                field_names: set[str] = set()
                flattened_by_id: dict[str, dict[str, str]] = {}
                for r in joined_rows:
                    rid = str(r["id"])
                    flat = _flatten_scalar_fields(r)
                    flattened_by_id[rid] = flat
                    field_names |= set(flat.keys())

                for fname in sorted(field_names):
                    val_to_keys: dict[str, set[str]] = {}
                    for rid, flat in flattened_by_id.items():
                        val = flat.get(fname, "<missing>")
                        val_to_keys.setdefault(val, set()).add(item_to_key[rid])

                    # If a field takes >= 2 distinct values and every value maps deterministically to a single key class,
                    # then that field leaks the key (or a class partition of the key) by content!
                    if len(val_to_keys) >= 2 and all(
                        len(kset) == 1 for kset in val_to_keys.values()
                    ):
                        raise refused_error(
                            "PILOT_B_SEALED_CONTENT_LEAK",
                            f"Field '{fname}' in '{file_path.name}' has per-item values that deterministically "
                            "predict the sealed key (ruling 5a content-sealing violation).",
                            "Remove per-item key-derived fields from public files; report aggregates only.",
                        )


def freeze_sealed_part(part_dir: Path, keys_sealed_path: Path) -> dict[str, Any]:
    """Run all sealed-part checks, mark part.json frozen=True, and update checksums.txt."""
    verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_sealed_path)
    part_json_path = part_dir / "part.json"
    part_meta: dict[str, Any] = json.loads(part_json_path.read_text(encoding="utf-8"))
    part_meta["frozen"] = True
    part_json_path.write_bytes(
        (json.dumps(part_meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
    )
    _write_checksums(part_dir)
    return part_meta
