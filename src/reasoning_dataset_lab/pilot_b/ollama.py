"""Part A: Real loopback Ollama writer adapter for Pilot B (Card gk Part A & Card gn).

Relies on the official Ollama HTTP API (`https://github.com/ollama/ollama/blob/main/docs/api.md`):
- `GET /api/tags` (`https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models`):
  reads `models[].name`, `models[].model`, and `models[].digest` to verify that the local model's
  digest equals the pinned `manifest_digest` before the first chat call. Never pulls (`/api/pull`
  is never called).
- `POST /api/chat` (`https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion`):
  sends `model` (pinned `ollama_tag`), `messages`, top-level `think` (`docs/api.md:499`), `stream: false`,
  and `options: {"temperature": 0, "num_predict": <cap + answer_budget>}` (`docs/api.md:393`, `:513`);
  reads `message.content`, `message.thinking` (`docs/api.md:505`), `done_reason` (`docs/api.md:658`),
  `eval_count`, and `thinking_tokens` / `thinking_eval_count`.
- `POST /api/chat` with `keep_alive: 0` (`https://github.com/ollama/ollama/blob/main/docs/api.md#unload-a-model`):
  unloads the model from VRAM after each wave.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import urlparse

from reasoning_dataset_lab.errors import refused_error
from reasoning_dataset_lab.pilot_b.fact_tables import (
    FactTable,
    ProvedClaim,
)
from reasoning_dataset_lab.pilot_b.rendering import (
    check_rendered_material,
    check_worded_claim,
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
DEFAULT_THINKING_TOKEN_CAP: int = 256

OllamaOperation = Literal["render_material", "word_claim", "read_claim"]
OllamaHttpTransport = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]

# Named answer token budgets per operation (Card gn FIX 1: options.num_predict = thinking_token_cap + answer_budget)
RENDER_ANSWER_BUDGET: int = 256
WORD_ANSWER_BUDGET: int = 96
READ_ANSWER_BUDGET: int = 32
RENDER_MATERIAL_ANSWER_BUDGET: int = RENDER_ANSWER_BUDGET
WORD_CLAIM_ANSWER_BUDGET: int = WORD_ANSWER_BUDGET
READ_CLAIM_ANSWER_BUDGET: int = READ_ANSWER_BUDGET

OPERATION_ANSWER_BUDGETS: dict[OllamaOperation, int] = {
    "render_material": RENDER_ANSWER_BUDGET,
    "word_claim": WORD_ANSWER_BUDGET,
    "read_claim": READ_ANSWER_BUDGET,
}

# Explicit num_ctx calculation and per-role ceilings (Card gp Part A):
# options.num_ctx = round_up(prompt_tokens_estimate + num_predict + margin, 1024)
PROMPT_TOKEN_ESTIMATE_RULE: str = "ceil(utf8_bytes / 3)"
NUM_CTX_MARGIN_TOKENS: int = 256
NUM_CTX_STEP_TOKENS: int = 1024

# Per-role max_num_ctx defaults planned so weights + KV stay at or below 26.0 GB (GPU-jobs rule):
# - writer (qwen3.6:35b, weights 22.6 GB, 3.4 GB under 26.0 GB): 8192
# - reader_1 (nemotron-3.5-lightning:30b, weights 25.4 GB, ~0.6 GB under 26.0 GB): 2048
# - reader_2 (granite4.2:8b, weights 5.3 GB, 20.7 GB under 26.0 GB): 8192
DEFAULT_ROLE_MAX_NUM_CTX: dict[str, int] = {
    "writer": 8192,
    "reader_1": 2048,
    "reader_2": 8192,
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
    """Per-call audit record for an Ollama writer or blind-reader invocation (Card gk Part A/B/C, Card gn, Card gp)."""

    model: str
    digest: str
    operation: OllamaOperation
    target_id: str
    prompt_sha256: str
    eval_tokens: int
    thinking_tokens: int
    thinking_token_cap: int
    wall_time_s: float
    think: ThinkSetting = True
    num_predict: int = 0
    num_ctx: int = 0
    max_num_ctx: int = 0
    prompt_tokens_estimate: int = 0
    done_reason: str = "stop"

    def to_dict(self, *, include_target_id: bool = True) -> dict[str, object]:
        out: dict[str, object] = {
            "model": self.model,
            "digest": self.digest,
            "operation": self.operation,
            "prompt_sha256": self.prompt_sha256,
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
            "think": self.think,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
            "max_num_ctx": self.max_num_ctx,
            "done_reason": self.done_reason,
            "eval_tokens": self.eval_tokens,
            "thinking_tokens": self.thinking_tokens,
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


def _extract_token_counts(
    resp: dict[str, Any],
    content: str,
    thinking_text: str,
) -> tuple[int, int]:
    raw_eval = resp.get("eval_count")
    if isinstance(raw_eval, int) and not isinstance(raw_eval, bool) and raw_eval >= 0:
        eval_tokens = raw_eval
    else:
        eval_tokens = len(content.split())

    raw_think = resp.get("thinking_tokens", resp.get("thinking_eval_count"))
    if isinstance(raw_think, int) and not isinstance(raw_think, bool) and raw_think >= 0:
        thinking_tokens = raw_think
    else:
        thinking_tokens = len(thinking_text.split()) if thinking_text.strip() else 0

    return eval_tokens, thinking_tokens


class OllamaWriterAdapter:
    """Real loopback Ollama writer adapter implementing `PilotBWriterAdapter` (Card gk Part A & Card gp Part A)."""

    is_fake: ClassVar[bool] = False

    def __init__(
        self,
        model: ModelIdentity | PinnedModelSpec,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        thinking_token_cap: int = DEFAULT_THINKING_TOKEN_CAP,
        max_num_ctx: int | None = None,
        http_transport: OllamaHttpTransport | None = None,
    ) -> None:
        if thinking_token_cap < 1:
            raise refused_error(
                "PILOT_B_INVALID_THINKING_CAP",
                f"thinking_token_cap must be >= 1, got {thinking_token_cap}.",
                "Pass a positive integer thinking_token_cap.",
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
        self._thinking_token_cap = thinking_token_cap
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
    def thinking_token_cap(self) -> int:
        return self._thinking_token_cap

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
        num_predict = self._thinking_token_cap + OPERATION_ANSWER_BUDGETS[operation]
        num_ctx = compute_num_ctx(
            prompt_tokens_estimate,
            num_predict,
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
                    thinking_tokens=0,
                    thinking_token_cap=self._thinking_token_cap,
                    wall_time_s=0.0,
                    think=self._think,
                    num_predict=num_predict,
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
                f"(prompt_tokens_estimate={prompt_tokens_estimate} + num_predict={num_predict} + "
                f"margin={NUM_CTX_MARGIN_TOKENS}), which exceeds max_num_ctx={self._max_num_ctx} "
                f"for '{self.ollama_tag}'.",
                "Reduce thinking_token_cap or prompt size, or configure a higher max_num_ctx if VRAM permits.",
            )

        chat_url = f"{self._base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.ollama_tag,
            "messages": messages,
            "think": self._think,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": num_predict,
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
        eval_tokens, thinking_tokens = _extract_token_counts(resp, content, thinking_text)

        record = OllamaCallRecord(
            model=self.ollama_tag,
            digest=self.manifest_digest,
            operation=operation,
            target_id=target_id,
            prompt_sha256=prompt_sha,
            eval_tokens=eval_tokens,
            thinking_tokens=thinking_tokens,
            thinking_token_cap=self._thinking_token_cap,
            wall_time_s=wall_s,
            think=self._think,
            num_predict=num_predict,
            num_ctx=num_ctx,
            max_num_ctx=self._max_num_ctx,
            prompt_tokens_estimate=prompt_tokens_estimate,
            done_reason=done_reason,
        )
        self._call_records.append(record)
        self._last_raw_output = content

        if done_reason == "length":
            raise refused_error(
                "PILOT_B_OLLAMA_LENGTH_STOP",
                f"Writer call '{target_id}' ({operation}) stopped with done_reason='length' "
                f"(num_predict={num_predict}).",
                "Increase thinking_token_cap or answer_budget.",
            )
        return content

    def render_material(self, table: FactTable) -> str:
        """Render `table` into numbered prose via `POST /api/chat` and run `check_rendered_material`."""
        table_json = json.dumps(
            table.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Pilot B writer. Render each fact in the formal FactTable into one numbered "
                    "sentence per line ('1. ...', '2. ...'), preserving exact entities, attributes, numbers, "
                    "dates, intervals, polarity, and quantifier direction. Introduce no extra numbers or proper nouns."
                ),
            },
            {
                "role": "user",
                "content": f"Render the following formal FactTable into numbered lines.\nFACTS_JSON:\n{table_json}",
            },
        ]
        rendered = self._chat(
            operation="render_material",
            target_id=table.table_id,
            messages=messages,
        )
        check_rendered_material(table, rendered)
        return rendered

    def word_claim(self, table: FactTable, claim: ProvedClaim) -> str:
        """Word `claim` into a single sentence via `POST /api/chat` and run `check_worded_claim`."""
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
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Pilot B writer. Word the formal claim into a single declarative English sentence "
                    "preserving its exact entity, attribute, value, unit, polarity, and quantifier direction. "
                    "Introduce no unlisted numbers or proper nouns."
                ),
            },
            {
                "role": "user",
                "content": f"Word the following formal ProvedClaim into one sentence.\nCLAIM_JSON:\n{claim_json}",
            },
        ]
        worded = self._chat(
            operation="word_claim",
            target_id=claim.claim_id,
            messages=messages,
        )
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
    """Real loopback Ollama blind-reader adapter implementing `BlindReaderAdapter` (Card gk Part B & Card gp Part A)."""

    is_fake: ClassVar[bool] = False

    def __init__(
        self,
        model: ModelIdentity | PinnedModelSpec,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        thinking_token_cap: int = DEFAULT_THINKING_TOKEN_CAP,
        max_num_ctx: int | None = None,
        http_transport: OllamaHttpTransport | None = None,
    ) -> None:
        if thinking_token_cap < 1:
            raise refused_error(
                "PILOT_B_INVALID_THINKING_CAP",
                f"thinking_token_cap must be >= 1, got {thinking_token_cap}.",
                "Pass a positive integer thinking_token_cap.",
            )
        self._base_url = require_loopback_ollama_url(base_url)
        self._model = _resolve_model_identity(model)
        default_role_max = (
            DEFAULT_ROLE_MAX_NUM_CTX["reader_1"]
            if self._model.normalized_family == "nemotron"
            else DEFAULT_ROLE_MAX_NUM_CTX["reader_2"]
        )
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
        self._thinking_token_cap = thinking_token_cap
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
    def thinking_token_cap(self) -> int:
        return self._thinking_token_cap

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
        num_predict = self._thinking_token_cap + READ_ANSWER_BUDGET
        num_ctx = compute_num_ctx(
            prompt_tokens_estimate,
            num_predict,
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
                thinking_tokens=0,
                thinking_token_cap=self._thinking_token_cap,
                wall_time_s=0.0,
                think=self._think,
                num_predict=num_predict,
                num_ctx=num_ctx,
                max_num_ctx=self._max_num_ctx,
                prompt_tokens_estimate=prompt_tokens_estimate,
                done_reason="max_num_ctx_exceeded",
            )
            self._call_records.append(record)
            raise refused_error(
                "PILOT_B_NUM_CTX_EXCEEDS_MAX",
                f"Reader call '{target_id}' (read_claim) requires num_ctx={num_ctx} "
                f"(prompt_tokens_estimate={prompt_tokens_estimate} + num_predict={num_predict} + "
                f"margin={NUM_CTX_MARGIN_TOKENS}), which exceeds max_num_ctx={self._max_num_ctx} "
                f"for '{self.ollama_tag}'.",
                "Reduce thinking_token_cap or prompt size, or configure a higher max_num_ctx if VRAM permits.",
            )

        chat_url = f"{self._base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.ollama_tag,
            "messages": messages,
            "think": self._think,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": num_predict,
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
        eval_tokens, thinking_tokens = _extract_token_counts(resp, raw_text, thinking_text)

        record = OllamaCallRecord(
            model=self.ollama_tag,
            digest=self.manifest_digest,
            operation="read_claim",
            target_id=target_id,
            prompt_sha256=prompt_sha,
            eval_tokens=eval_tokens,
            thinking_tokens=thinking_tokens,
            thinking_token_cap=self._thinking_token_cap,
            wall_time_s=wall_s,
            think=self._think,
            num_predict=num_predict,
            num_ctx=num_ctx,
            max_num_ctx=self._max_num_ctx,
            prompt_tokens_estimate=prompt_tokens_estimate,
            done_reason=done_reason,
        )
        self._call_records.append(record)
        return raw_text, record

    def read_claim(self, prompt: BlindReadPrompt) -> BlindReadResult:
        """Send only `prompt` (material + one claim, never the key) to `POST /api/chat` and parse the verdict."""
        messages = [
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
            thinking_tokens=record.thinking_tokens,
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

