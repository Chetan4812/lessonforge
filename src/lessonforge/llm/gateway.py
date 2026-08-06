"""LLM Gateway — the ONLY entry point for model calls in the entire codebase.

Rules (from AGENTS.md):
  - Never call a model provider directly. Always go through llm/gateway.py.
  - No prompt text inside .py files.
  - All prompts live in prompts/*.md and are loaded by name with SHA recorded.

The gateway is responsible for:
  1. Loading the rendered prompt text (passed in from the caller node).
  2. Routing to the correct model based on config role.
  3. Making the actual call (real LiteLLM or mock provider).
  4. Handling API-level retries (tenacity).
  5. Handling structured-output failures (one re-ask, then hard fail).
  6. Tracking tokens and cost.
  7. Emitting a JSONL trace event for every call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from lessonforge.config import OUT_DIR, AppConfig
from lessonforge.errors import CostLimitExceededError, LLMError
from lessonforge.llm.cost import estimate_cost
from lessonforge.llm.mock_provider import MockProvider
from lessonforge.llm.retry import call_with_reask, make_api_retry
from lessonforge.llm.schemas import CostAccumulator, LLMRequest, LLMResponse, TraceEvent

logger = logging.getLogger(__name__)


class Gateway:
    """Model-agnostic LLM gateway.

    Args:
        provider: "groq" | "openai" | "anthropic" | "mock"
        config:   AppConfig instance (defaults to singleton)
        cost_accumulator: shared per-run accumulator; pass the same instance
                          to all gateway calls within a run.
        run_id:   used for trace file path.
    """

    def __init__(
        self,
        provider: str = "groq",
        config: AppConfig | None = None,
        cost_accumulator: CostAccumulator | None = None,
        run_id: str = "unknown",
    ) -> None:
        self._cfg = config or AppConfig()
        self._provider = provider
        self._run_id = run_id
        self._cost = cost_accumulator or CostAccumulator()
        self._mock: MockProvider | None = None
        self._api_retry = make_api_retry()

        if provider == "mock":
            self._mock = MockProvider()

    # ── Public interface ──────────────────────────────────────────────────────

    def call(self, request: LLMRequest) -> LLMResponse:
        """Make a single structured LLM call and return a typed response.

        This is the only method nodes should call.
        """
        # ── Check cost budget ─────────────────────────────────────────────
        limit = self._cfg.max_cost_usd_per_run
        if self._cost.total_usd >= limit:
            raise CostLimitExceededError(self._cost.total_usd, limit)

        # ── Resolve model config ──────────────────────────────────────────
        model_cfg = self._cfg.model(request.role)
        model_id = request.model_override or str(model_cfg["id"])
        temperature = (
            request.temperature_override
            if request.temperature_override is not None
            else float(model_cfg["temperature"])
        )
        seed = request.seed or self._cfg.seed

        self._emit_trace(
            TraceEvent(
                ts=_now(),
                run_id=request.run_id,
                attempt=request.attempt,
                node=request.node,
                event="llm_call_start",
                role=request.role,
                model_id=model_id,
            ),
            request.run_id,
        )

        start_ms = _ms()

        # ── Dispatch ──────────────────────────────────────────────────────
        if self._mock is not None:
            response = self._call_mock(request, model_id, seed)
        else:
            response = self._call_litellm(request, model_id, temperature, seed)

        response.duration_ms = _ms() - start_ms

        # ── Accumulate cost ───────────────────────────────────────────────
        self._cost.add(response)

        # ── Emit trace end event ──────────────────────────────────────────
        self._emit_trace(
            TraceEvent(
                ts=_now(),
                run_id=request.run_id,
                attempt=request.attempt,
                node=request.node,
                event="llm_call_end",
                role=request.role,
                model_id=model_id,
                duration_ms=response.duration_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cost_usd=response.cost_usd,
                payload_sha=hashlib.sha256(
                    json.dumps(response.payload, sort_keys=True).encode()
                ).hexdigest()[:12],
                reask_triggered=response.reask_triggered,
            ),
            request.run_id,
        )

        return response

    @property
    def cost(self) -> CostAccumulator:
        return self._cost

    @property
    def mock(self) -> MockProvider | None:
        """Expose the mock provider for test injection."""
        return self._mock

    # ── Mock dispatch ─────────────────────────────────────────────────────────

    def _call_mock(self, request: LLMRequest, model_id: str, seed: int | None) -> LLMResponse:
        assert self._mock is not None

        def _raw_call(system: str, user: str) -> str:
            return self._mock.call_raw_with_override(  # type: ignore[union-attr]
                role=request.role,
                system=system,
                user=user,
                model_id=model_id,
                seed=seed,
            )

        payload, reask = call_with_reask(
            call_fn=_raw_call,
            system=request.system,
            user=request.user,
            parse_fn=_parse_json,
            node=request.node,
        )

        # Mock has no real token counts — estimate from text length
        prompt_tokens = (len(request.system) + len(request.user)) // 4
        completion_tokens = len(json.dumps(payload)) // 4

        return LLMResponse(
            role=request.role,
            node=request.node,
            run_id=request.run_id,
            attempt=request.attempt,
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=0.0,  # mock calls are free
            duration_ms=0,
            payload=payload,
            raw_text=json.dumps(payload),
            reask_triggered=reask,
        )

    # ── LiteLLM dispatch ──────────────────────────────────────────────────────

    def _call_litellm(
        self,
        request: LLMRequest,
        model_id: str,
        temperature: float,
        seed: int | None,
    ) -> LLMResponse:
        import litellm

        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ]

        api_retry = make_api_retry()

        def _raw_call(system: str, user: str) -> str:
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

            def _attempt() -> str:
                try:
                    resp = litellm.completion(
                        model=model_id,
                        messages=msgs,
                        temperature=temperature,
                        seed=seed,
                        response_format={"type": "json_object"},
                    )
                    return str(resp.choices[0].message.content)
                except Exception as exc:
                    logger.error("[gateway] LiteLLM error: %s", exc)
                    raise LLMError(str(exc)) from exc

            return str(api_retry(_attempt)())

        payload, reask = call_with_reask(
            call_fn=_raw_call,
            system=request.system,
            user=request.user,
            parse_fn=_parse_json,
            node=request.node,
        )

        # Re-fetch token usage from the last successful call
        # We do one more call to get the real response object with usage
        try:
            resp = litellm.completion(
                model=model_id,
                messages=messages,
                temperature=temperature,
                seed=seed,
                response_format={"type": "json_object"},
            )
            usage = resp.usage
            prompt_tokens = int(usage.prompt_tokens or 0)
            completion_tokens = int(usage.completion_tokens or 0)
        except Exception:  # noqa: BLE001
            # Fall back to estimates if usage not available
            prompt_tokens = (len(request.system) + len(request.user)) // 4
            completion_tokens = len(json.dumps(payload)) // 4

        total_tokens = prompt_tokens + completion_tokens
        cost = estimate_cost(model_id, prompt_tokens, completion_tokens)

        return LLMResponse(
            role=request.role,
            node=request.node,
            run_id=request.run_id,
            attempt=request.attempt,
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            duration_ms=0,  # filled in by caller
            payload=payload,
            raw_text=json.dumps(payload),
            reask_triggered=reask,
        )

    # ── Trace ─────────────────────────────────────────────────────────────────

    def _emit_trace(self, event: TraceEvent, run_id: str) -> None:
        try:
            trace_dir = OUT_DIR / run_id
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / "trace.jsonl"
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("trace write failed: %s", exc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ms() -> int:
    return int(time.monotonic() * 1000)


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse raw model text as JSON.

    Strips markdown fences if present (some models still emit them despite
    response_format=json_object instructions).
    """
    text = raw.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```json or ```) and last line (```)
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return dict(json.loads(text))
