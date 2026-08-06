"""Blueprint node — pure function (RunState) → RunState.

What this node does:
  1. Reads the grounding pack from state.grounding (set by the ground node).
  2. Loads the blueprint prompt from prompts/blueprint.md (records its SHA in trace).
  3. Calls the LLM gateway with role="blueprint".
  4. Validates and parses the response into a LessonBlueprint.
  5. Writes the result into state.blueprint and returns the updated state.

Contract:
  - Input:  state.grounding is not None (set by ground node).
  - Output: state.blueprint is a valid LessonBlueprint.
  - Fail:   raises MalformedOutputError (caught upstream by LangGraph orchestrator).
  - No side-effects other than the gateway trace log.
"""

from __future__ import annotations

import json
import logging

from lessonforge.config import AppConfig
from lessonforge.llm.gateway import Gateway as LLMGateway
from lessonforge.llm.schemas import LLMRequest
from lessonforge.prompts.loader import load_prompt
from lessonforge.state import LessonBlueprint, RunState

logger = logging.getLogger(__name__)

_NODE_NAME = "blueprint"


def run(state: RunState, gateway: LLMGateway, config: AppConfig | None = None) -> RunState:
    """Generate a LessonBlueprint from the grounding pack.

    This is a pure-ish function: same inputs → same structure of outputs
    (modulo LLM non-determinism when temperature > 0).

    Args:
        state:   The current RunState (must have state.grounding set).
        gateway: The LLMGateway instance (injected for testability).
        config:  AppConfig (optional; defaults to a fresh AppConfig()).

    Returns:
        Updated RunState with state.blueprint populated.
    """
    cfg = config or AppConfig()

    logger.info("[%s] node start — run_id=%s attempt=%d", _NODE_NAME, state.run_id, state.attempt)

    if state.grounding is None:
        raise ValueError(
            "Blueprint node requires state.grounding to be set. "
            "Run the ground node first."
        )

    # ── Build grounding context string ───────────────────────────────────────
    grounding_context = _format_grounding(state)

    # ── Build persona description string ─────────────────────────────────────
    persona_description = _format_persona(state)

    # ── Load prompt (SHA recorded by gateway via the request) ────────────────
    system_text, user_text, prompt_sha = load_prompt(
        role="blueprint",
        variables={
            "topic": state.topic,
            "persona_description": persona_description,
            "grounding_context": grounding_context,
        },
    )

    # ── Build and fire the LLM request ───────────────────────────────────────
    model_cfg = cfg.model("blueprint")
    request = LLMRequest(
        node=_NODE_NAME,
        run_id=state.run_id,
        attempt=state.attempt,
        role="blueprint",
        prompt_name="blueprint",
        prompt_sha=prompt_sha,
        model_override=str(model_cfg.get("id", "groq/llama-3.1-70b-versatile")),
        temperature_override=float(model_cfg.get("temperature", 0.4)),
        seed=cfg.seed,
        system=system_text,
        user=user_text,
        response_schema={},  # blueprint node validates its own output via Pydantic
    )

    response = gateway.call(request)

    # ── Parse and validate the blueprint payload ──────────────────────────────
    blueprint = _parse_blueprint(response.payload)

    logger.info(
        "[%s] node end — objectives=%d terms=%d cost_usd=%.6f",
        _NODE_NAME,
        len(blueprint.learning_objectives),
        len(blueprint.must_define_terms),
        response.cost_usd,
    )

    # ── Update state ──────────────────────────────────────────────────────────
    return state.model_copy(update={"blueprint": blueprint})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_grounding(state: RunState) -> str:
    """Render the grounding pack as a numbered list of passages for the prompt."""
    assert state.grounding is not None
    parts: list[str] = []
    for i, chunk in enumerate(state.grounding.chunks, start=1):
        header = f"[{chunk.id}] {chunk.title}"
        parts.append(f"{i}. {header}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def _format_persona(state: RunState) -> str:
    """Render the learner persona as a concise description string for the prompt."""
    p = state.persona
    return (
        f"{p.education}. "
        f"English level: {p.english_level}. "
        f"Unknown terms: {', '.join(p.unknown_terms[:6])}. "
        f"Motivation: {p.motivation}. "
        f"Reading budget: {p.reading_budget_minutes} minutes."
    )


def _parse_blueprint(payload: dict) -> LessonBlueprint:  # type: ignore[type-arg]
    """Parse the raw LLM payload dict into a validated LessonBlueprint.

    Raises MalformedOutputError (via Pydantic ValidationError propagation)
    if the payload does not conform to the schema.
    """
    from pydantic import ValidationError

    from lessonforge.errors import MalformedOutputError

    try:
        return LessonBlueprint.model_validate(payload)
    except (ValidationError, KeyError, TypeError, json.JSONDecodeError) as exc:
        short_payload = json.dumps(payload, default=str)[:400]
        raise MalformedOutputError(
            node=_NODE_NAME,
            validation_error=f"{exc} | payload={short_payload}",
        ) from exc
