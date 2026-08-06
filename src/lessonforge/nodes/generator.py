"""Generator node — pure function (RunState, LLMGateway) → RunState.

What this node does:
  1. Reads state.blueprint and state.grounding (both must be set).
  2. Loads prompts/generator.md, renders blueprint + persona + grounding into it.
  3. Calls the gateway with role="generator".
  4. Validates the response into a Lesson (all 11 sections present and non-empty).
  5. Appends the lesson to state.lesson_history, sets state.lesson, returns new state.

Contract:
  - Input:  state.blueprint is not None, state.grounding is not None.
  - Output: state.lesson is a valid Lesson with all 11 required sections.
  - Fail:   raises MalformedOutputError if the model returns invalid/incomplete output.
"""

from __future__ import annotations

import json
import logging

from lessonforge.config import AppConfig
from lessonforge.errors import MalformedOutputError
from lessonforge.llm.gateway import Gateway as LLMGateway
from lessonforge.llm.schemas import LLMRequest
from lessonforge.prompts.loader import load_prompt
from lessonforge.state import Lesson, RunState, SectionKey

logger = logging.getLogger(__name__)

_NODE_NAME = "generate"

_REQUIRED_SECTIONS: list[SectionKey] = [
    "hook",
    "what_it_is",
    "why_it_matters",
    "how_it_works",
    "analogy",
    "worked_example",
    "common_mistakes",
    "glossary",
    "recap",
    "check_yourself",
    "next_steps",
]


def run(state: RunState, gateway: LLMGateway, config: AppConfig | None = None) -> RunState:
    """Generate the full lesson from the blueprint and grounding pack.

    Args:
        state:   RunState with state.blueprint and state.grounding populated.
        gateway: LLMGateway instance (injected for testability).
        config:  AppConfig (defaults to a fresh AppConfig()).

    Returns:
        Updated RunState with state.lesson and updated state.lesson_history.
    """
    cfg = config or AppConfig()

    logger.info("[%s] node start — run_id=%s attempt=%d", _NODE_NAME, state.run_id, state.attempt)

    if state.blueprint is None:
        raise ValueError("Generator node requires state.blueprint. Run blueprint node first.")
    if state.grounding is None:
        raise ValueError("Generator node requires state.grounding. Run ground node first.")

    bp = state.blueprint

    # ── Render blueprint fields for the prompt ────────────────────────────────
    learning_objectives = "\n".join(f"- {o}" for o in bp.learning_objectives)
    must_define_terms = ", ".join(bp.must_define_terms)
    out_of_scope = "\n".join(f"- {s}" for s in bp.out_of_scope)
    section_plan = _format_section_plan(bp.section_plan)
    grounding_context = _format_grounding(state)
    persona_description = _format_persona(state)

    # ── Load and render the generator prompt ──────────────────────────────────
    system_text, user_text, prompt_sha = load_prompt(
        role="generator",
        variables={
            "topic": state.topic,
            "persona_description": persona_description,
            "learning_objectives": learning_objectives,
            "central_analogy": bp.central_analogy,
            "worked_example_scenario": bp.worked_example_scenario,
            "must_define_terms": must_define_terms,
            "out_of_scope": out_of_scope,
            "section_plan": section_plan,
            "grounding_context": grounding_context,
        },
    )

    # ── Call the LLM gateway ──────────────────────────────────────────────────
    model_cfg = cfg.model("generator")
    request = LLMRequest(
        node=_NODE_NAME,
        run_id=state.run_id,
        attempt=state.attempt,
        role="generator",
        prompt_name="generator",
        prompt_sha=prompt_sha,
        model_override=str(model_cfg.get("id", "groq/llama-3.1-70b-versatile")),
        temperature_override=float(model_cfg.get("temperature", 0.7)),
        seed=cfg.seed,
        system=system_text,
        user=user_text,
        response_schema={},
    )

    response = gateway.call(request)

    # ── Parse and validate the lesson payload ─────────────────────────────────
    lesson = _parse_lesson(response.payload)

    logger.info(
        "[%s] node end — sections=%d cost_usd=%.6f",
        _NODE_NAME,
        len(lesson.sections),
        response.cost_usd,
    )

    # ── Update state (immutably) ──────────────────────────────────────────────
    new_history = list(state.lesson_history) + [lesson]
    return state.model_copy(update={"lesson": lesson, "lesson_history": new_history})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_grounding(state: RunState) -> str:
    assert state.grounding is not None
    parts: list[str] = []
    for chunk in state.grounding.chunks:
        parts.append(f"[{chunk.id}] {chunk.title}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def _format_persona(state: RunState) -> str:
    p = state.persona
    return (
        f"{p.education}. "
        f"English level: {p.english_level}. "
        f"Unknown terms the reader does not know yet: {', '.join(p.unknown_terms[:6])}. "
        f"Motivation: {p.motivation}. "
        f"Reading budget: {p.reading_budget_minutes} minutes."
    )


def _format_section_plan(section_plan: list[dict]) -> str:  # type: ignore[type-arg]
    """Render section_plan entries as a numbered list for the prompt."""
    lines: list[str] = []
    for i, entry in enumerate(section_plan, start=1):
        key = entry.get("key", "?")
        heading = entry.get("heading", entry.get("note", ""))
        goal = entry.get("one_line_goal", "")
        if goal:
            lines.append(f"{i}. [{key}] \"{heading}\" — {goal}")
        else:
            lines.append(f"{i}. [{key}] \"{heading}\"")
    return "\n".join(lines)


def _parse_lesson(payload: dict) -> Lesson:  # type: ignore[type-arg]
    """Parse and validate the LLM payload into a Lesson.

    Validates:
    - Topic and title are present.
    - All 11 required section keys are present.
    - No section has empty body_md.

    Raises:
        MalformedOutputError on any validation failure.
    """
    from pydantic import ValidationError

    try:
        lesson = Lesson.model_validate(payload)
    except (ValidationError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise MalformedOutputError(
            node=_NODE_NAME,
            validation_error=f"Pydantic validation failed: {exc}",
        ) from exc

    # ── Check all required sections are present ───────────────────────────────
    found_keys = {s.key for s in lesson.sections}
    missing = [k for k in _REQUIRED_SECTIONS if k not in found_keys]
    if missing:
        raise MalformedOutputError(
            node=_NODE_NAME,
            validation_error=f"Missing required sections: {missing}",
        )

    # ── Check no section has empty body ───────────────────────────────────────
    empty = [s.key for s in lesson.sections if not s.body_md.strip()]
    if empty:
        raise MalformedOutputError(
            node=_NODE_NAME,
            validation_error=f"Sections with empty body_md: {empty}",
        )

    return lesson
