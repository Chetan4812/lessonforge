"""Repair node — pure function (RunState, LLMGateway) → RunState.

What this node does:
  1. Reads state.verdict.hard_fails to build a RepairPlan (via repair_planner).
  2. Loads prompts/repair.md, renders the repair plan + original lesson into it.
  3. Calls the gateway with role="generator" (same model, same output schema).
  4. Validates the repaired lesson.
  5. Increments attempt counter, updates state.lesson and lesson_history.
  6. Writes state.repair_plan and appends to state.rejection_log.

Contract:
  - Input:  state.verdict is not None (verdict.ship_decision must be RETRY).
  - Input:  state.lesson is not None (the lesson that failed evaluation).
  - Output: state.lesson is the repaired lesson, state.attempt incremented.
  - Output: state.repair_plan is set.
  - Fail:   raises ValueError if preconditions not met.
"""

from __future__ import annotations

import json
import logging

from lessonforge.config import AppConfig
from lessonforge.llm.gateway import Gateway as LLMGateway
from lessonforge.llm.schemas import LLMRequest
from lessonforge.nodes.repair_planner import plan as build_plan
from lessonforge.prompts.loader import load_prompt
from lessonforge.state import Lesson, RepairPlan, RunState

logger = logging.getLogger(__name__)

_NODE_NAME = "repair"


def run(state: RunState, gateway: LLMGateway, config: AppConfig | None = None) -> RunState:
    """Execute the repair loop for one attempt.

    Args:
        state:   RunState with state.verdict (RETRY) and state.lesson.
        gateway: LLMGateway instance.
        config:  AppConfig.

    Returns:
        Updated RunState with:
        - state.lesson = repaired lesson
        - state.lesson_history appended
        - state.attempt incremented
        - state.repair_plan set
        - state.rejection_log appended with the failed lesson + reasons
    """
    cfg = config or AppConfig()

    logger.info("[%s] node start — run_id=%s attempt=%d", _NODE_NAME, state.run_id, state.attempt)

    # ── Precondition checks ───────────────────────────────────────────────────
    if state.verdict is None:
        raise ValueError("Repair node requires state.verdict to be set. Run evaluate node first.")
    if state.verdict.ship_decision != "RETRY":
        raise ValueError(
            f"Repair node called but verdict is '{state.verdict.ship_decision}' — "
            "only RETRY verdicts should trigger repair."
        )
    if state.lesson is None:
        raise ValueError("Repair node requires state.lesson to be set.")

    # ── Build the repair plan (deterministic, no LLM) ─────────────────────────
    repair_plan = build_plan(
        hard_fails=state.verdict.hard_fails,
        lesson=state.lesson,
        attempt=state.attempt,
    )

    # ── Log the failed lesson to the rejection_log ────────────────────────────
    rejection_entry = {
        "attempt": state.attempt,
        "hard_fails": [
            {"check_id": r.check_id, "reason": r.reason}
            for r in state.verdict.hard_fails
        ],
        "lesson_word_count": _word_count(state.lesson),
    }

    # ── Render the repair prompt ──────────────────────────────────────────────
    system_text, user_text, prompt_sha = _render_prompt(state, repair_plan, cfg)

    # ── Call the LLM gateway ──────────────────────────────────────────────────
    new_attempt = state.attempt + 1
    model_cfg = cfg.model("generator")  # repair uses the same model as generator
    request = LLMRequest(
        node=_NODE_NAME,
        run_id=state.run_id,
        attempt=new_attempt,
        role="generator",
        prompt_name="repair",
        prompt_sha=prompt_sha,
        model_override=str(model_cfg.get("id", "groq/llama-3.1-70b-versatile")),
        temperature_override=float(model_cfg.get("temperature", 0.6)),
        seed=cfg.seed,
        system=system_text,
        user=user_text,
        response_schema={},
    )

    response = gateway.call(request)

    # ── Validate the repaired lesson ──────────────────────────────────────────
    from pydantic import ValidationError

    from lessonforge.errors import MalformedOutputError

    try:
        repaired_lesson = Lesson.model_validate(response.payload)
    except (ValidationError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise MalformedOutputError(
            node=_NODE_NAME,
            validation_error=f"Repaired lesson failed Pydantic validation: {exc}",
        ) from exc

    new_history = list(state.lesson_history) + [repaired_lesson]
    new_rejection_log = list(state.rejection_log) + [rejection_entry]

    logger.info(
        "[%s] node end — repaired lesson sections=%d new_attempt=%d",
        _NODE_NAME,
        len(repaired_lesson.sections),
        new_attempt,
    )

    return state.model_copy(update={
        "lesson": repaired_lesson,
        "lesson_history": new_history,
        "attempt": new_attempt,
        "repair_plan": repair_plan,
        "rejection_log": new_rejection_log,
        # Clear stale evaluation results — next evaluate node will repopulate
        "structural_report": None,
        "verdict": None,
    })


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_prompt(
    state: RunState,
    repair_plan: RepairPlan,
    cfg: AppConfig,
) -> tuple[str, str, str]:
    """Render the repair prompt with all required variables."""
    assert state.lesson is not None
    assert state.blueprint is not None

    bp = state.blueprint
    grounding_context = _format_grounding(state)

    # Format repair items for the prompt
    repair_items_text = _format_repair_items(repair_plan)
    keep_sections_text = (
        ", ".join(repair_plan.keep_sections)
        if repair_plan.keep_sections
        else "(none — full rewrite)"
    )

    learning_objectives = "\n".join(f"- {o}" for o in bp.learning_objectives)
    out_of_scope = "\n".join(f"- {s}" for s in bp.out_of_scope)

    return load_prompt(
        role="repair",
        variables={
            "strategy": repair_plan.strategy,
            "repair_items": repair_items_text,
            "keep_sections": keep_sections_text,
            "original_lesson": state.lesson.to_markdown(),
            "learning_objectives": learning_objectives,
            "central_analogy": bp.central_analogy,
            "out_of_scope": out_of_scope,
            "grounding_context": grounding_context,
        },
    )


def _format_repair_items(repair_plan: RepairPlan) -> str:
    if not repair_plan.items:
        return "(no specific items — see failed checks and use your judgment)"
    lines: list[str] = []
    for i, item in enumerate(repair_plan.items, start=1):
        lines.append(
            f"{i}. Section [{item.section_key}]\n"
            f"   Problem: {item.problem}\n"
            f"   Fix: {item.instruction}\n"
            f"   Triggered by: {', '.join(item.triggering_checks)}"
        )
    return "\n\n".join(lines)


def _format_grounding(state: RunState) -> str:
    if state.grounding is None:
        return "(no grounding pack)"
    parts = [f"[{c.id}] {c.title}\n{c.text}" for c in state.grounding.chunks]
    return "\n\n---\n\n".join(parts)


def _word_count(lesson: Lesson) -> int:
    return len(" ".join(s.body_md for s in lesson.sections).split())
