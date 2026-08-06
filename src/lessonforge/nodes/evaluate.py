"""Evaluate node — pure function (RunState, LLMGateway) → RunState.

Orchestrates:
  1. Deterministic checks (no LLM)
  2. All 4 LLM judges
  3. Verdict aggregation (SHIP / RETRY / ESCALATE)

Writes to: state.structural_report, state.verdict
"""

from __future__ import annotations

import logging

from lessonforge.config import AppConfig
from lessonforge.llm.gateway import Gateway as LLMGateway
from lessonforge.state import RunState, StructuralReport
from lessonforge.validators.deterministic import compute_metrics
from lessonforge.validators.deterministic import run_all as run_deterministic
from lessonforge.validators.judges import run_all_judges
from lessonforge.validators.verdict import aggregate

logger = logging.getLogger(__name__)

_NODE_NAME = "evaluate"


def run(state: RunState, gateway: LLMGateway, config: AppConfig | None = None) -> RunState:
    """Run the full evaluation pipeline and write results to state.

    Args:
        state:   RunState with state.lesson, state.blueprint, state.grounding populated.
        gateway: LLMGateway instance.
        config:  AppConfig.

    Returns:
        Updated RunState with state.structural_report and state.verdict populated.
    """
    cfg = config or AppConfig()

    logger.info("[%s] node start — run_id=%s attempt=%d", _NODE_NAME, state.run_id, state.attempt)

    if state.lesson is None:
        raise ValueError("Evaluate node requires state.lesson. Run generator node first.")

    lesson = state.lesson

    # ── Step 1: Deterministic checks ─────────────────────────────────────────
    det_results = run_deterministic(lesson, state.blueprint, cfg)
    logger.info("[%s] Deterministic checks: %d results", _NODE_NAME, len(det_results))

    # ── Step 2: LLM judges ────────────────────────────────────────────────────
    judge_results = run_all_judges(
        lesson=lesson,
        gateway=gateway,
        grounding=state.grounding,
        persona=state.persona,
        run_id=state.run_id,
        attempt=state.attempt,
        config=cfg,
    )
    logger.info("[%s] Judge results: %d results", _NODE_NAME, len(judge_results))

    # ── Step 3: Combine and compute metrics ───────────────────────────────────
    all_results = det_results + judge_results
    metrics = compute_metrics(lesson)

    structural_report = StructuralReport(
        results=all_results,
        metrics=metrics,
    )

    # ── Step 4: Aggregate verdict ─────────────────────────────────────────────
    verdict = aggregate(all_results, state.attempt, cfg)

    logger.info(
        "[%s] node end — verdict=%s hard_fails=%d",
        _NODE_NAME,
        verdict.ship_decision,
        len(verdict.hard_fails),
    )

    return state.model_copy(update={
        "structural_report": structural_report,
        "verdict": verdict,
    })
