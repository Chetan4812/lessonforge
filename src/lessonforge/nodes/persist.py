"""Persist node — writes every run's data into the memory SQLite database.

Runs at the END of every pipeline execution (both SHIP and ESCALATE paths).
Never makes an LLM call.

Responsibilities:
  1. Insert the run outcome into `runs`.
  2. Insert all lesson attempt snapshots into `attempts`.
  3. Insert all CheckResults into `check_results`.
  4. Insert a passing first-try exemplar into `exemplars` if applicable.
  5. Upsert failure signatures into `failure_modes`.
  6. Auto-promote recurring failure modes into `guardrails` when threshold is met.

Writes to: SQLite db. Returns state unchanged (pass-through node).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from lessonforge.config import AppConfig
from lessonforge.memory import db
from lessonforge.state import RunState

logger = logging.getLogger(__name__)

_NODE_NAME = "persist"


def run(state: RunState, config: AppConfig | None = None) -> RunState:
    """Persist the run to SQLite memory and auto-promote guardrails.

    Args:
        state:  Final RunState (all nodes complete).
        config: AppConfig.

    Returns:
        The same RunState (this node is a pass-through to allow easy chaining).
    """
    cfg = config or AppConfig()
    promotion_threshold = int(cfg.memory.get("guardrail_promotion_threshold", 3))

    logger.info("[%s] node start — run_id=%s", _NODE_NAME, state.run_id)

    # ── Init DB ───────────────────────────────────────────────────────────────
    db.init_db()

    # ── Determine outcome ─────────────────────────────────────────────────────
    ship = state.verdict.ship_decision if state.verdict else "error"
    outcome = {"SHIP": "shipped", "ESCALATE": "escalated"}.get(ship, "error")

    # ── Step 1: Insert run ────────────────────────────────────────────────────
    passed_first_try = outcome == "shipped" and state.attempt == 1
    _safe_insert_run(state, outcome, passed_first_try)

    # ── Step 2: Insert attempts ───────────────────────────────────────────────
    _insert_attempts(state)

    # ── Step 3: Insert check results ──────────────────────────────────────────
    if state.structural_report:
        result_dicts = [r.model_dump() for r in state.structural_report.results]
        db.insert_check_results(
            run_id=state.run_id,
            attempt=state.attempt,
            results=result_dicts,
        )
        logger.info("[%s] inserted %d check results", _NODE_NAME, len(result_dicts))

    # ── Step 4: Insert exemplar if first-try pass ─────────────────────────────
    if passed_first_try and state.lesson is not None:
        db.insert_exemplar(
            run_id=state.run_id,
            topic=state.topic,
            lesson_md=state.lesson.to_markdown(),
            passed_first_try=True,
        )
        logger.info("[%s] exemplar stored for topic '%s'", _NODE_NAME, state.topic)

    # ── Step 5 + 6: Failure mode clustering + guardrail promotion ─────────────
    _cluster_and_promote(state, promotion_threshold)

    logger.info("[%s] node end — run_id=%s outcome=%s", _NODE_NAME, state.run_id, outcome)

    return state  # pass-through


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_insert_run(state: RunState, outcome: str, passed_first_try: bool) -> None:
    """Insert the run row, catching errors so persist never crashes the pipeline."""
    try:
        db.insert_run(
            run_id=state.run_id,
            topic=state.topic,
            started_at=state.started_at,
            finished_at=datetime.now(tz=UTC),
            outcome=outcome,
            attempts_used=state.attempt,
            first_attempt_pass=passed_first_try,
            corpus_version=state.grounding.corpus_version if state.grounding else None,
            injected_error=state.injected_error,
        )
        logger.info("[%s] run row inserted: %s", _NODE_NAME, state.run_id)
    except Exception as exc:
        logger.error("[%s] failed to insert run: %s", _NODE_NAME, exc)


def _insert_attempts(state: RunState) -> None:
    """Insert all attempt snapshots from lesson_history + current lesson."""
    lessons = list(state.lesson_history)
    if state.lesson and state.lesson not in lessons:
        lessons.append(state.lesson)

    for i, lesson in enumerate(lessons, start=1):
        lesson_md = lesson.to_markdown()
        word_count = len(lesson_md.split())
        hard_fail_count = (
            len(state.verdict.hard_fails)
            if state.verdict and i == len(lessons)
            else 0
        )
        try:
            db.insert_attempt(
                run_id=state.run_id,
                attempt=i,
                lesson_md=lesson_md,
                word_count=word_count,
                fk_grade=0.0,    # metrics already computed by evaluate; not stored on Lesson
                hard_fail_count=hard_fail_count,
                repair_plan=(
                    state.repair_plan.model_dump() if state.repair_plan and i < len(lessons) else None
                ),
            )
        except Exception as exc:
            logger.error("[%s] failed to insert attempt %d: %s", _NODE_NAME, i, exc)


def _cluster_and_promote(state: RunState, promotion_threshold: int) -> None:
    """Upsert failure signatures and auto-promote to guardrails when threshold is crossed."""
    if state.verdict is None:
        return

    for fail in state.verdict.hard_fails:
        signature = _make_signature(fail.check_id, fail.reason)
        try:
            count = db.upsert_failure_mode(
                signature=signature,
                check_id=fail.check_id,
                description=fail.reason[:200],
            )
            logger.info(
                "[%s] failure mode '%s' count=%d (threshold=%d)",
                _NODE_NAME, signature, count, promotion_threshold,
            )
            if count >= promotion_threshold:
                _promote_to_guardrail(signature, fail.check_id, fail.reason)
        except Exception as exc:
            logger.error("[%s] failure clustering error: %s", _NODE_NAME, exc)


def _make_signature(check_id: str, reason: str) -> str:
    """Produce a stable, human-readable failure signature."""
    slug = re.sub(r"[^a-z0-9]+", "_", reason.lower())[:60].strip("_")
    return f"{check_id}::{slug}"


def _promote_to_guardrail(signature: str, check_id: str, reason: str) -> None:
    """Promote a recurring failure mode into an active guardrail."""
    pending = db.fetch_failure_modes_for_promotion()
    if not any(m["signature"] == signature for m in pending):
        return  # already promoted or below threshold

    guardrail_text = (
        f"[Guardrail auto-promoted from {check_id}] "
        f"Avoid: {reason[:150].strip('.')}."
    )
    try:
        db.insert_guardrail(text=guardrail_text, source_signature=signature)
        db.mark_failure_mode_promoted(signature)
        logger.info("[%s] promoted guardrail for signature '%s'", _NODE_NAME, signature)
    except Exception as exc:
        logger.error("[%s] guardrail promotion error: %s", _NODE_NAME, exc)
