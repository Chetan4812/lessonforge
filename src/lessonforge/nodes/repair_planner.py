"""Repair planner — deterministic, pure-Python.

Converts a Verdict's hard_fails into a structured RepairPlan.

No LLM call — the fail reasons from the judges already contain enough
information to produce a targeted plan.  This keeps the repair loop fast
and cheap: only the repair *executor* (repair.py node) uses a model call.

Strategy selection:
  - "surgical"    → ≤ 3 sections need work
  - "full_rewrite" → > 3 sections need work, or a structural check (FLW-02) failed

Each RepairItem maps a failing CheckResult to:
  - section_key:          which section to rewrite
  - problem:              plain-English description of the failure
  - instruction:          exactly what the rewriter must do to fix it
  - triggering_checks:    the check IDs that caused this item
"""

from __future__ import annotations

import logging

from lessonforge.state import CheckResult, Lesson, RepairItem, RepairPlan

logger = logging.getLogger(__name__)

# Maps check_id → the section key most likely responsible
_CHECK_SECTION_MAP: dict[str, str] = {
    "ACC-01": "how_it_works",
    "ACC-02": "how_it_works",   # most citation problems are in the technical section
    "ACC-03": "how_it_works",
    "ACC-04": "common_mistakes",
    "LNG-01": "how_it_works",   # longest / most complex section
    "LNG-02": "glossary",
    "LNG-03": "analogy",
    "LNG-04": "how_it_works",
    "EXM-01": "worked_example",
    "EXM-02": "worked_example",
    "EXM-03": "analogy",
    "EXM-04": "worked_example",
    "COV-01": "what_it_is",
    "COV-02": "why_it_matters",
    "COV-03": "how_it_works",
    "COV-04": "glossary",
    "FLW-01": "hook",            # forward reference most likely at the start
    "FLW-02": "",               # structural — needs full rewrite
    "FLW-03": "recap",           # too long → trim recap/check_yourself; too short → expand how_it_works
    "PRB-01": "check_yourself",
}

# Checks that force a full_rewrite strategy
_FORCE_FULL_REWRITE: set[str] = {"FLW-02"}

# Maximum distinct sections that can be handled surgically
_SURGICAL_SECTION_LIMIT = 3


def plan(
    hard_fails: list[CheckResult],
    lesson: Lesson,
    attempt: int,
) -> RepairPlan:
    """Convert a list of hard-fail CheckResults into a RepairPlan.

    Args:
        hard_fails:  Hard-fail results from the Verdict.
        lesson:      The current lesson (to determine keep_sections).
        attempt:     The attempt number this repair is for.

    Returns:
        A RepairPlan ready for the repair executor.
    """
    if not hard_fails:
        logger.warning("[planner] plan() called with no hard_fails — returning empty plan")
        return RepairPlan(
            attempt_for=attempt,
            strategy="surgical",
            items=[],
            keep_sections=[s.key for s in lesson.sections],
        )

    # ── Detect if any fail forces a full_rewrite ──────────────────────────────
    force_full = any(r.check_id in _FORCE_FULL_REWRITE for r in hard_fails)

    # ── Group fails by section ────────────────────────────────────────────────
    section_to_checks: dict[str, list[CheckResult]] = {}
    for result in hard_fails:
        section = _CHECK_SECTION_MAP.get(result.check_id, "")
        if not section:
            # Structural check — handled by full_rewrite
            continue
        if section not in section_to_checks:
            section_to_checks[section] = []
        section_to_checks[section].append(result)

    # ── Decide strategy ───────────────────────────────────────────────────────
    distinct_sections = len(section_to_checks)
    if force_full or distinct_sections > _SURGICAL_SECTION_LIMIT:
        strategy: str = "full_rewrite"
    else:
        strategy = "surgical"

    # ── Build repair items ────────────────────────────────────────────────────
    items: list[RepairItem] = []
    for section_key, checks in section_to_checks.items():
        # Aggregate problems and instructions from all failing checks on this section
        problems = "; ".join(r.reason for r in checks)
        instructions = "; ".join(
            f"[{r.check_id}] {r.repair_instruction or r.reason}"
            for r in checks
        )
        items.append(RepairItem(
            section_key=section_key,
            problem=problems,
            instruction=instructions,
            triggering_checks=[r.check_id for r in checks],
        ))

    # ── Determine keep_sections ───────────────────────────────────────────────
    if strategy == "full_rewrite":
        keep_sections: list[str] = []  # nothing kept verbatim in full rewrite
    else:
        sections_to_fix = {item.section_key for item in items}
        keep_sections = [s.key for s in lesson.sections if s.key not in sections_to_fix]

    logger.info(
        "[planner] attempt=%d strategy=%s items=%d keep=%d",
        attempt, strategy, len(items), len(keep_sections),
    )

    return RepairPlan(
        attempt_for=attempt,
        strategy=strategy,  # type: ignore[arg-type]
        items=items,
        keep_sections=keep_sections,
    )
