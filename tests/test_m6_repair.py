"""M6 tests — repair planner and repair node.

Acceptance criteria:
  1. Repair planner with ≤3 failing sections → strategy="surgical".
  2. Repair planner with >3 failing sections → strategy="full_rewrite".
  3. Repair planner with FLW-02 fail → strategy="full_rewrite".
  4. Repair planner: keep_sections is all sections NOT in the repair plan (surgical).
  5. Repair planner: keep_sections is empty for full_rewrite.
  6. Repair node with RETRY verdict → returns repaired lesson.
  7. Repair node increments state.attempt by 1.
  8. Repair node appends repaired lesson to lesson_history.
  9. Repair node sets state.repair_plan.
  10. Repair node appends to state.rejection_log.
  11. Repair node clears stale structural_report and verdict.
  12. Repair node is immutable — does not mutate input state.
  13. Repair node raises ValueError when verdict is None.
  14. Repair node raises ValueError when ship_decision is not RETRY.
  15. Repair node raises ValueError when lesson is None.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from lessonforge.llm.gateway import Gateway
from lessonforge.nodes.repair_planner import plan as build_plan
from lessonforge.state import (
    CheckResult,
    GroundingPack,
    Lesson,
    LessonBlueprint,
    RunState,
    SourceChunk,
    StructuralReport,
    Verdict,
)

# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_lesson() -> Lesson:
    from lessonforge.llm.mock_provider import MockProvider
    provider = MockProvider()
    payload = provider.call("generator", "", "", "mock")
    return Lesson.model_validate(payload)


def _make_blueprint() -> LessonBlueprint:
    return LessonBlueprint(
        learning_objectives=["Define RAG", "Explain RAG pipeline", "Identify limitations"],
        central_analogy="Open-book exam",
        worked_example_scenario="College FAQ chatbot",
        must_define_terms=["embedding", "vector", "index"],
        section_plan=[],
        out_of_scope=["Fine-tuning"],
    )


def _make_check(check_id: str, verdict: str = "FAIL", severity: str = "hard", section_key: str | None = None) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        dimension="test",
        verdict=verdict,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        reason=f"{check_id} failed",
        repair_instruction=f"Fix {check_id}",
        section_key=section_key,
        judged_by="test",
    )


def _make_verdict(
    hard_fails: list[CheckResult],
    attempt: int = 1,
    ship_decision: str = "RETRY",
) -> Verdict:
    return Verdict(
        attempt=attempt,
        all_results=hard_fails,
        hard_fails=hard_fails,
        advisory_fails=[],
        ship_decision=ship_decision,  # type: ignore[arg-type]
    )


def _make_state(
    ship_decision: str = "RETRY",
    hard_fails: list[CheckResult] | None = None,
    lesson: Lesson | None = None,
) -> RunState:
    if hard_fails is None:
        hard_fails = [_make_check("ACC-01"), _make_check("COV-04")]
    verdict = _make_verdict(hard_fails, ship_decision=ship_decision)
    return RunState(
        run_id="test-run-m6",
        topic="RAG",
        started_at=datetime.now(),
        config={},
        lesson=lesson or _make_lesson(),
        blueprint=_make_blueprint(),
        grounding=GroundingPack(
            chunks=[SourceChunk(id="S1-001", title="RAG", url=None, text="RAG is...", sha256="a" * 64)],
            corpus_version="v1",
        ),
        verdict=verdict,
        structural_report=StructuralReport(results=hard_fails, metrics={"word_count": 1000.0}),
        attempt=1,
    )


def _make_gateway() -> Gateway:
    return Gateway(provider="mock", run_id="test-run-m6")


# ── Repair planner tests ──────────────────────────────────────────────────────

def test_planner_surgical_for_few_sections() -> None:
    """≤3 distinct sections → strategy=surgical."""
    lesson = _make_lesson()
    fails = [_make_check("ACC-01"), _make_check("COV-04")]  # how_it_works + glossary = 2 sections
    repair_plan = build_plan(fails, lesson, attempt=1)
    assert repair_plan.strategy == "surgical"


def test_planner_full_rewrite_for_many_sections() -> None:
    """Many distinct sections → strategy=full_rewrite."""
    lesson = _make_lesson()
    fails = [
        _make_check("ACC-01"),   # how_it_works
        _make_check("COV-01"),   # what_it_is
        _make_check("COV-02"),   # why_it_matters
        _make_check("EXM-01"),   # worked_example
    ]
    repair_plan = build_plan(fails, lesson, attempt=1)
    assert repair_plan.strategy == "full_rewrite"


def test_planner_full_rewrite_on_flw02() -> None:
    """FLW-02 (structural) always forces full_rewrite."""
    lesson = _make_lesson()
    fails = [_make_check("FLW-02")]
    repair_plan = build_plan(fails, lesson, attempt=1)
    assert repair_plan.strategy == "full_rewrite"


def test_planner_surgical_keep_sections_excludes_failing() -> None:
    """Surgical plan: keep_sections must not include sections scheduled for repair."""
    lesson = _make_lesson()
    fails = [_make_check("COV-04")]  # maps to glossary
    repair_plan = build_plan(fails, lesson, attempt=1)

    assert repair_plan.strategy == "surgical"
    assert "glossary" not in repair_plan.keep_sections


def test_planner_full_rewrite_keep_sections_empty() -> None:
    """Full rewrite plan: keep_sections is empty."""
    lesson = _make_lesson()
    fails = [
        _make_check("ACC-01"), _make_check("COV-01"),
        _make_check("COV-02"), _make_check("EXM-01"),
    ]
    repair_plan = build_plan(fails, lesson, attempt=1)
    assert repair_plan.strategy == "full_rewrite"
    assert repair_plan.keep_sections == []


def test_planner_items_contain_triggering_checks() -> None:
    """Each RepairItem must reference its triggering check IDs."""
    lesson = _make_lesson()
    fails = [_make_check("ACC-04")]  # maps to common_mistakes
    repair_plan = build_plan(fails, lesson, attempt=1)
    assert any("ACC-04" in item.triggering_checks for item in repair_plan.items)


# ── Repair node tests ─────────────────────────────────────────────────────────

def test_repair_node_returns_lesson() -> None:
    """Repair node must return a new Lesson in state.lesson."""
    from lessonforge.nodes import repair as repair_node

    state = _make_state()
    result = repair_node.run(state, _make_gateway())

    assert result.lesson is not None
    assert isinstance(result.lesson, Lesson)


def test_repair_node_increments_attempt() -> None:
    """Repair node must increment state.attempt by 1."""
    from lessonforge.nodes import repair as repair_node

    state = _make_state()
    assert state.attempt == 1
    result = repair_node.run(state, _make_gateway())
    assert result.attempt == 2


def test_repair_node_appends_to_history() -> None:
    """Repair node must append the repaired lesson to lesson_history."""
    from lessonforge.nodes import repair as repair_node

    state = _make_state()
    initial_len = len(state.lesson_history)
    result = repair_node.run(state, _make_gateway())
    assert len(result.lesson_history) == initial_len + 1


def test_repair_node_sets_repair_plan() -> None:
    """Repair node must set state.repair_plan."""
    from lessonforge.nodes import repair as repair_node

    state = _make_state()
    result = repair_node.run(state, _make_gateway())
    assert result.repair_plan is not None


def test_repair_node_appends_rejection_log() -> None:
    """Repair node must record the failed lesson in state.rejection_log."""
    from lessonforge.nodes import repair as repair_node

    state = _make_state()
    initial_len = len(state.rejection_log)
    result = repair_node.run(state, _make_gateway())
    assert len(result.rejection_log) == initial_len + 1
    assert "hard_fails" in result.rejection_log[-1]


def test_repair_node_clears_stale_eval() -> None:
    """Repair node must clear structural_report and verdict so the next evaluate node starts fresh."""
    from lessonforge.nodes import repair as repair_node

    state = _make_state()
    result = repair_node.run(state, _make_gateway())
    assert result.structural_report is None
    assert result.verdict is None


def test_repair_node_does_not_mutate_state() -> None:
    """Repair node must not mutate the input state."""
    from lessonforge.nodes import repair as repair_node

    state = _make_state()
    original_attempt = state.attempt
    original_lesson = state.lesson

    result = repair_node.run(state, _make_gateway())

    assert state.attempt == original_attempt
    assert state.lesson is original_lesson
    assert result is not state


def test_repair_node_raises_without_verdict() -> None:
    """Repair node raises ValueError when state.verdict is None."""
    from lessonforge.nodes import repair as repair_node

    state = RunState(
        run_id="test", topic="RAG", started_at=datetime.now(),
        config={}, lesson=_make_lesson(), attempt=1,
    )
    with pytest.raises(ValueError, match="verdict"):
        repair_node.run(state, _make_gateway())


def test_repair_node_raises_when_not_retry() -> None:
    """Repair node raises ValueError when verdict.ship_decision is not RETRY."""
    from lessonforge.nodes import repair as repair_node

    state = _make_state(ship_decision="SHIP")
    with pytest.raises(ValueError, match="RETRY"):
        repair_node.run(state, _make_gateway())


def test_repair_node_raises_without_lesson() -> None:
    """Repair node raises ValueError when state.lesson is None."""
    from lessonforge.nodes import repair as repair_node

    hard_fails = [_make_check("ACC-01")]
    verdict = _make_verdict(hard_fails)
    state = RunState(
        run_id="test", topic="RAG", started_at=datetime.now(),
        config={}, lesson=None, verdict=verdict, attempt=1,
    )
    with pytest.raises(ValueError, match="lesson"):
        repair_node.run(state, _make_gateway())
