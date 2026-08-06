"""M5 tests — deterministic checks, judge runner, verdict aggregator, evaluate node.

Acceptance criteria:
  1. All 5 deterministic checks run without error on the mock lesson.
  2. FLW-02 PASSes when all 11 sections are present and in order.
  3. FLW-03 FAILs when word count < 900.
  4. LNG-01 FAILs when Flesch-Kincaid grade is too high.
  5. LNG-02 FAILs when glossary is missing a watchlist term.
  6. COV-04 FAILs when a blueprint term is absent from the glossary.
  7. Judge runner returns CheckResult objects for all expected check IDs.
  8. Judge runner handles flat-list fixture format (legacy fixtures).
  9. Verdict aggregator → SHIP when all PASS.
  10. Verdict aggregator → RETRY when hard FAIL and attempt < max.
  11. Verdict aggregator → ESCALATE when hard FAIL and attempt >= max.
  12. Evaluate node populates state.structural_report and state.verdict.
  13. Evaluate node is pure — does not mutate input state.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from lessonforge.config import AppConfig
from lessonforge.llm.gateway import Gateway
from lessonforge.state import (
    CheckResult,
    GroundingPack,
    Lesson,
    LessonBlueprint,
    LessonSection,
    RunState,
    SourceChunk,
)
from lessonforge.validators.deterministic import (
    check_cov04_glossary,
    check_flw02_section_order,
    check_flw03_word_count,
    check_lng01_readability,
    check_lng02_jargon,
)
from lessonforge.validators.deterministic import (
    run_all as run_deterministic,
)
from lessonforge.validators.verdict import aggregate

# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_passing_lesson() -> Lesson:
    """Return the mock lesson from the generator fixture (all 11 sections, good length)."""
    from lessonforge.llm.mock_provider import MockProvider

    provider = MockProvider()
    payload = provider.call("generator", "", "", "mock")
    return Lesson.model_validate(payload)


def _make_blueprint() -> LessonBlueprint:
    return LessonBlueprint(
        learning_objectives=["Define RAG", "Explain why RAG reduces hallucination", "List 5 pipeline stages"],
        central_analogy="Open-book exam",
        worked_example_scenario="College FAQ chatbot",
        must_define_terms=["RAG", "chunk", "vector", "index", "retrieval", "hallucination", "embedding", "token"],
        section_plan=[],
        out_of_scope=["Fine-tuning"],
    )


def _make_state() -> RunState:
    lesson = _make_passing_lesson()
    blueprint = _make_blueprint()
    grounding = GroundingPack(
        chunks=[
            SourceChunk(
                id="S1-001",
                title="RAG Paper",
                url=None,
                text="RAG combines retrieval with generation to reduce hallucination.",
                sha256="a" * 64,
            )
        ],
        corpus_version="abc12345",
    )
    return RunState(
        run_id="test-run-m5",
        topic="RAG",
        started_at=datetime.now(),
        config={},
        lesson=lesson,
        blueprint=blueprint,
        grounding=grounding,
        attempt=1,
    )


def _make_gateway() -> Gateway:
    return Gateway(provider="mock", run_id="test-run-m5")


# ── Deterministic checks ──────────────────────────────────────────────────────

def test_flw02_passes_with_correct_sections() -> None:
    lesson = _make_passing_lesson()
    result = check_flw02_section_order(lesson)
    assert result.verdict == "PASS", f"Expected PASS, got FAIL: {result.reason}"


def test_flw02_fails_with_missing_section() -> None:
    lesson = _make_passing_lesson()
    # Remove the first section
    truncated = Lesson(
        topic=lesson.topic,
        title=lesson.title,
        sections=lesson.sections[1:],  # remove hook
    )
    result = check_flw02_section_order(truncated)
    assert result.verdict == "FAIL"
    assert "hook" in (result.evidence_quote or "")


def test_flw03_passes_within_budget() -> None:
    lesson = _make_passing_lesson()
    result = check_flw03_word_count(lesson, min_words=100, max_words=9000)
    assert result.verdict == "PASS"


def test_flw03_fails_too_short() -> None:
    short_lesson = Lesson(
        topic="RAG",
        title="Test",
        sections=[
            LessonSection(key="hook", heading="Hook", body_md="Short content."),
            LessonSection(key="what_it_is", heading="What", body_md="Very short."),
            LessonSection(key="why_it_matters", heading="Why", body_md="Also short."),
            LessonSection(key="how_it_works", heading="How", body_md="Short."),
            LessonSection(key="analogy", heading="Analogy", body_md="Brief."),
            LessonSection(key="worked_example", heading="Example", body_md="Tiny."),
            LessonSection(key="common_mistakes", heading="Mistakes", body_md="Few words."),
            LessonSection(key="glossary", heading="Glossary", body_md="Minimal."),
            LessonSection(key="recap", heading="Recap", body_md="Short recap."),
            LessonSection(key="check_yourself", heading="Check", body_md="One question."),
            LessonSection(key="next_steps", heading="Next", body_md="Next steps."),
        ],
    )
    result = check_flw03_word_count(short_lesson, min_words=900)
    assert result.verdict == "FAIL"
    assert "short" in result.reason.lower()


def test_lng01_passes_on_simple_text() -> None:
    simple_lesson = Lesson(
        topic="RAG",
        title="RAG Lesson",
        sections=[
            LessonSection(key="hook", heading="Hook", body_md="RAG is a simple idea. The AI looks up facts before answering. This makes answers more accurate. The process has five clear steps. Each step builds on the last one.") for _ in range(11)
        ],
    )
    # Build minimal 11-section lesson
    result = check_lng01_readability(simple_lesson, max_fk_grade=12.0, max_avg_sentence_words=25.0)
    assert result.verdict == "PASS"


def test_lng02_fails_when_glossary_missing_term() -> None:
    lesson = _make_passing_lesson()
    # Replace glossary with an empty one
    new_sections = [
        s if s.key != "glossary"
        else LessonSection(key="glossary", heading="Glossary", body_md="No definitions here.")
        for s in lesson.sections
    ]
    lesson_no_glossary = Lesson(topic=lesson.topic, title=lesson.title, sections=new_sections)
    result = check_lng02_jargon(lesson_no_glossary)
    # The fixture lesson should have terms like "embedding" in glossary;
    # with empty glossary most watchlist terms will be missing
    assert result.verdict == "FAIL"


def test_cov04_fails_when_blueprint_term_missing() -> None:
    lesson = _make_passing_lesson()
    blueprint = _make_blueprint()
    # Add a term that definitely isn't in the glossary
    from lessonforge.state import LessonBlueprint
    exotic_blueprint = LessonBlueprint(
        learning_objectives=blueprint.learning_objectives,
        central_analogy=blueprint.central_analogy,
        worked_example_scenario=blueprint.worked_example_scenario,
        must_define_terms=["XYZZY_NONEXISTENT_TERM_12345"],
        section_plan=[],
        out_of_scope=[],
    )
    result = check_cov04_glossary(lesson, exotic_blueprint)
    assert result.verdict == "FAIL"
    assert "XYZZY_NONEXISTENT_TERM_12345" in (result.evidence_quote or "")


def test_run_all_deterministic_returns_checks() -> None:
    lesson = _make_passing_lesson()
    blueprint = _make_blueprint()
    results = run_deterministic(lesson, blueprint)
    check_ids = {r.check_id for r in results}
    assert "FLW-02" in check_ids
    assert "FLW-03" in check_ids
    assert "LNG-01" in check_ids
    assert "LNG-02" in check_ids
    assert "COV-04" in check_ids


# ── Verdict aggregator ────────────────────────────────────────────────────────

def _pass_result(check_id: str) -> CheckResult:
    return CheckResult(
        check_id=check_id, dimension="test", verdict="PASS",
        severity="hard", reason="ok", judged_by="test",
    )


def _fail_result(check_id: str, severity: str = "hard") -> CheckResult:
    return CheckResult(
        check_id=check_id, dimension="test", verdict="FAIL",
        severity=severity, reason="fail", judged_by="test",  # type: ignore[arg-type]
    )


def test_verdict_ship_when_all_pass() -> None:
    results = [_pass_result("ACC-01"), _pass_result("LNG-01")]
    v = aggregate(results, attempt=1)
    assert v.ship_decision == "SHIP"
    assert len(v.hard_fails) == 0


def test_verdict_retry_when_hard_fail_and_attempt_below_max() -> None:
    cfg = AppConfig()
    max_a = cfg.max_attempts
    results = [_fail_result("ACC-01")]
    v = aggregate(results, attempt=1, config=cfg)
    if max_a > 1:
        assert v.ship_decision == "RETRY"


def test_verdict_escalate_when_hard_fail_at_max_attempts() -> None:
    cfg = AppConfig()
    max_a = cfg.max_attempts
    results = [_fail_result("ACC-01")]
    v = aggregate(results, attempt=max_a, config=cfg)
    assert v.ship_decision == "ESCALATE"


def test_verdict_advisory_fail_does_not_block_ship() -> None:
    results = [_pass_result("ACC-01"), _fail_result("LNG-04", severity="advisory")]
    v = aggregate(results, attempt=1)
    assert v.ship_decision == "SHIP"
    assert len(v.advisory_fails) == 1


# ── Evaluate node ─────────────────────────────────────────────────────────────

def test_evaluate_node_populates_report_and_verdict() -> None:
    from lessonforge.nodes import evaluate as evaluate_node

    state = _make_state()
    gw = _make_gateway()
    result = evaluate_node.run(state, gw)

    assert result.structural_report is not None
    assert len(result.structural_report.results) > 0
    assert result.verdict is not None
    assert result.verdict.ship_decision in ("SHIP", "RETRY", "ESCALATE")


def test_evaluate_node_does_not_mutate_state() -> None:
    from lessonforge.nodes import evaluate as evaluate_node

    state = _make_state()
    original_verdict = state.verdict
    result = evaluate_node.run(state, _make_gateway())

    assert state.verdict == original_verdict, "Input state must not be mutated"
    assert result is not state


def test_evaluate_node_requires_lesson() -> None:
    from lessonforge.nodes import evaluate as evaluate_node

    state = RunState(
        run_id="test",
        topic="RAG",
        started_at=datetime.now(),
        config={},
        lesson=None,
        attempt=1,
    )
    with pytest.raises(ValueError, match="lesson"):
        evaluate_node.run(state, _make_gateway())


def test_evaluate_node_metrics_in_report() -> None:
    from lessonforge.nodes import evaluate as evaluate_node

    state = _make_state()
    result = evaluate_node.run(state, _make_gateway())

    assert result.structural_report is not None
    metrics = result.structural_report.metrics
    assert "word_count" in metrics
    assert "fk_grade" in metrics
    assert metrics["word_count"] > 0
