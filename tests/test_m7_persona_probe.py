"""M7 tests — persona probe (PRB-01).

Acceptance criteria:
  1. Persona probe prompt loads and renders correctly.
  2. Probe with passing fixture (score=5/5) → verdict=PASS CheckResult.
  3. Probe with failing fixture (score=2/5) → verdict=FAIL CheckResult.
  4. PRB-01 FAIL includes a repair_instruction.
  5. Probe with gateway error → synthetic FAIL (fail-closed).
  6. Evaluate node now includes PRB-01 in its results.
  7. PRB-01 appears in the structural_report after evaluation.
  8. Score below min_score triggers FAIL correctly.
  9. Score at min_score triggers PASS (boundary condition).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from lessonforge.llm.gateway import Gateway
from lessonforge.prompts.loader import load_prompt
from lessonforge.state import (
    GroundingPack,
    Lesson,
    LessonBlueprint,
    RunState,
    SourceChunk,
)
from lessonforge.validators.persona_probe import _parse_probe_response, run_persona_probe

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_lesson() -> Lesson:
    from lessonforge.llm.mock_provider import MockProvider
    provider = MockProvider()
    payload = provider.call("generator", "", "", "mock")
    return Lesson.model_validate(payload)


def _make_gateway() -> Gateway:
    return Gateway(provider="mock", run_id="test-run-m7")


def _make_state() -> RunState:
    lesson = _make_lesson()
    return RunState(
        run_id="test-run-m7",
        topic="RAG",
        started_at=datetime.now(),
        config={},
        lesson=lesson,
        blueprint=LessonBlueprint(
            learning_objectives=["Define RAG", "Explain RAG", "Identify limitations"],
            central_analogy="Open-book exam",
            worked_example_scenario="College FAQ chatbot",
            must_define_terms=["embedding", "vector", "index"],
            section_plan=[],
            out_of_scope=["Fine-tuning"],
        ),
        grounding=GroundingPack(
            chunks=[SourceChunk(id="S1-001", title="RAG", url=None, text="RAG is...", sha256="a" * 64)],
            corpus_version="v1",
        ),
        attempt=1,
    )


# ── Prompt loader tests ────────────────────────────────────────────────────────

def test_persona_probe_prompt_loads() -> None:
    """persona_probe.md must load and render without error."""
    system, user, sha = load_prompt(
        role="persona_probe",
        variables={
            "persona_description": "12th grade student",
            "lesson_text": "RAG is a method that retrieves documents before generating answers.",
            "question_count": "5",
            "min_score": "4",
        },
    )
    assert len(system) > 50
    assert "5" in system  # question_count rendered in system
    assert "4" in system  # min_score rendered in system
    assert len(sha) == 64


# ── Parse response tests ───────────────────────────────────────────────────────

def test_parse_response_pass_at_threshold() -> None:
    """Score == min_score → PASS (boundary condition)."""
    payload = {
        "total_score": 4,
        "max_score": 5,
        "scores": [1, 1, 1, 1, 0],
        "questions": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        "reason": "Got 4 right.",
        "verdict": "PASS",
    }
    result = _parse_probe_response(payload, min_score=4, max_score=5)
    assert result.verdict == "PASS"
    assert result.check_id == "PRB-01"


def test_parse_response_fail_below_threshold() -> None:
    """Score < min_score → FAIL."""
    payload = {
        "total_score": 2,
        "max_score": 5,
        "scores": [1, 0, 0, 1, 0],
        "questions": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        "reason": "Only got 2 right.",
        "verdict": "FAIL",
    }
    result = _parse_probe_response(payload, min_score=4, max_score=5)
    assert result.verdict == "FAIL"
    assert result.repair_instruction is not None


def test_parse_response_fail_provides_evidence() -> None:
    """FAIL result should have evidence_quote pointing to a failed question."""
    payload = {
        "total_score": 0,
        "max_score": 5,
        "scores": [0, 0, 0, 0, 0],
        "questions": ["What is RAG?", "Q2", "Q3", "Q4", "Q5"],
        "reason": "Learner did not understand anything.",
        "verdict": "FAIL",
    }
    result = _parse_probe_response(payload, min_score=4, max_score=5)
    assert result.verdict == "FAIL"
    assert result.evidence_quote is not None
    assert "What is RAG?" in (result.evidence_quote or "")


def test_parse_response_overrides_wrong_verdict() -> None:
    """The parser uses score comparison, not the model's stated verdict."""
    payload = {
        "total_score": 2,  # below min_score=4
        "max_score": 5,
        "scores": [1, 1, 0, 0, 0],
        "questions": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        "reason": "Partial understanding.",
        "verdict": "PASS",  # model incorrectly said PASS
    }
    result = _parse_probe_response(payload, min_score=4, max_score=5)
    assert result.verdict == "FAIL", "Parser should use score comparison, not model's stated verdict"


# ── run_persona_probe tests ────────────────────────────────────────────────────

def test_run_probe_with_passing_fixture() -> None:
    """Mock provider serves the passing fixture → PRB-01 PASS."""
    lesson = _make_lesson()
    gw = _make_gateway()
    result = run_persona_probe(lesson, gw)
    assert result.check_id == "PRB-01"
    assert result.verdict == "PASS"


def test_run_probe_returns_check_result() -> None:
    """run_persona_probe always returns a CheckResult (even on gateway error)."""
    from lessonforge.state import CheckResult

    lesson = _make_lesson()
    gw = _make_gateway()
    assert gw.mock is not None

    result = run_persona_probe(lesson, gw)
    assert isinstance(result, CheckResult)
    assert result.judged_by == "persona_probe"


def test_run_probe_fails_closed_on_gateway_error() -> None:
    """If the gateway call throws, run_persona_probe returns a synthetic FAIL (fail-closed)."""
    lesson = _make_lesson()
    gw = _make_gateway()

    with patch.object(gw, "call", side_effect=RuntimeError("network error")):
        result = run_persona_probe(lesson, gw)
    assert result.verdict == "FAIL"
    assert "Gateway error" in result.reason


# ── Integration with evaluate node ────────────────────────────────────────────

def test_evaluate_includes_prb01() -> None:
    """After evaluate.run, state.structural_report must contain a PRB-01 result."""
    from lessonforge.nodes import evaluate as evaluate_node

    state = _make_state()
    result = evaluate_node.run(state, _make_gateway())

    assert result.structural_report is not None
    check_ids = {r.check_id for r in result.structural_report.results}
    assert "PRB-01" in check_ids, f"PRB-01 missing from: {check_ids}"


def test_evaluate_prb01_uses_persona_fixture() -> None:
    """The PRB-01 result should reflect the passing persona fixture score."""
    from lessonforge.nodes import evaluate as evaluate_node

    state = _make_state()
    result = evaluate_node.run(state, _make_gateway())

    assert result.structural_report is not None
    prb = next((r for r in result.structural_report.results if r.check_id == "PRB-01"), None)
    assert prb is not None
    assert prb.verdict == "PASS", f"Expected PASS but got: {prb.reason}"
