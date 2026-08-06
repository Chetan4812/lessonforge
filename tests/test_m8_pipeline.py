"""M8 integration test — end-to-end pipeline with mock provider.

Acceptance criteria:
  1. pipeline.run with provider="mock" completes without raising.
  2. Final state.lesson is a valid Lesson.
  3. Final state.verdict is set (SHIP | ESCALATE).
  4. Final state.structural_report is populated.
  5. PRB-01 is present in the structural_report.
  6. output files are written: lesson.md, report.json, trace.json.
  7. lesson.md is non-empty.
  8. report.json is valid JSON with 'verdict' and 'checks' keys.
  9. trace.json is valid JSON with 'run_id' key.
  10. Output writer: write_outputs creates all 3 files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

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

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_passing_lesson() -> Lesson:
    from lessonforge.llm.mock_provider import MockProvider
    provider = MockProvider()
    payload = provider.call("generator", "", "", "mock")
    return Lesson.model_validate(payload)


def _make_state_with_lesson(tmp_dir: Path) -> RunState:
    """Build a state that already has lesson + verdict for output tests."""
    lesson = _make_passing_lesson()
    results = [
        CheckResult(
            check_id="FLW-02", dimension="flow", verdict="PASS",
            severity="hard", reason="ok", judged_by="deterministic"
        ),
        CheckResult(
            check_id="PRB-01", dimension="comprehension", verdict="PASS",
            severity="hard", reason="score 5/5", judged_by="persona_probe"
        ),
    ]
    verdict = Verdict(
        attempt=1, all_results=results, hard_fails=[],
        advisory_fails=[], ship_decision="SHIP",
    )
    return RunState(
        run_id="test-pipe-output",
        topic="RAG",
        started_at=datetime.now(),
        config={},
        lesson=lesson,
        blueprint=LessonBlueprint(
            learning_objectives=["a", "b", "c"], central_analogy="x",
            worked_example_scenario="y", must_define_terms=["z"],
            section_plan=[], out_of_scope=[],
        ),
        grounding=GroundingPack(
            chunks=[SourceChunk(id="S1", title="t", url=None, text="t", sha256="a" * 64)],
            corpus_version="v1",
        ),
        verdict=verdict,
        structural_report=StructuralReport(results=results, metrics={"word_count": 1000.0}),
        attempt=1,
    )


# ── Output writer tests ────────────────────────────────────────────────────────

def test_write_outputs_creates_three_files(tmp_path: Path) -> None:
    """write_outputs must create lesson.md, report.json, and trace.json."""
    from unittest.mock import patch

    from lessonforge.output import write_outputs

    state = _make_state_with_lesson(tmp_path)

    # Override OUT_DIR to the temp directory
    with patch("lessonforge.output.OUT_DIR", tmp_path):
        out_dir = write_outputs(state)

    assert (out_dir / "lesson.md").exists()
    assert (out_dir / "report.json").exists()
    assert (out_dir / "trace.json").exists()


def test_lesson_md_is_non_empty(tmp_path: Path) -> None:
    """lesson.md must contain the lesson title and section headings."""
    from unittest.mock import patch

    from lessonforge.output import write_outputs

    state = _make_state_with_lesson(tmp_path)

    with patch("lessonforge.output.OUT_DIR", tmp_path):
        out_dir = write_outputs(state)

    content = (out_dir / "lesson.md").read_text(encoding="utf-8")
    assert len(content) > 100
    assert "#" in content  # has headings


def test_report_json_is_valid(tmp_path: Path) -> None:
    """report.json must be parseable JSON with required keys."""
    from unittest.mock import patch

    from lessonforge.output import write_outputs

    state = _make_state_with_lesson(tmp_path)

    with patch("lessonforge.output.OUT_DIR", tmp_path):
        out_dir = write_outputs(state)

    data = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert "verdict" in data
    assert "checks" in data
    assert data["verdict"] == "SHIP"
    assert len(data["checks"]) == 2


def test_trace_json_is_valid(tmp_path: Path) -> None:
    """trace.json must be parseable JSON containing run_id."""
    from unittest.mock import patch

    from lessonforge.output import write_outputs

    state = _make_state_with_lesson(tmp_path)

    with patch("lessonforge.output.OUT_DIR", tmp_path):
        out_dir = write_outputs(state)

    data = json.loads((out_dir / "trace.json").read_text(encoding="utf-8"))
    assert data["run_id"] == "test-pipe-output"
    assert "topic" in data


# ── End-to-end pipeline tests ─────────────────────────────────────────────────

@pytest.mark.slow
def test_pipeline_runs_end_to_end_with_mock(tmp_path: Path) -> None:
    """Full pipeline with provider=mock must complete and produce a lesson."""
    from unittest.mock import patch

    from lessonforge import pipeline

    with patch("lessonforge.output.OUT_DIR", tmp_path), \
         patch("lessonforge.pipeline.OUT_DIR", tmp_path, create=True):
        state = pipeline.run(
            topic="Retrieval-Augmented Generation",
            provider="mock",
            write=True,
        )

    assert state.lesson is not None
    assert state.verdict is not None
    assert state.verdict.ship_decision in ("SHIP", "ESCALATE")
    assert state.structural_report is not None


@pytest.mark.slow
def test_pipeline_produces_prb01_result(tmp_path: Path) -> None:
    """Pipeline must include PRB-01 in the structural report."""
    from unittest.mock import patch

    from lessonforge import pipeline

    with patch("lessonforge.output.OUT_DIR", tmp_path):
        state = pipeline.run(
            topic="RAG",
            provider="mock",
            write=False,
        )

    assert state.structural_report is not None
    check_ids = {r.check_id for r in state.structural_report.results}
    assert "PRB-01" in check_ids


@pytest.mark.slow
def test_pipeline_writes_all_output_files(tmp_path: Path) -> None:
    """Pipeline must write lesson.md, report.json, trace.json to out/<run_id>/."""
    from unittest.mock import patch

    from lessonforge import pipeline

    run_id = "test-m8-pipe"
    with patch("lessonforge.output.OUT_DIR", tmp_path):
        _ = pipeline.run(
            topic="RAG",
            provider="mock",
            write=True,
            run_id=run_id,
        )

    run_dir = tmp_path / run_id
    assert (run_dir / "lesson.md").exists()
    assert (run_dir / "report.json").exists()
    assert (run_dir / "trace.json").exists()
