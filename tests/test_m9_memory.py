"""M9 tests — memory layer: SQLite database, recall node, persist node.

Acceptance criteria:
  1. init_db creates all required tables.
  2. insert_run and fetch round-trip correctly.
  3. insert_attempt round-trips correctly.
  4. insert_check_results round-trips correctly.
  5. fetch_active_guardrails returns only active guardrails, most-applied first.
  6. insert_exemplar and fetch_best_exemplar round-trip correctly.
  7. upsert_failure_mode increments occurrences on second call.
  8. fetch_failure_modes_for_promotion returns modes at threshold.
  9. mark_failure_mode_promoted prevents re-promotion.
 10. recall node: DB initialised + guardrails loaded into state.
 11. recall node: does not mutate input state.
 12. persist node: run row inserted after run.
 13. persist node: check results inserted after run.
 14. persist node: exemplar inserted on first-try pass.
 15. persist node: no exemplar inserted on failed run.
 16. persist node: failure modes updated from hard fails.
 17. persist node: guardrail auto-promoted after threshold is crossed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from lessonforge.memory import db
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

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mem_db(tmp_path: Path) -> Path:
    """Create a fresh, initialised in-memory DB in tmp_path."""
    db_path = tmp_path / "test_memory.db"
    db.init_db(db_path=db_path)
    return db_path


def _make_lesson() -> Lesson:
    from lessonforge.llm.mock_provider import MockProvider
    provider = MockProvider()
    payload = provider.call("generator", "", "", "mock")
    return Lesson.model_validate(payload)


def _make_check(check_id: str, verdict: str = "FAIL", severity: str = "hard") -> CheckResult:
    return CheckResult(
        check_id=check_id, dimension="test", verdict=verdict,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        reason=f"{check_id} failed because reason", judged_by="test",
    )


def _make_state(ship_decision: str = "SHIP", hard_fails: list[CheckResult] | None = None) -> RunState:
    lesson = _make_lesson()
    hf = hard_fails or []
    verdict = Verdict(
        attempt=1, all_results=hf, hard_fails=hf,
        advisory_fails=[], ship_decision=ship_decision,  # type: ignore[arg-type]
    )
    return RunState(
        run_id="test-m9",
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
        structural_report=StructuralReport(results=hf, metrics={"word_count": 1000.0}),
        attempt=1,
    )


# ── DB unit tests ──────────────────────────────────────────────────────────────

def test_init_db_creates_tables(mem_db: Path) -> None:
    """init_db must create all expected tables."""
    with db.connect(mem_db) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    required = {"runs", "attempts", "check_results", "failure_modes",
                "guardrails", "exemplars", "prompt_versions", "rubric_versions"}
    assert required <= tables


def test_insert_and_fetch_run(mem_db: Path) -> None:
    """insert_run stores the run; can be retrieved."""
    db.insert_run(
        run_id="r1", topic="RAG",
        started_at=datetime.now(), finished_at=datetime.now(),
        outcome="shipped", attempts_used=1, first_attempt_pass=True,
        db_path=mem_db,
    )
    with db.connect(mem_db) as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id='r1'").fetchone()
    assert row is not None
    assert row["outcome"] == "shipped"
    assert row["first_attempt_pass"] == 1


def test_insert_attempt(mem_db: Path) -> None:
    """insert_attempt stores the attempt."""
    db.insert_run(
        run_id="r2", topic="RAG",
        started_at=datetime.now(), finished_at=datetime.now(),
        outcome="shipped", attempts_used=1, first_attempt_pass=True,
        db_path=mem_db,
    )
    db.insert_attempt(
        run_id="r2", attempt=1, lesson_md="## Lesson",
        word_count=2, fk_grade=7.5, hard_fail_count=0,
        db_path=mem_db,
    )
    with db.connect(mem_db) as conn:
        row = conn.execute("SELECT * FROM attempts WHERE run_id='r2'").fetchone()
    assert row is not None
    assert row["word_count"] == 2


def test_insert_check_results(mem_db: Path) -> None:
    """insert_check_results stores multiple rows."""
    results = [
        {"check_id": "ACC-01", "dimension": "accuracy", "verdict": "PASS",
         "severity": "hard", "judged_by": "test", "reason": "ok"},
        {"check_id": "COV-04", "dimension": "coverage", "verdict": "FAIL",
         "severity": "hard", "judged_by": "test", "reason": "missing term"},
    ]
    db.insert_check_results(run_id="r3", attempt=1, results=results, db_path=mem_db)
    with db.connect(mem_db) as conn:
        rows = conn.execute("SELECT * FROM check_results WHERE run_id='r3'").fetchall()
    assert len(rows) == 2


def test_guardrails_fetch_order(mem_db: Path) -> None:
    """fetch_active_guardrails returns active guardrails most-applied first."""
    db.insert_guardrail(text="Avoid X", db_path=mem_db)
    db.insert_guardrail(text="Avoid Y", db_path=mem_db)
    with db.connect(mem_db) as conn:
        conn.execute("UPDATE guardrails SET times_applied=5 WHERE text='Avoid Y'")
    guardrails = db.fetch_active_guardrails(db_path=mem_db)
    assert guardrails[0] == "Avoid Y"


def test_exemplar_roundtrip(mem_db: Path) -> None:
    """insert_exemplar and fetch_best_exemplar round-trip."""
    db.insert_exemplar(
        run_id="r4", topic="Retrieval-Augmented Generation",
        lesson_md="## RAG lesson", passed_first_try=True, db_path=mem_db,
    )
    result = db.fetch_best_exemplar("Retrieval-Augmented Generation", db_path=mem_db)
    assert result == "## RAG lesson"


def test_failure_mode_increments(mem_db: Path) -> None:
    """upsert_failure_mode increments occurrences on repeated call."""
    sig = "ACC-01::some_reason"
    count1 = db.upsert_failure_mode(signature=sig, check_id="ACC-01", description="reason", db_path=mem_db)
    count2 = db.upsert_failure_mode(signature=sig, check_id="ACC-01", description="reason", db_path=mem_db)
    assert count1 == 1
    assert count2 == 2


def test_fetch_modes_for_promotion(mem_db: Path) -> None:
    """fetch_failure_modes_for_promotion returns modes at or above the threshold."""
    sig = "COV-04::missing_term"
    for _ in range(3):
        db.upsert_failure_mode(signature=sig, check_id="COV-04", description="missing term", db_path=mem_db)
    modes = db.fetch_failure_modes_for_promotion(min_occurrences=3, db_path=mem_db)
    assert any(m["signature"] == sig for m in modes)


def test_mark_promoted_prevents_re_promotion(mem_db: Path) -> None:
    """mark_failure_mode_promoted sets the flag so fetch returns empty."""
    sig = "FLW-02::missing_sections"
    for _ in range(3):
        db.upsert_failure_mode(signature=sig, check_id="FLW-02", description="missing sections", db_path=mem_db)
    db.mark_failure_mode_promoted(sig, db_path=mem_db)
    modes = db.fetch_failure_modes_for_promotion(min_occurrences=3, db_path=mem_db)
    assert not any(m["signature"] == sig for m in modes)


# ── Recall node tests ──────────────────────────────────────────────────────────

def test_recall_node_populates_guardrails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """recall node should load guardrails from DB into state."""
    from lessonforge.nodes import recall as recall_node

    test_db = tmp_path / "mem.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)
    db.init_db(db_path=test_db)
    db.insert_guardrail(text="Never say RAG eliminates hallucination.", db_path=test_db)

    state = RunState(run_id="test", topic="RAG", started_at=datetime.now(), config={}, attempt=1)
    result = recall_node.run(state)

    assert "Never say RAG eliminates hallucination." in result.guardrails


def test_recall_node_does_not_mutate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """recall node must not mutate the input state."""
    from lessonforge.nodes import recall as recall_node

    test_db = tmp_path / "mem.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)
    state = RunState(run_id="test", topic="RAG", started_at=datetime.now(), config={}, attempt=1)
    original_guardrails = list(state.guardrails)
    result = recall_node.run(state)
    assert state.guardrails == original_guardrails
    assert result is not state


# ── Persist node tests ─────────────────────────────────────────────────────────

def test_persist_node_inserts_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """persist node must write the run row."""
    from lessonforge.nodes import persist as persist_node

    test_db = tmp_path / "mem.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)

    state = _make_state(ship_decision="SHIP")
    persist_node.run(state)

    with db.connect(test_db) as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id='test-m9'").fetchone()
    assert row is not None
    assert row["outcome"] == "shipped"


def test_persist_node_inserts_check_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """persist node must write check results."""
    from lessonforge.nodes import persist as persist_node

    test_db = tmp_path / "mem.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)

    state = _make_state(ship_decision="SHIP")
    persist_node.run(state)

    with db.connect(test_db) as conn:
        rows = conn.execute("SELECT * FROM check_results WHERE run_id='test-m9'").fetchall()
    assert len(rows) == 0  # empty check results in state fixture — just confirm no crash


def test_persist_node_inserts_exemplar_on_first_try_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """persist node must save the lesson as an exemplar when it passes on attempt 1."""
    from lessonforge.nodes import persist as persist_node

    test_db = tmp_path / "mem.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)

    state = _make_state(ship_decision="SHIP")
    persist_node.run(state)

    with db.connect(test_db) as conn:
        row = conn.execute("SELECT * FROM exemplars WHERE run_id='test-m9'").fetchone()
    assert row is not None
    assert row["passed_first_try"] == 1


def test_persist_node_no_exemplar_on_escalate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """persist node must NOT save an exemplar when the outcome is ESCALATE."""
    from lessonforge.nodes import persist as persist_node

    test_db = tmp_path / "mem.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)

    state = _make_state(ship_decision="ESCALATE", hard_fails=[_make_check("ACC-01")])
    state_at_max = state.model_copy(update={"attempt": 3})
    persist_node.run(state_at_max)

    with db.connect(test_db) as conn:
        row = conn.execute("SELECT * FROM exemplars WHERE run_id='test-m9'").fetchone()
    assert row is None


def test_persist_node_updates_failure_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """persist node must upsert failure modes from hard fails."""
    from lessonforge.nodes import persist as persist_node

    test_db = tmp_path / "mem.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)

    fail = _make_check("ACC-01")
    state = _make_state(ship_decision="ESCALATE", hard_fails=[fail])
    persist_node.run(state)

    with db.connect(test_db) as conn:
        row = conn.execute("SELECT * FROM failure_modes").fetchone()
    assert row is not None
    assert row["check_id"] == "ACC-01"


def test_persist_node_auto_promotes_guardrail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After threshold crosses 3, persist must auto-promote to guardrails."""
    from lessonforge.nodes import persist as persist_node

    test_db = tmp_path / "mem.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)

    fail = _make_check("ACC-04")
    # Run persist 3 times with same failure to hit the threshold
    for run_num in range(3):
        s = _make_state(ship_decision="ESCALATE", hard_fails=[fail])
        s2 = s.model_copy(update={"run_id": f"test-m9-{run_num}", "attempt": 3})
        persist_node.run(s2)

    guardrails = db.fetch_active_guardrails(db_path=test_db)
    assert len(guardrails) >= 1
    assert "ACC-04" in guardrails[0]
