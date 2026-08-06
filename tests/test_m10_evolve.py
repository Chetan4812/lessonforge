"""M10 tests — self-evolution: miner, analyst, promoter, and CLI evolve command.

Acceptance criteria:
  1.  mine() returns clusters from failure_modes above the threshold.
  2.  mine() excludes clusters below min_occurrences.
  3.  mine() respects the since_days cutoff.
  4.  first_attempt_pass_rate() returns 1.0 when all runs pass first try.
  5.  first_attempt_pass_rate() returns 0.0 when no runs exist.
  6.  _parse_diagnoses() parses valid analyst output into Diagnosis objects.
  7.  _parse_diagnoses() skips malformed items without raising.
  8.  promote() inserts a guardrail for a prompt_rule diagnosis.
  9.  promote() dry_run does NOT write to the DB.
  10. promote() marks the failure mode as promoted.
  11. promote() returns a human-readable change description.
  12. promote() handles non-prompt_rule types without raising.
  13. promote() returns a [DRY RUN] prefix string in dry_run mode.
  14. evolve CLI: '--dry-run' exits cleanly with no DB changes.
  15. evolve CLI: no failures → prints 'Nothing to evolve' and returns 0.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lessonforge.cli import app
from lessonforge.evolve import analyst as analyst_mod
from lessonforge.evolve import miner as miner_mod
from lessonforge.evolve import promoter as promoter_mod
from lessonforge.evolve.analyst import Diagnosis, ProposedFix
from lessonforge.memory import db

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mem_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "mem.db"
    db.init_db(db_path=db_path)
    return db_path


def _seed_failure_mode(db_path: Path, sig: str, check_id: str, n: int = 3, days_ago: int = 1) -> None:
    """Upsert a failure mode N times with a recent last_seen timestamp."""

    from lessonforge.memory.db import connect

    last_seen = (datetime.now(tz=UTC) - timedelta(days=days_ago)).isoformat()
    for _ in range(n):
        db.upsert_failure_mode(signature=sig, check_id=check_id, description="test desc", db_path=db_path)
    # Force last_seen to the intended date so since_days filtering works
    with connect(db_path) as conn:
        conn.execute("UPDATE failure_modes SET last_seen=? WHERE signature=?", (last_seen, sig))


def _make_prompt_rule_diagnosis() -> Diagnosis:
    return Diagnosis(
        cluster_signature="ACC-04::rag_eliminates_hallucination",
        check_id="ACC-04",
        occurrences=4,
        root_cause="generator_prompt_gap",
        rationale="Model keeps claiming RAG eliminates hallucination.",
        proposed_fix=ProposedFix(
            fix_type="prompt_rule",
            description="Add a guardrail forbidding the claim.",
            guardrail_text="Never state that RAG removes hallucination; say it reduces it.",
        ),
    )


# ── Miner unit tests ───────────────────────────────────────────────────────────

def test_mine_returns_clusters(mem_db: Path) -> None:
    """mine() returns clusters that meet the min_occurrences threshold."""
    _seed_failure_mode(mem_db, "ACC-04::rag_removes_hallucination", "ACC-04", n=3)
    clusters = miner_mod.mine(since_days=7, min_occurrences=2, db_path=mem_db)
    assert len(clusters) >= 1
    assert any(c.check_id == "ACC-04" for c in clusters)


def test_mine_excludes_below_threshold(mem_db: Path) -> None:
    """mine() does not return clusters below min_occurrences."""
    _seed_failure_mode(mem_db, "LNG-03::idiom_used", "LNG-03", n=1)
    clusters = miner_mod.mine(since_days=7, min_occurrences=2, db_path=mem_db)
    assert all(c.check_id != "LNG-03" for c in clusters)


def test_mine_respects_since_cutoff(mem_db: Path) -> None:
    """mine() excludes failure modes older than since_days."""
    _seed_failure_mode(mem_db, "EXM-01::no_worked_example", "EXM-01", n=5, days_ago=60)
    clusters = miner_mod.mine(since_days=7, min_occurrences=2, db_path=mem_db)
    assert all(c.signature != "EXM-01::no_worked_example" for c in clusters)


def test_first_attempt_pass_rate_all_pass(mem_db: Path) -> None:
    """first_attempt_pass_rate() returns 1.0 when all runs pass first try."""
    now = datetime.now(tz=UTC)
    for i in range(3):
        db.insert_run(
            run_id=f"r-pass-{i}", topic="RAG",
            started_at=now, finished_at=now,
            outcome="shipped", attempts_used=1,
            first_attempt_pass=True, db_path=mem_db,
        )
    rate = miner_mod.first_attempt_pass_rate(since_days=1, db_path=mem_db)
    assert rate == 1.0


def test_first_attempt_pass_rate_no_runs(mem_db: Path) -> None:
    """first_attempt_pass_rate() returns 0.0 when no runs exist."""
    rate = miner_mod.first_attempt_pass_rate(since_days=1, db_path=mem_db)
    assert rate == 0.0


# ── Analyst unit tests ─────────────────────────────────────────────────────────

def test_parse_diagnoses_valid() -> None:
    """_parse_diagnoses() correctly parses a valid LLM payload."""
    payload = {
        "diagnoses": [
            {
                "cluster_signature": "ACC-04::rag_removes_hallucination",
                "check_id": "ACC-04",
                "occurrences": 4,
                "root_cause": "generator_prompt_gap",
                "rationale": "The model consistently claims RAG eliminates hallucination.",
                "proposed_fix": {
                    "type": "prompt_rule",
                    "description": "Add guardrail about hallucination.",
                    "guardrail_text": "Never state RAG removes hallucination.",
                },
            }
        ]
    }
    diagnoses = analyst_mod._parse_diagnoses(payload)
    assert len(diagnoses) == 1
    assert diagnoses[0].check_id == "ACC-04"
    assert diagnoses[0].proposed_fix.guardrail_text == "Never state RAG removes hallucination."


def test_parse_diagnoses_skips_malformed() -> None:
    """_parse_diagnoses() skips items with missing/bad fields without raising."""
    payload = {
        "diagnoses": [
            {"check_id": "ACC-04", "occurrences": "not-a-number"},  # bad occurrences
            {
                "cluster_signature": "valid",
                "check_id": "COV-04",
                "occurrences": 2,
                "root_cause": "rubric_ambiguity",
                "rationale": "Unclear.",
                "proposed_fix": {"type": "rubric_patch", "description": "Clarify."},
            },
        ]
    }
    diagnoses = analyst_mod._parse_diagnoses(payload)
    # The valid one should be parsed; bad one should not crash
    assert any(d.check_id == "COV-04" for d in diagnoses)


# ── Promoter unit tests ────────────────────────────────────────────────────────

def test_promote_inserts_guardrail(mem_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """promote() inserts a guardrail for a prompt_rule diagnosis."""
    monkeypatch.setattr(db, "DB_PATH", mem_db)
    diag = _make_prompt_rule_diagnosis()
    # Seed the failure mode first so mark_promoted can find it
    _seed_failure_mode(mem_db, diag.cluster_signature, diag.check_id, n=4)
    promoter_mod.promote([diag], dry_run=False, db_path=mem_db)
    guardrails = db.fetch_active_guardrails(db_path=mem_db)
    assert any("hallucination" in g for g in guardrails)


def test_promote_dry_run_no_db_write(mem_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """promote() in dry_run mode must NOT write to the DB."""
    monkeypatch.setattr(db, "DB_PATH", mem_db)
    diag = _make_prompt_rule_diagnosis()
    promoter_mod.promote([diag], dry_run=True, db_path=mem_db)
    guardrails = db.fetch_active_guardrails(db_path=mem_db)
    assert len(guardrails) == 0


def test_promote_marks_failure_mode_promoted(mem_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """promote() marks the failure mode as promoted so it isn't re-promoted."""
    monkeypatch.setattr(db, "DB_PATH", mem_db)
    diag = _make_prompt_rule_diagnosis()
    _seed_failure_mode(mem_db, diag.cluster_signature, diag.check_id, n=4)
    promoter_mod.promote([diag], dry_run=False, db_path=mem_db)
    modes = db.fetch_failure_modes_for_promotion(min_occurrences=2, db_path=mem_db)
    assert not any(m["signature"] == diag.cluster_signature for m in modes)


def test_promote_returns_change_description(mem_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """promote() returns a human-readable change description for each fix."""
    monkeypatch.setattr(db, "DB_PATH", mem_db)
    diag = _make_prompt_rule_diagnosis()
    _seed_failure_mode(mem_db, diag.cluster_signature, diag.check_id, n=4)
    changes = promoter_mod.promote([diag], dry_run=False, db_path=mem_db)
    assert len(changes) == 1
    assert "ACC-04" in changes[0]


def test_promote_handles_non_prompt_rule(mem_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """promote() handles rubric_patch and new_checkpoint without raising."""
    monkeypatch.setattr(db, "DB_PATH", mem_db)
    diag = Diagnosis(
        cluster_signature="COV-04::missing_glossary",
        check_id="COV-04",
        occurrences=3,
        root_cause="rubric_ambiguity",
        rationale="Judges disagree on what counts as a definition.",
        proposed_fix=ProposedFix(
            fix_type="rubric_patch",
            description="Add explicit ≤25-word constraint to COV-04.",
            guardrail_text=None,
        ),
    )
    changes = promoter_mod.promote([diag], dry_run=False, db_path=mem_db)
    assert len(changes) == 1
    assert "rubric_patch" in changes[0]


def test_promote_dry_run_prefix(mem_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """promote() dry_run returns a [DRY RUN] prefixed string."""
    monkeypatch.setattr(db, "DB_PATH", mem_db)
    diag = _make_prompt_rule_diagnosis()
    changes = promoter_mod.promote([diag], dry_run=True, db_path=mem_db)
    assert len(changes) == 1
    assert changes[0].startswith("[DRY RUN]")


# ── CLI integration tests ──────────────────────────────────────────────────────

runner = CliRunner()


def test_evolve_cli_no_failures_exits_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When no recurring failures exist, evolve exits cleanly with a 'nothing to evolve' message."""
    test_db = tmp_path / "mem.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)

    result = runner.invoke(app, ["evolve", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Nothing to evolve" in result.output or "failure cluster" in result.output


def test_evolve_cli_dry_run_no_db_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """evolve --dry-run: even with clusters present, no DB changes are made."""
    test_db = tmp_path / "mem.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)
    db.init_db(db_path=test_db)
    _seed_failure_mode(test_db, "ACC-04::rag_removes_hallucination", "ACC-04", n=4)

    result = runner.invoke(app, ["evolve", "--dry-run", "--provider", "mock"])
    assert result.exit_code == 0, result.output

    guardrails = db.fetch_active_guardrails(db_path=test_db)
    assert len(guardrails) == 0
