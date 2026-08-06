"""Memory layer — SQLite database access module.

Responsibilities:
  - Initialise the database from schema.sql (idempotent).
  - Provide simple DAO functions for insert and retrieval.
  - Expose DB_PATH so tests can override it via monkeypatching.

All SQL is in this file; no raw SQL leaks into nodes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lessonforge.config import CACHE_DIR

logger = logging.getLogger(__name__)

# ── Path resolution ───────────────────────────────────────────────────────────

DB_PATH: Path = CACHE_DIR / "memory.db"

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


# ── Connection / initialisation ───────────────────────────────────────────────

@contextmanager
def connect(db_path: Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Yield an open, row-factory-enabled SQLite connection."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    """Initialise the database from schema.sql (idempotent — safe to call every run)."""
    schema = _SCHEMA_PATH.read_text(encoding="utf-8")
    with connect(db_path) as conn:
        conn.executescript(schema)
    logger.info("[memory] database initialised at %s", db_path or DB_PATH)


# ── Runs ──────────────────────────────────────────────────────────────────────

def insert_run(
    *,
    run_id: str,
    topic: str,
    started_at: datetime,
    finished_at: datetime,
    outcome: str,
    attempts_used: int,
    first_attempt_pass: bool,
    corpus_version: str | None = None,
    total_tokens: int = 0,
    total_cost_usd: float = 0.0,
    wall_clock_s: float = 0.0,
    injected_error: str | None = None,
    db_path: Path | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, topic, started_at, finished_at, outcome,
                attempts_used, first_attempt_pass, corpus_version,
                total_tokens, total_cost_usd, wall_clock_s, injected_error
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id, topic,
                started_at.isoformat(), finished_at.isoformat(),
                outcome, attempts_used, int(first_attempt_pass),
                corpus_version, total_tokens, total_cost_usd,
                wall_clock_s, injected_error,
            ),
        )


# ── Attempts ──────────────────────────────────────────────────────────────────

def insert_attempt(
    *,
    run_id: str,
    attempt: int,
    lesson_md: str,
    word_count: int,
    fk_grade: float,
    hard_fail_count: int,
    repair_plan: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO attempts (run_id, attempt, lesson_md, word_count, fk_grade,
                hard_fail_count, repair_plan_json)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                run_id, attempt, lesson_md, word_count, fk_grade,
                hard_fail_count,
                json.dumps(repair_plan) if repair_plan else None,
            ),
        )


# ── Check results ─────────────────────────────────────────────────────────────

def insert_check_results(
    *,
    run_id: str,
    attempt: int,
    results: list[dict[str, Any]],
    db_path: Path | None = None,
) -> None:
    rows = [
        (
            run_id, attempt,
            r.get("check_id"), r.get("dimension"), r.get("verdict"),
            r.get("severity"), r.get("judged_by"), r.get("evidence_quote"),
            r.get("reason"), r.get("repair_instruction"), r.get("section_key"),
            r.get("vote_split"),
        )
        for r in results
    ]
    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO check_results (
                run_id, attempt, check_id, dimension, verdict, severity,
                judged_by, evidence_quote, reason, repair_instruction,
                section_key, vote_split
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )


# ── Guardrails ────────────────────────────────────────────────────────────────

def fetch_active_guardrails(
    limit: int = 12,
    db_path: Path | None = None,
) -> list[str]:
    """Return up to `limit` active guardrail texts, most-applied first."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT text FROM guardrails WHERE active=1 ORDER BY times_applied DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row["text"] for row in rows]


def insert_guardrail(
    *,
    text: str,
    source_signature: str | None = None,
    db_path: Path | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO guardrails (text, source_failure_signature, created_at)
            VALUES (?,?,?)
            """,
            (text, source_signature, datetime.now(tz=UTC).isoformat()),
        )


def increment_guardrail_usage(guardrail_text: str, db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE guardrails SET times_applied = times_applied + 1 WHERE text = ?",
            (guardrail_text,),
        )


# ── Failure mode clustering ───────────────────────────────────────────────────

def upsert_failure_mode(
    *,
    signature: str,
    check_id: str,
    description: str,
    db_path: Path | None = None,
) -> int:
    """Insert or increment a failure mode. Returns the new occurrences count."""
    now = datetime.now(tz=UTC).isoformat()
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id, occurrences FROM failure_modes WHERE signature=?",
            (signature,),
        ).fetchone()
        if existing:
            new_count = int(existing["occurrences"]) + 1
            conn.execute(
                "UPDATE failure_modes SET occurrences=?, last_seen=? WHERE signature=?",
                (new_count, now, signature),
            )
            return new_count
        else:
            conn.execute(
                """
                INSERT INTO failure_modes
                    (signature, check_id, canonical_description, first_seen, last_seen)
                VALUES (?,?,?,?,?)
                """,
                (signature, check_id, description, now, now),
            )
            return 1


def fetch_failure_modes_for_promotion(
    min_occurrences: int = 3,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return failure modes that have crossed the promotion threshold but not yet promoted."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT signature, check_id, canonical_description, occurrences
            FROM failure_modes
            WHERE occurrences >= ? AND promoted_to_guardrail = 0
            ORDER BY occurrences DESC
            """,
            (min_occurrences,),
        ).fetchall()
    return [dict(row) for row in rows]


def mark_failure_mode_promoted(signature: str, db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE failure_modes SET promoted_to_guardrail=1 WHERE signature=?",
            (signature,),
        )


# ── Exemplars ─────────────────────────────────────────────────────────────────

def insert_exemplar(
    *,
    run_id: str,
    topic: str,
    lesson_md: str,
    passed_first_try: bool,
    db_path: Path | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO exemplars (run_id, topic, lesson_md, passed_first_try)
            VALUES (?,?,?,?)
            """,
            (run_id, topic, lesson_md, int(passed_first_try)),
        )


def fetch_best_exemplar(
    topic: str,
    db_path: Path | None = None,
) -> str | None:
    """Return the lesson_md of a first-try-pass exemplar for the topic (simple keyword match)."""
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT lesson_md FROM exemplars
            WHERE passed_first_try=1
              AND topic LIKE ?
            ORDER BY id DESC LIMIT 1
            """,
            (f"%{topic.split()[0]}%",),
        ).fetchone()
    return row["lesson_md"] if row else None
