"""Failure miner — queries SQLite for recurring failure patterns.

Reads from:  check_results, failure_modes
Writes to:   nothing (read-only)

Returns a ranked list of failure clusters for the evolve analyst.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from lessonforge.memory import db

logger = logging.getLogger(__name__)


@dataclass
class FailureCluster:
    """A deduplicated, ranked failure pattern."""

    signature: str
    check_id: str
    canonical_description: str
    occurrences: int
    first_seen: str
    last_seen: str


def mine(
    since_days: int = 30,
    min_occurrences: int = 2,
    top_n: int = 10,
    db_path: object = None,
) -> list[FailureCluster]:
    """Mine the database for recurring failure patterns.

    Args:
        since_days:      Only count failures from the last N days.
        min_occurrences: Minimum occurrence count to include a cluster.
        top_n:           Return at most top_n clusters.
        db_path:         Override DB path (used in tests).

    Returns:
        A ranked list of FailureCluster objects, most frequent first.
    """
    cutoff = (datetime.now(tz=UTC) - timedelta(days=since_days)).isoformat()

    with db.connect(db_path) as conn:  # type: ignore[arg-type]
        rows = conn.execute(
            """
            SELECT signature, check_id, canonical_description, occurrences,
                   first_seen, last_seen
            FROM failure_modes
            WHERE last_seen >= ?
              AND occurrences >= ?
            ORDER BY occurrences DESC
            LIMIT ?
            """,
            (cutoff, min_occurrences, top_n),
        ).fetchall()

    clusters = [
        FailureCluster(
            signature=row["signature"],
            check_id=row["check_id"],
            canonical_description=row["canonical_description"],
            occurrences=int(row["occurrences"]),
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
        )
        for row in rows
    ]

    logger.info("[miner] found %d failure cluster(s)", len(clusters))
    return clusters


def first_attempt_pass_rate(since_days: int = 30, db_path: object = None) -> float:
    """Compute the first-attempt pass rate over recent runs (0.0–1.0)."""
    cutoff = (datetime.now(tz=UTC) - timedelta(days=since_days)).isoformat()

    with db.connect(db_path) as conn:  # type: ignore[arg-type]
        row = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(first_attempt_pass) AS passed
            FROM runs
            WHERE started_at >= ?
              AND outcome IN ('shipped', 'escalated')
            """,
            (cutoff,),
        ).fetchone()

    total = int(row["total"] or 0)
    passed = int(row["passed"] or 0)
    rate = passed / total if total > 0 else 0.0
    logger.info("[miner] first-attempt pass rate: %.1f%% (%d/%d)", rate * 100, passed, total)
    return rate
