"""Promoter — applies approved diagnoses and records versioned changes.

Responsibilities:
  1. Insert new guardrails into the memory DB (for prompt_rule fixes).
  2. Write prompt_version records for every prompt_rule change.
  3. Mark promoted failure modes.
  4. Print a human-readable summary of what was promoted.

Does NOT modify .md files in place — guardrails are injected at runtime via
the recall node, keeping the versioning in SQL rather than in files.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime

from lessonforge.config import AppConfig
from lessonforge.evolve.analyst import Diagnosis
from lessonforge.memory import db

logger = logging.getLogger(__name__)


def promote(
    diagnoses: list[Diagnosis],
    dry_run: bool = False,
    config: AppConfig | None = None,
    db_path: object = None,
) -> list[str]:
    """Apply approved diagnoses to the memory layer.

    Args:
        diagnoses: List of approved Diagnosis objects.
        dry_run:   If True, print what would change but do not write.
        config:    AppConfig.
        db_path:   Override DB path (used in tests).

    Returns:
        List of human-readable change descriptions.
    """
    changes: list[str] = []

    for diag in diagnoses:
        if diag.proposed_fix.fix_type == "prompt_rule" and diag.proposed_fix.guardrail_text:
            change = _promote_guardrail(diag, dry_run=dry_run, db_path=db_path)
            if change:
                changes.append(change)
        elif diag.proposed_fix.fix_type == "rubric_patch":
            change = f"[rubric_patch] {diag.proposed_fix.description} — manual review needed."
            changes.append(change)
            logger.info("[promoter] rubric_patch proposed: %s", diag.proposed_fix.description)
        elif diag.proposed_fix.fix_type == "new_checkpoint":
            change = f"[new_checkpoint] {diag.proposed_fix.description} — manual review needed."
            changes.append(change)
        else:
            change = f"[{diag.proposed_fix.fix_type}] {diag.proposed_fix.description}"
            changes.append(change)
            logger.info("[promoter] non-automated fix logged: %s", change)

    logger.info("[promoter] %d change(s) processed (dry_run=%s)", len(changes), dry_run)
    return changes


def _promote_guardrail(
    diag: Diagnosis,
    dry_run: bool,
    db_path: object,
) -> str | None:
    """Insert a guardrail into the DB and record a prompt_version entry."""
    guardrail_text = diag.proposed_fix.guardrail_text or ""
    if not guardrail_text.strip():
        return None

    description = (
        f"[guardrail] {diag.check_id} — {guardrail_text[:100]}"
    )

    if dry_run:
        logger.info("[promoter] DRY RUN — would insert guardrail: %s", guardrail_text)
        return f"[DRY RUN] {description}"

    try:
        db.insert_guardrail(
            text=guardrail_text,
            source_signature=diag.cluster_signature,
            db_path=db_path,  # type: ignore[arg-type]
        )
        db.mark_failure_mode_promoted(
            diag.cluster_signature,
            db_path=db_path,  # type: ignore[arg-type]
        )

        # Record in prompt_versions for full provenance
        sha = hashlib.sha256(guardrail_text.encode()).hexdigest()
        with db.connect(db_path) as conn:  # type: ignore[arg-type]
            conn.execute(
                """
                INSERT OR IGNORE INTO prompt_versions
                    (version, role, template, sha256, rationale, created_at, status)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    f"guardrail:{sha[:12]}",
                    "generator",
                    guardrail_text,
                    sha,
                    diag.rationale[:300],
                    datetime.now(tz=UTC).isoformat(),
                    "active",
                ),
            )
        logger.info("[promoter] guardrail promoted: %s", guardrail_text[:80])
        return description
    except Exception as exc:
        logger.error("[promoter] failed to promote guardrail: %s", exc)
        return None
