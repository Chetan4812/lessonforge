"""Output writer — persists the final lesson and evaluation report to disk.

Writes to: out/<run_id>/
  - lesson.md          — final lesson in readable Markdown
  - report.json        — structured evaluation results
  - trace.json         — full RunState serialised for debugging / replay

All paths come from config.  No magic strings.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from lessonforge.config import OUT_DIR, AppConfig
from lessonforge.render.html_report import render_html_report
from lessonforge.state import RunState

logger = logging.getLogger(__name__)


def write_outputs(state: RunState, config: AppConfig | None = None) -> Path:
    """Write all output artefacts for the completed run.

    Args:
        state:  Final RunState (must have state.lesson set).
        config: AppConfig.

    Returns:
        The run output directory path.
    """
    run_dir = OUT_DIR / state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_lesson_md(state, run_dir)
    _write_report_json(state, run_dir)
    _write_trace_json(state, run_dir)
    report_path = render_html_report(state, run_dir)

    logger.info("[output] wrote artefacts to %s", run_dir)
    logger.info("[output] report.html  → %s", report_path)
    return run_dir


# ── Writers ───────────────────────────────────────────────────────────────────

def _write_lesson_md(state: RunState, run_dir: Path) -> None:
    """Write the final lesson as Markdown."""
    if state.lesson is None:
        logger.warning("[output] state.lesson is None — skipping lesson.md")
        return

    path = run_dir / "lesson.md"
    md = state.lesson.to_markdown()
    path.write_text(md, encoding="utf-8")
    logger.info("[output] lesson.md  (%d words)", len(md.split()))


def _write_report_json(state: RunState, run_dir: Path) -> None:
    """Write the evaluation report as JSON."""
    report: dict = {  # type: ignore[type-arg]
        "run_id": state.run_id,
        "topic": state.topic,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "attempt": state.attempt,
        "verdict": state.verdict.ship_decision if state.verdict else "UNKNOWN",
        "metrics": state.structural_report.metrics if state.structural_report else {},
        "checks": [],
        "rejection_log": state.rejection_log,
    }

    if state.structural_report:
        for r in state.structural_report.results:
            report["checks"].append({
                "check_id": r.check_id,
                "dimension": r.dimension,
                "verdict": r.verdict,
                "severity": r.severity,
                "reason": r.reason,
                "evidence_quote": r.evidence_quote,
                "judged_by": r.judged_by,
            })

    path = run_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[output] report.json  (%d checks)", len(report["checks"]))


def _write_trace_json(state: RunState, run_dir: Path) -> None:
    """Write the full RunState as JSON for debugging and replay."""
    path = run_dir / "trace.json"
    # Use model_dump for Pydantic v2 serialisation
    data = state.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[output] trace.json")
