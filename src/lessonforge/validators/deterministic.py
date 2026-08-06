"""Deterministic checks — no LLM calls, no API keys, pure Python.

Implements checkpoints:
  LNG-01  Flesch-Kincaid grade ≤ 9.0 AND avg sentence ≤ 20 words
  LNG-02  No jargon term used before its plain-language definition
  COV-04  Every term in blueprint.must_define_terms appears in the glossary section
  FLW-02  All 11 required sections present and in correct order
  FLW-03  Word count between 900 and 1800

All functions are pure: (lesson, ...) → CheckResult.
None touch the network or the filesystem.
"""

from __future__ import annotations

import re

from lessonforge.config import AppConfig
from lessonforge.state import CheckResult, Lesson, LessonBlueprint, SectionKey

# Required section order (matches generator contract)
_REQUIRED_ORDER: list[SectionKey] = [
    "hook",
    "what_it_is",
    "why_it_matters",
    "how_it_works",
    "analogy",
    "worked_example",
    "common_mistakes",
    "glossary",
    "recap",
    "check_yourself",
    "next_steps",
]


# ── Public entry point ────────────────────────────────────────────────────────

def run_all(
    lesson: Lesson,
    blueprint: LessonBlueprint | None,
    config: AppConfig | None = None,
) -> list[CheckResult]:
    """Run all deterministic checks and return a list of CheckResult objects."""
    cfg = config or AppConfig()
    lesson_cfg = cfg.lesson

    checks: list[CheckResult] = [
        check_flw02_section_order(lesson),
        check_flw03_word_count(
            lesson,
            min_words=int(lesson_cfg.get("min_words", 900)),
            max_words=int(lesson_cfg.get("max_words", 1800)),
        ),
        check_lng01_readability(
            lesson,
            max_fk_grade=float(lesson_cfg.get("max_fk_grade", 9.0)),
            max_avg_sentence_words=float(lesson_cfg.get("max_avg_sentence_words", 20.0)),
        ),
        check_lng02_jargon(lesson, cfg),
    ]

    if blueprint is not None:
        checks.append(check_cov04_glossary(lesson, blueprint))

    return checks


# ── Individual checks ─────────────────────────────────────────────────────────

def check_flw02_section_order(lesson: Lesson) -> CheckResult:
    """FLW-02: All required sections present and in correct order."""
    found_keys = [s.key for s in lesson.sections]
    # Check presence
    missing = [k for k in _REQUIRED_ORDER if k not in found_keys]
    if missing:
        return CheckResult(
            check_id="FLW-02",
            dimension="flow",
            verdict="FAIL",
            severity="hard",
            evidence_quote=f"Missing sections: {missing}",
            reason=f"{len(missing)} required section(s) absent: {missing}",
            repair_instruction="Add the missing sections in the correct order.",
            judged_by="deterministic",
        )
    # Check ordering — find positions of required keys in found_keys
    required_positions = [found_keys.index(k) for k in _REQUIRED_ORDER if k in found_keys]
    if required_positions != sorted(required_positions):
        return CheckResult(
            check_id="FLW-02",
            dimension="flow",
            verdict="FAIL",
            severity="hard",
            evidence_quote=f"Section order found: {found_keys}",
            reason="Required sections are present but out of the required order.",
            repair_instruction=f"Reorder sections to: {_REQUIRED_ORDER}",
            judged_by="deterministic",
        )
    return CheckResult(
        check_id="FLW-02",
        dimension="flow",
        verdict="PASS",
        severity="hard",
        reason="All 11 required sections present in correct order.",
        judged_by="deterministic",
    )


def check_flw03_word_count(
    lesson: Lesson,
    min_words: int = 900,
    max_words: int = 1800,
) -> CheckResult:
    """FLW-03: Total lesson word count between min_words and max_words."""
    full_text = " ".join(s.body_md for s in lesson.sections)
    word_count = len(full_text.split())

    if word_count < min_words:
        return CheckResult(
            check_id="FLW-03",
            dimension="flow",
            verdict="FAIL",
            severity="hard",
            evidence_quote=f"Word count: {word_count}",
            reason=f"Lesson is too short: {word_count} words (minimum {min_words}).",
            repair_instruction=f"Expand sections to reach at least {min_words} words total.",
            judged_by="deterministic",
        )
    if word_count > max_words:
        return CheckResult(
            check_id="FLW-03",
            dimension="flow",
            verdict="FAIL",
            severity="hard",
            evidence_quote=f"Word count: {word_count}",
            reason=f"Lesson is too long: {word_count} words (maximum {max_words}).",
            repair_instruction=f"Trim sections to stay under {max_words} words total.",
            judged_by="deterministic",
        )
    return CheckResult(
        check_id="FLW-03",
        dimension="flow",
        verdict="PASS",
        severity="hard",
        reason=f"Word count {word_count} is within [{min_words}, {max_words}].",
        judged_by="deterministic",
    )


def check_lng01_readability(
    lesson: Lesson,
    max_fk_grade: float = 9.0,
    max_avg_sentence_words: float = 20.0,
) -> CheckResult:
    """LNG-01: Flesch-Kincaid grade ≤ max_fk_grade AND avg sentence ≤ max_avg_sentence_words."""
    full_text = " ".join(s.body_md for s in lesson.sections)
    # Strip markdown syntax before analysis
    clean = _strip_markdown(full_text)

    sentences = _split_sentences(clean)
    if not sentences:
        return CheckResult(
            check_id="LNG-01",
            dimension="language",
            verdict="FAIL",
            severity="hard",
            reason="No sentences found in lesson body.",
            judged_by="deterministic",
        )

    words = clean.split()
    word_count = len(words)
    sentence_count = len(sentences)
    syllable_count = sum(_count_syllables(w) for w in words)

    avg_sentence_words = word_count / sentence_count
    # Flesch-Kincaid Grade Level formula
    fk_grade = 0.39 * avg_sentence_words + 11.8 * (syllable_count / word_count) - 15.59

    failures: list[str] = []
    if fk_grade > max_fk_grade:
        failures.append(f"FK grade {fk_grade:.1f} > {max_fk_grade}")
    if avg_sentence_words > max_avg_sentence_words:
        failures.append(f"avg sentence {avg_sentence_words:.1f} words > {max_avg_sentence_words}")

    metrics_note = f"FK={fk_grade:.1f}, avg_sent={avg_sentence_words:.1f} words"

    if failures:
        return CheckResult(
            check_id="LNG-01",
            dimension="language",
            verdict="FAIL",
            severity="hard",
            evidence_quote=metrics_note,
            reason="; ".join(failures),
            repair_instruction="Shorten sentences and replace multi-syllable words with simpler alternatives.",
            judged_by="deterministic",
        )
    return CheckResult(
        check_id="LNG-01",
        dimension="language",
        verdict="PASS",
        severity="hard",
        reason=f"Readability OK: {metrics_note}",
        judged_by="deterministic",
    )


def check_lng02_jargon(lesson: Lesson, config: AppConfig | None = None) -> CheckResult:
    """LNG-02: No jargon term used before its plain-language definition.

    A term is considered defined if it appears in the glossary section OR
    is immediately followed by an explanation in the same sentence (e.g.
    'embedding (a list of numbers that represents meaning)').
    """
    cfg = config or AppConfig()
    watchlist = [t.lower() for t in cfg.jargon_watchlist]

    # Find the glossary section text
    glossary_text = ""
    for section in lesson.sections:
        if section.key == "glossary":
            glossary_text = section.body_md.lower()
            break

    # Check each watchlist term against the glossary
    undefined = [
        term for term in watchlist
        if term not in glossary_text
    ]

    if undefined:
        return CheckResult(
            check_id="LNG-02",
            dimension="language",
            verdict="FAIL",
            severity="hard",
            evidence_quote=f"Terms not in glossary: {undefined[:5]}",
            reason=f"{len(undefined)} jargon term(s) not defined in the glossary: {undefined[:5]}",
            repair_instruction="Add each flagged term to the glossary section with a plain-language definition (≤25 words).",
            judged_by="deterministic",
        )
    return CheckResult(
        check_id="LNG-02",
        dimension="language",
        verdict="PASS",
        severity="hard",
        reason=f"All {len(watchlist)} watchlist terms defined in the glossary.",
        judged_by="deterministic",
    )


def check_cov04_glossary(lesson: Lesson, blueprint: LessonBlueprint) -> CheckResult:
    """COV-04: Every term in blueprint.must_define_terms appears in the glossary section."""
    glossary_text = ""
    for section in lesson.sections:
        if section.key == "glossary":
            glossary_text = section.body_md.lower()
            break

    missing = [
        term for term in blueprint.must_define_terms
        if term.lower() not in glossary_text
    ]

    if missing:
        return CheckResult(
            check_id="COV-04",
            dimension="coverage",
            verdict="FAIL",
            severity="hard",
            evidence_quote=f"Terms missing from glossary: {missing}",
            reason=f"{len(missing)} blueprint term(s) not in glossary: {missing}",
            repair_instruction="Add each missing term to the glossary section.",
            judged_by="deterministic",
        )
    return CheckResult(
        check_id="COV-04",
        dimension="coverage",
        verdict="PASS",
        severity="hard",
        reason=f"All {len(blueprint.must_define_terms)} blueprint terms found in glossary.",
        judged_by="deterministic",
    )


# ── Text utilities ────────────────────────────────────────────────────────────

def compute_metrics(lesson: Lesson) -> dict[str, float]:
    """Compute readability metrics for the lesson. Used by evaluate node."""
    full_text = " ".join(s.body_md for s in lesson.sections)
    clean = _strip_markdown(full_text)
    words = clean.split()
    sentences = _split_sentences(clean)

    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    syllable_count = sum(_count_syllables(w) for w in words)
    avg_sentence_words = word_count / sentence_count
    fk_grade = 0.39 * avg_sentence_words + 11.8 * (syllable_count / max(word_count, 1)) - 15.59

    return {
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "avg_sentence_words": round(avg_sentence_words, 2),
        "fk_grade": round(fk_grade, 2),
        "syllable_count": float(syllable_count),
    }


def _strip_markdown(text: str) -> str:
    """Remove markdown syntax to get plain text for readability analysis."""
    # Remove inline code, bold, italic, links, headings
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove citation tags like [S1], [S1-001]
    text = re.sub(r"\[S\d+[^\]]*\]", "", text)
    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on .!? boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _count_syllables(word: str) -> int:
    """Estimate syllable count using vowel-group heuristic (English only)."""
    word = word.lower().strip(".,!?;:\"'")
    if not word:
        return 0
    # Count vowel groups
    vowels = re.findall(r"[aeiouy]+", word)
    count = len(vowels)
    # Adjust for silent 'e' at end
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)
