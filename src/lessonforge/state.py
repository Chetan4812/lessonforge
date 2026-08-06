from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ── Primitive aliases ─────────────────────────────────────────────────────────

CheckId = str  # e.g. "ACC-01"


# ── Learner persona ───────────────────────────────────────────────────────────

class LearnerPersona(BaseModel):
    label: str = "in_12th_grade_zero_background"
    education: str = "12th grade pass, India, non-English-medium schooling"
    english_level: str = (
        "CEFR A2-B1: understands short simple sentences; "
        "struggles with abstract nouns, idioms, latinate vocabulary"
    )
    prior_knowledge: list[str] = ["basic computer use", "has used ChatGPT once or twice"]
    unknown_terms: list[str] = [
        "embedding",
        "vector",
        "corpus",
        "inference",
        "parametric",
        "hallucination",
        "index",
        "token",
    ]
    motivation: str = "wants a job in AI, needs to feel the topic is learnable"
    reading_budget_minutes: int = 12


# ── Grounding ─────────────────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    id: str  # "S1"
    title: str
    url: str | None = None
    text: str
    sha256: str


class GroundingPack(BaseModel):
    chunks: list[SourceChunk]
    corpus_version: str


# ── Lesson structure ──────────────────────────────────────────────────────────

SectionKey = Literal[
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


class LessonSection(BaseModel):
    key: SectionKey
    heading: str
    body_md: str


class Lesson(BaseModel):
    topic: str
    title: str
    sections: list[LessonSection]

    def to_markdown(self) -> str:
        lines: list[str] = [f"# {self.title}\n"]
        for section in self.sections:
            lines.append(f"## {section.heading}\n")
            lines.append(section.body_md)
            lines.append("")
        return "\n".join(lines)


# ── Blueprint ─────────────────────────────────────────────────────────────────

class LessonBlueprint(BaseModel):
    learning_objectives: list[str] = Field(min_length=3, max_length=5)
    central_analogy: str
    worked_example_scenario: str
    must_define_terms: list[str]
    section_plan: list[dict]  # type: ignore[type-arg]
    out_of_scope: list[str]  # explicitly what NOT to cover — scope creep is a top failure mode


# ── Evaluation results ────────────────────────────────────────────────────────

class CheckResult(BaseModel):
    check_id: CheckId
    dimension: str
    verdict: Literal["PASS", "FAIL"]
    severity: Literal["hard", "advisory"]
    evidence_quote: str | None = None  # REQUIRED when verdict == FAIL
    reason: str
    repair_instruction: str | None = None
    section_key: str | None = None  # scopes the repair
    judged_by: str  # "deterministic" | "judge:accuracy" | "persona_probe"


class StructuralReport(BaseModel):
    results: list[CheckResult]
    metrics: dict  # type: ignore[type-arg]  # fk_grade, avg_sentence_len, jargon_hits, word_count, ...

    @property
    def all_passed(self) -> bool:
        return all(r.verdict == "PASS" for r in self.results)


class Verdict(BaseModel):
    attempt: int
    all_results: list[CheckResult]
    hard_fails: list[CheckResult]
    advisory_fails: list[CheckResult]
    ship_decision: Literal["SHIP", "RETRY", "ESCALATE"]


# ── Repair ────────────────────────────────────────────────────────────────────

class RepairItem(BaseModel):
    section_key: str
    problem: str
    instruction: str
    triggering_checks: list[CheckId]


class RepairPlan(BaseModel):
    attempt_for: int
    strategy: Literal["surgical", "full_rewrite"]
    items: list[RepairItem]
    keep_sections: list[str]  # explicitly preserved, do not touch


# ── Top-level run state ───────────────────────────────────────────────────────

class RunState(BaseModel):
    run_id: str
    topic: str
    started_at: datetime
    config: dict  # type: ignore[type-arg]
    persona: LearnerPersona = Field(default_factory=LearnerPersona)
    guardrails: list[str] = []  # from memory
    exemplar: str | None = None  # best past passing lesson, as few-shot
    blueprint: LessonBlueprint | None = None
    grounding: GroundingPack | None = None
    attempt: int = 1
    lesson: Lesson | None = None
    lesson_history: list[Lesson] = []
    structural_report: StructuralReport | None = None
    verdict: Verdict | None = None
    repair_plan: RepairPlan | None = None
    rejection_log: list[dict] = []  # type: ignore[type-arg]
    cost: dict = {}  # type: ignore[type-arg]
    injected_error: str | None = None  # demo mode
