# LessonForge

A deterministic-where-possible, LLM-where-necessary **agentic content pipeline** that takes a topic and ships a beginner lesson only after it survives a hard pass/fail rubric gate.

---

## Quick start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Copy and fill in API keys
cp .env.example .env

# 3. Run with Groq (free tier)
lessonforge run --topic "Retrieval-Augmented Generation" --provider groq

# 4. Run with mock provider (no API key needed — works in CI)
lessonforge run --topic "Retrieval-Augmented Generation" --provider mock

# 5. Run the self-evolution job (after collecting ≥2 runs of data)
lessonforge evolve --dry-run
```

---

## Architecture

```
TOPIC
  │
  ▼
[recall] ──► [ground] ──► [blueprint] ──► [generate] ──► [evaluate]
                                               ▲               │
                                               │ RETRY         │ SHIP/ESCALATE
                                          [repair] ◄───────────┘
                                                                │
                                                           [persist]
                                                                │
                                                          [write outputs]
```

### Nodes

| Node | Type | Responsibility |
|---|---|---|
| `recall` | deterministic | Load active guardrails + best exemplar from SQLite |
| `ground` | deterministic + embed | Retrieve top-k corpus chunks via FAISS |
| `blueprint` | LLM | Produce typed `LessonBlueprint` with objectives, analogy, worked example |
| `generate` | LLM | Write the full lesson in 11-section JSON schema |
| `evaluate` | deterministic + LLM ×4 + persona probe | 16 hard pass/fail checkpoints |
| `repair` | LLM | Surgical or full-rewrite repair targeting only failing sections |
| `persist` | deterministic | Write run data to SQLite; auto-promote recurring failures to guardrails |

---

## Commands

### `lessonforge run`

```
lessonforge run --topic "RAG" [--provider groq|mock] [-v]
```

Runs the full generate → evaluate → [repair → evaluate]* pipeline.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--topic` | required | The lesson topic |
| `--provider` | `groq` | LLM provider (`groq` or `mock`) |
| `--inject-error` | — | Inject a named defect for demo purposes |
| `-v` | — | Verbose logging |

Outputs to `out/<run_id>/`:
- `lesson_final.md` — the shipped lesson
- `report.json` — rubric matrix with all check verdicts
- `trace.json` — full structured trace (reproducible)

### `lessonforge evolve`

```
lessonforge evolve [--since 30d] [--dry-run] [--auto] [--provider groq]
```

Mines failure data, calls the LLM analyst, proposes guardrail fixes, and (with approval) promotes them to the memory layer so future runs avoid the same failures.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--since` | `30d` | Mine failures from the last N days |
| `--dry-run` | `false` | Show proposals without writing |
| `--auto` | `false` | Skip human approval gate (CI mode) |
| `--top-n` | `10` | Max failure clusters to analyse |

---

## The rubric — 16 hard pass/fail checkpoints

| ID | Dimension | Type | Description |
|---|---|---|---|
| ACC-01 | Accurate | hard | No factual errors about RAG |
| ACC-02 | Accurate | hard | Every factual claim is source-tagged `[Sx]` |
| ACC-03 | Accurate | hard | No invented statistics or product names |
| ACC-04 | Accurate | hard | Honest about limitations |
| LNG-01 | Language | hard | FK grade ≤ 9.0, avg sentence ≤ 20 words |
| LNG-02 | Language | hard | No unexplained jargon |
| LNG-03 | Language | hard | No idioms / culture-specific references |
| LNG-04 | Language | advisory | ≤ 15% of sentences have 2+ subordinate clauses |
| EXM-01 | Pedagogy | hard | ≥ 1 concrete end-to-end worked example |
| EXM-02 | Pedagogy | hard | Example uses culturally grounded scenario |
| EXM-03 | Pedagogy | hard | ≥ 1 analogy for the core mechanism |
| EXM-04 | Pedagogy | hard | Contrast case present |
| COV-01 | Coverage | hard | Plain RAG definition in first 200 words |
| COV-02 | Coverage | hard | Explains why RAG matters |
| COV-03 | Coverage | hard | All 5 pipeline stages explained |
| COV-04 | Coverage | hard | Every blueprint term in glossary |
| FLW-01 | Flow | hard | No forward references |
| FLW-02 | Flow | hard | All 11 sections present and ordered |
| FLW-03 | Flow | hard | 900–1800 words |
| PRB-01 | Comprehension | hard | Simulated learner scores ≥ 4/5 questions |

---

## Memory (three tiers)

| Tier | Store | What it changes |
|---|---|---|
| **Episodic** | `runs`, `attempts`, `check_results` tables | Nothing directly — raw evidence base |
| **Semantic** | `exemplars` table | Injected as few-shot example into generator |
| **Procedural** | `guardrails`, `failure_modes` tables | Directly rewrites generator system prompt |

**Guardrail promotion:** when a failure signature recurs ≥ 3 times, it is automatically promoted to a standing guardrail injected on every subsequent run.

---

## Self-evolution pipeline

```
1. MINE      Pull check_results from SQLite. Cluster hard fails by signature.
2. DIAGNOSE  LLM analyst classifies root cause (generator_prompt_gap /
             rubric_ambiguity / missing_checkpoint / grounding_gap / judge_over_strictness).
3. PROPOSE   Emit candidate guardrail text or rubric patch.
4. GATE      Human approval (or --auto). Minimum: ≥5pp first-pass improvement.
5. PROMOTE   Insert guardrail into DB. Mark failure mode as promoted.
             Next run picks it up automatically via the recall node.
```

---

## Environment variables

```env
GROQ_API_KEY=...          # Required for --provider groq
ANTHROPIC_API_KEY=...     # Optional: for Claude judges
OPENAI_API_KEY=...        # Optional: for OpenAI models
```

---

## Testing

```bash
# Fast unit tests (no API calls)
pytest tests/ -m "not slow" -q

# Slow integration tests (real API calls)
pytest tests/ -m slow -v

# Full quality gate
ruff check src/ tests/ && mypy src/ && pytest tests/ -m "not slow"
```

**Test counts by milestone:**

| Milestone | Tests | Focus |
|---|---|---|
| M2 Grounding | 3 slow | Corpus embedding + retrieval |
| M3 Blueprint | ~15 | Blueprint node + schema |
| M4 Generator | ~15 | Generator node + 11-section schema |
| M5 Evaluate | ~20 | All 16 deterministic + LLM checks |
| M6 Repair | ~15 | Surgical/full-rewrite strategies |
| M7 Persona Probe | ~10 | PRB-01 comprehension check |
| M8 Pipeline + CLI | ~15 | End-to-end pipeline + CLI |
| M9 Memory | 17 | SQLite, recall node, persist node |
| M10 Evolve | 15 | Miner, analyst, promoter, CLI |

---

## Design rationale

- **Cheap gates before expensive gates.** Deterministic validators run first (free, fast, ~55% of catches) before LLM judges.
- **Binary means binary.** Every checkpoint is PASS or FAIL. No 1–5 scores.
- **A failing judge must show its receipts.** Every FAIL requires a verbatim `evidence_quote` substring from the lesson.
- **The judge is blind.** Evaluators never see the attempt number or prior verdicts.
- **Repair, don't reroll.** Surgical repair preserves passing sections; `keep_sections` are reproduced verbatim.
- **Memory changes behaviour, or it isn't memory.** Failure modes that recur ≥ 3 times automatically become standing guardrails.
- **Everything is reproducible.** Model IDs pinned, temperature/seed recorded, prompt SHA recorded, corpus snapshot hashed.
