-- LessonForge Memory Layer — SQLite schema
-- Initialised once by db.py; versioned manually (schema v1).

CREATE TABLE IF NOT EXISTS runs (
  run_id             TEXT PRIMARY KEY,
  topic              TEXT NOT NULL,
  started_at         TEXT,
  finished_at        TEXT,
  outcome            TEXT,             -- shipped | escalated | error
  attempts_used      INTEGER,
  first_attempt_pass INTEGER,          -- 0/1 — headline learning metric
  prompt_version     TEXT DEFAULT 'v1',
  rubric_version     TEXT DEFAULT 'v1',
  corpus_version     TEXT,
  total_tokens       INTEGER DEFAULT 0,
  total_cost_usd     REAL    DEFAULT 0.0,
  wall_clock_s       REAL,
  injected_error     TEXT
);

CREATE TABLE IF NOT EXISTS attempts (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id           TEXT REFERENCES runs(run_id),
  attempt          INTEGER,
  lesson_md        TEXT,
  word_count       INTEGER,
  fk_grade         REAL,
  hard_fail_count  INTEGER,
  repair_plan_json TEXT
);

CREATE TABLE IF NOT EXISTS check_results (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id             TEXT,
  attempt            INTEGER,
  check_id           TEXT,
  dimension          TEXT,
  verdict            TEXT,
  severity           TEXT,
  judged_by          TEXT,
  evidence_quote     TEXT,
  reason             TEXT,
  repair_instruction TEXT,
  section_key        TEXT,
  vote_split         TEXT              -- for self-consistency checks
);

CREATE TABLE IF NOT EXISTS failure_modes (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  signature             TEXT UNIQUE,  -- e.g. "ACC-01::claims_rag_retrains_model"
  check_id              TEXT,
  canonical_description TEXT,
  occurrences           INTEGER DEFAULT 1,
  first_seen            TEXT,
  last_seen             TEXT,
  promoted_to_guardrail INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS guardrails (
  id                      INTEGER PRIMARY KEY AUTOINCREMENT,
  text                    TEXT NOT NULL,  -- injected into generator system prompt
  source_failure_signature TEXT,
  created_at              TEXT,
  active                  INTEGER DEFAULT 1,
  times_applied           INTEGER DEFAULT 0,
  effectiveness           REAL            -- recurrence rate after activation
);

CREATE TABLE IF NOT EXISTS exemplars (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id            TEXT,
  topic             TEXT,
  lesson_md         TEXT,
  passed_first_try  INTEGER,
  embedding         BLOB                  -- serialised numpy float32 vector
);

CREATE TABLE IF NOT EXISTS prompt_versions (
  version       TEXT PRIMARY KEY,
  role          TEXT,
  template      TEXT,
  sha256        TEXT,
  parent_version TEXT,
  rationale     TEXT,
  created_at    TEXT,
  eval_first_pass_rate REAL,
  status        TEXT                      -- candidate | active | rolled_back
);

CREATE TABLE IF NOT EXISTS rubric_versions (
  version        TEXT PRIMARY KEY,
  yaml           TEXT,
  sha256         TEXT,
  parent_version TEXT,
  rationale      TEXT,
  created_at     TEXT,
  status         TEXT                     -- candidate | active | rolled_back
);
