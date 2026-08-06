# Working agreement for Antigravity (the coding agent)

## Non-negotiables

- Python 3.11+. Full type hints. Pydantic v2 for every cross-node data structure.
- No dict-passing between nodes. Everything goes through `RunState`.
- No prompt text inside `.py` files. All prompts live in `prompts/*.md` and are loaded
  by name with their SHA recorded in the trace.
- No magic numbers. Every threshold comes from `config/settings.yaml`.
- Every node is a pure-ish function `(RunState) -> RunState` and independently testable.
- Never call a model provider directly. Always go through `llm/gateway.py`.
- Every new module ships with its unit test in the same commit.

## Definition of done for a milestone

- Acceptance criteria in the plan are met and demonstrated by a test.
- `ruff check`, `mypy src`, and `pytest` all pass.
- README updated if behaviour or commands changed.

## Style

- Fail closed. An inconclusive check is a FAIL, never a PASS.
- Prefer deterministic code over an LLM call wherever the check is mechanical.
- Log a structured trace event at the start and end of every node.

## Milestone ordering

Execute milestones strictly in order (M0 → M10).
Do not start a milestone until the previous one's acceptance criteria pass and the user
has reviewed and approved the output.

## File layout rules

- Prompts: `prompts/<role>.md`
- Config: `config/settings.yaml` (thresholds), `config/rubric.v1.yaml` (checkpoints)
- Corpus: `corpus/0N_<slug>.md` with YAML front-matter
- Output: `out/<run_id>/`
- Tests: `tests/unit/` for pure-function tests, `tests/test_*.py` for integration
