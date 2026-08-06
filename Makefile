.PHONY: setup install lint typecheck test demo run evolve report clean

# ── Setup ─────────────────────────────────────────────────────────────────────

setup:
	pip install -e ".[dev]"

install:
	pip install -e .

# ── Quality ───────────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

typecheck:
	mypy src/

test:
	pytest tests/ -v --tb=short

ci: lint typecheck test

# ── Run ───────────────────────────────────────────────────────────────────────

# Demo mode: full loop with mock provider, no API key required
demo:
	lessonforge run --topic "Introduction to RAG" --provider mock --verbose

# Real run — requires API keys in .env
run:
	lessonforge run --topic "Introduction to RAG" --verbose

# Error injection demo (for the video)
demo-error:
	lessonforge run --topic "Introduction to RAG" --provider mock --inject-error factual --verbose

# ── Grounding ─────────────────────────────────────────────────────────────────

ground:
	lessonforge ground --topic "Introduction to RAG"

# ── Batch (M7+) ───────────────────────────────────────────────────────────────

batch:
	lessonforge batch --topics-file config/regression_topics.txt --provider mock

# ── Self-evolution (M8+) ──────────────────────────────────────────────────────

evolve:
	lessonforge evolve --since 7d

evolve-dry:
	lessonforge evolve --since 7d --dry-run

# ── Reporting ─────────────────────────────────────────────────────────────────

report:
	@echo "Usage: make report RUN_ID=<run_id>"
	lessonforge report --run-id $(RUN_ID) --open

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache
	rm -rf .cache/index.faiss
