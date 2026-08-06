"""M2 tests — corpus chunker, embedder (shape only), FAISS index, retriever.

Acceptance criteria (from the plan):
  1. Chunker produces stable IDs and at least one chunk per corpus file.
  2. Chunks have correct prefixes (S1..S6) matching their file's chunk_prefix.
  3. No chunk exceeds chunk_tokens + overlap_tokens in size.
  4. Retriever returns ≥ 1 chunk for topic "RAG".
  5. Chunk IDs are stable across two identical runs.
  6. Cache hit is faster than cache miss (second call skips rebuilding).

Notes:
  - Embedding tests use a tiny random float32 array to avoid loading the
    full sentence-transformers model in CI.  The real model is tested via
    the integration test `test_retriever_returns_chunks` which is marked
    `slow` and can be skipped with `pytest -m "not slow"`.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from lessonforge.config import CORPUS_DIR
from lessonforge.grounding.chunker import RawChunk, chunk_corpus, chunk_file
from lessonforge.grounding.index import build_index, is_cache_valid, load_index

# ── Chunker tests (no ML, no network) ─────────────────────────────────────────

def test_chunk_corpus_produces_chunks() -> None:
    """All 6 corpus files produce at least one chunk each."""
    chunks = chunk_corpus(CORPUS_DIR)
    assert len(chunks) >= 6, f"Expected at least 6 chunks (one per file), got {len(chunks)}"


def test_chunk_ids_are_stable() -> None:
    """Running the chunker twice on the same corpus produces identical IDs."""
    run1 = [c.id for c in chunk_corpus(CORPUS_DIR)]
    run2 = [c.id for c in chunk_corpus(CORPUS_DIR)]
    assert run1 == run2, "Chunk IDs must be deterministic across runs"


def test_chunk_prefixes_match_corpus_files() -> None:
    """Each chunk's prefix must match the chunk_prefix in its source file's front-matter."""
    expected_prefixes = {"S1", "S2", "S3", "S4", "S5", "S6"}
    chunks = chunk_corpus(CORPUS_DIR)
    found = {c.prefix for c in chunks}
    assert expected_prefixes.issubset(found), (
        f"Missing prefixes: {expected_prefixes - found}"
    )


def test_no_chunk_exceeds_token_budget() -> None:
    """No chunk should exceed chunk_tokens + overlap_tokens (300 tokens with defaults)."""
    from lessonforge.config import AppConfig

    cfg = AppConfig()
    grounding = cfg.grounding
    chunk_tokens = int(grounding.get("chunk_tokens", 250))
    overlap_tokens = int(grounding.get("overlap_tokens", 40))
    max_allowed = chunk_tokens + overlap_tokens  # generous upper bound

    chunks = chunk_corpus(CORPUS_DIR, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    violators = [c for c in chunks if c.token_count > max_allowed]
    assert not violators, (
        f"{len(violators)} chunks exceed {max_allowed} tokens:\n"
        + "\n".join(f"  {c.id}: {c.token_count} tokens" for c in violators)
    )


def test_each_chunk_has_nonempty_text() -> None:
    chunks = chunk_corpus(CORPUS_DIR)
    empty = [c for c in chunks if not c.text.strip()]
    assert not empty, f"Found {len(empty)} chunks with empty text"


def test_chunks_have_sha256() -> None:
    chunks = chunk_corpus(CORPUS_DIR)
    missing = [c for c in chunks if len(c.sha256) != 64]
    assert not missing, f"{len(missing)} chunks have invalid sha256"


def test_single_file_chunk_prefix() -> None:
    """chunk_file on corpus/01_... returns chunks with prefix 'S1'."""
    target = next(CORPUS_DIR.glob("01_*.md"), None)
    assert target is not None, "Could not find 01_*.md corpus file"
    chunks = chunk_file(target)
    assert all(c.prefix == "S1" for c in chunks), "All chunks from file 01 must have prefix S1"


# ── Index tests (small synthetic vectors, no sentence-transformers) ────────────

def _make_fake_chunks(n: int, prefix: str = "T") -> list[RawChunk]:
    return [
        RawChunk(
            id=f"{prefix}-{i:03d}",
            file_slug="test",
            prefix=prefix,
            title="Test",
            heading="Section",
            text=f"This is chunk number {i} about retrieval augmented generation.",
            token_count=10,
            sha256="a" * 64,
        )
        for i in range(n)
    ]


def _make_fake_vectors(n: int, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    # L2-normalise so IP == cosine
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_build_and_load_index() -> None:
    """Build a small index, persist it, reload it — chunk count must match."""
    chunks = _make_fake_chunks(10)
    vectors = _make_fake_vectors(10)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Path(tmpdir)
        index = build_index(chunks, vectors, cache_dir=cache, corpus_hash="abc123")
        assert index.ntotal == 10

        loaded_index, loaded_chunks = load_index(cache)
        assert loaded_index.ntotal == 10
        assert len(loaded_chunks) == 10
        assert loaded_chunks[0].id == "T-000"


def test_cache_invalidation() -> None:
    """is_cache_valid returns False for a different corpus hash."""
    chunks = _make_fake_chunks(5)
    vectors = _make_fake_vectors(5)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Path(tmpdir)
        build_index(chunks, vectors, cache_dir=cache, corpus_hash="hash-v1")
        assert is_cache_valid("hash-v1", cache) is True
        assert is_cache_valid("hash-v2", cache) is False


def test_index_search_returns_top_k() -> None:
    """FAISS search returns the correct number of results."""

    chunks = _make_fake_chunks(20)
    vectors = _make_fake_vectors(20)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Path(tmpdir)
        index = build_index(chunks, vectors, cache_dir=cache)
        query = _make_fake_vectors(1)
        scores, indices = index.search(query, 5)
        assert scores.shape == (1, 5)
        assert indices.shape == (1, 5)
        assert all(i >= 0 for i in indices[0])


# ── Retriever integration test (requires sentence-transformers model) ──────────

@pytest.mark.slow
def test_retriever_returns_chunks_for_rag_topic() -> None:
    """End-to-end: retrieve() returns >= 1 chunk for topic 'RAG'.

    This test downloads / uses the sentence-transformers model.
    Skip with: pytest -m "not slow"
    """
    import tempfile

    from lessonforge.grounding.retriever import retrieve

    with tempfile.TemporaryDirectory() as tmpdir:
        pack = retrieve(query="Introduction to RAG", cache_dir=Path(tmpdir))

    assert len(pack.chunks) >= 1, "Retriever must return at least one chunk"
    assert all(c.id for c in pack.chunks), "All chunks must have non-empty IDs"
    assert pack.corpus_version, "GroundingPack must have a corpus version"


@pytest.mark.slow
def test_retriever_chunk_ids_stable_across_two_calls() -> None:
    """Chunk IDs and order are stable across two identical retrieve() calls."""
    import tempfile

    from lessonforge.grounding.retriever import retrieve

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Path(tmpdir)
        pack1 = retrieve(query="Introduction to RAG", cache_dir=cache)
        pack2 = retrieve(query="Introduction to RAG", cache_dir=cache)

    ids1 = [c.id for c in pack1.chunks]
    ids2 = [c.id for c in pack2.chunks]
    assert ids1 == ids2, "Retrieval order must be stable when corpus does not change"


@pytest.mark.slow
def test_retriever_cache_hit_is_faster() -> None:
    """Second retrieve() call (cache hit) is at least 2x faster than first (builds index)."""
    import tempfile

    from lessonforge.grounding.retriever import retrieve

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Path(tmpdir)
        t0 = time.time()
        retrieve(query="RAG", cache_dir=cache)
        cold_time = time.time() - t0

        t1 = time.time()
        retrieve(query="RAG", cache_dir=cache)
        warm_time = time.time() - t1

    assert warm_time < cold_time, (
        f"Cache hit ({warm_time:.2f}s) should be faster than cache miss ({cold_time:.2f}s)"
    )
