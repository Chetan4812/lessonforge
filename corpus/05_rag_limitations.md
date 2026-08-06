---
title: "RAG Limitations and Failure Modes"
slug: rag_limitations
chunk_prefix: S5
sources:
  - url: "https://arxiv.org/abs/2401.15884"
    citation: "Barnett, S. et al. Seven Failure Points When Engineering a Retrieval Augmented Generation System. ICSE 2024."
  - url: "https://arxiv.org/abs/2005.11401"
    citation: "Lewis et al., NeurIPS 2020."
  - url: "https://towardsdatascience.com/rag-pitfalls-and-how-to-avoid-them-a37c0d14f2a0"
    citation: "Towards Data Science: RAG Pitfalls and How to Avoid Them, 2024."
---

## RAG Is Not a Silver Bullet

RAG significantly improves the factual accuracy and traceability of LLM outputs for many tasks, but it introduces its own failure modes. Understanding these limitations is essential for building a trustworthy system.

## Failure Mode 1: Retrieval Miss

**What happens:** The relevant information exists in the document store but the retrieval step does not return it. The retrieved chunks do not contain the answer.

**Why it happens:**
- The embedding of the user's question is not similar enough to the embedding of the relevant passage, even though they are semantically related. This can occur when the question uses different vocabulary than the document.
- The chunk containing the answer was split at an awkward boundary, losing the context that makes it retrievable.
- k (the number of retrieved chunks) is too small to include the relevant passage.

**Consequence:** The generator either answers from its parametric memory (potentially hallucinating) or correctly states it does not know.

## Failure Mode 2: Retrieval of Wrong Passage

**What happens:** The retriever returns passages that are topically related but factually irrelevant or misleading for the specific question.

**Why it happens:** Embedding similarity captures topic similarity, not logical relevance. A passage about "RAG limitations" will score similarly for many different RAG questions, even if it does not answer the specific question asked.

**Consequence:** The generator is conditioned on irrelevant context and may produce a confident but incorrect answer.

## Failure Mode 3: Chunking Destroys Context

**What happens:** The relevant answer spans two chunks, but only one of them is retrieved. The retrieved chunk is incomplete and misleading without the adjacent context.

**Why it happens:** Fixed-size chunking does not respect semantic boundaries. A 250-token chunk may end in the middle of an important explanation.

**Mitigation:** Overlapping chunks reduce (but do not eliminate) this problem. Parent-child chunking (retrieve small chunks, return larger context windows) is a more advanced fix.

## Failure Mode 4: Document Store Is Stale or Incomplete

**What happens:** The document store is missing key information, or contains outdated information that contradicts the current truth.

**Why it happens:** RAG does not automatically update its document store. If documents are not maintained and re-indexed regularly, the system answers from outdated sources.

**Consequence:** The generator produces answers that were once correct but are now wrong. This is one of the hardest failure modes to detect without human review.

## Failure Mode 5: Generator Ignores Retrieved Context

**What happens:** The generator reads the retrieved passages but produces an answer that contradicts them or ignores them entirely in favour of its parametric knowledge.

**Why it happens:** Large language models are trained on vast amounts of text and have strong prior beliefs about many topics. When the retrieved context conflicts with the model's prior beliefs, some models give more weight to their training than to the retrieved evidence.

**Mitigation:** Explicit prompting helps: "Answer ONLY from the provided context. If the context does not contain the answer, say 'I don't know'." Smaller context windows (fewer retrieved chunks) can also reduce the model's tendency to drift.

## Failure Mode 6: Context Window Overflow

**What happens:** The total length of the retrieved chunks plus the system prompt plus the user question exceeds the model's context window limit.

**Why it happens:** Retrieving too many chunks (large k) or using an LLM with a small context window (e.g. 4k tokens) makes overflow likely.

**Consequence:** The API returns an error or silently truncates the context, potentially losing the most relevant chunks.

**Mitigation:** Cap the total grounding pack size in tokens (e.g. max 3,500 tokens for context, leaving room for the system prompt and the generated answer).

## Failure Mode 7: Hallucination Still Occurs

**What happens:** Even with retrieved context, the generator produces statements that are not supported by the retrieved passages.

**Why it happens:** The generator may blend retrieved facts with parametric knowledge, especially when the retrieved passages are incomplete or ambiguous. The model may also interpolate between facts in the context and produce plausible-sounding but unsupported claims.

**Key point:** RAG reduces hallucination. It does not eliminate it. Systems that claim "RAG completely eliminates hallucinations" are making a false claim.

## Summary of Failure Modes

| # | Failure Point | Stage | Result |
|---|---|---|---|
| 1 | Retrieval miss | Retrieval | Answer from memory or "I don't know" |
| 2 | Wrong passage retrieved | Retrieval | Confident wrong answer |
| 3 | Chunking destroys context | Indexing | Incomplete retrieved evidence |
| 4 | Stale document store | Maintenance | Outdated answers |
| 5 | Generator ignores context | Generation | Parametric override of retrieved facts |
| 6 | Context window overflow | Generation | Truncated or failed call |
| 7 | Hallucination persists | Generation | Unsupported claims despite retrieval |
