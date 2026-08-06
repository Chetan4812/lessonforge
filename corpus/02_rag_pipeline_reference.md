---
title: "The RAG Pipeline: A Step-by-Step Reference"
slug: rag_pipeline_reference
chunk_prefix: S2
sources:
  - url: "https://python.langchain.com/docs/concepts/rag/"
    citation: "LangChain documentation: RAG concepts, 2024."
  - url: "https://www.pinecone.io/learn/retrieval-augmented-generation/"
    citation: "Pinecone: Retrieval Augmented Generation (RAG) explained, 2024."
  - url: "https://arxiv.org/abs/2005.11401"
    citation: "Lewis et al., NeurIPS 2020."
---

## Overview

The RAG pipeline has two distinct phases: an **offline indexing phase** (run once or periodically to build the searchable knowledge base) and an **online inference phase** (run for every user query).

## Phase 1 — Offline Indexing

### Step 1: Document Ingestion

Collect the source documents to be made searchable. These can be PDFs, Word documents, web pages, database records, or any text source. Documents are loaded and converted to plain text.

### Step 2: Chunking (Splitting)

Each document is split into smaller pieces called **chunks**. A chunk is a short section of text, typically 100–500 words long. Chunking is necessary because:

- Embedding models have a maximum input length (e.g. 512 tokens for many popular models).
- Retrieving one small, focused chunk is more useful than retrieving an entire 50-page document.
- Smaller chunks allow the retriever to identify more precisely which part of a document is relevant.

Common chunking strategies:
- **Fixed-size chunking:** split every N tokens regardless of sentence or paragraph boundaries. Simple but may split mid-sentence.
- **Sentence or paragraph chunking:** respect natural language boundaries. More coherent but variable size.
- **Heading-aware chunking:** treat each section under a markdown or HTML heading as a chunk. Preserves document structure.

**Overlap:** Most implementations add an overlap of 10–20% of the chunk size. For example, a 250-token chunk may repeat the last 40 tokens at the start of the next chunk. This prevents key information from being lost when a relevant phrase sits at the boundary between chunks.

### Step 3: Embedding

Each chunk is converted into a **dense vector** — a list of floating-point numbers — using an embedding model. The embedding model maps text to a high-dimensional vector space such that semantically similar texts have vectors that are close to each other (in terms of cosine similarity or dot product).

Popular open-source embedding models include:
- `sentence-transformers/all-MiniLM-L6-v2` (fast, 384 dimensions, no API key required)
- `sentence-transformers/all-mpnet-base-v2` (higher quality, 768 dimensions)
- `text-embedding-3-small` (OpenAI API, 1536 dimensions)

### Step 4: Indexing

The computed vectors are stored in a **vector index** — a data structure optimised for fast similarity search. At small scale (under a million chunks), a simple flat index (exact search) works well. At large scale, approximate nearest-neighbour (ANN) indexes such as HNSW or IVF-PQ provide faster search at the cost of a small accuracy trade-off.

Popular vector indexes and databases:
- **FAISS** (Facebook AI Similarity Search) — open-source, in-memory, no external service needed.
- **Chroma, Weaviate, Qdrant, Pinecone** — managed or self-hosted vector databases with persistence.

## Phase 2 — Online Inference (per query)

### Step 5: Query Embedding

When the user asks a question, the question is embedded using the **same embedding model** used in the indexing phase. This is critical — using a different model would produce incomparable vectors and retrieval would fail.

### Step 6: Retrieval

The query vector is compared against all stored chunk vectors. The top-k most similar chunks (by cosine similarity or dot product) are returned. A typical value is k = 3 to 10.

### Step 7: Augmentation (Prompt Construction)

The retrieved chunks are inserted into the prompt that will be sent to the language model. A typical augmented prompt looks like:

```
Use the following context to answer the question.
Context:
[Chunk 1 text]
[Chunk 2 text]
[Chunk 3 text]
Question: [user's question]
Answer:
```

### Step 8: Generation

The language model reads the augmented prompt (which includes both the retrieved evidence and the user's question) and generates a response grounded in the retrieved text.

## The Grounding Contract

For RAG to work correctly, the generator must be instructed to answer **only from the provided context** and to indicate when the context does not contain enough information to answer. Without this instruction, models often blend retrieved facts with parametric knowledge, which defeats the purpose of grounding.

A well-designed RAG system also strips internal citation markers from the learner-facing output and replaces them with a clean Sources section, so readers can verify claims without seeing the internal tagging mechanism.
