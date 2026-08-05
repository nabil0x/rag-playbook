# Roadmap — Projects 21–36

> **Goal:** Move the curriculum past "swap one component" into the four skills that
> separate toy RAG from useful RAG: **query transformation**, **index & reranking
> engineering**, **agentic retrieval**, and **production + evaluation at scale**.
>
> Sixteen small projects, one concept each, built on top of the component library
> you already have. No project in this roadmap changes more than one idea at a
> time, so you can always isolate what worked.

---

## How to read this roadmap

- **One project = one concept.** The project card (`Topics/Project-NN-*/README.md`)
  is the execution guide: why, what to build, acceptance criteria.
- **Module stubs live in the component folders.** You fill in the `NotImplementedError`
  sites; the stub already carries the docstring, the import contract, and a
  no-network `__main__` smoke test.
- **Notebook specs** (`NoteBooks/Project-NN-*/01-*-spec.py`) render into runnable
  notebooks with `src/scripts/gen_notebook.py`.
- **Do them in order within a phase.** Phases build on each other; projects within
  a phase are independent enough that order matters only for narrative.
- **Skip freely.** The "Stretch" section of each card is optional. Items marked
  **deferred** in this roadmap are explicitly parked so the curriculum stays small.

---

## The seven phases

| Phase | Theme | Projects | What you can do afterwards |
|-------|-------|----------|----------------------------|
| 1 | Query Transformation | 21, 22 | Fix retrieval misses that come from *how the question is asked* |
| 2 | Index Optimization | 24 | Search millions of vectors with sub-linear latency and sane memory |
| 3 | Re-ranking | 25, 26 | Pull better chunks out of a top-50 candidate pool |
| 4 | Context Engineering | 27, 28 | Pack the right chunks in the right order, and make answers verifiable |
| 5 | Agentic RAG | 29, 30, 31 | Let the LLM decide when, what, and how to retrieve |
| 6 | Production | 32, 33, 34 | Serve it, ingest asynchronously, and see inside it |
| 7 | Evaluation at Scale | 35, 36 | Keep it correct while it runs in the wild |

---

## Phase map

| # | Project | Phase | Core concept | New module(s) |
|---|---------|-------|--------------|---------------|
| 21 | [Query Rewrite & Step-Back](Project-21-Query-Rewrite/README.md) | 1 | Rewrite vague/verbose queries; step back to a broader question | `src/retrieval/query_rewrite.py`, `src/retrieval/step_back.py` |
| 22 | [HyDE & Query Decomposition](Project-22-HyDE-Decomposition/README.md) | 1 | Embed a hypothetical answer; split multi-hop questions | `src/retrieval/hyde.py`, `src/retrieval/decompose.py` |
| 24 | [Vector Index Internals](Project-24-Vector-Index-Internals/README.md) | 2 | IVF, HNSW, PQ, OPQ — how ANN search actually works | `src/vectordb/faiss_index.py` |
| 25 | [CrossEncoder Reranking](Project-25-CrossEncoder-Reranking/README.md) | 3 | Bi-encoder to retrieve, cross-encoder to rerank | `src/retrieval/rerank.py` |
| 26 | [Late-Interaction & LLM Rerankers](Project-26-Late-Interaction-LLM-Rerankers/README.md) | 3 | ColBERT token-level matching; MonoT5 & LLM pointwise rerank | `src/retrieval/rerank_advanced.py` |
| 27 | [Context Assembly & Lost-in-the-Middle](Project-27-Context-Assembly/README.md) | 4 | Dedup, ordering, token budget; answer-position bias | `src/retrieval/context_assembly.py` |
| 28 | [Citations & Attribution](Project-28-Citations-Attribution/README.md) | 4 | Claim-level groundedness, hallucination detection | `src/evaluation/attribution.py` |
| 29 | [Tool-Calling RAG](Project-29-Tool-Calling-RAG/README.md) | 5 | LLM decides *when* to retrieve via tools | `src/agent/tool_calling.py` |
| 30 | [LangGraph Agentic RAG](Project-30-LangGraph-Agentic-RAG/README.md) | 5 | Stateful graph: plan → retrieve → answer → reflect | `src/agent/langgraph_agent.py` |
| 31 | [Multi-Agent RAG](Project-31-Multi-Agent-RAG/README.md) | 5 | Supervisor + specialist agents, parallel retrieval | `src/agent/multi_agent.py` |
| 32 | [RAG as a Service](Project-32-RAG-Service/README.md) | 6 | FastAPI, Redis cache, Docker | `src/api/main.py`, `docker/*` |
| 33 | [Async Ingestion](Project-33-Async-Ingestion/README.md) | 6 | Celery task queue for embedding pipelines | `src/celery_app.py` |
| 34 | [Observability](Project-34-Observability/README.md) | 6 | Traces, metrics, structured logs | `src/observability/*` |
| 35 | [Online Eval & A/B](Project-35-Online-Eval/README.md) | 7 | Feedback loops, A/B tests, significance | `src/evaluation/online.py` |
| 36 | [Drift & Prompt Regression](Project-36-Drift-Prompt-Regression/README.md) | 7 | Drift detection; CI regression gates | `src/evaluation/drift.py`, `src/evaluation/regression.py` |

> **Project 23 is intentionally skipped** (numbering keeps the original 20 as the
> foundation; the gap marks where the advanced curriculum begins).

---

## Stack decisions (apply to every project)

- **Local-first by default.** Ollama (`qwen2.5-coder:7b` or similar) for LLM calls,
  fastembed/BGE for embeddings. Zero API cost, works offline.
- **Gemini as the cloud alternative.** The repo's `GeminiLLM` / `GeminiEmbedding`
  components remain drop-in swaps; every project card notes where.
- **LangChain 1.3.14 note.** Agent/retriever tooling lives in the `langchain-classic`
  package (`langchain_classic.agents`, `langchain_classic.retrievers`). Notebooks
  import with a `try/except` fallback, same as Projects 09–15.
- **No new framework opinions.** If a project introduces a framework (FAISS,
  LangGraph, Celery, OpenTelemetry), it is the *point* of that project — nothing else
  in the roadmap depends on it except via the component boundary.

---

## Execution order

```
Phase 1 ── 21 → 22
Phase 2 ── 24            (needs only embeddings + faiss-cpu)
Phase 3 ── 25 → 26       (25 before 26: rerank-first thinking)
Phase 4 ── 27 → 28       (27 before 28: context quality precedes claims)
Phase 5 ── 29 → 30 → 31  (tool-calling → graph → multi-agent)
Phase 6 ── 32 → 33 → 34  (service → async → observability)
Phase 7 ── 35 → 36       (online → drift/regression; both capstones)
```

Recommended pauses: after Phase 1 (rewrite vs. rerank tradeoff), after Phase 4
(context engineering changes your prompts), and after Phase 6 (you now own a
deployable system worth evaluating).

---

## Deferred (explicitly parked)

- **DiskANN** (P24 stretch) — needs a specific OS/library footprint; IVF+HNSW+PQ
  already teach the full ANN toolkit.
- **Kubernetes** (P32 stretch) — a one-machine docker-compose deployment teaches
  the same service topology with 1/10th the moving parts.
- **Temporal** (P33 stretch) — Celery covers the queue pattern; durable workflows
  are a later add.
- **RankLLM / Cohere Rerank API** (P26 stretch) — hosted listwise reranking; the
  local ColBERT + MonoT5 + LLM variants keep the project free and offline.

---

## Artifacts per project

Every project card references the same four artifacts, in the same order:

1. **`README.md`** — this card (why / learn / execute / acceptance criteria).
2. **Stub module(s)** — importable class skeleton with `NotImplementedError` TODOs
   and a self-checking `__main__` block.
3. **Notebook spec** — `NoteBooks/Project-NN-*/01-*-spec.py`, rendered with:
   ```bash
   python src/scripts/gen_notebook.py NoteBooks/Project-NN-*/01-*-spec.py \
       NoteBooks/Project-NN-*/01-*.ipynb
   ```
4. **Article** — optional write-up drafted inside the project folder as the
   curriculum evolves (mirrors the existing Projects 01–20 pattern).
