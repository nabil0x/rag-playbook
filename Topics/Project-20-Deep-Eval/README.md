# Project 20 — Deep Eval (Capstone)

> **Goal:** Implement 4 RAGAS-style generation metrics (faithfulness, answer relevance, context precision, context recall) from scratch plus a RAGAS 0.4.3 cross-check, using a local Ollama judge (`qwen2.5-coder:7b`) and local fastembed embeddings (`BAAI/bge-base-en-v1.5`) — zero external API cost. Evaluates the SD-08 invoice RAG pipeline against a 23-question golden set.

```
             RAG Evaluation Framework

                 Loader
           ┌────────┴────────┐
           │                 │
         PDF              Web
           │                 │
           └────────┬────────┘
                    │
               Splitter
     Recursive / Token / Markdown / HTML
                    │
               Embeddings
    FastEmbed (BAAI/bge-base-en-v1.5)
                    │
               Vector Database
               Chroma / FAISS / Qdrant / LanceDB / PGVector
                    │
                 Retriever
    Similarity / MMR / Parent / MultiQuery / Compression
                    │
                 Prompt
    Basic / Citation / JSON / Few-shot
                    │
                   LLM
    Local Ollama (qwen2.5-coder:7b)
                    │
                 Evaluation
    Faithfulness • Answer Relevance • Context Precision • Context Recall
```

## Why

Proves you can **design, compare, and evaluate** RAG systems — not just build one.
Also prepares for interviews: *why* this embedding, *why* this retriever.

## Article

- [ ] `01-deep-eval.md`

## Code

New `src/evaluation/` folder (metrics, harness), reusing every component module.

- `golden.py` (`GOLDEN_QA` — 23 hand-checked QA pairs across 6 invoice docs; `load_golden()`)
- `judge.py` (`LocalEmbeddings` adapter over fastembed; `LLMJudge` over local Ollama `qwen2.5-coder:7b`)
- `metrics.py` (`FaithfulnessMetric`, `AnswerRelevanceMetric`, `ContextPrecisionMetric`, `ContextRecallMetric` — from-scratch RAGAS-style generation metrics)
- `ragas_metrics.py` (`ragas_scores()` — RAGAS 0.4.3 cross-check, fully local)
- `harness.py` (`MetricResult` plain class + `EvaluationHarness` with `run`/`aggregate`/`kappa`/`print_table`)

All components run locally — zero external API cost.

## Notebook

`NoteBooks/Project-20-Deep-Eval/01-deep-eval.ipynb` — capstone notebook, generated via `src/scripts/gen_notebook.py`, executed end-to-end on the local magus kernel (zero API cost)