# Project 18 — RAG Benchmark Suite (Capstone)

> **Goal:** Combine all projects into a benchmarking framework — not a single chatbot.

```
             RAG Benchmark Framework

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
 Gemini / BGE / E5 / Nomic / Jina / Voyage
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
 Gemini / GPT / Claude / Qwen / Mistral
                   │
               Evaluation
 Accuracy • Recall@K • MRR • nDCG • Latency • Cost
```

## Why

Proves you can **design, compare, and evaluate** RAG systems — not just build one.
Also prepares for interviews: *why* this embedding, *why* this retriever.

## Article

- [ ] `01-rag-benchmark-suite.md`

## Code

New `eval/` folder (metrics, harness), reusing every component module.

## Notebook

`NoteBooks/Project-18-RAG-Benchmark-Suite/` — capstone notebook not created yet
