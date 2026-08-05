# Project 22 — HyDE & Query Decomposition

> **Goal:** Improve retrieval for complex and indirect questions with two
> "generate-then-retrieve" strategies — HyDE embeds a hypothetical answer
> instead of the query, decomposition splits a multi-hop question into
> sub-questions and retrieves for each.

## Why

Embedding the question is a gamble when the question shares no vocabulary with
the answer chunk: "who wrote the sequel?" never mentions the author's name, so
its vector lands far from the right document. HyDE fixes this by having the LLM
write a short hypothetical passage that would answer the question — that
passage reads like a source document, and embedding it lands near the real
chunks. Decomposition fixes a different failure: multi-hop questions ("When was
the author of X born?") need facts from 2+ chunks, but plain top-k returns the
single most-similar chunk and the second fact's chunk never surfaces.
Retrieving once per sub-question pulls every fact's chunk into the candidate
set. Both are cheap wrappers around the retriever you already have — same
store, same embeddings, one extra LLM call (or a few) — and both sit in the
same spot in the pipeline as re-ranking (Projects 25–26) but attack the problem
at the query, not at the result list.

## Learn

- HyDE mechanics: one LLM call writes a hypothetical passage in source-document
  style, you embed that text instead of the query
- The lexical-gap failure HyDE fixes: question wording vs. document wording
- Decomposition patterns: parallel (all sub-questions in one call — fast, good
  for independent facts) vs. sequential (each next sub-question planned from
  the previous answer — slower, needed for dependent fact chains)
- When each wins: HyDE for indirect/paraphrased questions, decomposition for
  multi-hop questions — and how to tell the two failure modes apart
- Measuring retrieval quality: hit-rate (did every gold chunk appear in top-k?)
  on a multi-hop eval set

## Execute

1. **Setup** — no new dependencies. Reuses your Project 17 `RAGPipeline` blocks
   and an LLM from `src/llms/` (OpenAI / Gemini / local Ollama) — the LLM writes
   the hypothetical documents and the sub-questions.
2. **Read** `src/retrieval/hyde.py` and `src/retrieval/decompose.py` — both are stubs
   with `TODO`s and a no-network `__main__` smoke test.
3. **Implement** both wrappers:
   - `HyDERetriever` — LLM writes a hypothetical passage, embed the passage,
     delegate retrieval to the wrapped retriever.
   - `DecomposeRetriever` — LLM splits the question into 2–4 sub-questions,
     retrieve with the original question AND each sub-question, dedupe by page
     content (first occurrence wins), return top-k.
4. **Build the eval set** — 10–15 multi-hop questions: reuse 3–5 from
   `src/evaluation/golden.py` where possible, synthesize the rest from
   `Data/Waiting.txt`. Each question must need 2 chunks stitched together —
   "Who wrote it, and in what year?" — so plain top-k misses part of it.
5. **Run the comparison** — same vector store, same top-k: baseline vs HyDE
   vs decomposition. Record hit-rate per strategy and, for 3–5 questions, judge
   answer quality with your P20 `LLMJudge`.
6. **Acceptance criteria:**
   - Both wrappers return `list[Document]` and slot into
     `RAGPipeline(retriever=...)` unchanged.
   - A hit-rate table exists: baseline / HyDE / decomposition over your set.
   - You can explain one case where HyDE won and one where decomposition won.

## Stretch

- Hybrid HyDE + decomposition: decompose first, then run HyDE on each
  sub-question (and on the original)
- Sequential decomposition: one sub-question at a time, each planned from the
  previous answer (handles dependent fact chains)
- Log which strategy changed the top-5 and when (feeds Project 36 retrieval
  monitoring)

## Article

- [ ] `22-hyde-decomposition.md`

## Code

- `src/retrieval/hyde.py` — `HyDERetriever(hyde_llm, retriever, top_k)` — writes a
  hypothetical answer passage with an LLM, embeds it, then delegates retrieval
  to the wrapped retriever.
- `src/retrieval/decompose.py` — `DecomposeRetriever(decomposer_llm, retriever, top_k)`
  — splits a multi-hop question into sub-questions, retrieves with the original
  plus each sub-question, and merges the deduplicated results.

## Notebook

`NoteBooks/Project-22-HyDE-Decomposition/01-hyde-decomposition-spec.py` →
generate the notebook with:

```bash
python src/scripts/gen_notebook.py \
  NoteBooks/Project-22-HyDE-Decomposition/01-hyde-decomposition-spec.py \
  NoteBooks/Project-22-HyDE-Decomposition/01-hyde-decomposition.ipynb
```
