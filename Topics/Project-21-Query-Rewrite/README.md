# Project 21 — Query Rewrite & Step-Back

> **Goal:** Improve retrieval by rewriting the question before it is embedded —
> LLM rewrite fixes vague/conversational phrasing, step-back abstracts narrow
> questions into broader ones that surface more relevant chunks.

## Why

Vector search is only as good as the query you embed. A raw user query
("what about the second one?") embeds into the wrong region of vector space.
Rewriting it into a standalone, well-formed question — or retrieving with a
step-back question that finds the general context first — is the cheapest
retrieval-quality win before you add re-ranking (Projects 25–26). This project
builds two swappable retriever wrappers and measures the difference on a small
hand-labeled eval set.

## Learn

- Query rewrite: conversational → standalone, vague → specific (one LLM call)
- Step-back prompting: abstract the question, retrieve broadly, answer precisely
- When rewriting helps (vague/pronoun queries) vs. when it is a no-op (the
  answer chunk was already in top-k)
- Measuring retrieval quality: hit-rate (did the gold chunk appear in top-k?)
  on a hand-labeled set

## Execute

1. **Setup** — no new dependencies. Reuses your Project 17 `RAGPipeline` blocks
   and an LLM from `llms/` (OpenAI / Gemini / local Ollama).
2. **Read** `retrieval/query_rewrite.py` and `retrieval/step_back.py` — both are
   stubs with `TODO`s and a no-network `__main__` smoke test.
3. **Implement** both wrappers:
   - `QueryRewriteRetriever` — LLM rewrites the question, embed the rewrite,
     delegate retrieval to the wrapped retriever.
   - `StepBackRetriever` — LLM produces a step-back question, retrieve with it,
     merge with the original-query results, dedupe, return top-k.
4. **Build the eval set** — take 15–20 questions: reuse `evaluation/golden.py`
   where possible, plus a few deliberately vague/pronoun-heavy questions about
   `Data/Waiting.txt` (the kind a chatbot would really receive).
5. **Run the comparison** — same vector store, same top-k: baseline vs rewrite
   vs step-back. Record hit-rate per strategy and, for 3–5 questions, judge
   answer quality with your P20 `LLMJudge`.
6. **Acceptance criteria:**
   - Both wrappers return `list[Document]` and slot into
     `RAGPipeline(retriever=...)` unchanged.
   - You can name 2+ query types where rewrite changes the top-k — and 1 where
     it does not.
   - A hit-rate table exists: baseline / rewrite / step-back over your set.

## Stretch

- Rewrite + step-back combined (rewrite first, then step-back)
- Auto-decision: only rewrite when the query looks vague (cheap heuristic:
  pronoun present, length < 6 tokens) — the first step toward prompt routing
  (Project 27)
- Log how often rewriting changed the top-5 (feeds Project 36 retrieval monitoring)

## Article

- [ ] `21-query-rewrite-stepback.md`

## Code

- `retrieval/query_rewrite.py` — `QueryRewriteRetriever(rewriter_llm, retriever, top_k)`
  — rewrites the question with an LLM, then delegates retrieval to the wrapped
  retriever.
- `retrieval/step_back.py` — `StepBackRetriever(stepback_llm, retriever, top_k)`
  — generates a step-back question, retrieves with it, and merges with the
  original-query results.

## Notebook

`NoteBooks/Project-21-Query-Rewrite/01-query-rewrite-stepback-spec.py` →
generate the notebook with:

```bash
python scripts/gen_notebook.py \
  NoteBooks/Project-21-Query-Rewrite/01-query-rewrite-stepback-spec.py \
  NoteBooks/Project-21-Query-Rewrite/01-query-rewrite-stepback.ipynb
```
