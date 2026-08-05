# Project 26 — Late-Interaction & LLM Rerankers

> **Goal:** Add the two remaining reranking families to your toolbox — ColBERT's
> token-level "late interaction" matching, and pointwise reranking by a MonoT5
> or a plain LLM — and learn how their failure modes differ from a cross-encoder.

## Why

A cross-encoder (Project 25) reads the full query-document pair and is accurate,
but it scores every candidate with the whole model — expensive at 50 candidates.
ColBERT takes a different bet: embed the query and the document as *per-token*
vectors, then score by summing, for each query token, its best-matching document
token. That "MaxSim" score captures fine-grained term overlap a pooled vector
smears away, and it is faster than a cross-encoder at scale because document
token embeddings are precomputed. MonoT5 approaches reranking from a third
angle: it is a text-to-text model fine-tuned to answer "true or false: does
this passage answer this query?" — and because it is a language model, you can
also replace it with any local LLM that can be prompted to say yes/no. Three
rerankers, three mechanisms; this project makes the difference concrete.

## Learn

- Late interaction: token-level embeddings + MaxSim summation, and why it beats a pooled vector on fine-grained match
- ColBERT mechanics: document token embeddings precomputed at index time; only the query tokens are embedded per search
- Pointwise vs listwise reranking: MonoT5 and LLM yes/no prompts score one pair at a time
- How a seq2seq model (`castorini/monot5-base-msmarco`) reranks by decoding "true"/"false"
- The pragmatic pattern: run a fast index → ColBERT or cross-encoder to a shortlist → LLM rerank only the top 5-10

## Execute

1. **Setup** — `pip install sentence-transformers transformers torch` (Ollama optional for the LLM reranker)
2. **Read** — `src/retrieval/rerank_advanced.py` — `ColBERTReranker`, `MonoT5Reranker`, `LLMPointwiseReranker` stubs
3. **Implement** — token-embedding MaxSim scoring for ColBERT; MonoT5 "true/false" logit mapping; a yes/no prompt for the LLM reranker
4. **Run** — `python retrieval/rerank_advanced.py` for the smoke test; rerank the same 20-candidate pool with all three
5. **Measure** — same questions as Project 25; compare ColBERT vs cross-encoder vs LLM rerank on order agreement and wall-clock time
6. **Acceptance criteria** — ColBERT MaxSim is implemented from token embeddings (not a pooled-vector shortcut); all three rerankers return the same list order on at least 2 of 3 test questions; you can state which is fastest and which is most accurate for your corpus

## Stretch

- Run an LLM *listwise* reranker (RankLLM-style: "sort these passages") and compare against pointwise
- Precompute ColBERT document token embeddings to a FAISS index and measure the speedup
- Cross-check your ColBERT implementation against the reference `colbert-ir` package on one query

## Article

- [ ] `26-late-interaction-llm-rerankers.md`

## Code

- `src/retrieval/rerank_advanced.py` — `ColBERTReranker(model_name)` — MaxSim late interaction; `MonoT5Reranker(model_name)` — seq2seq true/false; `LLMPointwiseReranker(llm)` — yes/no prompt

## Notebook

`NoteBooks/Project-26-Late-Interaction-LLM-Rerankers/01-late-interaction-llm-rerank-spec.py` → generate with:

```bash
python src/scripts/gen_notebook.py NoteBooks/Project-26-Late-Interaction-LLM-Rerankers/01-late-interaction-llm-rerank-spec.py NoteBooks/Project-26-Late-Interaction-LLM-Rerankers/01-late-interaction-llm-rerank.ipynb
```
