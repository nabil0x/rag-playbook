# Project 25 — CrossEncoder Reranking

> **Goal:** Retrieve a wide candidate list cheaply with a bi-encoder, then
> re-score the top 20 with a cross-encoder that actually reads each
> query-document pair — and watch top-5 retrieval failures disappear.

## Why

A bi-encoder embeds query and document *separately*, then scores with a cosine
— fast (you pre-compute document vectors once), but the model never sees the
query and document together, so subtle relevance ("which part of this page
answers the question?") is lost. A cross-encoder feeds the *pair* into the
model — far more accurate, but it costs a full forward pass per candidate, so
you cannot run it over your whole corpus. The standard answer is a pipeline:
retrieve 20–50 candidates with a cheap bi-encoder, rerank the top handful with
a cross-encoder. This is the single highest-leverage accuracy win available to
a RAG system that already has decent retrieval, and it takes about twenty
lines around the retriever you already built.

## Learn

- Bi-encoder vs cross-encoder: what each one sees, and the speed/accuracy tradeoff
- Why "retrieve 5" is a bet — retrieve 20-50 and let the reranker decide
- The `cross-encoder/ms-marco-MiniLM-L-6-v2` model: trained on MS MARCO relevance pairs, tiny, local
- Storing the rerank score on `Document.metadata["score"]` so downstream code (and Project 27) can use it
- When reranking does *not* help: the candidates must actually contain the answer first

## Execute

1. **Setup** — `pip install sentence-transformers`
2. **Read** — `src/retrieval/rerank.py` — the `CrossEncoderReranker` and `RerankRetriever` stubs
3. **Implement** — `rerank(query, documents, top_k)`: load the cross-encoder lazily, score pairs, sort descending, keep `top_k`, attach scores
4. **Run** — `python retrieval/rerank.py` for the smoke test; then wire `RerankRetriever` (retrieve 20 → rerank to 5) into a pipeline
5. **Measure** — ask 3 questions through plain top-5 retrieval and through retrieve-20-rerank-5; compare answer quality and the score spread
6. **Acceptance criteria** — reranked list is monotonic in cross-encoder score; the reranker moves a relevant-but-low-ranked chunk into the top 5 on at least one question; you can state the added latency per query

## Stretch

- Compare `MiniLM-L-6-v2` against a larger reranker (e.g. `ms-marco-MiniLM-L-12-v2`)
- Add a minimum-score cutoff and answer "I don't know" when nothing clears it
- Measure how many candidates to retrieve: 10 vs 20 vs 50, same rerank budget

## Article

- [ ] `25-crossencoder-reranking.md`

## Code

- `src/retrieval/rerank.py` — `CrossEncoderReranker(model_name, top_k)` — pair-scoring rerank; `RerankRetriever(base, k_retrieve, top_k)` — retrieve-wide-rerank-short

## Notebook

`NoteBooks/Project-25-CrossEncoder-Reranking/01-crossencoder-reranking-spec.py` → generate with:

```bash
python src/scripts/gen_notebook.py NoteBooks/Project-25-CrossEncoder-Reranking/01-crossencoder-reranking-spec.py NoteBooks/Project-25-CrossEncoder-Reranking/01-crossencoder-reranking.ipynb
```
