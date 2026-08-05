# Project 28 — Citations & Attribution

> **Goal:** Make every answer verifiable — split the LLM's answer into claims,
> check each claim against the retrieved chunks, and emit inline citations
> `[1] [2]` that a reader can trace back to a source.

## Why

A RAG answer that sounds confident is not enough — you need to know *which part
of it is grounded in the retrieved documents and which part the model invented*.
Without attribution, one hallucinated number poisons an otherwise correct
answer and nobody can find it. The fix is a claim-level check: break the answer
into atomic claims, verify each against the context chunks (exact substring
match first, embedding similarity as the fuzzy fallback), and report the
groundedness score — the fraction of claims supported. Then surface it: inline
`[n]` markers in the answer plus a source list, so the failure mode becomes
visible instead of silent. Project 20 gave you answer-level metrics; this
project gets down to the sentence that is wrong.

## Learn

- Claim extraction: splitting an answer into atomic, checkable statements
- Attribution checking: exact-match first, embedding-similarity fallback, and why thresholds need tuning
- Groundedness = supported claims / total claims — and its complement, the hallucination rate
- Inline citation formatting `[1] [2]` tied to a sources list (the `CitationPrompt` in `src/prompts/citation.py` asks for these markers)
- The attribution boundary: a claim can be "supported" yet incomplete — coverage is a separate axis

## Execute

1. **Setup** — `pip install fastembed` (or reuse the repo embedder); optional LLM for claim extraction
2. **Read** — `src/evaluation/attribution.py` — `Citation`, `AttributionEvaluator`, `CitationFormatter` stubs
3. **Implement** — sentence-level claim splitting, the exact-then-fuzzy support check, and the `[n]` formatting
4. **Run** — `python evaluation/attribution.py` for the smoke test; then evaluate 3 answers: fully grounded, partially grounded, one hallucinated claim
5. **Measure** — groundedness % per answer, the list of unsupported claims, and the formatted cited answer
6. **Acceptance criteria** — the hallucinated claim is flagged unsupported on the partially-grounded answer; groundedness is 100% on the fully-grounded answer; formatted output maps every `[n]` to a real chunk

## Stretch

- Use an LLM to extract claims instead of naive sentence splitting — compare coverage
- Add a coverage score: did the chunks *contain* everything the answer needed?
- Wire the formatter into `src/main.py`'s prompt so every pipeline answer ships with citations

## Article

- [ ] `28-citations-attribution.md`

## Code

- `src/evaluation/attribution.py` — `Citation` dataclass; `AttributionEvaluator(embedder, threshold)` — claim-level support check; `CitationFormatter` — inline `[n]` + sources

## Notebook

`NoteBooks/Project-28-Citations-Attribution/01-citations-attribution-spec.py` → generate with:

```bash
python src/scripts/gen_notebook.py NoteBooks/Project-28-Citations-Attribution/01-citations-attribution-spec.py NoteBooks/Project-28-Citations-Attribution/01-citations-attribution.ipynb
```
