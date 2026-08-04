# Project 27 — Context Assembly & Lost-in-the-Middle

> **Goal:** Take the chunks your retriever returns — duplicates, near-duplicates,
> wrong order, over budget — and assemble the context your LLM actually reads,
> then measure the "lost in the middle" bias that silently tanks accuracy.

## Why

Retrieval hands you a list; the LLM reads a *context*. Between those two sit
decisions nobody teaches: the same chunk retrieved twice (multi-query and
hybrid retrievers are notorious for this), near-identical chunks from
overlapping splits, ordering that buries the answer, and a token budget you are
blowing past. Worse, LLMs are demonstrably bad at using the middle of a long
context — the "lost in the middle" effect from Liu et al. 2023: move the
answer-bearing chunk from the start to the middle of the context and accuracy
drops. That means *where* you put the evidence is a retrieval decision too.
This project builds the assembler that dedups, orders, and truncates your
context — and an experiment that proves ordering matters on your own data.

## Learn

- Exact vs near-duplicate removal: exact string match plus embedding-cosine dedup for overlapping splits
- Ordering strategies: best-first (score-descending), retrieval order, or answer-first — and when each wins
- Token budgeting: greedy fill up to the LLM's context limit, preserving order, and what you lose by truncating
- The lost-in-the-middle finding (Liu et al., 2023): answer position in context changes accuracy
- Why context assembly is the last mile of retrieval quality — a great retriever feeding a sloppy context still fails

## Execute

1. **Setup** — `pip install fastembed` (or reuse the repo's embedder), plus your usual LLM
2. **Read** — `retrieval/context_assembly.py` — the `ContextAssembler` and `LostInTheMiddleExperiment` stubs
3. **Implement** — `dedupe` (exact + cosine > 0.9), `reorder` (best_first / retrieval_order / answer_first), `truncate` (greedy token budget)
4. **Run** — `python retrieval/context_assembly.py` for the smoke test; then assemble a real context from a multi-query retrieval
5. **Measure** — the lost-in-the-middle experiment: answer chunk at start / middle / end of 8 chunks, same LLM, 3 questions; record accuracy per position
6. **Acceptance criteria** — dedupe removes duplicated and near-identical chunks from a hybrid retrieval; truncate respects the budget and keeps the strongest chunks; the experiment reproduces the lost-in-the-middle effect (start >= end > middle) on at least 2 of 3 questions

## Stretch

- Compare "best-first" vs "answer-first" ordering with the experiment harness
- Add recency metadata ordering for chat-style corpora (newer chunks first)
- Study where the lost-in-the-middle effect *starts*: sweep context length 4 → 8 → 16 chunks

## Article

- [ ] `27-context-assembly.md`

## Code

- `retrieval/context_assembly.py` — `ContextAssembler(embedder)` — dedupe/reorder/truncate; `LostInTheMiddleExperiment(llm)` — position-bias measurement

## Notebook

`NoteBooks/Project-27-Context-Assembly/01-context-assembly-spec.py` → generate with:

```bash
python scripts/gen_notebook.py NoteBooks/Project-27-Context-Assembly/01-context-assembly-spec.py NoteBooks/Project-27-Context-Assembly/01-context-assembly.ipynb
```
