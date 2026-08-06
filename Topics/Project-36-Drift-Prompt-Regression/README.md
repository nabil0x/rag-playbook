# Project 36 — Drift & Prompt Regression

> **Goal:** The capstone that keeps a deployed RAG system honest — detect when
> the data or the questions drift away from what the system was built for, and
> gate every prompt change behind an automated regression run against your
> golden set.

## Why

A deployed RAG system degrades silently. The corpus grows ("the docs changed
and users now ask about a new topic"), the questions drift ("last quarter's
queries are not this quarter's queries"), and — the quiet killer — someone edits
a prompt template and *nothing breaks loudly, everything gets worse slowly*.
Two tools stop all three. **Drift detection** watches the live query stream:
embed incoming questions, compare the recent centroid against the baseline,
and flag when the distribution moves — plus a retrieval hit-rate check against
the golden set to catch index problems. **Prompt regression** turns every
prompt edit into a CI event: run the golden QA set through the new template,
score it with the Project 20 judge, and fail the change if faithfulness or
relevance drops below baseline. This is how a production RAG system stays
correct without a human reading every answer.

## Learn

- Data drift vs retrieval drift: the corpus changed vs the index is failing — different causes, different fixes
- Query-embedding drift: centroid + mean cosine distance from baseline as a simple, effective signal
- Retrieval hit-rate monitoring: recall@k against the golden set over time windows
- Golden-set regression as a CI gate: run, score, compare to baseline, fail on regression
- The judge loop from Project 20 feeding the regression gate (`src/evaluation/judge.py`)
- Thresholds and alerting: flag, don't page — drift is a signal to investigate

## Execute

1. **Setup** — `pip install fastembed`; reuse the Project 20 judge stack (`src/evaluation/golden.py`, `src/evaluation/judge.py`)
2. **Read** — `src/evaluation/drift.py` (`DriftDetector`) and `src/evaluation/regression.py` (`PromptRegressionSuite`) stubs
3. **Implement** — query centroid tracking + threshold flag; retrieval hit-rate over windows; the regression suite (run → score → compare → pass/fail)
4. **Run** — `python evaluation/drift.py` and `python evaluation/regression.py` smoke tests; then simulate: shift the question topic, degrade a prompt
5. **Measure** — drift flagged on the shifted topic but not the original; regression suite passes the good prompt and fails the degraded one with a delta
6. **Acceptance criteria** — the drift detector fires on a simulated topic shift and stays quiet on baseline queries; the regression suite returns `{"passed": bool, "delta": ...}` and a non-zero exit code on regression; the hit-rate check catches a deliberately broken retriever

## Stretch

- Wire the regression suite into a CI workflow so every prompt PR runs it
- Feed drift alerts into the Project 34 metrics (a `drift` gauge Prometheus can scrape)
- Add an automated re-index trigger when corpus change is detected

## Article

- [ ] `36-drift-prompt-regression.md`

## Code

- `src/evaluation/drift.py` — `DriftDetector` — query-centroid drift + retrieval hit-rate monitoring
- `src/evaluation/regression.py` — `PromptRegressionSuite` — golden-set prompt regression gate

## Notebook

`NoteBooks/Projects/Project-36-Drift-Prompt-Regression/01-drift-regression-spec.py` → generate with:

```bash
python src/scripts/gen_notebook.py NoteBooks/Projects/Project-36-Drift-Prompt-Regression/01-drift-regression-spec.py NoteBooks/Projects/Project-36-Drift-Prompt-Regression/01-drift-regression.ipynb
```
