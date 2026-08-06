# Project 35 — Online Eval & A/B Testing

> **Goal:** Stop evaluating only in notebooks. Capture real user feedback on
> live answers, run an A/B test between two retrievers, and learn why "the new
> one feels better" is not a measurement.

## Why

Project 18 and 20 evaluated on a golden set — essential, but it is *offline*
evaluation: a snapshot of questions you wrote once. Production questions are
different, and your retrieval changes over time, so you need an *online* loop:
capture how real users rate real answers (thumbs up/down, a 1-5 star), and use
that signal to decide whether a change actually helped. The rigorous version is
an A/B test: serve variant A (e.g. `SimilarityRetriever`) to half the traffic
and variant B (e.g. `RerankRetriever` from Project 25) to the other half, then
compare mean ratings. The math lesson is the whole point: with small samples,
noise swamps the difference — you will watch a "real" improvement fail the
significance check and learn why `n` is the most important metric.

## Learn

- Online vs offline evaluation: golden sets vs live feedback, and why you need both
- Feedback capture: logging rating events (question, answer, rating, variant, latency) to SQLite/CSV
- A/B test design: variant assignment (random or round-robin), blinded comparison, one change at a time
- Comparing variants: mean rating, win rate, and the standard error / t-test for significance
- The sample-size lesson: small `n` → the confidence interval swallows the difference

## Execute

1. **Setup** — stdlib only (`sqlite3`/`csv`); optional `pip install scipy` for the t-test
2. **Read** — `src/evaluation/online.py` — `FeedbackEvent`, `OnlineEvaluator`, `ABTest` stubs
3. **Implement** — feedback logging to SQLite, `summary()` stats, `assign()` variant choice, and `compare()` with the significance check
4. **Run** — `python evaluation/online.py` for the smoke test; then simulate 100 feedback events with a real rating difference between variants
5. **Measure** — mean rating per variant, the win rate, and the significance verdict at n=20 vs n=100
6. **Acceptance criteria** — events persist across runs (SQLite file); at n=100 the simulated difference is detected; at n=20 the same difference is *not* significant — and you can explain why

## Stretch

- Log free-text user comments and surface the worst-rated answers for manual review
- Add a bandit-style allocation that shifts traffic toward the winning variant
- Run the A/B between real retrievers (Project 25's reranker vs plain top-k) for a day and compare

## Article

- [ ] `35-online-eval.md`

## Code

- `src/evaluation/online.py` — `OnlineEvaluator` (SQLite feedback log + summary), `ABTest` (assignment + significance)

## Notebook

`NoteBooks/Projects/Project-35-Online-Eval/01-online-eval-spec.py` → generate with:

```bash
python src/scripts/gen_notebook.py NoteBooks/Projects/Project-35-Online-Eval/01-online-eval-spec.py NoteBooks/Projects/Project-35-Online-Eval/01-online-eval.ipynb
```
