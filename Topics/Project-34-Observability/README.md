# Project 34 — Observability

> **Goal:** Make the service explainable — OpenTelemetry traces for every
> question's journey through the pipeline, Prometheus metrics for what is
> happening at a glance, and structured JSON logs for the long tail.

## Why

Project 32 and 33 gave you a service that runs. This project gives you the
ability to *see inside it*. When a user reports "the answer was wrong", you
need three different views: a **trace** — the exact path one question took
(ingest → retrieve → generate) with per-span latency and the number of chunks
used; **metrics** — rolling counters and histograms (queries per second,
latency distribution, retrieval hit rate) that tell you the system is healthy
without reading logs; and **structured logs** — JSON lines with fields a log
aggregator can query, instead of prose. These are the three pillars of
observability, and each answers a different question: traces answer "what
happened for this request?", metrics answer "what is happening across all
requests?", logs answer "what did the code actually say?".

## Learn

- Traces vs metrics vs logs — and the question each one answers
- OpenTelemetry spans: `ingest`, `retrieve`, `generate`, with attributes (chunk count, model, latency)
- Span context propagation: a query's spans share one trace ID end-to-end
- Prometheus metric types: Counter, Histogram (latency buckets), and labeled counters (retrieval hit/miss)
- Structured logging: JSON formatter with fields instead of free-text messages
- The local-friendly setup: console/OTLP span exporter and a plain Prometheus text endpoint — no SaaS required

## Execute

1. **Setup** — `pip install opentelemetry-sdk opentelemetry-api prometheus-client`
2. **Read** — `src/observability/tracing.py` (tracer + span helpers) and `src/observability/metrics.py` (metric definitions + JSON logging)
3. **Implement** — the `ingest`/`retrieve`/`generate` spans with attributes; the counter/histogram updates at the right pipeline points
4. **Run** — `python observability/tracing.py` and `python observability/metrics.py` smoke tests; then run 5 queries through a pipeline and export
5. **Measure** — one trace per query (trace ID + span latencies), the metrics text (counts + latency histogram), and a JSON log line per query
6. **Acceptance criteria** — every query produces one trace with 3 spans sharing an ID; metrics text exports Prometheus format with the defined counters/histogram; logs are valid JSON with query and latency fields

## Stretch

- Run a Prometheus server scraping the metrics endpoint and a Grafana dashboard
- Export traces to a local OTLP collector and inspect them in a trace viewer
- Add a "retrieval hit rate" gauge and alert when it drops (Project 36's drift detector can feed it)

## Article

- [ ] `34-observability.md`

## Code

- `src/observability/tracing.py` — `init_tracing(service_name)` + `trace_pipeline` spans (ingest/retrieve/generate)
- `src/observability/metrics.py` — Prometheus counters/histogram + `setup_logging()` JSON formatter

## Notebook

`NoteBooks/Projects/Project-34-Observability/01-observability-spec.py` → generate with:

```bash
python src/scripts/gen_notebook.py NoteBooks/Projects/Project-34-Observability/01-observability-spec.py NoteBooks/Projects/Project-34-Observability/01-observability.ipynb
```
