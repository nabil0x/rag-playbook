"""Prometheus metrics and structured JSON logging.

Observability block: counters and a latency histogram for "what is happening
across all requests", and JSON log lines with query/latency fields for the
long tail. prometheus-client is optional: without it, no-op stand-ins keep
pipeline code callable. See Topics/Project-34-Observability/README.md.
"""

from __future__ import annotations

import json
import logging

try:
    from prometheus_client import Counter, Histogram

    HAVE_PROMETHEUS = True
except ImportError:  # prometheus-client is optional: importing never fails
    HAVE_PROMETHEUS = False


class _NoopMetric:
    """Counter/Histogram stand-in when prometheus-client is missing."""

    def labels(self, *args, **kwargs):
        return self

    def inc(self, amount: float = 1) -> None:
        pass

    def observe(self, amount: float) -> None:
        pass

    def set(self, value: float) -> None:
        pass


def _make_counter(name: str, documentation: str, labelnames: tuple[str, ...] = ()):
    if HAVE_PROMETHEUS:
        return Counter(name, documentation, labelnames=labelnames)
    return _NoopMetric()


def _make_histogram(name: str, documentation: str, buckets: tuple[float, ...]):
    if HAVE_PROMETHEUS:
        return Histogram(name, documentation, buckets=buckets)
    return _NoopMetric()


#: Total /query requests, labeled by cache outcome.
QUERIES_TOTAL = _make_counter("rag_queries_total", "Total /query requests", labelnames=("hit",))

#: End-to-end /query latency in seconds.
QUERY_LATENCY = _make_histogram(
    "rag_query_latency_seconds",
    "End-to-end /query latency",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


def record_query(latency_seconds: float, hit: bool) -> None:
    """TODO(Project 34): update the counters for one query.

    ``QUERIES_TOTAL.labels(hit="hit" if hit else "miss").inc()`` and
    ``QUERY_LATENCY.observe(latency_seconds)``. Call this from the API's
    /query path (and anywhere else a query completes) — the "right pipeline
    points" the card asks you to find.
    """
    raise NotImplementedError("TODO(Project 34): implement observability.metrics.record_query")


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON with query and latency fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("question", "latency_ms", "trace_id", "chunks"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload)


def setup_logging(level: int = logging.INFO) -> logging.Handler:
    """Attach a JSON handler to the root logger and return it."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(level)
    return handler


if __name__ == "__main__":
    # No-network smoke test — formatter, logging, and metric stand-ins.
    record = logging.LogRecord("rag", logging.INFO, __file__, 0, "answer ok", None, None)
    record.question = "What is RAG?"
    record.latency_ms = 42.0
    line = json.loads(JsonFormatter().format(record))
    assert line["message"] == "answer ok"
    assert line["question"] == "What is RAG?" and line["latency_ms"] == 42.0

    QUERIES_TOTAL.labels(hit="miss").inc()
    QUERY_LATENCY.observe(0.3)

    handler = setup_logging(logging.WARNING)
    assert handler is not None

    if not HAVE_PROMETHEUS:
        print("SKIP: real Prometheus objects need prometheus-client: pip install prometheus-client")
    else:
        print("OK: JSON logging formats records and metrics objects respond to inc/observe")
