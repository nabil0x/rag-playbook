"""OpenTelemetry tracing for the RAG pipeline.

Observability block: give every question one trace — a root ``rag.query``
span with per-block child spans (``ingest``, ``retrieve``, ``generate``) that
share one trace ID. OTel is optional: without it, ``init_tracing`` returns a
no-op tracer so pipeline code runs unchanged. See Topics/Project-34-Observability/README.md.
"""

from __future__ import annotations

from contextlib import nullcontext


class _NoopTracer:
    """Tracer stand-in when opentelemetry is not installed.

    Implements the only surface the pipeline touches — ``start_as_current_span``
    — as a null context, so traced code never branches on availability.
    """

    def start_as_current_span(self, name: str, attributes: dict | None = None):
        return nullcontext()

    def start_span(self, name: str, attributes: dict | None = None):
        return nullcontext()


def init_tracing(service_name: str = "rag-service"):
    """Configure the OpenTelemetry SDK and return a tracer (or a no-op).

    Local-friendly setup: a ConsoleSpanExporter with a simple processor —
    spans print to stdout, no collector or SaaS required. Returns the real
    tracer when OTel is installed, else a ``_NoopTracer``.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    except ImportError:
        return _NoopTracer()
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


def trace_pipeline(question: str, tracer=None):
    """TODO(Project 34): wrap one pipeline run in a trace.

    Expected shape (see the card's acceptance criteria): one root span
    ``rag.query`` with the question as an attribute, and three child spans —
    ``ingest``, ``retrieve``, ``generate`` — each with attributes for chunk
    count / model / latency. All four share the same trace ID because they
    are started under one ``tracer.start_as_current_span("rag.query")``
    context. Return the trace ID (or a placeholder when the tracer is a no-op)
    so logs and the API response can reference it.
    """
    raise NotImplementedError("TODO(Project 34): implement observability.tracing.trace_pipeline")


if __name__ == "__main__":
    # No-network smoke test — a tracer must be usable either way.
    tracer = init_tracing("smoke")
    assert hasattr(tracer, "start_as_current_span")
    with tracer.start_as_current_span("rag.query"):
        pass
    print("OK: init_tracing returns a usable tracer (noop when OTel is absent)")
