"""RAG as a FastAPI service.

Production block: wrap the Project 17 pipeline in HTTP — ``POST /query``
(cached in Redis), ``POST /ingest`` (add documents at runtime), and
``GET /health`` — then ship the whole thing in Docker. Redis is optional:
with it down the service degrades to the cache-miss path, which is correct,
just slower. See Topics/Project-32-RAG-Service/README.md.
"""

from __future__ import annotations

import os

try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    HAVE_FASTAPI = True
except ImportError:  # fastapi is optional: importing this module never fails
    HAVE_FASTAPI = False

#: How long a cached answer lives before it is re-computed.
CACHE_TTL = int(os.environ.get("RAG_CACHE_TTL", "3600"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def normalize_question(question: str) -> str:
    """Lowercase a question and collapse whitespace — the cache-key form."""
    return " ".join(question.strip().lower().split())


def _cache_key(question: str) -> str:
    """Redis key for a question's cached answer."""
    return f"rag:query:{normalize_question(question)}"


def get_redis():
    """Lazily connect to Redis; return None when it is missing or down."""
    try:
        import redis
    except ImportError:
        return None
    try:
        return redis.Redis.from_url(
            REDIS_URL, socket_connect_timeout=2, socket_timeout=2, decode_responses=True
        )
    except Exception:  # noqa: BLE001 — Redis down is a cache miss, not a crash
        return None


def _cache_get(client, key: str) -> str | None:
    """TODO(Project 32): fetch a cached answer.

    Return the JSON string under ``key``, or None on miss. ``client`` may be
    None (Redis down) — treat that as a miss too. ``GET`` on a missing key
    already returns None, so this can be a thin wrapper.
    """
    raise NotImplementedError("TODO(Project 32): implement api.main._cache_get")


def _cache_set(client, key: str, value: str, ttl: int = CACHE_TTL) -> None:
    """TODO(Project 32): store an answer with an expiry.

    ``client.set(name, value, ex=ttl)`` — the TTL is the caching rule that
    matters: answers go stale when the corpus changes, so nothing lives
    forever. ``client`` may be None (Redis down) — skip silently.
    """
    raise NotImplementedError("TODO(Project 32): implement api.main._cache_set")


def _get_pipeline():
    """TODO(Project 32): assemble and cache the Project 17 RAGPipeline.

    Lazy-import the blocks (``loaders.*``, ``splitters.*``, ``embeddings.*``,
    ``vectordb.*``, ``retrieval.*``, ``llms.*``) and build a ``RAGPipeline``
    exactly like ``main.py`` does — then keep it on a module/global so the
    endpoints reuse it instead of rebuilding per request.
    """
    raise NotImplementedError("TODO(Project 32): implement api.main._get_pipeline")


if HAVE_FASTAPI:

    class QueryRequest(BaseModel):
        question: str

    class QueryResponse(BaseModel):
        answer: str
        sources: list[str]
        cached: bool
        latency_ms: float

    class IngestRequest(BaseModel):
        file_path: str

    class IngestResponse(BaseModel):
        status: str
        chunks_added: int

    def query_handler(request: QueryRequest) -> QueryResponse:
        """TODO(Project 32): answer a question, using the Redis cache.

        Flow: ``_cache_get`` on ``_cache_key(question)``; on hit return the
        cached answer with ``cached=True``; on miss call
        ``_get_pipeline().ask(question)``, ``_cache_set`` the answer, and
        return with ``cached=False``. ``sources`` come from the retrieved
        chunks (``metadata["source"]``) and ``latency_ms`` is wall time around
        the LLM call — the number the "second call is faster" demo measures.
        """
        raise NotImplementedError("TODO(Project 32): implement POST /query")

    def ingest_handler(request: IngestRequest) -> IngestResponse:
        """TODO(Project 32): add a document at runtime.

        Run the pipeline ingest steps for ``request.file_path`` (load -> split
        -> embed -> store, the same lazy imports as ``_get_pipeline``), then
        invalidate the cache (delete ``rag:query:*`` keys) so stale answers do
        not survive new data.
        """
        raise NotImplementedError("TODO(Project 32): implement POST /ingest")

    def health_handler() -> dict:
        """Liveness probe for orchestration (load balancers, compose healthcheck)."""
        return {"status": "ok", "service": "rag-api"}

    def create_app() -> FastAPI:
        app = FastAPI(title="RAG Service", version="1.0.0")
        app.post("/query", response_model=QueryResponse)(query_handler)
        app.post("/ingest", response_model=IngestResponse)(ingest_handler)
        app.get("/health")(health_handler)
        return app

    app = create_app()
else:
    app = None


if __name__ == "__main__":
    # No-network smoke test — pure-stdlib parts first, app checks when fastapi
    # is installed. No handler is invoked (the /query and /ingest bodies are TODOs).
    assert normalize_question("  What   IS RAG? ") == "what is rag?"
    assert _cache_key("What is RAG?") == "rag:query:what is rag?"

    if not HAVE_FASTAPI:
        print("SKIP: FastAPI app checks need fastapi: pip install fastapi uvicorn redis")
    else:
        paths = {route.path for route in app.routes}
        assert {"/query", "/ingest", "/health"} <= paths, paths
        assert health_handler()["status"] == "ok"
        print("OK: normalize_question, cache key, and /query /ingest /health routes registered")
