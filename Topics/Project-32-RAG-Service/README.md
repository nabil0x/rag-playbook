# Project 32 — RAG as a Service

> **Goal:** Wrap your pipeline in a FastAPI service — `POST /query`, `POST /ingest`,
> `GET /health` — add a Redis cache so repeated questions skip the LLM, and
> ship the whole thing in Docker.

## Why

The pipeline in `src/main.py` is a script: it answers one question in-process. A
real deployment is a service — something that can sit behind a load balancer,
be called by other programs, and be restarted without losing state. This
project turns the pipeline into HTTP: a `POST /query` endpoint that returns
answer + sources, a `POST /ingest` endpoint that adds documents at runtime, and
a `GET /health` for orchestration. Then the first production lesson: LLM calls
are the expensive part, and repeated questions waste them — so you cache
query results in Redis keyed by the normalized question, and you learn the
caching rule that actually matters (invalidate on ingest, TTL on everything).
Finally, Docker: one `Dockerfile`, one `docker-compose.yml` (api + redis), and
the same service runs anywhere.

## Learn

- FastAPI basics: routes, Pydantic request/response models, `uvicorn` as the server
- API contract design: what `/query`, `/ingest`, and `/health` accept and return, and why the contract matters
- Redis caching: key = normalized question (lowercase + collapse whitespace), TTL, and invalidation on ingest
- Graceful degradation: the service works with Redis down (cache miss path), and reports `SKIP` hints when optional deps are missing
- Docker: `Dockerfile` build, `docker-compose.yml` multi-service topology (api + redis), and env-based config (`REDIS_URL`)

## Execute

1. **Setup** — `pip install fastapi uvicorn redis`; Docker Desktop or daemon for the container step
2. **Read** — `src/api/main.py` (routes + lazy pipeline assembly), `docker/Dockerfile`, `docker/docker-compose.yml`
3. **Implement** — the `/query` handler (pipeline call + cache lookup/store) and `/ingest` handler (load → split → embed → store)
4. **Run** — `uvicorn src.api.main:app --reload`, then `curl localhost:8000/health` and a `POST /query`
5. **Measure** — query twice: second call is a cache hit (compare latency); `POST /ingest` a new file, then verify the cached answer is gone
6. **Acceptance criteria** — repeated identical question returns from cache (measurably faster, no new LLM call); ingest invalidates the cache; `docker compose up` serves the same API with redis on its own container

## Stretch

- Add `/query` streaming (SSE) and see the latency perception change
- Add auth via a simple API key header
- Kubernetes deployment of the same compose file — the deferred stretch from the roadmap

## Article

- [ ] `32-rag-as-a-service.md`

## Code

- `src/api/main.py` — FastAPI app: `/query` (cached), `/ingest`, `/health`
- `docker/Dockerfile` — python:3.11-slim build; `docker/docker-compose.yml` — api + redis

## Notebook

`NoteBooks/Projects/Project-32-RAG-Service/01-rag-service-spec.py` → generate with:

```bash
python src/scripts/gen_notebook.py NoteBooks/Projects/Project-32-RAG-Service/01-rag-service-spec.py NoteBooks/Projects/Project-32-RAG-Service/01-rag-service.ipynb
```
