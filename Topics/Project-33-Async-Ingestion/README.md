# Project 33 — Async Ingestion

> **Goal:** Move document ingestion off the request path — a Celery worker
> backed by Redis turns slow embedding into background jobs with status
> polling, retries, and idempotency.

## Why

In Project 32, `POST /ingest` blocks until the document is embedded and stored.
That is fine for one file and brutal for a real one: embedding 10,000 chunks
can take minutes, the HTTP client times out, and a crash mid-ingest leaves a
half-built index with no record of what finished. The production answer is a
task queue: the API enqueues an `ingest_document` job, returns a `task_id`
immediately, and a worker (separate process, scaled independently) does the
slow work. Three lessons come free with the pattern: status polling (the
client asks `GET /ingest/{task_id}` until it says SUCCESS), retries (a failed
job retries with backoff instead of silently vanishing), and idempotency (a
content hash prevents the same document from being embedded twice when a retry
re-enqueues it).

## Learn

- Sync vs async ingestion: what blocks, what doesn't, and where the timeout lives
- Celery fundamentals: `@app.task`, the Redis broker, and the worker process (`celery -A celery_app worker`)
- Status lifecycle: PENDING → STARTED → SUCCESS / FAILURE, and polling `AsyncResult`
- Retries with backoff: `autoretry_for=(Exception,)`, `retry_backoff=True`, `max_retries`
- Idempotency: content-hash the source document; skip the job if it is already indexed
- Keeping the API thin: FastAPI enqueues and polls; the worker owns the pipeline

## Execute

1. **Setup** — `pip install celery redis`; Redis running locally (or reuse the Project 32 compose file)
2. **Read** — `celery_app.py` — the app config, the `ingest_document` task skeleton, and the idempotency hook
3. **Implement** — the task body (load → split → embed → store, all lazy imports), the content-hash guard, and retry config
4. **Run** — start the worker (`celery -A celery_app worker --loglevel=info`), enqueue 3 files, poll their statuses
5. **Measure** — request returns instantly with a `task_id`; worker logs show per-document progress; re-enqueue the same file and watch it skip
6. **Acceptance criteria** — enqueuing returns before embedding completes; status polls move PENDING → SUCCESS; a forced exception (bad file) ends FAILURE then retries; a duplicate document is skipped, not re-embedded

## Stretch

- Add a Celery beat schedule for periodic re-indexing of a watched folder
- Add per-task progress reporting (chunks done / total) via `self.update_state`
- Wire the worker into Project 32's API (`POST /ingest` enqueues, `GET /ingest/{task_id}` polls)

## Article

- [ ] `33-async-ingestion.md`

## Code

- `celery_app.py` — Celery app + `ingest_document(file_path)` task with retries and content-hash idempotency

## Notebook

`NoteBooks/Project-33-Async-Ingestion/01-async-ingestion-spec.py` → generate with:

```bash
python scripts/gen_notebook.py NoteBooks/Project-33-Async-Ingestion/01-async-ingestion-spec.py NoteBooks/Project-33-Async-Ingestion/01-async-ingestion.ipynb
```
