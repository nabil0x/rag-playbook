"""Async document ingestion with Celery.

Production block: move load -> split -> embed -> store off the HTTP request
path. The API enqueues an ``ingest_document`` job and polls its status; a
worker (``celery -A src.celery_app worker``) executes it with retries and a
content-hash idempotency guard. Celery is optional: importing this module
never fails, it just leaves ``app`` as None. See Topics/Project-33-Async-Ingestion/README.md.
"""

from __future__ import annotations

import hashlib
import os

try:
    from celery import Celery

    HAVE_CELERY = True
except ImportError:  # celery is optional: importing this module never fails
    HAVE_CELERY = False

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def _content_hash(file_path: str) -> str:
    """SHA-256 of the file bytes — the idempotency key for a document."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


if HAVE_CELERY:

    app = Celery("rag_ingestion", broker=REDIS_URL, backend=REDIS_URL)

    @app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
    def ingest_document(self, file_path: str) -> dict:
        """TODO(Project 33): load, split, embed, and store one document.

        Body (all lazy imports, same blocks ``main.py`` uses):

        1. Compute ``_content_hash(file_path)`` and look it up in the store —
           if already indexed, return ``{"status": "skipped", "hash": ...}``
           (the idempotency guard: a retry must never double-embed).
        2. Load -> split -> embed -> store the document.
        3. Return ``{"status": "indexed", "chunks": n, "hash": ...}``.

        ``self`` (the bound task) gives ``self.update_state`` for the Stretch
        progress reporting; the decorator's ``autoretry_for``/``retry_backoff``
        already retry failures with exponential backoff.
        """
        raise NotImplementedError("TODO(Project 33): implement celery_app.ingest_document")

else:
    app = None


if __name__ == "__main__":
    # No-network smoke test — hash helper and Celery configuration only.
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        tmp_path = f.name
    try:
        h1 = _content_hash(tmp_path)
        assert h1 == _content_hash(tmp_path)
        # sha256("hello world") — deterministic, no newline.
        assert h1 == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    finally:
        os.unlink(tmp_path)

    if not HAVE_CELERY:
        print("SKIP: Celery task checks need celery + redis: pip install celery redis")
    else:
        assert app is not None and app.main == "rag_ingestion"
        task = app.tasks.get("celery_app.ingest_document")
        assert task is not None
        assert task.max_retries == 3 and task.retry_backoff is True
        assert task.autoretry_for == (Exception,)
        print("OK: Celery app + ingest_document task configured (run: celery -A src.celery_app worker)")
