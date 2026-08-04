"""Online evaluation and A/B testing.

Evaluation block: capture real user feedback on live answers (thumbs, stars)
persisted to SQLite, summarize it per variant, and run an A/B test between two
retrievers with a significance check — learning why small ``n`` swamps real
differences. Stdlib only (sqlite3); scipy optional for the t-test.
See Topics/Project-35-Online-Eval/README.md.
"""

from __future__ import annotations

import random
import sqlite3
import time
from dataclasses import dataclass, field

FEEDBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating REAL NOT NULL,
    variant TEXT NOT NULL,
    latency_ms REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
)
"""


@dataclass
class FeedbackEvent:
    """One user rating of one live answer."""

    question: str
    answer: str
    rating: float  # e.g. 0-5 stars, or 0/1 thumbs
    variant: str  # "A" or "B" — which retriever served the answer
    latency_ms: float = 0.0
    created_at: float = field(default_factory=time.time)


class OnlineEvaluator:
    """Append-only feedback log backed by SQLite, with per-variant summaries."""

    def __init__(self, db_path: str = "feedback.sqlite3"):
        self.db_path = db_path
        self._conn = self._connect()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute(FEEDBACK_SCHEMA)
        return conn

    def log(self, event: FeedbackEvent) -> int:
        """TODO(Project 35): persist one feedback event.

        ``INSERT`` the event fields into the ``feedback`` table and return the
        new row id (``cursor.lastrowid``). Commit after the insert so events
        survive a crash — the "persist across runs" acceptance criterion.
        """
        raise NotImplementedError("TODO(Project 35): implement OnlineEvaluator.log")

    def summary(self) -> dict[str, dict]:
        """TODO(Project 35): per-variant statistics over all logged events.

        Return e.g. ``{"A": {"count": n, "mean": x, "std": s}, "B": {...}}`` —
        one entry per variant present in the table. A ``GROUP BY variant``
        query with the count/avg/stddev aggregates keeps it to one statement.
        """
        raise NotImplementedError("TODO(Project 35): implement OnlineEvaluator.summary")

    def close(self) -> None:
        self._conn.close()


class ABTest:
    """Assign traffic to variants and compare their outcomes."""

    def __init__(self, variant_a: str = "A", variant_b: str = "B", seed: int | None = None):
        self.variant_a = variant_a
        self.variant_b = variant_b
        self._rng = random.Random(seed)

    def assign(self, user_id: str) -> str:
        """TODO(Project 35): pick a variant for ``user_id``.

        Random (``self._rng.random() < 0.5``) or round-robin — your call, as
        long as it is *stable per user* so the same user gets the same variant
        across sessions. That stability is what makes the comparison fair.
        """
        raise NotImplementedError("TODO(Project 35): implement ABTest.assign")

    def compare(self, events: list[FeedbackEvent], metric: str = "rating") -> dict:
        """TODO(Project 35): compare variants with a significance check.

        Return ``{"variant_a": {"count": .., "mean": ..}, "variant_b": {...},
        "delta": .., "significant": bool}``. Use the two-sample t-test
        (``scipy.stats.ttest_ind`` when scipy is installed; a standard-error
        heuristic otherwise). The point of the exercise: the same delta that
        is "significant" at n=100 is not at n=20 — note the n in the verdict.
        """
        raise NotImplementedError("TODO(Project 35): implement ABTest.compare")


if __name__ == "__main__":
    # No-network smoke test — schema + wiring only (log/summary/assign/compare
    # are learner TODOs and are not invoked).
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "feedback.sqlite3")
        evaluator = OnlineEvaluator(db_path)
        try:
            tables = [
                row["name"]
                for row in evaluator._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            ]
            assert "feedback" in tables, tables
        finally:
            evaluator.close()

    event = FeedbackEvent(question="q", answer="a", rating=4.0, variant="A")
    assert (event.question, event.answer, event.rating, event.variant) == ("q", "a", 4.0, "A")

    ab = ABTest(seed=1)
    assert ab.variant_a == "A" and ab.variant_b == "B"

    print("OK: feedback schema created, FeedbackEvent and ABTest constructed")
