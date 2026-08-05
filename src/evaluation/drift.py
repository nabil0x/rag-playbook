"""Query drift and retrieval hit-rate monitoring.

Evaluation block: watch the live query stream — embed incoming questions,
compare the recent centroid against the baseline, flag when the distribution
moves — plus a retrieval hit-rate check against the golden set to catch index
problems. Flag, don't page: drift is a signal to investigate, not an alert.
See Topics/Project-36-Drift-Prompt-Regression/README.md.
"""

from __future__ import annotations

from typing import Any


class DriftDetector:
    """Compare recent query embeddings against a baseline centroid.

    Baselines are given as raw query strings at construction: they are
    embedded once and averaged into ``baseline_centroid``. Every ``record()``
    call appends a new query vector to the rolling window; ``check()``
    (a TODO) turns the distance between the window centroid and the baseline
    into a drift verdict against ``threshold``.
    """

    #: Mean cosine distance above which the window is flagged as drifted.
    DEFAULT_THRESHOLD = 0.15

    def __init__(
        self,
        embedder,
        baseline_queries: list[str],
        threshold: float = DEFAULT_THRESHOLD,
        window: int = 100,
    ):
        # embedder: any object exposing embed_documents(texts) -> list[list[float]]
        # (e.g. evaluation.judge.LocalEmbeddings, or fastembed directly).
        self.embedder = embedder
        self.threshold = threshold
        self.window = window
        self._recent: list[list[float]] = []
        baseline_vectors = embedder.embed_documents(baseline_queries)
        self.baseline_centroid = self._centroid(baseline_vectors)

    @staticmethod
    def _centroid(vectors: list[list[float]]) -> list[float]:
        """Mean of the vectors — the reference point for drift distance."""
        n = len(vectors)
        if n == 0:
            return []
        dim = len(vectors[0])
        return [sum(v[i] for v in vectors) / n for i in range(dim)]

    def record(self, query: str) -> None:
        """Embed ``query`` and append it to the rolling window (trimmed to size)."""
        vector = self.embedder.embed_documents([query])[0]
        self._recent.append(vector)
        if len(self._recent) > self.window:
            self._recent = self._recent[-self.window :]

    def drift_score(self) -> float:
        """TODO(Project 36): mean cosine distance, window centroid vs baseline.

        Cosine distance = 1 - cosine similarity. Comparing the recent window's
        centroid against ``baseline_centroid`` smooths individual-query noise;
        averaging over ``min(len(window), k)`` random window vectors is the
        Stretch refinement. Returns a float in [0, 2].
        """
        raise NotImplementedError("TODO(Project 36): implement DriftDetector.drift_score")

    def check(self) -> dict[str, Any]:
        """TODO(Project 36): the drift verdict.

        Return ``{"drifted": bool, "score": float, "threshold": float,
        "window_size": int}`` where ``drifted = score > threshold``. Include
        the window size so the caller knows how much evidence the verdict
        rests on — a verdict on 3 queries is a hint, not a finding.
        """
        raise NotImplementedError("TODO(Project 36): implement DriftDetector.check")

    def hit_rate(self, golden_questions: list[str], retriever, k: int = 3) -> float:
        """TODO(Project 36): retrieval hit-rate over the golden set.

        For each golden question, retrieve top-k and score 1.0 when any
        returned chunk's metadata id matches the golden document id, else 0.0;
        return the mean. A dropping hit rate signals index/retriever problems
        (retrieval drift) rather than question drift — different fix.
        """
        raise NotImplementedError("TODO(Project 36): implement DriftDetector.hit_rate")


if __name__ == "__main__":
    # No-network smoke test with a fake embedder — exercises the implemented
    # scaffold only (drift_score/check/hit_rate are learner TODOs).
    class _FakeEmbedder:
        """Word-set vectors: overlapping vocab means overlapping vectors."""

        VOCAB = ["rag", "retrieval", "invoice", "tax", "pdf"]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0 if w in text.lower() else 0.0 for w in self.VOCAB] for text in texts]

    embedder = _FakeEmbedder()
    detector = DriftDetector(
        embedder,
        baseline_queries=["rag retrieval", "retrieval system", "rag pipeline"],
        threshold=0.15,
        window=5,
    )

    # Baseline centroid is the mean of the baseline vectors.
    expected = [2 / 3, 2 / 3, 0.0, 0.0, 0.0]
    assert len(detector.baseline_centroid) == 5
    assert all(abs(c - e) < 1e-9 for c, e in zip(detector.baseline_centroid, expected))

    # record() embeds queries and trims the window to size.
    for _ in range(6):
        detector.record("rag")
    assert len(detector._recent) == detector.window == 5
    assert all(v[0] == 1.0 for v in detector._recent)  # every query mentions "rag"

    print("OK: DriftDetector computes the baseline centroid and maintains the window")
