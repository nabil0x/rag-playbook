"""Retrieval metrics against qrels — Recall@k, MRR@k, nDCG@k.

Evaluation block: the shared, canonical implementations of the three retrieval
metrics every qrels-lab in the curriculum uses (track 07 lab 1, and the
reranking/graph/agentic labs that measure retrieval). Hand-rolled so the
definitions are explicit and dependency-free; all three take the same shape:

    ranked_ids: list of corpus ids in retrieval order (the ranked candidate list)
    gold: set of relevant corpus ids from the qrels (score >= 1)
    k: ranking depth the metric is truncated at

Each returns a float in [0, 1] — a per-query score; labs aggregate with
``sum(scores) / len(scores)``. Binary relevance throughout (a corpus id is
either relevant or not); graded qrels are collapsed the same way the BEIR
convention does for retrieval evaluation.

See Topics/Project-24-Retrieval-Evaluation/README.md.
"""

from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    """Fraction of the gold set found within the top ``k`` ranks.

    0.0 when ``gold`` is empty (nothing to recall) or ``k`` is 0. Otherwise
    ``|top-k ∩ gold| / |gold|``.
    """
    if not gold or k <= 0:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & gold) / len(gold)


def mrr_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    """Mean-reciprocal-rank-style score for one query: 1/rank of the first hit.

    Returns ``1 / (rank + 1)`` (1-based rank) for the first gold id found
    within the top ``k`` positions, else 0.0. Called MRR only when averaged
    over queries; per-query it is the reciprocal rank.
    """
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    """Discounted cumulative gain normalized by the ideal ranking.

    Binary relevance: each gold id in position ``i`` (1-based) contributes
    ``1 / log2(i + 1)``. The ideal is the same sum over the first ``min(k,
    |gold|)`` positions. Returns 0.0 when ``gold`` is empty; the ideal is
    truncated at ``k`` so scores land in [0, 1].
    """
    if not gold or k <= 0:
        return 0.0
    gain = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold:
            gain += 1.0 / math.log2(i + 1)
    ideal = 0.0
    for i in range(1, min(k, len(gold)) + 1):
        ideal += 1.0 / math.log2(i + 1)
    return gain / ideal if ideal > 0.0 else 0.0


if __name__ == "__main__":
    # No-network smoke tests.
    gold = {"a", "c"}
    ranked = ["x", "a", "b", "c", "y"]
    assert recall_at_k(ranked, gold, 1) == 0.0
    assert recall_at_k(ranked, gold, 2) == 0.5  # found 1 of 2 by rank 2
    assert recall_at_k(ranked, gold, 5) == 1.0
    assert recall_at_k([], gold, 5) == 0.0
    assert recall_at_k(ranked, set(), 5) == 0.0

    assert mrr_at_k(ranked, gold, 1) == 0.0
    assert mrr_at_k(ranked, gold, 2) == 1.0 / 2.0  # "a" at 1-based rank 2
    assert mrr_at_k(["x", "y"], gold, 5) == 0.0

    assert abs(ndcg_at_k(ranked, gold, 5) -
               (1.0 / math.log2(3) + 1.0 / math.log2(5)) /
               (1.0 / math.log2(2) + 1.0 / math.log2(3))) < 1e-9
    assert ndcg_at_k(["a", "c"], gold, 5) == 1.0  # ideal order, full gain
    assert ndcg_at_k([], gold, 5) == 0.0

    print("OK: recall_at_k, mrr_at_k, ndcg_at_k implemented")
