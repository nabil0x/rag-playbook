"""Lab 02 — nDCG@k lift: measuring what a reranker buys you.

Lab 01 showed the cross-encoder rescuing gold passages one query at a time.
This lab turns that anecdote into a number: normalized Discounted Cumulative
Gain at k (nDCG@k) over a pool of queries, for two systems —

* baseline — the bi-encoder (BGE) retrieves its top-5 directly;
* reranked — the bi-encoder retrieves a WIDE top-20, then the cross-encoder
  re-scores those 20 and keeps the top-5.

The reranked system sees strictly more candidates, re-ordered by a model that
reads each (query, passage) pair as one input — the question is whether that
extra, more precise pass improves the ranking, and by how much.

nDCG@k is the standard graded ranking metric: it rewards relevant passages in
the top positions (log-discounted) and normalizes by the score of a perfect
ranking, so it lands in [0, 1] and can be averaged across queries. Relevance
is binary here — a passage is relevant (1) if the qrels mark it gold for the
query, else 0 — and the ideal DCG puts every gold passage first.

The pool is split into two groups, because that split is the lesson:

* CEILING — queries where the bi-encoder already puts gold at rank 1
  (baseline nDCG = 1.0). Nothing to fix; the reranker can only agree or
  disagree, and disagreement costs.
* HEADROOM — queries where gold is buried or missing from the bi top-5.
  This is where the reranker earns its keep: it re-scores the wider 20-item
  pool and promotes gold that cosine pooling missed.

Corpus: beir-nfcorpus (medical QA, 3633 docs). We index a deterministic
600-doc head and evaluate every qrels-covered query with gold inside it.

Run from the repo root:
    python curriculum/06-re-ranking/02-ndcg-lift.py
    python curriculum/06-re-ranking/02-ndcg-lift.py --verify
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/06-re-ranking/02-ndcg-lift.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from tools.reranker import CrossEncoderReranker  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
NFCORPUS_DIR = Path("Data/corpus/beir-nfcorpus/nfcorpus")
CORPUS_PATH = NFCORPUS_DIR / "corpus.jsonl"
QUERIES_PATH = NFCORPUS_DIR / "queries.jsonl"
QRELS_PATH = NFCORPUS_DIR / "qrels" / "test.tsv"
NF_N_DOCS = 600  # deterministic head of the 3633-doc nfcorpus corpus
NF_MAX_QUERIES = 80  # pool cap: qrels-covered queries with gold inside
WIDE_K = 20  # stage-1 bi-encoder candidate list
EVAL_K = 5  # ranking depth both systems are scored at
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
PREVIEW = 62  # max characters of passage text shown next to each hit


# --------------------------------------------------------------------------
# 2. Load — nfcorpus corpus + queries + qrels
# --------------------------------------------------------------------------
def load_nfcorpus(path: Path, n: int) -> tuple[list[str], list[str]]:
    """Return (doc_texts, doc_ids) for the first ``n`` corpus docs.

    Field choice: ``title + " " + text`` — nfcorpus titles are short keyword
    phrases ("Breast Cancer Cells Feed on Cholesterol") that the queries are
    written against, so they belong in the indexed text.
    """
    texts: list[str] = []
    ids: list[str] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            doc = json.loads(line)
            ids.append(doc["_id"])
            texts.append(f'{doc["title"]} {doc["text"]}')
    return texts, ids


def load_nf_queries(path: Path) -> list[tuple[str, str]]:
    """Return [(query_id, query_text)] for every query, in file order."""
    out: list[tuple[str, str]] = []
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            out.append((q["_id"], q["text"]))
    return out


def load_qrels(path: Path) -> dict[str, set[str]]:
    """Return {query_id: {relevant_corpus_id, ...}} (qrels score >= 1)."""
    qrels: dict[str, set[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3 or parts[0] == "query-id":
                continue  # header row
            qid, cid = parts[0], parts[1]
            if int(parts[2]) >= 1:
                qrels.setdefault(qid, set()).add(cid)
    return qrels


# --------------------------------------------------------------------------
# 3. Metric — nDCG@k with binary relevance (implemented by hand)
# --------------------------------------------------------------------------
def dcg_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    """Discounted cumulative gain at depth ``k`` for a ranked id list."""
    gain = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold:
            gain += 1.0 / math.log2(i + 1)
    return gain


def ndcg_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    """nDCG@k in [0, 1]: DCG of the ranking / DCG of a perfect ranking.

    The ideal ranking places every gold passage first (binary relevance), so
    its gain is ``sum_{i=1}^{min(k, |gold|)} 1/log2(i+1)``. A ranking with no
    gold inside the top-k scores 0.0.
    """
    ideal = dcg_at_k(sorted(gold), gold, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(ranked_ids, gold, k) / ideal


# --------------------------------------------------------------------------
# 4. Experiment — embed, index, score both systems over the query pool
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    nf_texts, nf_ids = load_nfcorpus(CORPUS_PATH, NF_N_DOCS)
    nf_queries = load_nf_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)

    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME)
    t0 = time.perf_counter()
    nf_vecs = embedder.embed_documents(nf_texts)
    embed_s = time.perf_counter() - t0

    chunks = [
        Document(page_content=t, metadata={"id": cid})
        for t, cid in zip(nf_texts, nf_ids)
    ]
    store = FAISSVectorStore(embedding=embedder)
    t0 = time.perf_counter()
    store.add(chunks, embeddings=nf_vecs)
    index_s = time.perf_counter() - t0

    bi_wide = SimilarityRetriever(store, top_k=WIDE_K)
    reranker = CrossEncoderReranker()

    subset_ids = set(nf_ids)
    covered = [
        (qid, q) for qid, q in nf_queries
        if qid in qrels and qrels[qid] & subset_ids
    ][:NF_MAX_QUERIES]

    t0 = time.perf_counter()
    rows = []
    for qid, qtext in covered:
        gold = qrels[qid] & subset_ids
        wide = bi_wide.retrieve(qtext)

        baseline_ids = [d.metadata["id"] for d in wide[:EVAL_K]]
        reranked = reranker.rerank(qtext, wide, top_k=EVAL_K)
        reranked_ids = [d.metadata["id"] for d in reranked]

        rows.append(
            {
                "qid": qid,
                "question": qtext,
                "n_gold": len(gold),
                "ndcg_base": ndcg_at_k(baseline_ids, gold, EVAL_K),
                "ndcg_rerank": ndcg_at_k(reranked_ids, gold, EVAL_K),
            }
        )
    score_s = time.perf_counter() - t0

    ceiling = [r for r in rows if r["ndcg_base"] >= 1.0 - 1e-9]
    headroom = [r for r in rows if r["ndcg_base"] < 1.0 - 1e-9]

    def mean(rows_: list[dict], key: str) -> float:
        return sum(r[key] for r in rows_) / len(rows_) if rows_ else 0.0

    return {
        "rows": rows,
        "ceiling": ceiling,
        "headroom": headroom,
        "mean_base": mean(rows, "ndcg_base"),
        "mean_rerank": mean(rows, "ndcg_rerank"),
        "hr_mean_base": mean(headroom, "ndcg_base"),
        "hr_mean_rerank": mean(headroom, "ndcg_rerank"),
        "hr_improved": sum(1 for r in headroom if r["ndcg_rerank"] > r["ndcg_base"]),
        "hr_worse": sum(1 for r in headroom if r["ndcg_rerank"] < r["ndcg_base"]),
        "ceiling_worse": sum(1 for r in ceiling if r["ndcg_rerank"] < r["ndcg_base"]),
        "indexed": len(nf_texts),
        "embed_s": embed_s,
        "index_s": index_s,
        "score_s": score_s,
    }


# --------------------------------------------------------------------------
# 5. Demo — print the artifact
# --------------------------------------------------------------------------
def preview(text: str, limit: int = PREVIEW) -> str:
    """Flatten a passage for one-line printing."""
    flat = text.replace("\n", " ")
    return flat[:limit] + ("..." if len(flat) > limit else "")


def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 02 — nDCG@k lift: measuring what a reranker buys you")
    print(f"{BGE_MODEL_NAME} bi-encoder (top-{EVAL_K}) vs "
          f"bi top-{WIDE_K} -> cross-encoder top-{EVAL_K}")
    print("=" * 66)

    print(f"\n[1] Corpus (deterministic subset, no randomness):")
    print(f"    {exp['indexed']} nfcorpus docs (first {NF_N_DOCS} of 3633)")
    print(f"    embedded in {exp['embed_s']:.1f}s, indexed in {exp['index_s']:.3f}s")

    print(f"\n[2] Pool: {len(exp['rows'])} qrels-covered queries, "
          f"nDCG@{EVAL_K} per system (top-8 headroom lifts)")
    print(f"    {'query':<16}{'gold':>5}{'bi':>9}{'ce':>9}{'delta':>9}")
    top = sorted(exp["headroom"], key=lambda r: r["ndcg_rerank"] - r["ndcg_base"],
                 reverse=True)[:8]
    for r in top:
        delta = r["ndcg_rerank"] - r["ndcg_base"]
        print(f"    {r['qid']:<16}{r['n_gold']:>5}"
              f"{r['ndcg_base']:>9.3f}{r['ndcg_rerank']:>9.3f}{delta:>+9.3f}")

    print(f"\n[3] Mean nDCG@{EVAL_K}, by group:")
    print(f"    ALL queries ({len(exp['rows'])}): "
          f"bi {exp['mean_base']:.4f} -> ce {exp['mean_rerank']:.4f} "
          f"({exp['mean_rerank'] - exp['mean_base']:+.4f})")
    print(f"    CEILING  ({len(exp['ceiling'])} queries, bi already nDCG=1.0): "
          f"rerank {exp['ceiling_worse']} worse, rest tied — "
          f"no headroom to gain")
    print(f"    HEADROOM ({len(exp['headroom'])} queries, bi < 1.0): "
          f"{exp['hr_mean_base']:.4f} -> {exp['hr_mean_rerank']:.4f} "
          f"({exp['hr_mean_rerank'] - exp['hr_mean_base']:+.4f}), "
          f"{exp['hr_improved']} improved vs {exp['hr_worse']} worse")

    print(f"\n[4] Takeaway")
    print("    A reranker can only help where the bi-encoder left headroom.")
    print("    When cosine pooling already put gold at rank 1 (the ceiling")
    print("    group), the cross-encoder has nothing to fix and can even")
    print("    disagree. The lift concentrates in the headroom group — gold")
    print("    buried or missing from the bi top-5 — where pair-level scoring")
    print("    promotes it. That is why production systems evaluate rerank")
    print("    on nDCG@k over queries with room to improve, and why 'the")
    print("    reranker made it worse' usually means you measured a ceiling.")


# --------------------------------------------------------------------------
# 6. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    checks.append((f"exactly {NF_N_DOCS} nfcorpus docs indexed",
                   exp["indexed"] == NF_N_DOCS))
    checks.append((f"pool has {len(exp['rows'])} queries (>= 40)",
                   len(exp["rows"]) >= 40))
    checks.append(("ceiling group is defined (baseline nDCG == 1.0)",
                   all(r["ndcg_base"] == 1.0 for r in exp["ceiling"])))
    checks.append(("headroom group is defined (baseline nDCG < 1.0)",
                   all(r["ndcg_base"] < 1.0 for r in exp["headroom"])))

    for r in exp["rows"]:
        tag = f"Q{r['qid']}"
        checks.append((f"{tag} baseline nDCG in [0, 1]",
                       0.0 <= r["ndcg_base"] <= 1.0))
        checks.append((f"{tag} reranked nDCG in [0, 1]",
                       0.0 <= r["ndcg_rerank"] <= 1.0))

    checks.append(("HEADROOM: reranked mean nDCG > baseline mean nDCG",
                   exp["hr_mean_rerank"] > exp["hr_mean_base"]))
    checks.append(("HEADROOM: lift is non-trivial (>= +0.02)",
                   exp["hr_mean_rerank"] - exp["hr_mean_base"] >= 0.02))
    checks.append(("HEADROOM: at least 15 queries improve",
                   exp["hr_improved"] >= 15))
    checks.append(("CEILING: reranker hurts at most a third of ceiling queries",
                   exp["ceiling_worse"] <= max(1, len(exp["ceiling"]) // 3)))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
