"""Lab 01 — Recall@k, MRR, nDCG against the BEIR fiqa qrels.

The evaluation track opens with the three retrieval metrics every later lab
reuses. All three answer "did the retriever put the right documents near the
top?" — but they measure different failure modes:

* Recall@k — did the gold documents make it into the top-k *at all*? Ignores
  order inside the window. A recall ceiling means the retriever never even
  surfaces the answer.
* MRR@10 — how quickly does the FIRST gold document appear? Rewards getting
  one right document to the very top, ignores the rest.
* nDCG@10 — position-weighted overall ranking quality. Still rewards an early
  hit, but keeps counting the other gold documents further down.

Setup: the first 8,000 fiqa documents (deterministic head — embedding the
full 57,638 takes too long for a lab) are indexed in FAISS with the local BGE
embedder, then every test query whose gold documents are inside the subset is
retrieved at top-10 and scored with all three metrics. The shared
implementations live in ``evaluation/retrieval_metrics.py``.

Run from the repo root:
    python curriculum/07-evaluation/01-recall-mrr-ndcg.py
    python curriculum/07-evaluation/01-recall-mrr-ndcg.py --verify
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/07-evaluation/01-recall-mrr-ndcg.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from embeddings.bge import BGEEmbedding  # noqa: E402
from evaluation.retrieval_metrics import (  # noqa: E402
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from langchain_core.documents import Document  # noqa: E402
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
FIQA_DIR = Path("Data/corpus/beir-fiqa/fiqa")
CORPUS_PATH = FIQA_DIR / "corpus.jsonl"
QUERIES_PATH = FIQA_DIR / "queries.jsonl"
QRELS_PATH = FIQA_DIR / "qrels" / "test.tsv"
FIQA_N_DOCS = 8000  # deterministic head of the 57,638-doc fiqa corpus
FIQA_MAX_QUERIES = 60  # pool cap: first N qrels-covered queries with gold inside
EVAL_K = 10  # ranking depth every metric is computed at
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"


# --------------------------------------------------------------------------
# 2. Load — fiqa corpus + queries + qrels (BEIR layout, same as nfcorpus)
# --------------------------------------------------------------------------
def load_corpus(path: Path, n: int) -> tuple[list[str], list[str]]:
    """Return (doc_texts, doc_ids) for the first ``n`` corpus docs."""
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


def load_queries(path: Path) -> list[tuple[str, str]]:
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
                continue
            qid, cid = parts[0], parts[1]
            if int(parts[2]) >= 1:
                qrels.setdefault(qid, set()).add(cid)
    return qrels


# --------------------------------------------------------------------------
# 3. Experiment — index the subset, retrieve top-10 per query, score
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    fiqa_texts, fiqa_ids = load_corpus(CORPUS_PATH, FIQA_N_DOCS)
    fiqa_queries = load_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)

    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME)
    t0 = time.perf_counter()
    fiqa_vecs = embedder.embed_documents(fiqa_texts)
    embed_s = time.perf_counter() - t0

    chunks = [
        Document(page_content=t, metadata={"id": cid})
        for t, cid in zip(fiqa_texts, fiqa_ids)
    ]
    store = FAISSVectorStore(embedding=embedder)
    t0 = time.perf_counter()
    store.add(chunks, embeddings=fiqa_vecs)
    index_s = time.perf_counter() - t0

    retriever = SimilarityRetriever(store, top_k=EVAL_K)

    subset_ids = set(fiqa_ids)
    covered = [
        (qid, q) for qid, q in fiqa_queries
        if qid in qrels and qrels[qid] & subset_ids
    ][:FIQA_MAX_QUERIES]

    t0 = time.perf_counter()
    rows: list[dict] = []
    for qid, qtext in covered:
        gold = qrels[qid] & subset_ids
        ranked = [d.metadata["id"] for d in retriever.retrieve(qtext)]
        rows.append({
            "qid": qid,
            "gold": sorted(gold),
            "recall_1": recall_at_k(ranked, gold, 1),
            "recall_5": recall_at_k(ranked, gold, 5),
            "recall_10": recall_at_k(ranked, gold, 10),
            "mrr_10": mrr_at_k(ranked, gold, 10),
            "ndcg_10": ndcg_at_k(ranked, gold, 10),
        })
    retrieve_s = time.perf_counter() - t0

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return {
        "rows": rows,
        "indexed": len(fiqa_texts),
        "embed_s": embed_s,
        "index_s": index_s,
        "retrieve_s": retrieve_s,
        "metrics": {
            "recall@1": mean("recall_1"),
            "recall@5": mean("recall_5"),
            "recall@10": mean("recall_10"),
            "mrr@10": mean("mrr_10"),
            "ndcg@10": mean("ndcg_10"),
        },
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 07-01 — Recall@k, MRR, nDCG against the BEIR fiqa qrels")
    print(f"fiqa {exp['indexed']} docs, pool {len(exp['rows'])} queries, "
          f"k = {EVAL_K}")
    print("=" * 66)

    m = exp["metrics"]
    print(f"\n[1] Mean retrieval metrics over {len(exp['rows'])} queries:")
    for label in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"):
        print(f"    {label:10s}: {m[label]:.4f}")

    print(f"\n[2] Three example rows:")
    for row in exp["rows"][:3]:
        print(f"    {row['qid']}: gold={row['gold']}, R@5={row['recall_5']:.2f}, "
              f"MRR@10={row['mrr_10']:.2f}, nDCG@10={row['ndcg_10']:.2f}")

    print(f"\n[3] Timing: embed {exp['indexed']} docs {exp['embed_s']:.1f}s, "
          f"index {exp['index_s']:.1f}s, retrieve {exp['retrieve_s']:.1f}s")

    print(f"\n[4] Takeaway")
    print("    Recall and MRR/nDCG measure different things: recall@1 shows")
    print("    how often the BEST retriever position already holds a gold")
    print("    document, while nDCG@10 rewards the overall ordering of all")
    print("    gold documents. The gap between recall@10 and recall@1 is the")
    print("    headroom that reranking (track 06) exists to harvest.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    m = exp["metrics"]
    pool = len(exp["rows"])

    checks.append((f"exactly {FIQA_N_DOCS} fiqa docs indexed",
                   exp["indexed"] == FIQA_N_DOCS))
    checks.append((f"pool has {pool} queries (>= 40)", pool >= 40))

    for label in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"):
        checks.append((f"{label} in [0, 1]", 0.0 <= m[label] <= 1.0))

    checks.append(("recall monotonic: R@10 >= R@5 >= R@1",
                   m["recall@10"] >= m["recall@5"] >= m["recall@1"]))
    checks.append(("recall@10 > 0 (retriever actually finds gold)",
                   m["recall@10"] > 0.0))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
