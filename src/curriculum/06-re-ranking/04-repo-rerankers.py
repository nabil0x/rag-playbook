"""Lab 04 — The repo's reranker arsenal, wired end-to-end.

Labs 01-03 used ``tools/reranker.py`` (CrossEncoderReranker). The component
library ships more: ``retrieval/rerank.py`` wraps the same engine in a
composable ``RerankRetriever`` (retrieve wide, rerank short, one call), and
``retrieval/rerank_advanced.py`` adds three more families — ColBERT's
token-level MaxSim late interaction, a pointwise MonoT5, and LLM pointwise
scoring. This lab wires them all onto the same nfcorpus pool as lab 02 and
measures each on nDCG@5 against the bi-encoder baseline:

* RerankRetriever — bi top-20, cross-encoder top-5. The production shape.
* ColBERTReranker — bi top-20, MaxSim over per-token BGE embeddings, top-5.
  No extra model download: it reuses the same BGE embedder as retrieval.
* MonoT5 — pointwise seq2seq "is this relevant?" — OPTIONAL, only runs if
  the ~3GB checkpoint is already in the local HF cache (skipped otherwise).
* LLMPointwise — "yes/no" per candidate via Groq — OPTIONAL, only runs with
  ``--with-llm`` (uses ``GROQ_API_KEY``; O(n) calls, keep the pool tiny).

The gate covers only the two local systems; the optional sections print a
SKIP notice when their dependency is missing.

Run from the repo root:
    python curriculum/06-re-ranking/04-repo-rerankers.py
    python curriculum/06-re-ranking/04-repo-rerankers.py --verify
    python curriculum/06-re-ranking/04-repo-rerankers.py --with-llm
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/06-re-ranking/04-repo-rerankers.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from retrieval.rerank import CrossEncoderReranker as RepoCrossEncoder  # noqa: E402
from retrieval.rerank import RerankRetriever  # noqa: E402
from retrieval.rerank_advanced import (  # noqa: E402
    ColBERTReranker,
    LLMPointwiseReranker,
    MonoT5Reranker,
)
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
NFCORPUS_DIR = Path("Data/corpus/beir-nfcorpus/nfcorpus")
CORPUS_PATH = NFCORPUS_DIR / "corpus.jsonl"
QUERIES_PATH = NFCORPUS_DIR / "queries.jsonl"
QRELS_PATH = NFCORPUS_DIR / "qrels" / "test.tsv"
NF_N_DOCS = 600  # deterministic head of the 3633-doc nfcorpus corpus
NF_MAX_QUERIES = 40  # pool: first 40 qrels-covered queries with gold inside
WIDE_K = 20  # stage-1 bi-encoder candidate list
EVAL_K = 5  # ranking depth every system is scored at
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_POOL = 3  # LLM pointwise demo: first 3 pool queries, top-10 candidates
MONT5_CACHE = (
    Path.home() / ".cache/huggingface/hub/models--castorini--monot5-base-msmarco"
)


# --------------------------------------------------------------------------
# 2. Load — nfcorpus corpus + queries + qrels
# --------------------------------------------------------------------------
def load_nfcorpus(path: Path, n: int) -> tuple[list[str], list[str]]:
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
                continue
            qid, cid = parts[0], parts[1]
            if int(parts[2]) >= 1:
                qrels.setdefault(qid, set()).add(cid)
    return qrels


# --------------------------------------------------------------------------
# 3. Metric — nDCG@k (same implementation as lab 02)
# --------------------------------------------------------------------------
def ndcg_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    """nDCG@k in [0, 1] with binary relevance."""
    gain = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold:
            gain += 1.0 / math.log2(i + 1)
    ideal = 0.0
    for i in range(1, min(k, len(gold)) + 1):
        ideal += 1.0 / math.log2(i + 1)
    return gain / ideal if ideal > 0.0 else 0.0


# --------------------------------------------------------------------------
# 4. Experiment — index, then score each reranker on the same pool
# --------------------------------------------------------------------------
def run_experiment(with_llm: bool) -> dict:
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
    rerank_retriever = RerankRetriever(
        bi_wide, RepoCrossEncoder(), top_k=EVAL_K, k_retrieve=WIDE_K
    )
    colbert = ColBERTReranker(model_name=BGE_MODEL_NAME)

    subset_ids = set(nf_ids)
    covered = [
        (qid, q) for qid, q in nf_queries
        if qid in qrels and qrels[qid] & subset_ids
    ][:NF_MAX_QUERIES]

    t0 = time.perf_counter()
    baseline_scores, rr_scores, colbert_scores = [], [], []
    for qid, qtext in covered:
        gold = qrels[qid] & subset_ids
        wide = bi_wide.retrieve(qtext)
        baseline_scores.append(
            ndcg_at_k([d.metadata["id"] for d in wide[:EVAL_K]], gold, EVAL_K)
        )
        rr_scores.append(
            ndcg_at_k(
                [d.metadata["id"] for d in rerank_retriever.retrieve(qtext)],
                gold, EVAL_K,
            )
        )
        colbert_scores.append(
            ndcg_at_k(
                [d.metadata["id"] for d in colbert.rerank(qtext, wide, top_k=EVAL_K)],
                gold, EVAL_K,
            )
        )
    local_s = time.perf_counter() - t0

    def mean(scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    result = {
        "rows": covered,
        "baseline": mean(baseline_scores),
        "rerank_retriever": mean(rr_scores),
        "colbert": mean(colbert_scores),
        "indexed": len(nf_texts),
        "embed_s": embed_s,
        "index_s": index_s,
        "local_s": local_s,
        "optional": {},
    }

    # --- MonoT5: only if the ~3GB checkpoint is already cached locally -----
    if MONT5_CACHE.exists():
        t5 = MonoT5Reranker()
        t5_scores = []
        for qid, qtext in covered:
            gold = qrels[qid] & subset_ids
            wide = bi_wide.retrieve(qtext)
            t5_scores.append(
                ndcg_at_k(
                    [d.metadata["id"] for d in t5.rerank(qtext, wide, top_k=EVAL_K)],
                    gold, EVAL_K,
                )
            )
        result["optional"]["monot5"] = {"status": "run", "ndcg": mean(t5_scores)}
    else:
        result["optional"]["monot5"] = {
            "status": "skip",
            "ndcg": None,
            "note": "checkpoint not in HF cache (~3GB download; run it yourself)",
        }

    # --- LLM pointwise: only with --with-llm and a GROQ key -----------------
    if with_llm:
        from dotenv import load_dotenv  # import on demand, like llms/groq.py

        load_dotenv()
    if with_llm and os.getenv("GROQ_API_KEY"):
        from llms.groq import GroqLLM  # imported on demand, not at module level

        llm_rr = LLMPointwiseReranker(GroqLLM(temperature=0.0))
        hits = 0
        shown = []
        for qid, qtext in covered[:LLM_POOL]:
            gold = qrels[qid] & subset_ids
            wide = bi_wide.retrieve(qtext)
            kept = llm_rr.rerank(qtext, wide, top_k=10)
            kept_ids = [d.metadata["id"] for d in kept]
            hit = bool(set(kept_ids) & gold)
            hits += 1 if hit else 0
            shown.append({"qid": qid, "kept_gold": hit})
        result["optional"]["llm"] = {
            "status": "run",
            "queries": LLM_POOL,
            "kept_gold": hits,
            "rows": shown,
        }
    else:
        result["optional"]["llm"] = {
            "status": "skip",
            "queries": LLM_POOL,
            "note": "pass --with-llm with GROQ_API_KEY set to run the LLM judge",
        }
    return result


# --------------------------------------------------------------------------
# 5. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 04 — The repo's reranker arsenal, wired end-to-end")
    print(f"nfcorpus {exp['indexed']} docs, pool {len(exp['rows'])} queries, "
          f"nDCG@{EVAL_K}")
    print("=" * 66)

    print(f"\n[1] Local systems (mean nDCG@{EVAL_K} over {len(exp['rows'])} queries):")
    print(f"    baseline (bi top-{EVAL_K})              : {exp['baseline']:.4f}")
    print(f"    RerankRetriever (bi top-{WIDE_K} -> CE top-{EVAL_K})  : "
          f"{exp['rerank_retriever']:.4f} "
          f"({exp['rerank_retriever'] - exp['baseline']:+.4f})")
    print(f"    ColBERT MaxSim (BGE tokens, top-{WIDE_K} -> top-{EVAL_K}) : "
          f"{exp['colbert']:.4f} ({exp['colbert'] - exp['baseline']:+.4f})")

    print(f"\n[2] Optional systems:")
    t5 = exp["optional"]["monot5"]
    print(f"    MonoT5 (pointwise seq2seq): "
          f"{'ran' if t5['status'] == 'run' else 'SKIP — ' + t5['note']}"
          + (f", nDCG@5 {t5['ndcg']:.4f}" if t5["status"] == "run" else ""))
    llm = exp["optional"]["llm"]
    if llm["status"] == "run":
        print(f"    LLM pointwise (Groq, {llm['queries']} queries x 10 candidates):")
        for row in llm["rows"]:
            print(f"      {row['qid']}: gold {'kept' if row['kept_gold'] else 'dropped'}")
    else:
        print(f"    LLM pointwise (Groq): SKIP — {llm['note']}")

    print(f"\n[3] Takeaway")
    print("    Every reranker lifts the baseline on the same pool, because")
    print("    each re-scores the 20 candidates with a model that sees the")
    print("    query: the cross-encoder reads (query, passage) pairs, and")
    print("    ColBERT matches query tokens against passage tokens (MaxSim)")
    print("    — catching the fine-grained overlap a pooled cosine smears.")
    print("    ColBERT reuses the SAME BGE embedder as retrieval, so the")
    print("    token-level precision costs no extra model download.")
    print("    The trade-off is compute: all of them re-score every")
    print("    candidate, which is why they only ever see a short list.")


# --------------------------------------------------------------------------
# 6. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    checks.append((f"exactly {NF_N_DOCS} nfcorpus docs indexed",
                   exp["indexed"] == NF_N_DOCS))
    checks.append((f"pool has {len(exp['rows'])} queries (>= 40)",
                   len(exp["rows"]) >= 40))

    checks.append(("baseline nDCG in [0, 1]",
                   0.0 <= exp["baseline"] <= 1.0))
    checks.append(("RerankRetriever nDCG in [0, 1]",
                   0.0 <= exp["rerank_retriever"] <= 1.0))
    checks.append(("ColBERT nDCG in [0, 1]",
                   0.0 <= exp["colbert"] <= 1.0))

    checks.append(("RerankRetriever >= baseline",
                   exp["rerank_retriever"] >= exp["baseline"]))
    checks.append(("ColBERT >= baseline",
                   exp["colbert"] >= exp["baseline"]))

    checks.append(("MonoT5 section reports run or skip",
                   exp["optional"]["monot5"]["status"] in ("run", "skip")))
    checks.append(("LLM section reports run or skip",
                   exp["optional"]["llm"]["status"] in ("run", "skip")))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    with_llm = "--with-llm" in sys.argv
    exp = run_experiment(with_llm)
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
