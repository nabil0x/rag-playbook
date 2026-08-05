"""Lab 05 — Late interaction: MaxSim vs pooled cosine on the same tokens.

Lab 04 measured rerankers as *systems* (whole pipeline, different models).
This lab isolates the interaction function itself. Both scorers below consume
the EXACT SAME per-token BGE embeddings — one embedding pass per (query, doc)
— and differ only in how the tokens are combined:

* Pooled cosine — mean-pool the query tokens into one vector, mean-pool the
  document tokens into one vector, take the cosine. One number for the whole
  pair; fine-grained term overlap is smeared away. This is what a bi-encoder
  does.
* Late interaction (MaxSim) — for EVERY query token, find its best-matching
  document token and sum those maxima. Query tokens land only where they
  actually match, so "panther" in the query can score against "panther" in a
  long document even when mean-pooling would wash it out.

Both implementations are written inline in numpy (10 lines each) so the
mechanism is visible; the component-library ``ColBERTReranker`` from
``src/retrieval/rerank_advanced.py`` is then run on the same pool as a cross-check
— it should agree with the hand-rolled MaxSim to within float noise, which
verifies the teaching implementation against the shipped one.

The gate asserts: (1) MaxSim >= pooled cosine, (2) MaxSim >= bi-encoder
baseline, (3) hand-rolled MaxSim agrees with the component ColBERTReranker.

Run from the repo root:
    python src/curriculum/06-re-ranking/05-late-interaction.py
    python src/curriculum/06-re-ranking/05-late-interaction.py --verify
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/06-re-ranking/05-late-interaction.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from retrieval.rerank_advanced import ColBERTReranker  # noqa: E402
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — same pool as labs 02 and 04
# --------------------------------------------------------------------------
NFCORPUS_DIR = Path("Data/corpus/beir-nfcorpus/nfcorpus")
CORPUS_PATH = NFCORPUS_DIR / "corpus.jsonl"
QUERIES_PATH = NFCORPUS_DIR / "queries.jsonl"
QRELS_PATH = NFCORPUS_DIR / "qrels" / "test.tsv"
NF_N_DOCS = 600  # deterministic head of the 3633-doc nfcorpus corpus
NF_MAX_QUERIES = 40  # pool: first 40 qrels-covered queries with gold inside
WIDE_K = 20  # stage-1 bi-encoder candidate list both scorers re-rank
EVAL_K = 5  # ranking depth every system is scored at
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"


# --------------------------------------------------------------------------
# 2. Load — nfcorpus corpus + queries + qrels (same helpers as lab 04)
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
# 3. Metric — nDCG@k (same implementation as labs 02 and 04)
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
# 4. Interaction functions — same tokens in, one number out
# --------------------------------------------------------------------------
def pooled_cosine_score(q_tokens: list[list[float]], d_tokens: list[list[float]]) -> float:
    """Mean-pool both sides to single vectors, then cosine similarity."""
    if not q_tokens or not d_tokens:
        return 0.0
    q = np.mean(np.asarray(q_tokens, dtype=np.float32), axis=0)
    d = np.mean(np.asarray(d_tokens, dtype=np.float32), axis=0)
    q /= np.linalg.norm(q)
    d /= np.linalg.norm(d)
    return float(q @ d)


def maxsim_score(q_tokens: list[list[float]], d_tokens: list[list[float]]) -> float:
    """Late interaction: sum over query tokens of their best doc-token cosine.

    Same normalization and arithmetic as
    ``ColBERTReranker._maxsim_score`` in ``src/retrieval/rerank_advanced.py`` —
    this copy exists so the mechanism is visible in the lab.
    """
    if not q_tokens or not d_tokens:
        return 0.0
    q = np.asarray(q_tokens, dtype=np.float32)
    d = np.asarray(d_tokens, dtype=np.float32)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    sims = q @ d.T  # (n_query_tokens, n_doc_tokens) cosine matrix
    return float(sims.max(axis=1).sum())


# --------------------------------------------------------------------------
# 5. Experiment — one embedding pass per pair, both scorers, plus cross-check
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

    subset_ids = set(nf_ids)
    covered = [
        (qid, q) for qid, q in nf_queries
        if qid in qrels and qrels[qid] & subset_ids
    ][:NF_MAX_QUERIES]

    # Token-level BGE embedder — the single source of tokens for BOTH scorers.
    token_model = ColBERTReranker(model_name=BGE_MODEL_NAME)._get_model()

    def token_embeddings(text: str) -> list[list[float]]:
        emb = token_model.encode(text, output_value="token_embeddings")
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    baseline_scores: list[float] = []
    pooled_scores: list[float] = []
    maxsim_scores: list[float] = []
    winner_counts = {"maxsim": 0, "pooled": 0, "tie": 0}
    example: dict | None = None

    t0 = time.perf_counter()
    for qid, qtext in covered:
        gold = qrels[qid] & subset_ids
        wide = bi_wide.retrieve(qtext)
        q_tokens = token_embeddings(qtext)

        pooled_ranked = []
        maxsim_ranked = []
        for doc in wide:
            d_tokens = token_embeddings(doc.page_content)
            p_score = pooled_cosine_score(q_tokens, d_tokens)
            m_score = maxsim_score(q_tokens, d_tokens)
            doc.metadata["pooled_score"] = p_score
            doc.metadata["maxsim_score"] = m_score
            pooled_ranked.append((p_score, doc))
            maxsim_ranked.append((m_score, doc))

        pooled_ranked.sort(key=lambda pair: pair[0], reverse=True)
        maxsim_ranked.sort(key=lambda pair: pair[0], reverse=True)
        pooled_ids = [d.metadata["id"] for _, d in pooled_ranked[:EVAL_K]]
        maxsim_ids = [d.metadata["id"] for _, d in maxsim_ranked[:EVAL_K]]

        baseline_scores.append(
            ndcg_at_k([d.metadata["id"] for d in wide[:EVAL_K]], gold, EVAL_K)
        )
        pooled_scores.append(ndcg_at_k(pooled_ids, gold, EVAL_K))
        maxsim_scores.append(ndcg_at_k(maxsim_ids, gold, EVAL_K))

        if pooled_ids[0] != maxsim_ids[0]:
            winner_counts["maxsim" if maxsim_scores[-1] > pooled_scores[-1] else "pooled"] += 1
        else:
            winner_counts["tie"] += 1

        # First query where the top-1 differs: freeze it as the worked example.
        if example is None and pooled_ids[0] != maxsim_ids[0]:
            doc_by_id = {d.metadata["id"]: d for d in wide}
            example = {
                "qid": qid,
                "query": qtext,
                "pooled_top1": doc_by_id[pooled_ids[0]].page_content[:120],
                "maxsim_top1": doc_by_id[maxsim_ids[0]].page_content[:120],
                "best_doc_token": max(
                    pooled_ranked + maxsim_ranked, key=lambda p: p[0]
                )[1].page_content[:120],
                "query_token": qtext.split()[0],
                "maxsim_score": maxsim_scores[-1],
                "pooled_score": pooled_scores[-1],
            }

    local_s = time.perf_counter() - t0

    def mean(scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    result = {
        "rows": covered,
        "baseline": mean(baseline_scores),
        "pooled": mean(pooled_scores),
        "maxsim": mean(maxsim_scores),
        "winner_counts": winner_counts,
        "example": example,
        "indexed": len(nf_texts),
        "embed_s": embed_s,
        "index_s": index_s,
        "local_s": local_s,
    }

    # Cross-check: the component-library ColBERTReranker on the same pool
    # must agree with the hand-rolled MaxSim (identical arithmetic, same model).
    colbert = ColBERTReranker(model_name=BGE_MODEL_NAME)
    t0 = time.perf_counter()
    comp_scores = []
    for qid, qtext in covered:
        gold = qrels[qid] & subset_ids
        wide = bi_wide.retrieve(qtext)
        ranked = colbert.rerank(qtext, wide, top_k=EVAL_K)
        comp_scores.append(
            ndcg_at_k([d.metadata["id"] for d in ranked], gold, EVAL_K)
        )
    result["colbert_component"] = mean(comp_scores)
    result["crosscheck_s"] = time.perf_counter() - t0
    return result


# --------------------------------------------------------------------------
# 6. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 05 — Late interaction: MaxSim vs pooled cosine, same tokens")
    print(f"nfcorpus {exp['indexed']} docs, pool {len(exp['rows'])} queries, "
          f"nDCG@{EVAL_K}")
    print("=" * 66)

    print(f"\n[1] Same token embeddings, only the interaction function differs:")
    print(f"    bi-encoder baseline (pooled, top-{EVAL_K}) : "
          f"{exp['baseline']:.4f}")
    print(f"    pooled cosine on top-{WIDE_K} (lab 05)      : "
          f"{exp['pooled']:.4f} ({exp['pooled'] - exp['baseline']:+.4f})")
    print(f"    MaxSim late interaction on top-{WIDE_K}     : "
          f"{exp['maxsim']:.4f} ({exp['maxsim'] - exp['baseline']:+.4f})")

    print(f"\n[2] Cross-check — component ColBERTReranker: "
          f"{exp['colbert_component']:.4f} "
          f"(hand-rolled MaxSim {exp['maxsim']:.4f}, "
          f"diff {abs(exp['colbert_component'] - exp['maxsim']):.6f})")

    wc = exp["winner_counts"]
    print(f"\n[3] Per-query winner (top-1 differs): "
          f"MaxSim {wc['maxsim']}, pooled {wc['pooled']}, tie {wc['tie']}")

    ex = exp["example"]
    if ex:
        print(f"\n[4] Worked example — query {ex['qid']}: \"{ex['query']}\"")
        print(f"    query token examined: \"{ex['query_token']}\"")
        print(f"    pooled top-1: \"{ex['pooled_top1']}...\"")
        print(f"    MaxSim top-1 : \"{ex['maxsim_top1']}...\"")
        print(f"    MaxSim nDCG@{EVAL_K} {ex['maxsim_score']:.4f} vs "
              f"pooled {ex['pooled_score']:.4f}")

    print(f"\n[5] Takeaway")
    print("    Pooling is lossy: mean-pooling a passage's tokens turns a")
    print("    document with one precise match into the average of everything")
    print("    around it. MaxSim keeps each query token's best hit, so a")
    print("    single exact term match anywhere in the passage still scores.")
    print("    Same model, same tokens, same candidates — the interaction")
    print("    function alone explains the gap. The component library's")
    print("    ColBERTReranker reproduces the hand-rolled MaxSim, so this")
    print("    is the mechanism the production class ships.")


# --------------------------------------------------------------------------
# 7. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    checks.append((f"exactly {NF_N_DOCS} nfcorpus docs indexed",
                   exp["indexed"] == NF_N_DOCS))
    checks.append((f"pool has {len(exp['rows'])} queries (>= 40)",
                   len(exp["rows"]) >= 40))

    checks.append(("baseline nDCG in [0, 1]",
                   0.0 <= exp["baseline"] <= 1.0))
    checks.append(("pooled nDCG in [0, 1]",
                   0.0 <= exp["pooled"] <= 1.0))
    checks.append(("MaxSim nDCG in [0, 1]",
                   0.0 <= exp["maxsim"] <= 1.0))

    checks.append(("MaxSim >= bi-encoder baseline",
                   exp["maxsim"] >= exp["baseline"]))
    checks.append(("MaxSim >= pooled cosine (same tokens)",
                   exp["maxsim"] >= exp["pooled"]))

    checks.append(("component ColBERTReranker agrees with hand-rolled MaxSim "
                   "(|diff| < 0.005)",
                   abs(exp["colbert_component"] - exp["maxsim"]) < 0.005))

    checks.append(("winner counts sum to pool size",
                   sum(exp["winner_counts"].values()) == len(exp["rows"])))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
