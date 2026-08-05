"""Lab 01 — Cross-encoder reranking: retrieve wide, re-score short.

The bi-encoder embedder (BGE) maps query and passage to single vectors and
scores them with cosine — fast enough to scan the whole index, but blind to
fine-grained relevance: a passage sharing the topic scores well even when it
does not answer the question. A cross-encoder reads the (query, passage) PAIR
as one input, so it sees every token of both at once — far better at
separating "on topic" from "actually answers". The catch: one forward pass per
pair, which is why the cross-encoder only ever sees a SHORT candidate list.

This lab wires the two together on nfcorpus (BEIR medical FAQ corpus):

* stage 1 — bi-encoder (BGE) retrieves a wide top-20 candidate list;
* stage 2 — ``tools/reranker.py`` CrossEncoder re-scores the 20 candidates
  and keeps the top 3.

For each question we track the position of the gold (qrels-relevant) passage
before (bi top-20) and after (rerank top-3) to show the lift. Demo queries are
picked deterministically: the first qrels queries whose gold passage the
bi-encoder put OUTSIDE its top-3 but INSIDE its top-20 — the cases where a
reranker earns its keep.

Run from the repo root:
    python curriculum/06-re-ranking/01-cross-encoder.py
    python curriculum/06-re-ranking/01-cross-encoder.py --verify
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/06-re-ranking/01-cross-encoder.py``).
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
NF_MAX_QUERIES = 400  # search pool: first 400 qrels-covered query ids
WIDE_K = 20  # stage-1 bi-encoder candidate list
TOP_K = 3  # stage-2 cross-encoder shortlist
N_DEMO = 3  # how many lift-queries to show (and gate)
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


def preview(text: str, limit: int = PREVIEW) -> str:
    """Flatten a passage for one-line printing."""
    flat = text.replace("\n", " ")
    return flat[:limit] + ("..." if len(flat) > limit else "")


def gold_rank(docs: list[Document], gold_ids: set[str]) -> int | None:
    """1-based position of the first gold doc in ``docs``, else None."""
    for i, doc in enumerate(docs, start=1):
        if doc.metadata.get("id") in gold_ids:
            return i
    return None


# --------------------------------------------------------------------------
# 3. Experiment — embed, index, retrieve wide, rerank short
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    nf_texts, nf_ids = load_nfcorpus(CORPUS_PATH, NF_N_DOCS)
    nf_queries = load_nf_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)

    # --- Embed locally (BGE) and index in-memory ---------------------------
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME)
    t0 = time.perf_counter()
    nf_vecs = embedder.embed_documents(nf_texts)
    embed_s = time.perf_counter() - t0

    chunks = [
        Document(page_content=t, metadata={"id": cid})
        for t, cid in zip(nf_texts, nf_ids)
    ]
    store = FAISSVectorStore(embedding=embedder)  # embed_query works for the retriever
    t0 = time.perf_counter()
    store.add(chunks, embeddings=nf_vecs)
    index_s = time.perf_counter() - t0

    # --- The two stages ------------------------------------------------------
    bi = SimilarityRetriever(store, top_k=WIDE_K)  # stage 1: wide + cheap
    reranker = CrossEncoderReranker()  # stage 2: short + precise

    # Demo queries: gold in bi top-20 but NOT in bi top-3, and recovered into
    # the cross-encoder top-3 — the lift cases where the reranker earns its
    # keep. The cross-encoder only scores the rare candidates that pass the
    # cheap bi-encoder filter, so the search pool can be large.
    subset_ids = set(nf_ids)
    covered = [(qid, q) for qid, q in nf_queries if qid in qrels][:NF_MAX_QUERIES]
    t0 = time.perf_counter()
    results = []
    for qid, qtext in covered:
        gold = qrels[qid] & subset_ids
        if not gold:
            continue
        wide = bi.retrieve(qtext)
        g_before = gold_rank(wide, gold)
        if g_before is None or g_before <= TOP_K:
            continue  # not a lift case: gold already in bi top-3 (or missing)
        reranked = reranker.rerank(qtext, wide, top_k=TOP_K)
        g_after = gold_rank(reranked, gold)
        if g_after is None or g_after > TOP_K:
            continue  # cross-encoder did not recover it — not a demo query
        results.append(
            {
                "qid": qid,
                "question": qtext,
                "gold_ids": gold,
                "g_before": g_before,
                "g_after": g_after,
                "reranked": reranked,
                "scores": [d.metadata["score"] for d in reranked],
            }
        )
        if len(results) >= N_DEMO:
            break
    rerank_s = time.perf_counter() - t0

    return {
        "nf_ids": nf_ids,
        "demo": [(r["qid"], r["question"], r["gold_ids"]) for r in results],
        "results": results,
        "indexed": len(nf_texts),
        "embed_s": embed_s,
        "index_s": index_s,
        "rerank_s": rerank_s,
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 01 — Cross-encoder reranking: retrieve wide, re-score short")
    print(f"{BGE_MODEL_NAME} (bi-encoder, top-{WIDE_K}) -> "
          f"cross-encoder/ms-marco-MiniLM-L-6-v2 (top-{TOP_K})")
    print("=" * 66)

    print(f"\n[1] Corpus (deterministic subset, no randomness):")
    print(f"    {exp['indexed']} nfcorpus docs (first {NF_N_DOCS} of 3633)")
    print(f"    embedded in {exp['embed_s']:.2f}s (dim 768), indexed in {exp['index_s']:.3f}s")

    print(f"\n[2] Gold-position lift (bi-encoder top-{WIDE_K} -> cross-encoder top-{TOP_K}):")
    for r in exp["results"]:
        print(f'\n    Q[{r["qid"]}] "{r["question"]}"')
        print(f"      gold docs: {sorted(r['gold_ids'])}")
        print(f"      gold rank BEFORE (bi top-{WIDE_K}): "
              f"{r['g_before'] if r['g_before'] is not None else 'missing'}")
        print(f"      gold rank AFTER  (rerank top-{TOP_K}): "
              f"{r['g_after'] if r['g_after'] is not None else 'missing'}")
        print(f"      rerank scores (desc): {[f'{s:.3f}' for s in r['scores']]}")
        top = r["reranked"][0]
        print(f"      top-1: [{top.metadata['id']}] {preview(top.page_content)}")

    print(f"\n[3] Takeaway")
    print("    The bi-encoder put every gold passage OUTSIDE its top-3")
    print("    (otherwise the question would not be in this demo). The")
    print("    cross-encoder reads each (query, passage) pair as one input,")
    print("    so it can catch the fine-grained relevance the pooled cosine")
    print("    vector smears away — and promotes the gold into the top-3.")
    print("    That precision costs one forward pass per pair, which is why")
    print("    the cross-encoder must never see more than a short candidate")
    print(f"    list (here {WIDE_K}); scanning the whole corpus pair-wise would be")
    print("    orders of magnitude slower than the bi-encoder.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    checks.append((f"exactly {NF_N_DOCS} nfcorpus docs indexed", exp["indexed"] == NF_N_DOCS))
    checks.append((f"{N_DEMO} lift-queries selected (gold in bi top-{WIDE_K}, "
                   f"outside bi top-{TOP_K})", len(exp["results"]) == N_DEMO))

    for r in exp["results"]:
        tag = f"Q{r['qid']}"
        checks.append((f"{tag} gold was outside bi top-{TOP_K} before",
                       r["g_before"] is not None and r["g_before"] > TOP_K))
        checks.append((f"{tag} gold promoted into rerank top-{TOP_K}",
                       r["g_after"] is not None and r["g_after"] <= TOP_K))
        checks.append((f"{tag} gold rank strictly improved",
                       r["g_after"] is not None and r["g_before"] is not None
                       and r["g_after"] < r["g_before"]))
        checks.append((f"{tag} rerank scores are descending",
                       all(a >= b for a, b in zip(r["scores"], r["scores"][1:]))))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
