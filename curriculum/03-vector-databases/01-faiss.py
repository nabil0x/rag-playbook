"""Lab 01 — FAISS: the in-memory vector index.

A vector database stores embeddings and answers one question: "which stored
vectors are closest to this query vector?" FAISS (Facebook AI Similarity
Search) is the workhorse of the in-memory end of that spectrum: the index
lives entirely in RAM, builds in seconds, and searches exactly.

This lab builds a FAISS store over a deterministic subset of
``Data/corpus/rag-mini-wikipedia`` and inspects the three things that decide
how a vector database behaves:

* INDEX TYPE — FAISS ships dozens; the default here is ``IndexFlatL2``, a
  brute-force exact index: the query vector is compared against every stored
  vector. Exact search is the gold standard that every approximate method
  (IVF, HNSW, ...) is measured against. No training, no parameters, just a
  matrix of vectors and a loop.
* SCORE CONVENTION — ``IndexFlatL2`` reports the SQUARED Euclidean distance
  between the query and each passage vector: LOWER is more similar (a perfect
  match scores 0.0). Cosine-based stores (Qdrant — lab 03) flip both the
  scale and the direction. For unit-norm vectors the two are linked by
  ``cos = 1 - sqL2/2``, which is exactly the number lab 03 will reproduce.
* PERSISTENCE MODEL — FAISS is in-memory: build the index, query it, and when
  the process exits the index is gone (there is no on-disk format until you
  write one). That tradeoff is the takeaway — compare with Chroma's persistent
  store in lab 02.

The lab also runs MMR (Maximum Marginal Relevance) once: FAISS's
``max_marginal_relevance_search`` re-ranks candidates to trade pure similarity
for diversity, so a query can surface passages about different aspects of the
topic instead of near-duplicates.

Run from the repo root:
    python curriculum/03-vector-databases/01-faiss.py
    python curriculum/03-vector-databases/01-faiss.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/03-vector-databases/01-faiss.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PASSAGES = 100  # deterministic head of the 3200-passage corpus (keeps runtime low)
QUESTION_IDS = [1606, 1610, 1604]  # real questions from test.parquet, answers inside the subset
TOP_K = 3
MMR_K = 5
LAMBDA_MULT = 0.5  # MMR: 1.0 = pure similarity, 0.0 = pure diversity
PREVIEW = 62  # max characters of passage text shown next to each hit
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DIM = 768
FLOAT_BYTES = 4  # float32


# --------------------------------------------------------------------------
# 2. Load — corpus + questions from the fresh rag-mini-wikipedia parquet files
# --------------------------------------------------------------------------
def load_passages(path: Path, n: int) -> tuple[list[str], list[int]]:
    """Return (passage_texts, passage_ids) for the first ``n`` passages."""
    df = pd.read_parquet(path)
    subset = df.head(n)
    return subset["passage"].tolist(), subset.index.tolist()


def load_questions(path: Path, ids: list[int]) -> list[tuple[int, str]]:
    """Return [(question_id, question_text)] for the requested test rows."""
    df = pd.read_parquet(path)
    rows = df.loc[ids]
    return [(int(idx), row["question"]) for idx, row in rows.iterrows()]


def preview(text: str, limit: int = PREVIEW) -> str:
    """Flatten a passage for one-line printing."""
    flat = text.replace("\n", " ")
    return flat[:limit] + ("..." if len(flat) > limit else "")


def passage_lookup(texts: list[str], ids: list[int]) -> dict[int, str]:
    """Map passage id -> text, for turning a hit id back into printable text."""
    return dict(zip(ids, texts))


# --------------------------------------------------------------------------
# 3. Experiment — embed, index, query; returns every artifact the demo and
#    the verification gate need (no re-computation between the two paths)
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    passage_texts, passage_ids = load_passages(PASSAGES_PATH, N_PASSAGES)
    questions = load_questions(TEST_PATH, QUESTION_IDS)

    # --- Embed the whole subset once (batched) + each question once ---------
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME)
    t0 = time.perf_counter()
    passage_vecs = embedder.embed_documents(passage_texts)
    embed_s = time.perf_counter() - t0

    question_texts = [qtext for _, qtext in questions]
    query_vecs = [embedder.embed_query(q) for q in question_texts]

    # --- Build the FAISS index (in-memory) ----------------------------------
    chunks = [
        Document(page_content=t, metadata={"id": pid})
        for t, pid in zip(passage_texts, passage_ids)
    ]
    store = FAISSVectorStore()
    t0 = time.perf_counter()
    store.add(chunks, embeddings=passage_vecs)
    index_s = time.perf_counter() - t0

    # --- Query each question with scores ------------------------------------
    scored = [
        store.query_with_scores(qvec, top_k=TOP_K) for qvec in query_vecs
    ]

    # --- MMR on the first question ------------------------------------------
    mmr_docs = store.query_mmr(query_vecs[0], top_k=MMR_K, lambda_mult=LAMBDA_MULT)

    return {
        "passage_texts": passage_texts,
        "passage_ids": passage_ids,
        "questions": questions,
        "query_vecs": query_vecs,
        "embed_s": embed_s,
        "index_s": index_s,
        "scored": scored,
        "mmr_docs": mmr_docs,
        "dim": len(passage_vecs[0]),
        "indexed": len(passage_vecs),
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    passage_lk = passage_lookup(exp["passage_texts"], exp["passage_ids"])
    n_bytes = exp["indexed"] * exp["dim"] * FLOAT_BYTES

    print("=" * 66)
    print("Lab 01 — FAISS: the in-memory vector index")
    print(f"{BGE_MODEL_NAME} | exact flat-L2 index | in-memory only")
    print("=" * 66)

    print(f"\n[1] Corpus (deterministic subset, no randomness):")
    print(f"    {exp['indexed']} passages (first {N_PASSAGES} of 3200, ids {exp['passage_ids'][0]}..{exp['passage_ids'][-1]})")
    print(f"    {len(exp['questions'])} questions from test.parquet:")
    for qid, qtext in exp["questions"]:
        print(f"      [{qid}] {qtext}")

    print(f"\n[2] Embed + index:")
    print(f"    embedded {exp['indexed']} passages in {exp['embed_s']:.2f}s (dim {exp['dim']})")
    print(f"    FAISS index built in {exp['index_s']:.3f}s")
    print(f"    in-memory size ~ {n_bytes / 1024:.0f} KB "
          f"({exp['indexed']} x {exp['dim']} x {FLOAT_BYTES}B float32)")

    print(f"\n[3] Top-{TOP_K} per question (score = squared L2 distance, LOWER = more similar):")
    for i, (qid, qtext) in enumerate(exp["questions"]):
        print(f'\n    Q[{qid}] "{qtext}"')
        for doc, score in exp["scored"][i]:
            pid = doc.metadata.get("id", "?")
            print(f"      {score:8.4f}  [passage {pid}] {preview(doc.page_content)}")
        if i == 0:
            print("      ^ note: 0.0 would be a perfect match; these distances grow")
            print("        as relevance drops")

    print(f"\n[4] MMR on Q[{exp['questions'][0][0]}] (lambda_mult={LAMBDA_MULT}, k={MMR_K}):")
    for rank, doc in enumerate(exp["mmr_docs"], 1):
        pid = doc.metadata.get("id", "?")
        print(f"      {rank}. [passage {pid}] {preview(doc.page_content)}")
    print("      MMR re-ranks the candidates: raise lambda_mult toward 1.0 for")
    print("      pure relevance, lower it toward 0.0 for pure diversity.")

    print("\n[5] Takeaway")
    print("    FAISS's default flat-L2 index is exact, in-RAM, and reports")
    print("    squared-L2 distances (lower = better). It is the fastest store")
    print("    to build and the honest baseline for lab 04's benchmark — but")
    print("    everything vanishes at process exit. Chroma (lab 02) trades a")
    print("    few milliseconds for an on-disk store; Qdrant (lab 03) trades")
    print("    them for a cosine score and a full query language.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    # Dimension and count match the model / subset.
    checks.append(("embedding dimension is 768 (BGE base)", exp["dim"] == BGE_DIM))
    checks.append((f"exactly {N_PASSAGES} passages indexed", exp["indexed"] == N_PASSAGES))

    # Every question returned exactly TOP_K scored hits.
    checks.append(
        ("each question returns TOP_K scored hits",
         all(len(hits) == TOP_K for hits in exp["scored"]))
    )

    # Squared-L2 scores ascend with rank (0.0 would be a perfect match).
    scores_ascending = all(
        [s for _, s in hits] == sorted(s for _, s in hits) for hits in exp["scored"]
    )
    checks.append(("squared-L2 scores ascend per query (lower = more similar)", scores_ascending))

    # Content check: Q1610 "Who founded Montevideo?" must rank the passage
    # that says the Spanish founded Montevideo at #1 (passage id 2 lives
    # inside the first N_PASSAGES).
    q1610_top = exp["scored"][1][0][0].page_content.lower()
    checks.append(("Q1610 top-1 names the Spanish founder of Montevideo", "spanish" in q1610_top))

    # Q1606 "Is Uruguay's capital Montevideo?" must rank an Uruguay passage
    # that mentions Montevideo at #1.
    q1606_top = exp["scored"][0][0][0].page_content.lower()
    checks.append(("Q1606 top-1 mentions Montevideo", "montevideo" in q1606_top))

    # MMR returns exactly MMR_K distinct documents (no duplicates).
    mmr_texts = [d.page_content for d in exp["mmr_docs"]]
    checks.append(("MMR returns MMR_K distinct documents", len(mmr_texts) == MMR_K and len(set(mmr_texts)) == MMR_K))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
