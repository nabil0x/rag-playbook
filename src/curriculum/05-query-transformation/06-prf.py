"""Lab 06 — Pseudo-relevance feedback: expand the query with retrieved terms.

Every previous lab in this track pays for query transformation with an LLM
call. Pseudo-relevance feedback (PRF) is the LLM-free variant: it treats the
retriever's own first pass as feedback. Stage 1 retrieves with the raw query,
harvests the top terms from the top-k documents it found (minus stopwords and
terms already in the query), appends them to the query, and retrieves again.
The expansion terms pull stage 2 toward the vocabulary of the documents the
first stage already judged relevant — no LLM, no training, no index rebuild.

This lab wraps the ``SimilarityRetriever`` in the ``PRFRetriever``
(``src/tools/prf.py``) and compares, for the same questions:

* RAW — plain top-k: embed the question as the user typed it.
* PRF — two-stage: raw query -> harvest terms -> expanded query -> top-k.

Same three questions as labs 01/02/04/05 (1606/1610/1626) so you can compare
the transformations directly. No LLM anywhere — embeddings are local BGE and
every other step is pure Python (``re`` + ``Counter``).

Run from the repo root:
    python src/curriculum/05-query-transformation/06-prf.py
    python src/curriculum/05-query-transformation/06-prf.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/05-query-transformation/06-prf.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from tools.prf import PRFRetriever  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PASSAGES = 100  # deterministic head of the 3200-passage corpus (keeps runtime low)
QUESTION_IDS = [1606, 1610, 1626]  # same questions as labs 01/02/04/05, for comparison
TOP_K = 3  # stage-2 retrieval depth
FEEDBACK_K = 3  # how many stage-1 documents the expansion terms are harvested from
N_TERMS = 5  # how many terms the expanded query gains
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
PREVIEW = 62  # max characters of passage text shown next to each hit


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


# --------------------------------------------------------------------------
# 3. Experiment — raw vs PRF retrieval for the same questions
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    passage_texts, passage_ids = load_passages(PASSAGES_PATH, N_PASSAGES)
    questions = load_questions(TEST_PATH, QUESTION_IDS)

    # --- Embed locally (BGE) and index in-memory ---------------------------
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME)
    t0 = time.perf_counter()
    passage_vecs = embedder.embed_documents(passage_texts)
    embed_s = time.perf_counter() - t0

    chunks = [
        Document(page_content=t, metadata={"id": pid})
        for t, pid in zip(passage_texts, passage_ids)
    ]
    store = FAISSVectorStore(embedding=embedder)
    t0 = time.perf_counter()
    store.add(chunks, embeddings=passage_vecs)
    index_s = time.perf_counter() - t0

    # --- The two retrievers over the SAME store -----------------------------
    raw_retriever = SimilarityRetriever(store, top_k=TOP_K)
    prf_retriever = PRFRetriever(
        raw_retriever, top_k=TOP_K, feedback_k=FEEDBACK_K, n_terms=N_TERMS
    )

    # --- Per question: raw retrieval + the PRF expansion + stage-2 retrieval -
    results = []
    for qid, qtext in questions:
        raw_docs = raw_retriever.retrieve(qtext)

        # Replay PRF's two stages so the demo can show what changed.
        feedback = raw_retriever.retrieve(qtext)[:FEEDBACK_K]
        terms = prf_retriever._feedback_terms(qtext, feedback, N_TERMS)
        expanded = " ".join([qtext, *terms]) if terms else qtext

        t0 = time.perf_counter()
        prf_docs = prf_retriever.retrieve(qtext)
        prf_s = time.perf_counter() - t0

        results.append(
            {
                "qid": qid,
                "question": qtext,
                "terms": terms,
                "expanded": expanded,
                "prf_s": prf_s,
                "raw_docs": raw_docs,
                "prf_docs": prf_docs,
            }
        )

    return {
        "passage_texts": passage_texts,
        "passage_ids": passage_ids,
        "questions": questions,
        "indexed": len(passage_texts),
        "embed_s": embed_s,
        "index_s": index_s,
        "results": results,
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 06 — Pseudo-relevance feedback: expand the query with retrieved terms")
    print(f"{BGE_MODEL_NAME} (local) -> FAISS top-{TOP_K} -> PRF (no LLM)")
    print("=" * 66)

    print(f"\n[1] Corpus (deterministic subset, no randomness):")
    print(f"    {exp['indexed']} passages (first {N_PASSAGES} of 3200, ids {exp['passage_ids'][0]}..{exp['passage_ids'][-1]})")
    print(f"    embedded in {exp['embed_s']:.2f}s (dim 768), indexed in {exp['index_s']:.3f}s")

    print(f"\n[2] Raw vs PRF (per question):")
    for r in exp["results"]:
        print(f'\n    Q[{r["qid"]}] "{r["question"]}"')
        print(f"      feedback terms ({N_TERMS} max): {r['terms']}")
        print(f"      expanded query: {r['expanded']!r}")
        print(f"      raw  top-1: {preview(r['raw_docs'][0].page_content)}")
        print(f"      prf  top-1: {preview(r['prf_docs'][0].page_content)}")

    print("\n[3] Takeaway")
    print("    PRF is query expansion without an LLM: stage 1 retrieves with")
    print("    the raw query, stage 2 with the query plus the top terms of")
    print("    the stage-1 documents. The expansion terms move the second")
    print("    query toward the vocabulary of what the first pass already")
    print("    judged relevant — cheap, deterministic, and free of API calls.")
    print("    Trade-off vs the LLM-based labs: the terms come from retrieved")
    print("    text, so PRF amplifies whatever the first pass found, good or")
    print("    bad — garbage-in-garbage-out is stronger here.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    # Structural properties (fully deterministic — no LLM anywhere).
    checks.append((f"exactly {N_PASSAGES} passages indexed", exp["indexed"] == N_PASSAGES))
    checks.append(("each question returns TOP_K raw hits",
                   all(len(r["raw_docs"]) == TOP_K for r in exp["results"])))
    checks.append(("each question returns TOP_K PRF hits",
                   all(len(r["prf_docs"]) == TOP_K for r in exp["results"])))

    # The expansion must actually add terms, and they must be new information
    # (not stopwords, not the question's own tokens).
    for r in exp["results"]:
        tag = f"Q{r['qid']}"
        checks.append((f"{tag} harvested >= 1 feedback term", len(r["terms"]) >= 1))
        checks.append((f"{tag} expansion terms are not in the raw question",
                       not any(t in r["question"].lower() for t in r["terms"])))

    # Content checks: the expanded query must still surface the answer's
    # keyword. Q1606 -> Montevideo; Q1610 -> the Spanish; Q1626 -> 1930.
    for r in exp["results"]:
        tag = f"Q{r['qid']}"
        joined = " ".join(d.page_content for d in r["prf_docs"]).lower()
        if r["qid"] == 1606:
            kw = "montevideo"
        elif r["qid"] == 1610:
            kw = "spanish"
        else:  # 1626
            kw = "1930"
        checks.append((f"{tag} PRF top-{TOP_K} retains '{kw}'", kw in joined))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
