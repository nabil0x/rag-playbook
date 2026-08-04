"""Lab 03 — Qdrant: the in-memory cosine store.

Labs 01 and 02 covered the two ends of the persistence spectrum (pure RAM vs
a directory on disk). Qdrant sits in the middle: one API, two local modes —
``path=":memory:"`` keeps everything in RAM, ``path="<dir>"`` persists to
disk. This lab runs the in-memory mode and focuses on the second axis that
separates vector stores: **what the score means**.

FAISS (lab 01) and Chroma (lab 02) both default to the l2 space and report the
same raw SQUARED L2 distance — lower is more similar. Qdrant's default is
Cosine distance, so it reports a cosine SIMILARITY — higher is more similar.
Same embeddings, same corpus, same ranking, but the numbers are not
interchangeable.

Because every vector here is unit-norm (BGE normalizes), the two scores are
exactly linked:

    sqL2 = |a - b|^2 = 2 - 2 * cos(a, b)   =>   cos = 1 - sqL2 / 2

The lab exploits that identity as a cross-check: for each hit it prints the
FAISS squared-L2 score, the value ``1 - sqL2/2``, and the score Qdrant
actually reports. They agree to ~4 decimals — which proves both stores are
computing the same thing and only disagreeing about how to display it.

The other lesson is structural: ``:memory:`` writes nothing to disk and
forgets everything at process exit (run the lab twice — the second run builds
a brand-new index from scratch). Persistent Qdrant is literally a one-argument
change: ``path="qdrant_storage"`` instead of ``path=":memory:"``, reusing
lab 02's ``load()``-style reopen story.

Run from the repo root:
    python curriculum/03-vector-databases/03-qdrant-in-memory.py
    python curriculum/03-vector-databases/03-qdrant-in-memory.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/03-vector-databases/03-qdrant-in-memory.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402
from vectordb.qdrant import QdrantVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PASSAGES = 100  # deterministic head of the 3200-passage corpus
QUESTION_IDS = [1606, 1610]  # real questions; answers live inside the subset
TOP_K = 3
PREVIEW = 62
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DIM = 768
COSINE_TOL = 1e-3  # cos = 1 - sqL2/2 holds up to float32 rounding
COLLECTION = "lab03"


# --------------------------------------------------------------------------
# 2. Load — corpus + questions (same helpers as labs 01/02)
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


def expected_cosine(sq_l2: float) -> float:
    """cos(a,b) = 1 - |a-b|^2 / 2, exact for unit-norm vectors."""
    return 1.0 - sq_l2 / 2.0


# --------------------------------------------------------------------------
# 3. Experiment — embed once, index into FAISS (sq-L2) and Qdrant (cosine),
#    then cross-check the two score conventions on every hit
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    passage_texts, passage_ids = load_passages(PASSAGES_PATH, N_PASSAGES)
    questions = load_questions(TEST_PATH, QUESTION_IDS)
    chunks = [
        Document(page_content=t, metadata={"id": pid})
        for t, pid in zip(passage_texts, passage_ids)
    ]

    # --- Embed the subset once; BOTH stores index the same vectors ----------
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME)
    t0 = time.perf_counter()
    passage_vecs = embedder.embed_documents(passage_texts)
    embed_s = time.perf_counter() - t0
    query_vecs = [embedder.embed_query(q) for _, q in questions]

    # --- FAISS (sq-L2, lower = better) — the cross-check baseline -----------
    faiss_store = FAISSVectorStore()
    faiss_store.add(chunks, embeddings=passage_vecs)
    faiss_scored = [faiss_store.query_with_scores(q, top_k=TOP_K) for q in query_vecs]

    # --- Qdrant (cosine, higher = better) — in-memory, nothing on disk ------
    qdrant_store = QdrantVectorStore(
        collection_name=COLLECTION, path=":memory:"
    )
    t0 = time.perf_counter()
    qdrant_store.add(chunks, embeddings=passage_vecs)
    add_s = time.perf_counter() - t0
    qdrant_scored = [qdrant_store.query_with_scores(q, top_k=TOP_K) for q in query_vecs]

    return {
        "passage_texts": passage_texts,
        "passage_ids": passage_ids,
        "questions": questions,
        "embed_s": embed_s,
        "add_s": add_s,
        "faiss_scored": faiss_scored,
        "qdrant_scored": qdrant_scored,
        "dim": len(passage_vecs[0]),
        "indexed": len(passage_vecs),
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 03 — Qdrant: the in-memory cosine store")
    print(f"{BGE_MODEL_NAME} | cosine distance | path=:memory: (nothing on disk)")
    print("=" * 66)

    print(f"\n[1] Corpus + embedding:")
    print(f"    {exp['indexed']} passages, dim {exp['dim']}, embedded in {exp['embed_s']:.2f}s")
    print(f"    same vectors indexed into FAISS (sq-L2) AND Qdrant (cosine)")

    print(f"\n[2] Qdrant index build (in-memory):")
    print(f"    {exp['indexed']} passages added in {exp['add_s']:.3f}s")
    print("    no persist_dir: path=\":memory:\" writes zero bytes and forgets")
    print("    everything at process exit (persistent mode = path='<dir>' instead)")

    print(f"\n[3] Top-{TOP_K} per question — sq-L2 (FAISS) vs cosine (Qdrant):")
    print("    col '1 - sqL2/2' is the cosine value FAISS's score implies;")
    print("    col 'qdrant' is what Qdrant actually reports. They should match.")
    for i, (qid, qtext) in enumerate(exp["questions"]):
        print(f'\n    Q[{qid}] "{qtext}"')
        for (fdoc, fscore), (cdoc, cscore) in zip(
            exp["faiss_scored"][i], exp["qdrant_scored"][i]
        ):
            conv = expected_cosine(fscore)
            match = "SAME" if fdoc.metadata["id"] == cdoc.metadata["id"] else "DIFF"
            print(f"      faiss {fscore:8.4f} | 1-sqL2/2 {conv:8.4f} | qdrant {cscore:8.4f} "
                  f"| [passage {cdoc.metadata['id']}] {preview(cdoc.page_content)}  {match}")

    print("\n[4] Takeaway")
    print("    The ranking is identical to labs 01/02 — the store never changes")
    print("    WHAT ranks first, only the score scale and direction. For unit-")
    print("    norm vectors cos = 1 - sqL2/2, so '0.4255 sq-L2' and '0.7873")
    print("    cosine' are the same retrieval. Never compare scores across")
    print("    stores (or across embedding models); compare rankings. And:")
    print("    :memory: is a scratchpad — the same one-line API turns it into")
    print("    a persistent store when you pass a directory path.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    checks.append(("embedding dimension is 768 (BGE base)", exp["dim"] == BGE_DIM))
    checks.append((f"exactly {N_PASSAGES} passages indexed", exp["indexed"] == N_PASSAGES))

    # Same embeddings => same ranking, whatever the store.
    all_same_rank = True
    for fhits, qhits in zip(exp["faiss_scored"], exp["qdrant_scored"]):
        for (fdoc, _), (qdoc, _) in zip(fhits, qhits):
            all_same_rank &= fdoc.metadata["id"] == qdoc.metadata["id"]
    checks.append(("Qdrant ranks the same passages as FAISS, in the same order", all_same_rank))

    # The signature assertion: Qdrant's cosine score equals 1 - sqL2/2 for
    # every hit, i.e. both stores are computing the same similarity.
    cos_consistent = True
    descending = True
    for fhits, qhits in zip(exp["faiss_scored"], exp["qdrant_scored"]):
        q_scores = [s for _, s in qhits]
        descending &= q_scores == sorted(q_scores, reverse=True)
        for (_, fscore), (_, cscore) in zip(fhits, qhits):
            cos_consistent &= abs(expected_cosine(fscore) - cscore) < COSINE_TOL
    checks.append(("cosine scores descend with rank (higher = more similar)", descending))
    checks.append(("every Qdrant score matches 1 - sqL2/2 from the FAISS score", cos_consistent))

    # Content checks (same as labs 01/02).
    q1610_top = exp["qdrant_scored"][1][0][0].page_content.lower()
    checks.append(("Q1610 top-1 names the Spanish founder of Montevideo", "spanish" in q1610_top))
    q1606_top = exp["qdrant_scored"][0][0][0].page_content.lower()
    checks.append(("Q1606 top-1 mentions Montevideo", "montevideo" in q1606_top))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
