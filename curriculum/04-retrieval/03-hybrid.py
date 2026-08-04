"""Lab 03 — Hybrid retrieval: fusing dense (vector) + sparse (BM25) rankings.

Dense retrieval (embedding similarity) is strong on *semantic* matches: it
finds the passage that means the same thing as the question even when the
wording shares no words. Sparse retrieval (BM25) is strong on *exact term*
matches: it nails the passage that literally contains the rare keyword the
question names. Each arm alone is wrong in a different way — so we fuse them.

This lab builds both arms over the same deterministic subset of
``Data/corpus/rag-mini-wikipedia``:

* DENSE arm — ``FAISSVectorStore`` + ``SimilarityRetriever`` (repo
  ``retrieval/similarity.py``): BGE embeddings, cosine-style top-k.
* SPARSE arm — ``BM25Retriever`` from ``langchain_classic`` (rank_bm25
  backend): token-overlap top-k.

Then it fuses the two rankings with Reciprocal Rank Fusion (RRF): every
document earns ``weight / (c + rank)`` for each ranked list it appears in,
and the summed scores decide the final order. A document found by BOTH arms
collects two contributions and is deduplicated — the classic hybrid win.

Two fusion implementations are demonstrated side by side:

* ``EnsembleRetriever`` (langchain-classic, the plan-preferred path) — the
  dense arm is wrapped as a Runnable, weights ``[0.5, 0.5]``, ``c=60``.
* ``HybridRetriever`` (repo ``retrieval/hybrid.py``) — same RRF math
  (``1/(60+rank)``), dedup by ``page_content``; the sparse arm is adapted to
  its ``retrieve(question)`` contract with a 5-line ``BM25Adapter``.

Run from the repo root:
    python curriculum/04-retrieval/03-hybrid.py
    python curriculum/04-retrieval/03-hybrid.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/04-retrieval/03-hybrid.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_classic.retrievers import EnsembleRetriever  # noqa: E402
from langchain_classic.retrievers.bm25 import BM25Retriever  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from retrieval.hybrid import HybridRetriever  # noqa: E402
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PASSAGES = 100  # deterministic head of the 3200-passage corpus (keeps runtime low)
QUESTION_IDS = [1606, 1610, 1604]  # real questions from test.parquet, answers inside the subset
# A purely semantic question (no rare keyword shared with its target passage):
# passages 44/6 say "88% of the population are of European descent" — the
# question never says "European descent", so only the dense arm can bridge it.
SEMANTIC_QUESTION = "What share of the nation's people trace their roots to Europe?"
TOP_K = 5
RRF_C = 60  # RRF constant: how strongly lower ranks are discounted (k=60 is standard)
WEIGHTS = [0.5, 0.5]  # dense : sparse fusion weights
PREVIEW = 62  # max characters of passage text shown next to each hit
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DIM = 768


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


class BM25Adapter:
    """Expose langchain-classic ``BM25Retriever`` as the repo ``.retrieve()`` contract.

    ``HybridRetriever`` (``retrieval/hybrid.py``) calls
    ``sparse_retriever.retrieve(question)``; the classic BM25 retriever is a
    Runnable and speaks ``.invoke()`` instead. This 5-line adapter bridges the
    two so the repo fusion class can consume the same sparse arm.
    """

    def __init__(self, bm25: BM25Retriever):
        self._bm25 = bm25

    def retrieve(self, question: str) -> list[Document]:
        return self._bm25.invoke(question)


def overlap_analysis(
    dense_docs: list[Document], sparse_docs: list[Document]
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """Return (dense_rank_by_text, sparse_rank_by_text, texts_found_by_both)."""
    dense_rank = {d.page_content: r for r, d in enumerate(dense_docs, 1)}
    sparse_rank = {d.page_content: r for r, d in enumerate(sparse_docs, 1)}
    both = sorted(set(dense_rank) & set(sparse_rank))
    return dense_rank, sparse_rank, both


# --------------------------------------------------------------------------
# 3. Experiment — embed, index, build both arms, fuse; returns every artifact
#    the demo and the verification gate need (no re-computation between paths)
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    passage_texts, passage_ids = load_passages(PASSAGES_PATH, N_PASSAGES)
    questions = load_questions(TEST_PATH, QUESTION_IDS)
    questions.append((len(questions), SEMANTIC_QUESTION))  # 4th, purely semantic

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
    store = FAISSVectorStore(embedding=embedder)
    t0 = time.perf_counter()
    store.add(chunks, embeddings=passage_vecs)
    index_s = time.perf_counter() - t0

    # --- The two arms over the same corpus ----------------------------------
    dense = SimilarityRetriever(store, top_k=TOP_K)
    sparse = BM25Retriever.from_documents(chunks, k=TOP_K)

    # --- Fusion path 1: langchain-classic EnsembleRetriever (plan-preferred) -
    # The dense arm is wrapped as a Runnable; id_key=None dedups by page_content
    # (the same key the repo HybridRetriever uses).
    ensemble = EnsembleRetriever(
        retrievers=[RunnableLambda(dense.retrieve), sparse],
        weights=WEIGHTS,
        c=RRF_C,
        k=TOP_K,
        id_key=None,
    )

    # --- Fusion path 2: repo HybridRetriever (RRF 1/(60+rank), same math) ----
    repo_hybrid = HybridRetriever(dense, BM25Adapter(sparse), top_k=TOP_K)

    # --- Run every question through both arms and both fusions --------------
    per_question = []
    for qid, qtext in questions:
        dense_docs = dense.retrieve(qtext)
        sparse_docs = sparse.invoke(qtext)
        # EnsembleRetriever returns the full RRF-ranked list (union of both
        # arms); take the top TOP_K — the same contract the repo fusion uses.
        fused_ens = ensemble.invoke(qtext)[:TOP_K]
        fused_repo = repo_hybrid.retrieve(qtext)
        dense_rank, sparse_rank, both = overlap_analysis(dense_docs, sparse_docs)
        per_question.append(
            {
                "qid": qid,
                "qtext": qtext,
                "dense": dense_docs,
                "sparse": sparse_docs,
                "fused_ens": fused_ens,
                "fused_repo": fused_repo,
                "dense_rank": dense_rank,
                "sparse_rank": sparse_rank,
                "both": both,
            }
        )

    return {
        "passage_texts": passage_texts,
        "passage_ids": passage_ids,
        "questions": questions,
        "query_vecs": query_vecs,
        "embed_s": embed_s,
        "index_s": index_s,
        "per_question": per_question,
        "dim": len(passage_vecs[0]),
        "indexed": len(passage_vecs),
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    passage_lk = passage_lookup(exp["passage_texts"], exp["passage_ids"])

    print("=" * 66)
    print("Lab 03 — Hybrid retrieval: dense + sparse fusion (RRF)")
    print(f"{BGE_MODEL_NAME} dense | BM25 sparse | RRF c={RRF_C} weights={WEIGHTS}")
    print("=" * 66)

    print(f"\n[1] Corpus (deterministic subset, no randomness):")
    print(f"    {exp['indexed']} passages (first {N_PASSAGES} of 3200, ids {exp['passage_ids'][0]}..{exp['passage_ids'][-1]})")
    print(f"    {len(exp['questions'])} questions ({len(exp['questions']) - 1} from test.parquet + 1 semantic):")
    for qid, qtext in exp["questions"]:
        print(f"      [{qid}] {qtext}")

    print(f"\n[2] The two arms (same corpus, same top-{TOP_K}):")
    print(f"    embedded {exp['indexed']} passages in {exp['embed_s']:.2f}s (dim {exp['dim']})")
    print(f"    FAISS index built in {exp['index_s']:.3f}s")
    print("    dense  = FAISSVectorStore + SimilarityRetriever  (semantic match)")
    print("    sparse = BM25Retriever (rank_bm25)               (exact-term match)")

    for q in exp["per_question"]:
        print(f'\n[3] Q[{q["qid"]}] "{q["qtext"]}"')
        print(f'    dense top-{TOP_K}:')
        for rank, doc in enumerate(q["dense"], 1):
            pid = doc.metadata.get("id", "?")
            print(f"      {rank}. [p{pid}] {preview(doc.page_content)}")
        print(f'    sparse top-{TOP_K}:')
        for rank, doc in enumerate(q["sparse"], 1):
            pid = doc.metadata.get("id", "?")
            print(f"      {rank}. [p{pid}] {preview(doc.page_content)}")
        print(f'    fused top-{TOP_K} (EnsembleRetriever):')
        for rank, doc in enumerate(q["fused_ens"], 1):
            pid = doc.metadata.get("id", "?")
            print(f"      {rank}. [p{pid}] {preview(doc.page_content)}")
        print(f'    fused top-{TOP_K} (repo HybridRetriever):')
        for rank, doc in enumerate(q["fused_repo"], 1):
            pid = doc.metadata.get("id", "?")
            print(f"      {rank}. [p{pid}] {preview(doc.page_content)}")
        if q["both"]:
            fused_rank = {d.page_content: r for r, d in enumerate(q["fused_ens"], 1)}
            print(f'    found by BOTH arms ({len(q["both"])} docs — RRF boost):')
            for text in q["both"]:
                pid = next(
                    d.metadata.get("id", "?")
                    for d in q["dense"] + q["sparse"]
                    if d.page_content == text
                )
                print(f"      [p{pid}] dense#{q['dense_rank'][text]} sparse#{q['sparse_rank'][text]} "
                      f"-> fused#{fused_rank.get(text, '-')}")
        else:
            print("    found by BOTH arms: none (arms agree on nothing this query)")

    print("\n[4] Takeaway")
    print("    Dense alone misses exact-term needles; sparse alone misses")
    print("    paraphrases. RRF fuses the two rankings: a doc found by both")
    print("    arms collects two 1/(60+rank) contributions, is deduplicated,")
    print("    and outranks every doc found by only one arm — the hybrid win.")
    print("    EnsembleRetriever (langchain-classic) and the repo's")
    print("    HybridRetriever implement the same RRF math and agree here.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    # Dimension and count match the model / subset.
    checks.append(("embedding dimension is 768 (BGE base)", exp["dim"] == BGE_DIM))
    checks.append((f"exactly {N_PASSAGES} passages indexed", exp["indexed"] == N_PASSAGES))

    # Both arms return exactly TOP_K documents for every question.
    checks.append(
        ("dense arm returns exactly TOP_K docs per question",
         all(len(q["dense"]) == TOP_K for q in exp["per_question"]))
    )
    checks.append(
        ("sparse arm returns exactly TOP_K docs per question",
         all(len(q["sparse"]) == TOP_K for q in exp["per_question"]))
    )

    # Both fusions return exactly TOP_K documents with no duplicates.
    checks.append(
        ("EnsembleRetriever fused returns TOP_K docs, no duplicates",
         all(len(q["fused_ens"]) == TOP_K and len({d.page_content for d in q["fused_ens"]}) == TOP_K
             for q in exp["per_question"]))
    )
    checks.append(
        ("repo HybridRetriever fused returns TOP_K docs, no duplicates",
         all(len(q["fused_repo"]) == TOP_K and len({d.page_content for d in q["fused_repo"]}) == TOP_K
             for q in exp["per_question"]))
    )

    # RRF cannot invent documents: every fused doc must come from the union of
    # the two arms' result sets (checked for both fusion paths).
    def fused_within_union(fused: list[Document], dense: list[Document], sparse: list[Document]) -> bool:
        union = {d.page_content for d in dense} | {d.page_content for d in sparse}
        return all(d.page_content in union for d in fused)

    checks.append(
        ("every fused doc appears in the union of dense+sparse results (EnsembleRetriever)",
         all(fused_within_union(q["fused_ens"], q["dense"], q["sparse"]) for q in exp["per_question"]))
    )
    checks.append(
        ("every fused doc appears in the union of dense+sparse results (repo HybridRetriever)",
         all(fused_within_union(q["fused_repo"], q["dense"], q["sparse"]) for q in exp["per_question"]))
    )

    # Keyword-heavy query: Q1610 "Who founded Montevideo?" — the fused top-1
    # must be the passage that literally contains the rare term "Montevideo"
    # (passage id 2: "Montevideo was founded by the Spanish ...").
    q1610 = next(q for q in exp["per_question"] if q["qid"] == 1610)
    kw_top1 = q1610["fused_ens"][0].page_content.lower()
    checks.append(("keyword query fused top-1 contains the rare term 'montevideo'", "montevideo" in kw_top1))

    # Purely semantic query: the fused top-1 must be the passage about European
    # descent (passage id 6) even though the question never says those words.
    sem = exp["per_question"][-1]
    sem_top1 = sem["fused_ens"][0].page_content.lower()
    checks.append(("semantic query fused top-1 is the European-descent passage", "european descent" in sem_top1))

    # RRF boost: a document found by BOTH arms outranks every document found by
    # only one arm in the fused output (two contributions beat one, always).
    boost_ok = True
    any_overlap = False
    for q in exp["per_question"]:
        if not q["both"]:
            continue
        any_overlap = True
        fused_rank = {d.page_content: r for r, d in enumerate(q["fused_ens"], 1)}
        both_ranks = [fused_rank[t] for t in q["both"] if t in fused_rank]
        single_ranks = [r for t, r in fused_rank.items() if t not in q["both"]]
        if both_ranks and single_ranks and max(both_ranks) > min(single_ranks):
            boost_ok = False
    checks.append(("RRF boost: both-arm docs outrank single-arm docs in fused output", boost_ok and any_overlap))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)