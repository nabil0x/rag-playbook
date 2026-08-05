"""Lab 01 — Top-k retrieval: the retriever baseline.

Retrieval is the step that turns a query into candidate context: "which of
the indexed chunks are most relevant to this question?" The top-k similarity
retriever is the baseline every other retriever (MMR, hybrid, parent-child,
reranked) is measured against: embed the query, rank every stored passage,
keep the best k.

This lab wraps the FAISS store from lab 03 in the ``SimilarityRetriever``
block (``retrieval/similarity.py``) and does two things with it:

* QUALITATIVE — retrieve top-k passages for three real questions from
  ``rag-mini-wikipedia/test.parquet`` (answers live inside the first 100
  passages) and inspect the squared-L2 scores FAISS reports (LOWER = more
  similar; a perfect match scores 0.0).
* QUANTITATIVE — measure retrieval quality against qrels (gold relevance
  judgments) on beir-nfcorpus: embed the first 200 corpus documents once,
  retrieve top-5 per query, and compute
  Recall@k = |relevant ∩ retrieved| / |relevant| for the first 3 queries
  that have at least one relevant document inside the subset.

Deterministic, documented data choices (computed from the files at runtime):

* rag-mini passages: first N_PASSAGES=100 rows of ``passages.parquet``.
* nfcorpus documents: first 200 rows of ``corpus.jsonl``, joined as
  ``title + " " + text`` — the title carries the key terms the queries match
  on, so it belongs in the indexed text.
* nfcorpus queries: the first 20 ids (in ``queries.jsonl`` file order) that
  actually appear in ``qrels/test.tsv``. The qrels test split only covers the
  tail of queries.jsonl (the literal first 20 ids have no judgments), so we
  scan the file and keep the first 20 judged ids instead; from those we take
  the ones with >=1 relevant document inside the 200-doc subset (first 3).

Run from the repo root:
    python curriculum/04-retrieval/01-top-k.py
    python curriculum/04-retrieval/01-top-k.py --verify
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/04-retrieval/01-top-k.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PASSAGES = 100  # deterministic head of the 3200-passage corpus (keeps runtime low)
QUESTION_IDS = [1606, 1610, 1604]  # real questions from test.parquet, answers inside the subset
TOP_K = 3
PREVIEW = 62  # max characters of passage text shown next to each hit
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DIM = 768

NFCORPUS_DIR = Path("Data/corpus/beir-nfcorpus/nfcorpus")
CORPUS_PATH = NFCORPUS_DIR / "corpus.jsonl"
QUERIES_PATH = NFCORPUS_DIR / "queries.jsonl"
QRELS_PATH = NFCORPUS_DIR / "qrels" / "test.tsv"
NF_N_DOCS = 200  # deterministic head of the 3633-doc nfcorpus corpus
NF_MAX_QUERIES = 20  # first 20 qrels-covered query ids, in file order
NF_EVAL_QUERIES = 3  # of those, the first 3 with >=1 relevant doc in the subset
RECALL_K = 5


# --------------------------------------------------------------------------
# 2. Load — rag-mini corpus + questions, nfcorpus corpus + queries + qrels
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
    """Return {query_id: {relevant_corpus_id, ...}}.

    qrels/test.tsv is TAB-separated, headerless except for a literal first
    row ``query-id<TAB>corpus-id<TAB>score`` which is skipped.
    """
    qrels: dict[str, set[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3 or parts[0] == "query-id":
                continue  # header row
            qid, cid = parts[0], parts[1]
            qrels.setdefault(qid, set()).add(cid)
    return qrels


def preview(text: str, limit: int = PREVIEW) -> str:
    """Flatten a passage for one-line printing."""
    flat = text.replace("\n", " ")
    return flat[:limit] + ("..." if len(flat) > limit else "")


# --------------------------------------------------------------------------
# 3. Experiment — embed, index, retrieve, evaluate; returns every artifact
#    the demo and the verification gate need (no re-computation between the
#    two paths)
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    passage_texts, passage_ids = load_passages(PASSAGES_PATH, N_PASSAGES)
    questions = load_questions(TEST_PATH, QUESTION_IDS)

    # --- Embed the whole subset once (batched) + each question once ---------
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME)
    t0 = time.perf_counter()
    passage_vecs = embedder.embed_documents(passage_texts)
    embed_s = time.perf_counter() - t0

    # --- Build the FAISS index (in-memory), wrap it in the retriever --------
    chunks = [
        Document(page_content=t, metadata={"id": pid})
        for t, pid in zip(passage_texts, passage_ids)
    ]
    store = FAISSVectorStore(embedding=embedder)  # embed_query works for the retriever
    t0 = time.perf_counter()
    store.add(chunks, embeddings=passage_vecs)
    index_s = time.perf_counter() - t0

    # --- The retriever block: question in, top-k Documents out --------------
    retriever = SimilarityRetriever(store, top_k=TOP_K)
    retrieved = [retriever.retrieve(qtext) for _, qtext in questions]

    # --- Same queries, scored (squared L2, LOWER = more similar) ------------
    query_vecs = [store.embed_query(q) for _, q in questions]
    scored = [
        store.query_with_scores(qvec, top_k=TOP_K) for qvec in query_vecs
    ]

    # --- Qrels evaluation on nfcorpus: Recall@k against gold judgments ------
    nf_texts, nf_ids = load_nfcorpus(CORPUS_PATH, NF_N_DOCS)
    nf_queries = load_nf_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)

    nf_subset_ids = set(nf_ids)
    covered = [(qid, q) for qid, q in nf_queries if qid in qrels][:NF_MAX_QUERIES]
    nf_eval = [
        (qid, q) for qid, q in covered if qrels[qid] & nf_subset_ids
    ][:NF_EVAL_QUERIES]

    t0 = time.perf_counter()
    nf_vecs = embedder.embed_documents(nf_texts)
    nf_embed_s = time.perf_counter() - t0

    nf_chunks = [
        Document(page_content=t, metadata={"id": cid})
        for t, cid in zip(nf_texts, nf_ids)
    ]
    nf_store = FAISSVectorStore()
    t0 = time.perf_counter()
    nf_store.add(nf_chunks, embeddings=nf_vecs)
    nf_index_s = time.perf_counter() - t0

    nf_recalls: list[tuple[str, str, set[str], set[str], float]] = []
    for qid, qtext in nf_eval:
        qvec = embedder.embed_query(qtext)
        hits = nf_store.query_with_scores(qvec, top_k=RECALL_K)
        retrieved_ids = {d.metadata["id"] for d, _ in hits}
        relevant = qrels[qid] & nf_subset_ids
        recall = len(retrieved_ids & relevant) / len(relevant)
        nf_recalls.append((qid, qtext, relevant, retrieved_ids, recall))

    return {
        "passage_texts": passage_texts,
        "passage_ids": passage_ids,
        "questions": questions,
        "query_vecs": query_vecs,
        "embed_s": embed_s,
        "index_s": index_s,
        "retrieved": retrieved,
        "scored": scored,
        "dim": len(passage_vecs[0]),
        "indexed": len(passage_vecs),
        "nf_recalls": nf_recalls,
        "nf_embed_s": nf_embed_s,
        "nf_index_s": nf_index_s,
        "nf_mean_recall": (
            sum(r for *_, r in nf_recalls) / len(nf_recalls) if nf_recalls else 0.0
        ),
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 01 — Top-k retrieval: the retriever baseline")
    print(f"{BGE_MODEL_NAME} | FAISS flat-L2 | SimilarityRetriever")
    print("=" * 66)

    print(f"\n[1] Corpus (deterministic subset, no randomness):")
    print(f"    {exp['indexed']} passages (first {N_PASSAGES} of 3200, ids {exp['passage_ids'][0]}..{exp['passage_ids'][-1]})")
    print(f"    {len(exp['questions'])} questions from test.parquet:")
    for qid, qtext in exp["questions"]:
        print(f"      [{qid}] {qtext}")

    print(f"\n[2] Embed + index:")
    print(f"    embedded {exp['indexed']} passages in {exp['embed_s']:.2f}s (dim {exp['dim']})")
    print(f"    FAISS index built in {exp['index_s']:.3f}s")

    print(f"\n[3] Top-{TOP_K} per question (score = squared L2 distance, LOWER = more similar):")
    for i, (qid, qtext) in enumerate(exp["questions"]):
        print(f'\n    Q[{qid}] "{qtext}"')
        for doc, score in exp["scored"][i]:
            pid = doc.metadata.get("id", "?")
            print(f"      {score:8.4f}  [passage {pid}] {preview(doc.page_content)}")

    print(f"\n[4] SimilarityRetriever: the same {TOP_K} questions through the retriever block")
    print("    (Documents only — the scores live in the store, not the retriever):")
    for i, (qid, qtext) in enumerate(exp["questions"]):
        print(f'\n    Q[{qid}] "{qtext}"')
        for rank, doc in enumerate(exp["retrieved"][i], 1):
            pid = doc.metadata.get("id", "?")
            print(f"      {rank}. [passage {pid}] {preview(doc.page_content)}")

    print(f"\n[5] Qrels evaluation on nfcorpus (Recall@{RECALL_K} over {NF_N_DOCS} docs, {len(exp['nf_recalls'])} queries):")
    for qid, qtext, relevant, retrieved_ids, recall in exp["nf_recalls"]:
        print(f'\n    Q[{qid}] "{qtext}"')
        print(f"      relevant in subset: {sorted(relevant)}")
        print(f"      retrieved top-{RECALL_K}:  {sorted(retrieved_ids)}")
        print(f"      recall@{RECALL_K} = {recall:.2f}")
    print(f"\n    mean recall@{RECALL_K} = {exp['nf_mean_recall']:.3f}")
    print("    recall = |relevant ∩ retrieved| / |relevant| — how much of the")
    print("    gold-relevant context made it into the top-k candidate window.")
    print(f"    (nfcorpus embedded in {exp['nf_embed_s']:.2f}s, indexed in {exp['nf_index_s']:.3f}s)")

    print("\n[6] Takeaway")
    print("    The top-k similarity retriever is the baseline every other")
    print("    retriever is measured against: embed the query, rank the store,")
    print("    keep the best k. FAISS scores with squared-L2 (lower = more")
    print("    similar), and qrels say whether the ranking actually surfaces")
    print("    the gold-relevant documents — recall@k is the number to beat")
    print("    with MMR, hybrid search, and re-ranking in later labs.")


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
        ("each question returns exactly TOP_K scored hits",
         all(len(hits) == TOP_K for hits in exp["scored"]))
    )

    # Squared-L2 scores ascend with rank (0.0 would be a perfect match).
    scores_ascending = all(
        [s for _, s in hits] == sorted(s for _, s in hits) for hits in exp["scored"]
    )
    checks.append(("squared-L2 scores ascend per query (lower = more similar)", scores_ascending))

    # The retriever block agrees with the scored path on hit counts.
    checks.append(
        ("SimilarityRetriever returns exactly TOP_K documents per question",
         all(len(docs) == TOP_K for docs in exp["retrieved"]))
    )

    # Content check: Q1610 "Who founded Montevideo?" must rank the passage
    # that says the Spanish founded Montevideo at #1.
    q1610_top = exp["scored"][1][0][0].page_content.lower()
    checks.append(("Q1610 top-1 names the Spanish founder of Montevideo", "spanish" in q1610_top))

    # Q1606 "Is Uruguay's capital Montevideo?" must rank an Uruguay passage
    # that mentions Montevideo at #1.
    q1606_top = exp["scored"][0][0][0].page_content.lower()
    checks.append(("Q1606 top-1 mentions Montevideo", "montevideo" in q1606_top))

    # The retriever path ranks the same Spanish passage first for Q1610.
    q1610_ret = exp["retrieved"][1][0].page_content.lower()
    checks.append(("Q1610 top-1 through the retriever also names the Spanish founder", "spanish" in q1610_ret))

    # Qrels section: judgments exist, and every recall value is a probability.
    checks.append(
        (f"at least 2 nfcorpus queries have >=1 relevant doc in the {NF_N_DOCS}-doc subset",
         len(exp["nf_recalls"]) >= 2)
    )
    checks.append(
        ("every chosen query has >=1 relevant doc among the subset",
         all(len(relevant) >= 1 for _, _, relevant, _, _ in exp["nf_recalls"]))
    )
    checks.append(
        (f"every recall@{RECALL_K} lies in [0, 1]",
         all(0.0 <= r <= 1.0 for *_, r in exp["nf_recalls"]))
    )
    checks.append(
        (f"mean recall@{RECALL_K} lies in [0, 1]",
         0.0 <= exp["nf_mean_recall"] <= 1.0)
    )

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"    mean recall@{RECALL_K} over {len(exp['nf_recalls'])} queries = {exp['nf_mean_recall']:.3f}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
