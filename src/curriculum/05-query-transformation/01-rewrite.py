"""Lab 01 — Query rewrite: fix the query before you embed it.

Retrieval quality is capped by query quality: a vague or conversational
question ("did he win it?", "what about the second one?") embeds into the
wrong region of vector space and the top-k passages miss. Query rewrite is
the cheapest fix: an LLM rewrites the user's question into a standalone,
specific search query BEFORE it is embedded, so the retrieval step sees the
query the document store can actually answer.

This lab wraps the ``SimilarityRetriever`` block in the
``QueryRewriteRetriever`` (``src/retrieval/query_rewrite.py``) and compares, for
the same questions:

* RAW — plain top-k: embed the question as the user typed it.
* REWRITTEN — embed an LLM-rewritten version of the question instead.

The rewritten query reaches the inner retriever, so the vector store is
untouched — the whole technique lives in the retriever layer.

Questions: yes/no and short-answer questions from
``rag-mini-wikipedia/test.parquet`` whose answers live inside the first 100
passages. The rewrite turns each one into a standalone search query, and the
gate checks that the rewritten path still surfaces the passage carrying the
answer keyword.

Run from the repo root:
    python src/curriculum/05-query-transformation/01-rewrite.py
    python src/curriculum/05-query-transformation/01-rewrite.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/05-query-transformation/01-rewrite.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

load_dotenv(REPO_ROOT / ".env")  # GROQ_API_KEY lives in the repo-root .env

from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from llms.groq import GroqLLM  # noqa: E402
from retrieval.query_rewrite import QueryRewriteRetriever  # noqa: E402
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PASSAGES = 100  # deterministic head of the 3200-passage corpus (keeps runtime low)
# Vague/conversational questions whose gold answers live in the subset; the
# rewrite must resolve them into standalone queries that still find the answer.
QUESTION_IDS = [1606, 1610, 1626]
TOP_K = 3
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq is the *rewriter* LLM, never the embedder
# (Gemini alternative: LLM_MODEL = "gemini-2.5-flash" — needs GOOGLE_API_KEY in .env)
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
# 3. Experiment — raw vs rewritten retrieval for the same questions
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
    rewrite_llm = GroqLLM(model=LLM_MODEL)
    rewrite_retriever = QueryRewriteRetriever(rewrite_llm, raw_retriever, top_k=TOP_K)

    # --- Per question: raw retrieval + the rewritten query + its retrieval --
    results = []
    for qid, qtext in questions:
        raw_docs = raw_retriever.retrieve(qtext)
        t0 = time.perf_counter()
        rewritten = rewrite_retriever._rewrite(qtext)
        rewrite_s = time.perf_counter() - t0
        rewritten_docs = rewrite_retriever.retrieve(qtext)
        results.append(
            {
                "qid": qid,
                "question": qtext,
                "rewritten": rewritten,
                "rewrite_s": rewrite_s,
                "raw_docs": raw_docs,
                "rewritten_docs": rewritten_docs,
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
    print("Lab 01 — Query rewrite: fix the query before you embed it")
    print(f"{BGE_MODEL_NAME} (local) -> FAISS top-{TOP_K} -> {LLM_MODEL} rewriter")
    print("=" * 66)

    print(f"\n[1] Corpus (deterministic subset, no randomness):")
    print(f"    {exp['indexed']} passages (first {N_PASSAGES} of 3200, ids {exp['passage_ids'][0]}..{exp['passage_ids'][-1]})")
    print(f"    embedded in {exp['embed_s']:.2f}s (dim 768), indexed in {exp['index_s']:.3f}s")

    print(f"\n[2] Raw vs rewritten (per question):")
    for r in exp["results"]:
        print(f'\n    Q[{r["qid"]}] "{r["question"]}"')
        print(f"      rewritten in {r['rewrite_s']:.1f}s: {r['rewritten']!r}")
        print(f"      raw      top-1: {preview(r['raw_docs'][0].page_content)}")
        print(f"      rewritten top-1: {preview(r['rewritten_docs'][0].page_content)}")

    print("\n[3] Takeaway")
    print("    Query rewrite is a retriever-layer fix: rewrite the question")
    print("    into a standalone, specific search query, THEN embed it. The")
    print("    store and the embedding model never change — only the text")
    print("    that reaches them. It costs one cheap LLM call per question")
    print("    and pays for itself whenever users type pronouns, ellipses,")
    print("    or conversational phrasing instead of search queries.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    # Structural properties (no LLM involved).
    checks.append((f"exactly {N_PASSAGES} passages indexed", exp["indexed"] == N_PASSAGES))
    checks.append(("each question returns TOP_K raw hits",
                   all(len(r["raw_docs"]) == TOP_K for r in exp["results"])))
    checks.append(("each question returns TOP_K rewritten hits",
                   all(len(r["rewritten_docs"]) == TOP_K for r in exp["results"])))

    # The rewrite must actually produce a standalone question — non-empty
    # and different from the original (the LLM resolves the vague phrasing).
    for r in exp["results"]:
        tag = f"Q{r['qid']}"
        checks.append((f"{tag} rewrite is non-empty",
                       bool(r["rewritten"].strip())))
        checks.append((f"{tag} rewrite differs from the raw question",
                       r["rewritten"].strip() != r["question"].strip()))

    # Content checks: the rewritten path must still surface the answer's
    # keyword. Q1606 "Is Uruguay's capital Montevideo?" -> Montevideo;
    # Q1610 "Who founded Montevideo?" -> the Spanish; Q1626 "Did Uruguay
    # host the first ever World Cup?" -> 1930.
    for r in exp["results"]:
        tag = f"Q{r['qid']}"
        joined = " ".join(d.page_content for d in r["rewritten_docs"]).lower()
        if r["qid"] == 1606:
            kw = "montevideo"
        elif r["qid"] == 1610:
            kw = "spanish"
        else:  # 1626
            kw = "1930"
        checks.append((f"{tag} rewritten top-{TOP_K} retains '{kw}'", kw in joined))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
