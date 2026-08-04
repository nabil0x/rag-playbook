"""Lab 04 — HyDE: embed a hypothetical answer, not the question.

Lexical mismatch is a classic retrieval failure: the user asks "Who founded
Montevideo?" but the passage says "The city was established by the Spanish
in 1726". Query and passage share almost no tokens, so the question's
embedding lands far from the answer's embedding. HyDE (Hypothetical Document
Embeddings) closes the gap at the QUERY side: an LLM first writes a short
hypothetical passage that WOULD answer the question, and the retriever embeds
THAT passage instead of the raw query.

Why it works: a hypothetical passage is written in source-document language,
so its embedding lives in the same region of vector space as the real
chunks. The question only had to be understood once — by the LLM, which is
cheap and good at it — while the (expensive) embedding model only ever sees
document-shaped text.

This lab wraps the ``SimilarityRetriever`` in the ``HyDERetriever``
(``retrieval/hyde.py``) and compares, for the same questions:

* RAW — plain top-k: embed the question as the user typed it.
* HYDE — embed an LLM-written hypothetical passage instead.

Same three questions as labs 01–02 (1606/1610/1626) so you can compare the
transformations directly. The Groq LLM only writes the hypothetical passage;
embeddings stay local BGE.

Run from the repo root:
    python curriculum/05-query-transformation/04-hyde.py
    python curriculum/05-query-transformation/04-hyde.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/05-query-transformation/04-hyde.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")  # GROQ_API_KEY lives in the repo-root .env

from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from llms.groq import GroqLLM  # noqa: E402
from retrieval.hyde import HyDERetriever  # noqa: E402
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PASSAGES = 100  # deterministic head of the 3200-passage corpus (keeps runtime low)
QUESTION_IDS = [1606, 1610, 1626]  # same questions as labs 01–02, for comparison
TOP_K = 3
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq writes the hypothetical passage, never embeds
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
# 3. Experiment — raw vs HyDE retrieval for the same questions
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
    hyde_llm = GroqLLM(model=LLM_MODEL)
    hyde_retriever = HyDERetriever(hyde_llm, raw_retriever, top_k=TOP_K)

    # --- Per question: raw retrieval + the hypothetical passage + retrieval --
    results = []
    for qid, qtext in questions:
        raw_docs = raw_retriever.retrieve(qtext)
        t0 = time.perf_counter()
        hypothetical = hyde_retriever._hypothetical(qtext)
        hyde_s = time.perf_counter() - t0
        hyde_docs = hyde_retriever.retrieve(qtext)
        results.append(
            {
                "qid": qid,
                "question": qtext,
                "hypothetical": hypothetical,
                "hyde_s": hyde_s,
                "raw_docs": raw_docs,
                "hyde_docs": hyde_docs,
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
    print("Lab 04 — HyDE: embed a hypothetical answer, not the question")
    print(f"{BGE_MODEL_NAME} (local) -> FAISS top-{TOP_K} -> {LLM_MODEL} HyDE")
    print("=" * 66)

    print(f"\n[1] Corpus (deterministic subset, no randomness):")
    print(f"    {exp['indexed']} passages (first {N_PASSAGES} of 3200, ids {exp['passage_ids'][0]}..{exp['passage_ids'][-1]})")
    print(f"    embedded in {exp['embed_s']:.2f}s (dim 768), indexed in {exp['index_s']:.3f}s")

    print(f"\n[2] Raw vs HyDE (per question):")
    for r in exp["results"]:
        print(f'\n    Q[{r["qid"]}] "{r["question"]}"')
        print(f"      hypothetical ({r['hyde_s']:.1f}s): {r['hypothetical']!r}")
        print(f"      raw   top-1: {preview(r['raw_docs'][0].page_content)}")
        print(f"      hyde  top-1: {preview(r['hyde_docs'][0].page_content)}")

    print("\n[3] Takeaway")
    print("    HyDE converts the question into document-shaped text before")
    print("    embedding, so the embedding model sees what it is best at.")
    print("    The LLM does the understanding (one cheap call per question);")
    print("    the retriever then searches with a passage that shares the")
    print("    source corpus' vocabulary. Cost: one LLM call + one extra")
    print("    embed per question — no index rebuild, no store changes.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    # Structural properties (no LLM involved).
    checks.append((f"exactly {N_PASSAGES} passages indexed", exp["indexed"] == N_PASSAGES))
    checks.append(("each question returns TOP_K raw hits",
                   all(len(r["raw_docs"]) == TOP_K for r in exp["results"])))
    checks.append(("each question returns TOP_K HyDE hits",
                   all(len(r["hyde_docs"]) == TOP_K for r in exp["results"])))

    # The hypothetical passage must be a real passage — non-empty and longer
    # than a bare question (the LLM expands a question into document prose).
    for r in exp["results"]:
        tag = f"Q{r['qid']}"
        checks.append((f"{tag} hypothetical is non-empty",
                       bool(r["hypothetical"].strip())))
        checks.append((f"{tag} hypothetical is prose (>= 8 words), not a query",
                       len(r["hypothetical"].split()) >= 8))

    # Content checks: the HyDE path must still surface the answer's keyword.
    # Q1606 -> Montevideo; Q1610 -> the Spanish; Q1626 -> 1930.
    for r in exp["results"]:
        tag = f"Q{r['qid']}"
        joined = " ".join(d.page_content for d in r["hyde_docs"]).lower()
        if r["qid"] == 1606:
            kw = "montevideo"
        elif r["qid"] == 1610:
            kw = "spanish"
        else:  # 1626
            kw = "1930"
        checks.append((f"{tag} HyDE top-{TOP_K} retains '{kw}'", kw in joined))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
