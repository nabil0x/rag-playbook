"""Lab 05 — Contextual Compression: shrink retrieved context before the LLM reads it.

Retrieval returns top-k *documents* — whole passages that are only partly
about the question. Every irrelevant sentence is then paid for twice: once in
context tokens, once in the answer LLM's attention. Contextual compression
sits between the retriever and the answer step and cuts that waste.

The idea, from LangChain's ``ContextualCompressionRetriever``:

* a BASE RETRIEVER (here: FAISS over local BGE embeddings) returns the usual
  top-k passages;
* a COMPRESSOR — an LLM with an "extract the sentences relevant to this
  question" prompt (``LLMChainExtractor``) — reads each passage *and the
  question*, and returns only the query-relevant sentences, dropping the rest.

The compressor is called once per retrieved document, so the cost is ``#
questions x k`` LLM round-trips (2 x 3 = 6 here, a couple of seconds each on
Groq). That is the compression tax: you pay a small LLM bill to keep the
*answer* LLM's context small.

This lab quantifies the payoff: for every question it prints the raw
top-k context (characters + whitespace-token estimate) next to the compressed
context and the reduction percentage.

Run from the repo root:
    python curriculum/04-retrieval/05-compression.py
    python curriculum/04-retrieval/05-compression.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/04-retrieval/05-compression.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")  # GROQ_API_KEY lives in the repo-root .env

from langchain_classic.retrievers import (  # noqa: E402
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import (  # noqa: E402
    LLMChainExtractor,
)
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
# (Gemini alternative: from langchain_google_genai import ChatGoogleGenerativeAI)

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PASSAGES = 100  # deterministic head of the 3200-passage corpus (keeps runtime low)
QUESTION_IDS = [1606, 1610]  # 2 questions x K=3 = 6 compressor calls per run
TOP_K = 3
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq is the *compressor* LLM, never the embedder
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


def tokens(text: str) -> int:
    """Cheap token estimate: whitespace-split word count."""
    return len(text.split())


def total_chars(docs: list[Document]) -> int:
    """Total characters across a list of documents."""
    return sum(len(d.page_content) for d in docs)


def total_tokens(docs: list[Document]) -> int:
    """Total token estimate across a list of documents."""
    return sum(tokens(d.page_content) for d in docs)


# --------------------------------------------------------------------------
# 3. Experiment — raw retrieval vs LLM-compressed retrieval; returns every
#    artifact the demo and the verification gate need (no re-computation
#    between the two paths)
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    passage_texts, passage_ids = load_passages(PASSAGES_PATH, N_PASSAGES)
    questions = load_questions(TEST_PATH, QUESTION_IDS)

    # --- Embed locally (BGE) and index in-memory with langchain-native FAISS -
    chunks = [
        Document(page_content=t, metadata={"id": pid})
        for t, pid in zip(passage_texts, passage_ids)
    ]
    embedder = HuggingFaceEmbeddings(
        model_name=BGE_MODEL_NAME, encode_kwargs={"normalize_embeddings": True}
    )
    t0 = time.perf_counter()
    store = FAISS.from_documents(chunks, embedder)
    index_s = time.perf_counter() - t0

    # --- Base retriever (raw top-k) -----------------------------------------
    base_retriever = store.as_retriever(search_kwargs={"k": TOP_K})

    # --- Compressor LLM + the wrapped retriever ------------------------------
    # Groq only compresses; every embedding above is local BGE.
    # (Gemini alternative: llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0))
    llm = ChatGroq(model=LLM_MODEL, temperature=0.0)
    compressor = LLMChainExtractor.from_llm(llm)
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # --- Per question: raw vs compressed -------------------------------------
    results = []
    for qid, qtext in questions:
        raw_docs = base_retriever.invoke(qtext)
        t0 = time.perf_counter()
        comp_docs = compressed_retriever.invoke(qtext)
        comp_s = time.perf_counter() - t0
        results.append(
            {
                "qid": qid,
                "question": qtext,
                "raw_docs": raw_docs,
                "compressed_docs": comp_docs,
                "raw_chars": total_chars(raw_docs),
                "raw_tokens": total_tokens(raw_docs),
                "comp_chars": total_chars(comp_docs),
                "comp_tokens": total_tokens(comp_docs),
                "comp_s": comp_s,
            }
        )

    return {
        "passage_texts": passage_texts,
        "passage_ids": passage_ids,
        "questions": questions,
        "indexed": len(passage_texts),
        "index_s": index_s,
        "results": results,
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 05 — Contextual Compression: shrink context before the answer step")
    print(f"{BGE_MODEL_NAME} (local) -> FAISS top-{TOP_K} -> {LLM_MODEL} extractor")
    print("=" * 66)

    print(f"\n[1] Corpus (deterministic subset, no randomness):")
    print(f"    {exp['indexed']} passages (first {N_PASSAGES} of 3200, ids {exp['passage_ids'][0]}..{exp['passage_ids'][-1]})")
    print(f"    {len(exp['questions'])} questions from test.parquet:")
    for qid, qtext in exp["questions"]:
        print(f"      [{qid}] {qtext}")

    print(f"\n[2] Index:")
    print(f"    FAISS index built in {exp['index_s']:.3f}s over local BGE embeddings")

    print(f"\n[3] Raw vs compressed context (chars / token-estimate):")
    for r in exp["results"]:
        print(f'\n    Q[{r["qid"]}] "{r["question"]}"')
        raw_c, raw_t = r["raw_chars"], r["raw_tokens"]
        comp_c, comp_t = r["comp_chars"], r["comp_tokens"]
        print(f"      raw        {raw_c:5d} chars / {raw_t:4d} tokens  (k={len(r['raw_docs'])})")
        print(f"      compressed {comp_c:5d} chars / {comp_t:4d} tokens  (k={len(r['compressed_docs'])})")
        red_c = 100.0 * (raw_c - comp_c) / raw_c if raw_c else 0.0
        red_t = 100.0 * (raw_t - comp_t) / raw_t if raw_t else 0.0
        print(f"      reduction  {red_c:5.1f}% chars / {red_t:5.1f}% tokens "
              f"({r['comp_s']:.1f}s compressor time)")
        print("      raw top-1:  " + preview(r["raw_docs"][0].page_content))
        if r["compressed_docs"]:
            print("      compressed: " + preview(r["compressed_docs"][0].page_content))
        else:
            print("      compressed: (empty — extractor dropped every sentence)")

    print("\n[4] Takeaway")
    print("    The compressor keeps only the sentences the question is about,")
    print("    so the answer step reads a fraction of the original context.")
    print("    The price: one LLM call per retrieved document (2 questions x")
    print(f"    k={TOP_K} = {2 * TOP_K} calls here). Contextual compression trades")
    print("    that small LLM bill for a smaller, cleaner answer context —")
    print("    and the raw passages stay available if a question needs them.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    # Structural properties (no LLM involved).
    checks.append((f"exactly {N_PASSAGES} passages indexed", exp["indexed"] == N_PASSAGES))
    checks.append(("each question returns TOP_K raw hits",
                   all(len(r["raw_docs"]) == TOP_K for r in exp["results"])))

    # Compression stability, pinned to properties that survive LLM wording
    # variance (the extractor keeps *which* sentences, not exact text).
    for r in exp["results"]:
        tag = f"Q{r['qid']}"

        # The compressor must return at least one sentence.
        checks.append((f"{tag} compressed context is non-empty",
                       len(r["compressed_docs"]) > 0 and r["comp_chars"] > 0))

        # Compressed context must be strictly shorter than raw (in chars) —
        # LLMChainExtractor drops irrelevant sentences, so this holds with
        # margin; a small tolerance keeps it robust to 1-2 word overrides.
        checks.append((f"{tag} compressed chars < raw chars",
                       r["comp_chars"] < r["raw_chars"]))

        # The compressed context must still carry the answer's keyword —
        # the whole point is relevance, not just shrinkage. "Montevideo"
        # appears in the gold answer of both questions and survives the
        # extractor because the queries name it explicitly.
        joined = " ".join(d.page_content for d in r["compressed_docs"]).lower()
        checks.append((f"{tag} compressed context retains 'montevideo'",
                       "montevideo" in joined))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
