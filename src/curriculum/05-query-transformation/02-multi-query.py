"""Lab 02 — Multi-query: one question in, several search queries out.

A single user question is a single point in embedding space, but the answer
may live near several DIFFERENT points: the same fact phrased as "capital of
Uruguay", "Montevideo", "Uruguay's largest city" … each lands in a slightly
different region of the vector store. Top-k retrieval samples ONE region, so
phrasing luck decides what comes back.

Multi-query expansion (LangChain's ``MultiQueryRetriever``, from
``langchain-classic``) removes the luck: an LLM rewrites the user question
into 3+ search-query variants, every variant is retrieved, and the results
are merged into one deduplicated union. A fact reachable through ANY
phrasing now has a chance to surface.

The retrieval machinery is LangChain-native this time (instead of the
component library's ``SimilarityRetriever``): ``MultiQueryRetriever`` wraps a
LangChain ``BaseRetriever``, so the inner retriever is
``store.as_retriever(search_kwargs={"k": TOP_K})`` over the same local BGE
embeddings. This is the same shape as track 04 lab 05 — the base retriever
is just a different object, and the query-transformation layer sits on top.

Same three questions as lab 01 (1606/1610/1626) so you can compare the
transformations directly.

Run from the repo root:
    python curriculum/05-query-transformation/02-multi-query.py
    python curriculum/05-query-transformation/02-multi-query.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/05-query-transformation/02-multi-query.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")  # GROQ_API_KEY lives in the repo-root .env

from langchain_classic.retrievers.multi_query import (  # noqa: E402
    MultiQueryRetriever,
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
QUESTION_IDS = [1606, 1610, 1626]  # same questions as lab 01, for comparison
TOP_K = 3  # per-variant retrieval depth; the union is bigger than k
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq is the query *generator*, never the embedder
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
# 3. Experiment — LLM query variants -> per-variant retrieval -> union
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

    # --- Base retriever (LangChain contract) + the multi-query wrapper --------
    base_retriever = store.as_retriever(search_kwargs={"k": TOP_K})
    # include_original=True: the user's own query joins the LLM's variants, so
    # the union can never be WORSE than plain top-k.
    llm = ChatGroq(model=LLM_MODEL, temperature=0.0)
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm, include_original=True
    )

    # --- Per question: generated variants + the merged union ------------------
    results = []
    for qid, qtext in questions:
        t0 = time.perf_counter()
        variants = multi_retriever.llm_chain.invoke({"question": qtext})
        gen_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        union = multi_retriever.invoke(qtext)
        union_s = time.perf_counter() - t0
        results.append(
            {
                "qid": qid,
                "question": qtext,
                "variants": variants,
                "gen_s": gen_s,
                "union": union,
                "union_s": union_s,
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
    print("Lab 02 — Multi-query: one question in, several search queries out")
    print(f"{BGE_MODEL_NAME} (local) -> FAISS top-{TOP_K} -> {LLM_MODEL} generator")
    print("=" * 66)

    print(f"\n[1] Corpus (deterministic subset, no randomness):")
    print(f"    {exp['indexed']} passages (first {N_PASSAGES} of 3200, ids {exp['passage_ids'][0]}..{exp['passage_ids'][-1]})")
    print(f"    FAISS index built in {exp['index_s']:.3f}s over local BGE embeddings")

    print(f"\n[2] Generated variants -> union (per question):")
    for r in exp["results"]:
        print(f'\n    Q[{r["qid"]}] "{r["question"]}"')
        print(f"      variants ({len(r['variants'])}, {r['gen_s']:.1f}s):")
        for v in r["variants"]:
            print(f"        - {v}")
        print(f"      union: {len(r['union'])} unique docs "
              f"(k={TOP_K} per variant, {r['union_s']:.1f}s)")
        for rank, doc in enumerate(r["union"][:TOP_K], 1):
            pid = doc.metadata.get("id", "?")
            print(f"        {rank}. [passage {pid}] {preview(doc.page_content)}")
        if len(r["union"]) > TOP_K:
            print(f"        … {len(r['union']) - TOP_K} more unique docs beyond top-{TOP_K}")

    print("\n[3] Takeaway")
    print("    Multi-query trades one cheap LLM call per question for several")
    print("    retrieval passes over different phrasings, then merges the")
    print("    unique results. The union is never worse than plain top-k")
    print("    (the original query is included), and a fact reachable only")
    print("    through a different phrasing finally has a chance to surface.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    # Structural properties (no LLM involved).
    checks.append((f"exactly {N_PASSAGES} passages indexed", exp["indexed"] == N_PASSAGES))

    # Multi-query properties, pinned to what survives LLM wording variance.
    for r in exp["results"]:
        tag = f"Q{r['qid']}"

        # The generator must return at least one variant.
        checks.append((f"{tag} generated >=1 query variant", len(r["variants"]) >= 1))

        # The union is deduplicated: unique page content, no repeats.
        contents = [d.page_content for d in r["union"]]
        checks.append((f"{tag} union is deduplicated",
                       len(contents) == len(set(contents))))

        # include_original=True guarantees the union covers plain top-k, so it
        # is always >= TOP_K distinct documents.
        checks.append((f"{tag} union size >= TOP_K", len(r["union"]) >= TOP_K))

        # Content check: the union must carry the answer's keyword.
        joined = " ".join(d.page_content for d in r["union"]).lower()
        if r["qid"] == 1606:
            kw = "montevideo"
        elif r["qid"] == 1610:
            kw = "spanish"
        else:  # 1626
            kw = "1930"
        checks.append((f"{tag} union retains '{kw}'", kw in joined))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
