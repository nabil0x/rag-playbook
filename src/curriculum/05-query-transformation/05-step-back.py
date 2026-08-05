"""Lab 05 — Step-back: ask a broader question first, then merge.

Specific questions miss general context. "How does the parser handle
overlap?" might only match the sentence that mentions overlap, while the
passage that actually explains the splitter's design — the context the
specific question depends on — never makes the top-k. Step-back prompting
fixes this at the QUERY side: an LLM abstracts the question into a broader
"step-back" question ("How does the recursive text splitter work?"), the
retriever runs BOTH the step-back and the original question, and the results
are merged and deduplicated. The general context now has a retrieval pass of
its own; the original query keeps precision.

This lab wraps the ``SimilarityRetriever`` in the ``StepBackRetriever``
(``src/retrieval/step_back.py``) and compares, for the same questions:

* RAW — plain top-k: embed the question as the user typed it.
* STEP-BACK — merged retrieval over the step-back + original questions.

Same three questions as labs 01/02/04 (1606/1610/1626) so you can compare
the transformations directly. The Groq LLM only writes the step-back
question; embeddings stay local BGE.

Run from the repo root:
    python src/curriculum/05-query-transformation/05-step-back.py
    python src/curriculum/05-query-transformation/05-step-back.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/05-query-transformation/05-step-back.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

load_dotenv(REPO_ROOT / ".env")  # GROQ_API_KEY lives in the repo-root .env

from embeddings.bge import BGEEmbedding  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from llms.groq import GroqLLM  # noqa: E402
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from retrieval.step_back import StepBackRetriever  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the experiment
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PASSAGES = 100  # deterministic head of the 3200-passage corpus (keeps runtime low)
QUESTION_IDS = [1606, 1610, 1626]  # same questions as labs 01/02/04, for comparison
TOP_K = 3
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq writes the step-back question, never embeds
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
# 3. Experiment — raw vs step-back retrieval for the same questions
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
    stepback_llm = GroqLLM(model=LLM_MODEL)
    stepback_retriever = StepBackRetriever(stepback_llm, raw_retriever, top_k=TOP_K)

    # --- Per question: raw retrieval + the step-back question + its merge ----
    results = []
    for qid, qtext in questions:
        raw_docs = raw_retriever.retrieve(qtext)
        t0 = time.perf_counter()
        stepback = stepback_retriever._step_back(qtext)
        stepback_s = time.perf_counter() - t0
        stepback_docs = stepback_retriever.retrieve(qtext)
        results.append(
            {
                "qid": qid,
                "question": qtext,
                "stepback": stepback,
                "stepback_s": stepback_s,
                "raw_docs": raw_docs,
                "stepback_docs": stepback_docs,
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
    print("Lab 05 — Step-back: ask a broader question first, then merge")
    print(f"{BGE_MODEL_NAME} (local) -> FAISS top-{TOP_K} -> {LLM_MODEL} step-back")
    print("=" * 66)

    print(f"\n[1] Corpus (deterministic subset, no randomness):")
    print(f"    {exp['indexed']} passages (first {N_PASSAGES} of 3200, ids {exp['passage_ids'][0]}..{exp['passage_ids'][-1]})")
    print(f"    embedded in {exp['embed_s']:.2f}s (dim 768), indexed in {exp['index_s']:.3f}s")

    print(f"\n[2] Raw vs step-back (per question):")
    for r in exp["results"]:
        print(f'\n    Q[{r["qid"]}] "{r["question"]}"')
        print(f"      step-back ({r['stepback_s']:.1f}s): {r['stepback']!r}")
        print(f"      raw        top-1: {preview(r['raw_docs'][0].page_content)}")
        print(f"      step-back  top-1: {preview(r['stepback_docs'][0].page_content)}")

    print("\n[3] Takeaway")
    print("    Step-back adds ONE broader retrieval pass for general context,")
    print("    then merges it with the precise original-query results (dupes")
    print("    dropped, first occurrence wins). The merge guarantees the")
    print("    answer's passage still surfaces while giving the background")
    print("    chunk a chance to enter the candidate set. Cost: one LLM call")
    print("    + one extra embed + one extra FAISS query per question.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []

    # Structural properties (no LLM involved).
    checks.append((f"exactly {N_PASSAGES} passages indexed", exp["indexed"] == N_PASSAGES))
    checks.append(("each question returns TOP_K raw hits",
                   all(len(r["raw_docs"]) == TOP_K for r in exp["results"])))
    checks.append(("each question returns TOP_K step-back hits",
                   all(len(r["stepback_docs"]) == TOP_K for r in exp["results"])))

    # The step-back question must be non-empty.
    for r in exp["results"]:
        tag = f"Q{r['qid']}"
        checks.append((f"{tag} step-back is non-empty",
                       bool(r["stepback"].strip())))

    # The merged results must be deduplicated (step-back + original overlap).
    for r in exp["results"]:
        tag = f"Q{r['qid']}"
        contents = [d.page_content for d in r["stepback_docs"]]
        checks.append((f"{tag} merged results are deduplicated",
                       len(contents) == len(set(contents))))

    # Content checks: the merged path must still surface the answer's keyword.
    # Q1606 -> Montevideo; Q1610 -> the Spanish; Q1626 -> 1930.
    for r in exp["results"]:
        tag = f"Q{r['qid']}"
        joined = " ".join(d.page_content for d in r["stepback_docs"]).lower()
        if r["qid"] == 1606:
            kw = "montevideo"
        elif r["qid"] == 1610:
            kw = "spanish"
        else:  # 1626
            kw = "1930"
        checks.append((f"{tag} step-back top-{TOP_K} retains '{kw}'", kw in joined))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
