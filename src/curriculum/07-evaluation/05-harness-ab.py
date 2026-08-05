"""Lab 05 — EvaluationHarness A/B: which embedding model should you ship?

Labs 01-04 measured pieces (retrieval metrics, faithfulness, judge agreement,
regression). This lab runs the full generation-metric suite — the RAGAS
quartet faithfulness / answer relevance / context precision / context recall —
through ``evaluation/harness.py::EvaluationHarness`` on a hand-checked golden
set, for TWO retrieval variants:

* Variant A — BGE (BAAI/bge-base-en-v1.5), the repo default embedder.
* Variant B — E5 (intfloat/multilingual-e5-base), the alternative embedder.

Everything else is identical (same corpus, same questions, same references,
same judge, same generator, same top_k). Only the embedding model changes, so
any metric difference is attributable to the embedder — that is the A/B
discipline: change one knob, measure everything.

The golden set below is authored like ``evaluation/golden.py``: hand-written
references, but grounded — each question was kept only because its gold answer
is contained verbatim in a retrievable passage of the indexed corpus (checked
at authoring time), so a good retriever can genuinely answer it.

Run from the repo root:
    python curriculum/07-evaluation/05-harness-ab.py
    python curriculum/07-evaluation/05-harness-ab.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from embeddings.bge import BGEEmbedding  # noqa: E402
from embeddings.e5 import E5Embedding  # noqa: E402
from evaluation.harness import EvaluationHarness  # noqa: E402
from evaluation.judge import LLMJudge  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    FaithfulnessMetric,
)
from langchain_core.documents import Document  # noqa: E402
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

load_dotenv()

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
RAG_MINI = Path("Data/corpus/rag-mini-wikipedia")
PASSAGES_PATH = RAG_MINI / "passages.parquet"
N_PASSAGES = 800  # deterministic head of the corpus (keeps runtime low)
TOP_K = 3  # context chunks fed to the generator (harness.top_k)
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
E5_MODEL_NAME = "intfloat/multilingual-e5-base"

# The golden set: hand-authored references, validated at authoring time so the
# gold answer appears verbatim in a retrievable passage of the corpus.
GOLDEN_QA: list[dict] = [
    {
        "doc": "rag-mini",
        "question": "Who assassinated Lincoln?",
        "reference": "John Wilkes Booth, Lincoln's assassin, can be seen in "
        "the crowd at Lincoln's second inauguration on March 4, 1865.",
    },
    {
        "doc": "rag-mini",
        "question": "The Celsius crater on the Moon is what?",
        "reference": "The Celsius crater on the Moon is named after the "
        "scientist Anders Celsius.",
    },
    {
        "doc": "rag-mini",
        "question": "What period of rapid economic growth did the United "
        "States experience during Coolidge's presidency?",
        "reference": "During Coolidge's presidency the United States "
        "experienced the period of rapid economic growth known as the "
        "Roaring Twenties.",
    },
    {
        "doc": "rag-mini",
        "question": "In 1905 Coolidge met and married whom?",
        "reference": "In 1905 Coolidge met and married Grace Anna Goodhue, "
        "a local schoolteacher and fellow Vermonter.",
    },
    {
        "doc": "rag-mini",
        "question": "When did Islam become the dominant religion in Java "
        "and Sumatra?",
        "reference": "Islam became the dominant religion in Java and "
        "Sumatra by the end of the 16th century.",
    },
    {
        "doc": "rag-mini",
        "question": "Who was on the committee with Adams to draft a "
        "Declaration of Independence?",
        "reference": "Adams was appointed on a committee with Thomas "
        "Jefferson, Benjamin Franklin, Robert R. Livingston and Roger "
        "Sherman to draft a Declaration of Independence.",
    },
]


# --------------------------------------------------------------------------
# 2. Build one retriever per embedding model
# --------------------------------------------------------------------------
def build_retriever(embedder, passages: list[str]) -> SimilarityRetriever:
    """Index the passages and return a top-k similarity retriever."""
    vectors = embedder.embed_documents(passages)
    chunks = [
        Document(page_content=t, metadata={"id": i})
        for i, t in enumerate(passages)
    ]
    store = FAISSVectorStore(embedding=embedder)
    store.add(chunks, embeddings=vectors)
    return SimilarityRetriever(store, top_k=TOP_K)


# --------------------------------------------------------------------------
# 3. Experiment — run the harness once per variant
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    df = pd.read_parquet(PASSAGES_PATH)
    passages = [str(r["passage"]).strip() for _, r in df.iterrows()][
        :N_PASSAGES
    ]

    # device="cpu": the local Ollama judge holds the shared GPU (5.6 GiB
    # here); bulk-embedding on CPU avoids CUDA OOM and leaves the GPU for the
    # judge calls that follow.
    bge = BGEEmbedding(model_name=BGE_MODEL_NAME, device="cpu")
    e5 = E5Embedding(model_name=E5_MODEL_NAME, device="cpu")

    t0 = time.perf_counter()
    retriever_bge = build_retriever(bge, passages)
    retriever_e5 = build_retriever(e5, passages)
    embed_s = time.perf_counter() - t0

    judge = LLMJudge()
    harness_kwargs = dict(
        judge=judge,
        faithfulness_metric=FaithfulnessMetric(judge),
        answer_relevance_metric=AnswerRelevanceMetric(judge),
        context_precision_metric=ContextPrecisionMetric(judge),
        context_recall_metric=ContextRecallMetric(judge),
        top_k=TOP_K,
    )

    t0 = time.perf_counter()
    results_bge = EvaluationHarness(
        retriever=retriever_bge, **harness_kwargs
    ).run(GOLDEN_QA)
    results_e5 = EvaluationHarness(
        retriever=retriever_e5, **harness_kwargs
    ).run(GOLDEN_QA)
    eval_s = time.perf_counter() - t0

    return {
        "passages": len(passages),
        "embed_s": embed_s,
        "eval_s": eval_s,
        "results_bge": results_bge,
        "results_e5": results_e5,
    }


# --------------------------------------------------------------------------
# 4. Demo
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 72)
    print("Lab 07-05 — EvaluationHarness A/B: BGE vs E5 embeddings")
    print(f"rag-mini {exp['passages']} passages, "
          f"{len(exp['results_bge'])} golden questions, top_k={TOP_K}")
    print("=" * 72)

    harness = EvaluationHarness(
        judge=None, retriever=None, faithfulness_metric=None,
        answer_relevance_metric=None, context_precision_metric=None,
        context_recall_metric=None, top_k=TOP_K,
    )
    agg_bge = harness.aggregate(exp["results_bge"])["overall"]
    agg_e5 = harness.aggregate(exp["results_e5"])["overall"]

    print(f"\n[1] Per-question rows — variant A (BGE):")
    harness.print_table(exp["results_bge"])
    print(f"\n[2] Per-question rows — variant B (E5):")
    harness.print_table(exp["results_e5"])

    print(f"\n[3] A/B means (RAGAS quartet):")
    print(f"    {'metric':<20} {'A (BGE)':>9} {'B (E5)':>9}  delta")
    for key in ("faithfulness", "answer_relevance",
                "context_precision", "context_recall"):
        diff = agg_bge[key] - agg_e5[key]
        print(f"    {key:<20} {agg_bge[key]:>9.3f} {agg_e5[key]:>9.3f}  "
              f"{diff:+.3f}")

    print(f"\n[4] Timing: embed {exp['embed_s']:.0f}s, harness both "
          f"{exp['eval_s']:.0f}s")

    print(f"\n[5] Takeaway")
    print("    A/B on embeddings: same corpus, same questions, same judge,")
    print("    same generator — only the embedder differs, so metric deltas")
    print("    are attributable. The harness gives the full RAGAS quartet")
    print("    instead of a single number: retrieval quality shows up in")
    print("    context precision/recall, generation quality in faithfulness")
    print("    and answer relevance. Reading the rows beats reading the")
    print("    means: one bad retrieval on a question explains a dip better")
    print("    than a mean ever will. Note the judge is the same Ollama")
    print("    model used for both variants — the comparison stays fair even")
    print("    though the judge is not interchangeable with a reference.")
    print("    This is the same harness you would wire into a golden-set")
    print("    regression (lab 04) for every embedding/retriever change.")


# --------------------------------------------------------------------------
# 5. Verification gate
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    harness = EvaluationHarness(
        judge=None, retriever=None, faithfulness_metric=None,
        answer_relevance_metric=None, context_precision_metric=None,
        context_recall_metric=None, top_k=TOP_K,
    )
    agg_bge = harness.aggregate(exp["results_bge"])["overall"]
    agg_e5 = harness.aggregate(exp["results_e5"])["overall"]

    for label, results in (("BGE", exp["results_bge"]),
                           ("E5", exp["results_e5"])):
        checks.append(
            (f"{label}: {len(results)} questions evaluated (== 6)",
             len(results) == len(GOLDEN_QA)))
        for key in ("faithfulness", "answer_relevance",
                    "context_precision", "context_recall"):
            checks.append((f"{label} {key} in [0, 1]",
                           0.0 <= harness.aggregate(results)["overall"][key]
                           <= 1.0))

    # The golden set must be answerable by both variants (validated at
    # authoring time: gold contained in a retrievable passage).
    checks.append(("BGE context_recall > 0 (golden set is answerable)",
                   agg_bge["context_recall"] > 0.0))
    checks.append(("E5 context_recall > 0 (golden set is answerable)",
                   agg_e5["context_recall"] > 0.0))

    # The A/B is meaningful only if the variants differ somewhere.
    differs = any(abs(agg_bge[k] - agg_e5[k]) > 0.01
                  for k in ("faithfulness", "answer_relevance",
                            "context_precision", "context_recall"))
    checks.append(("A/B meaningful: variants differ on >= 1 metric", differs))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
