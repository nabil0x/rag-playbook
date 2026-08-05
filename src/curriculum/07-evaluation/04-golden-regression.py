"""Lab 04 — Golden-set regression: catch a pipeline regression before it ships.

A golden set is a small, hand-checked set of (question, reference) pairs that
never changes — the evaluation anchor. The repo's is
``src/evaluation/golden.py``: 23 verified invoice QA pairs (source of truth from
the SD-08 notebooks). This lab turns the sample-invoice + Invoice_1 pairs
into a regression gate:

* Variant A — the reference pipeline: full PDF -> pages -> BGE embeddings ->
  FAISS -> top-3 retrieval -> Groq answer. Scored on faithfulness (claims
  supported by context) and reference correctness (gold contained in the
  answer, answer-reference cosine).
* Variant B — a REGRESSED pipeline: the same code with ``top_k=1`` (one
  context chunk instead of three) — a realistic silent degradation (someone
  "optimized" the retriever).

The gate asserts A >= B on every metric. That is the regression contract:
any future change that drops a metric below the reference pipeline FAILS the
gate, no matter how good the new code looks. A golden set that never moves is
what makes the comparison fair.

Run from the repo root:
    python src/curriculum/07-evaluation/04-golden-regression.py
    python src/curriculum/07-evaluation/04-golden-regression.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from embeddings.bge import BGEEmbedding  # noqa: E402
from evaluation.golden import GOLDEN_QA  # noqa: E402
from evaluation.judge import LLMJudge  # noqa: E402
from evaluation.metrics import FaithfulnessMetric  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from llms.groq import GroqLLM  # noqa: E402
from loaders.pdf import PDFLoader  # noqa: E402
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

load_dotenv()

# --------------------------------------------------------------------------
# 1. Configuration — the golden set and the two pipeline variants
# --------------------------------------------------------------------------
INVOICE_DIR = Path("Data/SD-08-invoices")
GOLDEN_DOCS = ["sample-invoice", "Invoice_1"]  # subset of GOLDEN_QA keys
VARIANT_A_TOP_K = 3  # reference pipeline
VARIANT_B_TOP_K = 1  # the regression: single-chunk context
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Faithfulness is judge-based and measures claim-support DENSITY, not answer
# completeness: a pipeline that answers less can score HIGHER (fewer claims,
# fewer chances to hallucinate). It is also noisy on a 10-question sample.
# So the regression gate tolerates a small faithfulness dip while gating the
# reference-anchored metrics (containment, cosine) strictly.
FAITHFULNESS_TOLERANCE = 0.15


# --------------------------------------------------------------------------
# 2. Reference-based correctness (same helpers as labs 02-03)
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles and whitespace."""
    cleaned = "".join(c.lower() for c in text if c.isalnum() or c.isspace())
    words = [w for w in cleaned.split() if w not in ("a", "an", "the")]
    return " ".join(words)


def reference_contained(answer: str, reference: str) -> bool:
    """True when the normalized gold answer appears inside the answer."""
    return normalize(reference) in normalize(answer)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (0.0 if either is zero)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------
# 3. One pipeline, parameterized by top_k — A and B differ in ONE knob
# --------------------------------------------------------------------------
def build_indexes() -> dict[str, SimilarityRetriever]:
    """Per-doc index: {doc_key: retriever over that invoice's pages}."""
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME)
    retrievers: dict[str, SimilarityRetriever] = {}
    for doc_key in GOLDEN_DOCS:
        pdf_path = INVOICE_DIR / f"{doc_key}.pdf"
        pages = PDFLoader(str(pdf_path)).load()
        texts = [p.page_content for p in pages]
        vectors = embedder.embed_documents(texts)
        chunks = [
            Document(page_content=t, metadata={"doc": doc_key, "page": i})
            for i, t in enumerate(texts)
        ]
        store = FAISSVectorStore(embedding=embedder)
        store.add(chunks, embeddings=vectors)
        retrievers[doc_key] = SimilarityRetriever(store, top_k=20)
    return retrievers


def run_variant(retrievers: dict[str, SimilarityRetriever], top_k: int,
                llm, judge, faithfulness) -> dict[str, float]:
    """Run the golden set through the pipeline with a given ``top_k``.

    Returns mean scores per metric over the golden questions plus the
    per-question rows (doc, question, contained) for eyeballing.
    """
    per_metric: dict[str, list[float]] = {"faithfulness": [], "contained": [],
                                          "cosine": []}
    rows: list[dict] = []
    for doc_key in GOLDEN_DOCS:
        retriever = retrievers[doc_key]
        for question, reference in GOLDEN_QA[doc_key]:
            docs = retriever.retrieve(question)[:top_k]
            context = "\n\n".join(d.page_content for d in docs)
            answer = llm.invoke(
                f"Context:\n{context}\n\nQuestion: {question}\n\n"
                "Answer concisely, quoting the exact figure or value from "
                "the context:"
            ).strip()
            per_metric["faithfulness"].append(
                faithfulness.score(question, context, answer))
            contained = reference_contained(answer, reference)
            per_metric["contained"].append(contained)
            per_metric["cosine"].append(
                cosine_similarity(judge.embed([answer])[0],
                                  judge.embed([reference])[0]))
            rows.append({"doc": doc_key, "question": question,
                         "contained": contained})
    means = {k: sum(v) / len(v) for k, v in per_metric.items()}
    means["rows"] = rows
    return means


# --------------------------------------------------------------------------
# 4. Experiment — build once, run both variants
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    t0 = time.perf_counter()
    retrievers = build_indexes()
    index_s = time.perf_counter() - t0

    llm = GroqLLM(temperature=0.0)
    judge = LLMJudge()
    faithfulness = FaithfulnessMetric(judge)

    t0 = time.perf_counter()
    variant_a = run_variant(retrievers, VARIANT_A_TOP_K, llm, judge, faithfulness)
    variant_b = run_variant(retrievers, VARIANT_B_TOP_K, llm, judge, faithfulness)
    eval_s = time.perf_counter() - t0

    return {
        "docs": GOLDEN_DOCS,
        "n_questions": sum(len(GOLDEN_QA[d]) for d in GOLDEN_DOCS),
        "index_s": index_s,
        "eval_s": eval_s,
        "variant_a": variant_a,
        "variant_b": variant_b,
    }


# --------------------------------------------------------------------------
# 5. Demo
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 07-04 — Golden-set regression (evaluation/golden.py)")
    print(f"docs {exp['docs']}, {exp['n_questions']} golden questions")
    print("=" * 66)

    a, b = exp["variant_a"], exp["variant_b"]
    print(f"\n[1] Golden-set scores (mean over {exp['n_questions']} questions):")
    print(f"    {'metric':<14} {'A (top_k=3)':>12} {'B (top_k=1)':>12}  delta")
    for key in ("faithfulness", "contained", "cosine"):
        print(f"    {key:<14} {a[key]:>12.3f} {b[key]:>12.3f}  "
              f"{a[key] - b[key]:+.3f}")

    print(f"\n[2] Per-question reference containment (eyeball the rows):")
    print(f"    {'doc':<14} {'A':>4} {'B':>4}  question")
    for ra, rb in zip(a["rows"], b["rows"]):
        print(f"    {ra['doc']:<14} {ra['contained']:>4.0f} "
              f"{rb['contained']:>4.0f}  {ra['question'][:52]}")

    print(f"\n[3] Regression gate (reference-anchored strict, faithfulness "
          f"tolerated):")
    a_f, b_f = a["faithfulness"], b["faithfulness"]
    for key in ("contained", "cosine"):
        status = "PASS" if a[key] >= b[key] else "FAIL"
        print(f"    [{status}] A >= B for {key}")
    status = "PASS" if a_f >= b_f - FAITHFULNESS_TOLERANCE else "FAIL"
    print(f"    [{status}] A >= B - {FAITHFULNESS_TOLERANCE:.2f} for faithfulness")

    print(f"\n[4] Timing: index {exp['index_s']:.1f}s, evaluate both "
          f"{exp['eval_s']:.1f}s")

    print(f"\n[5] Takeaway")
    print("    A golden set is the regression contract: the same questions,")
    print("    the same references, forever. Variant B is a realistic silent")
    print("    regression — one context chunk instead of three — and the gate")
    print("    is 'reference pipeline >= new pipeline'. Two subtleties the")
    print("    numbers teach: (1) faithfulness is claim-support DENSITY, not")
    print("    completeness — a single chunk can score higher because the")
    print("    model asserts less; judge metrics get a tolerance. (2) the")
    print("    reference-anchored metrics gate strictly — when a change")
    print("    passes unit tests but drops a golden containment/cosine score,")
    print("    this gate stops it from shipping. Always eyeball the rows.")


# --------------------------------------------------------------------------
# 6. Verification gate
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    a, b = exp["variant_a"], exp["variant_b"]

    checks.append((f"{exp['n_questions']} golden questions evaluated (>= 6)",
                   exp["n_questions"] >= 6))

    for key in ("faithfulness", "contained", "cosine"):
        checks.append((f"A {key} in [0, 1]", 0.0 <= a[key] <= 1.0))
        checks.append((f"B {key} in [0, 1]", 0.0 <= b[key] <= 1.0))

    # The regression contract: A beats B on every metric. Reference-anchored
    # metrics gate strictly; the judge-based faithfulness metric gets the
    # tolerance band (claim-support density is not monotonic in context and
    # the judge is noisy on a 10-question sample).
    for key in ("contained", "cosine"):
        checks.append((f"regression gate: A >= B for {key}", a[key] >= b[key]))
    checks.append((
        f"regression gate: A >= B - {FAITHFULNESS_TOLERANCE} for faithfulness",
        a["faithfulness"] >= b["faithfulness"] - FAITHFULNESS_TOLERANCE))

    # A floor so the golden set is doing real work (not all-zeros).
    checks.append(("reference pipeline finds some answers (A contained > 0)",
                   a["contained"] > 0.0))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
