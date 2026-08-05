"""Lab 03 — LLM-as-judge vs reference: is the judge trustworthy?

LLM-as-judge is cheap and scales, but it is a MODEL making a judgment — it can
be sycophantic, pattern-match to phrasing, or disagree with the ground truth
in systematic ways. This lab applies the repo's standing rule (AGENTS.md #6):
before an LLM judge is trusted, cross-check it against a reference-based
metric and measure agreement with Cohen's kappa.

Setup: 15 questions from rag-mini-wikipedia. Each answer is judged "correct?"
by two independent raters:

* LLM judge — the Ollama qwen2.5-coder model, shown the question, the answer
  and the gold reference, asked for a binary correct/incorrect verdict
  (temperature 0). This is the judge whose trustworthiness we are auditing.
* Reference-based rater — the embedding cosine between the answer and the
  gold reference, binarized at a threshold. Deterministic, no model opinion,
  only geometric similarity to the human gold.

Cohen's kappa (from ``EvaluationHarness.kappa``) then answers: beyond the
agreement you would expect by chance, how much do the two raters agree? The
Landis-Koch scale turns the number into words (0.61+ substantial). A low
kappa means the LLM judge is NOT a drop-in replacement for the reference
metric — exactly the failure this check exists to catch.

Run from the repo root:
    python src/curriculum/07-evaluation/03-judge-vs-reference.py
    python src/curriculum/07-evaluation/03-judge-vs-reference.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from embeddings.bge import BGEEmbedding  # noqa: E402
from evaluation.harness import EvaluationHarness  # noqa: E402
from evaluation.judge import LLMJudge  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from llms.groq import GroqLLM  # noqa: E402
from retrieval.similarity import SimilarityRetriever  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

load_dotenv()

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
RAG_MINI = Path("Data/corpus/rag-mini-wikipedia")
PASSAGES_PATH = RAG_MINI / "passages.parquet"
TEST_PATH = RAG_MINI / "test.parquet"
N_SAMPLE = 15
TOP_K = 3
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
COSINE_THRESHOLD = 0.5  # reference rater: cosine >= this => "correct"


# --------------------------------------------------------------------------
# 2. Load
# --------------------------------------------------------------------------
def load_passages(path: Path) -> tuple[list[str], list[str]]:
    """Return (doc_texts, doc_ids) for every passage."""
    df = pd.read_parquet(path)
    texts = [str(row["passage"]).strip() for _, row in df.iterrows()]
    return texts, [str(i) for i in range(len(df))]


def load_test_qa(path: Path) -> list[dict]:
    """Return [{"question": ..., "answer": ...}] from test.parquet."""
    df = pd.read_parquet(path)
    return [{"question": r["question"], "answer": r["answer"]}
            for _, r in df.iterrows()]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (0.0 if either is zero)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------
# 3. The two raters
# --------------------------------------------------------------------------
def judge_verdict(judge: LLMJudge, question: str, answer: str,
                  reference: str) -> int:
    """LLM judge: is the answer correct? -> 1 correct, 0 incorrect.

    JSON schema ``{"correct": true|false}``; accepts bool or "yes"/"true"
    strings from the model.
    """
    instruction = (
        "You are an answer-quality judge. Decide whether the answer is "
        "CORRECT given the gold reference. Output ONLY JSON."
    )
    prompt = (
        f"Question: {question}\n\nAnswer: {answer}\n\n"
        f"Gold reference: {reference}\n\n"
        'Return JSON: {"correct": true} or {"correct": false}'
    )
    result = judge.judge(instruction, prompt)
    raw = result.get("correct", False)
    if isinstance(raw, bool):
        return 1 if raw else 0
    return 1 if str(raw).strip().lower() in ("yes", "true", "1", "correct") else 0


def reference_verdict(cosine: float) -> int:
    """Reference-based rater: cosine against gold >= threshold -> 1 else 0."""
    return 1 if cosine >= COSINE_THRESHOLD else 0


# --------------------------------------------------------------------------
# 4. Experiment
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    passages, passage_ids = load_passages(PASSAGES_PATH)
    test_qa = load_test_qa(TEST_PATH)

    # device="cpu": the local Ollama judge holds the shared GPU (5.6 GiB
    # here); bulk-embedding 3200 passages on CPU avoids CUDA OOM and leaves
    # the GPU for the judge calls that follow.
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME, device="cpu")
    t0 = time.perf_counter()
    vectors = embedder.embed_documents(passages)
    embed_s = time.perf_counter() - t0

    chunks = [
        Document(page_content=t, metadata={"id": cid})
        for t, cid in zip(passages, passage_ids)
    ]
    store = FAISSVectorStore(embedding=embedder)
    store.add(chunks, embeddings=vectors)
    retriever = SimilarityRetriever(store, top_k=TOP_K)

    llm = GroqLLM(temperature=0.0)
    judge = LLMJudge()

    judge_labels: list[int] = []
    reference_labels: list[int] = []
    rows: list[dict] = []

    t0 = time.perf_counter()
    for item in test_qa[:N_SAMPLE]:
        question, reference = item["question"], item["answer"]
        context = "\n\n".join(d.page_content
                              for d in retriever.retrieve(question))
        answer = llm.invoke(
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "Answer in one or two complete sentences, stating the key "
            "fact(s) from the context:"
        ).strip()

        cosine = cosine_similarity(judge.embed([answer])[0],
                                   judge.embed([reference])[0])
        jv = judge_verdict(judge, question, answer, reference)
        rv = reference_verdict(cosine)
        judge_labels.append(jv)
        reference_labels.append(rv)
        rows.append({
            "question": question,
            "reference": reference,
            "answer": answer,
            "cosine": cosine,
            "judge": jv,
            "reference": rv,
        })
    llm_s = time.perf_counter() - t0

    kappa = EvaluationHarness.kappa(judge_labels, reference_labels)
    n = len(judge_labels)
    agreement = sum(1 for a, b in zip(judge_labels, reference_labels)
                    if a == b) / n if n else 0.0
    return {
        "rows": rows,
        "indexed": len(passages),
        "embed_s": embed_s,
        "llm_s": llm_s,
        "kappa": kappa,
        "agreement": agreement,
        "judge_labels": judge_labels,
        "reference_labels": reference_labels,
    }


def landis_koch(kappa: float) -> str:
    """Human words for a kappa value (Landis & Koch, 1977)."""
    if kappa < 0.0:
        return "poor (worse than chance)"
    if kappa < 0.21:
        return "slight"
    if kappa < 0.41:
        return "fair"
    if kappa < 0.61:
        return "moderate"
    if kappa < 0.81:
        return "substantial"
    return "almost perfect"


# --------------------------------------------------------------------------
# 5. Demo
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 07-03 — LLM-as-judge vs reference: Cohen's kappa")
    print(f"rag-mini {exp['indexed']} passages, {len(exp['rows'])} questions")
    print("=" * 66)

    print("\n[1] Verdicts per question (judge vs reference-based):")
    for row in exp["rows"]:
        mark = "AGREE" if row["judge"] == row["reference"] else "DISAGREE"
        print(f"    judge={row['judge']} ref={row['reference']} "
              f"cos={row['cosine']:.2f} [{mark}] {row['question'][:50]}")

    k = exp["kappa"]
    print(f"\n[2] Agreement: observed {exp['agreement']:.3f}, "
          f"Cohen's kappa = {k:.3f} -> {landis_koch(k)}")

    print(f"\n[3] Takeaway")
    print("    Observed agreement alone flatters: two raters who both say")
    print("    'correct' most of the time agree by chance. Kappa subtracts")
    print("    that expected agreement, so it is the honest number. A kappa")
    print("    below ~0.6 means the LLM judge is not interchangeable with")
    print("    the reference metric — trust it only after this cross-check,")
    print("    and prefer it as a supplement, not a replacement. Note the")
    print("    threshold choice (cosine >= 0.5) also moves the reference")
    print("    rater's side of the table: kappa audits the PAIR.")


# --------------------------------------------------------------------------
# 6. Verification gate
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    n = len(exp["rows"])
    k = exp["kappa"]

    checks.append((f"{N_SAMPLE} questions judged (>= 10)", n >= 10))
    checks.append(("kappa is finite and in [-1, 1]",
                   -1.0 <= k <= 1.0 and k == k))
    checks.append(("both verdict lists have one label per question",
                   len(exp["judge_labels"]) == n
                   and len(exp["reference_labels"]) == n))
    checks.append(("reference rater found some correct answers (sum > 0)",
                   sum(exp["reference_labels"]) > 0))
    checks.append(("LLM judge produced some incorrect verdicts (sum < n) "
                   "- judge is not a yes-machine",
                   sum(exp["judge_labels"]) < n))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
