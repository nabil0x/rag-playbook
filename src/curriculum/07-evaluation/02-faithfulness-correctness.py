"""Lab 02 — Faithfulness and correctness: two halves of answer quality.

Retrieval metrics (lab 01) stop at the context window. This lab scores what
happens after generation — an answer can be factually right but unsupported by
the retrieved context (hallucination), or supported but wrong against the
reference answer (a retrieval gap that the generator faithfully amplified).

* Faithfulness — does the answer follow ONLY from the retrieved context?
  ``FaithfulnessMetric`` (evaluation/metrics.py) asks the local LLM judge to
  split the answer into atomic claims and mark each as supported/unsupported
  by the context. Score = supported / total claims. This is the hallucination
  guard: a high score means the generator stayed inside its evidence. It
  needs claim-rich answers — a terse "yes" has zero claims to check, which is
  why the generator is prompted to answer in complete sentences.
* Correctness — does the answer match the gold reference? Two reference-based
  measures: reference containment (the normalized gold answer appears inside
  the normalized answer — the only exact-style check that survives
  elaboration on a short-answer gold set) and embedding cosine between answer
  and reference (semantic tolerance for rephrasing).

Pipeline: the 3,200 rag-mini-wikipedia passages are embedded locally (BGE,
FAISS), each sampled question retrieves its top-3 context, Groq generates an
elaborated answer, the Ollama judge scores faithfulness, and both correctness
measures are computed against the gold answer.

Run from the repo root:
    python src/curriculum/07-evaluation/02-faithfulness-correctness.py
    python src/curriculum/07-evaluation/02-faithfulness-correctness.py --verify
"""

from __future__ import annotations

import os
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
from evaluation.judge import LLMJudge  # noqa: E402
from evaluation.metrics import FaithfulnessMetric  # noqa: E402
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
N_SAMPLE = 15  # judge-evaluated questions (Ollama is local and slow)
TOP_K = 3  # context chunks fed to the generator
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"


# --------------------------------------------------------------------------
# 2. Load — passages + test QA
# --------------------------------------------------------------------------
def load_passages(path: Path) -> tuple[list[str], list[str]]:
    """Return (doc_texts, doc_ids) for every passage."""
    df = pd.read_parquet(path)
    texts = [str(row["passage"]).strip() for _, row in df.iterrows()]
    ids = [str(i) for i in range(len(df))]
    return texts, ids


def load_test_qa(path: Path) -> list[dict]:
    """Return [{"question": ..., "answer": ...}] from test.parquet."""
    df = pd.read_parquet(path)
    return [{"question": r["question"], "answer": r["answer"]}
            for _, r in df.iterrows()]


# --------------------------------------------------------------------------
# 3. Reference-based correctness (no LLM — deterministic)
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles and whitespace."""
    cleaned = "".join(c.lower() for c in text if c.isalnum() or c.isspace())
    words = [w for w in cleaned.split() if w not in ("a", "an", "the")]
    return " ".join(words)


def reference_contained(answer: str, reference: str) -> bool:
    """True when the normalized gold answer appears inside the answer.

    The rag-mini gold answers are terse ("yes", "no", a number) while the
    generator is asked to elaborate, so exact equality can never hold.
    Containment is the exact-style check that survives elaboration: the
    gold's normalized words must appear in the answer's normalized words.
    """
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
# 4. Experiment — retrieve, generate, score
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
    faithfulness = FaithfulnessMetric(judge)

    rows: list[dict] = []
    t0 = time.perf_counter()
    for item in test_qa[:N_SAMPLE]:
        question, reference = item["question"], item["answer"]
        context_docs = retriever.retrieve(question)
        context = "\n\n".join(d.page_content for d in context_docs)
        answer = llm.invoke(
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "Answer in one or two complete sentences, stating the key "
            "fact(s) from the context:"
        ).strip()

        rows.append({
            "question": question,
            "reference": reference,
            "answer": answer,
            "faithfulness": faithfulness.score(question, context, answer),
            "contained": reference_contained(answer, reference),
            "cosine": cosine_similarity(
                judge.embed([answer])[0], judge.embed([reference])[0]
            ),
        })
    llm_s = time.perf_counter() - t0

    def mean(key: str) -> float:
        vals = [r[key] for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "rows": rows,
        "indexed": len(passages),
        "embed_s": embed_s,
        "llm_s": llm_s,
        "metrics": {
            "faithfulness": mean("faithfulness"),
            "containment": mean("contained"),
            "answer_cosine": mean("cosine"),
        },
    }


# --------------------------------------------------------------------------
# 5. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 07-02 — Faithfulness and correctness")
    print(f"rag-mini {exp['indexed']} passages, {len(exp['rows'])} questions")
    print("=" * 66)

    m = exp["metrics"]
    print(f"\n[1] Mean scores over {len(exp['rows'])} questions:")
    print(f"    faithfulness (claims supported by context): {m['faithfulness']:.3f}")
    print(f"    reference containment                     : {m['containment']:.3f}")
    print(f"    answer-reference cosine                   : {m['answer_cosine']:.3f}")

    print("\n[2] Three example rows:")
    for row in exp["rows"][:3]:
        print(f"    Q: {row['question'][:60]}")
        print(f"       gold={row['reference'][:40]!r} | "
              f"faith={row['faithfulness']:.2f} | "
              f"contained={row['contained']} | cos={row['cosine']:.2f}")

    print(f"\n[3] Timing: embed {exp['indexed']} passages {exp['embed_s']:.1f}s, "
          f"LLM+judge {len(exp['rows'])} Q {exp['llm_s']:.1f}s")

    print(f"\n[4] Takeaway")
    print("    Faithfulness and correctness are orthogonal: an answer can be")
    print("    perfectly faithful to a context that is itself irrelevant")
    print("    (high faithfulness, low containment), or wrong while fully")
    print("    supported by the evidence (the generator faithfully amplified")
    print("    a retrieval miss). Two operational notes: faithfulness needs")
    print("    claim-rich answers — a terse 'yes' has zero claims and scores")
    print("    0.0 by construction — and short-answer gold sets force you to")
    print("    soften exact-match into containment or cosine. High faithfulness")
    print("    + low correctness points at retrieval, not the LLM.")


# --------------------------------------------------------------------------
# 6. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    m = exp["metrics"]
    n = len(exp["rows"])

    checks.append((f"{N_SAMPLE} questions evaluated (>= 10)", n >= 10))
    checks.append(("faithfulness mean in [0, 1]", 0.0 <= m["faithfulness"] <= 1.0))
    checks.append(("faithfulness > 0 (elaborated answers carry claims)",
                   m["faithfulness"] > 0.0))
    checks.append(("reference containment in [0, 1]",
                   0.0 <= m["containment"] <= 1.0))
    checks.append(("answer cosine in [-1, 1]", -1.0 <= m["answer_cosine"] <= 1.0))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
