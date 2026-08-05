"""Lab 03 — BGE vs E5: head-to-head retrieval comparison.

The embedding model is the block that decides *which* passages a query can
ever reach: retrieval quality is bounded by embedding quality before any
retriever, prompt, or LLM gets a vote. This lab runs two local bi-encoder
models head to head on the same corpus and the same 60 real Q/A pairs:

* BGE — ``BAAI/bge-base-en-v1.5`` via ``src/embeddings/bge.BGEEmbedding``. bge
  models are *trained* for cosine similarity: embeddings are produced with
  ``normalize_embeddings=True``, so every vector leaves the model unit-length.
* E5 — ``intfloat/multilingual-e5-base`` via ``src/embeddings/e5.E5Embedding``.
  E5 models are trained with instruction prefixes: the wrapper prepends
  ``"query: "`` to questions and ``"passage: "`` to passages automatically.
  Unlike BGE, the E5 wrapper does **not** normalize the output vectors.

Two caveats the lab makes visible:

1. **Prefixes matter.** ``multilingual-e5-base`` was trained on
   ``query: ...`` / ``passage: ...`` pairs. Embedding a question without the
   ``query: `` prefix would put it in a different region of the space than
   the prefixed passages — the wrapper's auto-prefixing is exactly why it is
   used here.
2. **Normalization matters for fair cosine.** Cosine similarity is
   scale-invariant, so raw vs normalized vectors give identical *rankings*
   — but a raw E5 vector is several times longer than a unit vector, and
   raw dot products are not comparable to BGE's normalized ones. For a fair
   comparison (and to report mean/median similarity on the same scale), both
   models' matrices are L2-normalized here with numpy before scoring.

Metric: **answer-containment recall@1** — the fraction of questions whose
gold answer string (case-insensitive, whitespace-stripped) appears inside the
single most-similar passage. No LLM, no judge, no API: pure retrieval
measurement on real Q/A pairs.

Run from the repo root:
    python src/curriculum/02-embeddings/03-model-comparison.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Silence the "Loading weights" progress bar (transformers honors
# HF_HUB_DISABLE_PROGRESS_BARS, read at huggingface_hub import time — set it
# before any third-party import).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import pandas as pd

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/02-embeddings/03-model-comparison.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embeddings.bge import BGEEmbedding  # noqa: E402
from embeddings.e5 import E5Embedding  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the comparison
# --------------------------------------------------------------------------
NUM_PASSAGES = 400  # first N passages of the corpus (embedding budget)
NUM_QUESTIONS = 60  # first N test questions with non-empty answers
PREVIEW = 170  # max characters shown around the answer in the example table
MODEL_NAME = "BAAI/bge-base-en-v1.5"

PASSAGES_PARQUET = "Data/corpus/rag-mini-wikipedia/passages.parquet"
TEST_PARQUET = "Data/corpus/rag-mini-wikipedia/test.parquet"


# --------------------------------------------------------------------------
# 2. Load — deterministic subsets of the corpus
# --------------------------------------------------------------------------
def load_subsets() -> tuple[list[str], list[tuple[str, str]]]:
    """Return (passages, qa_pairs) with zero randomness.

    Passages: first ``NUM_PASSAGES`` rows of ``passages.parquet``. Questions:
    first ``NUM_QUESTIONS`` rows of ``test.parquet`` whose answer is a
    non-empty string, in file order.
    """
    passages_df = pd.read_parquet(PASSAGES_PARQUET)
    passages = [str(p) for p in passages_df["passage"].head(NUM_PASSAGES).tolist()]

    test_df = pd.read_parquet(TEST_PARQUET)
    qa_pairs: list[tuple[str, str]] = []
    for question, answer in zip(test_df["question"], test_df["answer"]):
        if isinstance(answer, str) and answer.strip():
            qa_pairs.append((str(question), answer.strip()))
        if len(qa_pairs) >= NUM_QUESTIONS:
            break
    return passages, qa_pairs


# --------------------------------------------------------------------------
# 3. Score — embed once per model, cosine via normalized dot products
# --------------------------------------------------------------------------
def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization to unit length."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def score_matrix(embedder: object, passages: list[str],
                 questions: list[str]) -> np.ndarray:
    """Return the (n_questions, n_passages) cosine-similarity matrix.

    E5's wrapper does not normalize its output (unlike BGE's); both matrices
    are L2-normalized here so that (a) dot product equals cosine similarity
    and (b) the mean/median similarity reported for each model lives on the
    same unit-scale.
    """
    passage_vecs = np.asarray(embedder.embed_documents(passages), dtype=np.float32)
    query_vecs = np.asarray([embedder.embed_query(q) for q in questions],
                            dtype=np.float32)
    return l2_normalize(query_vecs) @ l2_normalize(passage_vecs).T


def contains_answer(passage: str, answer: str) -> bool:
    """Answer containment: case-insensitive substring on stripped strings."""
    return answer.lower() in passage.lower()


class ModelScore:
    """Per-question retrieval outcomes for one embedding model."""

    def __init__(self, name: str, cosine: np.ndarray,
                 passages: list[str], qa_pairs: list[tuple[str, str]]) -> None:
        self.name = name
        self.cosine = cosine
        self.top1_idx = cosine.argmax(axis=1)
        rows = np.arange(cosine.shape[0])
        self.top1_sim = cosine[rows, self.top1_idx]
        self.hits = [contains_answer(passages[i], answer)
                     for i, (_, answer) in zip(self.top1_idx, qa_pairs)]

    def recall_at_1(self) -> float:
        return float(np.mean(self.hits))

    def mean_sim(self) -> float:
        return float(np.mean(self.top1_sim))

    def median_sim(self) -> float:
        return float(np.median(self.top1_sim))


def escape(s: str) -> str:
    """Make newlines visible so passage previews stay on one line."""
    return s.replace("\n", "\\n")


def passage_window(passage: str, answer: str, width: int = PREVIEW) -> str:
    """Preview the passage, centered on the answer span when it is present."""
    idx = passage.lower().find(answer.lower())
    if idx < 0:
        return escape(passage[:width])
    start = max(0, idx - width // 3)
    end = min(len(passage), idx + len(answer) + width // 2)
    preview = escape(passage[start:end])
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(passage) else ""
    return f"{prefix}{preview}{suffix}"


def disagreement_indices(bge: ModelScore, e5: ModelScore, limit: int) -> list[int]:
    """Question indices where the two models disagree on the top-1 hit."""
    return [i for i, (hb, he) in enumerate(zip(bge.hits, e5.hits))
            if hb != he][:limit]


# --------------------------------------------------------------------------
# 4. Print the artifact — runnable demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    passages, qa_pairs = load_subsets()
    questions = [q for q, _ in qa_pairs]

    print("=" * 66)
    print("Lab 03 — BGE vs E5: head-to-head retrieval comparison")
    print("=" * 66)

    # --- [1] Setup -------------------------------------------------------
    print(f"\n[1] Setup")
    models = [("BGE", BGEEmbedding(model_name=MODEL_NAME)), ("E5 ", E5Embedding())]
    dims: dict[str, int] = {}
    for label, embedder in models:
        probe = embedder.embed_query(questions[0])
        dims[label.strip()] = len(probe)
    print(f"    passages : {len(passages)} (first {NUM_PASSAGES} of rag-mini-wikipedia)")
    print(f"    questions: {len(qa_pairs)} (first {NUM_QUESTIONS} with non-empty answers)")
    print(f"    BGE : BAAI/bge-base-en-v1.5            dim={dims['BGE']}  "
          f"(wrapper normalizes output)")
    print(f"    E5  : intfloat/multilingual-e5-base   dim={dims['E5']}  "
          f"(wrapper auto-prefixes 'query:'/'passage:', no normalization)")
    print("    both matrices L2-normalized here so cosine is a fair dot product")

    # --- [2]-[4] Run each model -------------------------------------------
    results: dict[str, ModelScore] = {}
    for label, embedder in models:
        t0 = time.perf_counter()
        cosine = score_matrix(embedder, passages, questions)
        elapsed = time.perf_counter() - t0
        results[label.strip()] = ModelScore(label.strip(), cosine,
                                            passages, qa_pairs)
        print(f"    embedded {len(passages)} passages + {len(questions)} queries "
              f"with {label.strip()} in {elapsed:.1f}s")

    # --- [2] Recall@1 -----------------------------------------------------
    bge, e5 = results["BGE"], results["E5"]
    print("\n[2] Recall@1 — gold answer found in the top-1 passage")
    print(f"    {'model':<10}{'recall@1':>10}{'hits':>7}{'misses':>9}")
    for score in (bge, e5):
        hits = sum(score.hits)
        print(f"    {score.name:<10}{score.recall_at_1():>10.3f}"
              f"{hits:>7}{len(score.hits) - hits:>9}")

    # --- [3] Similarity of the retrieved (top-1) passage -------------------
    print("\n[3] Similarity of the retrieved passage (top-1 cosine)")
    print(f"    {'model':<10}{'mean':>8}{'median':>9}")
    for score in (bge, e5):
        print(f"    {score.name:<10}{score.mean_sim():>8.3f}"
              f"{score.median_sim():>9.3f}")

    # --- [4] Side-by-side disagreements ------------------------------------
    print("\n[4] Example retrievals where the models disagree")
    picked = disagreement_indices(bge, e5, limit=3)
    if not picked:
        print("    No disagreement found in this subset — both models agree "
              "on every question.")
    for i in picked:
        question, answer = qa_pairs[i]
        print(f"\n    Q: {escape(question[:110])}")
        print(f"       gold answer: {answer!r}")
        for score in (bge, e5):
            verdict = "HIT " if score.hits[i] else "MISS"
            snippet = passage_window(passages[score.top1_idx[i]], answer)
            print(f"       {score.name:<4}{verdict} sim={score.top1_sim[i]:.3f}  "
                  f"{snippet[:PREVIEW + 8]}{'...' if len(snippet) > PREVIEW + 8 else ''}")

    # --- [5] Takeaway ------------------------------------------------------
    print("\n[5] Takeaway")
    winner = "BGE" if bge.recall_at_1() > e5.recall_at_1() else \
        "E5" if e5.recall_at_1() > bge.recall_at_1() else "neither"
    print(f"    On {len(qa_pairs)} real questions over {len(passages)} passages, "
          f"{winner} wins recall@1 "
          f"(BGE {bge.recall_at_1():.3f} vs E5 {e5.recall_at_1():.3f}).")
    print("    Recall@1 only measures whether the top passage *contains* the")
    print("    gold answer — a coarse proxy (many answers here are 'yes'/'no').")
    print("    E5's top-1 similarities sit higher (median "
          f"{e5.median_sim():.3f} vs {bge.median_sim():.3f}) even with both")
    print("    matrices unit-length: absolute similarity does NOT transfer")
    print("    across models, only rankings do. And E5 needs its 'query:'/")
    print("    'passage:' prefixes plus L2 normalization to be compared")
    print("    fairly against BGE at all.")
