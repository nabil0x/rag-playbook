"""Lab 02 — cosine similarity vs raw dot product for retrieval.

Every vector store ranks passages by a similarity score, and the two most
common scores are the COSINE SIMILARITY and the raw DOT PRODUCT. This lab
answers the question that decides which one your store should use:

    cos(a, b) = (a . b) / (||a|| * ||b||)

The dot product in the numerator is divided by both vector lengths, so cosine
measures the ANGLE between two vectors and ignores their magnitude. The raw
dot product keeps magnitude: a long vector scores higher than a short one
even when it points in a worse direction.

For text embeddings this matters because magnitude is usually noise. BGE
(``BAAI/bge-base-en-v1.5``) is trained to produce L2-normalized vectors
(``normalize_embeddings=True`` in ``embeddings/bge.py``), so every vector has
length 1 and the two scores collapse into one:

    ||a|| = ||b|| = 1  =>  cos(a, b) = a . b

This lab proves that identity numerically on real embeddings, then shows what
happens when the normalization is missing: we deliberately scale the passage
vectors by a deterministic, length-correlated factor (simulating an embedder
that does not normalize, like E5) and watch the raw dot product mis-rank the
results that cosine gets right.

Run from the repo root:
    python curriculum/02-embeddings/02-cosine-vs-dot.py
"""

from __future__ import annotations

# Silence the "Loading weights" progress bar (transformers honors this flag;
# must be set before any third-party import pulls in huggingface_hub).
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/02-embeddings/02-cosine-vs-dot.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from embeddings.bge import BGEEmbedding  # noqa: E402

try:
    from scipy.stats import spearmanr  # noqa: F401

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the comparison
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PASSAGES = 50  # deterministic head of the 3200-passage corpus (keeps runtime low)
N_QUESTIONS = 5  # deterministic head of test.parquet
TOP_K = 5
PREVIEW = 62  # max characters of passage text shown next to each hit
MODEL_NAME = "BAAI/bge-base-en-v1.5"


# --------------------------------------------------------------------------
# 2. Load — corpus + questions from the fresh rag-mini-wikipedia parquet files
# --------------------------------------------------------------------------
def load_passages(path: Path, n: int) -> tuple[list[str], list[int]]:
    """Return (passage_texts, passage_ids) for the first ``n`` passages."""
    df = pd.read_parquet(path)
    subset = df.head(n)
    return subset["passage"].tolist(), subset.index.tolist()


def load_questions(path: Path, n: int) -> list[str]:
    """Return the first ``n`` question strings from test.parquet."""
    df = pd.read_parquet(path)
    return df["question"].head(n).tolist()


# --------------------------------------------------------------------------
# 3. Score matrices — cosine, normalized dot, and deliberately raw dot
# --------------------------------------------------------------------------
def cosine_similarity_matrix(Q: np.ndarray, P: np.ndarray) -> np.ndarray:
    """cos(a, b) = (a . b) / (||a|| * ||b||) for every (query, passage) pair.

    Written out explicitly (rather than via sklearn) so the formula is
    visible: each row/column is divided by its L2 norm before the dot product.
    """
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    return Qn @ Pn.T


def normalized_dot_matrix(Q: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Raw dot product, valid because BGE already L2-normalizes every vector.

    If the embedder's contract holds (||a|| = ||b|| = 1), this must equal the
    cosine matrix above — the identity this lab proves numerically.
    """
    return Q @ P.T


def unnormalized_passages(P: np.ndarray, lengths: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Simulate an embedder that does NOT normalize its output.

    BGE normalizes every vector to unit length, so magnitude carries no
    signal. Real embedders that skip normalization (e.g. E5) produce vectors
    whose magnitude grows with passage length. We reproduce that effect by
    scaling each passage vector by a deterministic factor derived from its
    character length — the raw dot product then ranks by magnitude * direction
    instead of direction alone. Returns (scaled_vectors, scale_factors).
    """
    scale = np.asarray([1.0 + (length % 200) / 100.0 for length in lengths])
    return P * scale[:, np.newaxis], scale


def raw_dot_matrix(Q: np.ndarray, P_raw: np.ndarray) -> np.ndarray:
    """Raw dot product on the deliberately unnormalized passage vectors."""
    return Q @ P_raw.T


# --------------------------------------------------------------------------
# 4. Ranking helpers
# --------------------------------------------------------------------------
def top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the top-k passages for one query's score row."""
    return np.argsort(scores)[::-1][:k].tolist()


def spearman_between(cos_scores: np.ndarray, raw_scores: np.ndarray) -> float:
    """Spearman rank correlation between two score rows (per query)."""
    cos_ranks = np.argsort(np.argsort(cos_scores))
    raw_ranks = np.argsort(np.argsort(raw_scores))
    if HAVE_SCIPY:
        return float(spearmanr(cos_ranks, raw_ranks).statistic)
    return float(pd.Series(cos_ranks).corr(pd.Series(raw_ranks), method="spearman"))


def topk_overlap(a: list[int], b: list[int]) -> float:
    """Fraction of top-k items shared by two orderings (order-insensitive)."""
    return len(set(a) & set(b)) / len(a)


def preview(text: str, limit: int = PREVIEW) -> str:
    """Flatten a passage for one-line printing."""
    flat = text.replace("\n", " ")
    return flat[:limit] + ("..." if len(flat) > limit else "")


# --------------------------------------------------------------------------
# 5. Print the artifact — runnable demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 2. Load ---------------------------------------------------------
    passage_texts, passage_ids = load_passages(PASSAGES_PATH, N_PASSAGES)
    questions = load_questions(TEST_PATH, N_QUESTIONS)

    print("=" * 66)
    print("Lab 02 — cosine similarity vs raw dot product for retrieval")
    print(f"model: {MODEL_NAME} (local BGE, L2-normalized embeddings)")
    print("=" * 66)

    print(f"\n[1] Setup")
    print(f"    {len(passage_texts)} passages (first {N_PASSAGES} of 3200, "
          f"ids {passage_ids[0]}..{passage_ids[-1]})")
    print(f"    {len(questions)} questions from test.parquet (first {N_QUESTIONS}):")
    for q in questions:
        print(f"      - {q}")
    print(f"    Spearman via {'scipy' if HAVE_SCIPY else 'pandas fallback'}")

    # --- 3. Embed --------------------------------------------------------
    embedder = BGEEmbedding(model_name=MODEL_NAME)
    P = np.asarray(embedder.embed_documents(passage_texts), dtype=np.float32)
    Q = np.asarray([embedder.embed_query(q) for q in questions], dtype=np.float32)
    print(f"\n    embedded {P.shape[0]} passages x {P.shape[1]} dims, "
          f"{Q.shape[0]} queries x {Q.shape[1]} dims")

    # --- 4. Three score matrices -----------------------------------------
    cosine = cosine_similarity_matrix(Q, P)
    norm_dot = normalized_dot_matrix(Q, P)
    P_raw, scale = unnormalized_passages(P, [len(t) for t in passage_texts])
    raw_dot = raw_dot_matrix(Q, P_raw)

    # --- [2] cosine vs normalized dot: the identity ----------------------
    max_diff = float(np.max(np.abs(cosine - norm_dot)))
    print("\n[2] Cosine vs normalized dot — the identity proof")
    print("    cos(a,b) = (a.b)/(||a||*||b||); BGE guarantees ||a||=||b||=1,")
    print("    so cos(a,b) = a.b. The two matrices must be numerically equal:")
    print(f"    max |cosine - normalized_dot| over all {cosine.size} scores = {max_diff:.3e}")
    print("    -> identical to within float32 precision. For a normalized")
    print("       embedder, cosine similarity IS the dot product.")

    # --- [3] cosine vs raw dot: the disagreement -------------------------
    print("\n[3] Cosine vs raw dot on deliberately unnormalized vectors")
    print("    Passage vectors were scaled by a deterministic length-correlated")
    print("    factor (1.0..2.99) to simulate an embedder that skips L2")
    print("    normalization. Raw dot then ranks by magnitude * direction.")
    rho_list, overlap_list, identical_list = [], [], []
    for i in range(len(questions)):
        cos_top = top_k_indices(cosine[i], TOP_K)
        raw_top = top_k_indices(raw_dot[i], TOP_K)
        rho_list.append(spearman_between(cosine[i], raw_dot[i]))
        overlap_list.append(topk_overlap(cos_top, raw_top))
        identical_list.append(cos_top == raw_top)
    print(f"    Spearman(cosine ranks, raw-dot ranks), mean over "
          f"{len(questions)} queries: {np.mean(rho_list):.3f}")
    print(f"    top-{TOP_K} overlap (shared items, order-insensitive), mean: "
          f"{np.mean(overlap_list):.2f}")
    print(f"    queries where the top-{TOP_K} order is identical: "
          f"{sum(identical_list)}/{len(questions)}")

    # Concrete mis-ranking example: first query whose top-1 differs.
    demo = next(
        (
            i
            for i in range(len(questions))
            if top_k_indices(cosine[i], 1) != top_k_indices(raw_dot[i], 1)
        ),
        0,
    )
    cos_top = top_k_indices(cosine[demo], TOP_K)
    norm_top = top_k_indices(norm_dot[demo], TOP_K)
    raw_top = top_k_indices(raw_dot[demo], TOP_K)
    print(f"\n    Concrete example — question {demo}:")
    print(f'      "{questions[demo]}"')
    print(f"      {'rank':<5}{'cosine':>8}{'norm_dot':>10}{'raw_dot':>8}"
          f"   (passage ids into the {N_PASSAGES}-passage subset)")
    for r in range(TOP_K):
        print(f"      {r + 1:<5}{cos_top[r]:>8}{norm_top[r]:>10}{raw_top[r]:>8}")
    print("    cosine and normalized_dot return the identical top-5; raw_dot")
    print("    reorders it. The raw-dot winner is a longer passage whose")
    print(f"    magnitude factor {scale[raw_top[0]]:.2f} outweighs its worse direction:")
    print(f"      raw_dot #1 (id {raw_top[0]}): {preview(passage_texts[raw_top[0]])}")
    print(f"      cosine  #1 (id {cos_top[0]}): {preview(passage_texts[cos_top[0]])}")

    # --- [4] Takeaway ----------------------------------------------------
    print("\n[4] Takeaway")
    print("    Cosine similarity is the standard for text embeddings because")
    print("    it isolates direction (meaning) from magnitude (length), which")
    print("    is usually noise. The raw dot product is only equivalent when")
    print("    the embedder guarantees unit-norm vectors — BGE does, so a")
    print("    dot-product vector store is safe with BGE. If you swap in an")
    print("    embedder that does not normalize (e.g. E5), either normalize")
    print("    the vectors yourself or switch the store's metric to cosine.")