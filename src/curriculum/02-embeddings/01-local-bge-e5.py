"""Lab 01 — local embeddings with BGE and E5.

An embedding turns text into a vector of numbers, and the whole RAG pipeline
stands on the assumption that "similar meaning => similar vector". This lab
runs two popular open-source embedders head to head on the same real corpus
(``Data/corpus/rag-mini-wikipedia``) and inspects the three numbers that
decide everything downstream:

* DIMENSION — how many floats per vector (768 for both models here). This is
  the storage cost of your vector database: 768 floats x 4 bytes per chunk.
* NORM — the length of the embedding vector. A unit-norm vector makes cosine
  similarity identical to a dot product, which matters when your vector
  database offers fast dot-product scoring. ``src/embeddings/bge.py`` normalizes
  BGE explicitly (``encode_kwargs`` ``normalize_embeddings``); the E5 model
  on the Hub ships its own ``2_Normalize`` layer, so it comes out unit-norm
  too. A surprising number of embedders do NOT normalize — always check.
* COSINE SIMILARITY — the retrieval score. We embed a small deterministic
  subset of passages once, embed 2-3 real questions from the corpus ``test``
  split, and rank passages by cosine similarity for each model.

Why local models: no API keys, no per-token cost, no data leaving your
machine. BGE (``BAAI/bge-base-en-v1.5``) is an English retrieval model
trained with normalized embeddings; E5 (``intfloat/multilingual-e5-base``)
covers many languages and is trained with instruction prefixes — E5 queries
are prefixed ``"query: "`` and passages ``"passage: "`` before embedding,
which ``src/embeddings/e5.py`` applies automatically. Prefix mismatches are a
classic silent retrieval killer.

Note on BGE construction: the shared ``src/embeddings/bge.py`` module builds the
model with the current universal ``HuggingFaceEmbeddings`` class (the same
class ``src/embeddings/e5.py`` uses) plus ``encode_kwargs`` ``normalize_embeddings``
= True, so its ``BGEEmbedding`` exposes the same contract as ``E5Embedding``.

Run from the repo root:
    python src/curriculum/02-embeddings/01-local-bge-e5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/02-embeddings/01-local-bge-e5.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embeddings.bge import BGEEmbedding  # noqa: E402
from embeddings.e5 import E5Embedding  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the comparison
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
TEST_PATH = Path("Data/corpus/rag-mini-wikipedia/test.parquet")
N_PASSAGES = 20  # deterministic head of the 3200-passage corpus (keeps runtime low)
QUESTION_IDS = [1606, 1610, 1604]  # real questions from test.parquet, picked to match
TOP_K = 3
PREVIEW = 62  # max characters of passage text shown next to each hit
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"


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


# --------------------------------------------------------------------------
# 3. Embed & compare — helpers shared by both models
# --------------------------------------------------------------------------
def run_model(
    model: object, passages: list[str], questions: list[str]
) -> tuple[list[list[float]], list[list[float]]]:
    """Embed all passages and all questions with one model.

    Returns (passage_vectors, query_vectors). Passages are embedded in one
    batched call; queries one at a time (``embed_query``) because in real RAG
    each incoming question is embedded individually.
    """
    passage_vecs = model.embed_documents(passages)
    query_vecs = [model.embed_query(q) for q in questions]
    return passage_vecs, query_vecs


def l2_norm(vector: list[float]) -> float:
    """Euclidean length of an embedding vector."""
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))


def top_k_results(
    query_vec: list[float],
    passage_vecs: list[list[float]],
    passage_ids: list[int],
    k: int,
) -> list[tuple[int, float]]:
    """Rank passages by cosine similarity to the query; return top-k (id, score)."""
    matrix = np.asarray(passage_vecs, dtype=np.float32)
    sims = cosine_similarity(np.asarray([query_vec], dtype=np.float32), matrix)[0]
    order = np.argsort(sims)[::-1][:k]
    return [(passage_ids[i], float(sims[i])) for i in order]


def preview(text: str, limit: int = PREVIEW) -> str:
    """Flatten a passage for one-line printing."""
    flat = text.replace("\n", " ")
    return flat[:limit] + ("..." if len(flat) > limit else "")


# --------------------------------------------------------------------------
# 4. Print the artifact — runnable demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 2. Load ---------------------------------------------------------
    passage_texts, passage_ids = load_passages(PASSAGES_PATH, N_PASSAGES)
    questions = load_questions(TEST_PATH, QUESTION_IDS)

    print("=" * 66)
    print("Lab 01 — local embeddings: BGE vs E5 on rag-mini-wikipedia")
    print(f"{BGE_MODEL_NAME}  vs  intfloat/multilingual-e5-base")
    print("=" * 66)

    print(f"\n[1] Corpus (deterministic subset, no randomness):")
    print(f"    {len(passage_texts)} passages (first {N_PASSAGES} of 3200, ids {passage_ids[0]}..{passage_ids[-1]})")
    print(f"    {len(questions)} questions from test.parquet:")
    for qid, qtext in questions:
        print(f"      [{qid}] {qtext}")

    # --- 3. Embed with both models ---------------------------------------
    models = {"BGE": BGEEmbedding(model_name=BGE_MODEL_NAME), "E5": E5Embedding()}
    question_texts = [qtext for _, qtext in questions]
    embedded = {
        name: run_model(model, passage_texts, question_texts)
        for name, model in models.items()
    }

    print("\n[2] Embedding vectors — dimension and norm:")
    print(f"    {'model':<6}{'dim':>6}{'passage norm':>14}{'query norm':>14}")
    for name, (pvecs, qvecs) in embedded.items():
        dim = len(pvecs[0])
        p_norm = l2_norm(pvecs[0])
        q_norm = l2_norm(qvecs[0])
        print(f"    {name:<6}{dim:>6}{p_norm:>14.4f}{q_norm:>14.4f}")
    print("    BGE is normalized explicitly (encode_kwargs); E5 unit-norm via its")
    print("    model's own 2_Normalize layer — cosine == dot product for both.")

    # --- 4. Top-k retrieval per question ---------------------------------
    print(f"\n[3] Top-{TOP_K} retrieval per question (cosine similarity):")
    for i, (qid, qtext) in enumerate(questions):
        print(f'\n    Q[{qid}] "{qtext}"')
        for name in models:
            pvecs, qvecs = embedded[name]
            hits = top_k_results(qvecs[i], pvecs, passage_ids, TOP_K)
            print(f"      {name:<4} " + "  ".join(
                f"id {pid} {score:.4f}" for pid, score in hits
            ))
            for pid, score in hits:
                idx = passage_ids.index(pid)
                print(f"            {score:.4f}  {preview(passage_texts[idx])}")

    # --- 5. Takeaway -----------------------------------------------------
    print("\n[4] Takeaway")
    print("    Same dimension (768) and both unit-norm, yet not interchangeable:")
    print("    E5's wrapper adds 'query: '/'passage: ' prefixes that BGE never")
    print("    sees, and BGE's normalization is explicit while E5's is baked")
    print("    into the model. When you swap embedders, check the vector norm")
    print("    and the prefix handling together — and remember retrieval scores")
    print("    are only comparable within one model's vector space, never across")
    print("    models.")
