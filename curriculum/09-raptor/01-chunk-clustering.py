"""Lab 01 — Chunk embeddings + GMM clustering (RAPTOR step 1).

RAPTOR (Sarthi et al., 2024) replaces a flat chunk list with a *summary
tree*. The first half of that index is unsupervised: embed every chunk and
group the embeddings so that related chunks land in the same cluster. Each
cluster is small enough to be compressed by a summary without losing the
details (the summarization happens in lab 02 — this lab never calls an LLM).

The clustering recipe from the paper is a recursive Gaussian mixture model:

* fit a 2-component ``GaussianMixture`` over the current chunk embeddings;
* if a resulting cluster is at most ``max_cluster_size`` chunks, keep it as
  a leaf cluster;
* otherwise split that cluster again (fit another GMM inside it).

The output is a *partition*: every chunk index appears in exactly one
cluster. That partition is the raw material for the tree — lab 02 turns
each cluster into a summary node.

Run from the repo root:
    python curriculum/09-raptor/01-chunk-clustering.py
    python curriculum/09-raptor/01-chunk-clustering.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/09-raptor/01-chunk-clustering.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from embeddings.bge import BGEEmbedding  # noqa: E402
from tools.raptor import cluster_embeddings  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
N_PASSAGES = 30  # deterministic head; embedding + clustering only, no LLM calls
MAX_CLUSTER_SIZE = 10  # clusters larger than this get split recursively
SEED = 42  # deterministic GMM splits
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DEVICE = "cpu"  # shared GPU: Ollama holds most of VRAM


# --------------------------------------------------------------------------
# 2. Load — first N passages of the rag-mini-wikipedia corpus
# --------------------------------------------------------------------------
def load_passages(path: Path, n: int) -> list[str]:
    """Return the first ``n`` passage texts (deterministic head)."""
    df = pd.read_parquet(path)
    return [str(text).strip() for text in df.head(n)["passage"].tolist()]


# --------------------------------------------------------------------------
# 3. Experiment — embed the chunks, then cluster them with the recursive GMM
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    passages = load_passages(PASSAGES_PATH, N_PASSAGES)
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME, device=BGE_DEVICE)

    t0 = time.perf_counter()
    embeddings = embedder.embed_documents(passages)
    embed_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    clusters = cluster_embeddings(
        embeddings, max_cluster_size=MAX_CLUSTER_SIZE, seed=SEED
    )
    cluster_s = time.perf_counter() - t0

    return {
        "passages": passages,
        "embeddings": embeddings,
        "clusters": clusters,
        "embed_s": embed_s,
        "cluster_s": cluster_s,
    }


# --------------------------------------------------------------------------
# 4. Demo — print the partition
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 09-01 — Chunk embeddings + GMM clustering (RAPTOR step 1)")
    n = len(exp["passages"])
    clusters = exp["clusters"]
    print(f"{n} passages embedded in {exp['embed_s']:.1f}s; "
          f"{len(clusters)} clusters (cap {MAX_CLUSTER_SIZE}) found in "
          f"{exp['cluster_s']:.1f}s")
    print("=" * 66)

    sizes = sorted((len(c) for c in clusters), reverse=True)
    print(f"\n[1] Cluster sizes (largest first): {sizes}")

    print(f"\n[2] One sample passage per cluster (first 120 chars):")
    for i, cluster in enumerate(clusters):
        sample = exp["passages"][cluster[0]]
        print(f"    C{i} (size {len(cluster)}): {sample[:120]}")

    print(f"\n[3] Takeaway")
    print("    Clustering is the unsupervised half of RAPTOR's index: each")
    print("    cluster groups chunks that a single summary can compress.")
    print("    The partition covers every chunk exactly once, so lab 02 can")
    print("    build one summary node per cluster without losing chunks.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    clusters = exp["clusters"]
    n = len(exp["passages"])
    flat = sorted(i for cluster in clusters for i in cluster)

    checks.append(("every chunk index assigned exactly once "
                   f"(union == set(range({n})))",
                   flat == list(range(n))))
    checks.append((f"cluster count in [2, {n // 2}] (got {len(clusters)})",
                   2 <= len(clusters) <= n // 2))
    checks.append(("every cluster is non-empty",
                   all(len(cluster) > 0 for cluster in clusters)))
    checks.append((f"every cluster size <= {MAX_CLUSTER_SIZE}",
                   all(len(cluster) <= MAX_CLUSTER_SIZE for cluster in clusters)))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
