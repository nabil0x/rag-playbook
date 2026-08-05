"""Lab 02 — Recursive tree build with LLM summarization (RAPTOR step 2).

Lab 01 produced a flat partition of chunks. This lab turns it into the
artifact RAPTOR actually queries: a *tree* where the leaves are the raw
chunks and every internal node is a short LLM summary of the cluster of
chunks beneath it.

The build is ``tools/raptor.build_tree``:

* level 0 — every chunk is a leaf node carrying its own text;
* cluster the leaf embeddings with the recursive GMM from lab 01;
* ask the LLM to summarize each cluster ("Summarize the following passages
  in 2-3 sentences, keeping key facts and names.") — that summary becomes a
  parent node whose children are the cluster's leaves;
* recurse on the summaries until one node (the root) remains.

Two properties make the tree useful instead of a glorified flat list:

1. **Coverage** — every chunk appears in exactly one leaf, so no information
   is dropped while building the tree.
2. **Compression** — a query can navigate the few summary nodes at the top
   and only descend into the cluster that actually matters (lab 03 tests
   that traversal). The whole build costs one LLM call per internal node,
   far cheaper than re-reading every chunk at query time.

Run from the repo root:
    python curriculum/09-raptor/02-recursive-tree.py
    python curriculum/09-raptor/02-recursive-tree.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/09-raptor/02-recursive-tree.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from embeddings.bge import BGEEmbedding  # noqa: E402
from llms.ollama import OllamaLLM  # noqa: E402
from tools.raptor import build_tree, collect_chunk_ids  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
N_PASSAGES = 24  # deterministic head; the tree costs one LLM call per cluster
MAX_CLUSTER_SIZE = 8  # max chunks a single summary node may cover
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DEVICE = "cpu"  # shared GPU: Ollama holds most of VRAM
ROOT_PREVIEW = 200  # characters of the root summary to print
MID_PREVIEW = 160  # characters of the mid-level summary to print


# --------------------------------------------------------------------------
# 2. Load — first N passages of the rag-mini-wikipedia corpus
# --------------------------------------------------------------------------
def load_passages(path: Path, n: int) -> list[str]:
    """Return the first ``n`` passage texts (deterministic head)."""
    df = pd.read_parquet(path)
    return [str(text).strip() for text in df.head(n)["passage"].tolist()]


# --------------------------------------------------------------------------
# 3. Experiment — embed, cluster, summarize recursively until one root remains
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    passages = load_passages(PASSAGES_PATH, N_PASSAGES)
    llm = OllamaLLM()  # local qwen2.5-coder:7b; temperature 0
    embedder = BGEEmbedding(model_name=BGE_MODEL_NAME, device=BGE_DEVICE)

    t0 = time.perf_counter()
    info = build_tree(
        passages,
        embedder,
        llm,
        max_cluster_size=MAX_CLUSTER_SIZE,
        progress=lambda level, count: print(
            f"  built level {level}: {count} node(s)", end="\r", flush=True
        ),
    )
    build_s = time.perf_counter() - t0
    print(" " * 40, end="\r")

    mid_nodes = [
        node
        for node in _walk(info["tree"])
        if node["level"] == 1 and node["children"]
    ]
    return {
        "passages": passages,
        "tree": info["tree"],
        "levels": info["levels"],
        "leaves": info["leaves"],
        "llm_calls": info["llm_calls"],
        "build_s": build_s,
        "node_counts": _node_counts_per_level(info["tree"]),
        "root_text": info["tree"]["text"],
        "mid_text": mid_nodes[0]["text"] if mid_nodes else "",
    }


def _walk(node: dict):
    """Depth-first iterator over every node of the tree."""
    yield node
    for child in node["children"]:
        yield from _walk(child)


def _node_counts_per_level(root: dict) -> dict[int, int]:
    """Number of nodes per level (level 0 = leaves)."""
    counts: dict[int, int] = {}
    for node in _walk(root):
        counts[node["level"]] = counts.get(node["level"], 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------
# 4. Demo — print the tree artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 09-02 — Recursive tree build with LLM summarization")
    print(f"{exp['leaves']} leaves, {exp['levels']} levels, "
          f"{exp['llm_calls']} LLM calls in {exp['build_s']:.1f}s")
    print("=" * 66)

    print(f"\n[1] Nodes per level:")
    for level, count in exp["node_counts"].items():
        role = "root" if level == exp["levels"] else (
            "leaves" if level == 0 else "summaries")
        print(f"    level {level}: {count:3d} node(s)  [{role}]")

    print(f"\n[2] Root summary (first {ROOT_PREVIEW} chars):")
    print(f"    {exp['root_text'][:ROOT_PREVIEW]}")

    print(f"\n[3] One mid-level summary (first {MID_PREVIEW} chars):")
    print(f"    {exp['mid_text'][:MID_PREVIEW] or '(no mid-level node)'}")

    print(f"\n[4] Takeaway")
    print("    The tree is a lossy but navigable index: level 0 keeps every")
    print("    chunk verbatim, and each higher level compresses a cluster")
    print("    into a few sentences. Retrieval walks the summaries and only")
    print("    expands the cluster that matches the question — lab 03")
    print("    compares that walk against scanning every chunk flat.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    root = exp["tree"]
    covered = sorted(collect_chunk_ids(root))

    checks.append((f"tree has >= 2 levels (got {exp['levels']})",
                   exp["levels"] >= 2))
    checks.append(("root exists and has non-empty text",
                   bool(root.get("text", "").strip())))
    checks.append((f"number of leaves == {len(exp['passages'])} "
                   f"(got {exp['leaves']})",
                   exp["leaves"] == len(exp["passages"])))
    checks.append(("every chunk covered exactly once across leaves",
                   covered == list(range(len(exp["passages"])))))
    checks.append(("every non-leaf node has non-empty summary text",
                   all(node["text"].strip()
                       for node in _walk(root) if node["children"])))
    checks.append((f"llm_calls <= 40 (got {exp['llm_calls']})",
                   exp["llm_calls"] <= 40))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
