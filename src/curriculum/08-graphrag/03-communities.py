"""Lab 03 — Communities + map/reduce summarization (the GraphRAG index step 2).

Lab 01 produced an entity graph; this lab turns it into the *index* GraphRAG
actually queries. The recipe:

1. Community detection — partition the entity graph with the Leiden
   algorithm so that entities inside a community are densely connected
   (they "belong together") while different communities stay loosely
   connected. This is the unsupervised structure discovery that makes
   GraphRAG useful over a whole corpus without a query in sight.
2. Map — summarize every community's internal relations into a few prose
   sentences (``src/tools/graphrag.community_summaries``).
3. Reduce — fold the community summaries into one global corpus summary
   (``src/tools/graphrag.global_summary``).

Community detection runs fully locally (sknetwork Leiden with a networkx
Louvain fallback — no model call); only the two summarization steps talk
to the LLM, so the whole index costs roughly one LLM call per passage plus
one per community.

Run from the repo root:
    python src/curriculum/08-graphrag/03-communities.py
    python src/curriculum/08-graphrag/03-communities.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/08-graphrag/03-communities.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from llms.ollama import OllamaLLM  # noqa: E402
from tools.graph import build_entity_graph  # noqa: E402
from tools.graphrag import (  # noqa: E402
    community_summaries,
    detect_communities,
    global_summary,
)

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
N_PASSAGES = 20  # deterministic head; each passage costs one extraction call
MAX_COMMUNITIES = 6  # map/reduce cap: largest communities get summarized
SEED = 42  # deterministic community detection (Leiden / Louvain)


# --------------------------------------------------------------------------
# 2. Load — first N passages of the rag-mini-wikipedia corpus
# --------------------------------------------------------------------------
def load_passages(path: Path, n: int) -> list[str]:
    """Return the first ``n`` passage texts (deterministic head)."""
    df = pd.read_parquet(path)
    return [str(text).strip() for text in df.head(n)["passage"].tolist()]


# --------------------------------------------------------------------------
# 3. Experiment — build graph, detect communities, map/reduce summaries
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    passages = load_passages(PASSAGES_PATH, N_PASSAGES)
    llm = OllamaLLM()  # local qwen2.5-coder:7b; map/reduce + extraction

    t0 = time.perf_counter()
    graph = build_entity_graph(passages, llm)
    build_s = time.perf_counter() - t0

    communities = detect_communities(graph, seed=SEED)
    t0 = time.perf_counter()
    summaries = community_summaries(
        llm, graph, communities, max_communities=MAX_COMMUNITIES
    )
    map_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    global_text = global_summary(llm, [s["summary"] for s in summaries])
    reduce_s = time.perf_counter() - t0

    covered = {node for community in communities for node in community}
    return {
        "graph": graph,
        "communities": communities,
        "summaries": summaries,
        "global_summary": global_text,
        "coverage": len(covered),
        "build_s": build_s,
        "map_s": map_s,
        "reduce_s": reduce_s,
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 08-03 — Communities + map/reduce summarization")
    graph = exp["graph"]
    print(f"{graph.number_of_nodes()} entities, {len(exp['communities'])} "
          f"communities, coverage {exp['coverage']}/{graph.number_of_nodes()}")
    print("=" * 66)

    sizes = sorted((len(c) for c in exp["communities"]), reverse=True)
    print(f"\n[1] Community sizes (largest first): {sizes}")

    print(f"\n[2] Map — community summaries (top {MAX_COMMUNITIES} by size):")
    for i, entry in enumerate(exp["summaries"], start=1):
        print(f"    C{i} (size {entry['size']}): {entry['summary']}")

    print(f"\n[3] Reduce — global corpus summary:")
    print(f"    {exp['global_summary']}")

    print(f"\n[4] Timing: build graph {exp['build_s']:.1f}s, "
          f"map {exp['map_s']:.1f}s, reduce {exp['reduce_s']:.1f}s")

    print(f"\n[5] Takeaway")
    print("    Communities partition the corpus by structure, not by query.")
    print("    The map step compresses every community into a few sentences;")
    print("    the reduce step folds them into one global summary. Lab 04")
    print("    queries this index two ways: local (walk the graph from the")
    print("    question's entities) and global (rank the summaries).")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    graph = exp["graph"]
    communities = exp["communities"]
    covered = exp["coverage"]

    checks.append(("communities cover every entity (no orphan nodes)",
                   covered == graph.number_of_nodes()))
    checks.append(("communities are disjoint",
                   sum(len(c) for c in communities) == covered))
    checks.append((f"at least 2 communities (got {len(communities)})",
                   len(communities) >= 2))
    checks.append((f"summaries cover {len(exp['summaries'])} communities "
                   f"(cap {MAX_COMMUNITIES})",
                   len(exp["summaries"]) <= MAX_COMMUNITIES))
    checks.append(("every summary is non-empty",
                   all(entry["summary"].strip()
                       for entry in exp["summaries"])))
    checks.append(("global summary is non-empty",
                   bool(exp["global_summary"].strip())))
    checks.append(("community sizes are consistent with members",
                   all(len(entry["members"]) == entry["size"]
                       for entry in exp["summaries"])))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
