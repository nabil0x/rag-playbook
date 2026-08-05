"""Lab 01 — Build an entity/relation graph from passages (the GraphRAG index).

GraphRAG changes what we index. Plain RAG embeds chunks and retrieves by
vector similarity; GraphRAG first extracts the *entities* and the *relations*
between them from the corpus and stores that structure as a graph. A query is
then answered by walking or summarizing the graph instead of (or in addition
to) dense retrieval.

This lab implements the index half of that recipe with ``tools/graph``:

* each passage is sent to a local-LLM-backed extractor (OllamaLLM +
  ``json_object``) which returns ``(head, relation, tail)`` triples;
* ``build_entity_graph`` folds every passage's triples into one networkx
  graph — nodes are entities, edges carry the relation phrases that link
  them, and each node remembers which passages mention it.

The graph is the artifact everything else in this track builds on: lab 03
detects communities over it and lab 04 queries it locally and globally.

Run from the repo root:
    python curriculum/08-graphrag/01-entity-graph.py
    python curriculum/08-graphrag/01-entity-graph.py --verify
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/08-graphrag/01-entity-graph.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from llms.ollama import OllamaLLM  # noqa: E402
from tools.graph import (  # noqa: E402
    build_entity_graph,
    graph_stats,
    sample_relations,
    top_entities,
)

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
PASSAGES_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
N_PASSAGES = 25  # deterministic head; each passage costs one LLM extraction call
TOP_K = 8  # how many highest-degree entities to print
N_SAMPLE_RELATIONS = 8  # how many example (head, relation, tail) triples to show


# --------------------------------------------------------------------------
# 2. Load — first N passages of the rag-mini-wikipedia corpus
# --------------------------------------------------------------------------
def load_passages(path: Path, n: int) -> list[str]:
    """Return the first ``n`` passage texts (deterministic head)."""
    df = pd.read_parquet(path)
    return [str(text).strip() for text in df.head(n)["passage"].tolist()]


# --------------------------------------------------------------------------
# 3. Experiment — extract triples per passage and fold them into one graph
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    passages = load_passages(PASSAGES_PATH, N_PASSAGES)
    llm = OllamaLLM()  # local qwen2.5-coder:7b; expose json_object for extraction

    t0 = time.perf_counter()
    graph = build_entity_graph(
        passages,
        llm,
        progress=lambda done, total: print(
            f"  extracted {done}/{total} passages", end="\r", flush=True
        ),
    )
    build_s = time.perf_counter() - t0

    return {
        "passages": passages,
        "graph": graph,
        "build_s": build_s,
        "stats": graph_stats(graph),
        "top": top_entities(graph, TOP_K),
        "examples": sample_relations(graph, N_SAMPLE_RELATIONS),
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 08-01 — Entity/relation graph from passages")
    print(f"{len(exp['passages'])} passages, built in {exp['build_s']:.1f}s")
    print("=" * 66)

    s = exp["stats"]
    print(f"\n[1] Graph structure over {s['nodes']} entities / {s['edges']} "
          f"edges:")
    print(f"    density      : {s['density']}")
    print(f"    avg degree   : {s['avg_degree']}")
    print(f"    max degree   : {s['max_degree']}")
    print(f"    components   : {s['connected_components']} "
          f"(largest {s['largest_component']})")

    print(f"\n[2] Highest-degree entities (hubs):")
    for name, degree in exp["top"]:
        print(f"    {degree:3d}  {name}")

    print(f"\n[3] Example extracted triples:")
    for head, relation, tail in exp["examples"]:
        print(f"    {head} -[{relation}]-> {tail}")

    print(f"\n[4] Takeaway")
    print("    The graph is a lossy but *structured* summary of the corpus:")
    print("    entities become nodes and their co-occurring relations become")
    print("    edges. Hubs (high degree) are the entities that connect many")
    print("    others — the bridges a multi-hop question must walk across.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    s = exp["stats"]
    graph = exp["graph"]

    checks.append(("every edge endpoint is a node",
                   all(graph.has_node(u) and graph.has_node(v)
                       for u, v in graph.edges())))
    checks.append(("every node remembers its passages",
                   all(graph.nodes[n].get("passages") for n in graph.nodes())))
    checks.append((f"graph has >= 15 entities (got {s['nodes']})",
                   s["nodes"] >= 15))
    checks.append((f"graph has >= 10 edges (got {s['edges']})",
                   s["edges"] >= 10))
    checks.append(("avg degree >= 1.0 (no fully isolated entity graph)",
                   s["avg_degree"] >= 1.0))
    checks.append(("at least one edge carries a relation phrase",
                   any(graph[u][v].get("relations") for u, v in graph.edges())))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
