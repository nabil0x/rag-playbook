"""Lab 02 — HotpotQA evidence graph: how gold paragraphs connect.

Lab 01 built a graph from an LLM's triples. This lab shows the same
idea using a benchmark's own annotations: HotpotQA multi-hop questions
come with ``supporting_facts`` — the exact (paragraph, sentence) pairs a
correct answer must combine. Most of those questions need *two* gold
paragraphs, which is the definition of multi-hop: the answer lives in the
link *between* paragraphs, not inside one.

We turn the dev set into a co-evidence graph:

* nodes are paragraphs (titles) that appear as gold evidence;
* an edge between two paragraphs means both were required by the same
  question (weight = how many questions share the pair);
* per question we print the gold evidence chain — the sentences a retriever
  would need to surface in order.

Nothing here needs a model: the graph comes straight from the dataset's
labels, which makes it a clean ground-truth view of what "multi-hop"
structure actually looks like.

Run from the repo root:
    python src/curriculum/08-graphrag/02-evidence-graph.py
    python src/curriculum/08-graphrag/02-evidence-graph.py --verify
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the repo-root component library importable when this file is run
# directly (``python src/curriculum/08-graphrag/02-evidence-graph.py``).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import networkx as nx  # noqa: E402

HOTPOT_PATH = Path("Data/corpus/hotpotqa/hotpot_dev_distractor_v1.json")
N_QUESTIONS = 50  # pool cap: first N multi-hop questions (deterministic head)
MAX_CHAINS_SHOWN = 5  # how many example evidence chains to print
TOP_HUBS = 6  # how many highest-degree paragraphs to print


# --------------------------------------------------------------------------
# 2. Load — dev set, keep only multi-hop questions (>= 2 gold paragraphs)
# --------------------------------------------------------------------------
def load_multi_hop(path: Path, n: int) -> list[dict]:
    """Return the first ``n`` questions whose gold evidence spans >= 2 paragraphs."""
    with open(path) as f:
        records = json.load(f)
    out: list[dict] = []
    for rec in records:
        gold_titles = {title for title, _ in rec["supporting_facts"]}
        if len(gold_titles) >= 2:
            out.append(rec)
        if len(out) >= n:
            break
    return out


def evidence_chain(rec: dict) -> list[tuple[str, int]]:
    """The (title, sentence_index) chain of a question's gold evidence."""
    facts = sorted(
        [(title, sent_idx) for title, sent_idx in rec["supporting_facts"]],
        key=lambda pair: (pair[0], pair[1]),
    )
    return facts


# --------------------------------------------------------------------------
# 3. Experiment — co-evidence graph over the sampled questions
# --------------------------------------------------------------------------
def run_experiment() -> dict:
    questions = load_multi_hop(HOTPOT_PATH, N_QUESTIONS)

    graph = nx.Graph()
    for rec in questions:
        titles = {title for title, _ in rec["supporting_facts"]}
        for title in titles:
            graph.add_node(title, questions=[])
            graph.nodes[title]["questions"].append(rec["_id"])
        for a in titles:
            for b in titles:
                if a < b:
                    if not graph.has_edge(a, b):
                        graph.add_edge(a, b, weight=0)
                    graph[a][b]["weight"] += 1

    degrees = [d for _, d in graph.degree()]
    components = list(nx.connected_components(graph))
    chains = [(rec["question"], rec["_id"], evidence_chain(rec))
              for rec in questions]

    return {
        "questions": questions,
        "graph": graph,
        "chains": chains,
        "stats": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "max_degree": max(degrees) if degrees else 0,
            "components": len(components),
            "multi_hop_in_pool": len(questions),
        },
        "hubs": sorted(graph.degree(), key=lambda pair: pair[1],
                       reverse=True)[:TOP_HUBS],
    }


# --------------------------------------------------------------------------
# 4. Demo — print the artifact
# --------------------------------------------------------------------------
def print_demo(exp: dict) -> None:
    print("=" * 66)
    print("Lab 08-02 — HotpotQA evidence graph (co-evidence of gold paragraphs)")
    s = exp["stats"]
    print(f"{s['multi_hop_in_pool']} multi-hop questions in the pool")
    print("=" * 66)

    print(f"\n[1] Co-evidence graph: {s['nodes']} paragraphs, {s['edges']} "
          f"edges, {s['components']} components")
    print(f"    max paragraph degree: {s['max_degree']}")

    print(f"\n[2] Highest-degree paragraphs (shared evidence hubs):")
    for title, degree in exp["hubs"]:
        print(f"    {degree:3d}  {title[:70]}")

    print(f"\n[3] Example evidence chains (question -> gold paragraphs):")
    for question, qid, chain in exp["chains"][:MAX_CHAINS_SHOWN]:
        print(f"    Q: {question[:80]}")
        for title, sent_idx in chain:
            print(f"       ({title[:60]}, sentence {sent_idx})")

    print(f"\n[4] Takeaway")
    print("    An edge here is a question that *demands* both paragraphs.")
    print("    High-degree paragraphs are reuse hubs: one paragraph answers")
    print("    many different questions. This is the ground truth GraphRAG")
    print("    community detection (lab 03) tries to rediscover from text.")


# --------------------------------------------------------------------------
# 5. Verification gate — run ``python <lab> --verify`` from the repo root
# --------------------------------------------------------------------------
def verify_gate(exp: dict) -> int:
    checks: list[tuple[str, bool]] = []
    s = exp["stats"]
    graph = exp["graph"]

    checks.append((f"pool has >= {N_QUESTIONS // 2} multi-hop questions "
                   f"(got {s['multi_hop_in_pool']})",
                   s["multi_hop_in_pool"] >= N_QUESTIONS // 2))
    checks.append((f"graph has >= 15 nodes (got {s['nodes']})",
                   s["nodes"] >= 15))
    checks.append((f"graph has >= 10 edges (got {s['edges']})",
                   s["edges"] >= 10))
    checks.append(("every edge weight is a positive integer",
                   all(graph[a][b]["weight"] >= 1 for a, b in graph.edges())))
    checks.append(("every node remembers its questions",
                   all(graph.nodes[n]["questions"] for n in graph.nodes())))
    checks.append(("every chain has >= 2 distinct paragraphs",
                   all(len({t for t, _ in chain}) >= 2
                       for _, _, chain in exp["chains"])))

    print("verification gate:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    exp = run_experiment()
    if "--verify" in sys.argv:
        sys.exit(verify_gate(exp))
    print_demo(exp)
