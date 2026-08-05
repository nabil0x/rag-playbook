"""GraphRAG index and query helpers built on ``tools/graph``.

Implements the second half of the GraphRAG recipe from curriculum/08-graphrag/:

1. Community detection over the entity graph (sknetwork Leiden primary,
   networkx Louvain fallback - both verified against networkx 3.6.1).
2. Map/reduce community summarization.
3. Two query strategies: local search (entity linking + 1-hop expansion)
   and global search (embedding-routed over community summaries).

Embedders are BGEEmbedding / E5Embedding from ``embeddings/``; both expose
``embed_documents(list[str]) -> list[list[float]]``. LLMs expose
``invoke(prompt) -> str`` (GroqLLM). The entity-graph builder needs the
additional ``json_object(prompt) -> dict`` method, so we re-export
``extract_triples``/``extract_entities`` from ``tools.graph`` here.
"""
from __future__ import annotations

from typing import Iterable, Protocol, Sequence

import networkx as nx

from tools.graph import JsonLLM, build_entity_graph, extract_entities  # noqa: F401


class StrLLM(Protocol):
    def invoke(self, prompt: str) -> str: ...


class GraphLLM(StrLLM, JsonLLM, Protocol):
    """LLM that can both extract JSON (json_object) and generate (invoke)."""


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def detect_communities(graph: nx.Graph, seed: int = 42) -> list[set[str]]:
    """Partition ``graph`` into communities.

    Uses sknetwork's Leiden when available; falls back to networkx Louvain.
    Isolated nodes (which Leiden drops from its membership matrix) are
    recovered as singleton communities so the partition always covers the
    whole node set.
    """
    nodes = list(graph.nodes())
    if not nodes:
        return []
    try:
        import numpy as np
        from scipy.sparse import csr_matrix
        from sknetwork.clustering import Leiden

        index = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        rows: list[int] = []
        cols: list[int] = []
        for u, v in graph.edges():
            rows.extend((index[u], index[v]))
            cols.extend((index[v], index[u]))
        adjacency = csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n, n)
        )
        # membership: n x k sparse; row i marks the column of node i's group
        membership = Leiden(random_state=seed).fit_transform(adjacency)
        dense = membership.toarray()
        labels = dense.argmax(axis=1)
        assigned = dense.sum(axis=1) > 0
    except Exception:
        # Fallback: networkx Louvain (unweighted, to match the ones/zeros above)
        parts = nx.community.louvain_communities(
            graph, weight=None, seed=seed
        )
        return [set(part) for part in parts]

    communities: dict[int, set[str]] = {}
    next_label = dense.shape[1]
    for node, label, is_assigned in zip(nodes, labels, assigned):
        if is_assigned:
            communities.setdefault(int(label), set()).add(node)
        else:
            communities[next_label] = {node}  # isolated -> own community
            next_label += 1
    return list(communities.values())


def community_text(graph: nx.Graph, community: Iterable[str]) -> str:
    """Render the intra-community edges of ``community`` as relation lines."""
    members = set(community)
    lines: set[str] = set()
    for node in members:
        for neighbor in graph.neighbors(node):
            if neighbor not in members:
                continue
            for relation in graph[node][neighbor].get("relations", ()):
                lines.add(f"{node} -[{relation}]-> {neighbor}")
    return "\n".join(sorted(lines)) or "no intra-community relations"


def summarize_community(llm: StrLLM, text: str) -> str:
    """Map step: condense one community's relation lines into prose."""
    prompt = (
        "Summarize the following entity-relation fragment in 2-3 sentences. "
        "Name the main entities and what connects them.\n\n"
        f"{text}"
    )
    return llm.invoke(prompt).strip()


def community_summaries(
    llm: StrLLM,
    graph: nx.Graph,
    communities: Sequence[Iterable[str]],
    max_communities: int = 8,
) -> list[dict]:
    """Map/reduce-ready list of ``{"members": [...], "summary": "..."}``.

    Communities are processed largest-first and truncated to
    ``max_communities`` so the reducer stays cheap.
    """
    ordered = sorted(communities, key=len, reverse=True)
    summaries: list[dict] = []
    for community in ordered[:max_communities]:
        summaries.append(
            {
                "members": sorted(community),
                "size": len(community),
                "summary": summarize_community(
                    llm, community_text(graph, community)
                ),
            }
        )
    return summaries


def global_summary(llm: StrLLM, summaries: Sequence[str]) -> str:
    """Reduce step: fold all community summaries into one global summary."""
    bullets = "\n".join(f"- {summary}" for summary in summaries)
    prompt = (
        "Combine the following community summaries into one global summary "
        "of the corpus in 3-4 sentences.\n\n"
        f"{bullets}"
    )
    return llm.invoke(prompt).strip()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def link_entities(
    embedder: Embedder,
    query_entities: Sequence[str],
    candidates: Sequence[str],
    threshold: float = 0.55,
) -> dict[str, str]:
    """Best-match each query entity to a graph entity via cosine similarity.

    Returns only links whose similarity meets ``threshold``. This is the
    lexical-alias bridge: the question says "Scott Derrickson" and the graph
    node is "Scott Derrickson" (or a near match like "Scott Derrickson's").
    """
    if not query_entities or not candidates:
        return {}
    query_vecs = embedder.embed_documents(list(query_entities))
    candidate_vecs = embedder.embed_documents(list(candidates))
    linked: dict[str, str] = {}
    for query, query_vec in zip(query_entities, query_vecs):
        best_name, best_score = None, threshold
        for name, vec in zip(candidates, candidate_vecs):
            score = _cosine(query_vec, vec)
            if score > best_score:
                best_name, best_score = name, score
        if best_name is not None:
            linked[query] = best_name
    return linked


def local_search(
    question: str,
    llm: GraphLLM,
    embedder: Embedder,
    graph: nx.Graph,
    passages: Sequence[str],
    top_n: int = 6,
    threshold: float = 0.55,
) -> dict:
    """Local search: link question entities into the graph, expand 1 hop.

    Returns the answer plus the retrieval trail so labs can evaluate it:
    linked entities, expanded neighborhood, and the passages surfaced.
    """
    query_entities = extract_entities(llm, question)
    linked = link_entities(embedder, query_entities, graph.nodes(), threshold)
    seed = set(linked.values())
    expanded = set(seed)
    for node in seed:
        expanded.update(graph.neighbors(node))

    passage_ids: set[int] = set()
    for node in expanded:
        passage_ids.update(graph.nodes[node].get("passages", ()))
    ordered_ids = sorted(passage_ids)[:top_n]
    retrieved = [passages[i] for i in ordered_ids]

    context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(retrieved))
    answer = llm.invoke(
        "Answer the question using ONLY the context paragraphs below. "
        "If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in one or two sentences."
    ).strip()
    return {
        "answer": answer,
        "retrieved": retrieved,
        "retrieved_ids": ordered_ids,
        "query_entities": query_entities,
        "linked": linked,
        "expanded": sorted(expanded),
    }


def global_search(
    question: str,
    llm: StrLLM,
    embedder: Embedder,
    community_summaries: Sequence[dict],
    top_n: int = 2,
) -> dict:
    """Global search: route the question to the most relevant summaries.

    ``community_summaries`` is the list returned by ``community_summaries``.
    Returns the answer and the summaries that were used.
    """
    if not community_summaries:
        return {"answer": "", "summaries_used": [], "scores": []}
    question_vec = embedder.embed_documents([question])[0]
    summary_vecs = embedder.embed_documents(
        [entry["summary"] for entry in community_summaries]
    )
    scored = [
        (_cosine(question_vec, vec), entry)
        for vec, entry in zip(summary_vecs, community_summaries)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    used = [entry for _, entry in scored[:top_n]]

    bullets = "\n".join(f"- {entry['summary']}" for entry in used)
    answer = llm.invoke(
        "Answer the question using ONLY the corpus summaries below. "
        "If the summaries do not contain the answer, say so.\n\n"
        f"Corpus summaries:\n{bullets}\n\n"
        f"Question: {question}\n\n"
        "Answer in one or two sentences."
    ).strip()
    return {
        "answer": answer,
        "summaries_used": [entry["summary"] for entry in used],
        "scores": [
            {"summary": entry["summary"], "cosine": round(score, 3)}
            for score, entry in scored[:top_n]
        ],
    }
