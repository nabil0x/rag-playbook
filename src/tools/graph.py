"""Entity/relation graph construction for GraphRAG-style indexing.

LangChain has no native entity-graph builder, so this module fills the gap:
documents -> LLM-extracted (head, relation, tail) triples -> a networkx graph.

The extraction step talks to any LLM exposing a ``json_object(prompt)``
method that returns a dict (e.g. GroqLLM from ``src/llms/groq.py``). Everything
after extraction is pure networkx and requires no model.

Used by src/curriculum/08-graphrag/ labs 01, 03 and 04.
"""
from __future__ import annotations

import random
import re
from typing import Callable, Protocol

import networkx as nx

#: JSON shape the LLM is asked to produce for triple extraction.
TRIPLE_SCHEMA = '{"triples": [{"head": "...", "relation": "...", "tail": "..."}]}'

#: JSON extraction LLM: anything exposing ``json_object(prompt) -> dict | list``.
#: GroqLLM from ``src/llms/groq.py`` matches this contract.
class JsonLLM(Protocol):
    def json_object(self, prompt: str) -> dict | list: ...


def extract_triples(llm: JsonLLM, text: str) -> list[tuple[str, str, str]]:
    """Extract ``(head, relation, tail)`` triples from one passage of text.

    Tolerates the ``{"triples": [...]}`` shape, a ``{"edges": ...}`` or
    ``{"data": [...]}`` alias, and a single triple given as a bare dict.
    Returns an empty list whenever the model fails to return valid JSON.
    """
    prompt = (
        "Extract the entity-relation triples from the text below.\n"
        "Rules:\n"
        "- Only entities explicitly named in the text.\n"
        "- Entities are people, places, organizations, works, events or "
        "concrete things (1-4 words).\n"
        "- Relations are short verbs or prepositional phrases (1-4 words), "
        "present tense.\n"
        f"- Output ONLY JSON: {TRIPLE_SCHEMA}\n"
        "- Output an empty list if the text has no meaningful triples.\n"
        "\n"
        f"Text:\n{text}"
    )
    result = llm.json_object(prompt)
    if isinstance(result, list):  # bare array of triples, no wrapper key
        raw = result
    elif isinstance(result, dict) and "error" not in result:
        raw = result.get("triples", result.get("edges", result.get("data", [])))
        if isinstance(raw, dict):  # single triple given without a list wrapper
            raw = [raw]
    else:
        return []
    triples: list[tuple[str, str, str]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        head = str(item.get("head", item.get("subject", ""))).strip()
        relation = str(item.get("relation", item.get("predicate", ""))).strip()
        tail = str(item.get("tail", item.get("object", ""))).strip()
        if head and tail:
            triples.append((head, relation or "related to", tail))
    return triples


def extract_entities(llm: JsonLLM, text: str, limit: int = 8) -> list[str]:
    """Ask the LLM for the salient named entities in ``text`` (up to ``limit``)."""
    prompt = (
        "List the salient named entities in the text below (people, places, "
        "organizations, works, events, concrete things).\n"
        "Rules:\n"
        "- Only entities explicitly named in the text.\n"
        "- Each entity is 1-4 words; use the most specific name that appears.\n"
        f"- Output ONLY JSON: {{\"entities\": [\"...\", \"...\"]}} with at most "
        f"{limit} entities.\n"
        "\n"
        f"Text:\n{text}"
    )
    result = llm.json_object(prompt)
    if isinstance(result, list):  # bare array of entity names, no wrapper key
        entities = result
    elif isinstance(result, dict) and "error" not in result:
        entities = result.get("entities", [])
    else:
        return []
    out: list[str] = []
    for ent in entities:
        if isinstance(ent, str) and ent.strip():
            out.append(ent.strip())
    return out[:limit]


def build_entity_graph(
    passages: list[str],
    llm: JsonLLM,
    progress: Callable[[int, int], None] | None = None,
) -> nx.Graph:
    """Build an entity/relation graph from a list of passage texts.

    Node attributes:
      - ``passages``: list of passage indices in which the entity appears.
    Edge attributes:
      - ``relations``: the extracted relation phrases linking the pair.
      - ``weight``: how many distinct relation phrases were extracted.
    """
    graph = nx.Graph()
    total = len(passages)
    for i, text in enumerate(passages):
        for head, relation, tail in extract_triples(llm, text):
            if head == tail:
                continue  # self-loops carry no structure
            graph.add_edge(head, tail, relations=set())
            for node in (head, tail):
                graph.nodes[node].setdefault("passages", [])
            graph.nodes[head]["passages"].append(i)
            graph.nodes[tail]["passages"].append(i)
            graph[head][tail]["relations"].add(relation)
        if progress is not None:
            progress(i + 1, total)
    for _, _, data in graph.edges(data=True):
        data["weight"] = len(data["relations"])
    return graph


def graph_stats(graph: nx.Graph) -> dict:
    """Compact summary of graph structure, safe for graphs of any size."""
    degrees = [d for _, d in graph.degree()]
    components = list(nx.connected_components(graph))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": round(nx.density(graph), 4),
        "avg_degree": round(sum(degrees) / len(degrees), 2) if degrees else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "connected_components": len(components),
        "largest_component": max((len(c) for c in components), default=0),
    }


def top_entities(graph: nx.Graph, k: int = 8) -> list[tuple[str, int]]:
    """The ``k`` highest-degree entities, as ``(name, degree)`` pairs."""
    ranked = sorted(graph.degree(), key=lambda pair: pair[1], reverse=True)
    return [(name, int(degree)) for name, degree in ranked[:k]]


def _clean_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()


def sample_relations(
    graph: nx.Graph, k: int = 8, seed: int = 42
) -> list[tuple[str, str, str]]:
    """``k`` example ``(head, relation, tail)`` triples from the graph."""
    rng = random.Random(seed)
    edges = list(graph.edges(data=True))
    rng.shuffle(edges)
    out: list[tuple[str, str, str]] = []
    for u, v, data in edges:
        for relation in sorted(data["relations"]):
            out.append((u, relation, v))
        if len(out) >= k:
            break
    return out[:k]
