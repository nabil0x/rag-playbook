"""RAPTOR: recursive abstractive processing for tree-organized retrieval.

LangChain has no native RAPTOR implementation, so this module fills the gap.
The recipe (Sarthi et al., 2024) replaces a flat chunk list with a summary
tree: cluster the chunk embeddings with a recursive Gaussian mixture model,
summarize each cluster with the LLM to form the parent level, and recurse
until one root remains. Retrieval then walks the tree — keeping the ``top_k``
most similar children per level — and returns leaf chunks (``collapse=True``)
or the chosen summaries (``collapse=False``).

The index costs roughly one LLM call per internal node; query time jumps to
the relevant leaves in a few similarity steps instead of scanning every
chunk. Embedders expose ``embed_documents(list[str]) -> list[list[float]]``
(e.g. BGEEmbedding from ``embeddings/``); LLMs expose ``invoke(prompt) -> str``
(e.g. OllamaLLM from ``llms/ollama.py``). Only summarization talks to the
LLM — ``cluster_embeddings`` alone is LLM-free.

Used by curriculum/09-raptor/ labs 01, 02 and 03.
"""
from __future__ import annotations

import sys
from typing import Callable, Protocol, Sequence

import numpy as np
from sklearn.mixture import GaussianMixture


class StrLLM(Protocol):
    def invoke(self, prompt: str) -> str: ...


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


#: Instruction shared by every summarization call in the tree build.
SUMMARY_PROMPT = (
    "Summarize the following passages in 2-3 sentences, keeping key facts "
    "and names."
)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def _split_indices(
    index_set: Sequence[int],
    embeddings: Sequence[Sequence[float]],
    max_cluster_size: int,
    seed: int,
) -> list[list[int]]:
    """Recursively split ``index_set`` into clusters of at most ``max_cluster_size``.

    Fits a 2-component full-covariance GaussianMixture on the current index
    set. A cluster small enough is kept as a leaf; anything larger is split
    again. When the fit fails (too few points, degenerate covariance, or a
    collapsed component) the whole set is returned as one cluster, so the
    partition always covers every index exactly once.
    """
    if len(index_set) <= max_cluster_size:
        return [sorted(index_set)]
    matrix = np.asarray([embeddings[i] for i in index_set], dtype=float)
    try:
        labels = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=seed,
        ).fit_predict(matrix)
    except (ValueError, np.linalg.LinAlgError):
        # Fallback (spec): keep the whole set as one cluster on fit failure
        # (ill-defined covariance, n_components > n_samples) so coverage holds.
        return [sorted(index_set)]

    groups: dict[int, list[int]] = {}
    for idx, label in zip(index_set, labels):
        groups.setdefault(int(label), []).append(idx)
    if len(groups) < 2:  # degenerate split: every point in one component
        return [sorted(index_set)]

    return [
        cluster
        for group in groups.values()
        for cluster in _split_indices(group, embeddings, max_cluster_size, seed)
    ]


def cluster_embeddings(
    embeddings: list[list[float]],
    max_cluster_size: int,
    seed: int = 42,
) -> list[list[int]]:
    """Partition chunk indices into GMM clusters of at most ``max_cluster_size``.

    Every index in ``range(len(embeddings))`` appears in exactly one cluster
    (the recursion is over a partition of the index set). Returns a single
    cluster when fitting fails or there are too few points.
    """
    if not embeddings:
        return []
    return _split_indices(
        list(range(len(embeddings))), embeddings, max_cluster_size, seed
    )


def _summarize(llm: StrLLM, texts: Sequence[str]) -> str:
    """One LLM call: compress ``texts`` into 2-3 sentences of prose."""
    joined = "\n".join(f"- {text}" for text in texts)
    return llm.invoke(f"{SUMMARY_PROMPT}\n\n{joined}").strip()


def collect_chunk_ids(node: dict) -> list[int]:
    """All chunk ids under ``node`` (leaf walk) — useful for gate checks."""
    if not node["children"]:
        return list(node["chunk_ids"])
    ids: list[int] = []
    for child in node["children"]:
        ids.extend(collect_chunk_ids(child))
    return ids


def build_tree(
    chunks: list[str],
    embedder: Embedder,
    llm: StrLLM,
    max_cluster_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Build the RAPTOR tree: leaves are chunks, parents are LLM summaries.

    Returns ``{"tree", "levels", "leaves", "llm_calls"}``. ``tree`` is a
    nested dict with shape ``{"text", "chunk_ids", "children", "level"}`` —
    level-0 nodes are leaves (one chunk each, empty ``children``); higher
    levels are summaries whose ``chunk_ids`` cover the whole subtree. Every
    chunk appears in exactly one leaf. ``progress(level, node_count)`` is
    called after each level is built.
    """
    nodes = [
        {"text": chunks[i], "chunk_ids": [i], "children": [], "level": 0}
        for i in range(len(chunks))
    ]
    llm_calls = 0
    level = 0
    while len(nodes) > 1:
        level += 1
        vectors = embedder.embed_documents([node["text"] for node in nodes])
        clusters = cluster_embeddings(vectors, max_cluster_size)
        if len(clusters) == len(nodes):
            # No merge happened (degenerate GMM split): collapse to a single
            # root so the recursion always terminates.
            clusters = [list(range(len(nodes)))]
        parents: list[dict] = []
        for cluster in clusters:
            members = [nodes[i] for i in cluster]
            parents.append(
                {
                    "text": _summarize(llm, [node["text"] for node in members]),
                    "chunk_ids": sorted(
                        cid for node in members for cid in node["chunk_ids"]
                    ),
                    "children": members,
                    "level": level,
                }
            )
            llm_calls += 1
        nodes = parents
        if progress is not None:
            progress(level, len(nodes))

    if not nodes:  # empty chunk list: keep a harmless empty root
        nodes = [{"text": "", "chunk_ids": [], "children": [], "level": 0}]
    root = nodes[0]
    return {
        "tree": root,
        "levels": level,
        "leaves": len(collect_chunk_ids(root)),
        "llm_calls": llm_calls,
    }


def retrieve(
    tree: dict,
    question: str,
    embedder: Embedder,
    top_k: int = 4,
    collapse: bool = True,
) -> dict:
    """Traverse the tree for ``question``.

    At each level the ``top_k`` children most similar to the question (by
    cosine) are kept and their children searched next. With ``collapse=True``
    the traversal keeps descending to the leaves and returns the best leaf
    chunk texts; with ``collapse=False`` it returns the summary texts of the
    deepest chosen nodes (the "tree-only" view).

    Returns ``{"texts", "ids", "scores", "path"}`` — ``path`` holds the node
    texts chosen at each level.
    """
    question_vec = embedder.embed_documents([question])[0]
    path: list[list[str]] = []
    frontier = [tree]
    last_selected = [tree]
    last_scores = [1.0]
    while frontier and frontier[0].get("children"):
        vectors = embedder.embed_documents([node["text"] for node in frontier])
        ranked = sorted(
            enumerate(frontier),
            key=lambda pair: _cosine(question_vec, vectors[pair[0]]),
            reverse=True,
        )[:top_k]
        last_selected = [frontier[i] for i, _ in ranked]
        last_scores = [_cosine(question_vec, vectors[i]) for i, _ in ranked]
        path.append([node["text"] for node in last_selected])
        frontier = [
            child for node in last_selected for child in node["children"]
        ]

    if not collapse:
        return {
            "texts": [node["text"] for node in last_selected],
            "ids": [
                cid for node in last_selected for cid in node["chunk_ids"]
            ],
            "scores": [round(score, 3) for score in last_scores],
            "path": path,
        }

    leaves = [node for node in frontier if not node.get("children")] or [tree]
    vectors = embedder.embed_documents([node["text"] for node in leaves])
    ranked = sorted(
        enumerate(leaves),
        key=lambda pair: _cosine(question_vec, vectors[pair[0]]),
        reverse=True,
    )[:top_k]
    return {
        "texts": [leaves[i]["text"] for i, _ in ranked],
        "ids": [leaves[i]["chunk_ids"][0] for i, _ in ranked],
        "scores": [
            round(_cosine(question_vec, vectors[i]), 3) for i, _ in ranked
        ],
        "path": path,
    }


def _smoke_test() -> int:
    """Standalone smoke: fake embedder + fake LLM, zero network, no heavy deps."""

    class _FakeEmbedder:
        VOCAB = ("cat", "dog", "bird", "sleep", "bark", "fly")

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [
                [1.0 if token in text.lower() else 0.0 for token in self.VOCAB]
                for text in texts
            ]

    class _FakeLLM:
        def invoke(self, prompt: str) -> str:
            lines = [
                line[2:]
                for line in prompt.splitlines()
                if line.startswith("- ")
            ]
            return " ".join(line[:24] for line in lines)[:90] or "summary"

    chunks = [
        "cats sleep a lot",
        "cats sleep during the day",
        "cats groom their fur",
        "dogs bark loudly",
        "dogs love long walks",
        "dogs chase cats",
        "birds fly south",
        "birds build nests",
        "birds sing at dawn",
    ]
    embedder = _FakeEmbedder()
    llm = _FakeLLM()

    embeddings = embedder.embed_documents(chunks)
    clusters = cluster_embeddings(embeddings, max_cluster_size=3)
    flat = sorted(i for cluster in clusters for i in cluster)

    info = build_tree(chunks, embedder, llm, max_cluster_size=3)
    leaves = sorted(collect_chunk_ids(info["tree"]))
    result = retrieve(info["tree"], "sleepy cat", embedder, top_k=2)

    checks: list[tuple[str, bool]] = [
        ("cluster_embeddings covers every index exactly once",
         flat == list(range(len(chunks)))),
        ("clustering produced more than one cluster",
         len(clusters) > 1),
        (f"every cluster respects max_cluster_size=3 (got "
         f"{sorted(len(c) for c in clusters)})",
         all(len(c) <= 3 for c in clusters)),
        ("build_tree terminated with a single non-empty root",
         info["levels"] >= 1 and bool(info["tree"]["text"].strip())),
        ("tree leaves cover every chunk exactly once",
         leaves == list(range(len(chunks)))),
        ("retrieve returns matching texts and ids",
         len(result["texts"]) > 0
         and len(result["ids"]) == len(result["texts"])
         and len(result["scores"]) == len(result["texts"])),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print(f"smoke: clusters={len(clusters)} levels={info['levels']} "
          f"leaves={info['leaves']} llm_calls={info['llm_calls']} "
          f"retrieved={len(result['texts'])}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_smoke_test())
