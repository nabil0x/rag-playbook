"""FAISS index internals.

Vector DB block: build and benchmark the four ANN index types faiss ships —
flat (exact), IVF, HNSW, PQ — and learn the recall/speed/memory tradeoffs.
See Topics/Project-24-Vector-Index-Internals/README.md.
"""

from __future__ import annotations

from langchain_core.documents import Document


class FaissIndex:
    """Thin teaching wrapper over raw ``faiss`` indexes.

    Builds one of ``flat`` / ``ivf`` / ``hnsw`` / ``pq`` on a matrix of
    vectors and benchmarks it against the flat (exact) baseline. The point
    is to make the index parameters visible and measurable, not to replace
    ``FAISSVectorStore`` (Project 07) — that one wraps the LangChain
    integration, this one drives raw faiss directly.
    """

    #: index_type names this class knows how to build.
    SUPPORTED = ("flat", "ivf", "hnsw", "pq")

    def __init__(self, index_type: str = "flat", dim: int = 128, **params):
        # index_type: one of SUPPORTED.
        # dim: embedding dimension — required to allocate any index.
        # params: index-specific knobs (nlist, nprobe, M, efSearch, nbits...).
        if index_type not in self.SUPPORTED:
            raise ValueError(f"index_type must be one of {self.SUPPORTED}, got {index_type!r}")
        self.index_type = index_type
        self.dim = dim
        self.params = params
        self._index = None
        self._trained = False

    def _build_index(self):
        """TODO(Project 24): construct the raw faiss index for self.index_type.

        Import ``faiss`` lazily here (it is an optional dependency). Then:
        - "flat": ``faiss.IndexFlatL2(self.dim)`` — the exact baseline.
        - "ivf": ``faiss.IndexIVFFlat(quantizer, self.dim, nlist, metric)``
          with a flat L2 quantizer and ``nlist`` (default 100) clusters;
          set ``nprobe`` (default 10) for queries. IVF needs ``train()``
          before ``add()``.
        - "hnsw": ``faiss.IndexHNSWFlat(self.dim, M)`` with ``M`` (default 32)
          and ``efSearch`` (default 64); HNSW builds incrementally, no train.
        - "pq": ``faiss.IndexIVFPQ(quantizer, self.dim, nlist, M, nbits)``
          with ``M`` (default 8) sub-vectors and ``nbits`` (default 8);
          also needs ``train()`` before ``add()``.

        Return the constructed (but possibly untrained) index object.
        """
        raise NotImplementedError("TODO(Project 24): implement FaissIndex._build_index")

    def build(self, vectors: list[list[float]]) -> None:
        """TODO(Project 24): train (if needed) and add ``vectors``.

        - Allocate the index via ``self._build_index()``.
        - IVF/PQ must be trained first: ``index.train(matrix)`` then
          ``index.add(matrix)``; flat/HNSW only need ``add(matrix)``.
        - Keep the built index on ``self._index`` and set ``self._trained``.
        - Assert every vector has length ``self.dim`` before adding.
        """
        raise NotImplementedError("TODO(Project 24): implement FaissIndex.build")

    def search(self, query_vector: list[float], k: int = 10) -> tuple[list[float], list[int]]:
        """TODO(Project 24): return (distances, ids) for the k nearest vectors.

        - For IVF, set ``index.nprobe`` from ``self.params`` (default 10)
          before searching.
        - For HNSW, set ``index.hnsw.efSearch`` (default 64) before searching.
        - Call ``index.search(matrix, k)`` and return the two arrays as lists.
        """
        raise NotImplementedError("TODO(Project 24): implement FaissIndex.search")

    def benchmark(
        self,
        queries: list[list[float]],
        ground_truth: list[list[int]],
        k: int = 10,
    ) -> dict:
        """TODO(Project 24): measure recall@k and mean latency vs flat baseline.

        For every query: compute the exact top-k ids with a fresh flat index
        (``faiss.IndexFlatL2``) as ground truth, then run ``self.search`` and
        count how many of the returned ids appear in the exact top-k.

        Return a dict with the keys:
            {"index_type", "recall_at_k", "mean_latency_ms", "n_queries"}
        where recall is the mean fraction of hits across queries and latency
        is measured with ``time.perf_counter()`` around the searches.
        """
        raise NotImplementedError("TODO(Project 24): implement FaissIndex.benchmark")

    def size_bytes(self) -> int | None:
        """Return the index's approximate memory footprint, or None if empty."""
        if self._index is None:
            return None
        return self._index.ntotal * self.dim * 4  # rough: 4 bytes per float


if __name__ == "__main__":
    # No-network smoke test. faiss-cpu is optional: skip gracefully.
    try:
        import faiss  # noqa: F401
    except ImportError:
        print("SKIP: FaissIndex demo needs faiss-cpu: pip install faiss-cpu")
        raise SystemExit(0)

    idx = FaissIndex(index_type="flat", dim=4)
    assert idx.index_type == "flat" and idx.dim == 4
    assert idx.size_bytes() is None  # nothing built yet

    try:
        FaissIndex(index_type="bogus", dim=4)
    except ValueError:
        pass  # unknown index types must be rejected up front
    else:
        raise AssertionError("FaissIndex must reject unknown index types")

    print("OK: FaissIndex validates index_type and reports size before build")
