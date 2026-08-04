"""Chunk-size sweep: retrieval quality vs chunk size.

Splitter block: sweep ``chunk_size`` and measure the RETRIEVAL impact — the
classic chunk-size tradeoff. Small chunks are precise but fragment a passage
across many vectors; large chunks are coherent but coarse (a whole passage
may collapse into one chunk, dragging neighbouring topics along).

Protocol: rag-mini-wikipedia has no qrels and its ``test.parquet`` answers are
yes/no, so we use a passage-sourced **self-recall** probe instead. We take
~50 random passages, derive a query from each (its first sentence), retrieve
top-5 chunks, and mark "source passage found" when any retrieved chunk belongs
to that passage. Each chunk size gets a FRESH split + embed + FAISS index so
the comparison is clean.

Embeddings are local BGE (BAAI/bge-base-en-v1.5) via the repo's
``embeddings/bge.py`` when the installed langchain-huggingface still ships
``HuggingFaceBgeEmbeddings``; otherwise we fall back to a direct
sentence-transformers wrapper with the same contract. Indexing reuses
``vectordb/faiss.py`` (FAISSVectorStore with precomputed embeddings).

Lab 6 (final) of track 01-chunking. See .omo/plans/layer1-rag-playbook.md.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path

# Keep the sweep table the star of the output: silence the "Loading weights"
# progress bar (transformers honors HF_HUB_DISABLE_PROGRESS_BARS) and the
# deprecation notice langchain_community.FAISS logs for precomputed
# embeddings (a logging call, so set the module logger to ERROR). The env var
# must be set before any third-party import — langchain_text_splitters already
# pulls in huggingface_hub, which reads the flag at import time.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("langchain_community.vectorstores.faiss").setLevel(logging.ERROR)

import pandas as pd
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/01-chunking/06-chunk-size-sweep.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from embeddings.bge import BGEEmbedding  # noqa: E402
from vectordb.faiss import FAISSVectorStore  # noqa: E402

# --- module-level constants: tweak these to re-run the experiment -----------
CORPUS_PATH = Path("Data/corpus/rag-mini-wikipedia/passages.parquet")
CORPUS_SUBSET = 1000  # first N passages — keeps CPU embedding time reasonable
CHUNK_SIZES = [100, 200, 400, 800]
OVERLAP = 50
PROBE_QUERIES = 50
TOP_K = 5
RANDOM_SEED = 42  # fixed seed so the probe set is reproducible
MODEL_NAME = "BAAI/bge-base-en-v1.5"


class _SentenceTransformerEmbedder:
    """BGE embedder with the repo's ``embed_query``/``embed_documents`` contract.

    Stand-in for ``BGEEmbedding`` when the installed langchain-huggingface no
    longer exports ``HuggingFaceBgeEmbeddings`` (removed in 1.x). Keeps the
    same lazy-build + normalized-embedding semantics.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_query(self, text: str) -> list[float]:
        return (
            self._get_model()
            .encode(text, normalize_embeddings=True, show_progress_bar=False)
            .tolist()
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return (
            self._get_model()
            .encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            .tolist()
        )


def make_embedder():
    """Return a BGE embedder exposing ``embed_query``/``embed_documents``.

    Prefers the repo's ``BGEEmbedding``; falls back to a direct
    sentence-transformers wrapper when the installed langchain-huggingface
    can't provide ``HuggingFaceBgeEmbeddings``.
    """
    try:
        from langchain_huggingface import HuggingFaceBgeEmbeddings  # noqa: F401

        return BGEEmbedding(model_name=MODEL_NAME)
    except (ImportError, AttributeError) as exc:
        print(
            f"note: repo BGEEmbedding not usable here ({exc}); "
            "falling back to sentence-transformers directly"
        )
        return _SentenceTransformerEmbedder(model_name=MODEL_NAME)


def load_passages(path: Path, limit: int) -> list[Document]:
    """Load the first ``limit`` passages as Documents carrying a ``passage_id``."""
    df = pd.read_parquet(path)
    return [
        Document(page_content=text, metadata={"passage_id": i})
        for i, text in enumerate(df["passage"].head(limit).tolist())
    ]


def derive_query(text: str, max_words: int = 40) -> str:
    """Derive a probe query from a passage: first ~40 words, trimmed to a sentence.

    Cuts at the last sentence-ending punctuation inside the first ``max_words``
    so the query is a complete, self-contained sentence.
    """
    first = " ".join(text.split()[:max_words])
    cut = max((first.rfind(p) for p in ".!?"), default=-1)
    if cut > 0:
        first = first[: cut + 1]
    return first.strip() or text[:200].strip()


def preview(text: str, limit: int = 100) -> str:
    """Truncate text for printing (never dump full chunk contents)."""
    return text[:limit] + ("..." if len(text) > limit else "")


def measure_chunk_size(
    size: int,
    passages: list[Document],
    probes: list[tuple[int, str]],
    embedder,
    enc,
) -> dict:
    """Split, embed, index and probe one chunk size; return one table row."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=OVERLAP
    )
    chunks = splitter.split_documents(passages)
    vectors = [list(v) for v in embedder.embed_documents(
        [c.page_content for c in chunks]
    )]

    store = FAISSVectorStore()
    store.add(chunks, embeddings=vectors)  # fresh index per chunk size

    avg_tokens = (
        sum(len(enc.encode(c.page_content)) for c in chunks) / len(chunks)
        if chunks
        else 0.0
    )

    found = 0
    latencies_ms: list[float] = []
    for passage_id, query in probes:
        query_vec = embedder.embed_query(query)
        t0 = time.perf_counter()
        hits = store.query(query_vec, top_k=TOP_K)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        if any(h.metadata.get("passage_id") == passage_id for h in hits):
            found += 1

    return {
        "chunk_size": size,
        "n_chunks": len(chunks),
        "avg_tokens": avg_tokens,
        "recall": found / len(probes),
        "latency_ms": sum(latencies_ms) / len(latencies_ms),
    }


if __name__ == "__main__":
    # --- 1. setup: what we are sweeping and how we measure it ---------------
    print("Chunk-size sweep — retrieval quality vs chunk size")
    print(f"  corpus      : {CORPUS_PATH} (subset: first {CORPUS_SUBSET} passages)")
    print(f"  embedder    : {MODEL_NAME} (local BGE, CPU)")
    print(f"  chunk sizes : {CHUNK_SIZES}  overlap={OVERLAP}")
    print(f"  probe       : {PROBE_QUERIES} passage-derived queries, "
          f"top-{TOP_K}, self-recall@5, seed={RANDOM_SEED}")
    print("  (first run downloads the BGE model, ~440MB)\n")

    # --- 2. load: a subset of passages as documents -------------------------
    passages = load_passages(CORPUS_PATH, CORPUS_SUBSET)
    avg_chars = sum(len(p.page_content) for p in passages) / len(passages)
    print(f"Loaded {len(passages)} passages "
          f"(avg {avg_chars:.0f} chars/passage)")

    # --- 3. probes: passage-derived queries (self-recall protocol) ----------
    rng = random.Random(RANDOM_SEED)
    probe_ids = rng.sample(range(len(passages)), PROBE_QUERIES)
    probes = [(pid, derive_query(passages[pid].page_content)) for pid in probe_ids]
    print(f"Derived {len(probes)} probe queries from random passages, e.g.:")
    for pid, query in probes[:2]:
        print(f"  [{pid}] {preview(query)!r}")

    # --- 4. sweep: fresh split + embed + index per chunk size ---------------
    embedder = make_embedder()
    enc = tiktoken.encoding_for_model("gpt-4")
    print("\nSweeping chunk sizes (fresh index per size)...\n")
    header = (
        f"{'chunk_size':>10} | {'n_chunks':>9} | {'avg_tokens':>10} | "
        f"{'self_recall@5':>12} | {'mean_latency_ms':>15}"
    )
    print(header)
    print("-" * len(header))
    rows = []
    for size in CHUNK_SIZES:
        row = measure_chunk_size(size, passages, probes, embedder, enc)
        rows.append(row)
        print(
            f"{row['chunk_size']:>10} | {row['n_chunks']:>9} | "
            f"{row['avg_tokens']:>10.1f} | {row['recall']:>12.3f} | "
            f"{row['latency_ms']:>15.2f}"
        )

    # --- 5. takeaway: what the table teaches --------------------------------
    small, large = rows[0], rows[-1]
    print("\n--- takeaway: chunk size trades granularity against coherence ---")
    print(
        f"chunk_size {small['chunk_size']} -> {large['chunk_size']}: "
        f"{small['n_chunks']} -> {large['n_chunks']} chunks "
        f"({small['n_chunks'] / large['n_chunks']:.1f}x fewer), "
        f"avg {small['avg_tokens']:.0f} -> {large['avg_tokens']:.0f} "
        "tokens/chunk"
    )
    print(
        "self-recall@5 stays high at every size because the probes are cut "
        "verbatim from passage openings — near-duplicate queries that any "
        "index finds. The real costs show in the other columns:"
    )
    print(
        "  * small chunks -> many vectors: more index memory, more candidates "
        "per query (latency), and a mid-passage fact gets torn across chunks, "
        "so retrieval returns pieces instead of the answer."
    )
    print(
        "  * large chunks -> few, fat vectors: fast and coherent, but coarse — "
        "a whole passage may be one chunk, so a query pulls in neighbouring "
        "topics and the context window fills with irrelevant text."
    )
    print(
        "The sweet spot is the smallest chunk size that still keeps each "
        "passage's facts inside one retrievable unit."
    )