"""Lab 05 — Semantic chunking: draw chunk boundaries where meaning shifts.

Splitter track, lab 5: compare *semantic* chunking against *recursive
character* chunking on two real long-prose documents — the opening of
``Pride and Prejudice`` and of ``Moby-Dick`` from the Gutenberg corpus
(``Data/corpus/gutenberg/``). Each book is bounded to its first
``SUBSET_CHARS`` characters: semantic splitting embeds every sentence
locally, so a full ~700KB novel would be far too slow — 50K characters of
prose is roughly 1,300 sentences and keeps one run in the low minutes.

Recursive character splitting cuts at a fixed character count, so a chunk
can end in the middle of a sentence. A semantic splitter instead embeds
every sentence and draws a chunk boundary exactly where the *meaning*
shifts: where the embedding distance between two consecutive sentences
jumps above the ``breakpoint_percentile`` percentile of all
consecutive-sentence distances (here the 95th).

What to look for in the output:

  * the semantic splitter produces more, smaller, topically coherent
    chunks — roughly one per chapter of each novel;
  * every boundary it draws sits at a sentence-level cosine-distance spike
    that clears the 95th-percentile threshold;
  * the recursive splitter's 500-char chunks are blind to those topic
    boundaries and cut mid-sentence.

All embeddings are local (BAAI/bge-base-en-v1.5 via sentence-transformers);
the model downloads on first use into ``~/.cache/huggingface``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/01-chunking/05-semantic.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from loaders.gutenberg import GutenbergLoader  # noqa: E402
from splitters.recursive import DocumentProcessor  # noqa: E402
from splitters.semantic import _SENTENCE_RE, SemanticSplitter  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
MODEL_NAME = "BAAI/bge-base-en-v1.5"  # local BGE model (splitters/semantic.py default)
BREAKPOINT_PERCENTILE = 95.0  # distance percentile above which a boundary is drawn
CHUNK_SIZE = 500  # recursive splitter: max characters per chunk
CHUNK_OVERLAP = 50  # recursive splitter: characters shared across chunks
SUBSET_CHARS = 50_000  # per-book prefix cap (keeps sentence embedding runtime sane)
DISTANCE_PRINT_CAP = 40  # max consecutive-sentence distances printed per doc
DOC_PATHS = [
    Path("Data/corpus/gutenberg/pride-and-prejudice.txt"),
    Path("Data/corpus/gutenberg/moby-dick.txt"),
]
PREVIEW = 200  # character cap for chunk previews
SENTENCE_PREVIEW = 80  # character cap for boundary-neighbour sentences


# ---------------------------------------------------------------------------
# Load: Gutenberg novels -> Documents with source metadata (bounded prefix)
# ---------------------------------------------------------------------------
def load_books(paths: list[Path]) -> list[Document]:
    """Load Gutenberg novels, boilerplate stripped, each bounded to a prefix.

    ``GutenbergLoader`` removes the Project Gutenberg header/footer, then
    each book's text is sliced to the first ``SUBSET_CHARS`` characters so
    the sentence-by-sentence BGE embedding stays within a sane runtime.
    """
    docs: list[Document] = []
    for path in paths:
        text = GutenbergLoader(path, strip=True).load()[0].page_content
        docs.append(
            Document(
                page_content=text[:SUBSET_CHARS],
                metadata={"source": str(path)},
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Boundary inspection: the teaching point
# ---------------------------------------------------------------------------
def consecutive_distances(model, sentences: list[str]) -> np.ndarray:
    """Cosine distance between consecutive sentences (1 - cosine similarity).

    Mirrors the internals of ``splitters/semantic.py`` so the numbers printed
    here are exactly the ones the SemanticSplitter used to draw boundaries.
    """
    vectors = np.asarray(model.encode(sentences), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard against zero-vector sentences
    normalized = vectors / norms
    dots = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return 1.0 - np.clip(dots, -1.0, 1.0)


def boundary_pairs(
    sentences: list[str], distances: np.ndarray, threshold: float, limit: int = 3
) -> list[tuple[int, int, float, str, str]]:
    """First ``limit`` boundaries as (pair_no, index, distance, prev, next)."""
    pairs: list[tuple[int, int, float, str, str]] = []
    boundary_count = 0
    for i, distance in enumerate(distances):
        if distance > threshold:
            boundary_count += 1
            pairs.append(
                (boundary_count, i, float(distance), sentences[i], sentences[i + 1])
            )
            if len(pairs) == limit:
                break
    return pairs


def preview(text: str, limit: int) -> str:
    """Collapse whitespace and cap a sentence/chunk preview at ``limit`` chars."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def avg_words(chunks: list[Document]) -> float:
    """Average word count across chunks (0.0 for an empty chunk list)."""
    if not chunks:
        return 0.0
    return sum(len(c.page_content.split()) for c in chunks) / len(chunks)


# ---------------------------------------------------------------------------
# Main: setup -> load -> split -> inspect boundaries -> compare -> print
# ---------------------------------------------------------------------------
def main() -> None:
    docs = load_books(DOC_PATHS)
    print(
        f"Loaded {len(docs)} Gutenberg books (each limited to the first "
        f"{SUBSET_CHARS} characters):"
    )
    for doc in docs:
        print(f"  {doc.metadata['source']} — {len(doc.page_content)} chars")

    try:
        import sentence_transformers
    except ImportError:
        print(
            "SKIP: semantic chunking needs sentence-transformers: "
            "pip install sentence-transformers"
        )
        return

    # One local BGE model shared by the splitter and the boundary analysis,
    # so both operate on identical sentence embeddings.
    model = sentence_transformers.SentenceTransformer(MODEL_NAME)
    semantic = SemanticSplitter(
        embedding=model, breakpoint_percentile=BREAKPOINT_PERCENTILE
    )
    recursive = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    semantic_chunks = semantic.split(docs)
    recursive_chunks = recursive.split_docs(docs)

    # --- Compare: chunk counts + average size -------------------------------
    print(
        f"Recursive split (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}): "
        f"{len(recursive_chunks)} chunk(s), avg {avg_words(recursive_chunks):.1f} words/chunk"
    )
    print(
        f"Semantic split (breakpoint_percentile={BREAKPOINT_PERCENTILE:.0f}): "
        f"{len(semantic_chunks)} chunk(s), avg {avg_words(semantic_chunks):.1f} words/chunk"
    )
    for doc in docs:
        source = doc.metadata["source"]
        n_recursive = sum(
            1 for c in recursive_chunks if c.metadata.get("source") == source
        )
        n_semantic = sum(1 for c in semantic_chunks if c.metadata.get("source") == source)
        print(f"  {source}: {n_recursive} recursive chunk(s) vs {n_semantic} semantic chunk(s)")

    # --- Teaching point: boundaries sit at cosine-distance spikes -----------
    print("\n--- Where does the semantic splitter draw boundaries? ---")
    for doc in docs:
        sentences = [
            s for s in _SENTENCE_RE.split(doc.page_content.strip()) if s
        ]
        if len(sentences) < 2:
            print(f"{doc.metadata['source']}: too few sentences to analyse")
            continue
        distances = consecutive_distances(model, sentences)
        threshold = float(np.percentile(distances, BREAKPOINT_PERCENTILE))
        print(
            f"\n{doc.metadata['source']} — {len(sentences)} sentence(s), "
            f"{len(distances)} consecutive-sentence distances"
        )
        print(
            f"  distance stats: mean={distances.mean():.3f}  "
            f"max={distances.max():.3f}  "
            f"p{BREAKPOINT_PERCENTILE:.0f} threshold={threshold:.3f}"
        )
        # The distance array with '*' marking every boundary the splitter
        # drew: the spike pattern is visible at a glance. Capped at
        # DISTANCE_PRINT_CAP so a ~1300-sentence book doesn't flood output.
        shown = distances[:DISTANCE_PRINT_CAP]
        marked = " ".join(
            f"{d:.2f}" + ("*" if d > threshold else "") for d in shown
        )
        n_more = len(distances) - len(shown)
        more_note = f" ... ({n_more} more)" if n_more > 0 else ""
        print(
            f"  distances (first {len(shown)} of {len(distances)}): "
            f"{marked}{more_note}   (* = boundary, distance > p95 threshold)"
        )
        for pair_no, i, distance, prev_sentence, next_sentence in boundary_pairs(
            sentences, distances, threshold
        ):
            print(
                f"  chunk pair {pair_no} -> {pair_no + 1}: boundary after sentence "
                f"{i + 1}, distance {distance:.3f} > p95 threshold {threshold:.3f} "
                f"(spike +{distance - threshold:.3f}, "
                f"{distance / distances.mean():.2f}x the mean distance)"
            )
            print(f"    ends: {preview(prev_sentence, SENTENCE_PREVIEW)}")
            print(f"    next: {preview(next_sentence, SENTENCE_PREVIEW)}")

    # --- Content preview: semantic chunks group related sentences -----------
    print("\n--- Content preview: semantic chunks group related sentences ---")
    # Same prose excerpt, two splitters: the semantic chunk keeps a stretch
    # of narrative (a chapter opening, a train of thought) whole because its
    # sentences embed closely; the recursive chunk cuts at a fixed 500
    # characters and can land mid-sentence.
    first_semantic = (
        semantic_chunks[1] if len(semantic_chunks) > 1 else semantic_chunks[0]
    )
    first_recursive = (
        recursive_chunks[1] if len(recursive_chunks) > 1 else recursive_chunks[0]
    )
    print(
        f"Semantic chunk [1] ({avg_words([first_semantic]):.0f} words, "
        f"source {first_semantic.metadata['source']}):"
    )
    print(f"  {preview(first_semantic.page_content, PREVIEW)}")
    print(
        f"Recursive chunk [1] ({avg_words([first_recursive]):.0f} words, "
        f"source {first_recursive.metadata['source']}):"
    )
    print(f"  {preview(first_recursive.page_content, PREVIEW)}")

    print(
        "\nTeaching takeaway: the semantic splitter draws a boundary at every "
        "sentence pair whose\ncosine distance clears the "
        f"{BREAKPOINT_PERCENTILE:.0f}th-percentile threshold — exactly where "
        "the topic shifts —\nwhile the recursive splitter cuts at a fixed "
        f"{CHUNK_SIZE} characters, blind to meaning."
    )


if __name__ == "__main__":
    main()
