"""Lab 01 — Fixed vs recursive text splitting.

The first decision in every RAG pipeline is: how do I cut a long document
into chunks? This lab runs the two classic answers head to head on the same
three public-domain Project Gutenberg novels — loaded through
``loaders/gutenberg.GutenbergLoader``, which strips the Gutenberg license
preamble/footer — with the same ``chunk_size``/``chunk_overlap``:

* FIXED splitting (``langchain_text_splitters.CharacterTextSplitter`` with
  ``separator=""``) is a pure character counter: it cuts at exactly
  ``chunk_size`` characters, no matter where that lands. Words and sentences
  get torn in half; the chunk count is purely a function of text length.
  (Note: ``CharacterTextSplitter`` defaults to a ``\\n\\n`` separator, which
  would quietly respect paragraphs — passing ``separator=""`` turns it into
  the blind character counter that "fixed splitting" really means.)
* RECURSIVE splitting (``splitters/recursive.DocumentProcessor`` →
  ``RecursiveCharacterTextSplitter``) climbs a ladder of separators —
  paragraphs (``\\n\\n``), newlines (``\\n``), spaces — and only falls back
  to characters when nothing else fits. Chunks end on paragraph/word
  boundaries instead of mid-word.

Why it matters: embeddings are trained on whole words and sentences. A chunk
that starts or ends mid-word feeds garbage tokens at exactly the boundary
points where neighbouring chunks meet, so retrieval quality degrades where
structure matters most.

Run from the repo root:
    python curriculum/01-chunking/01-fixed-vs-recursive.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/01-chunking/01-fixed-vs-recursive.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from loaders.gutenberg import GutenbergLoader  # noqa: E402
from splitters.recursive import DocumentProcessor  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the comparison
# --------------------------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PREVIEW = 100  # max characters shown on each side of a cut
DOC_PATHS = [
    Path("Data/corpus/gutenberg/pride-and-prejudice.txt"),
    Path("Data/corpus/gutenberg/moby-dick.txt"),
    Path("Data/corpus/gutenberg/a-tale-of-two-cities.txt"),
]


# --------------------------------------------------------------------------
# 2. Load — Gutenberg books via the shared loader
# --------------------------------------------------------------------------
def load_docs(paths: list[Path]) -> list[Document]:
    """Load each book through ``GutenbergLoader`` (strips the Gutenberg
    license preamble/footer, sets ``metadata["source"]`` to the book path)."""
    docs = []
    for path in paths:
        docs.extend(GutenbergLoader(path).load())
    return docs


# --------------------------------------------------------------------------
# 4. Compare — helpers for stats and boundary-quality detection
# --------------------------------------------------------------------------
def chunk_length_stats(chunks: list[Document]) -> tuple[int, float, int, int]:
    """Return (count, avg, min, max) chunk length in characters."""
    lengths = [len(c.page_content) for c in chunks]
    count = len(lengths)
    if count == 0:
        return 0, 0.0, 0, 0
    return count, sum(lengths) / count, min(lengths), max(lengths)


def escape(s: str) -> str:
    """Make newlines visible so cut positions are obvious in the output."""
    return s.replace("\n", "\\n")


def torn_word(tail: str, head: str) -> str:
    """Reconstruct the word torn across a cut from the letter runs on both sides."""
    trailing = []
    for ch in reversed(tail):
        if ch.isalnum():
            trailing.append(ch)
        else:
            break
    leading = []
    for ch in head:
        if ch.isalnum():
            leading.append(ch)
        else:
            break
    return "".join(reversed(trailing)) + "".join(leading)


def overlap_suffix(prev: str, next_: str) -> str:
    """Longest suffix of ``prev`` that is also a prefix of ``next_`` (the overlap repeat)."""
    for n in range(min(len(prev), len(next_)), 0, -1):
        if prev[-n:] == next_[:n]:
            return prev[-n:]
    return ""


# Gutenberg novels are plain text: structure shows up as "CHAPTER 1"/"CHAPTER I"
# headings, not markdown "#" headings.
CHAPTER_RE = re.compile(r"(?i)^chapter\s+[0-9ivxlcdm.]+")


def cut_stats(chunks: list[Document]) -> tuple[int, int]:
    """Return (in-document cuts, cuts where the next chunk opens a chapter heading)."""
    total = 0
    chapter_aligned = 0
    for i in range(len(chunks) - 1):
        if chunks[i].metadata.get("source") != chunks[i + 1].metadata.get("source"):
            continue
        total += 1
        if CHAPTER_RE.match(chunks[i + 1].page_content.lstrip()):
            chapter_aligned += 1
    return total, chapter_aligned


def find_midword_cut(chunks: list[Document]) -> tuple[int, str, str, str] | None:
    """First in-document cut where fixed splitting tears a word in half.

    Returns (index, tail_preview, fresh_head_preview, torn_word). The next
    chunk's preview has the ``CHUNK_OVERLAP`` repeat stripped so the reader
    sees the continuation of the torn word, not the duplicated tail.
    """
    for i in range(len(chunks) - 1):
        if chunks[i].metadata.get("source") != chunks[i + 1].metadata.get("source"):
            continue
        tail = chunks[i].page_content
        head = chunks[i + 1].page_content
        overlap = tail[-CHUNK_OVERLAP:]
        fresh = head[len(overlap):] if head.startswith(overlap) else head
        if tail and fresh and tail[-1].isalnum() and fresh[0].isalnum():
            return i, tail[-PREVIEW:], fresh[:PREVIEW], torn_word(tail, fresh)
    return None


def find_chapter_boundary(chunks: list[Document]) -> tuple[int, str, str] | None:
    """First in-document cut where the next chunk opens a chapter heading.

    Returns (index, tail_preview, head_preview). The tail has the overlap
    repeat stripped so the reader sees where the previous chunk truly ended.
    """
    for i in range(len(chunks) - 1):
        if chunks[i].metadata.get("source") != chunks[i + 1].metadata.get("source"):
            continue
        head = chunks[i + 1].page_content
        if CHAPTER_RE.match(head.lstrip()):
            tail = chunks[i].page_content
            overlap = overlap_suffix(tail, head)
            if overlap:
                tail = tail[: -len(overlap)]
            return i, tail[-PREVIEW:], head[:PREVIEW]
    return None


# --------------------------------------------------------------------------
# 5. Print the artifact — runnable demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 2. Load ---------------------------------------------------------
    docs = load_docs(DOC_PATHS)
    print("=" * 66)
    print("Lab 01 — fixed vs recursive text splitting")
    print(f"chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    print("=" * 66)
    print(f"\n[1] Loaded {len(docs)} document(s):")
    for doc in docs:
        print(f"    {Path(doc.metadata['source']).name:<22} {len(doc.page_content):>5} chars")

    # --- 3. Split --------------------------------------------------------
    fixed_chunks = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator=""
    ).split_documents(docs)
    recursive_chunks = DocumentProcessor(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_docs(docs)
    print("\n[2] Split with both splitters (same chunk_size/chunk_overlap):")
    print(f"    CharacterTextSplitter(separator='') : {len(fixed_chunks):>2} chunks")
    print(f"    DocumentProcessor (recursive)        : {len(recursive_chunks):>2} chunks")

    # --- 4. Compare lengths ----------------------------------------------
    f_count, f_avg, f_min, f_max = chunk_length_stats(fixed_chunks)
    r_count, r_avg, r_min, r_max = chunk_length_stats(recursive_chunks)
    print("\n[3] Chunk length stats (characters):")
    print(f"    {'':<12}{'chunks':>7}{'avg':>9}{'min':>7}{'max':>7}")
    print(f"    {'fixed':<12}{f_count:>7d}{f_avg:>9.1f}{f_min:>7d}{f_max:>7d}")
    print(f"    {'recursive':<12}{r_count:>7d}{r_avg:>9.1f}{r_min:>7d}{r_max:>7d}")

    # --- 5. Boundary quality ---------------------------------------------
    f_cuts, f_aligned = cut_stats(fixed_chunks)
    r_cuts, r_aligned = cut_stats(recursive_chunks)
    print("\n[4] Where do the cuts land?")
    print(f"    cuts landing at a chapter heading (next chunk opens a 'Chapter'): "
          f"fixed {f_aligned}/{f_cuts}, recursive {r_aligned}/{r_cuts}")

    cut = find_midword_cut(fixed_chunks)
    if cut is not None:
        i, tail, fresh, word = cut
        src = Path(fixed_chunks[i].metadata["source"]).name
        print(f"\n    FIXED cuts at exactly {CHUNK_SIZE} chars, mid-word:")
        print(f"      {src} chunk {i} ends    : ...{escape(tail)}")
        print(f"      {src} chunk {i + 1} (overlap stripped) starts: {escape(fresh)}...")
        print(f"      -> the word '{word}' is torn in half across chunks {i} and {i + 1}")
    else:
        print("\n    FIXED: no mid-word cut found (text too short or boundaries aligned).")

    boundary = find_chapter_boundary(recursive_chunks)
    if boundary is not None:
        j, tail, head = boundary
        src = Path(recursive_chunks[j].metadata["source"]).name
        print("\n    RECURSIVE climbs the separator ladder and lands on a chapter boundary:")
        print(f"      {src} chunk {j} ends  : ...{escape(tail)}")
        print(f"      {src} chunk {j + 1} starts: {escape(head)}...")
        print(f"      -> cut lands at a paragraph boundary; chunk {j + 1} opens a clean chapter")
    else:
        print("\n    RECURSIVE: no chapter boundary found (whole doc fits in one chunk).")

    print("\n[5] Takeaway")
    print("    Fixed splitting counts characters; recursive splitting counts")
    print("    structure. Same 500/50 budget, but recursive chunks keep words")
    print("    and paragraphs intact, so embeddings see clean text at every")
    print("    chunk boundary.")
