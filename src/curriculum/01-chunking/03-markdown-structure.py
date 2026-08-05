"""Structure-preserving markdown chunking.

Splitter block: chunk a markdown document by its header hierarchy instead of
by raw character counts. ``MarkdownHeaderTextSplitter`` cuts the text at every
``#`` / ``##`` / ``###`` heading and carries the heading chain into each
chunk's ``metadata`` (``{"H1": ..., "H2": ..., "H3": ...}``), so every chunk
knows which section it belongs to.

Why this matters for retrieval: a plain character splitter produces chunks
with no idea of their section, so a question about "persistence" can surface a
chunk from the wrong part of the doc. With header metadata, retrieval can
scope by section — e.g. filter to chunks whose ``H1 == "FAISS similarity
search"`` before ranking, or show the section path alongside every hit.

The primary demo here is the repo's own long markdown files — ``README.md``
and ``.omo/plans/layer1-rag-playbook.md`` — which have a real ``#``/``##``/
``###`` hierarchy worth splitting on. The tiny docs under
``Data/local-docs/`` stay as a second, side-by-side demo (their chunk indices
0–3 are stable, so the strip_headers comparison and section-scoped retrieval
blocks keep using them).

Compare against ``RecursiveCharacterTextSplitter`` (lab 01) and
``TokenTextSplitter`` (lab 02): those split on size, this one splits on
structure. See Topics/Project-04-Markdown-Documentation-RAG/README.md.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Module-level constant so the lab is tweakable: add ("####", "H4") to split
# deeper, or drop ("###", "H3") to merge H3 sections into their H2 parent.
HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

DOCS_DIR = Path("Data/local-docs/docs")
DOC_FILES = ["bge-embeddings.md", "faiss-search.md"]

# The repo's own long markdown files (paths relative to the repo root) — the
# primary demo documents with a real #/##/### hierarchy.
LONG_MD_PATHS = ["README.md", ".omo/plans/layer1-rag-playbook.md"]


def load_markdown(path: Path) -> str:
    """Read a markdown file as raw text (no loader needed for plain .md)."""
    return path.read_text(encoding="utf-8")


def distinct_h1_values(chunks: list[Document]) -> list[str]:
    """Distinct H1 values across chunks, in first-appearance order."""
    seen: list[str] = []
    for chunk in chunks:
        h1 = chunk.metadata.get("H1")
        if h1 is not None and h1 not in seen:
            seen.append(h1)
    return seen


def split_markdown(text: str, strip_headers: bool) -> list[Document]:
    """Split markdown text on its header hierarchy.

    Args:
        text: raw markdown content.
        strip_headers: if True the heading line is removed from the chunk
            content and kept only in metadata; if False the heading stays in
            the content too.

    Returns:
        List of Documents whose ``metadata`` carries the header chain
        (e.g. ``{"H1": "FAISS similarity search", "H2": "Persistence"}``).
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=strip_headers,
    )
    return splitter.split_text(text)


def preview(text: str, limit: int = 200) -> str:
    """Truncate a chunk's content for printing."""
    return text[:limit] + ("..." if len(text) > limit else "")


if __name__ == "__main__":
    # --- setup: one splitter config, two document sets -------------------
    print(f"Headers to split on: {HEADERS_TO_SPLIT_ON}")
    print(f"Local docs: {DOC_FILES}")
    print(f"Long repo docs: {LONG_MD_PATHS}\n")

    # --- load: raw markdown text from disk --------------------------------
    docs: dict[str, str] = {
        name: load_markdown(DOCS_DIR / name) for name in DOC_FILES
    }

    # --- split: structure-preserving, headers kept in content -------------
    # strip_headers=False (the default) keeps the heading line inside the
    # chunk, so the chunk is self-contained when read or embedded.
    chunks_kept: dict[str, list[Document]] = {
        name: split_markdown(text, strip_headers=False)
        for name, text in docs.items()
    }
    total_kept = sum(len(c) for c in chunks_kept.values())
    print(f"strip_headers=False -> {total_kept} chunk(s) across {len(docs)} file(s)")
    for name, chunks in chunks_kept.items():
        print(f"  {name}: {len(chunks)} chunk(s)")

    # --- inspect metadata: the header chain each chunk carries -------------
    first = chunks_kept[DOC_FILES[0]][0]
    print(f"\nFirst chunk of {DOC_FILES[0]} — metadata (header chain):")
    print(f"  {first.metadata}")
    print(f"  Content preview: {preview(first.page_content)!r}")

    # --- the real demo: the repo's own long markdown docs -----------------
    # README.md and the playbook plan are long, real markdown files with a
    # genuine #/##/### hierarchy — exactly what this splitter is for.
    long_chunks: dict[str, list[Document]] = {
        path: split_markdown(load_markdown(Path(path)), strip_headers=False)
        for path in LONG_MD_PATHS
    }
    total_long = sum(len(c) for c in long_chunks.values())
    print(f"\nLong repo markdown -> {total_long} chunk(s) across {len(LONG_MD_PATHS)} file(s)")
    for path, chunks in long_chunks.items():
        h1s = distinct_h1_values(chunks)
        print(f"  {path}: {len(chunks)} chunk(s), {len(h1s)} distinct H1 value(s)")

    readme_chunks = long_chunks["README.md"]
    print(f"\nFirst chunk of README.md — metadata (header chain):")
    print(f"  {readme_chunks[0].metadata}")
    print(f"  Content preview: {preview(readme_chunks[0].page_content)!r}")

    # --- side by side: strip_headers=False vs True for one section ---------
    # The same H2 section of faiss-search.md, split both ways.
    faiss_text = docs["faiss-search.md"]
    kept = split_markdown(faiss_text, strip_headers=False)
    stripped = split_markdown(faiss_text, strip_headers=True)
    # chunk 0 = "# FAISS similarity search" intro, chunk 1 = "## What it
    # does", chunk 2 = "## Persistence", chunk 3 = "## Querying".
    print("\nSide by side — '## Persistence' section of faiss-search.md:")
    print("  strip_headers=False (heading inside content):")
    print(f"    metadata: {kept[2].metadata}")
    print(f"    content : {preview(kept[2].page_content)!r}")
    print("  strip_headers=True  (heading only in metadata):")
    print(f"    metadata: {stripped[2].metadata}")
    print(f"    content : {preview(stripped[2].page_content)!r}")

    # --- the payoff: section-scoped retrieval ------------------------------
    # Because every chunk carries its H1/H2/H3 chain, a vector store can
    # filter before ranking — a fictional metadata-filter query:
    print("\nSection-scoped retrieval (fictional metadata filter):")
    print("  query = 'How do I persist a FAISS index?'")
    print("  filter = {'H1': 'FAISS similarity search'}  # scope to one doc")
    print("  -> only chunks whose metadata['H1'] matches are embedded/ranked,")
    print("     so a 'persistence' hit can never come from the BGE page.")
    print("\nTakeaway: structure-preserving splitting carries the section")
    print("hierarchy into chunk metadata — retrieval can scope by section.")