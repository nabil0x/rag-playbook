"""Structure-preserving HTML chunking with HTMLHeaderTextSplitter.

Lab 4 of track 01-chunking. The idea: HTML documents already carry their own
section hierarchy in the heading tags (``<h1>`` … ``<h4>``). Instead of
splitting on a fixed character/token budget and hoping a section boundary
lands inside a chunk, ``HTMLHeaderTextSplitter`` splits *on the headings
themselves*: every chunk maps to one document section, and the heading chain
that leads to it is stored as chunk metadata.

    HTML heading hierarchy  ->  chunk metadata  ->  section-scoped retrieval

A retriever can then filter on ``metadata["H2"] == "Liquidity and Capital
Resources"`` instead of hoping the right text happens to be in the top-k.

This lab runs the splitter on a real SEC 10-K filing (``aapl-20230930.htm``,
an inline-XBRL document). Real-world HTML is the point: SEC filings style
their headings as bold ``<span>`` elements, not ``<h1>``-``<h4>`` tags, so the
splitter finds no heading chain and collapses the whole filing into a single
chunk with empty metadata. The printed comparison shows exactly where the
``<title>`` and the styled top heading end up — and why structure-preserving
parsing for these documents needs XBRL/table-aware parsing (the SD-06 track).

See ``splitters/token_splitter.py`` for the sibling style; the plan is
``.omo/plans/layer1-rag-playbook.md``.
"""

from __future__ import annotations

from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import HTMLHeaderTextSplitter

# Module-level constants: the heading chain we split on, and the sample file.
HTML_PATH = "Data/SD-06-tables/aapl-20230930.htm"
HEADERS_TO_SPLIT_ON = [("h1", "H1"), ("h2", "H2"), ("h3", "H3")]
PREVIEW_CHARS = 200


def load_raw_html(path: str) -> str:
    """Read the raw HTML file as text.

    Plain ``open()`` on purpose: the splitter does the parsing, so the lab
    never touches the markup itself.
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_splitter() -> HTMLHeaderTextSplitter:
    """Build the splitter configured for the h1-h3 heading chain."""
    return HTMLHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)


def section_counts(chunks: list[Document], level: str) -> Counter[str]:
    """Count chunks per section, keyed by the metadata value for ``level``.

    Chunks with no value for that level (no matching heading tag in the
    document) fall into the ``"(no <tag> found)"`` bucket.
    """
    return Counter(
        chunk.metadata.get(level, f"(no <{level.lower()}> found)") for chunk in chunks
    )


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Collapse whitespace and cap a content preview at ~``limit`` chars."""
    return " ".join(text.split())[:limit]


if __name__ == "__main__":
    # --- setup -------------------------------------------------------------
    splitter = build_splitter()

    # --- load --------------------------------------------------------------
    html = load_raw_html(HTML_PATH)
    print(f"[1] Loaded raw HTML from {HTML_PATH} ({len(html):,} bytes)")

    # --- split -------------------------------------------------------------
    chunks = splitter.split_text(html)
    print(
        "[2] Split with HTMLHeaderTextSplitter("
        f"headers_to_split_on={HEADERS_TO_SPLIT_ON})"
    )
    print(f"    -> {len(chunks)} chunk(s)")

    # --- inspect metadata --------------------------------------------------
    print("\n[3] Chunk distribution across H1/H2 sections (from chunk metadata):")
    for level in ("H1", "H2"):
        counts = section_counts(chunks, level)
        print(f"    metadata[{level!r}]:")
        for section, count in counts.most_common():
            print(f"      {section!r}: {count} chunk(s)")

    print("\n[4] Metadata chain of the first few chunk(s):")
    for i, chunk in enumerate(chunks[:3]):
        print(f"    chunk {i}: metadata={chunk.metadata}")

    # --- print: content preview -------------------------------------------
    print("\n[5] Content preview (first ~200 chars of chunk 0):")
    if chunks:
        print(f"    {preview(chunks[0].page_content)}...")

    # --- print: comparison of <title> / top heading ------------------------
    print("\n[6] Where did the section headings go? (comparison)")
    print(
        "    - <title>aapl-20230930</title>: NOT a header (not in "
        "headers_to_split_on) and its text is dropped from the chunks "
        "entirely — the <head> is page metadata, not content."
    )
    if chunks:
        content = chunks[0].page_content
        top = content.find("UNITED STATES")
        if top != -1:
            print(
                "    - Top heading 'UNITED STATES / SECURITIES AND EXCHANGE "
                "COMMISSION / FORM 10-K' is a styled <span> (font-weight:700), "
                "NOT an <h1> tag -> it survives as plain text inside the single "
                "chunk but never becomes metadata."
            )
            print(f"      preview around it: {preview(content[top:top + 300])}...")

    # --- takeaway ----------------------------------------------------------
    print(
        "\n[7] Takeaway: HTML heading hierarchy becomes chunk metadata -> "
        "section-scoped retrieval — but only when the document uses real "
        "heading tags. SEC XBRL filings style headings as spans, so "
        "HTMLHeaderTextSplitter collapses the whole filing into one chunk "
        "with no metadata; structure-preserving parsing for these documents "
        "needs XBRL/table-aware parsing (the SD-06 track)."
    )