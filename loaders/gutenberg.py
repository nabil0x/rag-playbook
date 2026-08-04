"""Project Gutenberg plain-text loader.

Loader block: every loader returns `List[Document]` — nothing else in the
pipeline changes when you swap this block.

Gutenberg plain-text files (`https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt`)
ship wrapped in a fixed license preamble (the Project Gutenberg header/footer).
That boilerplate is noise for chunking and retrieval, so this loader strips
everything between the ``*** START OF THE PROJECT GUTENBERG EBOOK`` and
``*** END OF THE PROJECT GUTENBERG EBOOK`` markers by default.

Corpus provenance lives in `Data/.corpus-manifest.txt`; fetch new books with
`scripts/fetch_gutenberg.py`. Books are public domain.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"


def strip_gutenberg_boilerplate(text: str) -> str:
    """Return *text* with the Project Gutenberg preamble/footer removed.

    Slices from just after the ``*** START ...***`` marker to just before the
    ``*** END ...***`` marker. If either marker is missing the text is
    returned unchanged (defensive: some mirrors drop the footer), and the
    result is stripped of leading/trailing whitespace.
    """
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        return text.strip()
    return text[start + len(START_MARKER):end].strip()


class GutenbergLoader:
    """Load a Gutenberg plain-text book as a single Document.

    Args:
        path: path to a ``pg<id>.txt`` file (or any text file).
        strip: if True (default), remove the Gutenberg header/footer
            boilerplate; if False, keep the raw file content.
    """

    def __init__(self, path: str | Path, strip: bool = True):
        self.path = Path(path)
        self.strip = strip

    def load(self) -> list[Document]:
        text = self.path.read_text(encoding="utf-8")
        if self.strip:
            text = strip_gutenberg_boilerplate(text)
        return [
            Document(
                page_content=text,
                metadata={"source": str(self.path)},
            )
        ]


if __name__ == "__main__":
    # Self-check: loading a Gutenberg file yields one Document and the
    # boilerplate markers are gone from its content.
    import sys

    sample = Path("Data/corpus/gutenberg/pride-and-prejudice.txt")
    if not sample.exists():
        print("SKIP: no gutenberg corpus yet — run scripts/fetch_gutenberg.py first")
        sys.exit(0)
    docs = GutenbergLoader(sample).load()
    text = docs[0].page_content
    print(f"loaded {len(docs)} document(s) from {sample}")
    print(f"content: {len(text)} chars")
    print(f"boilerplate stripped: {START_MARKER not in text and END_MARKER not in text}")
    print(f"starts with: {text[:80]!r}")
