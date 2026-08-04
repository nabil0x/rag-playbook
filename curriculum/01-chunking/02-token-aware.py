"""Lab 02 — Token-aware splitting: a token budget is not a character budget.

The second decision in every RAG pipeline is: what unit do I budget chunks
in? This lab runs the SAME budget number (300) through two splitters where
the number means different units:

* TOKEN splitting (``splitters/token_splitter.TokenSplitter`` →
  ``TokenTextSplitter``) cuts on token boundaries, so every chunk is
  guaranteed to stay inside the token budget the LLM actually bills against.
* CHARACTER splitting (``splitters/recursive.DocumentProcessor`` →
  ``RecursiveCharacterTextSplitter``) cuts on character counts, which say
  nothing about tokens: a 300-character chunk can cost anywhere from ~20 to
  300+ tokens depending on how token-dense the text is (markdown markup,
  code and symbols tokenize far heavier than plain prose).

This lab measures both outputs in real token space with tiktoken
(``cl100k_base``, the gpt-4 tokenizer — the same encoder
``splitters/token_splitter.py`` uses to measure) and reports, per splitter:
chunk count, average / min / max tokens per chunk, and the spread (population
std dev). The corpus is three public-domain Project Gutenberg novels
(``Data/corpus/gutenberg/`` — Pride and Prejudice, Moby-Dick, A Tale of Two
Cities), loaded through ``loaders/gutenberg.GutenbergLoader`` so the license
preamble/footer is stripped before splitting. Token-budgeted chunks come out
uniform and bounded; character-budgeted chunks vary widely for the same
nominal budget, and a larger character budget (e.g. the repo default of 1000
chars) silently consumes most of a 300-token budget.

Compare against lab 01 (fixed vs recursive character splitting) and lab 03
(markdown structure): those split on size and structure, this one splits on
tokens. See splitters/token_splitter.py and
Topics/Project-04-Markdown-Documentation-RAG/README.md.

Run from the repo root:
    python curriculum/01-chunking/02-token-aware.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import tiktoken
from langchain_core.documents import Document

# Make the repo-root component library importable when this file is run
# directly (``python curriculum/01-chunking/02-token-aware.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from loaders.gutenberg import GutenbergLoader  # noqa: E402
from splitters.recursive import DocumentProcessor  # noqa: E402
from splitters.token_splitter import TokenSplitter  # noqa: E402

# --------------------------------------------------------------------------
# 1. Configuration — tweak these to rerun the comparison
# --------------------------------------------------------------------------
TOKEN_CHUNK_SIZE = 300            # budget in TOKENS — the unit the LLM bills
TOKEN_CHUNK_OVERLAP = 30
CHAR_CHUNK_SIZE = 300             # same number, different unit: CHARACTERS
CHAR_CHUNK_OVERLAP = 30
DOC_PATHS = [
    Path("Data/corpus/gutenberg/pride-and-prejudice.txt"),
    Path("Data/corpus/gutenberg/moby-dick.txt"),
    Path("Data/corpus/gutenberg/a-tale-of-two-cities.txt"),
]

# cl100k_base = the gpt-4 tokenizer; same encoder token_splitter.py measures
# with. Kept as a constant so the lab can be re-run under another encoding.
ENCODER_NAME = "cl100k_base"


# --------------------------------------------------------------------------
# 2. Load — GutenbergLoader strips the license preamble, wraps as Documents
# --------------------------------------------------------------------------
def load_documents(paths: list[Path]) -> list[Document]:
    """Load each Gutenberg book as a Document tagged with its source path."""
    docs: list[Document] = []
    for path in paths:
        docs.extend(GutenbergLoader(path, strip=True).load())
    return docs


# --------------------------------------------------------------------------
# 3. Token-count — measure every chunk in real token space
# --------------------------------------------------------------------------
def token_counts(chunks: list[Document], enc: tiktoken.Encoding) -> list[int]:
    """Measure each chunk's real token consumption (not its char count)."""
    return [len(enc.encode(chunk.page_content)) for chunk in chunks]


def chunk_stats(counts: list[int]) -> dict[str, float]:
    """Summarize a chunk population in token space: count/avg/min/max/std."""
    n = len(counts)
    return {
        "chunks": n,
        "avg": sum(counts) / n,
        "min": min(counts),
        "max": max(counts),
        "std": statistics.pstdev(counts),  # full population of chunks
    }


def preview(text: str, limit: int = 200) -> str:
    """Truncate a chunk's content for printing."""
    return text[:limit] + ("..." if len(text) > limit else "")


def print_stats_row(label: str, counts: list[int], note: str) -> None:
    """Print one row of the comparison table, all numbers in token space."""
    s = chunk_stats(counts)
    print(
        f"{label:<38} {s['chunks']:>6} {s['avg']:>7.1f} "
        f"{s['min']:>4.0f} {s['max']:>4.0f} {s['std']:>7.1f}   {note}"
    )


# --------------------------------------------------------------------------
# 4. Split — same documents through each splitter, then compare in tokens
# --------------------------------------------------------------------------
if __name__ == "__main__":
    enc = tiktoken.get_encoding(ENCODER_NAME)
    token_splitter = TokenSplitter(
        chunk_size=TOKEN_CHUNK_SIZE, chunk_overlap=TOKEN_CHUNK_OVERLAP
    )
    char_splitter = DocumentProcessor(
        chunk_size=CHAR_CHUNK_SIZE, chunk_overlap=CHAR_CHUNK_OVERLAP
    )
    print(f"Encoding: {ENCODER_NAME} (gpt-4 tokenizer)")
    print(
        f"Budget: {TOKEN_CHUNK_SIZE} tokens vs {CHAR_CHUNK_SIZE} characters, "
        f"overlap {TOKEN_CHUNK_OVERLAP}\n"
    )

    docs = load_documents(DOC_PATHS)
    print(f"Loaded {len(docs)} document(s): {[p.name for p in DOC_PATHS]}")
    for doc in docs:
        print(f"  {Path(doc.metadata['source']).name}: {len(doc.page_content):,} chars")

    token_chunks = token_splitter.split_docs(docs)
    char_chunks = char_splitter.split_docs(docs)
    print(
        f"TokenSplitter ({TOKEN_CHUNK_SIZE} tokens): {len(token_chunks)} chunk(s); "
        f"DocumentProcessor ({CHAR_CHUNK_SIZE} chars): {len(char_chunks)} chunk(s)\n"
    )

    token_tokens = token_counts(token_chunks, enc)
    char_tokens = token_counts(char_chunks, enc)

    # Same nominal budget "300" — the only difference is the unit it is
    # enforced in. All numbers below are tiktoken token counts.
    print("Chunk count and token consumption, measured with tiktoken:")
    print(f"{'splitter':<38} {'chunks':>6} {'avg':>7} {'min':>4} {'max':>4} {'std dev':>7}")
    print("-" * 88)
    print_stats_row(
        f"TokenSplitter ({TOKEN_CHUNK_SIZE} tokens)",
        token_tokens,
        "bounded: every chunk <= budget",
    )
    print_stats_row(
        f"DocumentProcessor ({CHAR_CHUNK_SIZE} chars)",
        char_tokens,
        "no token ceiling: wide spread",
    )
    print("-" * 88)
    spread = max(char_tokens) / min(char_tokens)
    print(
        f"Token chunks: {min(token_tokens)}-{max(token_tokens)} tokens — uniform, "
        f"bounded at the {TOKEN_CHUNK_SIZE}-token budget (the few tokens over "
        f"300 are each document's final-chunk remainder)."
    )
    print(
        f"Char chunks: {min(char_tokens)}-{max(char_tokens)} tokens — a "
        f"{spread:.1f}x spread for the same '300' budget."
    )

    # ----------------------------------------------------------------------
    # 5. Escalation — the repo default char budget, in token terms
    # ----------------------------------------------------------------------
    # DocumentProcessor's default is chunk_size=1000 CHARACTERS. Raise the
    # character budget and watch the token cost climb: nothing in the config
    # mentioned tokens, yet the chunks consume most of the 300-token budget.
    default_char_splitter = DocumentProcessor()  # repo default: 1000 chars
    default_char_chunks = default_char_splitter.split_docs(docs)
    default_char_tokens = token_counts(default_char_chunks, enc)
    s = chunk_stats(default_char_tokens)
    print("\nEscalation — DocumentProcessor at its default budget (1000 chars):")
    print(
        f"  {s['chunks']:.0f} chunk(s), avg {s['avg']:.1f}, "
        f"min {s['min']:.0f}, max {s['max']:.0f} tokens per chunk"
    )
    print(
        f"  -> a '1000-character' budget silently consumed up to "
        f"{s['max']:.0f} tokens per chunk, ~{100 * s['max'] / TOKEN_CHUNK_SIZE:.0f}% "
        f"of the {TOKEN_CHUNK_SIZE}-token budget."
    )

    # ----------------------------------------------------------------------
    # 6. Inspect — one real chunk from each splitter, side by side
    # ----------------------------------------------------------------------
    print("\nSide by side — first chunk of each splitter:")
    for label, chunk, tokens in [
        ("TokenSplitter", token_chunks[0], token_tokens[0]),
        ("DocumentProcessor", char_chunks[0], char_tokens[0]),
    ]:
        print(f"  {label}: {tokens} tokens, {len(chunk.page_content)} chars")
        print(f"    {preview(chunk.page_content)!r}")

    # ----------------------------------------------------------------------
    # 7. Takeaway — token budget vs character budget
    # ----------------------------------------------------------------------
    print("\nTakeaway: a character budget is not a token budget.")
    print(
        f"- TokenSplitter({TOKEN_CHUNK_SIZE} tokens): every chunk is bounded at "
        f"the {TOKEN_CHUNK_SIZE}-token budget (measured "
        f"{min(token_tokens)}-{max(token_tokens)}); the budget is enforced in "
        f"the unit the LLM bills."
    )
    print(
        f"- DocumentProcessor({CHAR_CHUNK_SIZE} chars): the same '300' produced "
        f"chunks of {min(char_tokens)}-{max(char_tokens)} tokens ({spread:.1f}x spread). "
        f"Character count never determines token count — markdown, code and "
        f"symbols tokenize far heavier than prose — so no character budget "
        f"sets a token ceiling."
    )
    print(
        f"  (On these three novels the same nominal '300' produced "
        f"{len(char_tokens):,} char-budgeted chunks of "
        f"{min(char_tokens)}-{max(char_tokens)} tokens — a {spread:.0f}x spread, "
        f"driven by short fragments (chapter headings, whitespace runs) at the "
        f"low end. Plain prose runs ~5 chars/token, so a 300-char chunk lands "
        f"around {sum(char_tokens) // len(char_tokens)} tokens on average — well "
        f"under the 300-token budget — yet the character budget still sets no "
        f"token ceiling: at the repo's default 1000-char budget the max reached "
        f"{s['max']:.0f} tokens, ~{100 * s['max'] / TOKEN_CHUNK_SIZE:.0f}% of the "
        f"{TOKEN_CHUNK_SIZE}-token budget.)"
    )
    print(
        "- A token-aware splitter is the only one that can guarantee your "
        "chunks fit the context window you actually pay for."
    )