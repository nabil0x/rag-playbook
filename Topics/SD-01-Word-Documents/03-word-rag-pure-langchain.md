> Source notebook: `NoteBooks/Special-Documents/SD-01-Word-Documents/03-word-rag-pure-langchain.ipynb`

---

# Word RAG with Pure LangChain — From Zero to a Measured Retrieval Pipeline

**Goal:** Build a complete RAG pipeline for Word documents using only
LangChain-native tools, measure what works, and understand why.

```
Loader      : python-docx naive + structure-aware (docx tables → markdown)
Splitter    : RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
Embedding   : fastembed BAAI/bge-base-en-v1.5 (local ONNX, 768-dim)
Vector DB   : Chroma (local, embed-once-reuse)
Retriever   : Similarity / MMR / Hybrid (dense + BM25 + RRF)
Prompt      : n/a (recall measurement, no LLM)
LLM         : none — fully local benchmark
```

No API keys. No proprietary services. Everything runs on your CPU. This notebook
walks you through building a retrieval-augmented generation (RAG) pipeline from
scratch, using only pure LangChain packages and open-source tools. By the end
you will know which strategies work, which fail, and why -- and you will have
the code to reproduce every number in this article.

Learn:

* **Structure-aware loaders** -- why naive `.docx` loaders silently destroy table data
* **Chunk size and overlap** -- the tradeoffs that determine whether answers survive splitting
* **Cosine similarity and embeddings** -- meaning as 768-dimensional vectors
* **The embed-once-reuse pattern** -- avoiding redundant embedding across strategies
* **Hybrid retrieval** -- dense + BM25 + Reciprocal Rank Fusion
* **Measuring what matters** -- recall@5 as a binary per-question metric

---

### How to work through this notebook

The pipeline has six steps: load, chunk, embed, store, retrieve, evaluate. We
compare **three loading strategies** (naive-char, structured-unit, structured-char)
against **three retrievers** (similarity, MMR, hybrid) across 14 hand-checked
questions on 4 real Word documents. Sections 0--4 build the pipeline once.
Section 5 defines the retrieval helpers. Section 6 runs the full matrix on a
single document. Section 7 shows the complete 4-document benchmark. Section 8
tells the stories behind the numbers.

---

## 0 · Setup — local, pure-LangChain pipeline

Everything here is local. The imports pull from `langchain_core`, `langchain_text_splitters`,
`langchain_community`, and `langchain_chroma` -- no API keys needed. We resolve
paths to four sample `.docx` files living in `Data/SD-01-word/` and verify they
exist before we start.

---

```python
# SETUP: imports and file verification.
import os
import sys
import tempfile
from pathlib import Path

# LangChain core: Document type and Embeddings interface.
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# LangChain text splitting.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain community: BM25 keyword retriever.
from langchain_community.retrievers import BM25Retriever

# LangChain Chroma integration.
from langchain_chroma import Chroma
```

---

```python
# Resolve the repo-relative paths to the sample .docx files.
# The notebook lives in NoteBooks/Special-Documents/SD-01-Word-Documents/,
# so the repo root is two levels up.
REPO_ROOT = Path.cwd().parent.parent
DOCX_DIR = REPO_ROOT / "Data" / "SD-01-word"

print(f"REPO_ROOT : {REPO_ROOT}")
print(f"DOCX_DIR  : {DOCX_DIR}")
print(f"Exists    : {DOCX_DIR.exists()}")
```

---

```python
# Verify every expected sample file exists before we start.
expected_files = [
    "fcc-nationwide-eas-test-2021.docx",
    "docx4j-getting-started.docx",
    "epa-combustion-turbine-tsd.docx",
    "undp-bda-evaluation-2025.docx",
]
for name in expected_files:
    path = DOCX_DIR / name
    status = "OK" if path.exists() else "MISSING"
    print(f"  {status}: {name}")
```

---

The benchmark uses 14 hand-checked QA pairs across 4 documents. Each pair is a
(question, expected_answer_substring) tuple. The expected answer must appear
(case-insensitive) in at least one retrieved chunk for recall@1 to fire.

---

```python
# 14 hand-checked QA pairs across 4 Word documents.
# Each pair is (question, expected_answer_substring).
# The expected answer must appear (case-insensitive) in at least one
# retrieved chunk for recall@1.

QA = {
    "fcc-nationwide-eas-test-2021": [
        ("What was the filing rate for radio broadcasters?", "79.9%"),
        ("How many radio broadcasters participated in the nationwide test?", "13,320"),
        ("What percentage of radio broadcasters successfully retransmitted the alert?", "87.0%"),
        ("What was the most commonly reported complication?", "Audio Quality Issues"),
    ],
    "docx4j-getting-started": [
        ("Which Java class represents a .docx document in docx4j?", "WordprocessingMLPackage"),
        ("Which docx4j series is the last to run under Java 1.8?", "8.x"),
        ("What does docx4j use for logging?", "slf4j"),
        ("Can docx4j handle legacy binary .doc files?", "binary"),
    ],
    "epa-combustion-turbine-tsd": [
        ("What is the ISO base load of the LM6000 PC gas turbine?", "46.6"),
        ("What is the efficiency of the LM6000 PF gas turbine?", "41.4%"),
        ("What is the Docket ID for this technical support document?", "EPA-HQ-OAR-2023-0072"),
    ],
    "undp-bda-evaluation-2025": [
        ("When was the final evaluation report prepared?", "September 2025"),
        ("Who prepared the evaluation report?", "M&N Consultancy"),
        ("What is the name of the project being evaluated?", "Bakenyezi"),
    ],
}
```

---

```python
# Count total questions across all documents.
total_q = sum(len(v) for v in QA.values())
print(f"{len(QA)} documents, {total_q} questions")
```

---

## 1 · Load — naive vs structure-aware .docx loaders

A `.docx` file is not a single document. It is a **zip archive** containing
XML files. The main content lives in `word/document.xml`, which describes
paragraphs (`<w:p>` tags) and tables (`<w:tbl>` tags) in document order.

When you open a .docx in Word, you see paragraphs and tables interleaved.
But when you use a naive loader that only reads `doc.paragraphs` from
python-docx, it grabs every `<w:p>` tag it can find -- and silently drops
every `<w:tbl>` tag. The tables are not damaged or corrupted. They simply
do not exist in the output.

This is the most common mistake in Word-document RAG: the pipeline looks
like it works (you get chunks, embeddings, answers) until you test it
against a document with tables. Then recall drops to zero for every question
that asks for a table value, and you have no idea why.

We demonstrate this failure, then fix it with a structure-aware loader
that walks the XML tree in document order and preserves everything.

### The naive loader: fast but destructive

LangChain gives us `Docx2txtLoader` for free. It reads the .docx, extracts
all text, and returns `Document` objects. The problem: it flattens everything.
Paragraphs and table cells get mixed together, and the structure is gone.

We also write a naive loader using python-docx that reads only
`doc.paragraphs`. This is even simpler -- and even more destructive, because
it silently drops every table.

Both approaches share the same failure: if your question asks for a value
that lives in a table, the answer simply does not exist in the corpus. No
amount of clever retrieval can find something that was never loaded.

**The teaching point:** LangChain gives us a naive loader for free. When the
document has tables, we must write our own structure-aware loader. But it
still returns the same `Document` type, so the rest of the pipeline never
changes.

---

```python
# Naive loader: extracts paragraphs only, drops tables entirely.
# Uses python-docx to iterate doc.paragraphs, which only returns
# body-level paragraphs. Text inside table cells lives in a different
# part of the XML tree and is silently discarded.

from docx import Document as DocxDocument

def load_naive(path):
    """Load a .docx by extracting paragraphs only (tables are dropped)."""
    doc = DocxDocument(str(path))
    return [Document(page_content=p.text.strip(), metadata={"type": "paragraph"})
            for p in doc.paragraphs if p.text.strip()]

# Load the FCC document with the naive loader.
fcc_naive = load_naive(DOCX_DIR / "fcc-nationwide-eas-test-2021.docx")
print(f"Naive loader: {len(fcc_naive)} paragraphs from FCC doc")
print(f"First chunk preview: {fcc_naive[0].page_content[:100]}...")
```

---

### The structure-aware loader: keeping everything

To fix the table problem, we walk the XML body in document order.
python-docx gives us access to the raw XML through `doc.element.body.iterchildren()`.
Each child element is either a paragraph (`<w:p>`) or a table (`<w:tbl>`).

For paragraphs, we check the style name. If it starts with "Heading" followed
by a number, we prefix the text with that many `#` characters (markdown heading).
Otherwise, we keep it as plain text.

For tables, we walk each row and cell, building a markdown table with
`| --- |` separators. This turns the table into searchable text that
embedding and retrieval can match.

The key insight: this function returns the same `Document` type as the naive
loader. The rest of the pipeline (splitting, embedding, storing, retrieving)
does not care how the Documents were created. It just sees text and metadata.

---

```python
# Structure-aware loader: walks body in document order.
# Headings become markdown headers, tables become markdown tables.
# Returns langchain Document objects -- same type as the naive loader.

from docx.table import Table
from docx.text.paragraph import Paragraph
```

---

```python
# table_to_markdown: convert a docx table object into a markdown table.
# The first row becomes a header; the second row is the | --- | separator.
def table_to_markdown(tbl):
    """Convert a docx table object to a markdown table string."""
    lines = []
    for i, row in enumerate(tbl.rows):
        cells_list = [c.text.replace("\n", " ").strip() for c in row.cells]
        lines.append("| " + " | ".join(cells_list) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * len(cells_list)) + " |")
    return "\n".join(lines)
```

---

```python
def load_structured(path):
    """Load a .docx preserving headings, paragraphs, and tables in order."""
    doc = DocxDocument(str(path))
    out, heading = [], None
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            # Check if this paragraph is a heading.
            try:
                style = (para.style.name or "").strip()
            except Exception:
                style = ""
            level = int(style.split()[-1]) if style.lower().startswith("heading") and style.split()[-1].isdigit() else None
            if level:
                # Prefix with # characters for markdown heading.
                out.append(Document(page_content="#" * level + " " + text, metadata={"type": "heading", "heading": text}))
                heading = text
            else:
                meta = {"type": "paragraph"}
                if heading:
                    meta["heading"] = heading
                out.append(Document(page_content=text, metadata=meta))
        elif child.tag.endswith("}tbl"):
            # Convert table to markdown format.
            tbl = Table(child, doc)
            meta = {"type": "table"}
            if heading:
                meta["heading"] = heading
            out.append(Document(page_content=table_to_markdown(tbl), metadata=meta))
    return out
```

---

```python
# Load the FCC document with the structured loader.
fcc_structured = load_structured(DOCX_DIR / "fcc-nationwide-eas-test-2021.docx")
print(f"Structured loader: {len(fcc_structured)} units from FCC doc")
table_chunks = [c for c in fcc_structured if c.metadata.get("type") == "table"]
print(f"  of which {len(table_chunks)} are tables")
```

---

**What to look for:** The naive loader sees paragraphs and zero tables. The
structured loader sees headings, paragraphs, and tables. The tables contain
data like "79.9%", "13,320", and "Audio Quality Issues" that the naive
loader completely dropped.

Let us show you the difference on a real table.

---

```python
# Show a table chunk that the naive loader dropped entirely.
for c in fcc_structured:
    if c.metadata.get("type") == "table":
        print("=== First table (first 600 chars) ===")
        print(c.page_content[:600])
        break

print()
print(f"Naive sees {len(fcc_naive)} paragraphs, zero tables.")
print(f"Structured sees {len(fcc_structured)} units, {len(table_chunks)} tables.")
print("The naive loader silently lost every table.")
```

---

## 2 · Split — why chunk size and overlap matter

Large language models have a limited context window -- typically 4K to 128K
tokens. If you feed an entire 50-page document into the model, it will either
overflow or bury the answer in a sea of irrelevant text. Even if the model
could handle it, you would pay for embedding and processing every word.

**Chunking** is the solution: split the document into smaller pieces (chunks)
that each fit comfortably in the context window. Each chunk becomes one unit
in the vector store, and retrieval returns the 5 most relevant chunks instead
of the entire document.

### chunk_size vs chunk_overlap

- **chunk_size** (we use 1000 characters): the maximum length of each chunk.
  Too small and you lose context (a sentence without its paragraph makes no
  sense). Too large and you lose precision (the answer drowns in noise).
- **chunk_overlap** (we use 200 characters): how much text is shared between
  consecutive chunks. This prevents information loss at chunk boundaries.
  Without overlap, a sentence that spans two chunks would be split in half,
  and neither half would contain the complete thought.

The `RecursiveCharacterTextSplitter` tries to split on paragraphs first,
then sentences, then words -- keeping related text together whenever possible.

---

```python
# Demonstrate text splitting on a structured chunk.
# RecursiveCharacterTextSplitter tries paragraph breaks first, then
# sentences, then words. chunk_size=1000, chunk_overlap=200.

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Pick a long chunk from the structured loader to show splitting in action.
long_chunks = [c for c in fcc_structured if len(c.page_content) > 500 and c.metadata.get("type") == "paragraph"]
if long_chunks:
    sample = long_chunks[0]
    chunks = splitter.split_text(sample.page_content)
    print(f"Original text: {len(sample.page_content)} chars")
    print(f"After splitting: {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks):
        print(f"\n  chunk {i}: {len(chunk)} chars")
        print(f"  starts with: {chunk[:80]}...")
        print(f"  ends with: ...{chunk[-80:]}")
else:
    # Fallback: use the first structured chunk.
    sample = fcc_structured[0]
    chunks = splitter.split_text(sample.page_content)
    print(f"Original text: {len(sample.page_content)} chars")
    print(f"After splitting: {len(chunks)} chunk(s)")
```

---

## 3 · Embed — meaning into 768-dim vectors

An **embedding** is a list of numbers (a vector) that captures the meaning
of a piece of text. Similar texts get similar vectors. For example:

- "What was the filing rate?" and "What percentage filed?" would have
  nearby vectors because they mean similar things.
- "The weather is nice today" would be far away from both, because the
  meaning is completely different.

Our model (`BAAI/bge-base-en-v1.5`) produces 768-dimensional vectors.
That means each text becomes a list of 768 floating-point numbers.

### Cosine similarity

When we search the vector store, we compute the **cosine similarity**
between the query vector and every chunk vector. Cosine similarity measures
the angle between two vectors -- not their length, just their direction.
Two vectors pointing in the same direction have cosine similarity 1.0.
Vectors at right angles have similarity 0.0. Opposite directions give -1.0.

This is why embedding works: the query "filing rate" will have a small
angle (high cosine similarity) with a chunk about "79.9% filing rate",
and a large angle (low similarity) with a chunk about "audio quality".

### The .tolist() gotcha

Chroma rejects numpy scalar types. When fastembed returns a vector, each
element is a numpy float64. We must call `.tolist()` to convert to plain
Python floats. This is a common footgun -- without it, Chroma raises a
cryptic error about unsupported types.

---

```python
# LocalEmbedder: wraps fastembed with the LangChain Embeddings interface.
# Model: BAAI/bge-base-en-v1.5 (768 dimensions, runs on CPU).
# First run downloads ~130 MB from HuggingFace; subsequent runs use local cache.
# No API key needed. No rate limits. 100 chunks embed in ~0.3 seconds.

class LocalEmbedder(Embeddings):
    """Local ONNX embedder using fastembed + BAAI/bge-base-en-v1.5."""
    def __init__(self, model="BAAI/bge-base-en-v1.5"):
        from fastembed import TextEmbedding
        self._m = TextEmbedding(model)

    def embed_documents(self, texts):
        """Embed a list of document texts. Returns list of lists of floats.
        Note: .tolist() converts numpy float64 to plain Python floats.
        Chroma rejects numpy scalars, so this conversion is required.
        """
        return [v.tolist() for v in self._m.embed(texts, batch_size=64)]

    def embed_query(self, text):
        """Embed a single query string. Returns a list of floats.
        query_embed returns a generator; next() gets the first (only) result.
        .tolist() converts to plain Python floats (required by Chroma).
        """
        return next(self._m.query_embed(text)).tolist()
```

---

```python
# Instantiate the embedder and run a 1-query smoke test.
embedder = LocalEmbedder()

# Test: embed a sample query and inspect the result.
sample_vec = embedder.embed_query("What was the filing rate?")
print(f"Embedding dimension: {len(sample_vec)}")
print(f"First 5 values: {sample_vec[:5]}")
```

---

## 4 · Store — Chroma + the embed-once-reuse pattern

A **vector store** (or vector database) is a database optimized for
similarity search. Given a query vector, it finds the K nearest chunk
vectors using **approximate nearest neighbor (ANN)** search. This is
much faster than comparing the query against every chunk one by one.

We use **Chroma**, a lightweight vector store that runs locally with
no server needed.

### The embed-once-reuse pattern

When building multiple strategies (naive-char, structured-unit,
structured-char), many chunks have identical text. Instead of re-embedding
the same text multiple times, we embed every unique chunk text once and
reuse the vectors. This is the `Precomputed` pattern: a passthrough class
that returns pre-computed vectors instead of computing new ones.

Without this trick, the FCC demo would take minutes instead of seconds,
because we would embed ~500 chunks three times instead of embedding ~200
unique chunks once.

---

```python
# Precomputed: a passthrough Embeddings class that returns pre-computed vectors.
# This lets us embed every unique chunk text once, then reuse the vectors
# when building stores for different strategies.

class Precomputed(Embeddings):
    """Passthrough: returns precomputed vectors instead of computing new ones."""
    def __init__(self, vecs):
        self.vecs = vecs

    def embed_documents(self, texts):
        """Return the precomputed vectors (one per input text)."""
        return self.vecs

    def embed_query(self, text):
        """Query vectors must be computed separately via embedder.embed_query()."""
        raise NotImplementedError("Query vectors must be computed separately")

print("Precomputed class defined. We will use it in the demo cell below.")
print("It allows Chroma to accept pre-computed vectors without re-embedding.")
```

---

## 5 · Retrieve — similarity, MMR, and hybrid (dense + BM25 + RRF)

Once the chunks are stored as vectors, we need a way to find the most
relevant ones for a given question. We use three different retrieval
strategies, each with its own strengths.

### 1. Similarity search (baseline)

The simplest approach: compute the cosine similarity between the query
vector and every chunk vector, return the top 5. Fast, straightforward,
and a good baseline.

### 2. MMR -- Maximum Marginal Relevance (diversity)

The problem with plain similarity: sometimes the top 5 results are all
almost the same thing. You asked about "filing rate" and got 5 different
paragraphs that all say "79.9%". That is not useful.

MMR fixes this by balancing relevance against diversity. The `lambda_mult`
parameter controls the tradeoff:
- `lambda_mult = 1.0`: pure relevance (same as similarity)
- `lambda_mult = 0.0`: pure diversity (just pick different things)
- `lambda_mult = 0.7`: lean toward relevance but avoid duplicates

### 3. Hybrid -- Dense + BM25 + RRF (the robust default)

**BM25** is a keyword-based retrieval method. It does not understand
meaning -- it matches exact words. This sounds primitive, but it has a
crucial advantage: it catches exact identifiers like class names
(`WordprocessingMLPackage`), library names (`slf4j`), and docket numbers
(`EPA-HQ-OAR-2023-0072`) that dense embeddings might miss.

**Reciprocal Rank Fusion (RRF)** merges the dense (vector) and sparse
(BM25) ranked lists into one. For each document, RRF sums `1/(60 + rank)`
from both lists. A document found by both lists gets two contributions and
wins ties. The constant 60 comes from the original paper (Cormack et al. 2009).

---

### recall_at_k: the evaluation metric

For each question, does the expected answer substring appear in at least
one of the top-5 retrieved chunks? This is a binary metric (0 or 1 per
question), averaged across all questions to get a recall score per strategy.

---

```python
# recall_at_k: binary metric per question.
# Returns 1 if the expected answer substring appears in any retrieved chunk.

def recall_at_k(retrieved, expected):
    """Did the expected substring appear in any of the top-k chunks?

    Args:
        retrieved: list of Document objects from the retriever.
        expected: the expected answer string (e.g. "79.9%").
    Returns:
        1 if expected.lower() appears in any chunk's page_content.lower(), else 0.
    """
    return int(any(expected.lower() in d.page_content.lower() for d in retrieved))
```

---

```python
# rrf_fuse: merge two ranked lists using Reciprocal Rank Fusion.
# k=60 is the standard constant from Cormack et al. 2009.

def rrf_fuse(dense, sparse, k=60, top_k=5):
    """Merge dense and sparse ranked lists using Reciprocal Rank Fusion.

    For each document, sum 1/(k+rank) from both ranked lists.
    Documents found by both lists get two contributions and win ties.
    Deduplicate by page_content (exact text match).
    """
    scores = {}
    for ranked in (dense, sparse):
        for rank, doc in enumerate(ranked, 1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores, key=scores.get, reverse=True)
    by_key = {d.page_content: d for d in dense + sparse}
    return [by_key[key] for key in ranked[:top_k]]
```

---

```python
# Sanity check: recall_at_k must work correctly.
class _FakeDoc:
    def __init__(self, text):
        self.page_content = text

assert recall_at_k([_FakeDoc("filing rate was 79.9%")], "79.9%") == 1
assert recall_at_k([_FakeDoc("the filing rate was high")], "79.9%") == 0
print("recall_at_k: sanity checks passed")
```

---

### RRF formula, worked example

The RRF score for a document is:

```
score(doc) = sum over all ranked lists: 1 / (k + rank_of_doc_in_that_list)
```

With k=60:

| Rank | Contribution |
|------|-------------|
| 1    | 1/61 = 0.01639 |
| 2    | 1/62 = 0.01613 |
| 3    | 1/63 = 0.01587 |
| 5    | 1/65 = 0.01538 |
| 10   | 1/70 = 0.01429 |

A document found at rank 1 by both dense and sparse gets 0.01639 + 0.01639 = 0.03278.
A document found at rank 1 by dense but not in sparse gets only 0.01639.
This means documents that appear in both lists are strongly favored.

The constant k=60 prevents rank-1 documents from dominating too much.
Without k, rank 1 would contribute 1.0 and rank 2 would contribute 0.5,
making the fusion essentially pick only the top-1 from each list.

---

```python
# Tiny worked example: what does RRF do in practice?
# Suppose two retrievers return these ranked lists for "logging in docx4j":

dense_list = ["Paragraph about SLF4J configuration", "Code example with logging", "Unrelated chunk"]
sparse_list = ["SLF4J setup guide", "Paragraph about SLF4J configuration", "Unrelated chunk"]

print("RRF fusion example (k=60):")
print()

scores = {}
for label, ranked in [("dense", dense_list), ("sparse", sparse_list)]:
    for rank, text in enumerate(ranked, 1):
        score = 1.0 / (60 + rank)
        scores[text] = scores.get(text, 0.0) + score
        print(f"  {label} rank {rank}: 1/(60+{rank}) = {score:.5f}  '{text}'")

print()
print("Final RRF scores (sorted by score):")
for text, score in sorted(scores.items(), key=lambda x: -x[1]):
    in_both = score > 2 * (1.0 / 61) - 0.0001  # found by both lists?
    tag = " (found by BOTH lists)" if in_both else ""
    print(f"  {score:.5f}{tag}  '{text}'")
```

---

### Quick comparison: all three retrievers on one question

Before running the full benchmark, let us compare all three retrievers on
one question from the FCC document. This lets you see the actual returned
chunks, not just numbers.

We build a store with the structured-char strategy (160 chunks), embed
them once, then run similarity, MMR, and hybrid on the question
"What was the filing rate for radio broadcasters?".

---

```python
# Compare all three retrievers on one FCC question.
# This lets you see the actual returned chunks, not just recall numbers.

# Build chunks for the structured-char strategy.
fcc_chunks_char = splitter.split_documents(
    load_structured(DOCX_DIR / "fcc-nationwide-eas-test-2021.docx")
)
print(f"Building store with {len(fcc_chunks_char)} chunks (structured-char)...")

# Embed all chunks once.
fcc_vecs = embedder.embed_documents([c.page_content for c in fcc_chunks_char])

# Build Chroma store with precomputed vectors.
store_demo = Chroma.from_documents(
    fcc_chunks_char,
    embedding=Precomputed(fcc_vecs),
    collection_name="nb03_inline_demo",
    persist_directory=tempfile.mkdtemp(prefix="nb03_demo_"),
)
```

---

```python
# BM25 keyword retriever + embed the question.
bm25_demo = BM25Retriever.from_documents(fcc_chunks_char, k=5)

# Embed the question.
question = "What was the filing rate for radio broadcasters?"
expected = "79.9%"
query_vec = embedder.embed_query(question)
```

---

```python
# Retrieve with all three methods and print the results.
dense_demo = store_demo.similarity_search_by_vector(query_vec, k=5)
mmr_demo = store_demo.max_marginal_relevance_search_by_vector(query_vec, k=5, lambda_mult=0.7)
sparse_demo = bm25_demo.invoke(question)
hybrid_demo = rrf_fuse(dense_demo, sparse_demo)

print(f"\nQuestion: {question}")
print(f"Expected answer: {expected}\n")

for label, results in [("Similarity", dense_demo), ("MMR", mmr_demo), ("Hybrid (RRF)", hybrid_demo)]:
    print(f"--- {label} top-3 ---")
    for i, d in enumerate(results[:3], 1):
        hit = expected.lower() in d.page_content.lower()
        marker = "HIT" if hit else "   "
        print(f"  {i}. [{marker}] {d.page_content[:90]}...")
    print()
```

---

## 6 · Evaluate — the FCC strategy x retriever matrix

Now we run the full strategy x retriever matrix on the FCC document only.
That is 3 strategies x 3 retrievers x 4 questions = 36 retrievals.

> **Timing note:** This cell takes approximately 3-4 minutes on CPU. Most of
> the time is embedding ~200 unique chunks (the first-time model download of
> ~130 MB from HuggingFace adds a bit more). Subsequent runs are faster because
> the model is cached locally.

We measure **recall@5**: for each question, does the expected answer substring
appear in at least one of the top-5 retrieved chunks? This is a binary metric
(0 or 1 per question), averaged across the 4 questions to get a recall score.

The expected results:
- **naive-char**: 0.50 / 0.50 / 0.50 (sim/mmr/hyb)
- **structured-unit**: 0.75 / 0.75 / 0.75
- **structured-char**: 0.75 / 0.75 / 0.75

The structured loaders gain one extra hit (3/4 = 0.75 vs 2/4 = 0.50) because
they preserve table data that the naive loader drops. All three retrievers
score the same on this document because the FCC doc is small enough that
dense embedding handles it well -- the hybrid advantage shows up on the
docx4j document (see Part 7).

---

```python
# === DEMO: Full strategy x retriever matrix on the FCC document ===
# These cells build chunks, embed them, and measure recall for all combinations.
# It takes about 3-4 minutes on CPU.

import time

DEMO_DOC = "fcc-nationwide-eas-test-2021"
strategies = ["naive-char", "structured-unit", "structured-char"]

print(f"Demo: {DEMO_DOC}")
print(f"  strategies : {strategies}")
print(f"  questions  : {len(QA[DEMO_DOC])}")
print()
```

---

```python
# Phase 1: build chunks for all strategies.
def build_chunks(docname, strategy):
    """Build chunks for one (document, strategy) pair."""
    path = DOCX_DIR / f"{docname}.docx"
    if strategy == "naive-char":
        return splitter.split_documents(load_naive(path))
    elif strategy == "structured-unit":
        return load_structured(path)
    elif strategy == "structured-char":
        return splitter.split_documents(load_structured(path))
    raise ValueError(f"unknown strategy: {strategy!r}")

all_chunks = {}
for strategy in strategies:
    all_chunks[strategy] = build_chunks(DEMO_DOC, strategy)
```

---

```python
# Phase 2: embed all unique chunks once (embed-once-reuse pattern).
unique_texts = []
seen = set()
for strategy in strategies:
    for c in all_chunks[strategy]:
        if c.page_content not in seen:
            seen.add(c.page_content)
            unique_texts.append(c.page_content)

print(f"Unique chunks to embed: {len(unique_texts)}")
t0 = time.time()
unique_vecs = embedder.embed_documents(unique_texts)
embed_map = dict(zip(unique_texts, unique_vecs))
print(f"Embedded in {time.time() - t0:.1f}s")
```

---

```python
# Phase 3: for each strategy, build store + BM25, run all retrievers.
demo_results = {}
for strategy in strategies:
    chunks = all_chunks[strategy]
    vecs = [embed_map[c.page_content] for c in chunks]

    store = Chroma.from_documents(
        chunks,
        embedding=Precomputed(vecs),
        collection_name=f"nb03_{strategy}",
        persist_directory=tempfile.mkdtemp(prefix="nb03_"),
    )
    bm25 = BM25Retriever.from_documents(chunks, k=5)

    demo_results[strategy] = {"n_chunks": len(chunks)}
    print(f"\n[{strategy}] {len(chunks)} chunks")

    for q, expected in QA[DEMO_DOC]:
        qv = embedder.embed_query(q)
        dense = store.similarity_search_by_vector(qv, k=5)
        mmr = store.max_marginal_relevance_search_by_vector(qv, k=5, lambda_mult=0.7)
        sparse = bm25.invoke(q)
        hyb = rrf_fuse(dense, sparse)

        demo_results[strategy][q] = {
            "sim": recall_at_k(dense, expected),
            "mmr": recall_at_k(mmr, expected),
            "hyb": recall_at_k(hyb, expected),
        }
        r = demo_results[strategy][q]
        print(f"  Q: {q[:58]}")
        print(f"     exp={expected!r:18} sim={r['sim']} mmr={r['mmr']} hyb={r['hyb']}")
```

---

```python
# Print the recall grid.
print("\n--- Recall grid ---")
print(f"{'strategy':16} {'sim':>4} {'mmr':>4} {'hyb':>4}")
for strategy in strategies:
    nq = len(QA[DEMO_DOC])
    r = demo_results[strategy]
    sim = sum(r[q]["sim"] for q, _ in QA[DEMO_DOC]) / nq
    mmr = sum(r[q]["mmr"] for q, _ in QA[DEMO_DOC]) / nq
    hyb = sum(r[q]["hyb"] for q, _ in QA[DEMO_DOC]) / nq
    print(f"{strategy:16} {sim:.2f} {mmr:.2f} {hyb:.2f}")
```

---

## 7 · Full picture — all 4 documents

The FCC demo shows the measurement loop in action. But one document is not
enough. Here are the complete results across all 4 documents, measured in
a full benchmark run that took about 10 minutes of embedding time.

We present these as static data (no embedding needed to view them). The code
cell below prints the per-document table and grand totals.

---

```python
# Full 4-document benchmark results (pre-computed, static data).
# These numbers are from a complete run of all 4 documents x 3 strategies x 3 retrievers.
# The demo cell above reproduces the FCC rows; this dict shows the full picture.

FULL_RESULTS = {
    "fcc-nationwide-eas-test-2021": {
        "naive-char":       {"sim": 0.50, "mmr": 0.50, "hyb": 0.50, "n_chunks": 146},
        "structured-unit":  {"sim": 0.75, "mmr": 0.75, "hyb": 0.75, "n_chunks": 147},
        "structured-char":  {"sim": 0.75, "mmr": 0.75, "hyb": 0.75, "n_chunks": 160},
    },
    "docx4j-getting-started": {
        "naive-char":       {"sim": 0.50, "mmr": 0.50, "hyb": 0.75, "n_chunks": 935},
        "structured-unit":  {"sim": 0.50, "mmr": 0.50, "hyb": 0.75, "n_chunks": 944},
        "structured-char":  {"sim": 0.50, "mmr": 0.50, "hyb": 0.75, "n_chunks": 946},
    },
    "epa-combustion-turbine-tsd": {
        "naive-char":       {"sim": 0.33, "mmr": 0.33, "hyb": 0.33, "n_chunks": 156},
        "structured-unit":  {"sim": 1.00, "mmr": 1.00, "hyb": 1.00, "n_chunks": 142},
        "structured-char":  {"sim": 1.00, "mmr": 1.00, "hyb": 1.00, "n_chunks": 176},
    },
    "undp-bda-evaluation-2025": {
        "naive-char":       {"sim": 0.00, "mmr": 0.00, "hyb": 0.00, "n_chunks": 334},
        "structured-unit":  {"sim": 0.00, "mmr": 0.00, "hyb": 0.00, "n_chunks": 338},
        "structured-char":  {"sim": 0.00, "mmr": 0.00, "hyb": 0.00, "n_chunks": 383},
    },
}

print(f"FULL_RESULTS loaded: {len(FULL_RESULTS)} documents")
```

---

```python
# Per-document recall table.
print(f"{'Document':30} {'Strategy':16} {'Sim':>4} {'MMR':>4} {'Hyb':>4} {'Chunks':>6}")
print("-" * 80)
for docname, strats in FULL_RESULTS.items():
    for sname, vals in strats.items():
        print(f"{docname[:30]:30} {sname:16} {vals['sim']:4.2f} {vals['mmr']:4.2f} {vals['hyb']:4.2f} {vals['n_chunks']:6d}")
```

---

```python
# Grand totals: sum recall hits across all 14 questions.
# We need the number of questions per document to convert aggregate scores to hit counts.
nq_map = {
    "fcc-nationwide-eas-test-2021": 4,
    "docx4j-getting-started": 4,
    "epa-combustion-turbine-tsd": 3,
    "undp-bda-evaluation-2025": 3,
}
grand_totals = {}
for docname, strats in FULL_RESULTS.items():
    nq = nq_map[docname]
    for sname, vals in strats.items():
        for ret in ["sim", "mmr", "hyb"]:
            key = (sname, ret)
            # Convert aggregate score to hit count (round to avoid floating point issues).
            hits = round(vals[ret] * nq)
            grand_totals[key] = grand_totals.get(key, 0) + hits

print("\n--- Grand totals (all 14 questions) ---")
print(f"{'Strategy':16} {'Retriever':12} {'Recall':>8}")
for (sname, ret), hits in sorted(grand_totals.items()):
    print(f"{sname:16} {ret:12} {hits}/14 = {hits/14:.2f}")
```

---

## 8 · Case studies — where each strategy wins and fails

### Case study 1: EPA combustion turbine (tables are decisive)

The EPA document is 90% tables. Every question asks for a value that lives
exclusively in a table cell: ISO base load "46.6", efficiency "41.4%", and
docket "EPA-HQ-OAR-2023-0072".

| Strategy | Sim | MMR | Hyb | Chunks |
|---|---|---|---|---|
| naive-char | 0.33 | 0.33 | 0.33 | 156 |
| structured-unit | 1.00 | 1.00 | 1.00 | 142 |
| structured-char | 1.00 | 1.00 | 1.00 | 176 |

The naive loader drops every table. Those values simply do not exist in the
corpus. Only 1 of 3 questions is answered (from a paragraph mention). With
the structured loader, all 3 values are found. This is not a subtle
difference. It is the difference between finding the answer and not finding
it at all.

**Lesson:** When your document IS a table, a naive loader destroys it.
Structure-aware loading is not optional -- it is essential.

### Case study 2: docx4j (identifiers need keyword search)

The docx4j document is a technical guide full of class names, library names,
and version numbers. Two questions ask for exact identifiers:
`WordprocessingMLPackage` (a Java class name) and `slf4j` (a logging library).

| Strategy | Sim | MMR | Hyb | Chunks |
|---|---|---|---|---|
| naive-char | 0.50 | 0.50 | 0.75 | 935 |
| structured-unit | 0.50 | 0.50 | 0.75 | 944 |
| structured-char | 0.50 | 0.50 | 0.75 | 946 |

Dense embedding represents meaning, not exact strings. The vector for
"logging framework" does not embed close to the vector for "slf4j". But
BM25's lexical matching catches exact term overlaps, and RRF fusion
promotes those chunks into the top-5.

Notice: all three strategies score the same. The loader choice does not
matter here. What matters is the retriever: only hybrid (dense + BM25)
reaches 0.75.

**Lesson:** Dense embedding is blind to exact tokens. BM25 is not. Hybrid
retrieval wins for identifier-heavy technical documents.

### Case study 3: FCC "13,320" -- the answer exists but ranks too deep

The FCC document has a 146-row table. The cell containing "13,320" exists
in the corpus, but it ranks 13th in similarity (score 0.624). The top-5
is filled by descriptive prose: the heading (0.683), three paragraphs
(0.682, 0.670, 0.666). The exact table value is semantically less similar
to the question than the surrounding prose.

All strategies and retrievers fail on this question. The problem is not
the loader or the retriever -- it is that the answer chunk is simply
outnumbered by semantically similar but unhelpful prose.

| Question | Expected | Result |
|---|---|---|
| How many radio broadcasters participated? | 13,320 | 0 across all strategies |

**Lesson:** Retrieval quality is not just about the loader. Sometimes the
answer exists but is buried. Possible fixes: increase top_k, use
table-aware retrieval (query tables separately by metadata), or apply
re-ranking after initial retrieval.

### Case study 4: UNDP (title pollution kills everything)

All three UNDP questions score 0.00 across every strategy and every retriever.
The answers ("September 2025", "M&N Consultancy", "Bakenyezi") do exist in
the corpus -- we verified they are in the chunks. But they rank 23rd to
297th.

The problem: the ALL-CAPS cover page chunk "FINAL EVALUATION REPORT" is the
highest-similarity hit for every question (sim scores 0.872, 0.747, 0.730)
because the questions themselves contain "evaluation report". The cover chunk
crowds out the actual answers.

| Strategy | Sim | MMR | Hyb | Chunks |
|---|---|---|---|---|
| naive-char | 0.00 | 0.00 | 0.00 | 334 |
| structured-unit | 0.00 | 0.00 | 0.00 | 338 |
| structured-char | 0.00 | 0.00 | 0.00 | 383 |

This is not a bug. It is a real limitation of embedding-based retrieval when
a noisy structural element dominates similarity. No strategy or retriever
variant we tested rescues these answers.

**Lesson:** Noisy structure hurts all strategies. Inspect what actually gets
retrieved -- sometimes the problem is the data, not the pipeline.

---

## Bad methods we hit first, and how we overcame them

This notebook looks clean, but getting here took three dead ends. Each one
taught us something important.

### Dead end #1: Google Gemini free-tier embedding

We started with Google Gemini's free-tier embedding API. The free tier allows
roughly 100 embed requests per minute, each holding up to 100 texts.

The problem: when you exceed the quota, the API returns HTTP 429
`RESOURCE_EXHAUSTED` with a message like "Please retry in 40.96s". The
client retries 429s automatically with exponential backoff. But every retry
burns more of the same quota that is already exhausted -- so the backoff
can never succeed. The retries pile up, each one pushing you further over
the limit, creating a retry storm.

We hit this when embedding 4 documents x 3 strategies worth of chunks:
hundreds of unique texts, batched in groups of 100. The 429s started
immediately and the retries made them permanent.

**Fix:** Local ONNX embedding via fastembed. No API key, no rate limits,
768 dimensions, 100 chunks in ~0.3 seconds on CPU. We embed every unique
chunk text once (SHA-1 keyed), then pass precomputed vectors to Chroma.
Zero network calls during the entire benchmark.

### Dead end #2: Naive loader drops tables

The python-docx `doc.paragraphs` approach silently discards every table.
We did not notice until we tested the EPA document and got 0.33 recall --
1 of 3 questions answered. The two table-only questions had no answer in
the corpus at all.

**Fix:** The structure-aware loader that walks `doc.element.body.iterchildren()`
in document order, converting tables to markdown. This is the single most
impactful change in the entire pipeline: EPA went from 0.33 to 1.00.

### Dead end #3: Title pollution in UNDP

We assumed the UNDP 0.00 scores were a bug. After hours of debugging, we
realized the answers exist in the chunks but rank 23rd to 297th. The
ALL-CAPS cover page "FINAL EVALUATION REPORT" dominates every query's
similarity scores because the queries contain "evaluation report".

This is not a bug to fix -- it is a limitation to understand. The lesson:
always inspect what the retriever actually returns, not just the recall
score.

---

## What you should notice

* **Structure-aware loaders win on table-heavy documents.** EPA went from
  0.33 to 1.00 recall when we preserved tables as markdown. The naive loader
  silently destroys data you cannot afford to lose.
* **Hybrid retrieval wins on identifier-heavy documents.** docx4j hit 0.75
  recall with hybrid (dense + BM25) while pure dense stayed at 0.50. Exact
  tokens like `WordprocessingMLPackage` and `slf4j` need lexical matching.
* **No strategy rescues title-polluted documents.** UNDP scored 0.00 across
  every strategy and retriever. The ALL-CAPS cover chunk dominates similarity
  for every query containing "evaluation report".
* **The embed-once-reuse pattern saves real time.** Building three strategies
  over the same documents would triple the embedding cost without it.
  Precomputed vectors cut the FCC demo from minutes to seconds.
* **MMR with lambda_mult=0.7 leans toward relevance** but avoids returning
  5 nearly-identical chunks. A good default for most use cases.
* **Reciprocal Rank Fusion is simple and effective.** Sum 1/(60+rank) from
  both ranked lists. Documents found by both lists get two contributions and
  win ties. No tuning needed.
* **Recall@5 is a binary per-question metric** -- fast to compute, easy to
  interpret, and honest about what actually works. No LLM required.
* **Grand totals tell the story.** Naive loaders get 5--6 of 14 questions
  right. Structured loaders get 8--9. The gap comes entirely from table data.
* **Local embedding makes benchmarking possible.** No API keys, no rate
  limits, fully reproducible.

---

## Exercises

1. **Change chunk_size to 500.** Re-run the FCC demo cell. How do the
   chunk counts change? Does recall improve or worsen? Think about why
   smaller chunks lose context.

2. **Swap lambda_mult to 0.2.** In the inline retriever comparison cell,
   change `lambda_mult=0.7` to `lambda_mult=0.2`. How does the MMR
   output change? You should see more diverse but less relevant chunks.

3. **Add a 5th question to the FCC QA dict.** Pick a fact from the FCC
   document, write a question with an exact answer, add it to
   `QA["fcc-nationwide-eas-test-2021"]`, and re-run the demo. Which
   strategies answer it?

4. **Compare Docx2txtLoader with our naive loader.** Import
   `from langchain_community.document_loaders import Docx2txtLoader`,
   load the FCC doc, and compare chunk counts. How does it differ from
   our python-docx naive loader?

5. **Implement RRF with k=10 instead of k=60.** Copy the `rrf_fuse`
   function, change k to 10, and run it on the inline comparison.
   How does the output change? Think about why a smaller k amplifies
   rank-1 results.
