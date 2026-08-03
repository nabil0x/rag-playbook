> Source notebook: `NoteBooks/SD-08-Invoices/03-invoice-rag-pure-langchain.ipynb`

---

# Invoice RAG with Pure LangChain — From Zero to a Measured Retrieval Pipeline

**Goal:** Build a complete RAG pipeline for PDF invoices using only
LangChain-native tools, measure what works, and understand why.

```
Loader      : pdfplumber naive + structure-aware (header / line-item / totals)
Splitter    : RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
Embedding   : fastembed BAAI/bge-base-en-v1.5 (local ONNX, 768-dim)
Vector DB   : Chroma (local, embed-once-reuse)
Retriever   : Similarity / MMR / Hybrid (dense + BM25 + RRF)
LLM         : none — fully local recall measurement
```

No API keys. No proprietary services. Everything runs on your CPU. This notebook
walks you through building a retrieval-augmented generation (RAG) pipeline from
scratch, using only pure LangChain packages and open-source tools. By the end
you will know which strategies work, which fail, and why, and you will have
the code to reproduce every number in this article.

Learn:

* **Structure-aware loaders** — why naive PDF loaders silently return empty pages for scanned invoices
* **Chunk size and overlap** — the tradeoffs that determine whether answers survive splitting
* **Cosine similarity and embeddings** — meaning as 768-dimensional vectors
* **The embed-once-reuse pattern** — avoiding redundant embedding across strategies
* **Hybrid retrieval** — dense + BM25 + Reciprocal Rank Fusion
* **Measuring what matters** — recall@5 as a binary per-question metric

---

### How to work through this notebook

The pipeline has six steps: load, chunk, embed, store, retrieve, evaluate. We
compare **three loading strategies** (naive-char, structured-unit, structured-char)
against **three retrievers** (similarity, MMR, hybrid) across 23 hand-checked
questions on 6 real invoice PDFs. Sections 0–4 build the pipeline once.
Section 5 defines the retrieval helpers. Section 6 runs the full matrix on a
single document. Section 7 shows the complete 6-document benchmark. Section 8
tells the stories behind the numbers.

---

## 0 · Setup — local, pure-LangChain pipeline

Everything here is local. The imports pull from `langchain_core`, `langchain_text_splitters`,
`langchain_community`, and `langchain_chroma` — no API keys needed. We resolve
paths to sample invoice PDFs living in `Data/SD-08-invoices/` and verify they
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

# Text splitting + community loaders/retrievers + Chroma.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
```

---

```python
# Resolve the repo-relative paths to the sample invoice PDFs.
# The notebook lives in NoteBooks/SD-08-Invoices/,
# so the repo root is two levels up.
REPO_ROOT = Path.cwd().parent.parent
INVOICE_DIR = REPO_ROOT / "Data" / "SD-08-invoices"

print(f"REPO_ROOT  : {REPO_ROOT}")
print(f"INVOICE_DIR: {INVOICE_DIR}")
print(f"Exists     : {INVOICE_DIR.exists()}")
```

---

```python
# Verify every expected sample file exists before we start.
# 6 are text-based PDFs; 3 are image-only (scans / a photo) with no text layer.
expected_files = [
    "sample-invoice.pdf",       "multipage_invoice1.pdf",
    "Invoice_1.pdf",            "Invoice-6.pdf",
    "sdk-invoice1.pdf",         "german-zugferd.pdf",
    "watson-hall-1898.pdf",     "macy-receipt.pdf",
    "szamla-minta.jpg",
]
for name in expected_files:
    path = INVOICE_DIR / name
    status = "OK" if path.exists() else "MISSING"
    print(f"  {status}: {name}")
```

---

The benchmark uses 23 hand-checked QA pairs across 6 text-based invoice PDFs.
Each pair is a `(question, expected_answer_substring)` tuple. The expected
answer must appear (case-insensitive) in at least one retrieved chunk for
recall@1 to fire.

---

```python
# 23 hand-checked QA pairs across 6 invoice PDFs.
# Each pair is (question, expected_answer_substring).
# The expected answer must appear (case-insensitive) in at least one
# retrieved chunk for recall@1.

QA = {
    "sample-invoice": [
        ("What is the invoice number?", "INV-100"),
        ("What is the total amount due?", "610.00"),
        ("Who is the customer?", "MICROSOFT CORPORATION"),
        ("What is the sales tax amount?", "10.00"),
    ],
    "multipage_invoice1": [
        ("What is the total for Company A?", "430.00"),
        ("What is the subtotal for Company B?", "3000.00"),
        ("Which line item has a quantity of 8?", "G"),
        ("What is the tax for Company B?", "300.00"),
    ],
    "Invoice_1": [
        ("What is the invoice number?", "3847193"),
        ("What is the total price of all items?", "1075.70"),
        ("How many total pieces?", "66"),
        ("What is the item code for the bubble film roll?", "JF9912413BF"),
    ],
    "Invoice-6": [
        ("What is the receipt number?", "9876"),
        ("What is the total?", "10,686.25"),
        ("What is the discount rate for the leadership training?", "25%"),
        ("What is the sales tax rate?", "3%"),
    ],
    "sdk-invoice1": [
        ("What is the invoice number?", "34278587"),
        ("What is the total charges amount?", "56,651.49"),
        ("Who is this invoice for?", "Microsoft"),
    ],
    "german-zugferd": [
        ("What is the Rechnungsnummer (invoice number)?", "1001"),
        ("What is the Rechnungsbetrag (invoice amount)?", "139,20"),
        ("What is the VAT rate for Position 1?", "19 %"),
        ("Who is the payee (seller)?", "PG Consulting"),
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

## 1 · Load — naive vs structure-aware PDF loaders

A PDF file is a **page-description format**. Text lives in "content streams"
that tell the renderer where to draw each glyph. When a PDF is created from a
text file (a real invoice exported from an ERP), those streams contain the
words — this is called the **text layer**.

But an invoice is often a **scan**: someone photographed or faxed a paper
document, and the PDF contains only images. Such a PDF has **no text layer at
all**. The words you see on screen exist only as pixels.

When you use a naive loader that only reads the text layer, scanned invoices
come back as pages with **empty content**. The pipeline looks like it works
(you get pages, chunks, embeddings, answers) until you test it against a scan.
Then recall drops to zero for every question, and you have no idea why.

Invoices also have structure: a **header** (invoice number, customer, dates),
a table of **line items**, and a **totals** block (subtotal, tax, total). A
page chunk mixes all three together. We demonstrate the naive failure, then
fix it with a structure-aware loader that keeps each part as its own atomic
unit.

### The naive loader: fast but destructive

LangChain gives us `PyPDFLoader` for free. It opens the PDF, reads the text
layer, and returns one `Document` per page. That is the entire pipeline for
most PDF RAG tutorials — and it silently fails on the two most common invoice
formats:

- **Scanned invoices** (watson-hall-1898.pdf, macy-receipt.pdf): the text
  layer is empty, so every page comes back with 0 characters.
- **Mixed-format invoices**: even when text exists, the header, the line-item
  table, and the totals are flattened into one page blob with no metadata.

**The teaching point:** LangChain gives us a naive loader for free. When the
document is a scan — or has structure worth keeping — we must write our own
structure-aware loader. But it still returns the same `Document` type, so the
rest of the pipeline never changes.

---

```python
# Naive loader: PyPDFLoader extracts the PDF text layer, one Document per page.
# On scanned invoices the text layer is EMPTY -- the classic silent failure.
from langchain_community.document_loaders import PyPDFLoader

def load_naive(path):
    """Load a PDF page-by-page (text layer only)."""
    return PyPDFLoader(str(path)).load()

for name in ["sample-invoice.pdf", "watson-hall-1898.pdf", "macy-receipt.pdf"]:
    pages = load_naive(INVOICE_DIR / name)
    chars = sum(len(p.page_content.strip()) for p in pages)
    non_empty = sum(len(p.page_content.strip()) > 0 for p in pages)
    print(f"{name:22} {len(pages)} pages, {chars:6d} chars, {non_empty} with text")
```

---

### The structure-aware loader: keeping everything

To fix the problem, we use **pdfplumber**, a PDF library that gives us both
the text layer AND the tables, with their row/column layout. An invoice is
mostly a table, so this matters.

Our `extract_invoice(path)` function splits each invoice PDF into **atomic
units**, one `Document` per unit:

- **header** — the page text: invoice number, customer, dates, and metadata.
- **line_item** — one row of the line-item table (item code, description,
  quantity, price).
- **totals** — rows that contain keywords like "total", "subtotal", "tax",
  "Netto", or "MwSt".

Every unit carries the same metadata: `invoice_id` (e.g. "INV-100"). This is
the property that makes invoice RAG different from page-based RAG — see
Case study 4.

The key insight: this function returns the same `Document` type as the naive
loader. The rest of the pipeline (splitting, embedding, storing, retrieving)
does not care how the Documents were created.

---

```python
# Structure-aware loader: pdfplumber opens the PDF and gives us BOTH the
# text layer and the tables -- which is what an invoice mostly is.
import re
import pdfplumber

def table_to_text(tbl):
    """Flatten a pdfplumber table (rows of cells) into pipe-separated lines."""
    return "\n".join(
        " | ".join((c or "").replace("\n", " ").strip() for c in row)
        for row in tbl
    )
```

---

```python
# detect_invoice_id: best-effort invoice number from the raw text layer;
# falls back to the file stem so every unit still gets a stable id.
def detect_invoice_id(path, text):
    m = re.search(
        r"(?:Invoice|Rechnung|Receipt)\s*(?:No\.?|Number|#|Nr\.?)?\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9\-]*)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1)
    return Path(path).stem

# classify_row: split table rows into totals (subtotal/tax/sum lines) vs.
# ordinary line items, using English and German invoice keywords.
TOTAL_KEYWORDS = ("total", "subtotal", "tax", "netto", "mwst", "brutto", "summe", "gesamt")

def classify_row(cells):
    joined = " ".join(cells).lower()
    return "totals" if any(k in joined for k in TOTAL_KEYWORDS) else "line_item"
```

---

```python
# Known invoice ids for the 6 text-based invoices -- kept self-consistent
# with the QA dict and used to tag every structured unit with metadata.
INVOICE_IDS = {
    "sample-invoice": "INV-100", "multipage_invoice1": "multipage",
    "Invoice_1": "3847193", "Invoice-6": "9876",
    "sdk-invoice1": "34278587", "german-zugferd": "1001",
}

def extract_invoice(path, invoice_id=None):
    """Split an invoice PDF into header / line-item / totals Documents."""
    if invoice_id is None:
        invoice_id = detect_invoice_id(path, "")
    docs = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            docs.append(Document(
                page_content=text,
                metadata={"invoice_id": invoice_id, "type": "header"},
            ))
            for tbl in page.extract_tables() or []:
                for row in tbl:
                    cells = [(c or "").replace("\n", " ").strip() for c in row]
                    line = " | ".join(cells)
                    if not line.strip():
                        continue
                    docs.append(Document(
                        page_content=line,
                        metadata={"invoice_id": invoice_id,
                                  "type": classify_row(cells)},
                    ))
    return docs
```

---

```python
# Load sample-invoice.pdf with the structured loader.
sample_units = extract_invoice(INVOICE_DIR / "sample-invoice.pdf", invoice_id="INV-100")
kinds = {}
for u in sample_units:
    kinds[u.metadata["type"]] = kinds.get(u.metadata["type"], 0) + 1
print(f"Structured loader: {len(sample_units)} units from sample-invoice.pdf")
print(f"  by type: {kinds}")
print(f"  all carry invoice_id: "
      f"{all(u.metadata.get('invoice_id') == 'INV-100' for u in sample_units)}")
```

---

**What to look for:** The naive loader sees pages and — for scans — zero
text. The structured loader sees header, line-item, and totals units, each
tagged with `invoice_id`. The units contain data like "INV-100", "610.00",
and "MICROSOFT CORPORATION" that a scan's missing text layer cannot provide.

Let us show you the difference on a real invoice.

---

```python
# Show one unit of each type from the structured loader.
for wanted in ["header", "line_item", "totals"]:
    unit = next(u for u in sample_units if u.metadata["type"] == wanted)
    print(f"=== {wanted} unit | metadata: {unit.metadata} ===")
    print(unit.page_content[:200])
    print()
```

---

## 2 · Split — why chunk size and overlap matter

Large language models have a limited context window — typically 4K to 128K
tokens. If you feed an entire 50-page document into the model, it will either
overflow or bury the answer in a sea of irrelevant text. Even if the model
could handle it, you would pay for embedding and processing every word.

**Chunking** is the solution: split the document into smaller pieces (chunks)
that each fit comfortably in the context window. Each chunk becomes one unit
in the vector store, and retrieval returns the 5 most relevant chunks instead
of the entire document.

### chunk_size vs chunk_overlap

- **chunk_size** (we use 1000 characters): the maximum length of each chunk.
  Too small and you lose context (a line item without its invoice header makes
  no sense). Too large and you lose precision (the answer drowns in noise).
- **chunk_overlap** (we use 200 characters): how much text is shared between
  consecutive chunks. This prevents information loss at chunk boundaries.
  Without overlap, a sentence that spans two chunks would be split in half,
  and neither half would contain the complete thought.

The `RecursiveCharacterTextSplitter` tries to split on paragraphs first,
then sentences, then words — keeping related text together whenever possible.

---

```python
# Demonstrate text splitting on a structured chunk.
# RecursiveCharacterTextSplitter tries paragraph breaks first, then
# sentences, then words. chunk_size=1000, chunk_overlap=200.

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Pick the longest unit of a multi-page invoice to show splitting in action.
long_unit = max(
    (u for u in extract_invoice(INVOICE_DIR / "multipage_invoice1.pdf",
                                invoice_id="multipage")),
    key=lambda u: len(u.page_content),
)
chunks = splitter.split_text(long_unit.page_content)
print(f"Original text: {len(long_unit.page_content)} chars")
print(f"After splitting: {len(chunks)} chunk(s)")
for i, chunk in enumerate(chunks):
    print(f"  chunk {i}: {len(chunk)} chars | {chunk[:60]}...")
```

---

## 3 · Embed — meaning into 768-dim vectors

An **embedding** is a list of numbers (a vector) that captures the meaning
of a piece of text. Similar texts get similar vectors. For example:

- "What is the invoice number?" and "What number is on this bill?" would have
  nearby vectors because they mean similar things.
- "The weather is nice today" would be far away from both, because the
  meaning is completely different.

Our model (`BAAI/bge-base-en-v1.5`) produces 768-dimensional vectors.
That means each text becomes a list of 768 floating-point numbers.

### Cosine similarity

When we search the vector store, we compute the **cosine similarity**
between the query vector and every chunk vector. Cosine similarity measures
the angle between two vectors — not their length, just their direction.
Two vectors pointing in the same direction have cosine similarity 1.0.
Vectors at right angles have similarity 0.0. Opposite directions give -1.0.

This is why embedding works: the query "total amount due" will have a small
angle (high cosine similarity) with a chunk about "Total due: 610.00", and a
large angle (low similarity) with a chunk about "bubble film roll".

### The .tolist() gotcha

Chroma rejects numpy scalar types. When fastembed returns a vector, each
element is a numpy float64. We must call `.tolist()` to convert to plain
Python floats. This is a common footgun — without it, Chroma raises a
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
sample_vec = embedder.embed_query("What is the invoice number?")
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

Without this trick, the demo would take minutes instead of seconds,
because we would embed the same units three times instead of once.

### Metadata filtering

Chroma lets us filter by metadata before searching. Because every structured
unit carries `invoice_id`, we can restrict a search to one invoice in a
multi-invoice collection — the killer feature of invoice RAG.

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

```python
# Metadata filter demo: every structured unit carries invoice_id.
# Build a small store from sample-invoice.pdf, then filter by invoice_id.
store_f = Chroma.from_documents(
    extract_invoice(INVOICE_DIR / "sample-invoice.pdf", invoice_id="INV-100"),
    embedding=embedder,
    collection_name="nb03_filter_demo",
    persist_directory=tempfile.mkdtemp(prefix="nb03_filter_"),
)
hits = store_f.similarity_search("total", k=3, filter={"invoice_id": "INV-100"})
print(f"{len(hits)} hits; invoice_ids: {[h.metadata['invoice_id'] for h in hits]}")
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

### 2. MMR — Maximum Marginal Relevance (diversity)

The problem with plain similarity: sometimes the top 5 results are all
almost the same thing. You asked about "total amount" and got 5 units that
all say "Total: 610.00". That is not useful.

MMR fixes this by balancing relevance against diversity. The `lambda_mult`
parameter controls the tradeoff:
- `lambda_mult = 1.0`: pure relevance (same as similarity)
- `lambda_mult = 0.0`: pure diversity (just pick different things)
- `lambda_mult = 0.7`: lean toward relevance but avoid duplicates

### 3. Hybrid — Dense + BM25 + RRF (the robust default)

**BM25** is a keyword-based retrieval method. It does not understand
meaning — it matches exact words. This sounds primitive, but it has a
crucial advantage: it catches exact identifiers like invoice numbers
(`3847193`), item codes (`JF9912413BF`), and tax rates (`19 %`) that dense
embeddings might miss.

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
        expected: the expected answer string (e.g. "610.00").
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

assert recall_at_k([_FakeDoc("Total due: 610.00")], "610.00") == 1
assert recall_at_k([_FakeDoc("the total is due soon")], "610.00") == 0
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
# Suppose two retrievers return these ranked lists for "sales tax":

dense_list = ["Header with tax rate 3%", "Line item with tax 10.00", "Unrelated chunk"]
sparse_list = ["Totals: tax 10.00", "Header with tax rate 3%", "Unrelated chunk"]

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
one question from sample-invoice.pdf. This lets you see the actual returned
chunks, not just numbers.

We build a store with the structured-char strategy, embed the chunks once,
then run similarity, MMR, and hybrid on the question
"What is the total amount due?" (expected answer "610.00").

---

```python
# Compare all three retrievers on one invoice question.
# This lets you see the actual returned chunks, not just recall numbers.

# Build chunks for the structured-char strategy (sample-invoice.pdf).
demo_chunks = splitter.split_documents(
    extract_invoice(INVOICE_DIR / "sample-invoice.pdf", invoice_id="INV-100")
)
print(f"Building store with {len(demo_chunks)} chunks (structured-char)...")

# Embed all chunks once.
demo_vecs = embedder.embed_documents([c.page_content for c in demo_chunks])

# Build Chroma store with precomputed vectors.
store_demo = Chroma.from_documents(
    demo_chunks,
    embedding=Precomputed(demo_vecs),
    collection_name="nb03_inline_demo",
    persist_directory=tempfile.mkdtemp(prefix="nb03_demo_"),
)
```

---

```python
# BM25 keyword retriever + embed the question.
bm25_demo = BM25Retriever.from_documents(demo_chunks, k=5)

# Embed the question.
question = "What is the total amount due?"
expected = "610.00"
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

## 6 · Evaluate — the sample-invoice demo

Now we run the full strategy x retriever matrix on sample-invoice.pdf only.
That is 3 strategies x 3 retrievers x 4 questions = 36 retrievals.

> **Timing note:** This cell takes approximately 1–2 minutes on CPU. Most of
> the time is embedding the unique chunks (the first-time model download of
> ~130 MB from HuggingFace adds a bit more). Subsequent runs are faster because
> the model is cached locally.

We measure **recall@5**: for each question, does the expected answer substring
appear in at least one of the top-5 retrieved chunks? This is a binary metric
(0 or 1 per question), averaged across the 4 questions to get a recall score.

The expected results on this short, well-formed invoice:
- **naive-char**: 1.00 / 1.00 / 1.00 (sim/mmr/hyb)
- **structured-unit**: 1.00 / 1.00 / 1.00
- **structured-char**: 1.00 / 1.00 / 1.00

All strategies tie here — a single-page invoice with a clean text layer is
easy for every pipeline. The differences show up on harder documents in Part 7.

---

```python
# === DEMO: Full strategy x retriever matrix on sample-invoice.pdf ===
# These cells build chunks, embed them, and measure recall for all combinations.
# It takes about 1-2 minutes on CPU.

import time

DEMO_DOC = "sample-invoice"
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
    path = INVOICE_DIR / f"{docname}.pdf"
    iid = INVOICE_IDS[docname]
    if strategy == "naive-char":
        return splitter.split_documents(load_naive(path))
    elif strategy == "structured-unit":
        return extract_invoice(path, invoice_id=iid)
    elif strategy == "structured-char":
        return splitter.split_documents(extract_invoice(path, invoice_id=iid))
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

## 7 · Full picture — all 6 documents

The sample-invoice demo shows the measurement loop in action. But one
invoice is not enough. Here are the complete results across all 6 text-based
invoices, measured in a full benchmark run that took about 10 minutes of
embedding time.

We present these as static data (no embedding needed to view them). The code
cell below prints the per-document table and grand totals.

Two numbers to watch:

- **Invoice_1** — the naive loader scores 4/4 while structured-unit scores
  only 2/4 on plain similarity. On this document structure does not help
  plain cosine; MMR and hybrid recover to 4/4.
- **german-zugferd** — structured-unit drops to 3/4 on similarity and hybrid
  (the payee question), while MMR recovers to 4/4.

Across all 23 questions: naive-char hits 23/23 everywhere; structured-unit
and structured-char hit 20/23 on similarity and 22/23 on hybrid — the
difference is metadata, not raw recall.

---

```python
# Full 6-document benchmark results (pre-computed, static data).
# These numbers are from a complete run of all 6 documents x 3 strategies
# x 3 retrievers. The demo cell above reproduces the sample-invoice rows.
# Grand totals: naive-char 23/23/23; structured-unit 20/23/22;
# structured-char 20/23/22 (sim/mmr/hyb) out of total_q=23.

FULL_RESULTS = {
    "sample-invoice": {
        "naive-char":      {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-unit": {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-char": {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
    },
    "multipage_invoice1": {
        "naive-char":      {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-unit": {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-char": {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
    },
    "Invoice_1": {
        "naive-char":      {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-unit": {"sim": 0.50, "mmr": 1.00, "hyb": 1.00},
        "structured-char": {"sim": 0.50, "mmr": 1.00, "hyb": 1.00},
    },
    "Invoice-6": {
        "naive-char":      {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-unit": {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-char": {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
    },
    "sdk-invoice1": {
        "naive-char":      {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-unit": {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-char": {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
    },
    "german-zugferd": {
        "naive-char":      {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-unit": {"sim": 0.75, "mmr": 1.00, "hyb": 0.75},
        "structured-char": {"sim": 0.75, "mmr": 1.00, "hyb": 0.75},
    },
}

print(f"FULL_RESULTS loaded: {len(FULL_RESULTS)} documents")
```

---

```python
# Per-document recall table (Q = number of questions for that document).
print(f"{'Document':20} {'Strategy':16} {'Sim':>4} {'MMR':>4} {'Hyb':>4} {'Q':>3}")
print("-" * 66)
for docname, strats in FULL_RESULTS.items():
    for sname, vals in strats.items():
        print(f"{docname[:20]:20} {sname:16} {vals['sim']:4.2f} "
              f"{vals['mmr']:4.2f} {vals['hyb']:4.2f} {len(QA[docname]):3d}")
```

---

```python
# Grand totals: sum recall hits across all 23 questions.
grand_totals = {}
for docname, strats in FULL_RESULTS.items():
    nq = len(QA[docname])
    for sname, vals in strats.items():
        for ret in ["sim", "mmr", "hyb"]:
            key = (sname, ret)
            # Convert aggregate score to hit count (round to avoid float issues).
            hits = round(vals[ret] * nq)
            grand_totals[key] = grand_totals.get(key, 0) + hits

print("\n--- Grand totals (all 23 questions) ---")
print(f"{'Strategy':16} {'Retriever':12} {'Recall':>8}")
for (sname, ret), hits in sorted(grand_totals.items()):
    print(f"{sname:16} {ret:12} {hits}/23 = {hits/23:.2f}")
```

---

## 8 · Case studies — where each strategy wins and fails

### Case study 1: Scanned invoices (no text layer, recall is zero)

Three files in `Data/SD-08-invoices/` are images, not documents:
watson-hall-1898.pdf and macy-receipt.pdf are scanned PDFs, and
szamla-minta.jpg is a photograph of a Hungarian invoice.

PyPDFLoader returns pages with **0 characters** for these. No loader,
splitter, or retriever can find an answer in text that does not exist:

| File | Text layer | Naive recall |
|---|---|---|
| watson-hall-1898.pdf | empty | 0 |
| macy-receipt.pdf | empty | 0 |
| szamla-minta.jpg | not a PDF | 0 |

**Lesson:** For scans, the retrieval pipeline is irrelevant until you add
OCR (e.g. pytesseract) or an image-understanding model. Structure-aware
loading cannot invent text that is not there.

### Case study 2: Invoice_1 totals — tiny units rank below top-5

Invoice_1.pdf is a dense line-item table. Two questions ask for values in
the totals block: "What is the total price of all items?" (1075.70) and
"How many total pieces?" (66).

| Strategy | Sim | MMR | Hyb |
|---|---|---|---|
| naive-char | 1.00 | 1.00 | 1.00 |
| structured-unit | 0.50 | 1.00 | 1.00 |
| structured-char | 0.50 | 1.00 | 1.00 |

Under structured-unit the totals row becomes a **tiny atomic unit** (one
line, e.g. "Total 1075.70 | 66 pcs"). Plain cosine similarity ranks that
unit below the top-5, which fills up with semantically similar line-item
rows — so sim scores 2/4. MMR's diversity pull brings the totals unit in
(4/4), and hybrid's BM25 side matches the words "total" and "pieces"
directly, also recovering to 4/4.

**Lesson:** Atomic units are precise but easy to drown. When a unit is too
small, plain similarity can miss it; MMR and hybrid are the fix.

### Case study 3: German invoice — the payee question

german-zugferd.pdf is a ZUGFeRD invoice written in German. Its QA pairs ask
for the Rechnungsnummer (1001), the Rechnungsbetrag (139,20), the VAT rate
of Position 1 (19 %), and the payee ("PG Consulting").

| Strategy | Sim | MMR | Hyb |
|---|---|---|---|
| naive-char | 1.00 | 1.00 | 1.00 |
| structured-unit | 0.75 | 1.00 | 0.75 |
| structured-char | 0.75 | 1.00 | 0.75 |

The payee lives in the header unit. Under structured-unit, plain similarity
ranks other units (the line-item table rows, which share more tokens with
the question's context) above the header — sim=0 for that question. Hybrid
also misses (BM25 has no "payee" term to match in German text), while MMR's
diversity pulls the header unit into the top-5 and recovers the answer.

BGE handles German well for most questions, but the payee question is hard
because the seller name appears in a small header unit that gets crowded out
by the larger line-item table.

**Lesson:** Multilingual invoices stress naive English tokenizers unevenly,
but BGE handles the German terms well for most questions. When a specific
unit is crowded out, diversity-based retrieval (MMR) is the escape hatch.

### Case study 4: invoice_id filtering — the structure's real win

The most valuable property of structured units is not recall — it is
**metadata**. Every header, line-item, and totals unit carries
`invoice_id`, so a single vector store can hold many invoices and still
answer "what is the total on invoice INV-100?" without any cross-invoice
contamination.

Naive page chunks carry no such metadata: a page from invoice A and a page
from invoice B are indistinguishable after embedding.

The cell below builds one index from three invoices and runs the same query
with a `filter={"invoice_id": ...}` — each query returns only units from
the requested invoice.

---

```python
# CS4: invoice_id filtering isolates one invoice in a multi-invoice index.
names_ids = [("sample-invoice.pdf", "INV-100"),
             ("Invoice_1.pdf", "3847193"),
             ("sdk-invoice1.pdf", "34278587")]
units = []
for name, iid in names_ids:
    units += extract_invoice(INVOICE_DIR / name, invoice_id=iid)
store_all = Chroma.from_documents(
    units, embedding=embedder, collection_name="nb03_three_invoices",
    persist_directory=tempfile.mkdtemp(prefix="nb03_multi_"),
)
for _, iid in names_ids:
    hits = store_all.similarity_search("total", k=3, filter={"invoice_id": iid})
    got = sorted({h.metadata["invoice_id"] for h in hits})
    print(f"filter invoice_id={iid!r}: {len(hits)} hits, ids={got}")
```

---

## Bad methods we hit first, and how we overcame them

This notebook looks clean, but getting here took three dead ends. Each one
taught us something important.

### Dead end #1: Assuming naive PDF text extraction "just works"

We started with PyPDFLoader on the whole `Data/SD-08-invoices/` folder and
watched the pipeline produce pages, chunks, and embeddings without a single
error. Everything looked fine — until we tested a question against
watson-hall-1898.pdf and got recall 0.

The failure was silent: PyPDFLoader does not raise on a scan, it returns
pages with `page_content == ""`. No error message, no warning, nothing.
The corpus was empty without telling us.

**Fix:** Always print a character-count sanity check per document before
trusting a PDF pipeline. If a page has 0 chars, that file needs OCR, not
retrieval tuning.

### Dead end #2: The first structured extractor dropped totals

Our first version of `extract_invoice` only pulled the line-item table rows
as units and kept the page text as the header. It looked great on
sample-invoice.pdf — and silently lost two classes of answers:

- **multipage_invoice1.pdf** — Company B's subtotal (3000.00) and tax
  (300.00) sit in the second page's totals block. Our extractor only kept
  page 1's table, so those questions scored 0.
- **german-zugferd.pdf** — the Netto and MwSt sums live in a totals block
  that the first extractor treated as plain header text.

**Fix:** Extend the extractor to classify every table row by keyword
("total", "subtotal", "tax", "Netto", "MwSt", ...) into header, line_item,
or totals units — and do it on every page, not just page 1.

### Dead end #3: Assuming structured always wins

After the loader work, we assumed structured-unit would beat naive-char
everywhere. It did not. On sample-invoice.pdf, multipage_invoice1.pdf,
Invoice-6.pdf, and sdk-invoice1.pdf the two pipelines tie at 4/4 or 3/3.
And on Invoice_1.pdf naive-char's page chunks actually beat
structured-unit's plain similarity (4/4 vs 2/4).

**Fix:** Stop treating recall as the only metric. The structure's real win
is metadata (invoice_id filtering, per-unit granularity), not raw recall on
short invoices. The lesson: measure, and look at the table before
concluding.

---

## Summary: What you should remember

1. **A PDF may have no text layer at all.** Scanned invoices come back as
   empty pages from naive loaders. Always check extracted character counts
   before building a pipeline — and reach for OCR for scans.

2. **Structure means metadata plus atomic units.** Splitting an invoice into
   header / line-item / totals units, each tagged with `invoice_id`, turns a
   page blob into a queryable, filterable collection.

3. **Atomic units dilute plain similarity.** A one-line totals unit ranks
   below the top-5 even when it is the exact answer. Plain cosine similarity
   favors fat, semantically-similar units.

4. **MMR and hybrid fix the dilution.** MMR's diversity pulls in crowded-out
   units; hybrid's BM25 side matches exact tokens ("total", "pieces",
   "19 %"). Between them they recovered every question the benchmark missed.

5. **invoice_id filtering is the killer feature for invoice RAG.** One
   vector store, many invoices, and a `filter={"invoice_id": ...}` turns
   retrieval into a per-invoice search — something page chunks cannot do.

---

## Exercises

1. **Add OCR for a scanned invoice.** Install pytesseract and try to extract
   text from watson-hall-1898.pdf or macy-receipt.pdf (render each page to an
   image with pdf2image, then OCR it). Embed the OCR text and re-run the
   pipeline: which strategy wins on OCR output?

2. **Chunk by line-item groups.** Modify `extract_invoice` so consecutive
   line-item rows are grouped into one unit (e.g. 5 rows per unit) instead of
   one unit per row. Re-run the Invoice_1 demo: does plain similarity now
   find the totals?

3. **Metadata filter vs no filter.** Build one index from three invoices and
   run the same question with and without `filter={"invoice_id": ...}`.
   Count how often an answer from the wrong invoice leaks into the top-5
   without the filter. Is filtering always correct?
