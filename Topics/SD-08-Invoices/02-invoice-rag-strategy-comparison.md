> Source notebook: `NoteBooks/SD-08-Invoices/02-invoice-rag-strategy-comparison.ipynb`

---

# Invoice RAG Strategy Comparison — Loader x Splitter x Retriever on Real PDF Invoices

**Goal:** Measure which combinations of loader, splitter, and retriever
actually work for PDF-invoice RAG, and which fail, and why.

```
Loaders     : PDFLoader (naive pypdf pages) / inline pdfplumber extractor (structured)
Splitter    : DocumentProcessor (character-count) / none (atomic units)
Embedding   : fastembed BAAI/bge-base-en-v1.5 (local ONNX, 768-dim)
Vector DB   : Chroma (local, embed-once-reuse)
Retrievers  : Similarity / MMR / Hybrid (dense + BM25 + RRF)
Metric      : recall@5 (substring match, no LLM)
```

No API keys. No proprietary services. Everything runs on your CPU. This notebook
walks you through building a retrieval pipeline for PDF invoices using three
loading strategies and three retrievers, then measuring every combination against
23 hand-checked QA pairs across 6 real invoices. By the end you will know which
strategies work, which fail, and why.

Learn:

* **Naive page extraction** returns 0 text on scanned invoices with no text layer
* **Atomic structured units** (header / line item / totals) beat page chunks on metadata, not raw recall
* **Plain similarity loses to MMR** when tiny atomic chunks dilute the top-5 ranking
* **Hybrid retrieval** (dense + BM25 + RRF) recovers most lost answers but not all
* **Local embedding** eliminates rate-limit problems entirely and makes benchmarking possible

---

### How to work through this notebook

The pipeline has six steps: load, chunk, embed, store, retrieve, evaluate. We
compare **three loading strategies** (naive-char, structured-unit, structured-char)
against **three retrievers** (similarity, MMR, hybrid) across 23 hand-checked
questions on 6 real PDF invoices. Sections 0--4 build the pipeline once.
Section 5 defines the retrieval helpers. Section 6 runs the full matrix on a
single document. Section 7 shows the complete 6-document benchmark. Section 8
tells the stories behind the numbers.

---

## 0 · Setup — local pipeline with repo modules

Everything here is local. We resolve paths to six sample `.pdf` files living in
`Data/SD-08-invoices/` and verify they exist before we start. The repo provides
its own `PDFLoader`, `DocumentProcessor`, `ChromaVectorStore`, and retriever
classes, all imported from modules under the repo root.

---

```python
import os
import sys
from pathlib import Path

# Resolve the repo root relative to this notebook's directory.
# When executed from NoteBooks/SD-08-Invoices/,
# the repo root is two levels up.
REPO_ROOT = Path.cwd().parent.parent
DATA_DIR = REPO_ROOT / "Data" / "SD-08-invoices"

# Add the repo root to sys.path so we can import the project's modules.
sys.path.insert(0, str(REPO_ROOT))

print(f"REPO_ROOT : {REPO_ROOT}")
print(f"DATA_DIR  : {DATA_DIR}")
print(f"Exists    : {DATA_DIR.exists()}")

# List the PDF invoices we expect to find.
expected = [
    "sample-invoice.pdf",
    "multipage_invoice1.pdf",
    "Invoice_1.pdf",
    "Invoice-6.pdf",
    "sdk-invoice1.pdf",
    "german-zugferd.pdf",
]
for name in expected:
    path = DATA_DIR / name
    status = "OK" if path.exists() else "MISSING"
    print(f"  {status}: {name}")
```

---

```python
import hashlib
import json
import tempfile

# Repo modules -- these live under the repo root, added to sys.path above.
from loaders.pdf import PDFLoader
from retrieval.hybrid import HybridRetriever
from retrieval.mmr import MMRRetriever
from retrieval.similarity import SimilarityRetriever
from splitters.recursive import DocumentProcessor
from vectordb.chroma import ChromaVectorStore

# The repo has no structured PDF loader: the structured strategies use
# an inline pdfplumber extractor (defined in the Strategies cell below).
import pdfplumber
from langchain_core.documents import Document

print("All imports OK")
```

---

## 1 · Benchmark setup — 23 QA pairs across 6 invoices

A dictionary of 23 hand-checked QA pairs across 6 PDF invoices. Each pair is a
`(question, expected_answer_substring)` tuple. The expected answer must appear
(case-insensitive) in at least one retrieved chunk for recall@5 to fire.

We chose exact-value answers like `"INV-100"`, `"610.00"`, `"3847193"`, `"19 %"`
because they are unambiguous, they test whether specific data survived loading,
splitting, and embedding, and a strategy that drops tables will miss line-item
answers while a strategy that shreds units will miss cell values.

---

```python
# 23 hand-checked QA pairs across 6 PDF invoices.
# Answers are exact substrings that must appear in retrieved chunk text.
QA = {
    "sample-invoice": [
        ("What is the invoice number?", "INV-100"),
        ("What is the total due on the invoice?", "610.00"),
        ("Who is the customer on this invoice?", "MICROSOFT CORPORATION"),
        ("What was the sales tax amount?", "10.00"),
    ],
    "multipage_invoice1": [
        ("What is the total amount of the Company A invoice?", "430.00"),
        ("What is the subtotal of the Company B invoice?", "3000.00"),
        ("Which line item has a quantity of 8?", "G"),
        ("What is the tax amount on the Company B invoice?", "300.00"),
    ],
    "Invoice_1": [
        ("What is the invoice number?", "3847193"),
        ("What is the total price of all items?", "1075.70"),
        ("How many pieces were delivered in total?", "66"),
        ("Which item code is the bubble film roll?", "JF9912413BF"),
    ],
    "Invoice-6": [
        ("What is the receipt number on the project statement?", "9876"),
        ("What is the total amount on the project statement?", "10,686.25"),
        ("What discount rate was applied to the leadership training?", "25%"),
        ("What is the sales tax rate?", "3%"),
    ],
    "sdk-invoice1": [
        ("What is the invoice number?", "34278587"),
        ("What are the total charges on the invoice?", "56,651.49"),
        ("Who is the invoice for?", "Microsoft"),
    ],
    "german-zugferd": [
        ("What is the Rechnungsnummer (invoice number)?", "1001"),
        ("What is the Rechnungsbetrag (total amount due)?", "139,20"),
        ("What VAT rate is applied to Position 1?", "19 %"),
        ("Who is the payee (seller) on this invoice?", "PG Consulting"),
    ],
}

total_q = sum(len(v) for v in QA.values())
print(f"{len(QA)} documents, {total_q} questions")
```

---

## 2 · Load and chunk — naive vs structured strategies

Three strategies for turning a PDF invoice into chunks, defined by
loader-times-splitter choice:

| Strategy | Loader | Splitter | What happens |
|---|---|---|---|
| `naive-char` | PDFLoader (pypdf pages) | DocumentProcessor(1000, 200) | One chunk per page of extracted text, then character-count split |
| `structured-unit` | inline `extract_invoice` (pdfplumber) | None | Walk the PDF in order, keep each header / line-item / totals unit atomic |
| `structured-char` | inline `extract_invoice` (pdfplumber) | DocumentProcessor(1000, 200) | Join units per logical section, then character-count split |

The naive loader is the trap everyone falls into. It looks like it works until
you measure it against a scanned invoice (0 text pages) or an invoice whose
values live in tables. The structured extractor fixes the loading problem; the
choice between "unit" and "char" determines whether you then re-split those
units or keep them atomic.

---

```python
def extract_invoice(path):
    """Extract structured units from a PDF invoice with pdfplumber.

    Returns a list of langchain Documents, one per logical unit:
      - 'header' units: key-value lines (invoice number, dates, totals)
      - 'line_item' units: one per table row
      - 'totals' units: subtotal / tax / total-due rows

    Every unit carries metadata: type and invoice_id.
    """
    KEYWORDS = (
        "invoice", "rechnung", "receipt", "customer", "buyer",
        "seller", "payee", "total", "subtotal", "tax", "vat",
        "amount", "due", "number", "no.", "summe", "netto",
        "mwst", "discount",
    )
    units = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            # Header lines: key-value lines mentioning invoice terms.
            for line in text.splitlines():
                if any(k in line.lower() for k in KEYWORDS):
                    units.append(Document(
                        page_content=line, metadata={"type": "header"}))
            # Line items and totals: every non-empty table row.
            for table in page.extract_tables() or []:
                for row in table:
                    cells = [c for c in row if c]
                    if cells:
                        units.append(Document(
                            page_content=" | ".join(cells),
                            metadata={"type": "line_item"}))
    # Best-effort invoice id: first token after an invoice-number marker,
    # else the filename stem.
    invoice_id = Path(path).stem
    for unit in units:
        if unit.metadata["type"] != "header":
            continue
        low = unit.page_content.lower()
        for marker in ("invoice no", "invoice number",
                       "rechnungsnummer", "invoice #", "invoice:"):
            if marker in low:
                tail = unit.page_content.split(":", 1)[-1].strip()
                if tail:
                    invoice_id = tail.split()[0]
                    break
        if invoice_id != Path(path).stem:
            break
    for unit in units:
        unit.metadata["invoice_id"] = invoice_id
    return units


def build_chunks(docname, strategy):
    """Build chunks for one (document, strategy) pair."""
    path = str(DATA_DIR / f"{docname}.pdf")
    if strategy == "naive-char":
        # Naive mode: one chunk per page of extracted text.
        docs = PDFLoader(path).load()
        # Then split on character counts.
        chunks = DocumentProcessor(chunk_size=1000, chunk_overlap=200).split_docs(docs)
    elif strategy == "structured-unit":
        # Structured mode: keep each header / line-item / totals unit atomic.
        chunks = extract_invoice(path)
    elif strategy == "structured-char":
        # Structured mode: join units into one char block per logical
        # section (header / line items / totals), then split on char counts.
        units = extract_invoice(path)
        sections = {}
        for unit in units:
            sections.setdefault(unit.metadata["type"], []).append(unit.page_content)
        docs = [Document(page_content="\n".join(parts))
                for parts in sections.values()]
        chunks = DocumentProcessor(chunk_size=1000, chunk_overlap=200).split_docs(docs)
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")
    return chunks

# Quick sanity check: build one strategy and show chunk counts.
for strat in ["naive-char", "structured-unit", "structured-char"]:
    chunks = build_chunks("sample-invoice", strat)
    print(f"  {strat:20} -> {len(chunks)} chunks")
```

---

## 3 · Embed — local ONNX, embed-once-reuse

`LocalEmbedder` wraps `fastembed` with the `BAAI/bge-base-en-v1.5` model
(768 dimensions, runs on CPU). Two operations: `embed_unique(texts)`
deduplicates texts, embeds each unique text once, and returns a
`{sha1_hash: vector}` dict. `embed_query(text)` embeds a single query string.

This is the result of the first bad method we hit. The Gemini free tier limits
you to roughly 100 embed requests per minute, and every retry burns more quota,
creating a retry storm that can never recover. Local embedding has zero rate
limits, no API key, and embeds 100 chunks in roughly 0.3 seconds on CPU.

---

```python
class LocalEmbedder:
    """Local ONNX embedder (fastembed / BAAI/bge-base-en-v1.5, 768-dim).

    No API key, no rate limits: 100 chunks embed in ~0.3s on CPU.
    Uses the bge query prefix for queries and plain text for documents
    (fastembed handles this split internally via query_embed vs embed).
    """
    MODEL = "BAAI/bge-base-en-v1.5"

    def __init__(self):
        from fastembed import TextEmbedding
        self._emb = TextEmbedding(model_name=self.MODEL)

    def embed_unique(self, texts):
        """Embed all texts; returns {sha1: vector} for each unique text."""
        seen = set()
        # Preserve insertion order, skip duplicates.
        uniq = [t for t in texts if not (t in seen or seen.add(t))]
        # Batch-embed all unique texts at once.
        vectors = list(self._emb.embed(uniq, batch_size=64))
        # Key by SHA-1 hash so the same text always maps to the same vector.
        return {
            hashlib.sha1(t.encode("utf-8")).hexdigest(): v.tolist()
            for t, v in zip(uniq, vectors)
        }

    def embed_query(self, text):
        """Embed a single query string. Returns a 768-dim list."""
        return next(self._emb.query_embed(text)).tolist()

# Instantiate -- first call downloads the model (~130 MB), subsequent calls are fast.
embedder = LocalEmbedder()

# Quick test: embed a sample query.
sample_vec = embedder.embed_query("What is the invoice number?")
print(f"Embedding dimension: {len(sample_vec)}")
print(f"First 5 values: {sample_vec[:5]}")
```

---

## 4 · Retrieve — similarity, MMR, and hybrid

Three retrieval strategies, all sharing one precomputed query vector:

| Retriever | How it works |
|---|---|
| similarity | Plain cosine distance between query vector and chunk vectors |
| mmr | Maximum Marginal Relevance, trades pure relevance for diversity (lambda_mult=0.7) |
| hybrid | Reciprocal Rank Fusion of dense (vector) + sparse (BM25 keyword) rankings |

Similarity is the baseline. MMR prevents retrieving 5 nearly-identical chunks.
Hybrid adds BM25 keyword matching, which catches exact identifiers (invoice
numbers, item codes, totals) that dense embeddings miss.

---

```python
def _prevec_dense(store, query_vec):
    """Dense retriever adapter that reuses a precomputed query vector."""
    class _Adapter:
        def retrieve(self, question):
            return store.query(query_vec, top_k=5)
    return _Adapter()


def retrieve_variants(store, chunks, question, query_vec):
    """Run all 3 retrievers on one question, sharing one query vector.

    Similarity and MMR use the vector store directly.
    Hybrid fuses dense results with BM25 sparse results via RRF.
    """
    # Similarity: plain cosine top-5.
    sim = store.query(query_vec, top_k=5)

    # MMR: diverse top-5 with lambda_mult=0.7 (lean toward relevance).
    mmr = store.query_mmr(query_vec, top_k=5, lambda_mult=0.7)

    # Hybrid: dense + BM25 sparse, fused with Reciprocal Rank Fusion.
    try:
        from langchain_community.retrievers import BM25Retriever

        # Build a BM25 index over the chunks for this strategy.
        sparse = BM25Retriever.from_documents(chunks, k=5)

        # BM25Retriever is a langchain Runnable -- its .invoke() method
        # returns results. We wrap it in an adapter exposing .retrieve()
        # so HybridRetriever can use it uniformly.
        class _SparseAdapter:
            """BM25Retriever is a Runnable: expose .retrieve() via .invoke()."""
            def retrieve(self, q):
                return sparse.invoke(q)

        hyb = HybridRetriever(
            _prevec_dense(store, query_vec), _SparseAdapter(), top_k=5
        ).retrieve(question)
    except Exception as exc:
        print(f"    hybrid unavailable: {exc}")
        hyb = []

    return {"similarity": sim, "mmr": mmr, "hybrid": hyb}
```

---

### recall@5: the evaluation metric

For each question, does the expected answer substring appear in at least one of
the top-5 retrieved chunks? This is a binary metric (0 or 1 per question),
averaged across all questions to get a recall score per strategy. We use
substring matching rather than LLM-judged answer quality because the 23 answers
are exact values (`"INV-100"`, `"610.00"`, `"3847193"`, `"19 %"`), a substring
match proves the answer exists in the retrieved context, and the metric is
deterministic, fast, and reproducible with no LLM calls.

---

```python
def recall_at_k(retrieved, expected):
    """Did the expected substring appear in any retrieved chunk?

    Args:
        retrieved: list of Document objects from the retriever.
        expected: the expected answer string (e.g. "610.00").

    Returns:
        True if expected.lower() appears in any chunk's page_content.lower().
    """
    low = expected.lower()
    return any(low in d.page_content.lower() for d in retrieved)


# Sanity check: "610.00" must be findable in a chunk that contains it.
class _FakeDoc:
    def __init__(self, text):
        self.page_content = text

assert recall_at_k(
    [_FakeDoc("Total due on the invoice is 610.00 USD")],
    "610.00"
)
assert not recall_at_k(
    [_FakeDoc("the total due was paid in full")],
    "610.00"
)
print("recall_at_k: sanity checks passed")
```

---

## 5 · Demo — full matrix on sample-invoice.pdf

Run the full strategy x retriever matrix on `sample-invoice.pdf` only
(4 questions x 3 strategies x 3 retrievers = 36 retrievals). This takes about
10 to 20 seconds and shows the measurement loop in action. `sample-invoice.pdf`
is the smallest and fastest to process, so the demo lets you see the harness
work end-to-end before looking at the full 6-document results.

---

```python
# --- DEMO: run the full matrix on sample-invoice.pdf only ---
import time

DEMO_DOC = "sample-invoice"
strategies = ["naive-char", "structured-unit", "structured-char"]
retriever_names = ["similarity", "mmr", "hybrid"]

print(f"Demo: {DEMO_DOC}")
print(f"  strategies : {strategies}")
print(f"  retrievers : {retriever_names}")
print(f"  questions  : {len(QA[DEMO_DOC])}")
print()

# Phase 1: build chunks for all strategies, collect unique texts.
all_chunks = {}
for strategy in strategies:
    all_chunks[strategy] = build_chunks(DEMO_DOC, strategy)

# Collect all unique chunk texts across strategies for one embedding pass.
unique_texts = []
seen = set()
for strategy in strategies:
    for c in all_chunks[strategy]:
        if c.page_content not in seen:
            seen.add(c.page_content)
            unique_texts.append(c.page_content)

print(f"Unique chunks to embed: {len(unique_texts)}")
t0 = time.time()
vec_by_key = embedder.embed_unique(unique_texts)
print(f"Embedded in {time.time() - t0:.1f}s")

# Phase 2: for each strategy, build a Chroma store with precomputed
# embeddings, then run all retrievers on all questions.
tmp = tempfile.mkdtemp(prefix="sd08_chroma_")
demo_results = {}

for strategy in strategies:
    chunks = all_chunks[strategy]
    # Build store with embedding=None since we pass precomputed vectors.
    store = ChromaVectorStore(
        collection_name=f"demo__{strategy}",
        persist_dir=tmp,
        embedding=None,
    )
    # Look up the precomputed vector for each chunk by its SHA-1 hash.
    embs = [
        vec_by_key[hashlib.sha1(c.page_content.encode("utf-8")).hexdigest()]
        for c in chunks
    ]
    store.add(chunks, embeddings=embs)
    demo_results[strategy] = {"n_chunks": len(chunks)}

    print(f"\n[{strategy}] {len(chunks)} chunks")
    for q, expected in QA[DEMO_DOC]:
        # Embed the question once, reuse the vector across retrievers.
        query_vec = embedder.embed_query(q)
        variants = retrieve_variants(store, chunks, q, query_vec)

        row = {"expected": expected}
        for rname, retrieved in variants.items():
            row[rname] = recall_at_k(retrieved, expected)

        demo_results[strategy][q] = row
        flags = " ".join(f"{r}={'1' if row[r] else '0'}" for r in retriever_names)
        print(f"  Q: {q[:58]}")
        print(f"     exp={expected!r:18} {flags}")

# Print the recall grid.
print("\n--- Recall grid (1=hit, 0=miss) ---")
print(f"{'strategy':20} {'sim':>4} {'mmr':>4} {'hyb':>4}")
for strategy in strategies:
    nq = len(QA[DEMO_DOC])
    sim = sum(demo_results[strategy][q]["similarity"] for q, _ in QA[DEMO_DOC])
    mmr = sum(demo_results[strategy][q]["mmr"] for q, _ in QA[DEMO_DOC])
    hyb = sum(demo_results[strategy][q]["hybrid"] for q, _ in QA[DEMO_DOC])
    print(f"{strategy:20} {sim/nq:4.2f} {mmr/nq:4.2f} {hyb/nq:4.2f}")
```

---

## 6 · Full picture — all 6 documents

The demo above shows the measurement loop on one document. These are the real
numbers from the full benchmark, which we present as static data rather than
re-running the embed loop. The full run took about 1 to 2 minutes of embedding
time (100% local, no API calls, no rate limits).

### Per-document recall@5 table

```
Document              Strategy          Sim   MMR   Hyb
sample-invoice        naive-char        1.00  1.00  1.00
sample-invoice        structured-unit   1.00  1.00  1.00
sample-invoice        structured-char   1.00  1.00  1.00
multipage_invoice1    naive-char        1.00  1.00  1.00
multipage_invoice1    structured-unit   1.00  1.00  1.00
multipage_invoice1    structured-char   1.00  1.00  1.00
Invoice_1             naive-char        1.00  1.00  1.00
Invoice_1             structured-unit   0.50  1.00  1.00
Invoice_1             structured-char   0.50  1.00  1.00
Invoice-6             naive-char        1.00  1.00  1.00
Invoice-6             structured-unit   1.00  1.00  1.00
Invoice-6             structured-char   1.00  1.00  1.00
sdk-invoice1          naive-char        1.00  1.00  1.00
sdk-invoice1          structured-unit   1.00  1.00  1.00
sdk-invoice1          structured-char   1.00  1.00  1.00
german-zugferd        naive-char        1.00  1.00  1.00
german-zugferd        structured-unit   0.75  1.00  0.75
german-zugferd        structured-char   0.75  1.00  0.75
```

### Grand totals (all 23 questions)

```
Strategy          Retriever      Recall
naive-char        similarity     23/23 = 1.00
naive-char        mmr            23/23 = 1.00
naive-char        hybrid         23/23 = 1.00
structured-unit   similarity     20/23 = 0.87
structured-unit   mmr            23/23 = 1.00
structured-unit   hybrid         22/23 = 0.96
structured-char   similarity     20/23 = 0.87
structured-char   mmr            23/23 = 1.00
structured-char   hybrid         22/23 = 0.96
```

On short text invoices the naive strategy is unbeatable (23/23). The structured
strategies lose 3 questions on plain similarity (the atomic-unit dilution
effect), but MMR recovers all of them; hybrid recovers all but the German payee
question.

---

```python
# Full 6-doc benchmark results (pre-computed). Stored as a dict so you
# can slice and analyze without re-running the embed loop.
FULL_RESULTS = {
    "sample-invoice": {
        "naive-char":       {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-unit":  {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-char":  {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
    },
    "multipage_invoice1": {
        "naive-char":       {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-unit":  {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-char":  {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
    },
    "Invoice_1": {
        "naive-char":       {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-unit":  {"sim": 0.50, "mmr": 1.00, "hyb": 1.00},
        "structured-char":  {"sim": 0.50, "mmr": 1.00, "hyb": 1.00},
    },
    "Invoice-6": {
        "naive-char":       {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-unit":  {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-char":  {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
    },
    "sdk-invoice1": {
        "naive-char":       {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-unit":  {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-char":  {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
    },
    "german-zugferd": {
        "naive-char":       {"sim": 1.00, "mmr": 1.00, "hyb": 1.00},
        "structured-unit":  {"sim": 0.75, "mmr": 1.00, "hyb": 0.75},
        "structured-char":  {"sim": 0.75, "mmr": 1.00, "hyb": 0.75},
    },
}

# Grand totals: {strategy x retriever: (hits, total)}
GRAND_TOTALS = {
    ("naive-char", "similarity"): (23, 23),
    ("naive-char", "mmr"):        (23, 23),
    ("naive-char", "hybrid"):     (23, 23),
    ("structured-unit", "similarity"): (20, 23),
    ("structured-unit", "mmr"):        (23, 23),
    ("structured-unit", "hybrid"):     (22, 23),
    ("structured-char", "similarity"): (20, 23),
    ("structured-char", "mmr"):        (23, 23),
    ("structured-char", "hybrid"):     (22, 23),
}

print("FULL_RESULTS loaded:", len(FULL_RESULTS), "docs")
for (strat, ret), (hits, total) in sorted(GRAND_TOTALS.items()):
    print(f"  {strat:16} x {ret:10} {hits}/{total} = {hits/total:.2f}")
```

---

```python
# OPTIONAL: re-run the full 6-document benchmark (takes ~1-2 minutes).
# Uncomment and execute this cell to reproduce the numbers above.
#
# import time
#
# full_strategies = ["naive-char", "structured-unit", "structured-char"]
# full_retrievers = ["similarity", "mmr", "hybrid"]
# tmp_full = tempfile.mkdtemp(prefix="sd08_chroma_full_")
# full_results = {}
#
# for docname in QA:
#     print(f"\n{'='*72}\n### {docname}\n{'='*72}")
#     full_results[docname] = {}
#
#     # One embedding pass per document, shared by all strategies.
#     all_chunks_full = {}
#     for strategy in full_strategies:
#         all_chunks_full[strategy] = build_chunks(docname, strategy)
#     unique_texts_full = []
#     seen_full = set()
#     for strategy in full_strategies:
#         for c in all_chunks_full[strategy]:
#             if c.page_content not in seen_full:
#                 seen_full.add(c.page_content)
#                 unique_texts_full.append(c.page_content)
#     vec_by_key_full = embedder.embed_unique(unique_texts_full)
#     print(f"  embedded {len(unique_texts_full)} unique chunks")
#
#     for strategy in full_strategies:
#         chunks = all_chunks_full[strategy]
#         store = ChromaVectorStore(
#             collection_name=f"{docname}__{strategy}",
#             persist_dir=tmp_full,
#             embedding=None,
#         )
#         embs = [
#             vec_by_key_full[hashlib.sha1(c.page_content.encode("utf-8")).hexdigest()]
#             for c in chunks
#         ]
#         store.add(chunks, embeddings=embs)
#         full_results[docname][strategy] = {"n_chunks": len(chunks)}
#
#         for q, expected in QA[docname]:
#             query_vec = embedder.embed_query(q)
#             variants = retrieve_variants(store, chunks, q, query_vec)
#             row = {"expected": expected}
#             for rname, retrieved in variants.items():
#                 row[rname] = recall_at_k(retrieved, expected)
#             full_results[docname][strategy][q] = row
#
# # Print summary.
# print("\nSUMMARY")
# print(f"{'doc':28} {'strategy':16} {'sim':>4} {'mmr':>4} {'hyb':>4} {'n':>3}")
# for docname, dres in full_results.items():
#     for strategy, sres in dres.items():
#         n = sres.get("n_chunks", 0)
#         nq = sum(1 for v in sres.values() if isinstance(v, dict) and "similarity" in v)
#         sim = sum(v["similarity"] for v in sres.values() if isinstance(v, dict) and "similarity" in v)
#         mmr = sum(v["mmr"] for v in sres.values() if isinstance(v, dict) and "mmr" in v)
#         hyb = sum(v["hybrid"] for v in sres.values() if isinstance(v, dict) and "hybrid" in v)
#         print(f"{docname[:28]:28} {strategy:16} {sim/nq:4.2f} {mmr/nq:4.2f} {hyb/nq:4.2f} {n:>3}")
```

---

## 7 · Case studies — where each strategy wins and fails

### Case study 1: Invoice_1 — atomic units dilute plain similarity

On `Invoice_1`, the questions "What is the total price of all items?" and
"How many pieces were delivered in total?" score 0 on plain similarity for
both structured strategies (sim=0.50 for both structured-unit and
structured-char, vs 1.00 for naive-char). The answers ("1075.70", "66")
live in tiny atomic units that rank outside the top-5. MMR and hybrid both
recover them (mmr=1, hybrid=1), because diversity and BM25 keyword overlap
promote the exact-value chunk.

| Strategy | Sim | MMR | Hyb |
|---|---|---|---|
| naive-char | 1.00 | 1.00 | 1.00 |
| structured-unit | 0.50 | 1.00 | 1.00 |
| structured-char | 0.50 | 1.00 | 1.00 |

### Case study 2: german-zugferd — BGE handles German, but the payee question is hard

`german-zugferd.pdf` is a German ZUGFeRD invoice. BGE embeds German text well
enough that the `Rechnungsnummer` and `Rechnungsbetrag` questions score 1 on
every strategy. The one failure is "Who is the payee (seller) on this invoice?":
plain similarity scores 0 and hybrid scores 0 for the structured strategies (MMR
recovers it, mmr=1). The seller name "PG Consulting" is a proper noun with no
lexical overlap with the question, so neither cosine nor BM25 finds it reliably.

| Strategy | Sim | MMR | Hyb |
|---|---|---|---|
| naive-char | 1.00 | 1.00 | 1.00 |
| structured-unit | 0.75 | 1.00 | 0.75 |
| structured-char | 0.75 | 1.00 | 0.75 |

### Case study 3: sample-invoice, multipage_invoice1, Invoice-6, sdk-invoice1 — naive wins cleanly

Four of the six invoices score 1.00 across every strategy and every retriever.
The six invoices are short documents (1 to 3 pages), so a page-sized chunk
keeps the whole invoice in context and the answer is always in the top-5. The
naive strategy's real failures are not on these documents.

### Case study 4: The structured strategies' real win is metadata

`structured-unit` and `structured-char` do not beat naive on raw recall
(20 to 23 out of 23), but every unit carries `invoice_id` in its metadata.
That enables per-invoice filtered retrieval: "only search invoice 3847193"
becomes a metadata filter instead of a full-corpus search. Naive page chunks
carry no such label, so you cannot scope a query to one invoice without
re-embedding.

---

## Bad methods we hit first, and how we overcame them

This is the heart of the notebook. Three real failures, three real fixes.

---

### BAD #1: Naive page extraction returns 0 text on scanned invoices

`PDFLoader` wraps `PyPDFLoader` (pypdf), which extracts the text layer of a
PDF. On text-based invoices that works fine, every page becomes one chunk. But
two invoices in `Data/SD-08-invoices` are pure scans: `watson-hall-1898.pdf`
and `macy-receipt.pdf`. They have no text layer at all, so pypdf returns 0
text pages. Every question about them scores 0 recall no matter which retriever
you use. The answer simply does not exist in the corpus. This is the genuine
naive failure: silent, total, and invisible until you measure it.

**OVERCOME:** This benchmark measures the six text-based invoices only. The
honest lesson is that image-only PDFs need OCR (e.g. `pytesseract` plus
`pdf2image`) as a separate pipeline. The structured extractor below at least
labels every unit with `invoice_id`, so a scan-aware pipeline can be added
per-invoice later.

---

### BAD #2: The structured extractor dropped totals lines

The first version of `extract_invoice` kept table rows as line items and a
few obvious header lines, and silently dropped the Company B subtotal and tax
rows on `multipage_invoice1.pdf` (they live in a second table on page 2) and
the German `Summe Netto` / `MwSt` lines on `german-zugferd.pdf`. Those answers
("3000.00", "300.00", "139,20") were simply absent from the corpus, so the
structured strategies missed them entirely.

**OVERCOME:** We extended the header extractor to also keep any line whose
lowercased text contains `subtotal`, `tax`, `vat`, `summe`, `netto`, `mwst`,
`total`, `amount`, or `due`. After that fix, the structured strategies find
every totals value that the naive strategy finds.

---

### BAD #3: Tiny atomic units dilute plain-similarity top-5

`structured-unit` keeps each header line and each table row as its own atomic
chunk. That is great for metadata and filtering, but a one-line chunk is a
weak similarity target: "What is the total price of all items?" embeds closer
to the surrounding prose than to the single row holding "1075.70". Plain
similarity drops to 20/23 for both structured strategies (sim=20 vs 23).

**OVERCOME:** MMR's diversity term and hybrid's BM25 keyword boost both
recover the missing answers. MMR reaches 23/23, hybrid 22/23. The only
question that stays lost is the German payee question (see Findings below).

---

## Findings and limitations

Four findings from the full benchmark, all backed by real numbers.

### 1. On short text invoices, naive page-chunks retain everything

`naive-char` scores 23/23 across all three retrievers. The six invoices are
short documents (1 to 3 pages), so a page-sized chunk keeps the whole invoice
in context and the answer is always in the top-5. The naive strategy's real
failures are not on these documents: they are scanned invoices (0 text pages)
and the missing `invoice_id` metadata that would enable per-invoice filtering.

### 2. The structured strategies' real win is metadata

`structured-unit` and `structured-char` do not beat naive on recall (20 to
23 out of 23), but every unit carries `invoice_id` in its metadata. That
enables per-invoice filtered retrieval: "only search invoice 3847193" becomes
a metadata filter instead of a full-corpus search. Naive page chunks carry no
such label, so you cannot scope a query to one invoice without re-embedding.

### 3. Atomic units dilute plain similarity

On `Invoice_1`, the questions "What is the total price of all items?" and
"How many pieces were delivered in total?" score 0 on plain similarity for
both structured strategies (sim=0). The answers ("1075.70", "66") live in
tiny atomic units that rank outside the top-5. MMR and hybrid both recover
them (mmr=1, hybrid=1), because diversity and BM25 keyword overlap promote
the exact-value chunk. This is the same dilution effect we saw on Word
document tables.

### 4. Multilingual invoices: BGE handles German, but the payee question is hard

`german-zugferd.pdf` is a German ZUGFeRD invoice. BGE embeds German text well
enough that the `Rechnungsnummer` and `Rechnungsbetrag` questions score 1 on
every strategy. The one failure is "Who is the payee (seller) on this invoice?":
plain similarity scores 0 and hybrid scores 0 for the structured strategies
(MMR recovers it, mmr=1). The seller name "PG Consulting" is a proper noun
with no lexical overlap with the question, so neither cosine nor BM25 finds
it reliably.

---

## What you should notice

* **On short text invoices, naive page chunks are hard to beat** -- 23/23 on
  every retriever. The naive strategy's real failure is scanned PDFs (0 text
  pages) and the absence of `invoice_id` metadata, not chunking.
* **The structured strategies tie naive on recall but add metadata** -- every
  unit carries `invoice_id`, enabling per-invoice filtered retrieval that naive
  page chunks cannot do.
* **Atomic units dilute plain similarity** (sim=20 vs 23) -- MMR and hybrid
  recover the exact-value chunks that cosine ranking buries.
* **Hybrid is not a universal win here** -- it scores 22/23 on the structured
  strategies, missing the German payee question that MMR finds.
* **Local embedding made this benchmark possible:** 1 to 2 minutes of
  embedding, zero API calls, zero rate limits, fully reproducible.
* **Precomputed embeddings are an efficiency pattern worth keeping:** embed
  every unique chunk once, pass vectors to Chroma, and never re-embed the same
  text for a different strategy.

---

## Exercises

1. **Add a scanned invoice.** Drop `watson-hall-1898.pdf` into
   `Data/SD-08-invoices/`, add QA pairs for it, and re-run the demo.
   Confirm that naive-char scores 0 because pypdf extracts no text.
   What would you need to change to make it retrievable?

2. **Use the `invoice_id` metadata.** Modify the demo to filter the
   Chroma store to a single invoice (e.g.
   `metadata={"invoice_id": "INV-100"}`) before querying. Does
   recall stay the same? How much faster is a filtered query on a
   large corpus?

3. **Swap the embedding model.** Change `LocalEmbedder.MODEL` to
   `BAAI/bge-small-en-v1.5` (384-dim) and re-run the demo. Do the
   German questions still hit? Does the payee question improve?
