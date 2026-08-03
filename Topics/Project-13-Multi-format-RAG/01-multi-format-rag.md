> Source notebook: `NoteBooks/Project-13-Multi-format-RAG/01-multi-format-rag.ipynb`


---

# Project 13 — Multi-format RAG

**Goal:** Mix every loader into one vector database.

```
Loader      : PyPDFLoader + TextLoader + CSVLoader + WebBaseLoader + JSONLoader
Splitter    : RecursiveCharacterTextSplitter
Embedding   : Gemini
Vector DB   : Chroma (one store, five formats)
Retriever   : Similarity (Top-K) + metadata filter
Prompt      : Basic Context + Question
LLM         : Gemini
```

Every project so far fed the pipeline **one kind** of file. Real knowledge
bases are messier: PDFs, markdown notes, CSV tables, web pages and JSON blobs
all live side by side. This project loads **all five** formats, tags every
document with where it came from, pours them into **one** Chroma index, and
then answers from any of them — including scoping a search to a single format
with a metadata filter.

Learn:

* **Metadata** — extra labels that travel with each `Document`
* **Source tracking** — knowing which file or URL a chunk came from
* **Filtering** — restricting a search to a subset before any comparison

---

### How to work through this notebook

This notebook runs the **full pipeline** but swaps the loader block for **five
loaders at once** — one per format. Sections `1.1`–`1.5` each load one format
into a list, and Section `1.6` tags every document and merges the lists. The
split, embed, store, retrieve, prompt and answer blocks (Sections 2–7) are the
same pipeline you built in Projects 01–02, running on the combined corpus. The
two new ideas are **source tracking** and **metadata filtering**, both
demonstrated in Section 5.

---

## 0 · Setup — environment, keys & one install

The pipeline calls the **Gemini embedding API** and the **Gemini LLM**, so you
need a `GOOGLE_API_KEY`. Copy `.env.example` to `.env` at the repo root and
paste your key from <https://aistudio.google.com/>. The check below only prints
a masked preview (`key[:4]…`), never the full key.

One package is needed **for this project only**:

* `jq` — `JSONLoader` evaluates its `jq_schema` (a tiny query language for
  picking keys out of JSON) with this package. It ships as a small prebuilt
  wheel, so the install is quick.

`PyPDFLoader` additionally needs `pypdf` (already in `requirements.txt` — the
install cell re-installs it as a safety net). `WebBaseLoader` uses `requests` +
`beautifulsoup4` and `CSVLoader` uses the stdlib `csv` module, both already
part of the `langchain` stack.

---

```python
from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("GOOGLE_API_KEY", "")
if key:
    print(f"GOOGLE_API_KEY set: {key[:4]}…")
else:
    print("GOOGLE_API_KEY missing — copy .env.example to .env and add your key.")
```

---

```python
# Needed for THIS project only:
#   jq     → JSONLoader evaluates its jq_schema with this package
#   pypdf  → PyPDFLoader (already in requirements.txt; safety net)
%pip install jq pypdf
```

---

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
    JSONLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
```

---

## 1 · Load — five formats, five loaders, one list

Every loader does the same job: turn some external format into a list of
`Document`s. A `Document` is just text (`page_content`) plus a dictionary of
labels (`metadata`) that travels with it. Because all five loaders speak the
same `Document` language, you can pile PDF, markdown, CSV, web and JSON output
into **one list** and feed it to a single splitter, embedder and store.

Sections `1.1`–`1.5` each load one format into a `*_docs` list. A helper in
`1.6` then tags every document with its format, and the lists are concatenated.

---

### 1.1 · PDF — PyPDFLoader

A PDF's words live in a **text layer**; `PyPDFLoader` reads it and returns **one
`Document` per page**, each carrying `source` (the file path) and `page`
(zero-based) in its metadata. Drop your own PDF into `NoteBooks/Data/` and set
`PDF_PATH`, or run the fallback cell to download the public-domain
*Declaration of Independence* from Project Gutenberg.

---

```python
PDF_PATH = "NoteBooks/Data/my-document.pdf"

if os.path.exists(PDF_PATH):
    print(f"Found your PDF: {PDF_PATH}")
else:
    print(f"No file at {PDF_PATH!r} — drop one there, or run the")
    print("fallback cell below to download a Gutenberg sample.")
```

---

```python
# No PDF handy? Download a public-domain sample from Project Gutenberg.
import urllib.request

URL = "https://www.gutenberg.org/files/16780/16780-pdf.pdf"

if not os.path.exists(PDF_PATH):
    print(f"No file at {PDF_PATH!r} — using the Gutenberg sample instead.")
    PDF_PATH = "NoteBooks/Data/declaration-of-independence.pdf"

if not os.path.exists(PDF_PATH):
    print("Downloading a sample PDF …")
    urllib.request.urlretrieve(URL, PDF_PATH)

print(f"PDF ready: {PDF_PATH} ({os.path.getsize(PDF_PATH):,} bytes)")
```

---

```python
loader = PyPDFLoader(PDF_PATH)
pdf_docs = loader.load()

print(f"PDF → {len(pdf_docs)} Documents (one per page)")
print("First page metadata:", pdf_docs[0].metadata)
```

---

### 1.2 · Markdown — TextLoader

A `.md` file is plain text with a bit of syntax. `TextLoader` reads it as-is and
tags each `Document` with its `source` path. We load one page from the repo's
sample docs tree, `NoteBooks/Data/local-docs/docs/bge-embeddings.md`.

---

```python
MD_PATH = "NoteBooks/Data/local-docs/docs/bge-embeddings.md"

md_docs = TextLoader(MD_PATH, encoding="utf-8").load()

print(f"Markdown → {len(md_docs)} Document")
print("Metadata:", md_docs[0].metadata)
```

---

### 1.3 · CSV — CSVLoader

CSV data is tabular. `CSVLoader` turns **each row** into its own `Document`,
rendering the row as `column: value` lines, and tags it with the file `source`
and a `row` number. The sample file `NoteBooks/Data/sample.csv` holds a tiny
table of landmark ML papers. The next two cells define that file (writing it
only if it is missing), so the notebook stays self-contained.

---

```python
import csv
from pathlib import Path

CSV_PATH = Path("NoteBooks/Data/sample.csv")
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

rows = [
    ["title", "author", "year", "summary"],
    ["Attention Is All You Need", "Vaswani et al.", 2017,
     "Introduces the Transformer, a self-attention architecture."],
    ["BERT", "Devlin et al.", 2018,
     "Pre-trains a bidirectional Transformer for transfer learning."],
    ["Retrieval-Augmented Generation", "Lewis et al.", 2020,
     "Combines a generator with a retriever over a corpus."],
    ["Reciprocal Rank Fusion", "Cormack et al.", 2009,
     "Fuses ranked lists by summing the reciprocal of each rank."],
]
print(f"{len(rows) - 1} sample rows defined")
```

---

```python
if not CSV_PATH.exists():
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"Created {CSV_PATH}")

print(f"CSV ready: {CSV_PATH} ({CSV_PATH.stat().st_size} bytes)")
```

---

```python
csv_loader = CSVLoader(file_path=str(CSV_PATH))
csv_docs = csv_loader.load()

print(f"CSV → {len(csv_docs)} Documents (one per row)")
print("First row metadata:", csv_docs[0].metadata)
```

---

### 1.4 · Web — WebBaseLoader

`WebBaseLoader` fetches a URL, strips the HTML with BeautifulSoup and returns
one `Document` whose `source` is the URL. The sample is the article the repo's
own README links to — a write-up of building RAG systems from scratch. This
step needs internet access, and the extracted text will include some navigation
noise from the page (expected — real web data is messy).

---

```python
WEB_URL = "https://dev.to/gautamvhavle/building-production-rag-systems-from-zero-to-hero-2f1i"

web_docs = WebBaseLoader(WEB_URL).load()

print(f"Web → {len(web_docs)} Document")
print("Metadata:", web_docs[0].metadata)
print("Snippet:", web_docs[0].page_content.strip()[:80])
```

---

### 1.5 · JSON — JSONLoader

`JSONLoader` picks values out of a JSON file using a **`jq_schema`** — a tiny
query string. With `jq_schema=".[].content"` it selects the `content` key of
*every* object in the root array, so each object becomes one `Document`. This
is the fastest route for "records whose text lives in one named key". The
sample file `NoteBooks/Data/sample.json` holds three short concept notes.

---

```python
import json

JSON_PATH = Path("NoteBooks/Data/sample.json")

notes = [
    {"content": "A vector database stores embeddings and returns the nearest neighbours of a query vector."},
    {"content": "RecursiveCharacterTextSplitter cuts documents by character count while respecting paragraph boundaries."},
    {"content": "Metadata filters restrict which chunks a search considers before any similarity comparison is made."},
]

if not JSON_PATH.exists():
    JSON_PATH.write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(f"Created {JSON_PATH}")

print(f"JSON ready: {JSON_PATH} ({JSON_PATH.stat().st_size} bytes)")
```

---

```python
json_loader = JSONLoader(
    file_path=str(JSON_PATH),
    jq_schema=".[].content",   # take the "content" key of every array element
)

json_docs = json_loader.load()

print(f"JSON → {len(json_docs)} Documents")
print("First note metadata:", json_docs[0].metadata)
print("First note text     :", json_docs[0].page_content)
```

---

### 1.6 · Tag & combine — one list, with provenance

All five lists are in hand. Now we stamp every `Document` with a `format` label
while keeping the loader's own metadata (`source`, `page`, …). Because
`Document`s are immutable, we build new ones with merged metadata. The result
is one list, `all_docs`, that knows **exactly where every chunk came from** —
the foundation for the source tracking and filtering in Section 5.

---

```python
def tag(docs, fmt):
    """Return new Documents with `format` added to their metadata."""
    return [
        Document(
            page_content=d.page_content,
            metadata={**d.metadata, "format": fmt},
        )
        for d in docs
    ]

pdf_tagged = tag(pdf_docs, "pdf")
md_tagged = tag(md_docs, "md")
csv_tagged = tag(csv_docs, "csv")
web_tagged = tag(web_docs, "web")
json_tagged = tag(json_docs, "json")

all_docs = pdf_tagged + md_tagged + csv_tagged + web_tagged + json_tagged

print(f"Combined corpus: {len(all_docs)} Documents")
```

---

```python
from collections import Counter

counts = Counter(d.metadata.get("format") for d in all_docs)
for fmt in ["pdf", "md", "csv", "web", "json"]:
    print(f"{fmt:4} → {counts.get(fmt, 0)} Documents")
```

---

```python
csv_example = next(d for d in all_docs if d.metadata["format"] == "csv")
print("Example tagged Document (a CSV row):")
print("  metadata:", csv_example.metadata)
print("  content :", csv_example.page_content)
```

---

## 2 · Split — one splitter for the whole mix

The splitter is format-agnostic: it only sees text. `RecursiveCharacterTextSplitter`
cuts each `Document` into overlapping `chunk_size`-character chunks. The tag you
added in `1.6` is part of `metadata`, and the splitter **carries metadata
through** — every chunk inherits its parent's `format` and `source`. That is
what makes per-format filtering possible after splitting.

---

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(all_docs)

print(f"Split {len(all_docs)} Documents into {len(chunks)} chunks")
```

---

```python
from collections import Counter

print("Chunk formats:", dict(Counter(c.metadata["format"] for c in chunks)))
print("Chunk 0 metadata:", chunks[0].metadata)
print("Chunk 0 text:", chunks[0].page_content[:100])
```

---

## 3 · Embed — text → vectors

Same embedding model as Projects 01–02: `GoogleGenerativeAIEmbeddings` sends
each chunk to the Gemini API and gets a 3072-dimension vector back. The format
a chunk came from does not matter to the embedder — text is text — which is why
one model can index a mixed corpus.

---

```python
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

sample_vec = embeddings.embed_query("multi-format vector search")
print(f"Embedding dimension: {len(sample_vec)}")
```

---

## 4 · Store — one Chroma store, five formats

Every chunk — PDF page, markdown note, CSV row, web paragraph, JSON record — is
embedded and indexed into **the same** Chroma collection. Chroma stores the
`(text, vector, metadata)` triple, so the `format` / `source` labels survive in
the index and become searchable filters in Section 5. A distinct
`collection_name` keeps this project's index separate from other notebooks' if
you run several of them in the same kernel.

---

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="project13_multi_format",
)
print(f"Chroma index created ({len(chunks)} chunks stored)")
```

---

## 5 · Retrieve — similarity search, source tracking & filtering

Retrieval works exactly as before: embed the query, return the `k` closest
chunks. Two new tricks:

* **Source tracking** — every retrieved chunk's `metadata` tells you which
  format and file/URL it came from, so you can show provenance alongside the
  answer.
* **Metadata filtering** — a `where` dict (`{"format": "md"}`) makes Chroma
  only compare against markdown chunks, skipping every other format *before*
  the similarity computation. This is a cheap pre-filter.

---

### 5.1 · The filter keyword: `where` vs `filter`

Chroma metadata filters are **`where` dictionaries**: `{"format": "md"}`.
Different `langchain-chroma` versions name the *Python keyword* differently —
older ones accept `where=`, newer ones `filter=` (matching `langchain-core`).
The cell below probes the installed version once, and a tiny `scoped_search`
helper hides the difference so every call below reads as the canonical
`where={...}` form.

---

```python
import inspect

FILTER_KW = "where" if "where" in inspect.signature(Chroma.similarity_search).parameters else "filter"
print(f"This install accepts `{FILTER_KW}=` (canonical form: where={{...}})")


def scoped_search(store, query, k=4, where=None):
    """similarity_search_with_score, `where` dict on any langchain-chroma."""
    return store.similarity_search_with_score(query, k=k, **{FILTER_KW: where})
```

---

```python
query = "Which embedding model runs locally for documentation search?"

hits = scoped_search(vector_store, query, k=4)

for doc, score in hits:
    fmt = doc.metadata["format"]
    src = doc.metadata["source"].split("/")[-1]
    print(f"[{score:.4f}] {fmt:4} | {src[:40]} | {doc.page_content[:45]}")
```

---

```python
from collections import Counter

formats_in_results = Counter(d.metadata["format"] for d, _ in hits)
print("Source tracking — formats retrieved:", dict(formats_in_results))
```

---

```python
md_only = scoped_search(vector_store, query, k=4, where={"format": "md"})

for doc, score in md_only:
    src = doc.metadata["source"].split("/")[-1]
    print(f"[{score:.4f}] {doc.metadata['format']:4} | {src} | {doc.page_content[:45]}")
```

---

```python
print("After the filter, every hit is markdown:",
      all(d.metadata["format"] == "md" for d, _ in md_only))
print(f"{len(md_only)} chunks retrieved, all from {{'format': 'md'}}")
```

---

## 6 · Prompt — package context + question

Identical template to the earlier projects: fuse the retrieved chunks into
`{context}`, add the `{question}`, and instruct the LLM to answer *only* from
the context. The prompt does not care whether the chunks came from a PDF, a
web page or a JSON record — and here it is built from the **markdown-filtered**
context, so the answer will be grounded in the markdown source.

---

```python
context = "\n\n".join(doc.page_content for doc, _ in md_only)
print(f"Context from {len(md_only)} markdown-only chunks ({len(context)} chars)")
```

---

```python
template = """
You are a helpful assistant.

Answer the question using ONLY the provided context.
If the answer is not contained in the context, say:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
print(prompt.format(context=context[:300], question=query))
```

---

## 7 · Answer — Gemini LLM

The last block sends the rendered prompt to `ChatGoogleGenerativeAI`
(`gemini-2.5-flash`). Because retrieval was scoped to markdown with
`where={"format": "md"}`, the model answers from the markdown page about BGE
embeddings — a grounded answer that also proves the filter shaped the result.

---

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = prompt.invoke({"context": context, "question": query})
response = llm.invoke(messages)

print(response.content)
```

---

## 8 · Try it yourself

Change the query, change `k`, or change the `where` filter to another format
(`{"format": "pdf"}`, `{"format": "web"}`, …) and re-run retrieval + answer.
The three cells below query three different formats on purpose — notice that
source tracking tells you *which* format answered each time.

---

```python
pdf_query = "What rights do the people have?"

pdf_hits = scoped_search(vector_store, pdf_query, k=3, where={"format": "pdf"})

for doc, score in pdf_hits:
    page = doc.metadata.get("page", "?")
    print(f"[page {page}] {doc.page_content[:70]}")
```

---

```python
csv_query = "Who introduced the Transformer architecture?"

csv_hits = scoped_search(vector_store, csv_query, k=2)

for doc, score in csv_hits:
    print(f"{doc.metadata['format']:4} | {doc.page_content[:80]}")
```

---

```python
web_query = "What did the author learn building RAG systems?"

web_hits = scoped_search(vector_store, web_query, k=3, where={"format": "web"})

for doc, score in web_hits:
    print(f"[{score:.4f}] {doc.metadata['format']:4} | {doc.page_content[:60]}")
```

---

## What you should notice

* **All five loaders speak `Document`.** PDF pages, markdown files, CSV rows,
  web pages and JSON records all become the same object, so one splitter, one
  embedder and one store handle the whole mix — the "swap one block" design at
  its widest.
* **Metadata is the glue.** The `format` tag added in `1.6` survives splitting,
  storage and retrieval untouched, which is what makes both source tracking and
  filtering possible.
* **Source tracking is free provenance.** After any search you can show the
  user which format and file/URL each retrieved chunk came from — no extra
  indexing work needed.
* **The filter is a pre-filter, not a re-rank.** `where={"format": "md"}`
  narrows the candidate set *before* similarity comparison, so it is cheap even
  when one format dominates a large corpus.
* **`JSONLoader` needs `jq`.** The `jq_schema` (`".[].content"`) is evaluated by
  the `jq` package; that is the one extra install this project needs.
* **Mixed queries pull mixed formats.** Unfiltered, one question can return PDF
  *and* web chunks; the filter is how you force a single source.

---

## Exercises

1. **Add a sixth format.** Point a `TextLoader` at a `.txt` file (or any new
   `.md`), tag it `format="txt"`, add it to `all_docs`, and confirm the splitter
   and store accept it with zero other changes.
2. **Filter by something other than format.** Tag a couple of chunks with an
   extra key (e.g. `"topic"`) and run `where={"topic": ...}` — does the filter
   compose with `format`?
3. **Compare unfiltered vs filtered answers.** Ask the same question with and
   without `where={"format": "pdf"}`, and check whether the retrieved chunks
   and the final answer differ.
