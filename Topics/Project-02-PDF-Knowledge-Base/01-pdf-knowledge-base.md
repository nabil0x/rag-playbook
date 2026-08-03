> Source notebook: `NoteBooks/Project-02-PDF-Knowledge-Base/01-pdf-knowledge-base.ipynb`


---

# Project 02 — PDF Knowledge Base

**Goal:** Swap the loader only — PDFs instead of web pages.

```
Loader      : PyPDFLoader
Splitter    : RecursiveCharacterTextSplitter
Embedding   : Gemini
Vector DB   : Chroma
Retriever   : Similarity (Top-K)
Prompt      : Basic Context + Question
LLM         : Gemini
```

Everything below the loader is identical to Project 01 — that is the whole
point. You change **one block** (the loader) and the rest of the pipeline just
works. A university regulation, a research paper, a book or your course notes
can all become a knowledge base this way.

---

## 0 · Setup — environment & keys

The pipeline calls the **Gemini embedding API** and the **Gemini LLM**, so you
need a `GOOGLE_API_KEY`. Create a `.env` file at the repo root (copy
`.env.example`) and paste your key from <https://aistudio.google.com/>. The
cell below only prints a masked preview (`key[:4]…`), never the full key.

PDF loading uses `PyPDFLoader` from `langchain-community`, which needs `pypdf`
under the hood. Both are already in `requirements.txt`, so **no extra install
is needed for this project**.

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
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
```

---

## 1 · Load — PDFs via PyPDFLoader

A PDF is a container: pages, each with a **text layer** (the selectable text)
plus fonts and layout. `PyPDFLoader` reads that text layer. **Why it matters:**
RAG needs plain text, so the loader's job is to pull the words out of the PDF.

`PyPDFLoader.load()` returns **one `Document` per page** — a 12-page PDF becomes
12 Documents. Each Document's metadata carries the `source` path and the
zero-based `page` number, which you can use later to tell the user *which page*
an answer came from.

* Drop your own PDF into `NoteBooks/Data/` and point `PDF_PATH` at it, **or**
* run the fallback cell below to download a public-domain sample from Project
  Gutenberg.

---

```python
# Point this at a PDF you dropped into NoteBooks/Data/.
PDF_PATH = "NoteBooks/Data/my-document.pdf"

if os.path.exists(PDF_PATH):
    print(f"Found your PDF: {PDF_PATH}")
else:
    print(f"No file at {PDF_PATH!r} — either drop one there,")
    print("or run the fallback cell below to download a sample.")
```

---

```python
# No PDF handy? Download a public-domain sample from Project Gutenberg:
# "The Declaration of Independence of the United States of America".
import urllib.request

PDF_PATH = "NoteBooks/Data/declaration-of-independence.pdf"
URL = "https://www.gutenberg.org/files/16780/16780-pdf.pdf"

if not os.path.exists(PDF_PATH):
    print("Downloading a sample PDF …")
    urllib.request.urlretrieve(URL, PDF_PATH)

print(f"PDF ready: {PDF_PATH} ({os.path.getsize(PDF_PATH):,} bytes)")
```

---

```python
loader = PyPDFLoader(PDF_PATH)
pdf_docs = loader.load()

print(f"Loaded {len(pdf_docs)} Document objects")
```

---

```python
print("One Document per page — the metadata tells you which page:")
print(pdf_docs[0].metadata)
print()
print("First page snippet:")
print(pdf_docs[0].page_content[:180])
```

---

## 2 · Split — RecursiveCharacterTextSplitter

A page can be hundreds of words — too big to encode into a single useful vector
and too noisy to retrieve precisely. The splitter chops the PDF text into
**overlapping chunks** of `chunk_size` characters, carrying the previous
`chunk_overlap` characters across each seam so a thought that straddles a
boundary is not lost.

This is the same splitter as Project 01 — only the *loader* changed in this
project.

---

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(pdf_docs)

print(f"Split {len(pdf_docs)} pages into {len(chunks)} chunks")
```

---

```python
print("Sample chunk (first 200 chars):")
print(chunks[0].page_content[:200])
print()
print("It still remembers which page it came from:")
print(chunks[0].metadata)
```

---

## 3 · Embed — text → vectors

Embeddings turn text into a list of numbers (a *vector*). Texts that mean
similar things end up **near each other** in vector space, so "What does the
declaration say about rights?" and "the unalienable rights of the people" land
close together even though they share few words. `GoogleGenerativeAIEmbeddings`
sends each chunk to the Gemini embedding API and returns one vector per chunk.

Expect a long vector: `gemini-embedding-2-preview` returns **3072** floats.

---

```python
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

sample_vec = embeddings.embed_query("PDF knowledge base with vector search")
print(f"Embedding dimension: {len(sample_vec)}")
```

---

## 4 · Store — Chroma

Chroma takes every chunk, calls the embedding model on it, and stores the
`(text, vector, metadata)` triple in an index it can search by similarity.
`Chroma.from_documents` does the embedding and indexing in one call.

Here we keep the index in memory, exactly like Project 01. Chroma can also
persist to a folder with `persist_directory="chroma_langchain_db/"` so you do
not re-embed the whole PDF on every run.

---

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)
print("Chroma index created")
```

---

## 5 · Retrieve — similarity search

Retrieval is the RAG "search" step: the query is embedded the same way as the
chunks, then Chroma returns the `k` stored chunks whose vectors are **closest**
to the query vector. `k` (top-k) is your only lever here — a bigger `k` gives
more context but also more noise for the LLM.

`similarity_search_with_score` also returns a distance per hit; for Chroma's
default metric **lower is closer** (more relevant).

---

```python
query = "What did the declaration of independence announce?"

hits = vector_store.similarity_search_with_score(query, k=3)

for doc, score in hits:
    print(f"[{score:.4f}] (page {doc.metadata['page']}) {doc.page_content[:80]}")
```

---

## 6 · Prompt — package context + question

The LLM only ever sees a prompt. The prompt fuses the **retrieved chunks** (the
context) with the **user's question** and instructs the model to answer *only*
from the context — if the context has no answer, the model should say so instead
of inventing one. This is the same basic template as Project 01.

---

```python
context = "\n\n".join(doc.page_content for doc, _ in hits)
print(f"Context is {len(context)} characters from {len(hits)} chunks")
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
```

---

```python
rendered = prompt.format(context=context[:400], question=query)
print(rendered)
```

---

## 7 · Answer — Gemini LLM

The last block sends the rendered prompt to the Gemini chat model
(`gemini-2.5-flash`), which reads the context you retrieved from the PDF and
answers. Loading, splitting and retrieval all ran locally — the LLM call (like
the embeddings) is the only part that leaves your machine.

---

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = prompt.invoke({"context": context, "question": query})
response = llm.invoke(messages)

print(response.content)
```

---

## 8 · Try it yourself

Ask your own questions about *your* PDF — you changed the loader, so now any PDF
can become a knowledge base. Change the query, the `k` in retrieval, or the
`chunk_size` / `chunk_overlap` above and re-run to see how the answers shift.

---

```python
query2 = "Who wrote the document we just indexed?"

hits2 = vector_store.similarity_search(query2, k=2)
context2 = "\n\n".join(d.page_content for d in hits2)

answer2 = llm.invoke(prompt.format_messages(context=context2, question=query2))
print(answer2.content)
```

---

```python
query3 = "How many pages does the document have?"

hits3 = vector_store.similarity_search(query3, k=2)
context3 = "\n\n".join(d.page_content for d in hits3)

answer3 = llm.invoke(prompt.format_messages(context=context3, question=query3))
print(answer3.content)
```

---

## What you should notice

* **One Document per page.** `PyPDFLoader` returned as many Documents as the PDF
  has pages — a PDF is *already* chunked by its layout before you even split.
* **Page metadata is free provenance.** Every chunk remembers its `source` and
  `page`, so you can tell the user *which page* an answer came from.
* **Text layer ≠ scanned image.** PDF parsing only works when the file has a real
  text layer. Scanned PDFs return empty `page_content` until you add OCR — a
  classic RAG gotcha.
* **Only the loader changed.** Split, embed, store, retrieve, prompt and LLM are
  the Project 01 blocks untouched — the "swap one block" design in action.
* **Chunks can span pages.** The splitter does not care about PDF page
  boundaries, so a 1000-character chunk may mix the end of one page and the
  start of the next.

---

## Exercises

1. Change `chunk_size` to `500` and `chunk_overlap` to `100`, re-run Split →
   Answer, and compare how the answer to the same query changes.
2. Ask a question whose answer spans two pages (e.g. "who signed it and why?")
   and inspect the `page` metadata of the retrieved chunks — did the splitter
   keep the answer together?
3. Ask a question *not* in the document and confirm the model says it does not
   know, instead of guessing.
