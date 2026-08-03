> Source notebook: `NoteBooks/Project-04-Markdown-Documentation-RAG/01-markdown-documentation-rag.ipynb`


---

# Project 04 — Markdown Documentation RAG

**Goal:** Split on structure instead of randomly — Markdown already has sections.

```
Loader      : DirectoryLoader (Topics/**/*.md)
Splitter    : MarkdownHeaderTextSplitter (vs naive RecursiveCharacterTextSplitter)
Embedding   : BGE (local)
Vector DB   : FAISS
Retriever   : Similarity Search (Top-K)
Prompt      : Citation Prompt (source-aware)
LLM         : Gemini
```

The whole pipeline stays the same as Project 01 — only the splitter changes,
and the prompt becomes source-aware. Splitting on Markdown headers keeps each
section intact, so a retrieved chunk is a *whole section* instead of a random
slice of text.

Learn:

* Structure-preserving splitting (headers → chunks with `H1`/`H2`/`H3` metadata)
* Naive vs header-aware chunking, side by side
* Indexing your own repository's documentation as a corpus

---

## 0 · Setup — environment & keys

Gemini needs a `GOOGLE_API_KEY` in a `.env` file at the repo root (copy
`.env.example` and paste your key from Google AI Studio). The key is only used
at the very end, in the Answer step — everything before it runs fully offline.

Two optional packages are needed for THIS project only:

* `faiss-cpu` — the FAISS vector index (Project 01 used Chroma instead)
* `sentence-transformers` — the local BGE embedding model

Both are installed with `%pip` below. If you already ran earlier projects, only
the pieces you do not have yet are needed.

---

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY", "")
if api_key:
    print(f"GOOGLE_API_KEY found: {api_key[:4]}…{api_key[-2:]} ({len(api_key)} chars)")
else:
    print("GOOGLE_API_KEY missing — copy .env.example to .env and add your key.")
```

---

```python
# Optional deps for THIS project only.
%pip install faiss-cpu
%pip install sentence-transformers
```

---

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
```

---

## 1 · Load — the repo's own curriculum as a corpus

What better test corpus than the curriculum itself? `DirectoryLoader` walks a
folder and loads every file matching a glob into LangChain `Document`s. Here we
point it at `Topics/` — the folder holding every project card — and ask for all
markdown files (`**/*.md`).

Two WHYs:

* It is a **self-demo**: you immediately see the whole curriculum indexed, and
  every chunk carries a `source` metadata key pointing at its file.
* It is **real documentation RAG**: docs that mix prose, headings and code
  blocks — exactly the shape this splitter was designed for.

`TextLoader` reads a `.md` file as plain text; the structure-preserving work
happens in the Split step, not here. (An alternative like
`UnstructuredMarkdownLoader` would also parse the markdown into elements, but
adds the heavy `unstructured` dependency.)

---

```python
def find_dir(name, start=os.getcwd()):
    """Walk upward from `start` until a folder named `name` is found."""
    d = os.path.abspath(start)
    while True:
        candidate = os.path.join(d, name)
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            raise FileNotFoundError(f"'{name}' not found above {start}")
        d = parent

TOPICS_DIR = find_dir("Topics")
print(TOPICS_DIR)
```

---

```python
loader = DirectoryLoader(
    TOPICS_DIR,
    glob="**/*.md",
    loader_cls=TextLoader,
)

docs = loader.load()
print(f"Loaded {len(docs)} markdown documents from Topics/")
```

---

```python
first = docs[0]
print(first.metadata)
print(first.page_content[:140])
```

---

## 2 · Split — naive vs structure-preserving

Here is the whole point of Project 04. A `RecursiveCharacterTextSplitter` cuts
by character count: it starts at character 0 and chops every `chunk_size`
characters, with `chunk_overlap` characters of bleed between chunks. It knows
nothing about Markdown, so chunk boundaries land **mid-section** — one chunk
may start inside the `## Why` section and end inside `## Code`.

The `MarkdownHeaderTextSplitter` is the opposite: it splits **on headings**
(`#`, `##`, `###`). Each chunk is a complete section, and the header path is
recorded in the chunk's metadata (`H1: Project 04…`, `H2: Why`). Retrieval then
returns whole, self-contained sections — and the prompt can even cite which
section it used.

---

```python
naive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
naive_chunks = naive_splitter.split_documents(docs)
print(f"Naive splitting → {len(naive_chunks)} chunks")
print("metadata of chunk 0:", naive_chunks[0].metadata)
print(naive_chunks[0].page_content[:120])
```

---

Now the header-aware version. Note that in current `langchain-text-splitters`,
`MarkdownHeaderTextSplitter` exposes `split_text(text)`, not
`split_documents(docs)` — so we call it once per loaded document and **merge the
file-level `source` metadata into every section chunk** by hand. That merge is
important: it is what lets the prompt cite a file, not just a section.

---

```python
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ]
)

header_chunks = []
for doc in docs:
    for part in md_splitter.split_text(doc.page_content):
        part.metadata = {**doc.metadata, **part.metadata}
        header_chunks.append(part)

print(f"Header-based splitting → {len(header_chunks)} chunks")
```

---

```python
sample = header_chunks[0]
print(sample.metadata)
print(sample.page_content[:160])
```

---

## 3 · Embed — BGE

Embeddings turn text into vectors: numbers where similar text lands near each
other. Project 04 switches the embedding model to **BGE** (`bge-small-en`), a
small multilingual model from BAAI that runs **locally** via
`sentence-transformers` — no API key needed.

We embed one sample string to prove the model works and to read the vector
dimensionality (BGE small uses 384 numbers per chunk).

---

```python
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en",
)

vec = embeddings.embed_query("What is retrieval-augmented generation?")
print(f"BGE small embedding dims: {len(vec)}")
print(vec[:5])
```

---

## 4 · Store — FAISS

FAISS (Facebook AI Similarity Search) is an in-memory vector index: it stores
the chunk vectors and answers "which vector is nearest to the query" very fast.
It is the classic choice when your corpus fits in RAM.

* The index lives in memory — close the kernel and it is gone (rebuilt by the
  Load + Split + Embed steps above).
* `FAISS.from_documents` embeds every chunk and builds the index in one call.
* Later projects persist it with `save_local` and reload it with
  `FAISS.load_local`.

---

```python
vector_store = FAISS.from_documents(
    documents=header_chunks,
    embedding=embeddings,
)
print(f"FAISS index built over {len(header_chunks)} chunks")
```

---

## 5 · Retrieve

A query gets embedded the same way, and FAISS returns the `k` nearest chunk
vectors. `similarity_search_with_score` also returns the raw distance — smaller
is closer, so it tells you how confident the retrieval is.

---

```python
query = "What is the goal of the HTML Documentation project?"
results = vector_store.similarity_search_with_score(query, k=3)

for i, (chunk, score) in enumerate(results, 1):
    src = chunk.metadata.get("source", "?")
    snippet = chunk.page_content[:80].replace("\n", " ")
    print(f"[{i}] score={score:.3f} source={src}\n    {snippet}\n")
```

---

## 6 · Prompt — the source-aware citation prompt

Project 01 used a "basic context + question" prompt. Project 04 upgrades to the
**Citation Prompt**: it asks the model to answer only from the context AND to
name the source of every claim. To make that possible we package the retrieved
chunks **with their source metadata** — the `[source: …]` tags you see below
are exactly what the model cites back.

---

```python
prompt = ChatPromptTemplate.from_template(
    """You are a documentation assistant.

Answer the question using ONLY the provided context.
For every claim, cite the source file it came from.

Context:
{context}

Question:
{question}

Answer (with sources):
"""
)
```

---

```python
def pack_for_prompt(docs):
    """Join chunks, tagging each one with its source file."""
    return "\n\n".join(
        f"[source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

context = pack_for_prompt([r[0] for r in results])
print(f"{len(context)} chars of context, each chunk tagged with its source file")
```

---

```python
messages = prompt.invoke({"context": context, "question": query})
print(messages.content[:300])
```

---

## 7 · Answer — Gemini

The last block is the LLM. We use the same Gemini model as Project 01 —
`gemini-2.5-flash` — because everything in front of it has already changed.
The prompt with tagged sources goes in, and the answer comes out with citations
you can check against the corpus.

---

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

answer = llm.invoke(messages)
print(answer.content)
```

---

## 8 · Try it yourself

Change the query, change `k`, or ask a question that spans two sections — the
header metadata will show you exactly which sections got retrieved.

---

```python
query2 = "How is Project 05 different from Project 04?"
for hit in vector_store.similarity_search(query2, k=2):
    print(hit.metadata.get("source"), "→", hit.page_content[:70].replace("\n", " "))
```

---

## What you should notice

* `MarkdownHeaderTextSplitter` turns Markdown structure into chunk *metadata*
  (`H1`, `H2`, `H3`) — retrieval now returns whole sections, and the prompt can
  cite them.
* The naive splitter produced more, smaller, mid-section chunks; the
  header-aware splitter produced fewer, self-contained chunks.
* Every header chunk carries the file-level `source` because we merged it
  manually — `split_text` alone would drop it.
* Embeddings (BGE) and search (FAISS) run fully offline; only the final LLM
  call needs `GOOGLE_API_KEY`.
* A citation prompt turns retrieval quality into verifiable answers: if the
  right section is retrieved, the answer carries the right source.
* Self-indexing the curriculum is a neat demo: your own READMEs become the
  knowledge base.

---

## Exercises

1. Add `("####", "H4")` to `headers_to_split_on` and re-run
   Split → Store → Retrieve. How does the chunk count change?
2. Persist the FAISS index (`vector_store.save_local("faiss_index")`) and load
   it back in a fresh kernel with `FAISS.load_local(..., embeddings=embeddings)`.
3. Swap the prompt back to Project 01's basic template. Compare answer style:
   does the citation prompt actually change how the model responds?
