> Source notebook: `NoteBooks/Projects/Project-07-Compare-Vector-Databases/01-comparing-vector-databases.ipynb`


---

# Project 07 — Compare Vector Databases

**Goal:** Same documents, same embeddings — only the store changes.

```
Loader      : Directory Loader (Topics/*.md)   ← fixed corpus
Splitter    : RecursiveCharacterTextSplitter   ← fixed
Embedding   : Gemini Embedding (embedded once) ← fixed
Vector DB   : Chroma  ↓  FAISS  ↓  Qdrant  (↓ LanceDB later)
Retriever   : Similarity Search (top-k)
LLM         : (none — this project measures the store, not the answer)
```

Learn:

- Why the vector database matters in RAG (it decides which chunks you retrieve)
- How to benchmark index time, query speed, correctness, persistence, filtering
- What the difference between an embedded store, a library index and a server store is

---

## 0 · Setup — environment & keys

Every notebook starts the same way: load the `.env` file, confirm the Google
key is present (masked), and import the libraries. This project adds **two
optional packages** that are NOT in `requirements.txt` yet:

- `faiss-cpu` — the FAISS index (CPU build)
- `langchain-qdrant` — LangChain's Qdrant integration (pulls `qdrant-client`, so we
  can run Qdrant fully in memory without starting a server)

You still need `GOOGLE_API_KEY` because the corpus vectors come from Gemini
(embedded exactly once, then shared by all three stores).

---

```python
from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("GOOGLE_API_KEY", "")
if key:
    print("GOOGLE_API_KEY set:", key[:4] + "…" + key[-4:])
else:
    print("GOOGLE_API_KEY missing — copy .env.example to .env and add your key.")
```

---

### Optional installs for THIS project only

FAISS and Qdrant are used only in Project 07. Run the cell below once; if they
are already installed, pip reports *Requirement already satisfied* and nothing
is re-downloaded. No server is started for Qdrant — we use `location=":memory:"`.

---

```python
# pip install faiss-cpu langchain-qdrant
%pip install -q faiss-cpu langchain-qdrant
```

---

### Imports

Grouped, one cell. Everything here is a real library from the cheat-sheet.
`Qdrant` comes from `langchain_qdrant` (installed above); `FAISS` lives in
`langchain_community.vectorstores`.

---

```python
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_qdrant import Qdrant

import time
import os
```

---

## 1 · Load — one fixed corpus

The whole point of Project 07 is **isolation**: only the store changes. So the
corpus must be identical, small enough to rebuild fast, and available offline.
The repo's own `Topics/*.md` files (the 18 project cards) are perfect: dozens of
small documents, stable, and already on disk.

We resolve the repo root by walking up from the notebook's working directory
until we find a folder that contains both `Topics/` and `NoteBooks/`, then load
every markdown file with `DirectoryLoader` + `TextLoader`.

---

```python
def find_repo_root():
    d = Path.cwd()
    for _ in range(6):
        if (d / "Topics").is_dir() and (d / "NoteBooks").is_dir():
            return d
        d = d.parent
    return None

REPO_ROOT = find_repo_root()
print("repo root:", REPO_ROOT)
```

---

```python
loader = DirectoryLoader(
    str(REPO_ROOT / "Topics"),
    glob="**/*.md",
    loader_cls=TextLoader,
    silent_errors=True,
)
docs = loader.load()
print("documents:", len(docs))

# Give every doc a short, filterable field (full path varies by machine).
for doc in docs:
    doc.metadata["source_file"] = Path(doc.metadata["source"]).name

print("sample metadata:", docs[0].metadata)
print("first chars:", docs[0].page_content[:60].replace(chr(10), " "))
```

---

## 2 · Split

Same splitter as Project 01. Chunks are the unit the store indexes and the
retriever searches — if two stores disagree, it will not be because of the
splitting. `chunk_size=500` keeps the corpus to a few hundred chunks, so the
index build is fast enough to re-run while you experiment.

---

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)
print("chunks:", len(chunks))
```

---

```python
print("sample chunk:")
print(chunks[0].page_content[:120].replace(chr(10), " "))
print("…")
print("source_file:", chunks[0].metadata["source_file"])
```

---

## 3 · Embed once — shared vectors

Here is the key fairness trick. If each store embedded the chunks by itself, the
timing would mix **embedding** cost (network) with **indexing** cost (local).
Instead we embed the whole corpus a single time with Gemini, then hand the same
vectors to every store. Any difference we measure is the store, not the embedder.

`embed_documents` returns one vector per chunk. The dimensionality (768 for
`gemini-embedding-2-preview`) is a property of the model, not the store.

---

```python
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

texts = [chunk.page_content for chunk in chunks]
vectors = embeddings.embed_documents(texts)
print("chunks:", len(texts))
print("dim:  ", len(vectors[0]))
```

---

### Reuse the precomputed vectors

`FAISS.from_documents` / `Qdrant.from_documents` / `Chroma.from_documents`
normally call `embed_documents` internally. To force them to use **our** vectors
instead, we pass a tiny adapter: `embed_documents` returns the cached vectors
from a `text → vector` map, while `embed_query` still calls the real model for
new questions. This way all three stores index byte-identical vectors.

---

```python
class ReuseEmbeddings:
    """Return cached corpus vectors; embed new queries with the real model."""

    def __init__(self, model, text_to_vector):
        self._model = model
        self._cache = dict(text_to_vector)

    def embed_documents(self, texts):
        return [self._cache[t] for t in texts]

    def embed_query(self, text):
        return self._model.embed_query(text)


cached_emb = ReuseEmbeddings(embeddings, zip(texts, vectors))
print("cached vectors ready:", len(cached_emb._cache))
```

---

## 4 · The rotation — same vectors, three stores

We define one builder per store. This is the *only* code that differs:

- **Chroma** — embedded store, persists to a local directory by default.
- **FAISS** — a library index held in RAM (fast, no persistence unless you call
  `save_local`).
- **Qdrant** — a server-style store; we run it in memory (`location=":memory:"`),
  no server needed.

---

```python
def build_chroma(chunks, embedding):
    return Chroma.from_documents(chunks, embedding=embedding,
                                 collection_name="p07_compare",
                                 persist_directory="chroma_db_p07")


def build_faiss(chunks, embedding):
    return FAISS.from_documents(chunks, embedding=embedding)


def build_qdrant(chunks, embedding):
    return Qdrant.from_documents(chunks, embedding=embedding,
                                 collection_name="p07_compare",
                                 location=":memory:")
```

---

### Measure index (build) time

`time.perf_counter()` around each builder. Because the embedder is cached, this
number is **pure indexing**: how long the store takes to add the vectors and
build its search structure.

---

```python
builders = {
    "chroma": build_chroma,
    "faiss": build_faiss,
    "qdrant": build_qdrant,
}

stores = {}
index_ms = {}
for name, builder in builders.items():
    t0 = time.perf_counter()
    stores[name] = builder(chunks, cached_emb)
    index_ms[name] = (time.perf_counter() - t0) * 1000
    print(f"{name:7s} index {index_ms[name]:8.1f} ms")
```

---

## 5 · Query — one fixed question, every store

Retrieval quality is checked with a single question and `k=3`. We measure how
long each store takes to answer it and look at the top hit. Same vectors → the
top-3 should be identical across stores; the *speed* is what differs.

---

```python
QUERY = "What does the RAG curriculum teach about vector databases?"

query_ms = {}
for name, store in stores.items():
    t0 = time.perf_counter()
    hits = store.similarity_search(QUERY, k=3)
    query_ms[name] = (time.perf_counter() - t0) * 1000
    top = hits[0].page_content.replace(chr(10), " ")[:55]
    print(f"{name:7s} query {query_ms[name]:7.2f} ms | top: {top}…")
```

---

### Correctness check

"Correctness" here means: did the store return the same relevant chunks as its
siblings? Since the vectors are identical, a correct store must agree with the
others on the top-3 set. Disagreement would point at a store bug or a different
distance metric.

---

```python
def top3_signature(store):
    hits = store.similarity_search(QUERY, k=3)
    return tuple(h.page_content[:40] for h in hits)


sigs = {name: top3_signature(store) for name, store in stores.items()}
for name in stores:
    print(name, "→", sigs[name][0][:35])
print("same top-3 across stores:", len(set(sigs.values())) == 1)
```

---

### The comparison table

A markdown table is printed from the measured numbers. Rows = the thing we
rotate (the store); columns = the things we measure. `same top-3` is the
correctness column.

Keep in mind: numbers depend on the machine, the corpus size and the vector
dimension — the *ranking* between stores is the lesson, not the absolute ms.

---

```python
print("| store   | index (ms) | query (ms) | same top-3 |")
print("|---------|-----------:|-----------:|:----------:|")
for name in stores:
    same = "yes" if len(set(sigs.values())) == 1 else "no"
    print(f"| {name} | {index_ms[name]:8.1f} | {query_ms[name]:7.2f} | {same} |")
```

---

## 6 · Deep-dive — persistence

Persistence decides what survives a restart. Check it per store:

- **Chroma** wrote a real directory (`chroma_db_p07/`): reopen later and the
  vectors are still there — it is an *embedded persistent* store.
- **FAISS** is RAM-only: we explicitly `save_local()` a folder; reloading needs
  `FAISS.load_local(path, embeddings=..., allow_dangerous_deserialization=True)`.
- **Qdrant** in-memory lives and dies with this notebook — durability only comes
  with a real Qdrant server (or an embedded mode).

---

```python
print("chroma persisted dir:", os.path.isdir("chroma_db_p07"))

stores["faiss"].save_local("faiss_p07_index")
print("faiss saved files:  ", sorted(os.listdir("faiss_p07_index")))
```

---

## 7 · Deep-dive — filtering

Filtering narrows a search by metadata *before* ranking (e.g. "only the README
files"). The stores differ here a lot:

- **Chroma** supports a plain dict filter on metadata.
- **FAISS** has no native metadata filter in `similarity_search` — you must
  filter after retrieval (or filter the ids yourself).
- **Qdrant** has rich filters, but the syntax is its own
  `filter_={"must": [{"key": …, "match": …}]}` structure.

Let's see the Chroma one live; the others are noted below.

---

```python
hits = stores["chroma"].similarity_search(
    QUERY, k=3,
    filter={"source_file": "README.md"},
)
print("chroma filtered hits:", len(hits))
for h in hits:
    print("  -", h.metadata["source_file"], "|", h.page_content[:45].replace(chr(10), " "))
```

---

### Qualitative notes on filtering

- **FAISS** — no built-in filter. Typical pattern: `similarity_search(QUERY, k=10)`
  then keep only hits whose `metadata` matches. (There is also a
  `filter`-style keyword in some wrappers, but it is not part of the core index.)
- **Qdrant** — native payload filters, e.g.
  `similarity_search(QUERY, k=3, filter_={"must": [{"key": "source_file", "match": {"value": "README.md"}}]})`.
- **LanceDB** — the 4th rotatable store in the card (`Chroma ↓ FAISS ↓ Qdrant ↓
  LanceDB`). It is a columnar embedded store with built-in filtering. Not
  installed here; if you want to try it:
  `pip install lancedb`, then
  `from langchain_community.vectorstores import LanceDB`.

---

## What you should notice

- **Index time**: an embedded store (Chroma) and a library index (FAISS) usually
  build faster than the server-style store, even in-memory — serialization and
  structure overhead differ.
- **Query speed**: all three answer a top-3 search in milliseconds on a small
  corpus; differences widen as the corpus grows, so re-run with more chunks.
- **Correctness is the same**: identical vectors → identical top-3. The store
  does not change *what* is retrieved, only *how fast and how durable* it is.
- **Persistence is the biggest real difference**: Chroma keeps a directory on
  disk; FAISS forgets everything unless you `save_local`; in-memory Qdrant is
  ephemeral until you attach a server.
- **Filtering varies by API**: Chroma dict filter is the simplest to write;
  FAISS makes you filter after retrieval; Qdrant's filter language is powerful
  but verbose.
- **How to choose**: for a small local project, Chroma (default) is the least
  effort; for huge in-memory similarity, FAISS shines; for teams/servers and
  advanced filtering, Qdrant (or LanceDB) is worth the setup.

---

## Exercises

1. **Scale it up.** Lower `chunk_size` to 200 (more, smaller chunks) and
   re-run the index-time loop. Which store degrades fastest? Why?
2. **Check recall under filtering.** Use the FAISS "filter after retrieval"
   pattern with `k=10` and count how many of the kept docs are actually from
   `README.md`. Compare with Chroma's `filter=` on the same question.
3. **Rotate in LanceDB.** Install `lancedb`, add a fourth builder with the same
   cached embedding wrapper, and extend the comparison table with an extra row.
4. **Persistence round-trip.** Restart the kernel, reload Chroma from
   `chroma_db_p07` with `Chroma(persist_directory=…, embedding_function=…)` and
   confirm the same top-3 still come back.
