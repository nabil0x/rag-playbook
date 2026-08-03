> Source notebook: `NoteBooks/Project-17-Modular-RAG-Framework/01-modular-rag-framework.ipynb`


---

# Project 17 — Modular RAG Framework

**Goal:** Implement your own architecture — pluggable interfaces for every block.

```
BaseLoader       → load    → list of documents
BaseSplitter     → split   → list of chunks
BaseEmbedding    → embed   → vectors
BaseVectorStore  → add/query → store & search vectors
BaseRetriever    → retrieve → top-k chunks
BaseLLM          → generate → answer
        └───────── RAGPipeline wires them together
```

This is the capstone of the framework-building thread. In Project 16 you built
every block with raw libraries; here you give those blocks **interfaces** and
assemble them into your own `RAGPipeline`. The result mirrors the repo's
`main.py` — the aspirational wiring shown in the README, where swapping
`WebLoader()` for `PDFLoader()` changes only one line.

---

### How to work through this notebook

The notebook has four movements:

1. **Define the interfaces** — six small abstract classes that say *what* every
   block must do, but not *how*.
2. **Implement concrete blocks** — thin classes that wrap the raw libraries you
   used in Project 16 (`requests`+`bs4`, a character splitter,
   `sentence-transformers`, `numpy`, `google-genai`).
3. **Assemble `RAGPipeline`** — one class that receives the six blocks and
   chains them: load → split → embed → store → retrieve → prompt → generate.
4. **Run it end to end** — one document, one question, and a demonstration that
   any block can be swapped without touching the rest.

Each section explains **what** the step does, **why** it matters for RAG, and
**what to expect** when you run it.

## 0 · Setup — environment, installs & imports

**WHAT:** Loads `.env` (for the Gemini API key), installs the two optional
packages from Project 16 (`sentence-transformers`, `google-genai`), and imports
the plain Python libraries the concrete blocks will wrap.

**WHY:** Same kitchen prep as every notebook. The imports are deliberately the
*raw* libraries — this framework is yours, built on the same pieces you mastered
in Project 16, not on another framework.

**WHAT TO EXPECT:** A masked key check (and a warning if `.env` is missing), a
pip message, then a silent imports cell.

---

```python
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
if key:
    print(f"GOOGLE_API_KEY is set (starts with {key[:4]}…)")
else:
    print("GOOGLE_API_KEY missing — create a .env file with GOOGLE_API_KEY=...")
```

---

```python
# Optional deps for THIS project only:
#   sentence-transformers → local MiniLM embeddings
#   google-genai          → the raw Gemini SDK
# %pip install sentence-transformers google-genai
```

---

```python
import os
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
```

---

## 1 · Define the interfaces

**WHAT:** Six abstract base classes, one per pipeline block. Each declares a
single method contract and raises `NotImplementedError` — an interface says
*what* a block must do, never *how*.

**WHY:** Interfaces are what make a pipeline pluggable. The rest of the pipeline
only ever talks to `BaseRetriever.retrieve(...)`; it never knows whether the
retriever is a naive top-k search, MMR, or hybrid — so swapping the retriever
means writing a new concrete class, not touching the pipeline. This is the same
idea as the repo's stub modules (`loaders/web.py`, `splitters/recursive.py`,
`vectordb/faiss.py`, …): they each declare a contract and raise
`NotImplementedError` until a project fills them in. `main.py` will eventually
wire concrete versions of these very classes.

**WHAT TO EXPECT:** Six class definitions, then a cell proving you cannot call an
interface directly — it raises `NotImplementedError`.

---

```python
class BaseLoader:
    """Contract: every loader returns a list of documents."""

    def load(self) -> list[dict]:
        raise NotImplementedError


class BaseSplitter:
    """Contract: every splitter turns documents into chunks."""

    def split(self, docs: list[dict]) -> list[dict]:
        raise NotImplementedError
```

---

```python
class BaseEmbedding:
    """Contract: every embedder turns text into vectors."""

    def embed(self, text: str):
        raise NotImplementedError


class BaseVectorStore:
    """Contract: stores embeddings and answers similarity queries."""

    def add(self, embeddings, chunks: list[dict]) -> None:
        raise NotImplementedError

    def query(self, query_vector, top_k: int = 3):
        raise NotImplementedError
```

---

```python
class BaseRetriever:
    """Contract: every retriever returns top-k chunks for a question."""

    def retrieve(self, question: str):
        raise NotImplementedError


class BaseLLM:
    """Contract: every LLM block turns a prompt string into an answer string."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError
```

---

```python
try:
    BaseLoader().load()
except NotImplementedError as e:
    print("BaseLoader.load() raises:", type(e).__name__)
```

---

## 2 · Implement concrete blocks

**WHAT:** One small class per interface, each wrapping a raw library from Project
16: `WebLoader` (requests + bs4), `CharSplitter` (sliding window),
`MiniLMLocalEmbedder` (sentence-transformers), `NumpyVectorStore` (a numpy
matrix), `SimilarityRetriever` (cosine top-k), and `GeminiClient` (google-genai).

**WHY:** A concrete class is an interface + an implementation. Every class here
subclasses its base, so each one can be dropped straight into the pipeline in
section 3. These are exactly the classes the repo's stubs will become — the
stubs raise `NotImplementedError`; these don't.

**WHAT TO EXPECT:** Six class definitions with no output. A document is just a
`dict` (`{"content": ..., "metadata": {...}}`) — our own lightweight
`Document`, chosen so the framework depends on nothing external.

---

### Block 1 · Load — WebLoader

**WHAT:** `requests.get` fetches a URL and `BeautifulSoup` keeps only headings
and paragraphs — the same loader you wrote in Project 16, now wrapped in a class
that honours the `BaseLoader.load()` contract.

**WHY:** Loading is where data enters the pipeline. Because the loader is
encapsulated, swapping a web page for a PDF later means writing one new
`BaseLoader` subclass — nothing else changes.

---

```python
class WebLoader(BaseLoader):
    def __init__(self, url: str):
        self.url = url

    def load(self) -> list[dict]:
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, "html.parser")
        text = "\n\n".join(
            t.get_text(" ", strip=True)
            for t in soup.find_all(["h1", "h2", "h3", "h4", "p"])
            if t.get_text(" ", strip=True)
        )
        return [{"content": text, "metadata": {"source": self.url}}]
```

---

### Block 2 · Split — CharSplitter

**WHAT:** A fixed-size window slides over the text, advancing by
`chunk_size - overlap` characters each step so consecutive chunks share a tail.

**WHY:** Retrieval needs small focused chunks. The `overlap` preserves sentences
that fall across a boundary, exactly like `chunk_overlap` did in earlier
projects — but here it is four lines you can tune yourself.

---

```python
class CharSplitter(BaseSplitter):
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, docs: list[dict]) -> list[dict]:
        chunks = []
        for doc in docs:
            text = doc["content"]
            step = self.chunk_size - self.overlap
            for start in range(0, max(len(text), 1), step):
                chunk = text[start:start + self.chunk_size]
                if chunk:
                    chunks.append({"content": chunk, "metadata": doc["metadata"]})
        return chunks
```

---

### Block 3 · Embed — MiniLMLocalEmbedder

**WHAT:** Wraps `SentenceTransformer("all-MiniLM-L6-v2")`; `embed` handles one
string, `embed_many` a whole list (the model encodes a batch far faster than one
at a time).

**WHY:** Embeddings turn text into the numbers that similarity search compares.
`embed_many` is a concrete-class convenience — the interface only requires
`embed`, and extra methods never hurt.

---

```python
class MiniLMLocalEmbedder(BaseEmbedding):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts)
```

---

### Block 4 · Store — NumpyVectorStore

**WHAT:** `add` stacks embeddings into one matrix and remembers the chunks;
`query` computes cosine similarity with a matrix–vector dot product and returns
the top-k `(chunk, score)` pairs.

**WHY:** The matrix *is* the vector store. This is the same math as
`Chroma.similarity_search` — a real database only adds persistence and
approximate search for millions of vectors.

---

```python
class NumpyVectorStore(BaseVectorStore):
    def __init__(self):
        self._matrix = None
        self._chunks = []

    def add(self, embeddings, chunks: list[dict]) -> None:
        self._matrix = np.asarray(embeddings)
        self._chunks = chunks

    def query(self, query_vector, top_k: int = 3):
        scores = self._matrix @ query_vector
        scores = scores / (
            np.linalg.norm(self._matrix, axis=1) * np.linalg.norm(query_vector)
        )
        order = np.argsort(scores)[::-1][:top_k]
        return [(self._chunks[i], float(scores[i])) for i in order]
```

---

### Block 5 · Retrieve — SimilarityRetriever

**WHAT:** Embeds the question and asks the injected vector store for the top-k
most similar chunks. Note that the retriever *depends on two other blocks* — the
store and the embedder — handed to it through `__init__`.

**WHY:** This is dependency injection in miniature: `SimilarityRetriever` knows
only the `BaseVectorStore` / `BaseEmbedding` contracts, so it will work with any
implementation. An MMR retriever (Exercise 2) can reuse the exact same store.

---

```python
class SimilarityRetriever(BaseRetriever):
    def __init__(self, vector_store, embedder, top_k: int = 3):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, question: str):
        qvec = self.embedder.embed(question)
        return self.vector_store.query(qvec, top_k=self.top_k)
```

---

### Block 6 · Answer — GeminiClient

**WHAT:** Wraps the raw `google-genai` SDK: `generate(prompt)` calls
`client.models.generate_content(...)` and returns `response.text`.

**WHY:** This is the block that turns retrieved evidence into a grounded
answer — and, being behind the `BaseLLM` contract, it can be swapped for an
OpenAI client or a local model without touching the pipeline.

---

```python
class GeminiClient(BaseLLM):
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )
        return response.text
```

---

## 3 · Assemble RAGPipeline

**WHAT:** One class that owns the six blocks and chains them. `index(source)`
runs load → split → embed → store (the one-time ingestion); `ask(question,
source=None)` runs retrieve → prompt → generate, and also runs `index` first if
a `source` is given, so a single `ask(question, source=url)` can drive the whole
chain.

**WHY:** This is the wiring the repo's `main.py` dreams of:

```python
pipeline = RAGPipeline(
    loader=PDFLoader(), splitter=SemanticSplitter(), embedder=BGEEmbedding(),
    vector_store=FAISSVectorStore(), retriever=MMRRetriever(db), llm=GeminiLLM(),
)
pipeline.ask("What is Task Decomposition?")
```

The pipeline never names a concrete class — it receives them via `__init__`
(dependency injection). That one sentence is why every block is swappable.

**WHAT TO EXPECT:** A prompt template string, then the `RAGPipeline` class, then
a tiny proof that the class wires blocks without knowing what they are.

---

```python
PROMPT_TEMPLATE = """
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
```

---

```python
class RAGPipeline:
    def __init__(self, loader, splitter, embedder, vector_store, retriever, llm):
        self.loader, self.splitter, self.embedder = loader, splitter, embedder
        self.vector_store, self.retriever, self.llm = vector_store, retriever, llm

    def index(self, source):
        """Ingest: load -> split -> embed -> store."""
        docs = self.loader.load()
        chunks = self.splitter.split(docs)
        vectors = self.embedder.embed_many([c["content"] for c in chunks])
        self.vector_store.add(vectors, chunks)
        print(f"Indexed {len(chunks)} chunks from {len(docs)} document(s)")

    def ask(self, question, source=None):
        """Answer: (index if a source is given) -> retrieve -> prompt -> generate."""
        if source is not None:
            self.index(source)
        hits = self.retriever.retrieve(question)
        context = "\n\n".join(c["content"] for c, _ in hits)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        return self.llm.generate(prompt)
```

---

```python
class FakeLoader(BaseLoader):
    def load(self):
        return [{"content": "Hello from a fake loader.", "metadata": {}}]


pipeline = RAGPipeline(
    loader=FakeLoader(),
    splitter=CharSplitter(),
    embedder=None,  # not needed for this smoke test
    vector_store=None,
    retriever=None,
    llm=None,
)
print("RAGPipeline wired OK with a fake loader")
```

---

## 4 · Run it end to end — one document, one query

**WHAT:** We wire the six *real* concrete blocks into a `RAGPipeline` (exactly
the `main.py` style), index one Project Gutenberg book, and ask it a question.

**WHY:** End-to-end is the proof the architecture works: the same pipeline that
accepted a fake loader accepts the real one, because both implement
`BaseLoader`. First `ask(question, source=url)` drives the *entire* chain
load → split → embed → store → retrieve → prompt → generate; later `ask`
calls skip ingestion and reuse the store.

**WHAT TO EXPECT:** On the first `ask`, a line showing how many chunks were
indexed, then a grounded answer about the book's moral; on the second `ask` no
indexing line (the store is reused). The retrieval cell shows the top-3 chunks
and their cosine scores.

---

```python
URL = "https://www.gutenberg.org/cache/epub/79247/pg79247-images.html"

store = NumpyVectorStore()
embedder = MiniLMLocalEmbedder()
pipeline = RAGPipeline(
    loader=WebLoader(URL),
    splitter=CharSplitter(chunk_size=1000, overlap=200),
    embedder=embedder,
    vector_store=store,
    retriever=SimilarityRetriever(store, embedder, top_k=3),
    llm=GeminiClient(),
)
print("pipeline built — ready to ask")
```

---

```python
QUESTION = "What is the moral of the story?"

answer = pipeline.ask(QUESTION, source=URL)  # runs the whole chain in one call
print(answer)
```

---

```python
hits = pipeline.retriever.retrieve(QUESTION)
print(f"{len(hits)} chunks retrieved:")
for i, (chunk, score) in enumerate(hits, 1):
    print(f"{i}. score={score:.3f} :: {chunk['content'][:90]}...")
```

---

```python
answer2 = pipeline.ask("Who is the narrator's partner?")
print(answer2)
```

---

### Try it yourself

Change `QUESTION`, change `top_k` in the retriever, or swap in a second
document with a second `pipeline.index(source)`. The cell below is your sandbox.

---

```python
# your scratch cell — reuse pipeline, store, embedder, ...
```

---

## What you should notice

- **Swap any block without touching the rest.** The pipeline code only ever
  references `loader.load()`, `splitter.split()`, … — never a concrete class.
  Replacing `WebLoader` with a PDF loader means writing one subclass, then one
  line of wiring.
- **Dependency injection is the whole trick.** Blocks receive their dependencies
  through `__init__` instead of constructing them. That single decision is what
  makes every block replaceable — it is the entire architecture of this project.
- **Interfaces are contracts, classes are implementations.** `BaseRetriever`
  says *what*; `SimilarityRetriever` says *how*. The repo's stub modules
  (`loaders/`, `splitters/`, `embeddings/`, `vectordb/`, `retrieval/`, `llms/`)
  will become exactly these concrete classes.
- **The framework adds structure, not functionality.** Every block here wraps a
  raw library from Project 16 — the RAG math is unchanged; only the *shape* of
  the code changed. That is what a framework is.
- **Separate ingestion from query.** `index()` (load→split→embed→store) runs
  once per document; `ask()` runs per question. Keeping them separate is why RAG
  can serve many questions cheaply after one indexing pass.
- **`main.py` is the endgame.** Wiring `RAGPipeline(PDFLoader(), SemanticSplitter(),
  BGEEmbedding(), FAISSVectorStore(), MMRRetriever(), GeminiLLM())` is now one
  keyword swap away from the classes you have just written.

---

## Exercises

1. **Add a `PDFLoader` block.** Write `class PDFLoader(BaseLoader)` that opens a
   PDF with `pypdf` (e.g. `PdfReader(open(path, "rb"))`, joining
   `page.extract_text()` per page into one document dict) and swap it into the
   pipeline — *only* the `loader=` line changes. Point it at a PDF in
   `Data/`.
2. **Add MMR to the retriever.** Write `MMRRetriever(BaseRetriever)` that
   re-ranks the top results: each pick is the most similar remaining chunk
   *minus* a penalty for being similar to chunks already chosen (diversity).
   Compare its top-3 with `SimilarityRetriever`'s on a single-topic question.
3. **Persist the vector store.** Give `NumpyVectorStore` `save(path)` and
   `load(path)` methods using `np.savez` (matrix + chunk texts/metadata). Restart
   the kernel, load the store, and re-run `pipeline.ask(...)` without re-embedding
   — the first sign of a real production vector DB.
