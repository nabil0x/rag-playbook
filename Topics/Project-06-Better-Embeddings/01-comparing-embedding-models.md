> Source notebook: `NoteBooks/Project-06-Better-Embeddings/01-comparing-embedding-models.ipynb`


---

# Project 06 — Better Embeddings

**Goal:** Keep everything identical — only the embedding model changes.

```
Same pipeline, fixed small corpus, fixed query. Only this block rotates:

  Gemini  →  BGE  →  E5  →  (Nomic / Jina later)
```

Projects 01–05 changed loaders and splitters. This project changes **nothing
except the embedding model** — and measures what that single swap does to
retrieval quality, latency, and vector size.

Learn:

* Rotating one block while freezing the rest (the curriculum's core method)
* `GoogleGenerativeAIEmbeddings` vs `HuggingFaceBgeEmbeddings` vs
  `HuggingFaceEmbeddings`
* Comparing dims, latency, and retrieval accuracy side by side

---

## 0 · Setup — environment & keys

The Gemini embedding variant (and any LLM call) needs `GOOGLE_API_KEY` in
`.env`. The local BGE and E5 variants run fully offline. Optional installs for
THIS project: `faiss-cpu` and `sentence-transformers` (the local models).

---

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY", "")
if api_key:
    print(f"GOOGLE_API_KEY found: {api_key[:4]}…{api_key[-2:]} ({len(api_key)} chars)")
else:
    print("GOOGLE_API_KEY missing — the cloud Gemini variant will be skipped.")
```

---

```python
# Optional deps for THIS project only.
%pip install faiss-cpu
%pip install sentence-transformers
```

---

```python
import os
import time

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
```

---

## 1 · Load — a fixed, small corpus

Every variant must run against the *same* corpus or the comparison is
meaningless. We load a curated handful of the repo's own project cards from
`Topics/` — small enough to embed quickly on a CPU, big enough to be a real
retrieval test.

The corpus is chosen so each variant faces the same challenge: find the card
that matches a fixed query.

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
CARD_FILES = [
    "Project-01-Baseline-RAG/README.md",
    "Project-04-Markdown-Documentation-RAG/README.md",
    "Project-05-HTML-Documentation/README.md",
    "Project-06-Better-Embeddings/README.md",
    "Project-07-Compare-Vector-Databases/README.md",
]

docs = []
for rel in CARD_FILES:
    docs.append(TextLoader(os.path.join(TOPICS_DIR, rel)).load()[0])

print(f"Loaded {len(docs)} project cards")
for d in docs:
    print(" -", os.path.basename(d.metadata["source"]))
```

---

## 2 · Split

A plain `RecursiveCharacterTextSplitter` — deliberately boring. Splitting is
not what this project studies, so we keep it fixed and small: every variant
sees identical chunks.

---

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=40,
)
chunks = splitter.split_documents(docs)
print(f"{len(chunks)} chunks — the same fixed corpus for every variant")
```

---

## 3 · The rotation — one function, three embedders

The comparison lives in a single reusable function, `run_retrieval(embedder)`.
Everything else — corpus, chunks, query, store, retriever — is frozen inside
the function, so the *only* variable between runs is the embedding model.

Per variant we measure:

* **dims** — vector size from `embed_query` (drives storage + memory)
* **top-1 hit?** — is the expected card the #1 retrieved result?
* **latency ms** — query embed + similarity search, wall-clock
* **score** — raw distance of the top hit

Then we define the three variants: Gemini (cloud), BGE (local), E5 (local).
Gemini is skipped automatically if `GOOGLE_API_KEY` is unset.

---

```python
QUERY = "Which project keeps everything identical and only changes the embedding model?"
EXPECTED = "Project-06-Better-Embeddings"

def run_retrieval(embedder, query=QUERY, expected=EXPECTED):
    """Same pipeline, one embedder; returns dims / hit / latency / score."""
    db = FAISS.from_documents(chunks, embedder)
    dims = len(embedder.embed_query(query))
    t0 = time.perf_counter()
    top = db.similarity_search_with_score(query, k=1)
    latency_ms = (time.perf_counter() - t0) * 1000
    source = top[0][0].metadata.get("source", "")
    return dict(dims=dims, hit=expected in source,
                latency_ms=latency_ms, score=round(top[0][1], 3))
```

---

```python
variants = []

if os.getenv("GOOGLE_API_KEY"):
    variants.append(
        ("Gemini", GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview"))
    )
else:
    print("GOOGLE_API_KEY not set — skipping the cloud Gemini variant.")

variants += [
    ("BGE", HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en")),
    ("E5", HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")),
]

print("Variants:", [name for name, _ in variants])
```

---

```python
NOTES = {
    "Gemini": "cloud · 3072 dims · needs API key",
    "BGE": "local · bge-small-en",
    "E5": "local · e5-small-v2",
}

rows = []
for name, embedder in variants:
    result = run_retrieval(embedder)
    rows.append(
        (name, result["dims"], "yes" if result["hit"] else "no",
         f"{result['latency_ms']:.0f}", NOTES[name])
    )
    print(f"{name:6s} dims={result['dims']:>4}  top-1 hit={'yes' if result['hit'] else 'no':>3}"
          f"  latency={result['latency_ms']:7.0f} ms  score={result['score']}")
```

---

```python
print("| Embedding | dims | top-1 hit? | latency (ms) | notes |")
print("|---|---|---|---|---|")
for name, dims, hit, latency, note in rows:
    print(f"| {name} | {dims} | {hit} | {latency} | {note} |")
```

---

How to read the table:

* **dims** — 3072 (Gemini) vs 384 (BGE/E5). Fewer dims means ~8× less storage
  and memory for the same number of chunks, and often faster query embedding.
* **top-1 hit?** — with a tiny 5-card corpus, every variant should find the
  right card. The metric only becomes discriminating once the corpus grows.
* **latency (ms)** — Gemini pays a network round-trip per query; the local
  models pay only a CPU embed. First runs also include a one-time model
  download to the HuggingFace cache.
* **score** — distance units are not comparable *across* models; use it only to
  sanity-check retrieval within a model.

---

## 4 · Deep-dive — the three variants

**Gemini (`gemini-embedding-2-preview`)** — a cloud embedding model: 3072
dimensions, high quality, and a network call on every embed. Nothing to
download, but every query costs latency and an API key is mandatory.

**BGE (`BAAI/bge-small-en`)** — a local sentence-transformers model with 384
dimensions. Runs on CPU, no key, no network. BGE is a solid default for offline
documentation search (the Project 03 stack).

**E5 (`intfloat/e5-small-v2`)** — also local and 384-dimensional. E5 is trained
with `query:` / `passage:` prefixes for best results — the prompt-side tweak
that `HuggingFaceEmbeddings` can apply for you in production.

The rotation is designed to keep growing: the project card lists **Nomic**
(Matryoshka embeddings, truncatable to any dimension) and **Jina** as the next
two models you can drop into the same `run_retrieval` function.

---

## What you should notice

* Freezing every other block is what makes this comparison trustworthy: any
  difference you see is caused by the embedder alone.
* Cloud vs local is a real trade-off: Gemini's 3072 dims and network latency
  against BGE/E5's instant local 384-dim embeds.
* `run_retrieval` is the pattern you will reuse for every rotation project
  (07 vector DBs, 12 prompts): one function, one swappable block, one table.
* The first run of a local model downloads weights into the HuggingFace cache;
  later runs are fast.
* At this tiny corpus size every model hits the right card — retrieval quality
  separates them only as the corpus grows (try the Exercises).

---

## Exercises

1. Add a fourth variant — Nomic
   (`HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")`) or a
   Jina model — and re-run the table.
2. Measure *index build* time too: time `FAISS.from_documents` separately per
   variant. Which model is slowest to embed the corpus?
3. Use three queries (one per project card) and report hit@3 instead of top-1.
4. Re-run with `chunk_size=100`. Does top-1 hit change for any model? Why?
