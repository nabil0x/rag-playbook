> Source notebook: `NoteBooks/Projects/Project-16-Build-Without-LangChain/01-rag-without-langchain.ipynb`


---

# Project 16 — Build Without LangChain

**Goal:** Replace every block of the RAG pipeline with raw libraries — no framework.

```
requests + BeautifulSoup → Load
hand-rolled chunker      → Split
SentenceTransformer      → Embed
numpy cosine similarity  → Store & Search
hand-built template      → Prompt
google-genai SDK         → Answer
```

Every project before this one leaned on LangChain for at least one block. This
project removes **all** of it: you call `requests`, `BeautifulSoup`,
`sentence-transformers`, `numpy` and the `google-genai` SDK directly. There is
not a single `langchain` import in this notebook — proof that a RAG pipeline is
just six libraries stitched together, and that LangChain is a thin convenience
wrapper on top of them.

---

### How to work through this notebook

This notebook rebuilds the **baseline pipeline** — load → chunk → embed → store →
retrieve → prompt → answer — from Project 01, but block by block in raw Python.
Each section explains **what** the step does, **why** it matters for RAG, and
**what to expect** when you run it.

By the end you will have implemented every block by hand — which is exactly what
a framework hides from you. When you later use LangChain's
`RecursiveCharacterTextSplitter` or `Chroma.similarity_search`, you will know
what they do under the hood, because you wrote the simpler version first.

## 0 · Setup — environment, installs & imports

**WHAT:** Loads `.env` (so the Gemini API key is available), installs the two
optional packages this project needs (`sentence-transformers` for local
embeddings, `google-genai` for the raw LLM SDK), and imports the plain Python
libraries.

**WHY:** One place for keys and imports, so the pipeline cells below stay short.
Look at what is *not* imported: no `langchain`, no `langchain_community`, no
`langchain_core` — only `os`, `requests`, `bs4`, `numpy` and `dotenv`.

**WHAT TO EXPECT:** The first cell prints `True` (or `False` if `.env` is
missing) and a masked key check; the install cell prints a pip message; the
imports cell produces no output.

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
#   sentence-transformers → local MiniLM embeddings (no cloud call)
#   google-genai          → the raw Gemini SDK
# %pip install sentence-transformers google-genai
```

---

```python
import os
import requests
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
```

---

## 1 · Load — requests + BeautifulSoup

**WHAT:** `requests.get(url)` fetches the raw HTML of a Project Gutenberg book,
and a small `extract_text()` helper uses BeautifulSoup to keep only the
headings and paragraphs — the main content, with navigation dropped.

**WHY:** This is the same document the Project 01 baseline loaded with
`WebBaseLoader` — but `WebBaseLoader` was itself just a wrapper around exactly
these two libraries. Writing the loader yourself shows what it was hiding.

**WHAT TO EXPECT:** A `raw_text` string (tens of thousands of characters) whose
first paragraph is the book's opening line.

---

```python
URL = "https://www.gutenberg.org/cache/epub/79247/pg79247-images.html"


def extract_text(html: str) -> str:
    """Keep headings and paragraphs from a raw HTML page, joined by blank lines."""
    soup = BeautifulSoup(html, "html.parser")
    parts = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p"]):
        text = tag.get_text(" ", strip=True)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


html = requests.get(URL).text
raw_text = extract_text(html)
```

---

```python
print(len(raw_text), "characters loaded")
print(raw_text[:300])
```

---

## 2 · Split by hand — paragraphs → ~1000-char chunks

**WHAT:** Three tiny functions you write yourself, each doing one job:
`split_paragraphs` cuts the text on blank lines; `merge_paragraphs` packs the
paragraphs into chunks of roughly `chunk_size` characters; `add_overlap`
re-introduces the tail of each chunk into the start of the next one.

**WHY:** Retrieval works on small focused chunks, not whole books. This is what
`RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` did for you
in Project 01 — here you can see and change every decision.

**WHAT TO EXPECT:** A few hundred paragraphs at first, then a smaller number of
~1000-character chunks, and finally the same chunk count with each chunk slightly
longer because it now carries the previous chunk's tail.

---

```python
def split_paragraphs(text: str) -> list[str]:
    """Split raw text into non-empty paragraphs on blank lines."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


paragraphs = split_paragraphs(raw_text)
print(len(paragraphs), "paragraphs")
print("first:", paragraphs[0][:100])
```

---

```python
def merge_paragraphs(paragraphs: list[str], chunk_size: int = 1000) -> list[str]:
    """Pack paragraphs into chunks, breaking only between paragraphs."""
    chunks, current, size = [], [], 0
    for para in paragraphs:
        if current and size + len(para) + 2 > chunk_size:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks
```

---

```python
chunks = merge_paragraphs(paragraphs)
print(len(chunks), "chunks")
print("sizes:", sorted({len(c) for c in chunks})[:5], "...")
```

---

```python
def add_overlap(base_chunks: list[str], overlap: int = 200) -> list[str]:
    """Prepend the previous chunk's last `overlap` characters to each chunk."""
    out = []
    for i, chunk in enumerate(base_chunks):
        if i > 0:
            chunk = base_chunks[i - 1][-overlap:] + "\n\n" + chunk
        out.append(chunk)
    return out


chunks = add_overlap(chunks)
print(len(chunks), "chunks after overlap")
print(chunks[2][:200])
```

---

## 3 · Embed — SentenceTransformer, directly

**WHAT:** `SentenceTransformer("all-MiniLM-L6-v2")` loads a small local model,
and `model.encode(text)` turns any string into a 384-dimensional vector. A
`embed()` helper keeps later cells short.

**WHY:** Semantic search compares *vectors*, not text. Similar sentences land
near each other in this 384-dimensional space, so "moral of the story" sits
close to chunks that actually talk about the moral. Everything is computed on
your machine — no API call.

**WHAT TO EXPECT:** A model object (first run downloads it, ~80 MB), then a
vector of shape `(384,)` with the first few numbers printed.

---

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
```

---

```python
def embed(text: str) -> np.ndarray:
    """Embed one string into a 384-dim vector using the local model."""
    return model.encode(text)


vec = embed("What is the moral of the story?")
print("vector shape:", vec.shape)
print("first 5 values:", vec[:5])
```

---

## 4 · Store & Search — numpy cosine similarity

**WHAT:** `np.vstack` stacks every chunk's vector into one `(n_chunks, 384)`
matrix — that matrix *is* your vector store. To search, `cosine_similarity`
computes the dot product scaled by the vector lengths, and `search` sorts the
scores to return the top-k chunks. No vector database involved.

**WHY:** Cosine similarity is the math underneath every vector store you have
used (Chroma, FAISS, Qdrant). Writing `np.dot` and `np.linalg.norm` yourself
demystifies what `similarity_search(k=3)` did for you in Project 01.

**WHAT TO EXPECT:** A matrix of shape `(n_chunks, 384)`, then the top-3 chunks
for the query with their similarity scores — the same evidence the baseline
retriever produced.

---

```python
matrix = np.vstack([embed(chunk) for chunk in chunks])
print("matrix shape:", matrix.shape)
```

---

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, in [-1, 1]."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search(query_text: str, matrix: np.ndarray, chunks: list[str], k: int = 3):
    """Return the top-k (chunk, score) pairs for a query."""
    qvec = embed(query_text)
    scores = [cosine_similarity(qvec, row) for row in matrix]
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i], scores[i]) for i in top]
```

---

```python
query = "What is the moral of the story?"
hits = search(query, matrix, chunks, k=3)
for i, (chunk, score) in enumerate(hits, 1):
    print(f"{i}. score={score:.3f} :: {chunk[:90]}...")
```

---

```python
context = "\n\n".join(chunk for chunk, _ in hits)
print(len(context), "chars of context")
```

---

## 5 · Prompt by hand — a plain template string

**WHAT:** A plain string with `{context}` and `{question}` placeholders, filled
in with `.format()`. No `ChatPromptTemplate`, no `MessagesPlaceholder` — the
template *is* the prompt.

**WHY:** The wording ("answer using ONLY the provided context", with an explicit
"I don't know" fallback) is what turns a generic LLM into a grounded one. This
is the same contract the baseline used; here it is just a string you can read
and edit in one line.

**WHAT TO EXPECT:** The rendered prompt printed — the instructions, the retrieved
chunks, and the question, exactly as the model will see them.

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

prompt = PROMPT_TEMPLATE.format(context=context, question=query)
```

---

```python
print(prompt[:500])
```

---

## 6 · Answer — the google-genai SDK, raw

**WHAT:** `genai.Client(api_key=...)` builds the raw Gemini client and
`client.models.generate_content(model, contents=prompt)` sends the rendered
prompt. `response.text` is the answer. No LangChain wrapper — this is the actual
HTTP call under `ChatGoogleGenerativeAI`.

**WHY:** This is the final block: the model reads the retrieved evidence plus
the question and produces a grounded answer. Seeing the raw SDK makes clear that
a LangChain chat model was just this client in a prettier costume.

**WHAT TO EXPECT:** A natural-language answer to "What is the moral of the
story?" grounded in the retrieved chunks — the same kind of answer the baseline
produced.

---

```python
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
```

---

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)
print(response.text)
```

---

## 7 · Try it yourself — your sandbox

The whole pipeline, minus the framework. Change `query` below, change `k` in
`search`, or rewrite `PROMPT_TEMPLATE`, then re-run the search → prompt → answer
cells. The next cell is yours to edit; the empty cell after it is your scratch
space.

---

```python
query2 = "Who is the narrator's partner?"
hits2 = search(query2, matrix, chunks, k=3)
context2 = "\n\n".join(chunk for chunk, _ in hits2)
prompt2 = PROMPT_TEMPLATE.format(context=context2, question=query2)
answer2 = client.models.generate_content(model="gemini-2.5-flash", contents=prompt2)
print(answer2.text)
```

---

```python
# your scratch cell — reuse any of the helpers defined above
```

---

## What you should notice

- **LangChain is a thin wrapper.** Every block you used through 15 projects is
  six plain libraries: `requests`, `bs4`, a text splitter, an embedder, `numpy`,
  and an HTTP client. Nothing here is magic.
- **You now understand every block.** `RecursiveCharacterTextSplitter` =
  paragraph merge + overlap. `Chroma.similarity_search` = cosine similarity +
  argsort. `ChatGoogleGenerativeAI` = `genai.Client(...).generate_content(...)`.
- **A RAG pipeline needs no framework at all.** The whole thing fits in a handful
  of small functions — but the trade-off is that *you* now own edge cases
  (long paragraphs, zero vectors, token budgets) that the framework handled.
- **`all-MiniLM-L6-v2` is local and free.** No API cost, no latency from the
  network, 384-dim vectors — at the price of lower quality than a large cloud
  embedding model.
- **`np.dot` on a matrix is fast enough here.** With a few hundred chunks a
  brute-force scan is instant; real vector databases exist because that stops
  being true at millions of vectors.
- **The prompt did the grounding, not the model.** The answer quality comes from
  the "answer ONLY from context" contract plus good retrieval — the LLM is
  interchangeable.

---

## Exercises

1. **Tune the chunker.** Change `chunk_size` to 500 and `overlap` to 100, rebuild
   the matrix, and re-run retrieval. How does the answer change when the evidence
   is finer-grained?
2. **Ask an out-of-context question.** Query something not in the book (e.g.
   "What is the capital of France?") — does the "I don't know" fallback in the
   template hold up?
3. **Swap the embedder to a cloud model.** Point the `embed` helper at
   `google-genai`'s `models.embed_content` and re-run the same search. Compare
   the retrieved chunks (and the install/API cost) with the local MiniLM run.
