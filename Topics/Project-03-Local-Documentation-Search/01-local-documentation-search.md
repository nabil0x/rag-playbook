> Source notebook: `NoteBooks/Projects/Project-03-Local-Documentation-Search/01-local-documentation-search.ipynb`


---

# Project 03 — Local Documentation Search

**Goal:** Index a whole directory tree offline — no cloud embedding API.

```
Loader      : DirectoryLoader
Splitter    : RecursiveCharacterTextSplitter
Embedding   : BGE (BAAI/bge-small-en, local)
Vector DB   : FAISS (in-memory)
Retriever   : Similarity (Top-K)
Prompt      : Basic Context + Question
LLM         : Gemini
```

This is the **offline** project: loading, splitting, embedding and vector
search all run on your machine. The *only* online step is the final LLM call
to Gemini — and you can swap that block too if you ever want a fully offline
pipeline.

---

## 0 · Setup — environment, keys & optional installs

Three things to know before running:

* **`GOOGLE_API_KEY` is still needed** — the LLM step calls Gemini. Copy
  `.env.example` to `.env` at the repo root and paste your key. The check below
  only prints a masked preview (`key[:4]…`), never the full key.
* **`sentence-transformers`** runs the local BGE embeddings — install cell below.
* **`faiss-cpu`** is the local vector index — same install cell.

The embedding and search steps need **no API key and no internet** after the
one-time model download.

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
#   sentence-transformers → local BGE embeddings
#   faiss-cpu             → local FAISS index
#   langchain-huggingface → HuggingFaceBgeEmbeddings (also in requirements.txt)
%pip install sentence-transformers faiss-cpu langchain-huggingface
```

---

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
```

---

## 1 · Load — index a whole directory tree

`DirectoryLoader` turns an *entire folder* into Documents: give it a path and a
`glob` pattern, and it walks the tree, loading every matching file with a
per-format loader (`TextLoader` for `.md` / `.txt`). That is what makes a
"local documentation search": point it at your docs folder and every file
becomes searchable.

We index `Data/local-docs/` — a tiny sample knowledge base with a
`README.md` and two topic pages under `docs/`. Add your own files there later
to grow the index.

---

```python
DATA_DIR = "Data/local-docs"

loader = DirectoryLoader(
    DATA_DIR,
    glob="**/*.md",             # recursive — picks up docs/*.md too
    loader_cls=TextLoader,      # reads plain text / markdown files
    loader_kwargs={"encoding": "utf-8"},
)

docs = loader.load()
print(f"Loaded {len(docs)} files")
```

---

```python
for d in docs:
    first_line = d.page_content.splitlines()[0][:80]
    print(f"• {d.metadata['source']}")
    print(f"    {first_line}")
```

---

## 2 · Split — RecursiveCharacterTextSplitter

The files here are small (a few hundred words each), so we use a tighter budget
than Projects 01–02: `chunk_size=500`, `chunk_overlap=50`. A chunk is the
retrieval unit — each one gets embedded and later compared to the query. Keep it
small enough to be *about one thing*, yet large enough to contain an answer.

---

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)

print(f"Split {len(docs)} files into {len(chunks)} chunks")
```

---

```python
print("Sample chunk:")
print(chunks[0].page_content[:200])
```

---

## 3 · Embed — BGE, fully local

Projects 01–02 embedded with the **Gemini API**; here we embed with **BGE** — a
small open-source model run locally by `sentence-transformers`
(`HuggingFaceBgeEmbeddings`). No API key, no network per query. BGE also
recommends **normalizing** the vectors, which makes cosine similarity
well-behaved (`normalize_embeddings=True`).

First use downloads the model (~130 MB) into `~/.cache/huggingface`; every later
run uses the cached copy. `BAAI/bge-small-en` outputs **384**-dimension vectors
— versus Gemini's 3072 — small enough to embed thousands of docs on a laptop.

---

```python
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en",
    encode_kwargs={"normalize_embeddings": True},
)
print("BGE embeddings ready (model cached after first download)")
```

---

```python
sample_vec = embeddings.embed_query("vector search over local documentation")
print(f"Embedding dimension: {len(sample_vec)}  (bge-small-en → 384)")
print("First 5 values:", [round(v, 4) for v in sample_vec[:5]])
```

---

## 4 · Store — FAISS

**FAISS** (Facebook AI Similarity Search) is a library for fast vector search.
It lives **in memory** — no server, no database process — which is exactly right
for a local docs index. `FAISS.from_documents` embeds every chunk with our local
BGE model and builds the index in one step.

Persistence is explicit: `vector_store.save_local("faiss_index/")` writes the
index to disk, and `FAISS.load_local("faiss_index/", embeddings)` reads it back
(needs `allow_dangerous_deserialization=True`) — so you can build once and reuse.

---

```python
vector_store = FAISS.from_documents(chunks, embeddings)

print(f"Indexed {vector_store.index.ntotal} vectors in memory")
```

---

## 5 · Retrieve — similarity search, no network

Same retriever concept as before: embed the query with the *same* local model,
then ask FAISS for the `k` nearest chunks. Because everything is in-process,
search is instant and fully offline. `similarity_search_with_score` reports the
distance per hit — **lower is closer** (more relevant).

---

```python
query = "How do I run similarity search over my local documentation?"

hits = vector_store.similarity_search_with_score(query, k=3)

for doc, score in hits:
    src = doc.metadata["source"]
    print(f"[{score:.4f}] {src}: {doc.page_content[:80]}")
```

---

## 6 · Prompt — context + question

Identical template to the previous projects: fuse the retrieved chunks into
`{context}`, add the `{question}`, and instruct the LLM to answer only from the
context. The prompt does not care whether the chunks came from a web page, a
PDF or a markdown tree — the "swap one block" design at work again.

---

```python
context = "\n\n".join(doc.page_content for doc, _ in hits)
print(f"Context: {len(context)} characters from {len(hits)} chunks")
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

## 7 · Answer — Gemini LLM (the one online step)

Only the generation step goes online: `ChatGoogleGenerativeAI` sends the prompt
to Gemini and returns an answer. The entire knowledge-base side — loading,
splitting, BGE embedding, FAISS search — ran locally. For a fully offline
pipeline, swap this block for a local model (e.g. via Ollama); the prompt and
the retrieved context stay exactly the same.

---

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = prompt.invoke({"context": context, "question": query})
response = llm.invoke(messages)

print(response.content)
```

---

## 8 · Try it yourself

Add your own `.md` / `.txt` files to `Data/local-docs/`, re-run Load →
Store (no API cost — embedding is local), then ask new questions. Or change `k`
in retrieval to widen or narrow the context given to the LLM.

---

```python
query2 = "Which embedding model does this knowledge base recommend and why?"

hits2 = vector_store.similarity_search(query2, k=2)
context2 = "\n\n".join(d.page_content for d in hits2)

answer2 = llm.invoke(prompt.format_messages(context=context2, question=query2))
print(answer2.content)
```

---

```python
query3 = "What does the FAISS section say about persisting the index?"

hits3 = vector_store.similarity_search(query3, k=2)
context3 = "\n\n".join(d.page_content for d in hits3)

answer3 = llm.invoke(prompt.format_messages(context=context3, question=query3))
print(answer3.content)
```

---

## What you should notice

* **Truly offline search.** Embedding (BGE) and retrieval (FAISS) ran with no
  API key and no network — the only online call was the LLM. Everything up to
  Section 7 works even unplugged.
* **One-time model download.** The first BGE embed fetched ~130 MB into
  `~/.cache/huggingface`; later runs are instant and free.
* **384 vs 3072 dimensions.** `bge-small-en` vectors are ~8× smaller than
  Gemini's — faster to compute and search, at a small quality trade-off.
* **FAISS is an in-memory index.** No server process; persistence is manual via
  `save_local` / `load_local`.
* **DirectoryLoader scales to whole trees.** One `glob` indexed every markdown
  file under `Data/local-docs/` — point it at a bigger folder and you
  have a local docs search.

---

## Exercises

1. Add 2–3 of your own `.md` files under `Data/local-docs/` (or switch
   `glob` to `"**/*.txt"`) and re-index. No API cost — only the final LLM answer
   needs a key.
2. Build the index once with `vector_store.save_local("faiss_index/")`, then in
   a fresh cell load it back with `FAISS.load_local` and query without
   re-embedding.
3. Compare: answer the same question through FAISS + BGE (this project) and
   through Chroma + Gemini (Project 02). Which chunks are retrieved? Which
   answer is better, and why?
