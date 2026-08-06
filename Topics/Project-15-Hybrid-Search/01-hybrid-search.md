> Source notebook: `NoteBooks/Projects/Project-15-Hybrid-Search/01-hybrid-search.ipynb`


---

# Project 15 — Hybrid Search

**Goal:** Combine keyword and semantic search.

```
Loader      : Inline synthetic corpus (RAG glossary)
Splitter    : RecursiveCharacterTextSplitter (passages already small)
Embedding   : Gemini (dense side)
Vector DB   : Chroma (dense) + BM25 (keyword)
Retriever   : EnsembleRetriever — reciprocal rank fusion
Prompt      : Basic Context + Question
LLM         : Gemini
```

Dense retrieval (embeddings) is great at **meaning** but can miss exact terms;
BM25 is great at **exact terms** but blind to meaning. Real search engines fuse
both. This project builds a **dense retriever** (Chroma similarity) and a
**keyword retriever** (`BM25Retriever`), then combines their rankings with
`EnsembleRetriever` — a weighted **reciprocal rank fusion** of the two lists.

Learn:

* **BM25** — sparse keyword scoring (term frequency + inverse document frequency)
* **Dense retrieval** — vector similarity on embeddings
* **Reciprocal rank fusion** — merging ranked lists into one

---

### How to work through this notebook

This notebook runs the **full pipeline** but the retriever block is the story:
Sections 1–4 build one small corpus and index it **twice** — once as vectors in
Chroma, once as a BM25 keyword index. Section 5 runs the *same query* through
BM25, through dense, and through the fused `EnsembleRetriever`, printing the
three lists side by side so you can see which retriever catches what. Sections
6–7 answer from the fused context.

---

## 0 · Setup — environment, keys & one install

Two online pieces — Gemini embeddings and the Gemini LLM — need a
`GOOGLE_API_KEY` (copy `.env.example` to `.env`; the check below only prints a
masked preview `key[:4]…`).

One package is needed **for this project only**:

* `rank-bm25` — a fast implementation of the BM25 ranking function used by
  `BM25Retriever` from `langchain_community.retrievers`. The install cell is
  idempotent, so re-running it is harmless.

Also note: `EnsembleRetriever` historically lives in `langchain.retrievers`;
newer `langchain` (1.x) moved it to `langchain_classic.retrievers`. The import
cell tries both.

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
#   rank-bm25 → BM25Retriever scores documents with BM25Okapi
%pip install rank-bm25
```

---

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# EnsembleRetriever: `langchain.retrievers` on langchain ≤ 1.0, moved to
# `langchain_classic.retrievers` on newer langchain 1.x.
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
```

---

## 1 · Load — a glossary built for the keyword-vs-semantic showdown

The corpus is ten short, factual passages about RAG internals. Several contain
**rare, technical tokens** (`HNSW`, `quantization`, `TF-IDF`, `reciprocal rank
fusion`). That is deliberate: an exact-token matcher (BM25) locks onto these
tokens, while an embedding model must still find the passage *semantically*.
The setup is rigged so the two retrievers can disagree — which is what makes
the fusion visible.

---

```python
docs = [
    Document(page_content="HNSW is a graph-based index that vector databases use for approximate nearest neighbour search."),
    Document(page_content="BM25 is a keyword ranking function that scores documents by term frequency and inverse document frequency."),
    Document(page_content="Reciprocal rank fusion combines ranked lists by summing the reciprocal of each document's rank."),
    Document(page_content="Quantization compresses embedding vectors into fewer bits, cutting memory use at some accuracy cost."),
    Document(page_content="A vector database stores dense embeddings and searches them by cosine similarity."),
    Document(page_content="Tokenization splits raw text into tokens before an embedding or language model processes it."),
    Document(page_content="The top-k parameter decides how many chunks are handed to the prompt as context."),
    Document(page_content="Chunk overlap preserves context that would otherwise be lost at chunk boundaries."),
    Document(page_content="TF-IDF weighs terms that appear often in a document but rarely across the whole corpus."),
    Document(page_content="Semantic search matches meaning; keyword search matches exact terms."),
]
print(f"Corpus: {len(docs)} Documents")
```

---

```python
for i, d in enumerate(docs):
    print(f"[{i}] {d.page_content}")
```

---

## 2 · Split — a formality here (passages are already small)

Each passage is already about one idea, so `RecursiveCharacterTextSplitter`
with a large budget keeps them intact. We split anyway to stay consistent with
the full-pipeline template — and so both indexes below are built from the same
`chunks`.

---

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

print(f"{len(docs)} passages → {len(chunks)} chunks")
```

---

## 3 · Embed — text → vectors (the dense side)

Same embedding model as the previous projects: `GoogleGenerativeAIEmbeddings`
returns a 3072-dimension vector per chunk. These vectors power the **dense**
side of the hybrid retriever. The keyword side (BM25) needs none of this — it
works on raw text.

---

```python
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

sample_vec = embeddings.embed_query("hybrid search and rank fusion")
print(f"Embedding dimension: {len(sample_vec)}")
```

---

## 4 · Store — a dense index AND a keyword index

Two very different indexes over the *same* chunks:

* **Chroma** — embeds every chunk and stores the vectors. Search is by distance
  in vector space (dense).
* **BM25Retriever** — tokenizes the same chunks into a sparse term-count index.
  Search is by exact-term scoring (keyword). No embeddings involved.

Both expose the same retriever interface (`invoke(query) → list[Document]`),
which is exactly what the fusion step in Section 5 needs.

---

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="project15_hybrid",
)
print(f"Chroma dense index built ({len(chunks)} vectors)")
```

---

```python
bm25_retriever = BM25Retriever.from_documents(chunks, k=4)
print("BM25 keyword index built")
```

---

## 5 · Retrieve — BM25 vs dense vs fused

The query is chosen to favour the keyword side: it contains **rare technical
tokens** (`HNSW`) that appear verbatim in only one passage. Predictions before
you run it:

* **BM25** should rank the exact-term passage first — it literally counts tokens.
* **Dense** may rank a *related* passage higher, because embeddings stretch for
  meaning when an exact token appears nowhere else.
* **Fused** should put the best of both at the top.

`EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever],
weights=[0.5, 0.5])` implements **reciprocal rank fusion**: each retriever
produces a ranked list, every document in a list scores `weight / (rank + c)`
(`c` a small constant), and the weighted sums are re-ranked. With equal weights
the two signals are simply averaged.

---

```python
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5],
)
print("EnsembleRetriever built (reciprocal rank fusion, 50/50)")
```

---

```python
query = "Which HNSW graph index does vector search use for approximate nearest neighbour search?"

bm25_hits = bm25_retriever.invoke(query)
dense_hits = dense_retriever.invoke(query)
fused_hits = ensemble.invoke(query)

print(f"BM25 → {len(bm25_hits)} | Dense → {len(dense_hits)} | Fused → {len(fused_hits)}")
```

---

```python
def show(name, hits):
    print(f"--- {name} ---")
    for i, d in enumerate(hits, 1):
        print(f"  #{i} | {d.page_content[:70]}")
    print()

show("BM25", bm25_hits)
show("Dense", dense_hits)
show("Fused", fused_hits)
```

---

## 6 · Prompt — package context + question

The prompt template is identical to the earlier projects. Here the context is
built from the **fused** ranking — the model sees the passages that *both*
retrievers collectively considered most relevant.

---

```python
context = "\n\n".join(d.page_content for d in fused_hits)
print(f"Fused context: {len(context)} chars from {len(fused_hits)} chunks")
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
print(prompt.format(context=context[:250], question=query))
```

---

## 7 · Answer — Gemini LLM

The final block sends the rendered prompt to `ChatGoogleGenerativeAI`
(`gemini-2.5-flash`) and prints the answer grounded in the fused context.

---

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = prompt.invoke({"context": context, "question": query})
response = llm.invoke(messages)

print(response.content)
```

---

## 8 · Try it yourself

The `weights` list is a knob between the two extremes. Swapping in `[1.0, 0.0]`
or `[0.0, 1.0]` reproduces a **pure keyword** or **pure dense** retriever — so
the next cells let you answer a *semantic* question (where meaning matters) with
either side alone, to see where each retriever is strong or blind.

---

```python
kw_only = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[1.0, 0.0],
)
sem_only = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.0, 1.0],
)
print("Built keyword-only (1.0, 0.0) and dense-only (0.0, 1.0) ensembles")
```

---

```python
semantic_query = "What is the idea of finding text by meaning rather than by exact words?"

print("--- keyword-only (BM25 drives everything) ---")
for i, d in enumerate(kw_only.invoke(semantic_query), 1):
    print(f"  #{i} | {d.page_content[:70]}")
print("--- dense-only (embeddings drive everything) ---")
for i, d in enumerate(sem_only.invoke(semantic_query), 1):
    print(f"  #{i} | {d.page_content[:70]}")
```

---

## What you should notice

* **Two retrievers, one corpus.** Chroma (dense) and `BM25Retriever` (sparse)
  both speak the same `invoke(query)` interface, so swapping and fusing them is
  trivial — the "swap one block" design again.
* **Keyword catches exact terms.** Rare tokens like `HNSW` are BM25's home
  turf: it matches them verbatim, with no embedding required.
* **Dense catches semantics.** Paraphrases and related meaning that share few
  words still land near each other in vector space.
* **Fusion is robust.** `EnsembleRetriever` sums reciprocal ranks, so even when
  one retriever misses, the other still contributes — the fused list rarely
  loses a passage that *either* side found.
* **Weights are a dial.** `[0.5, 0.5]` blends; `[1.0, 0.0]` and `[0.0, 1.0]`
  isolate each retriever, which is a great debugging technique.
* **One install.** `BM25Retriever` needs the `rank-bm25` package — the only
  extra dependency in this project.

---

## Exercises

1. **Tune the weights.** Change `weights` to `[0.8, 0.2]` and re-run Section 5's
   query — how does the fused ranking shift toward the keyword side?
2. **Build a disagreement case.** Add a passage containing an acronym (e.g.
   `"RRF"`) and ask a query that only makes sense semantically; compare the
   three lists and confirm fusion still surfaces the acronym passage.
3. **Answer from each context.** Answer the same question from BM25-only,
   dense-only and fused context, and judge which answer is best grounded.
