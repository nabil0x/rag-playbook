> Source notebook: `NoteBooks/Projects/Project-14-Metadata-Filtering/01-metadata-filtering.ipynb`


---

# Project 14 — Metadata Filtering

**Goal:** Scope retrieval by metadata instead of searching everything.

```
Loader      : Inline synthetic corpus (tagged AI / Finance / Medicine)
Splitter    : RecursiveCharacterTextSplitter (passages already small)
Embedding   : Gemini
Vector DB   : Chroma (metadata-aware)
Retriever   : Similarity (Top-K) + where={...} filter
Prompt      : Basic Context + Question
LLM         : Gemini
```

So far every query searched the *whole* corpus. Real deployments are bigger and
messier: a law-firm RAG has contracts *and* memos, a hospital has cardiology
*and* dermatology notes. You usually want an answer from **one category**, not
a blend. This project tags every document with a `category` (`AI`, `Finance`,
`Medicine`) and uses Chroma's metadata filter to scope the *same* query to a
single category.

Learn:

* **Metadata as a cheap pre-filter** — narrow the candidate set before ranking
* **Chroma `where` syntax** — the dictionary rules for filtering
* **Filtered vs unfiltered retrieval** — the same query, two different answers

---

### How to work through this notebook

This notebook runs the **full pipeline** (Load → Split → Embed → Store →
Retrieve → Prompt → Answer) with a deliberately tiny, hand-written corpus: 12
short passages, 4 per category, so you can read every retrieval result at a
glance. The whole point lives in **Section 5**: the *same query* is run twice —
once against everything, once against `{"category": "Finance"}` — and the two
result lists are compared.

---

## 0 · Setup — environment & keys

Two online pieces: Gemini embeddings and the Gemini LLM. Copy `.env.example` to
`.env` at the repo root and add `GOOGLE_API_KEY` from
<https://aistudio.google.com/>. The check below prints only a masked preview
(`key[:4]…`), never the full key.

**No optional packages** are needed for this project — Chroma metadata
filtering is built into `langchain-chroma`.

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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
```

---

## 1 · Load — a tiny tagged corpus, written inline

Instead of a loader and files, this project defines the corpus **directly in
the notebook**: twelve `Document` objects, four per category. Every one carries
a `category` label in its metadata — that label is the whole point of the
project. The passages are short and self-contained so you can read the
retrieval results below at a glance.

---

### 1.1 · AI — four passages

Retrieval, transformers and embeddings — the stack you have been building
through this curriculum.

---

```python
ai_docs = [
    Document(
        page_content="Retrieval-augmented generation couples a retriever over a corpus with a generator, so the model can cite evidence it was never trained on.",
        metadata={"category": "AI"},
    ),
    Document(
        page_content="Transformer models process entire sequences in parallel with self-attention, letting every token attend to every other token.",
        metadata={"category": "AI"},
    ),
    Document(
        page_content="Embedding models map sentences into a vector space where text with similar meaning lands close together.",
        metadata={"category": "AI"},
    ),
    Document(
        page_content="Chunk size and overlap control retrieval granularity: too big blurs the topic, too small loses the answer.",
        metadata={"category": "AI"},
    ),
]
print(f"{len(ai_docs)} AI passages")
```

---

### 1.2 · Finance — four passages

Money basics: interest, diversification, inflation and bonds.

---

```python
finance_docs = [
    Document(
        page_content="Compound interest earns interest on interest, so savings grow exponentially rather than linearly over time.",
        metadata={"category": "Finance"},
    ),
    Document(
        page_content="Diversification spreads money across asset classes so that a single sector's downturn does not sink the whole portfolio.",
        metadata={"category": "Finance"},
    ),
    Document(
        page_content="Inflation measures how the general price level rises, which erodes the purchasing power of a fixed sum of money.",
        metadata={"category": "Finance"},
    ),
    Document(
        page_content="A bond is a fixed-income instrument: the issuer pays periodic coupons and repays the principal at maturity.",
        metadata={"category": "Finance"},
    ),
]
print(f"{len(finance_docs)} Finance passages")
```

---

### 1.3 · Medicine — four passages

Public-health and nutrition facts.

---

```python
medicine_docs = [
    Document(
        page_content="Vitamin D supports calcium absorption, which keeps bones dense and reduces the risk of fractures in older adults.",
        metadata={"category": "Medicine"},
    ),
    Document(
        page_content="Antibiotics kill bacteria; they have no effect on viruses, which is why they do not treat the common cold.",
        metadata={"category": "Medicine"},
    ),
    Document(
        page_content="Regular aerobic exercise strengthens the heart muscle and improves how efficiently blood is pumped around the body.",
        metadata={"category": "Medicine"},
    ),
    Document(
        page_content="Hand hygiene remains one of the most effective measures for preventing the spread of infectious disease.",
        metadata={"category": "Medicine"},
    ),
]
print(f"{len(medicine_docs)} Medicine passages")
```

---

```python
docs = ai_docs + finance_docs + medicine_docs

print(f"Corpus: {len(docs)} Documents "
      f"({len(ai_docs)} AI / {len(finance_docs)} Finance / {len(medicine_docs)} Medicine)")
```

---

```python
print("Every document carries a category label:")
for d in docs:
    print(f"  {d.metadata['category']:8} | {d.page_content[:42]}…")
```

---

## 2 · Split — a formality here (passages are already small)

The corpus is 12 short, single-topic passages — each is already about the size
of a useful chunk. We still run `RecursiveCharacterTextSplitter` with a budget
large enough that every passage survives as its own chunk, because it proves an
important property: **metadata (including `category`) is copied onto every
chunk**, so the filter still works after splitting.

---

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)

print(f"{len(docs)} passages → {len(chunks)} chunks (metadata preserved)")
```

---

```python
from collections import Counter

print("Categories among chunks:",
      dict(Counter(c.metadata["category"] for c in chunks)))
print("Sample chunk metadata:", chunks[0].metadata)
```

---

## 3 · Embed — text → vectors

Same embedding model as the previous projects: `GoogleGenerativeAIEmbeddings`
sends each chunk to the Gemini API and returns a 3072-dimension vector. The
`category` label is **not** part of the embedding — it is metadata, stored
alongside the vector and used as a filter in Section 5.

---

```python
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

sample_vec = embeddings.embed_query("metadata filtering in RAG")
print(f"Embedding dimension: {len(sample_vec)}")
```

---

## 4 · Store — Chroma keeps the metadata

`Chroma.from_documents` embeds each chunk and stores the `(text, vector,
metadata)` triple. You do not need to do anything special — the `category`
label rides along automatically. That stored metadata is what Section 5 will
filter on. A distinct `collection_name` keeps this project's index separate
from other notebooks' if you run several in the same kernel.

---

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="project14_categories",
)
print("Chroma index created with category metadata stored")
```

---

## 5 · Retrieve — the SAME query, filtered and unfiltered

The experiment: run one query twice.

1. **Unfiltered** — the whole 12-doc corpus is a candidate set, ranked purely
   by vector similarity.
2. **Filtered** — with `where={"category": "Finance"}`, Chroma narrows
   candidates to the four Finance chunks **before** any similarity comparison.

Watch the second list: every hit is Finance. The filter is a *pre-filter* — it
does not re-rank, it *excludes*.

---

### 5.1 · The filter keyword: `where` vs `filter`

Chroma metadata filters are **`where` dictionaries**: `{"category": "Finance"}`.
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
query = "How does spreading money across assets reduce risk?"

unfiltered = scoped_search(vector_store, query, k=4)

for doc, score in unfiltered:
    print(f"[{score:.4f}] {doc.metadata['category']:8} | {doc.page_content[:55]}")
```

---

```python
filtered = scoped_search(
    vector_store,
    query,
    k=4,
    where={"category": "Finance"},
)

for doc, score in filtered:
    print(f"[{score:.4f}] {doc.metadata['category']:8} | {doc.page_content[:55]}")
```

---

```python
from collections import Counter

def cat_dist(hits):
    """Count categories among retrieved (doc, score) tuples."""
    return dict(Counter(d.metadata["category"] for d, _ in hits))

print("Unfiltered categories:", cat_dist(unfiltered))
print("Filtered categories  :", cat_dist(filtered))
print("Filtered pulls only Finance:",
      all(d.metadata["category"] == "Finance" for d, _ in filtered))
```

---

### 5.2 · Chroma `where` syntax

`where` is a plain dict of rules on metadata values:

```python
where={"category": "Finance"}           # shorthand for $eq
where={"category": {"$eq": "Finance"}}  # explicit operator form
```

Both mean the same thing. Chroma also supports `$ne`, `$in`, `$nin`, `$gt`,
`$lt`, and the logical combiners `$and` / `$or` — e.g.
`{"category": {"$in": ["AI", "Medicine"]}}` keeps two categories.

---

```python
explicit_eq = scoped_search(
    vector_store,
    query,
    k=4,
    where={"category": {"$eq": "Finance"}},
)

print("Explicit $eq form gives the same scope:",
      dict(Counter(d.metadata["category"] for d, _ in explicit_eq)))
```

---

## 6 · Prompt — package context + question

The prompt template is identical to the earlier projects: fuse the retrieved
chunks into `{context}`, add the `{question}`, and instruct the LLM to answer
*only* from the context. Here the context comes from the **Finance-filtered**
retrieval, so the model's answer will be grounded in Finance passages alone.

---

```python
context = "\n\n".join(doc.page_content for doc, _ in filtered)
print(f"Context from {len(filtered)} Finance-only chunks ({len(context)} chars)")
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
(`gemini-2.5-flash`). Because retrieval was scoped to `{"category": "Finance"}`,
the answer comes from the Finance passages — a grounded, single-domain answer.

---

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = prompt.invoke({"context": context, "question": query})
response = llm.invoke(messages)

print(response.content)
```

---

## 8 · Try it yourself

Change the filter category (`"Medicine"`, `"AI"`), combine categories with
`$in`, or drop the filter and compare the answers. Two ready-made variants
below.

---

```python
med_query = "How can I keep my bones strong?"

med_hits = scoped_search(vector_store, med_query, k=3, where={"category": "Medicine"})

for doc, score in med_hits:
    print(f"{doc.metadata['category']:8} | {doc.page_content[:55]}")
```

---

```python
two_cats = scoped_search(
    vector_store,
    "Which models learn from large text collections?",
    k=4,
    where={"category": {"$in": ["AI", "Medicine"]}},
)

for doc, score in two_cats:
    print(f"{doc.metadata['category']:8} | {doc.page_content[:55]}")
```

---

## What you should notice

* **The filter excludes; it does not re-rank.** With `{"category": "Finance"}`,
  the four Finance chunks are the *only* candidates — unrelated-but-similar
  chunks are gone before any scoring happens.
* **Metadata is a cheap precision lever.** As the corpus grows, filtering first
  means the retriever ranks within a small, relevant set instead of everything.
* **Same query, different scope, different answer.** The unfiltered run mixed
  categories; the filtered run was Finance-only — proof that retrieval, not the
  question, decides what evidence the LLM sees.
* **`where` is just a dict.** Shorthand (`"Finance"`), explicit operator
  (`{"$eq": "Finance"}`), membership (`{"$in": [...]}`) and the logical
  combiners all compose.
* **Metadata survives splitting and storage.** The `category` tag was copied
  onto every chunk and into the Chroma index with zero extra code.
* **One keyword-rename gotcha.** Recent `langchain-chroma` renamed the Python
  keyword from `where=` to `filter=`; the dict syntax is unchanged (see 5.1).

---

## Exercises

1. **Add a category.** Define three new `Document`s tagged `"Law"`, store them
   (or just re-run Store), then run a query filtered with
   `where={"category": "Law"}` — do you retrieve only the new passages?
2. **Filter on a numeric field too.** Add a `year` value to the metadata of a
   few docs and use `where={"category": "Finance", "year": {"$gt": 2010}}`
   (combine equality + `$gt` in one dict).
3. **Same question, three scopes.** Ask one question through `Medicine`, `AI`
   and unfiltered retrieval, and compare the three answers — this is exactly
   how filtering changes a multi-domain product.
