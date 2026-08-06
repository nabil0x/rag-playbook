> Source notebook: `NoteBooks/Projects/Project-09-Parent-Child-Retrieval/01-parent-child-retrieval.ipynb`


---

# Project 09 — Parent-Child Retrieval

**Goal:** Retrieve on small chunks, return large parent context.

```
Loader      : PDF (PyPDFLoader)  ← NEW
Splitter    : Parent + Child     ← NEW
Embedding   : Gemini Embedding   (BGE alternative noted)
Vector DB   : Chroma
Retriever   : ParentDocumentRetriever  ← NEW
Prompt      : Basic Context + Question
LLM         : Gemini 2.5 Flash
```

Learn:

- Why a *single* chunk size is a compromise between precision and context
- Large chunks = context, small chunks = precision
- How Parent-Child retrieval keeps both: search on the small, return the large

---

## 0 · Setup — environment & keys

Standard setup: load `.env`, verify the masked Google key, import the libraries.

This project needs `pypdf` (already in `requirements.txt`) for the PDF loader,
and one **compat shim**: in langchain 1.x the classic retrievers moved out of
`langchain.retrievers` into `langchain_classic`, and `InMemoryStore` lives in
`langchain_core.stores`. The import below tries the canonical path first and
falls back to the new one, so it runs on both old and new langchain.

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

### Imports

`ParentDocumentRetriever` is the star here: it stores two copies of the data
(small chunks in the vector store, big parents in a docstore) and reconnects
them at retrieval time. `InMemoryStore` is the docstore — a plain key→document
map held in RAM, perfect for learning before you switch to a real database.

---

```python
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
```

---

### Retriever + docstore imports

`ParentDocumentRetriever` and `InMemoryStore` moved packages in langchain 1.x:
`langchain.retrievers`/`langchain.storage` became
`langchain_classic.retrievers`/`langchain_core.stores`. The try/except below
runs on both old and new versions.

---

```python
try:
    from langchain.retrievers import ParentDocumentRetriever
except ImportError:
    from langchain_classic.retrievers import ParentDocumentRetriever

try:
    from langchain.storage import InMemoryStore
except ImportError:
    from langchain_core.stores import InMemoryStore
```

---

## 1 · Load — a PDF

This project's loader is a **PDF** loader (`PyPDFLoader`, backed by `pypdf`).

We point it at a `PDF_PATH` variable. If the file is missing, the notebook
falls back to downloading a public-domain book from Project Gutenberg as a PDF
— or, if the network is unavailable, to the same Gutenberg book via
`WebBaseLoader` (the Project 01 corpus). Set `PDF_PATH` to your own PDF to
replace the sample entirely.

---

```python
from pathlib import Path

# PDF_PATH -> set this to your own PDF, or leave it to use the Gutenberg sample.
PDF_PATH = Path("Data/pg79247.pdf")

GUTENBERG_PDF = "https://www.gutenberg.org/cache/epub/79247/pg79247-pdf.pdf"
GUTENBERG_HTML = "https://www.gutenberg.org/cache/epub/79247/pg79247-images.html"

print("PDF_PATH:", PDF_PATH)
print("pdf exists:", PDF_PATH.exists())
```

---

```python
if PDF_PATH.exists():
    loader = PyPDFLoader(str(PDF_PATH))
else:
    print("PDF not found — trying the Gutenberg download…")
    try:
        import urllib.request
        PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GUTENBERG_PDF, PDF_PATH)
        loader = PyPDFLoader(str(PDF_PATH))
    except Exception as exc:  # network unavailable -> last-resort HTML fallback
        print("download failed:", type(exc).__name__)
        loader = WebBaseLoader(GUTENBERG_HTML)

docs = loader.load()
print("documents:", len(docs))
```

---

```python
print("metadata of page 0:", docs[0].metadata)
print("first chars:", docs[0].page_content[:60].replace(chr(10), " "))
```

---

## 2 · Split — Parent AND Child

Normal RAG splits once and must pick a compromise chunk size:

- **Small chunks** (≈400 chars) match precisely, but carry too little context
  for the LLM to answer well.
- **Large chunks** (≈2000 chars) carry context, but are so coarse that the most
  relevant part is diluted by surrounding text.

Parent-Child retrieval uses **both**: split into large *parent* chunks, then
split each parent into small *child* chunks. Small children get embedded and
searched; the large parents are what the LLM actually sees.

---

```python
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
)

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
)
print("child splitter  → ~400 chars (precision)")
print("parent splitter → ~2000 chars (context)")
```

---

```python
sample = docs[0].page_content

children = child_splitter.split_text(sample)
parents = parent_splitter.split_text(sample)

print("children from page 0:", len(children))
print("parents  from page 0:", len(parents))
print("avg child  len:", round(sum(map(len, children)) / len(children)))
print("avg parent len:", round(sum(map(len, parents)) / len(parents)))
```

---

## 3 · Embed

Children are embedded (they are what similarity compares). The embedding model
is Gemini, same as the other projects; the card also lists **BGE** as an
alternative (`pip install sentence-transformers`, then
`HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en")`) for a fully offline
setup — swap it in and nothing else needs to change.

---

```python
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

probe = embeddings.embed_query("a quick check")
print("vector dimension:", len(probe))
```

---

## 4 · Store — vectorstore + docstore

Two stores work together:

1. **Chroma** — the *vector store*. Only the small child chunks live here, so
   similarity search is precise.
2. **InMemoryStore** — the *docstore*. The large parent chunks live here, keyed
   by id. It is plain memory, so it is cheap and easy to inspect.

---

```python
vectorstore = Chroma(
    collection_name="project09",
    embedding_function=embeddings,
)

docstore = InMemoryStore()
print("vectorstore ready, docstore ready")
```

---

## 5 · ParentDocumentRetriever

This retriever glues the two stores together. On `add_documents` it:

1. splits each document into parents, then children;
2. embeds + stores the **children** in the vector store;
3. stores the **parents** in the docstore under the same ids.

On `invoke(query)` it searches the children, then returns the *parent* of every
matched child.

---

```python
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs)
print("documents added")
```

---

```python
child_count = vectorstore._collection.count()
parent_keys = list(docstore.yield_keys())

print("child chunks in vectorstore:", child_count)
print("parent chunks in docstore:  ", len(parent_keys))
print("sample parent key:", parent_keys[0] if parent_keys else None)
```

---

## 6 · Retrieve — small in, large out

Querying looks just like any other retriever. The difference is in what comes
back: each hit is a **parent** chunk (≈2000 chars) even though the match was
made on a **child** chunk (≈400 chars). That is the core lesson — *retrieve on
small chunks, return large parent context*.

---

```python
QUERY = "What is the moral of the story?"

hits = retriever.invoke(QUERY)
print("retrieved documents:", len(hits))
for i, d in enumerate(hits):
    print(f"{i+1}. {len(d.page_content)} chars | {d.page_content[:60].replace(chr(10), ' ')}…")
```

---

```python
print("child  chunk size ~", child_splitter.chunk_size, "chars (what is searched)")
print("parent chunk size ~", parent_splitter.chunk_size, "chars (what is returned)")
print("retrieved sizes:  ", [len(d.page_content) for d in hits])
```

---

### Peek at one full parent

The LLM reads this bigger context block — enough surrounding text to actually
answer, without the noise of a whole page. Compare it with the tiny child chunk
that triggered the match.

---

```python
print("PARENT (returned to the LLM):")
print(hits[0].page_content[:400])
print("…")
```

---

```python
child_hits = vectorstore.similarity_search(QUERY, k=3)
print("CHILD (what matched in Chroma):")
print(child_hits[0].page_content[:200])
```

---

## 7 · Prompt

Same Basic prompt as always: context + question. The improvement over Project 01
is not the template — it is the **quality of the context**: a large coherent
parent instead of a small fragmented chunk.

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
print("template ready")
```

---

```python
context = "\n\n".join(doc.page_content for doc in hits)

messages = prompt.invoke({"context": context, "question": QUERY})
print("prompt ready,", len(context), "context chars")
```

---

## 8 · Answer

One LLM call. The Gemini model reads the parent-sized context and answers. If
you swap `child_splitter.chunk_size` down to 200 later, the retrieved parents
stay ~2000 chars — the children only change *which* parents get chosen.

---

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

response = llm.invoke(messages)
print(response.content)
```

---

## 9 · Try it yourself

Two knobs control the precision/context trade-off:

1. `child_splitter.chunk_size` — smaller children = more precise matching.
2. `parent_splitter.chunk_size` — bigger parents = more context, more tokens.

Changing either requires rebuilding the retriever (re-run section 5). Try a
different question too, and check the returned sizes stay "parent-sized".

---

```python
# A second query through the same retriever.
for q in ["What is the moral of the story?", "Who is Magpie?"]:
    r = retriever.invoke(q)
    print("Q:", q)
    print("  retrieved:", len(r), "docs, sizes:", [len(d.page_content) for d in r])
```

---

## What you should notice

- **One splitter forces a compromise.** Small chunks match well but answer
  poorly; large chunks answer well but match coarsely. Parent-Child splits both
  ways to avoid the trade-off.
- **The vector store only holds children.** `child_count` in Chroma is large and
  precise; the docstore holds far fewer, much bigger parents.
- **Returned size ≠ searched size.** Retrieval matched a ~400-char child but
  every `hits` document is ~2000 chars — *small in, large out*.
- **`ParentDocumentRetriever` needs both splitter sizes.** `child_splitter` and
  `parent_splitter` are separate objects; mixing up the two is the classic bug
  (a "child" bigger than its "parent" makes no sense).
- **The docstore is where context lives.** Chroma finds *what* to return;
  InMemoryStore supplies the *content*. A real deployment can swap InMemoryStore
  for a persistent key-value store without touching the retriever.

---

## Exercises

1. **Shrink the children.** Set `child_splitter.chunk_size = 200` and re-run the
   build + retrieve. Does the matched child get more specific? Do the returned
   parents change?
2. **Read the plumbing.** After `add_documents`, inspect one parent in the
   docstore (`docstore.mget([parent_keys[0]])`) and confirm its size and that it
   contains its children's text.
3. **Compare with the baseline.** Run the same query through a plain
   `Chroma` retriever from Project 01 and through `ParentDocumentRetriever`, then
   compare the two answers for completeness.
4. **Swap to BGE.** Install `sentence-transformers` and switch to
   `HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en")` — the pipeline
   should work unchanged, fully offline.
