> Source notebook: `NoteBooks/Projects/Project-10-MultiQuery-Retrieval/01-multiquery-retrieval.ipynb`


---

# Project 10 — MultiQuery Retrieval

> **Goal:** The LLM rephrases the question into several variants; we retrieve
> with all variants and dedupe the results.

```
Loader      : DirectoryLoader (Topics/*.md)
Splitter    : RecursiveCharacterTextSplitter (1000 / 200)
Embedding   : Gemini Embedding
Vector DB   : Chroma
Retriever   : Base similarity (k=4) → MultiQueryRetriever
Prompt      : Basic Context + Question
LLM         : Gemini 2.5 Flash
```

**What changes in this project:** only the **Retriever** block. Everything before
it (load → split → embed → store) is identical to the baseline pipeline, so any
difference you observe comes from the retriever alone.

---

## 0 · Setup — environment & keys

We load the `.env` file that holds our `GOOGLE_API_KEY` and confirm the key is
present. We only ever print a **masked** preview — never the full key. The whole
pipeline needs this key because both the embeddings and the LLM are Gemini
models running in Google's cloud.

---

```python
from dotenv import load_dotenv

load_dotenv()
```

---

```python
import os

key = os.getenv("GOOGLE_API_KEY", "")
if not key:
    raise SystemExit("GOOGLE_API_KEY not found — copy .env.example to .env and add your key.")
print(f"GOOGLE_API_KEY set: {key[:4]}… (len {len(key)})")
```

---

These are the same real libraries Project 01 uses. One version note: in
langchain 1.x the classic retrievers (`MultiQueryRetriever`, …) moved into the
`langchain_classic` package, which is installed automatically as a dependency of
`langchain-community`. The `try/except` below also supports older langchain
versions where the same classes live under `langchain.retrievers`.

---

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_classic.retrievers import MultiQueryRetriever
except ImportError:  # older langchain
    from langchain.retrievers import MultiQueryRetriever
```

---

## 1 · Load

Our corpus is the repo's own `Topics/*.md` files — the project cards of this RAG
curriculum. They describe the same concepts (retrievers, chunking, embeddings…)
with **different wording from file to file**, which is exactly the situation that
makes plain similarity search miss things and query expansion help.

`DirectoryLoader` + `TextLoader` read every markdown file and produce one
`Document` per file. A `Document` is just text (`page_content`) plus metadata
(here: the source path).

---

```python
from pathlib import Path

topics_dir = Path("../../Topics")
print("corpus dir exists:", topics_dir.exists())

loader = DirectoryLoader(
    str(topics_dir),
    glob="**/*.md",
    loader_cls=TextLoader,
)
docs = loader.load()
print("documents loaded:", len(docs))
```

---

```python
print("first source:", docs[0].metadata["source"])
print("first doc chars:", len(docs[0].page_content))
print("--- snippet ---")
print(docs[0].page_content[:300])
```

---

## 2 · Split

Chunks must be small enough that the embedding stays focused and the retrieved
piece is a precise answer fragment. `RecursiveCharacterTextSplitter` first tries
paragraph breaks, then sentences, then words, so it keeps natural boundaries.

We use the baseline `chunk_size=1000` with `chunk_overlap=200` so the split point
never cuts a thought in half.

---

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)
print("chunks:", len(chunks))
```

---

```python
print("--- sample chunk (chars:", len(chunks[3].page_content), ") ---")
print(chunks[3].page_content[:300])
print("--- metadata ---")
print(chunks[3].metadata)
```

---

## 3 · Embed

Embeddings turn text into vectors (lists of numbers) where similar *meaning*
sits near similar numbers. When every chunk is embedded we can search by "closest
vector to the question's vector". Gemini's embedding model returns one vector per
text — we embed one sample string just to see its size.

---

```python
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

sample_vec = embeddings.embed_query("What is a retriever?")
print("vector dimensions:", len(sample_vec))
print(sample_vec[:3], "…")
```

---

## 4 · Store

Chroma stores each chunk's vector and text together. `from_documents` embeds
every chunk, indexes the vectors, and returns a `Chroma` object we can search.
No persistence directory is used here — the store lives in memory for this
session, which is fine for a single experiment.

---

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)
print("vector store ready")
```

---

## 5 · Retrieve — one query or many?

**This is the block this project changes.**

A plain similarity retriever embeds the query once and returns the `k` closest
chunks. The problem: the user's wording and the document's wording can differ
("How can I make search better?" vs. "swap one component at a time"), so the
best chunk may score poorly just because it uses different words.

Query expansion fixes this by asking the LLM for **several paraphrases** of the
question, retrieving with each one, and merging the results — more coverage,
still deduplicated. That is exactly what `MultiQueryRetriever` does.

Because the retriever needs the LLM to *generate* the variants, we create the
Gemini chat model here (before the Answer section), not later.

---

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
question = "How can retrieval quality be improved beyond plain vector similarity?"

print("LLM ready (also used to expand the query)")
```

---

**Baseline first — one query, top-4 chunks.** Note which files these come
from: these are the only 4 chunks a plain retriever would feed to the LLM.

---

```python
base_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
single_docs = base_retriever.invoke(question)
print("single-query chunks:", len(single_docs))
for d in single_docs:
    print("  -", Path(d.metadata["source"]).name)
```

---

**Step 1 of multi-query: expand the question.** The prompt below mirrors the
default template `MultiQueryRetriever` uses internally — we call it ourselves
first so we can see the generated variants with our own eyes.

---

```python
expansion_template = """You are an AI assistant. Generate 3 different versions of
the given question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the question you help overcome the
limitations of distance-based similarity search. Provide the alternative
questions separated by newlines. Original question: {question}"""

expansion_chain = ChatPromptTemplate.from_template(expansion_template) | llm | StrOutputParser()
variants = expansion_chain.invoke({"question": question})
variants = [v.strip() for v in variants.splitlines() if v.strip()]
print("generated variants:")
for i, v in enumerate(variants, 1):
    print(f"  {i}. {v}")
```

---

**Step 2 + 3: let `MultiQueryRetriever` do it all.** It wraps the base
retriever, uses the LLM to generate the variants, retrieves with each variant,
and returns only the *unique* documents (`unique_union`). `include_original=True`
keeps the user's original question in the list, so nothing is lost.

First we ask it to show us the queries it generated — this is the same LLM we
already saw produce variants above, now working inside the retriever.

---

```python
from uuid import uuid4
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

mq_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True,
)

run_manager = CallbackManagerForRetrieverRun(
    run_id=uuid4(), handlers=[], inheritable_handlers=[]
)
print("queries generated by MultiQueryRetriever:")
for q in mq_retriever.generate_queries(question, run_manager):
    print("  -", q)
```

---

```python
mq_docs = mq_retriever.invoke(question)
print("unique documents returned:", len(mq_docs))
for d in mq_docs:
    print("  -", Path(d.metadata["source"]).name)
```

---

**Single-query vs. multi-query, side by side.** The number of *unique* chunks
usually grows because each variant can hit a different part of the corpus. The
extra chunks are exactly the "different wording" pieces a single embedding of
the question would have missed.

---

```python
print("single-query chunks :", len(single_docs))
print("multi-query unique  :", len(mq_docs))
print("extra coverage      :", len(mq_docs) - len(single_docs), "more chunk(s)")
```

---

## 6 · Prompt

The prompt packages the retrieved context plus the question for the LLM. This is
the same template as the baseline: instruct the model to answer ONLY from the
context and to admit ignorance otherwise. We build it with `ChatPromptTemplate`
and render one example so you can see the exact text the model receives.

---

```python
template = """You are a helpful assistant.

Answer the question using ONLY the provided context.
If the answer is not contained in the context, say:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)
```

---

```python
context = "\n\n".join(doc.page_content for doc in mq_docs)
messages = prompt.invoke({"context": context, "question": question})
print(messages.content[:150], "…")
```

---

## 7 · Answer

Now the LLM call. We reuse the `llm` we created in the Retrieve section (a
`ChatGoogleGenerativeAI` on `gemini-2.5-flash`) — the same model that expanded
the query. `invoke` returns a response whose `.content` is the answer text.

---

```python
response = llm.invoke(messages)
print(response.content)
```

---

## 8 · Try it yourself

Change the question, the top-k, or how many variants the LLM is asked for, then
rerun the retrieve + answer cells. Below are two more questions worth comparing
single-query vs. multi-query.

---

```python
q2 = "What does token reduction mean for a RAG pipeline?"
docs2 = mq_retriever.invoke(q2)
print("multi-query docs:", len(docs2))
for d in docs2[:6]:
    print("  -", Path(d.metadata["source"]).name)
```

---

```python
context2 = "\n\n".join(d.page_content for d in docs2)
messages2 = prompt.invoke({"context": context2, "question": q2})
print(llm.invoke(messages2).content)
```

---

## What you should notice

- **Query expansion changes the query, not the retriever.** The base retriever is
  untouched; `MultiQueryRetriever` simply feeds it several paraphrases.
- **The variants matter.** With plain similarity, chunks whose wording matches the
  query win; expansion pulls in chunks that *mean* the same thing but are worded
  differently.
- **Deduplication is built in.** `MultiQueryRetriever` returns the unique union of
  all variant results — the same chunk is not repeated even if several variants
  hit it.
- **Cost / latency tradeoff.** `N` variants ≈ `N` retrieval calls plus `1` LLM
  call for the expansion itself. Better recall, slightly slower and pricier.
- **Downstream blocks are untouched.** The prompt and answer sections are
  identical to the baseline — the effect we observed comes only from the
  retriever block.

---

## Exercises

1. Set `include_original=False` and compare the unique documents with and
   without the original question included.
2. Change `k` on the base retriever (e.g. 2 and 8) and watch how the extra
   coverage from multi-query shrinks or grows.
3. Ask a question whose answer lives in *one* project card and check whether the
   variants still broaden the result set, or just add noise.
