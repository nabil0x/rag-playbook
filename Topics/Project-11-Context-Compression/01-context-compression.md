> Source notebook: `NoteBooks/Projects/Project-11-Context-Compression/01-context-compression.ipynb`


---

# Project 11 — Context Compression

> **Goal:** Replace 5 full chunks with only the sentences that are actually
> useful for the question.

```
Loader      : DirectoryLoader (Topics/*.md)
Splitter    : RecursiveCharacterTextSplitter (1000 / 200)
Embedding   : Gemini Embedding
Vector DB   : Chroma
Retriever   : Base similarity (k=5) → ContextualCompressionRetriever
             + LLMChainExtractor
Prompt      : Basic Context + Question
LLM         : Gemini 2.5 Flash
```

**What changes in this project:** only the **Retriever** block. Everything before
it is identical to the baseline pipeline. The retriever now *compresses* the
retrieved chunks so the LLM only ever sees the relevant sentences.

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
langchain 1.x the classic retrievers and document compressors moved into the
`langchain_classic` package (installed automatically as a dependency of
`langchain-community`). The `try/except` below also supports older langchain
versions where they lived under `langchain.retrievers` /
`langchain_community.document_compressors`.

---

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
except ImportError:  # older langchain
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.document_compressors import LLMChainExtractor
```

---

## 1 · Load

Our corpus is the repo's own `Topics/*.md` files — the project cards of this RAG
curriculum. They describe retrieval concepts with different wording from file to
file, which gives the retriever realistic work to do.

`DirectoryLoader` + `TextLoader` read every markdown file and produce one
`Document` per file. A `Document` is just text (`page_content`) plus metadata
(here: the source path).

---

```python
from pathlib import Path

loader = DirectoryLoader(
    str(Path("../../Topics")),
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
print(docs[0].page_content[:200])
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
print("--- sample chunk (chars:", len(chunks[5].page_content), ") ---")
print(chunks[5].page_content[:250])
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

sample_vec = embeddings.embed_query("What is context compression?")
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

## 5 · Retrieve — compress before you answer

**This is the block this project changes.**

A top-5 retriever returns **5 full chunks** — but usually only a sentence or two
inside each chunk is actually relevant to the question. Feeding all of it to the
LLM wastes tokens (and therefore money and latency), and the extra noise can even
hurt answer quality.

Context compression fixes this: after retrieval, a **compressor** passes each
chunk through the LLM and keeps only the parts that answer the question.

Because the compressor needs the LLM, we create the Gemini chat model here
(before the Answer section).

---

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
question = "What is context compression and what are its benefits in a RAG pipeline?"

print("LLM ready (also used to compress the chunks)")
```

---

**Baseline first — top-5, full chunks.** Measure how much text the LLM would
have received. We count characters and estimate tokens with a rough rule of thumb
(~4 characters per English token) because we do not want a second API call just
to count tokens.

---

```python
base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
base_docs = base_retriever.invoke(question)


def total_chars(docs):
    return sum(len(d.page_content) for d in docs)


def est_tokens(docs):
    return total_chars(docs) // 4  # rough: ~4 chars per English token

print("base retrieval (full chunks):")
for i, d in enumerate(base_docs, 1):
    print(f"  chunk {i}: {len(d.page_content):5d} chars  <- {Path(d.metadata['source']).name}")
print(f"TOTAL chars: {total_chars(base_docs)}  |  est. tokens: {est_tokens(base_docs)}")
```

---

**Now wrap the base retriever in a `ContextualCompressionRetriever`.** It keeps
the exact same retrieval step, then runs an `LLMChainExtractor` over the
results. The extractor asks the LLM to pull out, verbatim, "any part of the
context that is relevant to the question" and drops the rest. Chunks with nothing
relevant are removed entirely.

---

```python
compressor = LLMChainExtractor.from_llm(llm)

cc_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

compressed_docs = cc_retriever.invoke(question)
print("compressed documents:", len(compressed_docs), "of", len(base_docs), "original chunks")
```

---

**Measure the savings.** Compare total characters and estimated tokens before
and after compression, and print the percentage of text removed.

---

```python
print("compressed retrieval (useful sentences only):")
for i, d in enumerate(compressed_docs, 1):
    print(f"  doc {i}: {len(d.page_content):5d} chars  <- {Path(d.metadata['source']).name}")
print(f"TOTAL chars: {total_chars(compressed_docs)}  |  est. tokens: {est_tokens(compressed_docs)}")

base_chars = total_chars(base_docs)
compressed_chars = total_chars(compressed_docs)
saved = 100 * (1 - compressed_chars / base_chars)
print(f"characters saved: {base_chars - compressed_chars} ({saved:.1f}%)")
```

---

**Look at what survived.** Each compressed `Document` should contain only the
sentences that actually help answer the question — and its metadata (the source
file) is carried through, so you can still trace every sentence back to a chunk.

---

```python
for i, d in enumerate(compressed_docs, 1):
    print(f"--- doc {i} <- {Path(d.metadata['source']).name} ---")
    print(d.page_content)
    print()
```

---

**Cost / latency tradeoff.** Fewer tokens going into the answer call means a
cheaper and faster answer. But the compression step itself costs one LLM call
*per chunk* (here: 5 calls), so the tradeoff only pays off when the savings in
the answer call outweigh the cost of compressing. The cell below converts our
token estimate into money using an example input price.

---

```python
price_per_1m = 0.30  # example USD / 1M input tokens — check live Gemini pricing

tokens_saved = est_tokens(base_docs) - est_tokens(compressed_docs)
cost_saved = tokens_saved / 1_000_000 * price_per_1m
print(f"est. tokens saved per query : {tokens_saved}")
print(f"est. cost saved per query   : ${cost_saved:.6f}")
print("note: compression costs one LLM call per chunk — the real tradeoff.")
```

---

## 6 · Prompt

The prompt packages the retrieved context plus the question for the LLM. This is
the same template as the baseline: instruct the model to answer ONLY from the
context and to admit ignorance otherwise. The only difference from Project 10 is
that `context` here is built from the *compressed* documents.

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
context = "\n\n".join(doc.page_content for doc in compressed_docs)
messages = prompt.invoke({"context": context, "question": question})
print(messages.content[:150], "…")
```

---

## 7 · Answer

Now the LLM call. We reuse the `llm` from the Retrieve section — the same model
that compressed the chunks now reads the compressed context and answers. The
answer call receives far fewer tokens than it would have without compression.

---

```python
response = llm.invoke(messages)
print(response.content)
```

---

## 8 · Try it yourself

Change the question, the top-k, or the chunk size and rerun the retrieve +
answer cells. A different question will keep a different subset of sentences —
notice how compression adapts to the query.

---

```python
q2 = "How do rerankers or compressors reduce the token budget of a query?"
docs2 = cc_retriever.invoke(q2)
print("compressed docs:", len(docs2))
print("chars:", total_chars(docs2), "| est. tokens:", est_tokens(docs2))
```

---

```python
context2 = "\n\n".join(d.page_content for d in docs2)
messages2 = prompt.invoke({"context": context2, "question": q2})
print(llm.invoke(messages2).content)
```

---

## What you should notice

- **Compression preserves meaning, not text.** Only the sentences relevant to the
  *question* survive; filler is dropped by design.
- **The numbers are the point.** A top-5 full-chunk context of ~2 000 characters
  can shrink to a few hundred — that is a large token saving on every single
  question.
- **Smaller input, cheaper and faster answers.** Fewer tokens per answer call
  means lower cost and lower latency per request — and less room for the LLM to
  get distracted by irrelevant text.
- **Compression is not free.** It costs one LLM call per chunk, so there is a
  real latency/cost tradeoff; it pays off when the saved answer tokens exceed the
  compression cost.
- **Metadata survives.** Each compressed `Document` still carries its source, so
  answers remain traceable to their origin.

---

## Exercises

1. Change `k` from 5 to 10 and see how total saved characters scale — does the
   percentage stay similar?
2. Ask a question that is *not* answerable from the corpus and watch the
   compressor drop chunks until the context is nearly empty.
3. Compare the answer quality from full chunks vs. compressed chunks for the same
   question: is the compressed answer worse, better, or equal?
