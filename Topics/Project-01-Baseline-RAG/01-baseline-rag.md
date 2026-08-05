# Project 1 — The Baseline RAG

> Stop thinking about LangChain classes. Think of RAG as a pipeline where **every block is replaceable**.

## The pipeline

```
            Documents
                │
                ▼
         Document Loader
                │
                ▼
          Text Splitter
                │
                ▼
           Embedding Model
                │
                ▼
           Vector Database
                │
                ▼
             Retriever
                │
                ▼
              Prompt
                │
                ▼
               LLM
                │
                ▼
              Answer
```

Every one of these blocks is replaceable. That single sentence is the whole
course. The baseline project below uses the *simplest possible* component in
each slot, so you can see exactly what each one does before you start swapping.

## The baseline stack

| Block | Component | Why it's the baseline |
|-------|-----------|------------------------|
| Loader | `WebBaseLoader` | One URL in, `List[Document]` out |
| Splitter | `RecursiveCharacterTextSplitter` | Sensible default for prose |
| Embedding | Gemini Embedding | Good quality, zero local setup |
| Vector DB | Chroma | Local, persistent, zero-config |
| Retriever | Similarity Search (Top-K) | The naive baseline every strategy improves on |
| Prompt | Basic Context + Question | The minimum viable prompt |
| LLM | Gemini 2.5 Flash | Fast, cheap, good enough |

The rule of Stage 1: **only the block you're studying changes.** Everything
else stays identical, so any difference you observe is caused by that one block.

## Build it

### 1. Load

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.gutenberg.org/cache/epub/79247/pg79247-images.html")
docs = loader.load()
```

Every loader — web, PDF, CSV, Markdown — returns `List[Document]`. Nothing else
in the pipeline cares how the text got in.

### 2. Split

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
```

Chunk size and overlap are the first two knobs you will ever tune. `1000/200`
means "chunks of ~1000 characters, keeping 200 characters of overlap so no
sentence is cut across a boundary."

### 3. Embed

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
```

An embedding turns text into a vector — a list of numbers — such that texts
with similar *meaning* land close together in vector space. That distance is
what retrieval is built on.

### 4. Store

```python
from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_langchain_db",
)
```

The vector database stores the chunks *and* their embeddings. At query time it
answers one question: "which stored chunks are closest to this query vector?"

### 5. Retrieve

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
```

Top-K similarity: embed the question, find the 5 nearest chunks, return them.
Naive — but every advanced retriever (MMR, parent-child, multi-query) is
measured against this baseline.

### 6. Prompt

```python
prompt = f"""
Context:
{context}

Question:
{question}
"""
```

The minimum viable RAG prompt. No tricks, no few-shots — just "here is the
evidence, here is the question, answer."

### 7. Generate

```python
answer = llm.invoke(prompt)
```

The LLM answers using only the retrieved context. Retrieval quality and prompt
quality are now the two levers that decide whether the answer is right.

## What you should notice

1. **The contract between blocks is tiny.** Loader → `List[Document]`, splitter
   → `List[Document]`, store → "nearest neighbors". That's why every block is
   swappable with one line.
2. **The LLM never sees the whole book.** It sees ~5 chunks (≈5k characters).
   RAG is a *memory management* strategy for LLMs: bring the relevant evidence
   to a fixed-size context window.
3. **Garbage in, garbage out.** If the splitter breaks a section mid-paragraph
   or the embeddings are weak, no prompt will fix the answer.

## Where this goes next

- Project 2 — same pipeline, `PyPDFLoader` instead of `WebBaseLoader`
- Project 3 — offline embeddings (BGE) + FAISS, no API calls
- Project 6 — keep everything, rotate embedding models, measure retrieval accuracy
- Project 8 — replace similarity with MMR, observe diversity

The complete walkthrough lives in
[`NoteBooks/Project-01-Baseline-RAG/04-baseline-rag.ipynb`](../../NoteBooks/Project-01-Baseline-RAG/04-baseline-rag.ipynb).
The component modules live in `src/loaders/`, `src/splitters/`, `src/embeddings/`,
`src/vectordb/`, `src/retrieval/`, `src/prompts/`, `src/llms/` — each one a class you can swap.
