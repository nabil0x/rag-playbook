# Project 03 — Local Documentation Search

> **Goal:** Index a whole directory tree offline — no cloud embedding API.

## Stack

```
Loader      : DirectoryLoader
Splitter    : Recursive
Embedding   : BGE
Vector DB   : FAISS
Retriever   : Similarity
Prompt      : Basic
LLM         : Gemini
```

## Example corpus

```
project/
├── README.md
├── docs/
├── examples/
└── tutorials/
```

## New concepts

- Offline embeddings (sentence-transformers / BGE)
- Offline vector search (FAISS)

## Article

- [ ] `01-local-documentation-search.md`

## Code

`src/loaders/` (add `directory.py`), `src/embeddings/bge.py`, `src/vectordb/faiss.py`

## Notebook

`NoteBooks/Projects/Project-03-Local-Documentation-Search/01-local-documentation-search.ipynb`
