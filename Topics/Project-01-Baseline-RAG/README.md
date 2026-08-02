# Project 01 — The Baseline RAG

> **Goal:** Understand the complete pipeline with the simplest components.

## Stack

```
Loader      : Web Loader
Splitter    : RecursiveCharacterTextSplitter
Embedding   : Gemini Embedding
Vector DB   : Chroma
Retriever   : Similarity Search (Top-K)
Prompt      : Basic Context + Question
LLM         : Gemini 2.5 Flash
```

## Learn

- LangChain basics
- Document lifecycle
- Retrieval flow

## Article

- [ ] `01-baseline-rag.md` — walk the full pipeline end to end (DRAFTED)
- [ ] `02-what-is-rag.md` — why RAG, pipeline diagram, when to use it

## Code

`loaders/web.py` (`WebLoader`), `splitters/recursive.py`, `embeddings/gemini.py`,
`vectordb/chroma.py`, `retrieval/similarity.py`, `prompts/basic.py`, `llms/gemini.py`

## Notebook

`NoteBooks/Project-01-Baseline-RAG/04-baseline-rag.ipynb`
