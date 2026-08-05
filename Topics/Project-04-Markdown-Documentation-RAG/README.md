# Project 04 — Markdown Documentation RAG

> **Goal:** Split on structure instead of randomly — Markdown already has sections.

## Stack

```
Loader      : UnstructuredMarkdownLoader
Splitter    : MarkdownHeaderTextSplitter
Embedding   : BGE
Vector DB   : FAISS
Retriever   : Similarity
Prompt      : Citation Prompt
LLM         : Gemini
```

## Why

Instead of splitting blindly —

```
Installation ↓ Usage ↓ API
```

— preserve sections as chunks, so retrieval respects document structure.

## Article

- [ ] `01-markdown-documentation-rag.md`

## Code

`src/splitters/` (add `markdown_header.py`), `src/prompts/citation.py`

## Notebook

`NoteBooks/Project-04-Markdown-Documentation-RAG/01-markdown-documentation-rag.ipynb`
