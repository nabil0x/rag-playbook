# Project 05 — HTML Documentation

> **Goal:** A custom BeautifulSoup loader + structure-aware HTML splitting.

## Stack

```
Loader      : Custom BeautifulSoup Loader
Splitter    : HTMLHeaderTextSplitter
Embedding   : E5
Vector DB   : Chroma
Retriever   : Similarity
Prompt      : Citation Prompt
LLM         : Gemini
```

## Example corpora

Python docs, FastAPI docs, LangGraph docs

## New concept

- HTML hierarchy (h1/h2 → section → chunk)

## Article

- [ ] `01-html-documentation.md`

## Code

`loaders/web.py` (`WebLoader` — extend for generic sites), `splitters/` (add `html_header.py`), `embeddings/e5.py`

## Notebook

`NoteBooks/Project-05-HTML-Documentation/01-html-documentation.ipynb`
