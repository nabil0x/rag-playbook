# Project 02 — PDF Knowledge Base

> **Goal:** Swap the loader only — PDFs instead of web pages.

## Stack

```
Loader      : PyPDFLoader
Splitter    : RecursiveCharacterTextSplitter
Embedding   : Gemini
Vector DB   : Chroma
Retriever   : Similarity
Prompt      : Basic
LLM         : Gemini
```

## Example sources

University regulations, research papers, books, course notes

## New concept

- PDF parsing (text layer extraction, multi-page documents)

## Article

- [ ] `01-pdf-knowledge-base.md`

## Code

`src/loaders/pdf.py` (`PDFLoader`) — implement with `PyPDFLoader`

## Notebook

`NoteBooks/Projects/Project-02-PDF-Knowledge-Base/01-pdf-knowledge-base.ipynb`
