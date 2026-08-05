# Project 09 — Parent-Child Retrieval

> **Goal:** Retrieve on small chunks, return large parent context.

## Stack

```
Loader      : PDF
Splitter    : Parent + Child
Embedding   : BGE
Vector DB   : Chroma
Retriever   : ParentDocumentRetriever
Prompt      : Basic
LLM         : Gemini
```

## Learn

- Large chunks (context)
- Small chunks (precision)
- Recover the parent after retrieval

## Article

- [ ] `01-parent-child-retrieval.md`

## Code

`src/retrieval/` (add `parent_child.py`)

## Notebook

`NoteBooks/Project-09-Parent-Child-Retrieval/01-parent-child-retrieval.ipynb`
