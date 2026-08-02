# Project 17 — Modular RAG Framework

> **Goal:** Implement your own architecture — pluggable interfaces for every block.

## Structure

```
loaders/  splitters/  embeddings/  vectordb/  retrieval/  prompt/  llm/
```

## Interfaces

```python
class BaseLoader:
    def load(self):
        raise NotImplementedError

class BaseSplitter:
    def split(self, docs):
        raise NotImplementedError

class BaseEmbedding:
    def embed(self, text):
        raise NotImplementedError
```

## Wiring

```python
pipeline = RAGPipeline(
    loader=WebLoader(),
    splitter=SemanticSplitter(),
    embedder=BGEEmbedding(),
    vector_db=FAISSStore(),
    retriever=MMRRetriever(),
    llm=GeminiLLM(),
)
```

## Article

- [ ] `01-modular-rag-framework.md`

## Code

This is what `loaders/`, `splitters/`, `embeddings/`, `vectordb/`, `retrieval/`,
`prompts/`, `llms/` + `main.py` become — every block in the repo root is a
working implementation, and the interface each one follows is the contract
you'd enforce here.

## Notebook

`NoteBooks/Project-17-Modular-RAG-Framework/01-modular-rag-framework.ipynb`
