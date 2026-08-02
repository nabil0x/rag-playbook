# BGE embeddings

BGE (BAAI General Embedding) is a family of open-source embedding models
trained by the Beijing Academy of Artificial Intelligence. They convert text
into vectors: sentences that mean similar things end up near each other in
vector space.

## Why use BGE?

- Runs locally via `sentence-transformers` — no API key, no per-query cost.
- `BAAI/bge-small-en` is a small model (about 130 MB) that returns
  384-dimension vectors, fast enough to embed thousands of documents on a
  laptop.
- It is a general-purpose English model, well suited to documentation search.

## Usage notes

- Enable `encode_kwargs={"normalize_embeddings": True}` so cosine similarity
  behaves well.
- The model is downloaded once into `~/.cache/huggingface` and cached for all
  later runs.

## In LangChain

Use `HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en")` from
`langchain_huggingface`, then pass it to a vector store the same way you would
pass Gemini embeddings.
