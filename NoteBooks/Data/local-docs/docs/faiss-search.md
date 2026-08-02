# FAISS similarity search

FAISS (Facebook AI Similarity Search) is a library for fast similarity search
over dense vectors. It is the workhorse behind many vector databases.

## What it does

Given an index of vectors (one per chunk) and a query vector, FAISS returns the
k nearest neighbours — the chunks most similar to the query. Everything lives in
memory, so lookups are instant and fully offline.

## Persistence

FAISS in LangChain is an in-memory index. To keep it between sessions, save and
load it explicitly:

    vector_store.save_local("faiss_index/")
    vector_store = FAISS.load_local("faiss_index/", embeddings,
                                    allow_dangerous_deserialization=True)

## Querying

    hits = vector_store.similarity_search_with_score(query, k=3)

`similarity_search_with_score` also returns a distance per hit — lower means
closer, i.e. more relevant to the query.
