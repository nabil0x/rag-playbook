"""RAGPipeline — the endgame assembly (Project 17).

Wires the six pipeline blocks into one pluggable question-answering system.
Every block is swappable without touching the others, so this class is the
"final boss" of the curriculum: by Project 17 you build exactly this from
the components you implemented along the way.

See Topics/Project-17-Modular-RAG-Framework/README.md.
"""

from prompts.basic import BasicPrompt


class RAGPipeline:
    """Assemble loader -> splitter -> embedder -> store -> retriever -> llm."""

    def __init__(
        self,
        loader,
        splitter,
        embedder,
        db,
        retriever,
        llm,
        prompt=None,
    ):
        self.loader = loader
        self.splitter = splitter
        self.embedder = embedder
        self.db = db
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt if prompt is not None else BasicPrompt()

    def ingest(self) -> int:
        """Load the loader's source, split it, embed the chunks, store them.

        Returns the number of chunks added to the vector store.
        """
        docs = self.loader.load()
        chunks = self.splitter.split_docs(docs)
        embeddings = self.embedder.embed_documents([c.page_content for c in chunks])
        self.db.add(chunks, embeddings=embeddings)
        return len(chunks)

    def ask(self, question: str) -> str:
        """Retrieve relevant chunks, build the prompt, return the answer."""
        context_docs = self.retriever.retrieve(question)
        context = "\n\n".join(d.page_content for d in context_docs)
        prompt_text = self.prompt.format(context=context, question=question)
        return self.llm.invoke(prompt_text)


if __name__ == "__main__":
    # End-to-end smoke test. Uses only installed dependencies:
    # CSVLoader (stdlib csv) + Gemini (needs GOOGLE_API_KEY in .env).
    from loaders.csv_loader import CSVLoader
    from splitters.recursive import DocumentProcessor
    from embeddings.gemini import GeminiEmbedding
    from vectordb.chroma import ChromaVectorStore
    from retrieval.similarity import SimilarityRetriever
    from llms.gemini import GeminiLLM

    embedder = GeminiEmbedding()
    db = ChromaVectorStore(embedding=embedder)
    pipeline = RAGPipeline(
        loader=CSVLoader("Data/sample.csv"),
        splitter=DocumentProcessor(),
        embedder=embedder,
        db=db,
        retriever=SimilarityRetriever(db),
        llm=GeminiLLM(),
    )
    n = pipeline.ingest()
    print(f"ingested {n} chunks")
    print(pipeline.ask("What is the sample about?")[:400])
