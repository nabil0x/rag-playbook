loader = PDFLoader()
splitter = SemanticSplitter()
embedder = BGEEmbedding()
db = FAISSVectorStore()
retriever = MMRRetriever(db)
llm = GeminiLLM()

pipeline = RAGPipeline(
    loader,
    splitter,
    embedder,
    db,
    retriever,
    llm,
)

pipeline.ask("What is Task Decomposition?")