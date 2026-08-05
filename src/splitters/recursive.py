from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:

    def __init__(self, chunk_size:int=1000,chunk_overlap:int=200):
        """Initialize the document processor with chunk size and chunk overlap. """
        self.text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_docs(self,docs:List[Document])->List[Document]:
        return self.text_splitter.split_documents(docs)