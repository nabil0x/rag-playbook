"""Web document loader.

Loader block: every loader returns `List[Document]` — nothing else in the
pipeline changes when you swap this block.
See Topics/Project-01-Baseline-RAG/README.md and
Topics/Project-16-Build-Without-LangChain/README.md.
"""

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


class WebLoader:
    """Fetch a web page and return its main text as a list of Documents."""

    def __init__(self, url: str):
        self.url = url

    def _fetch_text(self) -> str:
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, "html.parser")

        # DEV.to stores the core article text inside div#article-body
        article_body = soup.find(id="article-body") or soup.find("article")

        texts = []
        # Include "li" along with headers, paragraphs, and code blocks
        for tag in article_body.find_all(["h1", "h2", "h3", "h4", "p", "pre", "li"]):
            # Avoid picking up nested elements inside p/li/pre that get matched twice
            if tag.parent.name in ["p", "li", "pre"]:
                continue
            texts.append(tag.get_text("\n", strip=True))

        return "\n\n".join(texts)

    def load(self) -> list[Document]:
        text = self._fetch_text()
        return [Document(page_content=text, metadata={"source": self.url})]


if __name__ == "__main__":
    url = "https://dev.to/gautamvhavle/building-production-rag-systems-from-zero-to-hero-2f1i"
    docs = WebLoader(url).load()
    print(docs[0].page_content)
