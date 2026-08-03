> Source notebook: `NoteBooks/Project-05-HTML-Documentation/01-html-documentation.ipynb`


---

# Project 05 — HTML Documentation

**Goal:** A custom BeautifulSoup loader + structure-aware HTML splitting.

```
Loader      : WebBaseLoader + custom BeautifulSoup loader
Splitter    : HTMLHeaderTextSplitter (h1/h2 → sections)
Embedding   : E5 (local)
Vector DB   : Chroma
Retriever   : Similarity Search (Top-K)
Prompt      : Citation Prompt (with source URLs)
LLM         : Gemini
```

Project 04 split Markdown by its headings. Project 05 does the same for HTML:
web pages already have a hierarchy (`h1` page title, `h2` section headings),
and `HTMLHeaderTextSplitter` turns that hierarchy into chunks — a whole `<h2>`
section becomes one retrievable unit.

Learn:

* Fetching a real docs page into Documents (`WebBaseLoader`)
* Rolling your own tiny BeautifulSoup loader
* HTML hierarchy → section chunks
* Citation prompts that cite source URLs

---

## 0 · Setup — environment & keys

Same key setup as before: `GOOGLE_API_KEY` lives in `.env` and is only needed
at the final Answer step. HTML parsing needs `beautifulsoup4` (+ `lxml` as a
parser backend) — optional installs for THIS project only.

---

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY", "")
if api_key:
    print(f"GOOGLE_API_KEY found: {api_key[:4]}…{api_key[-2:]} ({len(api_key)} chars)")
else:
    print("GOOGLE_API_KEY missing — copy .env.example to .env and add your key.")
```

---

```python
# Optional deps for THIS project only.
%pip install beautifulsoup4 lxml
```

---

```python
import requests
from bs4 import BeautifulSoup

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
```

---

## 1 · Load — fetching a real docs page

We fetch a real documentation page: the "More Control Flow Tools" chapter of
the Python tutorial — a stable URL with a rich `h1`/`h2` structure.
`WebBaseLoader` is the one-liner path: it downloads the page and extracts the
visible text into `Document`s, with the URL saved in `metadata["source"]`.

**Fallback note:** if the network is unavailable (or the site blocks the
request), save the page once (`curl -o Data/controlflow.html
https://docs.python.org/3/tutorial/controlflow.html`) and load it from disk —
the custom loader we build next reads from a saved file just as easily.

---

```python
url = "https://docs.python.org/3/tutorial/controlflow.html"

loader = WebBaseLoader(url)
html_docs = loader.load()

print(len(html_docs))
print(html_docs[0].metadata)
print(html_docs[0].page_content[:150])
```

---

`WebBaseLoader` is convenient, but it gives you extracted *text* — it has
already thrown away the HTML markup. That matters, because
`HTMLHeaderTextSplitter` (the next block) needs the **raw HTML** to see the
`h1`/`h2` structure. So we also build the project's "custom BeautifulSoup
loader": fetch the page ourselves with `requests`, and parse it with
`BeautifulSoup`.

This is the same idea as `loaders/web.py` in the repo — you control exactly
what gets extracted. Here we keep only the `<div role="main">` content and drop
the site navigation.

---

```python
resp = requests.get(url, timeout=20)
resp.raise_for_status()
raw_html = resp.text
print(f"Fetched {len(raw_html):,} chars of raw HTML from {url}")
```

---

```python
soup = BeautifulSoup(raw_html, "html.parser")

main = soup.find("div", role="main") or soup.body
text = main.get_text("\n", strip=True)

custom_doc = Document(
    page_content=text,
    metadata={
        "source": url,
        "title": soup.title.string if soup.title else url,
    },
)
print(len(text), "chars of text extracted")
print(custom_doc.metadata)
```

---

## 2 · Split — HTML hierarchy → sections

`HTMLHeaderTextSplitter` reads the raw HTML and splits on HTML headings. Each
returned `Document` holds one section, and the metadata records the header path
— `Header 1: More Control Flow Tools`, `Header 2: Defining Functions`. The
first section may be page chrome that appears *before* any heading (look for
its empty `{}` metadata).

One detail worth noticing: the splitter works on an HTML *string* (not on
`Document`s), so it cannot know the source URL. We attach the URL to every
section's metadata afterwards — that is what lets the citation prompt cite
real URLs.

---

```python
splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "Header 1"),
        ("h2", "Header 2"),
    ]
)

sections = splitter.split_text(raw_html)
print(f"Split HTML into {len(sections)} section chunks")
```

---

```python
for section in sections:
    section.metadata["source"] = url

print("source attached to every section:")
print(sections[1].metadata)
```

---

```python
for section in sections[1:3]:
    print(section.metadata)
    print(section.page_content[:120].replace("\n", " "))
    print("---")
```

---

## 3 · Embed — E5

E5 (`intfloat/e5-small-v2`) is a compact embedding model from Microsoft that
runs locally via `langchain-huggingface` / `sentence-transformers`. Like BGE in
Project 04, no API key is needed for embeddings. E5 was trained with `query:` /
`passage:` prefixes for best results — a production tweak you can apply through
`HuggingFaceEmbeddings` configuration.

As always, we embed one sample to confirm the model loads and to read the
vector size (E5 small is 384 dimensions).

---

```python
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-small-v2",
)

vec = embeddings.embed_query("How do I define a function in Python?")
print(f"E5 small-v2 embedding dims: {len(vec)}")
```

---

## 4 · Store — Chroma

Chroma is the persistent vector store introduced in Project 01. It is used here
exactly the same way — the only things that changed in this project are the
loader and the splitter. `Chroma.from_documents` embeds the section chunks and
stores them for similarity search. (Without a `persist_directory`, the store
lives in memory for this session.)

---

```python
vector_store = Chroma.from_documents(
    documents=sections,
    embedding=embeddings,
)
print(f"Chroma store built over {len(sections)} sections")
```

---

## 5 · Retrieve

Same retrieval as before: embed the query, return the `k` nearest sections,
and show the raw similarity score alongside each hit so you can judge how
confident the match is.

---

```python
query = "Which chapter explains the range() function?"
results = vector_store.similarity_search_with_score(query, k=3)

for i, (chunk, score) in enumerate(results, 1):
    src = chunk.metadata.get("source", "?")
    snippet = chunk.page_content[:90].replace("\n", " ")
    print(f"[{i}] score={score:.3f} source={src}\n    {snippet}\n")
```

---

## 6 · Prompt — citation prompt with source URLs

The citation prompt from Project 04, adapted for the web: the model must answer
from the context only and **cite the source URL** for each claim. Because every
section carries `source` in its metadata, we can tag each chunk with its URL
before it goes into the prompt — the same "source-aware packaging" idea, now
with addresses you can open instead of file paths.

---

```python
prompt = ChatPromptTemplate.from_template(
    """You are an HTML documentation assistant.

Answer the question using ONLY the provided context.
For every claim, cite the source URL it came from.

Context:
{context}

Question:
{question}

Answer (with source URLs):
"""
)
```

---

```python
def pack_for_prompt(docs):
    """Join chunks, tagging each one with its source URL."""
    return "\n\n".join(
        f"[source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

context = pack_for_prompt([r[0] for r in results])
print(f"{len(context)} chars of context, each chunk tagged with its URL")
```

---

```python
messages = prompt.invoke({"context": context, "question": query})
print(messages.content[:300])
```

---

## 7 · Answer — Gemini

The final block is the LLM — the same `gemini-2.5-flash` model as every project
so far. Feed it the tagged context and the question; the answer comes back with
source URLs you can open to verify.

---

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

answer = llm.invoke(messages)
print(answer.content)
```

---

## 8 · Try it yourself

Swap the corpus or the question. Good follow-ups: load a different stable docs
page (the tutorial index, FastAPI, LangGraph) or ask something that lives deep
inside an `<h2>` section — the metadata will show you which section answered.

---

```python
query2 = "Where do I learn about the Python standard library?"
for hit in vector_store.similarity_search(query2, k=2):
    print(hit.metadata.get("source"), "→", hit.page_content[:70].replace("\n", " "))
```

---

## What you should notice

* `HTMLHeaderTextSplitter` consumes the **raw HTML string**, not the text that
  `WebBaseLoader` already extracted — so the custom loader is not optional
  decoration, it is what feeds the splitter.
* Each section chunk carries `Header 1` / `Header 2` metadata — the HTML
  equivalent of Project 04's `H1`/`H2`.
* The source URL is NOT in the splitter output — we attached it manually.
* E5 embeddings run locally (384 dims); only the Gemini answer needs a key.
* The citation prompt now cites URLs, turning a retrieved section into a
  verifiable link.
* One pipeline, one swap: Loader + Splitter changed; Store, Embed and Retrieve
  stayed identical to the baseline.

---

## Exercises

1. Add `("h3", "Header 3")` to `headers_to_split_on` and re-index. How many more
   (and smaller) sections do you get?
2. Write a custom loader that strips the site navigation (exclude `<nav>`
   elements) before splitting, and compare the retrieved snippets.
3. Fetch a different stable docs page (e.g. https://fastapi.tiangolo.com/) and
   run the pipeline on it unchanged. What breaks, and what survives?
4. Persist the Chroma store with `persist_directory="chroma_langchain_db"` and
   reload it in a fresh kernel.
