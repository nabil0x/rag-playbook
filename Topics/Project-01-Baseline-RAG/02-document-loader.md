> Source notebook: `NoteBooks/Projects/Project-01-Baseline-RAG/02-document-loader.ipynb`


---

# Project 01 · Notebook 2 — Build Your Own Document Loader

> **Goal:** Understand exactly what a "document loader" does by writing one from
> scratch — `requests` + BeautifulSoup → clean text → a LangChain `Document` →
> chunks.

The first notebook used LangChain's `WebBaseLoader`. Here you open the hood: a
loader is just *code that turns a URL into structured text*. By the end you will
hold a `Document` you built yourself, plus chunks you can feed straight into any
embedding model and vector store.

## What you will build, step by step

| Step | What happens                                   | Output you should see       |
|------|------------------------------------------------|-----------------------------|
| 0    | Load `.env`                                    | `True`                      |
| 1    | Fetch a page & extract the article with bs4    | first ~100 chars of the doc |
| 2    | Wrap the text in a LangChain `Document`        | a `Document` with metadata  |
| 3    | Split into overlapping chunks                  | a list of chunk `Document`s |
| 4    | Inspect the chunks                             | chunk count + one sample    |
| 5    | Your turn — empty cell                         | your own experiment         |

---

## 0 · Setup — environment

**WHAT:** Loads `.env` so any API key you need later in the project is
available.

**WHY:** Even though this notebook only fetches and chunks (no API calls yet),
the pipeline you are building will need `GOOGLE_API_KEY` in the next step, and a
dedicated setup cell keeps every notebook self-contained.

**WHAT TO EXPECT:** `True` if `.env` was found and read.

---

```python
from dotenv import load_dotenv
load_dotenv()
```

---

```text
True
```

---

## 1 · Load — fetch the page and extract its article text

**WHAT:** `requests.get(url)` downloads the raw HTML, BeautifulSoup parses it,
and a loop walks the `h1`–`h4`, `p`, `pre` and `li` tags inside the article body
(skipping nested duplicates) and joins them into one text blob. The first 100
characters are printed.

**WHY (for RAG):** This is the real job of a loader: **raw HTML in, clean text
out**. RAG quality depends on this step — grab the page's content area and drop
the navigation, ads and scripts before anything downstream sees the text.

**WHAT TO EXPECT:** A short print-out with the opening of the DEV.to article.

---

```python
import requests
from bs4 import BeautifulSoup

url = "https://dev.to/gautamvhavle/building-production-rag-systems-from-zero-to-hero-2f1i"

html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

# DEV.to stores the core article text inside div#article-body
article_body = soup.find(id="article-body") or soup.find("article")

texts = []
# Include "li" along with headers, paragraphs, and code blocks
for tag in article_body.find_all(["h1", "h2", "h3", "h4", "p", "pre", "li"]):
    # Avoid picking up nested elements inside pre/li that get matched twice
    if tag.parent.name in ["p", "li", "pre"]:
        continue
    texts.append(tag.get_text("\n", strip=True))

document = "\n\n".join(texts)
print(document[:100])
```

---

```text
What I learned building RAG systems from scratch—and how you can too

The Journey That Changed How I
```

---

## 2 · Wrap the text in a LangChain `Document`

**WHAT:** The plain string is wrapped in
`Document(page_content=document, metadata={"source": url})` and put in a list
`docs`.

**WHY (for RAG):** Every later block (splitter, embedder, store) expects the
LangChain `Document` shape: `page_content` (the text) plus `metadata`
(provenance — where the text came from). Metadata is how you can cite sources in
answers later.

**WHAT TO EXPECT:** A one-element list `docs` holding your document.

---

```python
from langchain_core.documents import Document

doc = Document(
    page_content=document,
    metadata={"source": url}
)

docs = [doc]
```

---

## 3 · Split — chunk the document

**WHAT:** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`
breaks the document into ~1000-character chunks that overlap by 200 characters.

**WHY (for RAG):** Retrieval is chunk-based. Keeping chunks focused (~1000
chars) makes each retrieved piece readable and relevant; the overlap preserves
sentences that straddle a cut.

**WHAT TO EXPECT:** `splits` — a list of chunk `Document`s, each still carrying
the source metadata.

---

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splits=text_splitter.split_documents(docs)
```

---

## 4 · Inspect the chunks — verify the splitter's work

**WHAT:** `len(splits)` prints the chunk count, and `print(splits[1])` shows one
full chunk including its metadata.

**WHY (for RAG):** Always eyeball your chunks once. If chunks are empty, cut
mid-sentence, or carry the wrong metadata, nothing downstream can recover the
quality loss.

**WHAT TO EXPECT:** A number (dozens of chunks) and a printed chunk with
`page_content='...'` and `metadata={'source': ...}`.

---

```python
len(splits)
```

---

```text
42
```

---

```python
print(splits[1])
```

---

```text
page_content='Here's what I wish someone had told me before I started, and what I've learned along the way.

Why RAG? The Problem I Kept Running Into

During my course, the instructor kept hammering home one point: LLMs are amazing at reasoning, terrible at remembering. I nodded along, but I didn't
really
get it until my first project.

I was building a chatbot for a company's internal documentation. Simple, right? Feed GPT-4 a question, get an answer. Except:

It hallucinated constantly.
Made up API endpoints that didn't exist. Confidently cited documentation sections that were never written.

It didn't know about latest updates.
We could have shipped a major feature last week. The model? Clueless.

That's when RAG clicked. Instead of expecting the model to memorize everything, I'd give it a search engine. When someone asks a question, search the docs first, then feed the relevant content to the model.

Suddenly: no hallucinations, always up-to-date, and token utilization also was optimized.' metadata={'source': 'https://dev.to/gautamvhavle/building-production-rag-systems-from-zero-to-hero-2f1i'}
```

---

## 5 · Your turn — experiment in the empty cell

The last cell is intentionally empty. Try loading a second URL, tweaking the
splitter parameters, or adding your own `title` to the metadata — then inspect
the result here.

---

## What you should notice

- **A loader is just plain Python.** `requests` + BeautifulSoup and a loop —
  "document loader" is a fancy name for *HTML → clean text*.
- **The `Document` shape is the handoff.** Everything from here on (splitter →
  embedder → store) consumes `page_content` + `metadata`, so wrapping text
  correctly is the glue of the whole pipeline.
- **Metadata is provenance.** Attaching `source` now means answers can cite
  where the text came from — a requirement for trustworthy RAG.
- **Chunking is observable.** A couple of print statements tell you whether the
  split is healthy before you spend tokens embedding it.

---

## Exercises

1. **Add metadata.** Extend the `Document` with a `title` key (scrape the
   `<title>` tag) and verify it shows up in `print(splits[1])`.
2. **Change the chunk size.** Split with `chunk_size=500, chunk_overlap=50` and
   compare the chunk count and a sample chunk. Which one reads better?
3. **Load another page.** Point the same code at a second article (or any public
   page) and confirm the extracted text is clean — no nav, no scripts.
