> Source notebook: `NoteBooks/Projects/Project-01-Baseline-RAG/01-langchain-intro.ipynb`


---

# Project 01 · Notebook 1 — LangChain Intro

> **Goal:** Get comfortable with the core LangChain building blocks — document
> loaders, text splitters, embeddings, a vector store, a retriever, a prompt and
> an LLM — and watch them snap together into a RAG chain.

RAG in one sentence: *give the LLM a small set of relevant passages, then let it
answer from those passages instead of from memory.* This notebook shows every
piece of that pipeline using real LangChain components against one blog article.

## What you will build, step by step

| Step | Block    | Component used                                          |
|------|----------|---------------------------------------------------------|
| 1    | Load     | `WebBaseLoader` + `bs4`                                 |
| 2    | Split    | `RecursiveCharacterTextSplitter`                        |
| 3    | Embed    | `GoogleGenerativeAIEmbeddings`                          |
| 4    | Store    | `Chroma` (persistent)                                   |
| 5    | Retrieve | vector store as retriever (top-k)                       |
| 6    | Prompt   | context + question template                             |
| 7    | Answer   | `ChatGoogleGenerativeAI` + `StrOutputParser`            |
| 8    | Measure  | token counting with `tiktoken`, cosine similarity       |

You need a `GOOGLE_API_KEY` in your `.env` file — both the embedding model and
the LLM call the Gemini API.

---

## 0 · Setup — environment & imports

**WHAT:** Loads the `.env` file so your `GOOGLE_API_KEY` reaches the Gemini API,
and imports every LangChain class used in this notebook (loader, splitter,
embeddings, `Chroma`, prompt, chat model, output parser).

**WHY:** Every RAG pipeline has two ingredients: *your data* and *an LLM that can
read it*. This cell imports the tools for both halves, so all later cells stay
short and focused on one idea.

**WHAT TO EXPECT:** Running the cell returns `True` — that is `load_dotenv()`
confirming your `.env` was read. If you see `False`, create a `.env` file in the
repo root containing `GOOGLE_API_KEY=...`.

---

```python
import os
from dotenv import load_dotenv
load_dotenv()
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
```

---

```text
True
```

---

> **Note on the next cell.** The original cell below is intentionally *all in
> one* — a single running example that walks the entire pipeline in order:
> load → split → embed → store → retrieve → prompt → answer, followed by two
> measurement exercises (token counting and cosine similarity) and a re-index
> using a token budget. The sections below break that flow into small steps so
> you can read one concept at a time, then watch the cell demonstrate them all.

---

## 1 · Load — turn a web page into a `Document`

**WHAT:** `WebBaseLoader` fetches the Lilian Weng "LLM Powered Autonomous
Agents" post. A `bs4.SoupStrainer` keeps only the `post-content`,
`post-title` and `post-header` elements, so you get the article text instead of
the whole HTML page.

**WHY (for RAG):** Loading is the first block of the pipeline — if the loader
keeps junk (nav, scripts, ads), every later step is polluted. Scoping the parser
to the content area is the cheapest quality win in all of RAG.

**WHAT TO EXPECT:** `docs` — a list with one LangChain `Document` whose
`page_content` holds the article text.

---

## 2 · Split — cut the document into overlapping chunks

**WHAT:** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`
slices the article into ~1000-character pieces that overlap by 200 characters.

**WHY (for RAG):** LLMs read bounded contexts and retrievers return *chunks*, not
whole books. Chunks keep each retrieved piece focused on one idea; the overlap
stops a relevant thought from being chopped in half at a boundary.

**WHAT TO EXPECT:** `splits` — a list of chunk `Document`s. A long page yields
many chunks rather than one giant document.

---

## 3 · Embed — turn text into vectors

**WHAT:** `GoogleGenerativeAIEmbeddings` converts a piece of text into a list of
numbers (a vector). The `gemini-embedding-2-preview` model is used here.

**WHY (for RAG):** Machines do not "read" — they compare numbers. Embeddings
place *similar* text near each other in vector space, which is what lets the
retriever find relevant chunks by proximity instead of keyword matching.

**WHAT TO EXPECT:** An embeddings object you can call with
`embeddings.embed_query(...)` to get a vector (hundreds of floats long).

---

## 4 · Store — persist the vectors in Chroma

**WHAT:** A persistent Chroma collection (`agent_blog`) is created at
`./chroma_langchain_db` and the chunk vectors are added with
`vectorstore.add_documents(splits)`.

**WHY (for RAG):** A vector database is the pipeline's *memory*. Persisting the
embeddings means you index the data once and search it many times — and the
folder is regenerable, so you can rebuild it any time.

**WHAT TO EXPECT:** The Chroma collection is created on disk; `as_retriever()`
returns an object you can `.invoke(...)` or pipe into a chain.

---

## 5 · Retrieve — find the top-k chunks for a question

**WHAT:** `vectorstore.as_retriever()` wraps the store in a retriever, and
`retriever.invoke(question)` returns the most relevant chunks.

**WHY (for RAG):** Retrieval is what makes the model *grounded*. Instead of
guessing, the LLM gets a handful of relevant passages to answer from — this is
the "R" in RAG.

**WHAT TO EXPECT:** A short list of `Document`s ranked by similarity to the
query "What is Task Decomposition?".

---

## 6 · Prompt — package context + question

**WHAT:** A prompt template (pulled from LangSmith's `rlm/rag-prompt` here) is
filled with the retrieved context and the user's question. The `format_docs`
helper joins the chunks with blank lines so the model can read them.

**WHY (for RAG):** The prompt is the *contract* between the retriever and the
LLM: "here is the evidence, answer only from it". How you word this sentence
directly controls hallucination and answer quality.

**WHAT TO EXPECT:** A `ChatPromptTemplate` you can `.invoke({...})` to produce
the final message the model will read.

---

## 7 · Answer — run the chain end to end

**WHAT:** The pieces snap together with `|`: retriever → `format_docs` → prompt →
`ChatGoogleGenerativeAI` → `StrOutputParser`, then `rag_chain.invoke(question)`.

**WHY (for RAG):** This is the whole point — a *chain* that takes a raw question
in and returns a grounded answer out. `StrOutputParser` unwraps the model's
message into plain text.

**WHAT TO EXPECT:** A natural-language answer to "What is Task Decomposition?"
written from the retrieved chunks.

---

## 8 · Token counting — why context has a budget

**WHAT:** `tiktoken` counts the tokens in a string via
`num_tokens_from_string` — first for a short question, then for a longer
document.

**WHY (for RAG):** Context windows are finite and tokens are what you pay for.
Token counting is how you reason about chunk sizes: chunks that are too big
overflow the window, chunks that are too small starve the model of context.

**WHAT TO EXPECT:** A small integer — the token count of the sample question.

---

## 9 · Cosine similarity — "similar text" in numbers

**WHAT:** The query and a short document are embedded, then compared with
`cosine_similarity` (dot product divided by the product of the norms). The score
runs from -1 to 1; closer to 1 means more similar.

**WHY (for RAG):** This is *exactly* what the vector store computes internally
when it retrieves. Understanding cosine similarity demystifies "semantic
search" — it is just measuring the angle between vectors.

**WHAT TO EXPECT:** A similarity score printed for the query vs. the document.

---

## 10 · Re-indexing with a token budget

**WHAT:** A second indexing run uses a token-based length function
(`length_function=lambda text: len(enc.encode(text))`) with
`chunk_size=300, chunk_overlap=50`, then a retriever with
`search_kwargs={"k": 1}`.

**WHY (for RAG):** Character counts and token counts are not the same thing.
Splitting by tokens gives tight control over what the model actually pays to
read; `k` controls how many chunks get retrieved. Smaller `k` = cheaper, narrower
answers.

**WHAT TO EXPECT:** `len(docs) == 1` — exactly one chunk retrieved, then printed
together with its content and metadata.

---

```python
#Load Documents

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs=loader.load()
# Split

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splits=text_splitter.split_documents(docs)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
import chromadb

client = chromadb.PersistentClient(path="./chroma_langchain_db")
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 1. Provide the model string WITHOUT the 'models/' prefix
vectorstore = Chroma(
    collection_name="agent_blog",
    embedding_function=embeddings,
)

vectorstore.add_documents(splits)

retriever = vectorstore.as_retriever()

# LLM
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
# Create a LangSmith API in Settings > API Keys
# Make sure API key env var is set:
# import os; os.environ["LANGSMITH_API_KEY"] = "<your-api-key>"
from langsmith import Client

client = Client()

prompt = client.pull_prompt(
    "rlm/rag-prompt",
    dangerously_pull_public_prompt=True,
)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("What is Task Decomposition?")
docs = retriever.invoke("What is Task Decomposition?")

context = format_docs(docs)

prompt_value = prompt.invoke({
    "context": context,
    "question": "What is Task Decomposition?"
})

response = llm.invoke(prompt_value)

answer = StrOutputParser().invoke(response)

print(answer)


# Documents
question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num_tokens_from_string(question, "cl100k_base")
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embd = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)
len(query_result)
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print("Cosine Similarity:", similarity)
#### INDEXING ####

# Load blog
import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

enc = tiktoken.get_encoding("cl100k_base")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=lambda text: len(enc.encode(text)),
)
token_splits = text_splitter.split_documents(blog_docs)
# Index
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents=token_splits,
                                    embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview"))

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
docs = retriever.invoke("What is Task Decomposition?")
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
prompt
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0
)
# Chain
chain = prompt | llm
chain.invoke({"context": format_docs(docs), "question": "What is Task Decomposition?"})
from langsmith import Client

client = Client()

prompt = client.pull_prompt(
    "rlm/rag-prompt",
    dangerously_pull_public_prompt=True,
)
prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")
```

---

## What you should notice

- **The pipeline is a data flow.** Every stage passes `Document`s / chunks down
  the line, and each stage only transforms the previous stage's output.
- **Embeddings are the hidden contract.** The loader, splitter, store and
  retriever all "speak" through vectors: everything before the store is about
  making good chunks, everything after is about matching them.
- **Chunking trades focus against cost.** Small chunks (300 tokens) give precise
  retrieval but little context; large chunks (1000 chars) give more context per
  hit. Both are valid — the trade-off is yours to make.
- **Similarity is measurable.** Cosine similarity is the same operation the
  vector store runs during retrieval — you just ran the retriever's core math by
  hand.
- **`k` is a dial, not a constant.** With `k=1` the answer is built from a single
  chunk; with `k=3` or more the model sees more evidence and pays more tokens.

---

## Exercises

1. **Change the question.** Ask the retriever a different question about the
   same article (for example about how agents plan tasks) and compare the chunks
   it returns. Do the retrieved chunks match the topic?
2. **Tune the splitter.** Try `chunk_size=500, chunk_overlap=50` and re-index.
   How does the number of chunks change? Does retrieval quality change?
3. **Compare similarity scores.** Embed two unrelated sentences and two related
   ones, and compare their cosine similarities. Which pair scores higher?
