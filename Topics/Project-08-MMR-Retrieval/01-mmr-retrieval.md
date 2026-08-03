> Source notebook: `NoteBooks/Project-08-MMR-Retrieval/01-mmr-retrieval.ipynb`


---

# Project 08 — MMR Retrieval

**Goal:** Replace plain similarity with Maximum Marginal Relevance.

```
Loader      : Web Loader (Project Gutenberg)
Splitter    : RecursiveCharacterTextSplitter
Embedding   : Gemini Embedding
Vector DB   : Chroma
Retriever   : Plain Similarity  →  SWAPPED TO  MMR
Prompt      : Basic Context + Question
LLM         : Gemini 2.5 Flash
```

Learn:

- Why plain top-k similarity can return near-duplicate chunks
- What Maximum Marginal Relevance (MMR) is and the trade-off it makes
- How `lambda_mult` tunes "diverse" vs "relevant" in MMR

---

## 0 · Setup — environment & keys

Same setup as every project: load `.env`, verify the masked Google key, import
the real libraries. Everything used here is already in `requirements.txt` — no
optional installs this time. The only thing we swap later is the retriever.

---

```python
from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("GOOGLE_API_KEY", "")
if key:
    print("GOOGLE_API_KEY set:", key[:4] + "…" + key[-4:])
else:
    print("GOOGLE_API_KEY missing — copy .env.example to .env and add your key.")
```

---

### Imports

Grouped, one cell. Matches the reference notebook exactly. No retriever import
needed — the MMR retriever is created from the vector store itself with
`as_retriever(...)`, which is the whole point: the retriever is a *wrapper*
around the store, so nothing else in the pipeline changes.

---

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
```

---

## 1 · Load

We use the same public-domain Gutenberg story as Project 01. A story is a good
corpus for this project because it repeats themes, characters and events — so
plain similarity search will happily return chunks that all say roughly the same
thing. That repetition is exactly what MMR is built to fix.

---

```python
url = "https://www.gutenberg.org/cache/epub/79247/pg79247-images.html"

loader = WebBaseLoader(url)
docs = loader.load()

print("documents:", len(docs))
print("first chars:", docs[0].page_content[:60].replace(chr(10), " "))
```

---

```python
print("metadata:", docs[0].metadata)
print("characters in doc:", len(docs[0].page_content))
```

---

## 2 · Split

Same splitter as the baseline: `chunk_size=1000`, `chunk_overlap=200`. These
chunks are what get embedded and searched. With a ~1000-char window the story
produces several dozen chunks that discuss overlapping content — plenty of
near-duplicates for MMR to skip.

---

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)

print("chunks:", len(chunks))
```

---

```python
print("sample chunk:")
print(chunks[0].page_content[:150].replace(chr(10), " "))
print("…")
```

---

## 3 · Embed

Embeddings turn text into vectors so the store can rank chunks by similarity:
text that means similar things lands near each other in vector space. We embed
once and reuse the vectors for the whole notebook.

---

```python
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

sample_vector = embeddings.embed_query("a quick check")
print("vector dimension:", len(sample_vector))
```

---

## 4 · Store

Chroma, the default store. No `persist_directory` is passed, so the collection
lives in memory for this notebook (regenerable artifact — same as Project 01).

---

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)
print("chunks stored:", vector_store._collection.count())
```

---

## 5 · Retrieve — plain similarity first

The baseline retriever is plain top-k similarity: embed the question, find the
`k` chunks whose vectors are closest, return them. We keep it as `baseline` so
we can compare it against MMR on the very same store.

Run this with a question about a topic the story mentions many times — e.g. the
moral, poker, or a recurring character — so the corpus has many similar chunks.

---

```python
baseline = vector_store.as_retriever(
    search_kwargs={"k": 4},
)

QUERY = "What is the moral of the story?"
base_hits = baseline.invoke(QUERY)

print(f"baseline returned {len(base_hits)} chunks")
for i, d in enumerate(base_hits):
    print(f"{i+1}. {d.page_content[:70].replace(chr(10), ' ')}…")
```

---

### Spot the problem

Look at the chunk *snippets* above. With plain similarity, several of the top-4
often say the same thing — the story hammers the same point again and again, and
similarity only cares about "close to the question", never about "already shown
the user something like this". You get redundant context and the LLM reads the
same idea four times.

MMR attacks this directly.

---

### The swap: Maximum Marginal Relevance

MMR re-scores the candidates with two terms at once:

```
score = λ · similarity(query, doc)          ← keep it relevant
        − (1−λ) · max similarity(doc, chosen) ← penalize redundancy
```

After picking the most relevant chunk, MMR re-ranks the rest so that each new
pick is both relevant **and** the least similar to chunks already chosen. The
only change in the pipeline is `search_type="mmr"` on `as_retriever` — the
store, the embeddings and the question stay identical.

---

```python
mmr = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
)

mmr_hits = mmr.invoke(QUERY)

print(f"MMR returned {len(mmr_hits)} chunks")
for i, d in enumerate(mmr_hits):
    print(f"{i+1}. {d.page_content[:70].replace(chr(10), ' ')}…")
```

---

### Side-by-side: similarity vs MMR

The two retrievers are asked the exact same question against the same store.
`fetch_k=20` means MMR may look at 20 candidate chunks before picking the final
4 — it has a bigger pool to diversify from. Compare the *overlap*: plain
similarity often repeats itself; MMR spreads the 4 hits over different parts of
the story.

---

```python
def first_words(d):
    return d.page_content.split()[:8]


print("baseline top-4:")
for i, d in enumerate(base_hits):
    print(f"  {i+1}. {' '.join(first_words(d))}…")
print()
print("mmr top-4:")
for i, d in enumerate(mmr_hits):
    print(f"  {i+1}. {' '.join(first_words(d))}…")
```

---

```python
base_bodies = {d.page_content for d in base_hits}
mmr_bodies = {d.page_content for d in mmr_hits}

print("chunks shared by both retrievers:", len(base_bodies & mmr_bodies), "of", len(base_bodies))
```

---

## 6 · The `lambda_mult` knob

`lambda_mult` (the `λ` in the formula) balances the two terms:

- **High** (e.g. 0.9) → trust similarity almost fully → closer to plain top-k.
- **Low** (e.g. 0.5) → penalize redundancy harder → more diverse context.

There is no universal "best" value: technical Q&A usually prefers a higher
`lambda_mult` (you want the exact answer chunk), while summarizing or
question-answering over a story benefits from lower values that pull in different
angles.

---

```python
mmr_focused = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.9},
)

print("lambda_mult = 0.9 (relevance-heavy):")
for d in mmr_focused.invoke(QUERY):
    print("  -", " ".join(first_words(d)), "…")
print()
print("lambda_mult = 0.5 (diversity-heavy):")
for d in mmr_hits:
    print("  -", " ".join(first_words(d)), "…")
```

---

## 7 · Prompt

The prompt is the same Basic "answer from context" template. It packages the
retrieved chunks plus the question. With MMR the context handed to the LLM
covers more distinct parts of the story — which usually gives a more complete,
less repetitive answer.

---

```python
template = """
You are a helpful assistant.

Answer the question using ONLY the provided context.
If the answer is not contained in the context, say:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
print("template ready")
```

---

```python
context = "\n\n".join(doc.page_content for doc in mmr_hits)

messages = prompt.invoke({
    "context": context,
    "question": QUERY,
})
print("prompt length:", len(messages.to_string()))
```

---

## 8 · Answer

Call the LLM with the MMR-built prompt. If you want, re-run the same cell but
replace `mmr_hits` with `base_hits` and compare: the plain-similarity answer is
usually more repetitive, while the MMR answer reads more like a summary of
different scenes.

---

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

response = llm.invoke(messages)
print(response.content)
```

---

## 9 · Try it yourself

Play with the two knobs that change MMR behavior:

1. `k` — how many chunks the answer uses.
2. `fetch_k` — how big the candidate pool is before diversity kicks in.
3. `lambda_mult` — relevance vs diversity.

Also try a question where the story has many similar passages (a recurring
character, the word "poker", a repeated event) and watch the overlap column.

---

```python
probe = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 30, "lambda_mult": 0.5},
)

for q in ["What is the moral of the story?", "Tell me about Magpie"]:
    hits = probe.invoke(q)
    print("Q:", q)
    for d in hits:
        print("  -", " ".join(first_words(d)), "…")
    print()
```

---

## What you should notice

- **Plain similarity repeats itself.** On a corpus with repeated themes, the
  top-4 often contains near-duplicate chunks — same information, re-worded.
- **MMR trades a little relevance for diversity.** The MMR top-4 may not be the
  4 most similar chunks, but they cover more distinct content. That is the whole
  deal: a bit of precision, a lot of redundancy removed.
- **`fetch_k` matters.** With `fetch_k=20`, MMR can reject the obvious near-
  duplicates and still fill `k=4` slots with relevant-but-different chunks.
- **`lambda_mult` is a slider, not a switch.** 0.9 ≈ plain similarity; 0.5 pulls
  in more diverse context. Neither is "correct" — it depends on whether you want
  the single best chunk or broad coverage.
- **The pipeline barely changed.** Only `search_type="mmr"` and `search_kwargs`
  on `as_retriever` were different — proof that the retriever block is fully
  swappable.

---

## Exercises

1. **Measure the redundancy.** For the same question, count how many of the
   top-8 baseline chunks share their first sentence (or a distinctive phrase)
   with another top-8 chunk, then do the same for MMR with `k=8, fetch_k=40`.
   Which retriever has fewer duplicates?
2. **Sweep `lambda_mult`.** Try 0.3 / 0.5 / 0.7 / 0.9 for the same question and
   print the first words of each top-4. At what value does MMR start to look like
   plain similarity?
3. **Answer both.** Build one prompt from `base_hits` and one from `mmr_hits`,
   invoke the LLM with both, and write a sentence about how the two answers
   differ in completeness vs repetition.
4. **Compare with the previous project.** Re-open Project 07's notebook and note
   that changing the *store* did not change the chunks, while changing the
   *retriever* did — that is the "one block at a time" design in action.
