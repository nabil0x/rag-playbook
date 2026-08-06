> Source notebook: `NoteBooks/Projects/Project-01-Baseline-RAG/04-baseline-rag.ipynb`


---

# Project 1 — The Baseline RAG

**Goal:** Understand the complete pipeline with the simplest components.

```
Loader      : Web Loader
Splitter    : RecursiveCharacterTextSplitter
Embedding   : Gemini Embedding
Vector DB   : Chroma
Retriever   : Similarity Search (Top-K)
Prompt      : Basic Context + Question
LLM         : Gemini 2.5 Flash
```

Learn:

* LangChain basics
* Document lifecycle
* Retrieval flow

---

### How to work through this notebook

This notebook runs the **complete baseline pipeline** — load → chunk → embed →
store → retrieve → prompt → answer — with the simplest component in each block.
Every section below explains **what** the step does, **why** it exists in RAG,
and **what to expect** when you run it. The two empty cells at the end are your
sandbox.

## 0 · Setup — environment & imports

**WHAT:** Loads `.env` (so the Gemini API key is available) and imports all
LangChain classes: `WebBaseLoader`, `RecursiveCharacterTextSplitter`, `Chroma`,
Gemini embeddings + chat model, and `ChatPromptTemplate`.

**WHY:** This is the kitchen prep of every pipeline — one place where keys and
imports live, so the pipeline cells below stay short and readable.

**WHAT TO EXPECT:** The first cell prints `True` (`load_dotenv()` succeeded).
If you see `False`, create a `.env` with `GOOGLE_API_KEY=...`.

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

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
```

---

```text
/tmp/ipykernel_3260637/412520464.py:1: DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details and migration guidance toward standalone integration packages.
  from langchain_community.document_loaders import WebBaseLoader
USER_AGENT environment variable not set, consider setting it to identify your requests.
```

---

## 1 · Load — fetch a public-domain book

**WHAT:** `WebBaseLoader` fetches a Project Gutenberg HTML book and `.load()`
returns a list of `Document`s. The cell prints the document count, the first 10
characters of text, and the metadata.

**WHY:** Loading is where data enters the pipeline. A Gutenberg page is a clean,
public-domain source — ideal for a baseline you can re-run without worrying
about licensing or changing content.

**WHAT TO EXPECT:** `1` document, a snippet of the book's opening text, and
metadata such as `{'source': ..., 'title': ..., 'language': 'en'}`.

---

```python
url = "https://www.gutenberg.org/cache/epub/79247/pg79247-images.html"

loader = WebBaseLoader(url)
docs = loader.load()

print(len(docs))
print(docs[0].page_content[:10])
print(docs[0].metadata)
```

---

```text
1


With the
{'source': 'https://www.gutenberg.org/cache/epub/79247/pg79247-images.html', 'title': 'With the Joker Wild | Project Gutenberg', 'language': 'en'}
```

---

## 2 · Split — chunk the book with overlap

**WHAT:** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`
cuts the document into overlapping chunks; the cell prints the chunk count and a
sample chunk.

**WHY:** Retrieval works on chunks, not whole books. A 1000-character chunk is
focused enough to be relevant, and the 200-character overlap preserves content
that falls on a boundary.

**WHAT TO EXPECT:** A much larger chunk count than the document count (here 74
chunks), plus a sample chunk printed in full.

---

```python
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks=splitter.split_documents(docs)
print(len(chunks))
print(chunks[2].page_content)
```

---

```text
74
This ain’t no poker story. I ain’t going to try and cold-deck anybody into thinking it is, so I’ll orate right now that she ain’t. I aims to prove uh little moral, and yuh can’t prove no morals in uh poker story, ’cause there ain’t no morals in that game.
All I wants to do is to cinch the fact that honesty is the best policy, and that wisdom ain’t uh better hand than luck. I played in uh poker game once, where the deuces run wild. It wasn’t poker—it was suicide. Over on the coast they sometimes plays with the joker wild. They opens the game with uh silent prayer and deals five cards apiece. Before they discards they wires for the police, coroner, and notifies their families. Then they spends the interim—that word seems to fit exactly, ’cause it looks like interment—trying to find out where the joker fits into uh straight.
```

---

## 3 · Embed — text → vectors

**WHAT:** `GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")`
creates the embedding model used to vectorize every chunk.

**WHY:** Semantic search compares *vectors*, not text. This model turns each
chunk into a list of numbers so similar content lands close together in vector
space.

**WHAT TO EXPECT:** An embeddings object — no output on its own; it is used in
the store step next.

---

```python
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
```

---

## 4 · Store — index the vectors in Chroma

**WHAT:** `Chroma.from_documents(documents=chunks, embedding=embeddings)` embeds
the chunks and indexes the vectors in a Chroma collection.

**WHY:** The vector store is the pipeline's memory: it lets the retriever find
the most relevant chunks in milliseconds.

**WHAT TO EXPECT:** A `Chroma` vector store object assigned to `vector_store`
(and a `chroma_langchain_db/` directory on disk).

---

```python
from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)
```

---

## 5 · Retrieve — top-k similarity search

**WHAT:** `vector_store.similarity_search(query, k=3)` returns the 3 chunks most
similar to the question, and the next cell joins their text into one `context`
string separated by blank lines.

**WHY:** This is the "R" in RAG — evidence gathering. Retrieving *k* chunks
means the prompt gets the most relevant passages, no more and no less. The
`context` string is what the prompt will hand to the model.

**WHAT TO EXPECT:** `docs` holds 3 retrieved chunks; `context` is their joined
text.

---

## 6 · Prompt — package context + question

**WHAT:** A `ChatPromptTemplate` wraps the instructions ("answer using ONLY the
provided context", with an explicit "I don't know" fallback) around the
`context` and `question` slots. The next cells fill it: `prompt.invoke({...})`
produces the final `messages`, which the notebook prints so you can see exactly
what the model receives.

**WHY:** The prompt is the contract that stops hallucination — telling the model
it may *only* answer from the retrieved context. Reading the rendered `messages`
is the best way to understand what your model is actually seeing.

**WHAT TO EXPECT:** A `ChatPromptTemplate`, then a printed `messages` object
showing your instructions, the retrieved context, and the question.

---

## 7 · Answer — the LLM reads the prompt

**WHAT:** `ChatGoogleGenerativeAI(model="gemini-2.5-flash")` invokes the filled
`messages` and the answer is printed.

**WHY:** This is the final block: the model reads the retrieved evidence plus
the question and produces a grounded answer. Without retrieval the model would
guess; with retrieval it answers from the book.

**WHAT TO EXPECT:** A natural-language answer to "What is the moral of the
story?" — derived from the retrieved chunks, not from the model's memory.

---

```python
query = "What is the moral of the story?"

docs = vector_store.similarity_search(
    query=query,
    k=3
)
```

---

```python
context = "\n\n".join(doc.page_content for doc in docs)
```

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
```

---

```python
messages = prompt.invoke({
    "context": context,
    "question": query
})
```

---

```python
print(messages)
```

---

```text
messages=[HumanMessage(content='\nYou are a helpful assistant.\n\nAnswer the question using ONLY the provided context.\nIf the answer is not contained in the context, say:\n"I don\'t know based on the provided context."\n\nContext:\nThis ain’t no poker story. I ain’t going to try and cold-deck anybody into thinking it is, so I’ll orate right now that she ain’t. I aims to prove uh little moral, and yuh can’t prove no morals in uh poker story, ’cause there ain’t no morals in that game.\nAll I wants to do is to cinch the fact that honesty is the best policy, and that wisdom ain’t uh better hand than luck. I played in uh poker game once, where the deuces run wild. It wasn’t poker—it was suicide. Over on the coast they sometimes plays with the joker wild. They opens the game with uh silent prayer and deals five cards apiece. Before they discards they wires for the police, coroner, and notifies their families. Then they spends the interim—that word seems to fit exactly, ’cause it looks like interment—trying to find out where the joker fits into uh straight.\n\nHonesty is the best policy. The feller who first uttered them words said more in five movements of his vocal cords than the person did in eight, when “A fool and his money are soon parted” was coined. Magpie Simpkins was uh wise man. I say, “was wise,” because after he got wise to himself he found that he was uh fool—but he’ll never admit it.\nMagpie is six feet several inches of wisdom and deduction and so forth and so on. He’s tried every course from hair-oil to hypnotism to make himself admired of everybody, but he still wiggles along, sluicing gravel, herding uh burro and predicting that he’s got Solomon looking half-witted.\nMe? I’m Ike Harper, Magpie’s pardner. I wished him on to me one time over in Helena, when he was suffering with uh rash, brought on by trying to convince uh wildcat that the human eye is stronger than four feet full uh claws. Me and that animated fish-pole has seen ups and downs ever since.\n\n“My soul wallers in the depths uh remorse, Ike,” he sobs. “It’s uh good thing that we wasn’t both born with brains. I was wise, but——”\n“Ike Harper was uh danged fool,” I finishes for him again. “Uh danged fool and his money is sometimes hard to part, Magpie. Them four burros didn’t—say, Magpie, it was uh case uh three of uh kind trying to beat four jacks and the joker.”\n“Five jacks, Ike,” corrects Magpie. “The joker was wild.”\nAll of which shows—well, why preach?\n\nQuestion:\nWhat is the moral of the story?\n\nAnswer:\n', additional_kwargs={}, response_metadata={})]
```

---

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

response = llm.invoke(messages)

print(response.content)
```

---

```text
The moral of the story is that honesty is the best policy, and that wisdom is not a better hand than luck.
```

---

## 8 · Try it yourself — your sandbox

The two empty cells are yours. Change the query, change `k` in the retrieval
step, or re-run the pipeline on a different question — then observe how the
answer changes.

---

## What you should notice

- **The whole pipeline fits in about ten small cells.** Each block does one job
  and hands its output to the next — that modularity is the point of RAG.
- **The prompt is doing heavy lifting.** The "answer ONLY from context" wording
  is what turns a generic LLM into a grounded one.
- **`k` shapes the answer.** With `k=3` the model sees three chunks of evidence;
  changing `k` changes both the answer's breadth and the token cost.
- **`gemini-embedding-2-preview` + `gemini-2.5-flash` is the repo's default
  stack.** Later projects rotate one of these blocks and compare.
- **The vector store is regenerable.** If you re-run from the top, `Chroma`
  rebuilds the index from scratch.

---

## Exercises

1. **Ask a different question.** Change `query` (for example ask about the
   characters or the setting) and re-run retrieval + prompt + answer. Is the
   answer still grounded in the book?
2. **Tune `k`.** Retrieve `k=1` and `k=5` and compare answer quality and how
   much context the prompt contains.
3. **Swap the prompt.** Rewrite the template with a stricter "if the answer is
   not in the context, say exactly 'I don't know'" instruction and see whether
   an out-of-context question is handled more safely.
