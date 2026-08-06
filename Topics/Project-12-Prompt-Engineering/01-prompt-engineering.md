> Source notebook: `NoteBooks/Projects/Project-12-Prompt-Engineering/01-prompt-engineering.ipynb`


---

# Project 12 — Prompt Engineering

> **Goal:** Everything stays the same — only the prompt changes.

```
Loader      : DirectoryLoader (Topics/*.md)     ← fixed
Splitter    : RecursiveCharacterTextSplitter    ← fixed
Embedding   : Gemini Embedding                  ← fixed
Vector DB   : Chroma                            ← fixed
Retriever   : Similarity (k=5)                  ← fixed
Prompt      : Basic → JSON → Few-shot → Citation → Reasoning   ← ROTATES
LLM         : Gemini 2.5 Flash                  ← fixed
```

**The rotation:** we build the pipeline once, retrieve the same chunks for the
same question, and then call the *same* LLM with **five different prompt
templates**. Any difference between the answers is caused by the prompt alone.

Learn: hallucination risk, format control, citation grounding, accuracy.

---

## 0 · Setup — environment & keys

We load the `.env` file that holds our `GOOGLE_API_KEY` and confirm the key is
present. We only ever print a **masked** preview — never the full key. The whole
pipeline needs this key because both the embeddings and the LLM are Gemini
models running in Google's cloud.

---

```python
from dotenv import load_dotenv

load_dotenv()
```

---

```python
import os

key = os.getenv("GOOGLE_API_KEY", "")
if not key:
    raise SystemExit("GOOGLE_API_KEY not found — copy .env.example to .env and add your key.")
print(f"GOOGLE_API_KEY set: {key[:4]}… (len {len(key)})")
```

---

These are the same real libraries used across the curriculum. No special
retrievers are needed in this project — we only rotate the prompt — so the import
list is the same one from the baseline pipeline.

---

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
```

---

## 1 · Load + Split (short)

Corpus: the repo's own `Topics/*.md` project cards. We load every markdown file
and split it with the baseline settings. This section is kept short because the
point of this project is the prompt rotation, not the ingestion.

---

```python
from pathlib import Path

loader = DirectoryLoader(
    str(Path("../../Topics")),
    glob="**/*.md",
    loader_cls=TextLoader,
)
docs = loader.load()
print("documents loaded:", len(docs))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print("chunks:", len(chunks))
```

---

```python
print("sample chunk source:", chunks[0].metadata["source"])
print(chunks[0].page_content[:150])
```

---

## 2 · Embed + Store + Retrieve (fixed)

We embed every chunk, store the vectors in Chroma, and retrieve the same top-5
chunks for one fixed question. **This context will be reused by all five
prompts**, so the retrieval step cannot bias the comparison.

Because every prompt needs a number, the context is rendered with `[n]` markers
and the source file name next to each chunk — the Citation prompt will point at
these markers.

---

```python
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
print("vector store ready")
```

---

```python
QUESTION = "What is a retriever in a RAG system and how can it be improved?"

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke(QUESTION)
print("retrieved chunks:", len(docs))
for d in docs:
    print("  -", Path(d.metadata["source"]).name)
```

---

```python
def format_context(docs):
    numbered = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        numbered.append(f"[{i}] (source: {Path(source).name})\n{doc.page_content}")
    return "\n\n".join(numbered)


context = format_context(docs)
print(context[:300], "…")
```

---

## 3 · The rotation — only the prompt changes

Five templates, same context, same question, same LLM. Each variant adds one new
instruction on top of the previous idea:

1. **Basic** — bare "answer from context" instruction. No format constraints.
2. **JSON output** — demand a strict JSON object so downstream code can parse it.
3. **Few-shot** — two example Q/A pairs that set the answer *style*.
4. **Citation** — require `[n]` markers pointing at the numbered sources.
5. **Reasoning** — force a step-by-step chain of thought before the final answer.

We store every template in a `prompts` dict; the iteration order *is* the order
above, so the comparison stays readable.

---

```python
prompts = {}

basic_template = """Answer the question using ONLY the context below. If the context does
not contain the answer, say: "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:"""

prompts["basic"] = ChatPromptTemplate.from_template(basic_template)
print("basic prompt ready")
```

---

**Variant 2 — JSON output.** Same grounding rule, but the output must be a
single JSON object. This gives us a machine-readable answer (`answer`), a list
of which sources were used, and a self-assessed confidence — all parseable with
`json.loads`.

---

```python
json_template = """Answer the question using ONLY the context below. Respond with a
single JSON object and nothing else:
{"answer": "...", "sources_used": [...], "confidence": "high|medium|low"}

Context:
{context}

Question:
{question}

JSON:"""

prompts["json"] = ChatPromptTemplate.from_template(json_template)
print("json prompt ready")
```

---

**Variant 3 — Few-shot.** Two worked examples teach the model the expected
length, tone, and structure of an answer. Few-shot prompting is the cheapest way
to steer style without changing the model.

---

```python
fewshot_template = """Answer using ONLY the context. Style follows the examples.
Example 1:
Question: What is RAG?
Answer: RAG retrieves documents and feeds them to the LLM.
Example 2:
Question: What is chunking?
Answer: Chunking splits documents into small pieces.
Now answer:
Question: {question}
Context:
{context}
Answer:"""

prompts["fewshot"] = ChatPromptTemplate.from_template(fewshot_template)
print("fewshot prompt ready")
```

---

**Variant 4 — Citation.** The context is already numbered (`[1]`, `[2]`, …)
with a source file next to each chunk. This prompt demands that every claim carry
a bracket marker, so the answer is *grounded*: we can verify that the numbers it
cites actually exist in the provided context.

---

```python
citation_template = """Answer the question using ONLY the numbered context below. Cite
the source of every claim with a bracket marker, e.g. "retrievers expand coverage [2]".
Use only numbers that appear in the context.

Context:
{context}

Question:
{question}

Answer with citations:"""

prompts["citation"] = ChatPromptTemplate.from_template(citation_template)
print("citation prompt ready")
```

---

**Variant 5 — Reasoning.** Ask the model to think step by step: first list the
relevant facts from the context, then reason from them, and only then give the
final answer. A visible reasoning chain makes the answer easier to verify and
usually reduces confident-but-wrong hallucination.

---

```python
reasoning_template = """Answer the question using ONLY the context below. Think step by
step: first list the relevant facts from the context, then reason from them, then give
the final answer in the last sentence.

Context:
{context}

Question:
{question}

Step-by-step reasoning:"""

prompts["reasoning"] = ChatPromptTemplate.from_template(reasoning_template)
print("reasoning prompt ready")
```

---

**One reusable call.** Every variant plugs into the same function: render the
prompt with the *same* `context` and `QUESTION`, call the *same* LLM, return the
answer text. Nothing else differs between the five runs.

---

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def run(question, prompt, context):
    messages = prompt.invoke({"context": context, "question": question})
    return llm.invoke(messages).content
```

---

**Run every variant.** The loop walks the `prompts` dict in insertion order,
calls the same `run()` for each, and stores the answers for the comparison table
below. Read the answers side by side: same question, same context, same model —
only the instructions changed.

---

```python
results = {}
for name, prompt in prompts.items():
    answer = run(QUESTION, prompt, context)
    results[name] = answer
    print(f"===== {name} =====")
    print(answer)
    print()
```

---

**Comparison table.** We turn the collected answers into a markdown table.
"Answer format" is measured, not guessed: we try `json.loads` and report whether
the answer is valid JSON. "Citations?" checks for `[n]` markers. The notes column
is our commentary on what each prompt produced. **Hallucination** is harder to
measure automatically — the deep-dive below explains the honest manual method.

---

```python
import json
import re


def detect_format(answer):
    try:
        json.loads(answer)
        return "JSON"
    except Exception:
        return "text"


def has_citations(answer):
    return "yes" if re.search(r"\[\d+\]", answer) else "no"
```

---

```python
notes = {
    "basic": "free-form; may drift from the context",
    "json": "strict machine-readable format",
    "fewshot": "answer style follows the examples",
    "citation": "grounded in numbered sources",
    "reasoning": "verifiable step-by-step chain",
}

print("| Prompt | Answer format | Citations? | ~Answer chars | Notes |")
print("|---|---|---|---|---|")
for name, answer in results.items():
    print(f"| {name} | {detect_format(answer)} | {has_citations(answer)} | {len(answer)} | {notes[name]} |")
```

---

## 4 · Deep-dive — hallucination, format control, citations

**Grounding check (Citation prompt).** The strongest automatic signal for
hallucination is a citation that points at a chunk that does *not exist*. The
cell below extracts every `[n]` a prompt produced and keeps only the numbers that
are inside the 1..k range of the context we actually provided. Citations that
survive are verifiable; numbers outside the range are fabricated.

---

```python
def cited_sources(answer, n_chunks):
    nums = [int(n) for n in re.findall(r"\[\d+\]", answer)]
    return sorted(set(n for n in nums if 1 <= n <= n_chunks))

print("context chunks provided:", len(docs))
for name, answer in results.items():
    print(f"{name:10s} cites chunk(s): {cited_sources(answer, len(docs))}")
```

---

**Reading the comparison.**

- **Hallucination** — the risk that the model answers from its *own* knowledge
  instead of the context. The Basic prompt has the highest risk; the Citation and
  Reasoning prompts constrain the model by forcing it to point at (or reason from)
  the supplied chunks. The honest test is manual: read each answer against the
  context and flag claims that are not supported.
- **Format control** — the JSON prompt is the only one that reliably yields
  parseable output; the rest return free-form text. If a downstream program must
  consume the answer, a format instruction is what makes that possible.
- **Citations** — only the Citation prompt reliably emits `[n]` markers, and the
  grounding check verifies they are real. A citation you can trace back to a
  chunk is a checkable, non-hallucinated claim.
- **Accuracy** — the few-shot and reasoning answers are typically the most
  complete, because examples give a shape to follow and a reasoning chain keeps
  the model close to the evidence.

---

## What you should notice

- **The prompt is a lever you can pull without touching anything else.** Same
  retrieval, same model — the five answers differ only because of the template.
- **Format is achievable by instruction.** The JSON prompt shows that output
  structure is a prompt concern, not a code concern.
- **Grounding is verifiable.** Citation markers turn "trust me" into a check: a
  marker that resolves to a real chunk is evidence, one that does not is a
  hallucination.
- **Constraints reduce hallucination.** Asking for citations or a reasoning chain
  keeps the model closer to the provided context than the bare Basic prompt.
- **There is no single best prompt.** JSON is best for machines, citation for
  verifiability, reasoning for quality, few-shot for style — pick by the use case.

---

## Exercises

1. Add a sixth variant, e.g. a **role prompt** ("You are a RAG debugging expert")
   or a **length-constrained** prompt ("answer in at most two sentences"), and
   rerun the comparison.
2. Ask a question that the context cannot answer, and compare how each prompt
   handles it — which ones admit ignorance, which ones hallucinate?
3. Swap the question while keeping the same five prompts and see whether the
   ranking of "best prompt" changes with the question.
