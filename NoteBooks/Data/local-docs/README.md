# local-docs — sample knowledge base

This folder is the tiny sample corpus for Project 03
(Local Documentation Search). It stands in for a real project's
documentation tree:

- `README.md` — this file, the overview of the knowledge base
- `docs/` — topic pages that answer real questions about the stack

To make the search interesting, try asking things like:

- Which embedding model does this knowledge base recommend?
- How do I run similarity search?
- How do I persist a FAISS index?

Everything here is plain markdown, so the `DirectoryLoader` can load it with
`TextLoader` and the `RecursiveCharacterTextSplitter` can chunk it. Adding more
files to this tree is the easiest way to make the index bigger.
