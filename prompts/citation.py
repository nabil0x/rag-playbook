"""Citation prompt template.

Prompt block: force the LLM to answer from context and cite sources.
No retrieval changes.
See Topics/Project-12-Prompt-Engineering/README.md.
"""


class CitationPrompt:
    """Build a prompt that requires source citations in the answer."""

    def __init__(self):
        self.template = (
            "Answer using ONLY the context below. Cite the source of each "
            "claim as [source].\n\nContext:\n{context}\n\nQuestion:\n"
            "{question}\n\nAnswer:"
        )

    def format(self, context: str, question: str) -> str:
        return self.template.format(context=context, question=question)
