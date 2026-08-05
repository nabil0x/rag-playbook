"""Basic prompt template: Context + Question -> Answer.

Prompt block: the default RAG prompt shape.
See Topics/Project-12-Prompt-Engineering/README.md.
"""


class BasicPrompt:
    """Build the standard Context/Question/Answer prompt."""

    def __init__(self):
        self.template = (
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )

    def format(self, context: str, question: str) -> str:
        return self.template.format(context=context, question=question)
