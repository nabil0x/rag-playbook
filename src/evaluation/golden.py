"""Golden QA set for RAG answer evaluation.

Evaluation block: the 23 hand-checked question/answer pairs used to score
generation quality. Source of truth is
NoteBooks/SD-08-Invoices/02-invoice-rag-strategy-comparison.ipynb cell 5 (the
``QA`` dict) — extracted verbatim, not hand-rewritten.
See Topics/Project-20-Deep-Eval/README.md.
"""

GOLDEN_QA: dict[str, list[tuple[str, str]]] = {
    "sample-invoice": [
        ("What is the invoice number?", "INV-100"),
        ("What is the total due on the invoice?", "610.00"),
        ("Who is the customer on this invoice?", "MICROSOFT CORPORATION"),
        ("What was the sales tax amount?", "10.00"),
    ],
    "multipage_invoice1": [
        ("What is the total amount of the Company A invoice?", "430.00"),
        ("What is the subtotal of the Company B invoice?", "3000.00"),
        ("Which line item has a quantity of 8?", "G"),
        ("What is the tax amount on the Company B invoice?", "300.00"),
    ],
    "Invoice_1": [
        ("What is the invoice number?", "3847193"),
        ("What is the total price of all items?", "1075.70"),
        ("How many pieces were delivered in total?", "66"),
        ("Which item code is the bubble film roll?", "JF9912413BF"),
    ],
    "Invoice-6": [
        ("What is the receipt number on the project statement?", "9876"),
        ("What is the total amount on the project statement?", "10,686.25"),
        ("What discount rate was applied to the leadership training?", "25%"),
        ("What is the sales tax rate?", "3%"),
    ],
    "sdk-invoice1": [
        ("What is the invoice number?", "34278587"),
        ("What are the total charges on the invoice?", "56,651.49"),
        ("Who is the invoice for?", "Microsoft"),
    ],
    "german-zugferd": [
        ("What is the Rechnungsnummer (invoice number)?", "1001"),
        ("What is the Rechnungsbetrag (total amount due)?", "139,20"),
        ("What VAT rate is applied to Position 1?", "19 %"),
        ("Who is the payee (seller) on this invoice?", "PG Consulting"),
    ],
}

#: The document keys, in the order they appear in ``GOLDEN_QA``.
DOC_KEYS: tuple[str, ...] = tuple(GOLDEN_QA.keys())


def load_golden() -> list[dict]:
    """Return one dict per QA pair: {"doc", "question", "reference"}."""
    pairs: list[dict] = []
    for doc, qa in GOLDEN_QA.items():
        for question, reference in qa:
            pairs.append(
                {"doc": doc, "question": question, "reference": reference}
            )
    return pairs


if __name__ == "__main__":
    pairs = load_golden()
    print(f"{len(GOLDEN_QA)} documents, {len(pairs)} questions")
    for doc, qa in GOLDEN_QA.items():
        print(f"  {doc}: {len(qa)}")