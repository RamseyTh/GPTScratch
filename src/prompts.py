from __future__ import annotations


def baseline_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def rag_prompt(question: str, retrieved_context: str, numeric_candidate: str | None = None) -> str:
    candidate_line = f"\nCandidate numeric answer from context: {numeric_candidate}\n" if numeric_candidate else ""
    return (
        "Context:\n"
        f"{retrieved_context}\n\n"
        f"{candidate_line}"
        f"Question: {question}\n\n"
        'Answer using the context. If the context does not contain the answer, say "not enough information."\n'
        "Answer:"
    )


def oracle_prompt(question: str, gold_context: str) -> str:
    return f"Gold evidence:\n{gold_context}\n\nQuestion: {question}\nAnswer:"


def random_context_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
