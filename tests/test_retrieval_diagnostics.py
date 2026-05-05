from __future__ import annotations


def test_context_budget_truncates_long_context():
    from src.retrieval import retrieval_context

    results = [
        {"rank": 1, "text": " ".join(["alpha"] * 100), "metadata": {"ticker": "AAPL", "year": "2025"}},
        {"rank": 2, "text": " ".join(["beta"] * 100), "metadata": {"ticker": "AAPL", "year": "2025"}},
    ]
    context = retrieval_context(results, token_budget=20)

    assert len(context.split()) <= 20
    assert "alpha" in context


def test_retrieval_diagnostic_reports_coverage_and_noise():
    from src.retrieval import retrieval_diagnostic

    question = {
        "question_id": "q1",
        "question": "What were Apple's Services net sales?",
        "answer": "$109.158 billion",
        "answer_aliases": ["109.158 billion"],
        "ticker": "AAPL",
        "year": "2025",
        "question_type": "numeric_fact",
    }
    results = [
        {
            "chunk_id": "c1",
            "score": 1.0,
            "text": "Apple Services net sales were $109.158 billion.",
            "metadata": {"ticker": "AAPL", "year": "2025", "source_type": "table_row"},
        }
    ]
    diag = retrieval_diagnostic(question, "query", {"ticker": "AAPL", "year": "2025"}, results, 0.01)

    assert diag["answer_coverage_at_3"] is True
    assert diag["source_accuracy"] is True
    assert diag["noise_reason"] == "none"
