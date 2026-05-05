from __future__ import annotations


def test_evaluation_metrics():
    from src.evaluation import answer_coverage, exact_match, is_refusal, numeric_match, token_f1

    assert exact_match("The Apple", ["Apple"])
    assert token_f1("Apple Services revenue", ["Services revenue"]) > 0.7
    assert numeric_match("$109,158 million", ["$109.158 billion"])
    assert is_refusal("not enough information")
    assert answer_coverage(
        "The filing says Services net sales of $109.158 billion.",
        {
            "answer": "$109.158 billion",
            "answer_aliases": ["109158 million"],
            "gold_evidence_text": "Services net sales",
        },
    )
