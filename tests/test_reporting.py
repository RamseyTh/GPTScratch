from __future__ import annotations

import pandas as pd


def test_low_coverage_report_recommends_retrieval_fixes():
    from src.reporting import render_report

    summary = pd.DataFrame(
        [
            {"system": "baseline_gpt", "answer_coverage_at_3": float("nan"), "average_gold_answer_perplexity": 10},
            {"system": "oracle_gpt", "answer_coverage_at_3": 1.0, "average_gold_answer_perplexity": 8},
            {"system": "rag_gpt_tfidf_top3", "answer_coverage_at_3": 0.2, "average_gold_answer_perplexity": 11},
        ]
    )
    metadata = {"num_questions": 3, "is_smoke_test": True, "valid_for_research": False, "invalid_reasons": ["num_questions<50"]}
    report = render_report("run", summary, [], [{"noise_reason": "wrong_section"}], {"run_metadata": metadata}, {})

    assert "This run verifies pipeline execution" in report
    assert "Retrieval is the main bottleneck" in report


def test_valid_report_can_state_rag_improved():
    from src.reporting import render_report

    summary = pd.DataFrame(
        [
            {"system": "baseline_gpt", "token_f1": 0.1, "answer_coverage_at_3": float("nan"), "average_gold_answer_perplexity": 10},
            {"system": "oracle_gpt", "token_f1": 0.2, "answer_coverage_at_3": 1.0, "average_gold_answer_perplexity": 8},
            {"system": "rag_gpt_tfidf_top3", "token_f1": 0.3, "answer_coverage_at_3": 0.8, "average_gold_answer_perplexity": 7},
        ]
    )
    metadata = {"num_questions": 100, "is_smoke_test": False, "valid_for_research": True, "invalid_reasons": []}
    report = render_report("run", summary, [], [], {"run_metadata": metadata}, {})

    assert "RAG improved" in report
