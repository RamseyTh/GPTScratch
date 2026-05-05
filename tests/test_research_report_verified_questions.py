from __future__ import annotations

import pandas as pd


def test_research_report_has_verified_dataset_sections():
    from src.reporting import render_report

    summary = pd.DataFrame(
        [
            {"system": "baseline_gpt", "token_f1": 0.2, "numeric_accuracy": 0.1, "answer_coverage_at_3": float("nan"), "average_gold_answer_perplexity": 10, "average_total_latency": 1.0},
            {"system": "oracle_gpt", "token_f1": 0.4, "numeric_accuracy": 0.2, "answer_coverage_at_3": 1.0, "average_gold_answer_perplexity": 8, "average_total_latency": 1.0},
            {"system": "rag_gpt_tfidf_top3", "token_f1": 0.3, "numeric_accuracy": 0.2, "answer_coverage_at_3": 0.8, "average_gold_answer_perplexity": 7, "average_total_latency": 1.2},
            {"system": "random_context_gpt", "token_f1": 0.0, "numeric_accuracy": 0.0, "answer_coverage_at_3": 0.0, "average_gold_answer_perplexity": 12, "average_total_latency": 1.0},
        ]
    )
    metadata = {
        "question_file": "data/questions/questions_verified.remapped.jsonl",
        "question_source": "verified_remapped",
        "num_questions": 271,
        "question_type_counts": {"numeric_fact": 67},
        "ticker_counts": {"AAPL": 43},
        "company_counts": {"Apple": 43},
        "missing_evidence_rate": 0.01,
        "is_smoke_test": False,
        "valid_for_research": True,
        "invalid_reasons": [],
        "oracle_answer_coverage_at_3": 1.0,
        "rag_answer_coverage_at_3": 0.8,
    }
    config = {
        "run_metadata": metadata,
        "checkpoint_path": "model/model_weights.pt",
        "tokenizer_dir": "model/hftokenizer",
        "model_config": {"vocab_size": 10000},
        "chunking_summary": {
            "num_chunks": 100,
            "by_source_type": {"narrative": 50, "table_row": 30, "table_summary": 20},
            "retrievable_count": 90,
            "non_retrievable_count": 10,
        },
    }
    tokenizer_metadata = {"vocab_size": 10000, "tokenizer_class": "GPT2TokenizerFast"}

    report = render_report("verified", summary, [], [], config, tokenizer_metadata)

    assert report.startswith("# Final Report: Retrieval-Augmented Generation with Financial Filings")
    assert "## Dataset" in report
    assert "number of questions: 271" in report
    assert "verified_remapped" in report
    assert "## Tokenizer and Checkpoint" in report
    assert "compatibility: PASS" in report
    assert "## RAG vs Baseline" in report
    assert "Training logs were not provided" in report
