from __future__ import annotations

import pandas as pd


def test_quick_run_report_includes_required_sections_and_metadata():
    from src.reporting import render_report

    summary = pd.DataFrame(
        [
            {"system": "baseline_gpt", "token_f1": 0.1, "numeric_accuracy": 0.1, "answer_coverage_at_3": float("nan"), "average_gold_answer_perplexity": 10, "average_total_latency": 1.0},
            {"system": "rag_gpt_tfidf_top3", "token_f1": 0.2, "numeric_accuracy": 0.2, "answer_coverage_at_3": 0.8, "average_gold_answer_perplexity": 8, "average_total_latency": 1.3},
            {"system": "oracle_gpt", "token_f1": 0.3, "numeric_accuracy": 0.2, "answer_coverage_at_3": 1.0, "average_gold_answer_perplexity": 7, "average_total_latency": 1.0},
            {"system": "random_context_gpt", "token_f1": 0.0, "numeric_accuracy": 0.0, "answer_coverage_at_3": 0.0, "average_gold_answer_perplexity": 12, "average_total_latency": 1.0},
        ]
    )
    metadata = {
        "run_id": "final_verified_run",
        "question_file": "data/questions/questions_verified.remapped.jsonl",
        "question_source": "verified_remapped",
        "num_questions": 271,
        "limit": None,
        "is_smoke_test": False,
        "valid_for_research": True,
        "invalid_reasons": [],
        "checkpoint": "model/model_weights.pt",
        "checkpoint_path": "model/model_weights.pt",
        "tokenizer_dir": "model/hftokenizer",
        "systems": ["baseline_gpt", "rag_gpt_tfidf_top3", "oracle_gpt", "random_context_gpt"],
        "missing_evidence_rate": 0.01,
    }
    report = render_report(
        "final_verified_run",
        summary,
        [],
        [],
        {
            "run_metadata": metadata,
            "checkpoint_path": "model/model_weights.pt",
            "tokenizer_dir": "model/hftokenizer",
            "model_config": {"vocab_size": 10000},
            "chunking_summary": {"num_chunks": 10, "by_source_type": {"table_row": 3}, "retrievable_count": 10, "non_retrievable_count": 0},
        },
        {"vocab_size": 10000, "tokenizer_class": "GPT2TokenizerFast"},
    )

    for section in (
        "## Hypothesis",
        "## Experiment Setup",
        "## Dataset Validity",
        "## Tokenizer and Checkpoint",
        "## Chunking Summary",
        "## Retrieval Quality",
        "## Generation Quality",
        "## Latency Tradeoff",
        "## RAG vs Baseline Conclusion",
        "## Training and Convergence",
        "## Limitations",
        "## Academic Honesty",
    ):
        assert section in report
    assert "Baseline GPT" in report
    assert "RAG-GPT TF-IDF top-3" in report
    assert "same checkpoint at model/model_weights.pt" in report
