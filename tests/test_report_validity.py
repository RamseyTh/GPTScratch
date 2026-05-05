from __future__ import annotations

import pandas as pd


def test_three_question_run_is_smoke_and_invalid(tmp_path):
    from src.evaluation import build_run_metadata
    from src.reporting import render_report

    questions = [{"question_id": f"q{i}"} for i in range(3)]
    predictions = []
    for question in questions:
        for system in ("baseline_gpt", "rag_gpt_tfidf_top3", "oracle_gpt", "random_context_gpt"):
            predictions.append({"question_id": question["question_id"], "system": system})
    summary = pd.DataFrame(
        [
            {"system": "baseline_gpt", "num_questions": 3, "answer_coverage_at_3": float("nan"), "average_gold_answer_perplexity": 10},
            {"system": "oracle_gpt", "num_questions": 3, "answer_coverage_at_3": 1.0, "average_gold_answer_perplexity": 8},
            {"system": "rag_gpt_tfidf_top3", "num_questions": 3, "answer_coverage_at_3": 0.33, "average_gold_answer_perplexity": 11},
            {"system": "random_context_gpt", "num_questions": 3, "answer_coverage_at_3": 0.0, "average_gold_answer_perplexity": 12},
        ]
    )
    metadata = build_run_metadata(questions, predictions, summary, "data/questions/questions_verified.jsonl", limit=3)

    assert metadata["is_smoke_test"] is True
    assert metadata["valid_for_research"] is False

    report = render_report("smoke", summary, predictions, [], {"run_metadata": metadata}, {})
    assert "This run is a smoke test" in report
    assert "pipeline executed successfully" in report


def test_large_synthetic_run_can_be_valid_when_gates_pass():
    from src.evaluation import build_run_metadata

    questions = [{"question_id": f"q{i}"} for i in range(100)]
    predictions = []
    for question in questions:
        for system in ("baseline_gpt", "rag_gpt_tfidf_top3", "oracle_gpt", "random_context_gpt"):
            predictions.append({"question_id": question["question_id"], "system": system})
    summary = pd.DataFrame(
        [
            {"system": "baseline_gpt", "num_questions": 100, "answer_coverage_at_3": float("nan")},
            {"system": "oracle_gpt", "num_questions": 100, "answer_coverage_at_3": 0.99},
            {"system": "rag_gpt_tfidf_top3", "num_questions": 100, "answer_coverage_at_3": 0.75},
            {"system": "random_context_gpt", "num_questions": 100, "answer_coverage_at_3": 0.10},
        ]
    )
    for index, question in enumerate(questions):
        question.update(
            {
                "question_type": ["numeric_fact", "text_fact", "risk_factor", "explanation", "negative"][index % 5],
                "ticker": "AAPL",
                "gold_evidence_text": "evidence" if index % 5 != 4 else "",
                "relevant_chunk_ids": ["c1"] if index % 5 != 4 else [],
                "expected_refusal": index % 5 == 4,
            }
        )
    metadata = build_run_metadata(questions, predictions, summary, "data/questions/questions_verified.jsonl")

    assert metadata["is_smoke_test"] is False
    assert metadata["valid_for_research"] is True
    assert metadata["invalid_reasons"] == []


def test_rag_low_coverage_is_reported_but_not_invalid_when_verified():
    from src.evaluation import build_run_metadata

    questions = [
        {
            "question_id": f"q{i}",
            "question_type": ["numeric_fact", "text_fact", "risk_factor", "explanation", "negative"][i % 5],
            "ticker": "AAPL",
            "gold_evidence_text": "evidence" if i % 5 != 4 else "",
            "relevant_chunk_ids": ["c1"] if i % 5 != 4 else [],
            "expected_refusal": i % 5 == 4,
        }
        for i in range(100)
    ]
    predictions = [
        {"question_id": question["question_id"], "system": system}
        for question in questions
        for system in ("baseline_gpt", "rag_gpt_tfidf_top3", "oracle_gpt", "random_context_gpt")
    ]
    summary = pd.DataFrame(
        [
            {"system": "baseline_gpt", "num_questions": 100, "answer_coverage_at_3": float("nan")},
            {"system": "oracle_gpt", "num_questions": 100, "answer_coverage_at_3": 1.0},
            {"system": "rag_gpt_tfidf_top3", "num_questions": 100, "answer_coverage_at_3": 0.20},
            {"system": "random_context_gpt", "num_questions": 100, "answer_coverage_at_3": 0.10},
        ]
    )

    metadata = build_run_metadata(questions, predictions, summary, "data/questions/questions_verified.jsonl")

    assert metadata["valid_for_research"] is True
    assert metadata["retrieval_quality"] == "weak"
    assert "rag_answer_coverage_at_3<0.70" not in metadata["invalid_reasons"]
