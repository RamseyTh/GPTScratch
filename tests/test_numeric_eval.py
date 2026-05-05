from __future__ import annotations


def test_numeric_eval_uses_extraction_or_generation():
    from src.evaluation import evaluate_predictions

    questions = [
        {
            "question_id": "q1",
            "answer": "$109.158 billion",
            "answer_aliases": ["$109,158 million"],
            "question_type": "numeric_fact",
        }
    ]
    predictions = [
        {
            "question_id": "q1",
            "system": "rag_gpt_tfidf_top3",
            "prediction": "wrong",
            "extracted_numeric_answer": "$109,158 million",
            "retrieved_context": "Services $109,158 million",
        }
    ]
    summary, details = evaluate_predictions(predictions, questions)

    assert details[0]["numeric_extraction_accuracy"] is True
    assert details[0]["numeric_generation_accuracy"] is False
    assert details[0]["numeric_accuracy"] is True
    assert float(summary.iloc[0]["numeric_extraction_accuracy"]) == 1.0
