from __future__ import annotations


def test_rejects_vague_question():
    from src.evaluation import validate_question_rows

    questions = [{"question_id": "q1", "question": "What does the company disclose?", "answer": "Revenue"}]
    _, rejected, _ = validate_question_rows(questions, [])

    assert rejected
    assert "vague_question" in rejected[0]["rejection_reasons"]


def test_rejects_numeric_without_numeric_answer():
    from src.evaluation import validate_question_rows

    questions = [
        {
            "question_id": "q1",
            "question": "What were Apple's Services net sales?",
            "answer": "a lot",
            "question_type": "numeric_fact",
            "gold_evidence_text": "Services net sales were high",
            "relevant_chunk_ids": ["c1"],
            "answer_aliases": ["a lot"],
            "source_type": "table_row",
        }
    ]
    chunks = [{"chunk_id": "c1", "text": "Services net sales were high"}]
    _, rejected, _ = validate_question_rows(questions, chunks)

    assert rejected
    assert "numeric_answer_required" in rejected[0]["rejection_reasons"]


def test_accepts_table_row_numeric_question_with_aliases():
    from src.evaluation import validate_question_rows

    questions = [
        {
            "question_id": "q1",
            "question": "What were Apple's Services net sales?",
            "answer": "$109.158 billion",
            "normalized_answer": "109158 million",
            "answer_aliases": ["$109,158 million"],
            "question_type": "numeric_fact",
            "source_type": "table_row",
            "gold_evidence_text": "Services net sales were $109.158 billion",
            "relevant_chunk_ids": ["c1"],
        }
    ]
    chunks = [{"chunk_id": "c1", "text": "Services net sales were $109.158 billion in 2025."}]
    valid, rejected, _ = validate_question_rows(questions, chunks)

    assert valid
    assert not rejected


def test_accepts_plausible_negative_question():
    from src.evaluation import validate_question_rows

    questions = [
        {
            "question_id": "q1",
            "question": "Did Apple report a Cloud Infrastructure segment?",
            "answer": "not enough information",
            "question_type": "negative",
            "expected_refusal": True,
        }
    ]
    valid, rejected, _ = validate_question_rows(questions, [])

    assert valid
    assert not rejected
