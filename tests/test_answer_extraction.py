from __future__ import annotations


def test_unit_conversion_and_alias_matching():
    from src.answer_extraction import extract_numeric_answer, normalize_numeric_phrase

    assert normalize_numeric_phrase("$109.158 billion") == "109158 million"
    assert normalize_numeric_phrase("$109,158 million") == "109158 million"
    assert normalize_numeric_phrase("36%") == "36 percent"

    question = {
        "question_type": "numeric_fact",
        "answer_aliases": ["$109.158 billion"],
        "row_label": "Services net sales",
    }
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Apple table. Services net sales were $109.158 billion in 2025.",
            "metadata": {"row_label": "Services net sales"},
        }
    ]
    result = extract_numeric_answer(question, chunks)

    assert result is not None
    assert result["answer"] == "$109.158 billion"
    assert result["source_chunk_id"] == "c1"


def test_extraction_skips_negative_questions():
    from src.answer_extraction import extract_numeric_answer

    result = extract_numeric_answer({"question_type": "negative"}, [{"chunk_id": "c1", "text": "$10 million"}])

    assert result is None
