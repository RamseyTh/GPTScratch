from __future__ import annotations

import json


def test_validate_verified_questions_report_includes_match_rate(tmp_path):
    from scripts.validate_verified_questions import remap_verified_questions

    questions = [
        {
            "question_id": "q1",
            "question_type": "text_fact",
            "question": "What did Apple report?",
            "answer": "Services revenue",
            "answer_aliases": ["Services revenue"],
            "ticker": "AAPL",
            "year": "2025",
            "source_type": "narrative",
            "gold_evidence_text": "Apple reported Services revenue",
            "relevant_chunk_ids": ["old_id"],
            "expected_refusal": False,
        },
        {
            "question_id": "q2",
            "question_type": "negative",
            "question": "Did Apple report a bank segment?",
            "answer": "not enough information",
            "ticker": "AAPL",
            "year": "2025",
            "gold_evidence_text": "",
            "relevant_chunk_ids": [],
            "expected_refusal": True,
        },
    ]
    chunks = [
        {
            "chunk_id": "new_id",
            "text": "Apple reported Services revenue in its fiscal 2025 filing.",
            "ticker": "AAPL",
            "year": "2025",
            "source_type": "narrative",
        }
    ]
    questions_path = tmp_path / "questions_verified.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    out_path = tmp_path / "questions_verified.remapped.jsonl"
    questions_path.write_text("\n".join(json.dumps(row) for row in questions) + "\n", encoding="utf-8")
    chunks_path.write_text("\n".join(json.dumps(row) for row in chunks) + "\n", encoding="utf-8")

    result = remap_verified_questions(questions_path, chunks_path, out_path)
    report = (tmp_path / "evidence_remap_report.md").read_text(encoding="utf-8")

    assert result["evidence_match_rate"] == 1.0
    assert "answerable questions: 1" in report
    assert "negative questions: 1" in report
    assert "evidence match rate: 1.0000" in report
