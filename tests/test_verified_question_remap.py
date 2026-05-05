from __future__ import annotations

import json


def test_evidence_remapping_updates_stale_chunk_id(tmp_path):
    from scripts.validate_verified_questions import remap_verified_questions
    from src.utils import read_jsonl

    questions_path = tmp_path / "questions_verified.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    out_path = tmp_path / "questions_verified.remapped.jsonl"
    question = {
        "question_id": "AAPL_2025_001",
        "question": "What were Apple's Services net sales in fiscal 2025?",
        "answer": "$109.158 billion",
        "answer_aliases": ["109.158 billion"],
        "ticker": "AAPL",
        "company": "Apple",
        "year": "2025",
        "question_type": "numeric_fact",
        "source_section": "Item 7",
        "source_type": "table_row",
        "gold_evidence_text": "Services net sales were $109.158 billion in 2025",
        "relevant_chunk_ids": ["old_table_aware_256_id"],
        "expected_refusal": False,
    }
    chunk = {
        "chunk_id": "new_chunk_id",
        "text": "Apple fiscal 2025 table. Services net sales were $109.158 billion in 2025.",
        "ticker": "AAPL",
        "year": "2025",
        "section_id": "Item 7",
        "source_type": "table_row",
    }
    questions_path.write_text(json.dumps(question) + "\n", encoding="utf-8")
    chunks_path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")

    remap_verified_questions(questions_path, chunks_path, out_path)
    remapped = read_jsonl(out_path)[0]

    assert remapped["relevant_chunk_ids"] == ["new_chunk_id"]
    assert remapped["gold_chunk_id"] == "new_chunk_id"
    assert remapped["evidence_valid"] is True
    assert "remapped" in remapped["validation_notes"]
    assert (tmp_path / "evidence_remap_report.md").exists()
    assert (tmp_path / "evidence_remap_summary.csv").exists()
