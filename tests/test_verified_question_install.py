from __future__ import annotations

import json
from pathlib import Path


def test_install_verified_questions_writes_default_files(tmp_path):
    from scripts.install_verified_questions import install_verified_questions
    from src.utils import read_json

    source = tmp_path / "questions_verified.jsonl"
    rows = [
        {"question_id": "q1", "question_type": "numeric_fact", "ticker": "AAPL", "company": "Apple"},
        {"question_id": "q2", "question_type": "text_fact", "ticker": "MSFT", "company": "Microsoft"},
        {"question_id": "q3", "question_type": "negative", "ticker": "AAPL", "company": "Apple"},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    csv_path = tmp_path / "questions_verified.csv"
    csv_path.write_text("question_id\nq1\n", encoding="utf-8")
    audit = tmp_path / "evidence_audit.md"
    audit.write_text("# Evidence", encoding="utf-8")
    report = tmp_path / "validation_report.md"
    report.write_text("Verified questions available: 3.", encoding="utf-8")

    summary = install_verified_questions(
        verified_jsonl=str(source),
        verified_csv=str(csv_path),
        evidence_audit=str(audit),
        validation_report=str(report),
        out_dir=tmp_path / "data" / "questions",
    )

    out_dir = tmp_path / "data" / "questions"
    assert (out_dir / "questions_verified.jsonl").exists()
    assert (out_dir / "questions.jsonl").exists()
    assert (out_dir / "install_summary.json").exists()
    assert (out_dir / "question_distribution.csv").exists()
    assert summary["num_questions"] == 3
    assert read_json(out_dir / "install_summary.json")["question_type_counts"]["numeric_fact"] == 1
