from __future__ import annotations

import json


def test_install_from_source_dir_auto_discovers_jsonl(tmp_path):
    from scripts.install_verified_questions import install_verified_questions

    source_dir = tmp_path / "uploaded"
    source_dir.mkdir()
    rows = [
        {"question_id": "q1", "question_type": "numeric_fact", "ticker": "AAPL", "expected_refusal": False},
        {"question_id": "q2", "question_type": "negative", "ticker": "MSFT", "expected_refusal": True},
    ]
    (source_dir / "questions_verified.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = install_verified_questions(source_dir=source_dir, out_dir=tmp_path / "data" / "questions", project_root=tmp_path)

    out_dir = tmp_path / "data" / "questions"
    assert (out_dir / "questions_verified.jsonl").exists()
    assert (out_dir / "questions.jsonl").exists()
    assert (out_dir / "install_summary.json").exists()
    assert summary["num_questions"] == 2
    assert summary["expected_refusal_counts"]["true"] == 1


def test_missing_required_jsonl_error_lists_searches_and_candidates(tmp_path):
    from scripts.install_verified_questions import install_verified_questions

    source_dir = tmp_path / "uploaded"
    source_dir.mkdir()
    (source_dir / "some_questions.jsonl").write_text("{}\n", encoding="utf-8")

    try:
        install_verified_questions(source_dir=source_dir, out_dir=tmp_path / "data" / "questions", project_root=tmp_path)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected missing verified JSONL error")

    assert "Searched locations" in message
    assert "Found candidate files" in message
    assert "some_questions.jsonl" in message
    assert "Copy questions_verified.jsonl" in message
