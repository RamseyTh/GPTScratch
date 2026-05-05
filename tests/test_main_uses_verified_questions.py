from __future__ import annotations

import json


def test_main_defaults_to_remapped_verified_then_verified(tmp_path, monkeypatch):
    from main import question_source, resolve_questions_path

    questions_dir = tmp_path / "data" / "questions"
    questions_dir.mkdir(parents=True)
    (questions_dir / "questions_verified.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert resolve_questions_path(None) == "data/questions/questions_verified.jsonl"
    assert question_source(resolve_questions_path(None)) == "verified"

    (questions_dir / "questions_verified.remapped.jsonl").write_text("{}\n", encoding="utf-8")
    assert resolve_questions_path(None) == "data/questions/questions_verified.remapped.jsonl"
    assert question_source(resolve_questions_path(None)) == "verified_remapped"


def test_main_defaults_to_sample_when_only_sample_exists(tmp_path, monkeypatch):
    from main import question_source, resolve_questions_path

    questions_dir = tmp_path / "data" / "questions"
    questions_dir.mkdir(parents=True)
    (questions_dir / "sample_questions.jsonl").write_text(json.dumps({"question_id": "q1"}) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    path = resolve_questions_path(None)
    assert path == "data/questions/sample_questions.jsonl"
    assert question_source(path) == "sample"
