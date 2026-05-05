from __future__ import annotations

import inspect
import json
from pathlib import Path


def test_research_questions_preferred_over_verified_and_fallback(tmp_path, monkeypatch):
    from src.utils import resolve_question_file

    qdir = tmp_path / "data" / "questions"
    qdir.mkdir(parents=True)
    (qdir / "questions.jsonl").write_text("{}\n", encoding="utf-8")
    (qdir / "questions_verified.remapped.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    path, source = resolve_question_file()
    assert path == Path("data/questions/questions_verified.remapped.jsonl")
    assert source == "verified_remapped"

    (qdir / "questions_research.validated.jsonl").write_text("{}\n", encoding="utf-8")
    path, source = resolve_question_file()
    assert path == Path("data/questions/questions_research.validated.jsonl")
    assert source == "research_validated"


def test_main_stays_as_entry_point_only():
    import main

    source = inspect.getsource(main)
    assert "PipelineRunner" in source
    for forbidden in ("class Retriever", "LocalGPT(", "evaluate_predictions(", "retrieval_context("):
        assert forbidden not in source
