from __future__ import annotations

import json
import sys
from pathlib import Path


def test_quick_run_uses_verified_questions_over_default(tmp_path, monkeypatch):
    metadata = _run_fake_quick_run(tmp_path, monkeypatch, verified_count=60, default_count=3, run_id="quick_verified")

    assert metadata["question_source"] == "verified"
    assert metadata["num_questions"] == 60
    assert metadata["is_smoke_test"] is False


def test_quick_run_prefers_remapped_verified(tmp_path, monkeypatch):
    metadata = _run_fake_quick_run(tmp_path, monkeypatch, verified_count=60, remapped_count=61, run_id="quick_remapped")

    assert metadata["question_source"] == "verified_remapped"
    assert metadata["num_questions"] == 61


def test_quick_run_limit_marks_smoke(tmp_path, monkeypatch):
    metadata = _run_fake_quick_run(tmp_path, monkeypatch, verified_count=60, run_id="quick_limit", extra_args=["--limit", "3"])

    assert metadata["num_questions"] == 3
    assert metadata["is_smoke_test"] is True
    assert metadata["valid_for_research"] is False


def test_quick_run_falls_back_to_sample(tmp_path, monkeypatch):
    metadata = _run_fake_quick_run(tmp_path, monkeypatch, sample_count=3, run_id="quick_sample")

    assert metadata["question_source"] == "sample"
    assert metadata["is_smoke_test"] is True
    assert metadata["valid_for_research"] is False


def _run_fake_quick_run(
    tmp_path,
    monkeypatch,
    run_id: str,
    verified_count: int | None = None,
    remapped_count: int | None = None,
    default_count: int | None = None,
    sample_count: int | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    import main
    import src.pipeline as pipeline
    from src.utils import read_json

    monkeypatch.chdir(tmp_path)
    qdir = tmp_path / "data" / "questions"
    qdir.mkdir(parents=True)
    (tmp_path / "outputs" / "chunks").mkdir(parents=True)
    (tmp_path / "outputs" / "chunks" / "chunks.jsonl").write_text('{"chunk_id":"c1","text":"evidence","retrievable":true}\n', encoding="utf-8")
    if verified_count is not None:
        _write_questions(qdir / "questions_verified.jsonl", verified_count)
    if remapped_count is not None:
        _write_questions(qdir / "questions_verified.remapped.jsonl", remapped_count)
    if default_count is not None:
        _write_questions(qdir / "questions.jsonl", default_count)
    if sample_count is not None:
        _write_questions(qdir / "sample_questions.jsonl", sample_count)

    monkeypatch.setattr(pipeline.PipelineRunner, "check_tokenizer_and_checkpoint", lambda self: setattr(self, "compatibility_pass", True))
    monkeypatch.setattr(pipeline.PipelineRunner, "ensure_chunks", lambda self: None)
    monkeypatch.setattr(pipeline, "run_experiment", _fake_run_experiment)
    argv = ["main.py", "--quick-run", "--run-id", run_id, "--report", *(extra_args or [])]
    monkeypatch.setattr(sys, "argv", argv)
    main.main()
    return read_json(tmp_path / "outputs" / "runs" / run_id / "run_metadata.json")


def _write_questions(path: Path, count: int) -> None:
    rows = []
    qtypes = ["numeric_fact", "text_fact", "risk_factor", "explanation", "negative"]
    for index in range(count):
        qtype = qtypes[index % len(qtypes)]
        expected_refusal = qtype == "negative"
        rows.append(
            {
                "question_id": f"q{index}",
                "question": f"Question {index}?",
                "answer": "answer" if not expected_refusal else "not enough information",
                "answer_aliases": ["answer"],
                "ticker": "AAPL",
                "company": "Apple",
                "year": "2025",
                "question_type": qtype,
                "gold_evidence_text": "evidence" if not expected_refusal else "",
                "relevant_chunk_ids": ["c1"] if not expected_refusal else [],
                "expected_refusal": expected_refusal,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _fake_run_experiment(args) -> dict:
    import pandas as pd

    from src.evaluation import build_run_metadata
    from src.reporting import build_report
    from src.utils import ensure_dir, read_jsonl, write_json, write_jsonl

    run_dir = ensure_dir(Path("outputs") / "runs" / args.run_id)
    report_dir = ensure_dir(Path("outputs") / "reports" / args.run_id)
    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    systems = args.systems
    predictions = [
        {
            "question_id": question["question_id"],
            "system": system,
            "question": question["question"],
            "gold_answer": question["answer"],
            "prediction": question["answer"],
            "retrieved_context": "evidence" if system in {"rag_gpt_tfidf_top3", "oracle_gpt"} else "",
            "retrieved_texts": ["evidence"] if system == "rag_gpt_tfidf_top3" else [],
            "retrieved_metadata": [{"ticker": "AAPL", "year": "2025", "source_type": "table_row"}] if system == "rag_gpt_tfidf_top3" else [],
            "generation_latency_seconds": 0.01,
            "retrieval_latency_seconds": 0.01 if system == "rag_gpt_tfidf_top3" else 0.0,
            "total_latency_seconds": 0.02,
            "gold_answer_loss": 1.0,
            "gold_answer_perplexity": 2.0,
        }
        for question in questions
        for system in systems
    ]
    summary = pd.DataFrame(
        [
            {"system": "baseline_gpt", "num_questions": len(questions), "token_f1": 0.2, "numeric_accuracy": 0.1, "answer_coverage_at_3": float("nan"), "average_gold_answer_perplexity": 3.0, "average_total_latency": 0.01},
            {"system": "rag_gpt_tfidf_top3", "num_questions": len(questions), "token_f1": 0.3, "numeric_accuracy": 0.2, "answer_coverage_at_3": 0.8, "average_gold_answer_perplexity": 2.0, "average_total_latency": 0.02},
            {"system": "oracle_gpt", "num_questions": len(questions), "token_f1": 0.4, "numeric_accuracy": 0.2, "answer_coverage_at_3": 1.0, "average_gold_answer_perplexity": 1.8, "average_total_latency": 0.01},
            {"system": "random_context_gpt", "num_questions": len(questions), "token_f1": 0.0, "numeric_accuracy": 0.0, "answer_coverage_at_3": 0.0, "average_gold_answer_perplexity": 4.0, "average_total_latency": 0.01},
        ]
    )
    metadata = build_run_metadata(
        questions,
        predictions,
        summary,
        args.questions,
        limit=args.limit,
        expected_systems=systems,
        question_source=args.question_source,
    )
    metadata.update({"run_id": args.run_id, "checkpoint": args.checkpoint, "checkpoint_path": args.checkpoint, "tokenizer_dir": args.tokenizer_dir, "systems": systems})
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_jsonl(run_dir / "retrieval_diagnostics.jsonl", [])
    write_jsonl(run_dir / "evaluation_details.jsonl", predictions)
    summary.to_csv(run_dir / "evaluation_summary.csv", index=False)
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(run_dir / "tokenizer_metadata.json", {"vocab_size": 10000, "tokenizer_class": "GPT2TokenizerFast"})
    write_json(
        run_dir / "run_config.json",
        {
            "run_id": args.run_id,
            "checkpoint_path": args.checkpoint,
            "tokenizer_dir": args.tokenizer_dir,
            "model_config": {"vocab_size": 10000},
            "run_metadata": metadata,
            "chunking_summary": {"num_chunks": 1, "by_source_type": {"table_row": 1}, "retrievable_count": 1, "non_retrievable_count": 0},
        },
    )
    if args.report:
        build_report(args.run_id)
    return {"run_dir": str(run_dir), "report_dir": str(report_dir), "report_path": str(report_dir / "final_report.md") if args.report else None}
