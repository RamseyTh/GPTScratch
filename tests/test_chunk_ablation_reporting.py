from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _write_config_outputs(root: Path, run_id: str, config: str, coverage: float, wrong_section: int) -> None:
    run_dir = root / "outputs" / "runs" / "chunk_ablation" / run_id / config
    run_dir.mkdir(parents=True)
    summary = pd.DataFrame(
        [
            {
                "system": "baseline_gpt",
                "token_f1": 0.1,
                "numeric_accuracy": 0.0,
                "answer_coverage_at_3": float("nan"),
                "average_gold_answer_perplexity": 10.0,
                "average_total_latency": 0.1,
                "average_generation_latency": 0.1,
                "average_retrieval_latency": 0.0,
            },
            {
                "system": "rag_gpt_tfidf_top3",
                "token_f1": 0.2,
                "numeric_accuracy": 0.5,
                "answer_coverage_at_1": coverage,
                "answer_coverage_at_3": coverage,
                "answer_coverage_at_5": coverage,
                "source_accuracy_at_3": 1.0,
                "section_accuracy_at_3": 0.8,
                "table_row_recall_at_3": 1.0,
                "average_gold_answer_perplexity": 8.0,
                "average_total_latency": 0.2,
                "average_generation_latency": 0.15,
                "average_retrieval_latency": 0.05,
            },
        ]
    )
    summary.to_csv(run_dir / "evaluation_summary.csv", index=False)
    diagnostics = [
        {"question_id": f"q{i}", "noise_reason": "wrong_section" if i < wrong_section else "none", "answer_coverage_at_3": i >= wrong_section}
        for i in range(3)
    ]
    (run_dir / "retrieval_diagnostics.jsonl").write_text("\n".join(json.dumps(row) for row in diagnostics) + "\n", encoding="utf-8")
    details = [
        {"question_id": "q1", "system": "rag_gpt_tfidf_top3", "question_type": "numeric_fact", "token_f1": 0.2, "numeric_accuracy": 0.5, "answer_coverage_at_3": coverage, "gold_answer_perplexity": 8.0}
    ]
    (run_dir / "evaluation_details.jsonl").write_text("\n".join(json.dumps(row) for row in details) + "\n", encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps({"chunking_summary": {"num_chunks": 10, "retrievable_count": 9, "average_chunk_length": 180}}),
        encoding="utf-8",
    )


def test_chunk_ablation_report_writes_tables(tmp_path, monkeypatch):
    from src.reporting import build_chunk_ablation_report

    monkeypatch.chdir(tmp_path)
    run_id = "ablation_report"
    _write_config_outputs(tmp_path, run_id, "fixed_128", coverage=0.3, wrong_section=2)
    _write_config_outputs(tmp_path, run_id, "table_aware_clean", coverage=0.8, wrong_section=0)

    report_path = build_chunk_ablation_report(run_id)
    report_dir = tmp_path / "outputs" / "reports" / "chunk_ablation" / run_id

    for filename in (
        "chunk_ablation_summary.csv",
        "retrieval_by_chunk_config.csv",
        "generation_by_chunk_config.csv",
        "question_type_breakdown.csv",
        "failure_by_chunk_config.csv",
        "latency_by_chunk_config.csv",
        "best_config_summary.csv",
        "final_report.md",
    ):
        assert (report_dir / filename).exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Best chunking configuration" in text
    assert "table_aware_clean" in text
