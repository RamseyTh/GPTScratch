from __future__ import annotations

import json
import sys
from pathlib import Path

from conftest import build_tiny_hf_tokenizer


def test_smoke_pipeline_runs_with_hf_tokenizer(tmp_path, monkeypatch):
    import torch

    from main import main
    from src.gpt import GPTModel
    from src.hftokenizer import HFTokenizer

    root = Path(__file__).resolve().parents[1]
    tokenizer_dir = build_tiny_hf_tokenizer(tmp_path / "hftokenizer")
    tokenizer = HFTokenizer(tokenizer_dir)
    checkpoint_path = tmp_path / "model_weights.pt"
    torch.save(
        GPTModel(d_model=16, n_heads=8, layers=1, vocab_size=tokenizer.vocab_size, max_seq_len=16).state_dict(),
        checkpoint_path,
    )

    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "question_id": "SMOKE_001",
                "question": "What was Apple revenue?",
                "answer": "10 billion",
                "answer_aliases": ["10 billion"],
                "ticker": "AAPL",
                "company": "Apple",
                "year": "2025",
                "question_type": "numeric_fact",
                "gold_evidence_text": "Apple revenue was 10 billion",
                "expected_refusal": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        json.dumps(
            {
                "chunk_id": "aapl_1",
                "text": "Apple revenue was 10 billion in 2025.",
                "source_file": "aapl.txt",
                "ticker": "AAPL",
                "company": "Apple",
                "year": "2025",
                "chunk_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_id = "smoke_hf_test"
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--questions",
            str(questions_path),
            "--chunks",
            str(chunks_path),
            "--checkpoint",
            str(checkpoint_path),
            "--tokenizer-dir",
            str(tokenizer_dir),
            "--limit",
            "1",
            "--run-id",
            run_id,
            "--report",
            "--max-new-tokens",
            "2",
            "--device",
            "cpu",
        ],
    )
    main()
    assert (root / "outputs" / "runs" / run_id / "predictions.jsonl").exists()
    assert (root / "outputs" / "runs" / run_id / "evaluation_summary.csv").exists()
    assert (root / "outputs" / "reports" / run_id / "final_report.md").exists()
