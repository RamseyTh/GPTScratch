from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd


def test_verified_run_metadata_not_smoke_without_limit():
    from src.evaluation import build_run_metadata

    questions = [
        {
            "question_id": f"q{i}",
            "question_type": ["numeric_fact", "text_fact", "risk_factor", "explanation", "negative"][i % 5],
            "ticker": "AAPL",
            "gold_evidence_text": "evidence" if i % 5 != 4 else "",
            "relevant_chunk_ids": ["c1"] if i % 5 != 4 else [],
            "expected_refusal": i % 5 == 4,
        }
        for i in range(60)
    ]
    predictions = [
        {"question_id": question["question_id"], "system": system}
        for question in questions
        for system in ("baseline_gpt", "rag_gpt_tfidf_top3", "oracle_gpt", "random_context_gpt")
    ]
    summary = pd.DataFrame(
        [
            {"system": "baseline_gpt", "answer_coverage_at_3": float("nan")},
            {"system": "oracle_gpt", "answer_coverage_at_3": 1.0},
            {"system": "rag_gpt_tfidf_top3", "answer_coverage_at_3": 0.75},
            {"system": "random_context_gpt", "answer_coverage_at_3": 0.0},
        ]
    )

    metadata = build_run_metadata(questions, predictions, summary, "data/questions/questions_verified.jsonl")

    assert metadata["is_smoke_test"] is False
    assert metadata["question_source"] == "verified"
    assert metadata["valid_for_research"] is True


def test_limit_keeps_verified_run_as_smoke():
    from src.evaluation import build_run_metadata

    questions = [{"question_id": f"q{i}", "question_type": "numeric_fact", "ticker": "AAPL", "gold_evidence_text": "e", "relevant_chunk_ids": ["c1"]} for i in range(3)]
    predictions = [
        {"question_id": question["question_id"], "system": system}
        for question in questions
        for system in ("baseline_gpt", "rag_gpt_tfidf_top3", "oracle_gpt", "random_context_gpt")
    ]
    summary = pd.DataFrame(
        [
            {"system": "baseline_gpt", "answer_coverage_at_3": float("nan")},
            {"system": "oracle_gpt", "answer_coverage_at_3": 1.0},
            {"system": "rag_gpt_tfidf_top3", "answer_coverage_at_3": 1.0},
            {"system": "random_context_gpt", "answer_coverage_at_3": 0.0},
        ]
    )

    metadata = build_run_metadata(questions, predictions, summary, "data/questions/questions_verified.jsonl", limit=3)

    assert metadata["is_smoke_test"] is True
    assert metadata["valid_for_research"] is False


def test_report_labels_baseline_vs_rag():
    from src.reporting import render_report

    summary = pd.DataFrame(
        [
            {"system": "baseline_gpt", "token_f1": 0.1, "answer_coverage_at_3": float("nan"), "average_gold_answer_perplexity": 10, "average_total_latency": 1.0},
            {"system": "oracle_gpt", "token_f1": 0.2, "answer_coverage_at_3": 1.0, "average_gold_answer_perplexity": 8, "average_total_latency": 1.0},
            {"system": "rag_gpt_tfidf_top3", "token_f1": 0.3, "answer_coverage_at_3": 0.8, "average_gold_answer_perplexity": 7, "average_total_latency": 1.2},
            {"system": "random_context_gpt", "token_f1": 0.0, "answer_coverage_at_3": 0.0, "average_gold_answer_perplexity": 12, "average_total_latency": 1.0},
        ]
    )
    metadata = {
        "num_questions": 60,
        "question_source": "verified",
        "is_smoke_test": False,
        "valid_for_research": True,
        "invalid_reasons": [],
    }
    report = render_report("verified", summary, [], [], {"run_metadata": metadata, "checkpoint_path": "model/model_weights.pt"}, {})

    assert "Baseline GPT" in report
    assert "RAG-GPT TF-IDF top-3" in report
    assert "same GPT architecture" in report
    assert "same checkpoint at model/model_weights.pt" in report


def test_small_fake_verified_run_writes_research_metadata(tmp_path, monkeypatch):
    import src.experiments as experiments
    from src.checkpoint import GPTConfig
    from src.utils import read_json, write_jsonl

    class FakeLocalGPT:
        checkpoint_format = "fake"
        tokenizer_metadata = {"vocab_size": 10}
        gpt_config = GPTConfig(d_model=8, n_heads=1, layers=1, vocab_size=10, max_seq_len=16)

        def __init__(self, config):
            self.config = config
            self.device = "cpu"

            class Tokenizer:
                vocab_size = 10

                def encode(self, text):
                    return text.split()

            self.tokenizer = Tokenizer()

        def generate(self, prompt):
            return {
                "text": "answer",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "latency_seconds": 0.0,
                "checkpoint_path": "model/model_weights.pt",
                "tokenizer_dir": "model/hftokenizer",
                "tokenizer_vocab_size": 10,
            }

        def score_loss(self, prompt, target):
            return {"loss": 1.0, "perplexity": 2.0, "num_tokens": 1}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiments, "LocalGPT", FakeLocalGPT)
    questions = []
    chunks = []
    qtypes = ["numeric_fact", "text_fact", "risk_factor", "explanation", "negative"]
    for i in range(60):
        qtype = qtypes[i % len(qtypes)]
        question_id = f"q{i}"
        chunk_id = f"c{i}"
        expected_refusal = qtype == "negative"
        questions.append(
            {
                "question_id": question_id,
                "question": f"What is verified fact {i}?",
                "answer": "answer" if not expected_refusal else "not enough information",
                "answer_aliases": ["answer"],
                "ticker": "AAPL",
                "company": "Apple",
                "year": "2025",
                "question_type": qtype,
                "gold_evidence_text": f"verified fact {i} answer" if not expected_refusal else "",
                "relevant_chunk_ids": [chunk_id] if not expected_refusal else [],
                "expected_refusal": expected_refusal,
            }
        )
        chunks.append({"chunk_id": chunk_id, "text": f"verified fact {i} answer", "ticker": "AAPL", "year": "2025", "retrievable": True})
    questions_path = tmp_path / "data" / "questions" / "questions_verified.jsonl"
    chunks_path = tmp_path / "outputs" / "chunks" / "chunks.jsonl"
    write_jsonl(questions_path, questions)
    write_jsonl(chunks_path, chunks)

    args = SimpleNamespace(
        run_id="verified_fake",
        chunks=str(chunks_path),
        questions=str(questions_path),
        question_source="verified",
        data_dir="data/raw",
        chunk_size=180,
        overlap=40,
        max_new_tokens=1,
        checkpoint="model/model_weights.pt",
        tokenizer_dir="model/hftokenizer",
        d_model=None,
        n_heads=None,
        layers=None,
        vocab_size=None,
        max_seq_len=None,
        device="cpu",
        temperature=1.0,
        top_k_sampling=50,
        greedy=True,
        allow_random_init=True,
        retrieval_method="tfidf",
        retrieval_top_k=3,
        context_token_budget=100,
        open_corpus=False,
        numeric_extraction=True,
        numeric_extraction_only=False,
        oracle_first=False,
        strict_oracle_gate=False,
        oracle_latency_warning_seconds=5.0,
        min_research_questions=50,
        limit=None,
        report=True,
        allow_small_final_report=False,
        allow_small_research_report=False,
        strict_research_report=False,
        ablation=False,
    )
    result = experiments.run_experiment(args)
    metadata = read_json(tmp_path / result["run_dir"] / "run_metadata.json")
    report = (tmp_path / result["report_dir"] / "final_report.md").read_text(encoding="utf-8")

    assert metadata["num_questions"] == 60
    assert metadata["is_smoke_test"] is False
    assert metadata["question_source"] == "verified"
    assert "Baseline GPT" in report
    assert "RAG-GPT TF-IDF top-3" in report
