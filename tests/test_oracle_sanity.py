from __future__ import annotations

import pandas as pd


def test_oracle_sanity_passes_when_coverage_and_perplexity_improve():
    from src.evaluation import oracle_sanity_check

    summary = pd.DataFrame(
        [
            {"system": "baseline_gpt", "answer_coverage_at_3": float("nan"), "average_gold_answer_perplexity": 10, "average_total_latency": 1},
            {"system": "oracle_gpt", "answer_coverage_at_3": 1.0, "average_gold_answer_perplexity": 8, "average_total_latency": 1},
        ]
    )
    result = oracle_sanity_check(summary, {"num_questions": 100, "min_research_questions": 50, "is_smoke_test": False})

    assert result["passed"] is True


def test_oracle_sanity_fails_without_answer_coverage():
    from src.evaluation import oracle_sanity_check

    summary = pd.DataFrame(
        [
            {"system": "baseline_gpt", "answer_coverage_at_3": float("nan"), "average_gold_answer_perplexity": 10, "average_total_latency": 1},
            {"system": "oracle_gpt", "answer_coverage_at_3": 0.0, "average_gold_answer_perplexity": 8, "average_total_latency": 1},
        ]
    )
    result = oracle_sanity_check(summary, {"num_questions": 100, "min_research_questions": 50, "is_smoke_test": False})

    assert result["passed"] is False
    assert "oracle_answer_coverage<0.95" in result["reasons"]


def test_strict_oracle_gate_stops_rag_systems(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import src.experiments as experiments
    from src.checkpoint import GPTConfig
    from src.utils import read_jsonl, write_jsonl

    class FakeLocalGPT:
        checkpoint_format = "fake"
        tokenizer_metadata = {}
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
                "checkpoint_path": "fake.pt",
                "tokenizer_dir": "fake_tokenizer",
                "tokenizer_vocab_size": 10,
            }

        def score_loss(self, prompt, target):
            return {"loss": 1.0, "perplexity": 2.0, "num_tokens": 1}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiments, "LocalGPT", FakeLocalGPT)
    monkeypatch.setattr(experiments, "oracle_sanity_check", lambda summary, metadata, latency_warning_seconds=5.0: {"passed": False, "reasons": ["oracle_answer_coverage<0.95"]})
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        questions_path,
        [{"question_id": "q1", "question": "Question?", "answer": "answer", "gold_evidence_text": "", "expected_refusal": False}],
    )
    write_jsonl(chunks_path, [{"chunk_id": "c1", "text": "answer", "retrievable": True}])

    args = SimpleNamespace(
        run_id="oracle_gate",
        chunks=str(chunks_path),
        questions=str(questions_path),
        data_dir="data/raw",
        chunk_size=180,
        overlap=40,
        max_new_tokens=1,
        checkpoint="fake.pt",
        tokenizer_dir="fake_tokenizer",
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
        oracle_first=True,
        strict_oracle_gate=True,
        oracle_latency_warning_seconds=5.0,
        min_research_questions=50,
        limit=None,
        report=False,
        allow_small_final_report=False,
    )
    result = experiments.run_experiment(args)
    predictions = read_jsonl(tmp_path / result["run_dir"] / "predictions.jsonl")
    systems = {row["system"] for row in predictions}

    assert systems == {"baseline_gpt", "oracle_gpt"}
