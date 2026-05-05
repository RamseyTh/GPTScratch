from __future__ import annotations

from types import SimpleNamespace


def test_chunk_ablation_runner_iterates_configs(tmp_path, monkeypatch):
    import src.experiments as experiments
    from src.utils import ensure_dir, write_json

    monkeypatch.chdir(tmp_path)
    calls: list[str] = []

    def fake_run_config(args, config):
        calls.append(config)
        run_dir = ensure_dir(tmp_path / "outputs" / "runs" / "chunk_ablation" / args.run_id / config)
        for system in ("baseline_gpt", "rag_gpt_tfidf_top3"):
            system_dir = ensure_dir(run_dir / system)
            (system_dir / "predictions.jsonl").write_text("", encoding="utf-8")
            (system_dir / "retrieval_diagnostics.jsonl").write_text("", encoding="utf-8")
        return {"chunk_config": config, "run_dir": str(run_dir), "report_dir": ""}

    monkeypatch.setattr(experiments, "_run_chunk_config", fake_run_config)
    args = SimpleNamespace(
        experiment="chunk_ablation",
        run_id="ablation_smoke",
        chunk_configs="fixed_128,table_aware_clean",
        questions="questions.jsonl",
        checkpoint="model/model_weights.pt",
        tokenizer_dir="model/hftokenizer",
        retrieval_method="tfidf",
        retrieval_top_k=3,
        context_token_budget=700,
        systems=["baseline_gpt", "rag_gpt_tfidf_top3"],
        report=False,
    )
    result = experiments.run_chunk_ablation(args)

    assert calls == ["fixed_128", "table_aware_clean"]
    assert result["chunk_configs"] == ["fixed_128", "table_aware_clean"]
    metadata = tmp_path / "outputs" / "runs" / "chunk_ablation" / "ablation_smoke" / "run_metadata.json"
    assert metadata.exists()
