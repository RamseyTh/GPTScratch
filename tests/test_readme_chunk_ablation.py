from __future__ import annotations

from pathlib import Path


def test_readme_documents_chunking_ablation():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Chunking Ablation" in readme
    assert "scripts/build_all_chunk_configs.py" in readme
    assert "chunk_ablation_summary.csv" in readme
    for config in (
        "fixed_128",
        "fixed_256",
        "fixed_512",
        "section_180",
        "table_row_only",
        "table_aware_mixed",
        "table_aware_clean",
        "table_aware_clean_context",
    ):
        assert config in readme
    assert "GPT-5" not in readme
    assert "OpenAI API key is required" not in readme
