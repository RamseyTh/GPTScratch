from __future__ import annotations

from pathlib import Path


def _document() -> dict:
    return {
        "text": (
            "TABLE OF CONTENTS Item 1 Business 3 Item 1A Risk Factors 12 Item 7 Management's Discussion and Analysis 40\n"
            "Item 1. Business\n"
            + "Apple sells products and services. " * 30
            + "\nItem 1A. Risk Factors\n"
            + "Supply chain risks may affect Apple. " * 30
            + "\nItem 7. Management's Discussion and Analysis\n"
            + "Products and Services Performance dollars in millions\n"
            + "2025 2024 2023\n"
            + "Services $109,158 $96,169 $85,200\n"
            + "Revenue increased primarily due to higher Services net sales compared to 2024. " * 8
        ),
        "source_file": "AAPL_Apple_2025_10K.txt",
        "ticker": "AAPL",
        "company": "Apple",
        "year": "2025",
    }


def test_fixed_chunk_sizes_differ():
    from src.chunking import chunk_documents_for_config, chunk_stats

    fixed_128 = chunk_documents_for_config([_document()], "fixed_128")
    fixed_512 = chunk_documents_for_config([_document()], "fixed_512")

    assert fixed_128
    assert fixed_512
    assert chunk_stats(fixed_128)["average_chunk_length"] <= chunk_stats(fixed_512)["average_chunk_length"]
    assert {chunk["source_type"] for chunk in fixed_128} == {"fixed"}
    assert all(chunk["chunk_config"] == "fixed_128" for chunk in fixed_128)


def test_table_row_only_excludes_narrative():
    from src.chunking import chunk_documents_for_config

    chunks = chunk_documents_for_config([_document()], "table_row_only")
    source_types = {chunk["source_type"] for chunk in chunks}

    assert source_types
    assert source_types <= {"table_row", "table_summary"}
    assert "narrative" not in source_types


def test_table_aware_clean_filters_table_of_contents():
    from src.chunking import chunk_documents_for_config

    chunks = chunk_documents_for_config([_document()], "table_aware_clean")
    non_retrievable = [chunk for chunk in chunks if chunk.get("retrievable") is False]

    assert non_retrievable
    assert any(chunk.get("non_retrievable_reason") in {"table_of_contents", "navigation_only", "cover_page"} for chunk in non_retrievable)


def test_table_aware_clean_context_adds_local_context():
    from src.chunking import chunk_documents_for_config

    chunks = chunk_documents_for_config([_document()], "table_aware_clean_context")

    assert any(chunk["source_type"] == "local_context" for chunk in chunks)


def test_chunk_audits_are_written(tmp_path: Path):
    from src.chunking import chunk_documents_for_config, write_chunk_audit

    chunks = chunk_documents_for_config([_document()], "table_aware_clean")
    write_chunk_audit(chunks, tmp_path)

    assert (tmp_path / "chunk_audit.csv").exists()
    assert (tmp_path / "chunk_samples.md").exists()
    assert (tmp_path / "non_retrievable_audit.csv").exists()
