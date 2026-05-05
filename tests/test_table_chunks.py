from __future__ import annotations


def test_financial_table_rows_become_self_contained_chunks():
    from src.chunking import chunk_document

    document = {
        "text": (
            "Item 7. Management's Discussion and Analysis\n"
            "Products and Services Performance dollars in millions\n"
            "2025 2024 2023\n"
            "Services $109,158 $96,169 $85,200\n"
            "Total net sales $416,161 $391,035 $383,285\n"
        ),
        "source_file": "AAPL_Apple_2025_10K.txt",
        "ticker": "AAPL",
        "company": "Apple",
        "year": "2025",
    }
    chunks = chunk_document(document)
    table_rows = [chunk for chunk in chunks if chunk["source_type"] == "table_row"]
    summaries = [chunk for chunk in chunks if chunk["source_type"] == "table_summary"]

    assert table_rows
    services = next(chunk for chunk in table_rows if chunk["row_label"] == "Services")
    assert services["section_id"] == "Item 7"
    assert services["retrievable"] is True
    assert "$109.158 billion" in services["text"]
    assert "2025" in services["text"]
    assert summaries
