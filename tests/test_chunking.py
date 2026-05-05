from __future__ import annotations


def test_section_aware_chunks_do_not_cross_item_boundaries():
    from src.chunking import chunk_document

    document = {
        "text": (
            "Item 1. Business\n"
            "Apple sells products and services. " * 20
            + "\nItem 1A. Risk Factors\n"
            + "Supply chain risks may affect Apple. " * 20
        ),
        "source_file": "AAPL_Apple_2025_10K.txt",
        "ticker": "AAPL",
        "company": "Apple",
        "year": "2025",
    }
    chunks = chunk_document(document, chunk_size=40, overlap=5)
    sections = {chunk["section_id"] for chunk in chunks}

    assert "Item 1" in sections
    assert "Item 1A" in sections
    assert all("Item 1A" not in chunk["text"] for chunk in chunks if chunk["section_id"] == "Item 1")


def test_boilerplate_chunk_is_not_retrievable():
    from src.chunking import chunk_document

    document = {
        "text": "Item 1. Business\nForward-looking statements are subject to risks and uncertainties.",
        "source_file": "AAPL_Apple_2025_10K.txt",
        "ticker": "AAPL",
        "company": "Apple",
        "year": "2025",
    }
    chunks = chunk_document(document)

    assert chunks
    assert any(chunk["retrievable"] is False for chunk in chunks)
