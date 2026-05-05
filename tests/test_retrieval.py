from __future__ import annotations


def test_tfidf_retrieves_relevant_chunk():
    from src.retrieval import Retriever

    chunks = [
        {
            "chunk_id": "aapl_1",
            "text": "Apple Services net sales were 109.158 billion dollars in fiscal 2025.",
            "ticker": "AAPL",
            "year": "2025",
            "company": "Apple",
            "source_file": "aapl.txt",
            "chunk_index": 0,
        },
        {
            "chunk_id": "msft_1",
            "text": "Microsoft discusses gaming revenue and cloud products.",
            "ticker": "MSFT",
            "year": "2025",
            "company": "Microsoft",
            "source_file": "msft.txt",
            "chunk_index": 0,
        },
    ]
    retriever = Retriever(chunks)
    results = retriever.retrieve("What were Apple's Services net sales?", top_k=1)
    assert results[0]["chunk_id"] == "aapl_1"


def test_numeric_query_prefers_table_row_and_filters_company():
    from src.retrieval import Retriever, build_retrieval_query, retrieval_filters

    chunks = [
        {
            "chunk_id": "aapl_table",
            "text": "Apple fiscal 2025 table. Services net sales were $109.158 billion in 2025.",
            "ticker": "AAPL",
            "year": "2025",
            "company": "Apple",
            "source_type": "table_row",
            "section_id": "Item 7",
            "retrievable": True,
        },
        {
            "chunk_id": "msft_table",
            "text": "Microsoft Cloud revenue was $100 billion in 2025.",
            "ticker": "MSFT",
            "year": "2025",
            "company": "Microsoft",
            "source_type": "table_row",
            "section_id": "Item 7",
            "retrievable": True,
        },
    ]
    question = {
        "question": "What were Apple's Services net sales in fiscal 2025?",
        "ticker": "AAPL",
        "company": "Apple",
        "year": "2025",
        "question_type": "numeric_fact",
        "row_label": "Services net sales",
    }
    retriever = Retriever(chunks)
    results = retriever.retrieve(build_retrieval_query(question), top_k=3, filters=retrieval_filters(question), question=question)

    assert results[0]["chunk_id"] == "aapl_table"
    assert all(result["metadata"]["ticker"] == "AAPL" for result in results)


def test_risk_query_prefers_item_1a_and_boilerplate_not_indexed():
    from src.retrieval import Retriever, build_retrieval_query

    chunks = [
        {
            "chunk_id": "risk",
            "text": "Apple Item 1A risk factors include supply chain risks.",
            "ticker": "AAPL",
            "year": "2025",
            "source_type": "narrative",
            "section_id": "Item 1A",
            "retrievable": True,
        },
        {
            "chunk_id": "boilerplate",
            "text": "This report contains forward-looking statements.",
            "ticker": "AAPL",
            "year": "2025",
            "source_type": "narrative",
            "section_id": "Front Matter",
            "retrievable": False,
        },
    ]
    question = {
        "question": "What supply chain risk did Apple disclose?",
        "ticker": "AAPL",
        "year": "2025",
        "question_type": "risk_factor",
    }
    retriever = Retriever(chunks)
    results = retriever.retrieve(build_retrieval_query(question), top_k=3, question=question)

    assert [result["chunk_id"] for result in results] == ["risk"]
