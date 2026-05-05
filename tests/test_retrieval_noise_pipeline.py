from __future__ import annotations


def test_retrieval_noise_resistant_pipeline():
    from src.answer_extraction import extract_numeric_answer
    from src.evaluation import answer_coverage_for_results, source_accuracy_for_results
    from src.prompts import rag_prompt
    from src.retrieval import Retriever, build_retrieval_query, retrieval_filters

    chunks = [
        {
            "chunk_id": "answer_table",
            "text": "Apple fiscal 2025 table. Services net sales were $109.158 billion in 2025.",
            "ticker": "AAPL",
            "company": "Apple",
            "year": "2025",
            "section_id": "Item 7",
            "source_type": "table_row",
            "row_label": "Services net sales",
            "retrievable": True,
        },
        {
            "chunk_id": "boilerplate",
            "text": "This report contains forward-looking statements...",
            "ticker": "AAPL",
            "company": "Apple",
            "year": "2025",
            "section_id": "Front Matter",
            "source_type": "narrative",
            "retrievable": False,
        },
        {
            "chunk_id": "risk",
            "text": "Apple risk factors discuss supply chain risks.",
            "ticker": "AAPL",
            "company": "Apple",
            "year": "2025",
            "section_id": "Item 1A",
            "source_type": "narrative",
            "retrievable": True,
        },
        {
            "chunk_id": "wrong_company",
            "text": "Microsoft Cloud revenue was $100 billion.",
            "ticker": "MSFT",
            "company": "Microsoft",
            "year": "2025",
            "section_id": "Item 7",
            "source_type": "table_row",
            "retrievable": True,
        },
    ]
    question = {
        "question_id": "q1",
        "question": "What were Apple's Services net sales in fiscal 2025?",
        "answer": "$109.158 billion",
        "answer_aliases": ["109.158 billion"],
        "ticker": "AAPL",
        "company": "Apple",
        "year": "2025",
        "question_type": "numeric_fact",
        "row_label": "Services net sales",
    }
    retriever = Retriever(chunks)
    results = retriever.retrieve(build_retrieval_query(question), top_k=3, filters=retrieval_filters(question), question=question)
    prompt = rag_prompt(question["question"], "\n".join(result["text"] for result in results))
    extracted = extract_numeric_answer(question, results)

    ids = [result["chunk_id"] for result in results]
    assert "answer_table" in ids
    assert "boilerplate" not in ids
    assert "wrong_company" not in ids
    assert answer_coverage_for_results(results, question, k=3) is True
    assert source_accuracy_for_results(results, question, k=3) is True
    assert "Services net sales were $109.158 billion" in prompt
    assert extracted is not None
    assert extracted["answer"] == "$109.158 billion"
