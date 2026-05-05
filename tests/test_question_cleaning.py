from __future__ import annotations

from pathlib import Path


def _chunk(chunk_id: str, text: str, **metadata):
    return {"chunk_id": chunk_id, "text": text, "retrievable": True, **metadata}


def test_toc_item_number_question_is_rejected():
    from scripts.clean_verified_questions import clean_questions

    questions = [
        {
            "question_id": "bad_toc_4",
            "question": "What value did Apple report for legal proceedings item mine safety disclosures part ii in fiscal 2025?",
            "answer": "4",
            "question_type": "numeric_fact",
            "ticker": "AAPL",
            "company": "Apple",
            "year": "2025",
            "gold_evidence_text": "TABLE OF CONTENTS Item 3 Legal Proceedings Item 4 Mine Safety Disclosures Part II",
            "relevant_chunk_ids": ["toc"],
        }
    ]
    chunks = [_chunk("toc", questions[0]["gold_evidence_text"])]
    accepted, rejected = clean_questions(questions, chunks)

    assert not accepted
    assert rejected[0]["question_quality"] == "structural_only"
    assert rejected[0]["reject_reason"] == "toc_item_number_answer"


def test_item_9b_toc_answer_is_rejected():
    from scripts.clean_verified_questions import clean_questions

    questions = [
        {
            "question_id": "bad_9b",
            "question": "What value did Apple report for controls and procedures item other information item c in fiscal 2025?",
            "answer": "9B",
            "question_type": "numeric_fact",
            "ticker": "AAPL",
            "company": "Apple",
            "year": "2025",
            "gold_evidence_text": "Item 9A. Controls and Procedures Item 9B. Other Information Item 9C. Disclosure",
            "relevant_chunk_ids": ["toc"],
        }
    ]
    chunks = [_chunk("toc", questions[0]["gold_evidence_text"])]
    accepted, rejected = clean_questions(questions, chunks)

    assert not accepted
    assert rejected[0]["reject_reason"] == "toc_item_number_answer"


def test_incidental_year_answer_is_rejected():
    from scripts.clean_verified_questions import clean_questions

    questions = [
        {
            "question_id": "bad_year",
            "question": "What value did Apple report for fiscal year in fiscal 2025?",
            "answer": "2025",
            "question_type": "numeric_fact",
            "ticker": "AAPL",
            "company": "Apple",
            "year": "2025",
            "gold_evidence_text": "Apple Inc. Form 10-K fiscal 2025",
            "relevant_chunk_ids": ["cover"],
        }
    ]
    chunks = [_chunk("cover", questions[0]["gold_evidence_text"])]
    accepted, rejected = clean_questions(questions, chunks)

    assert not accepted
    assert rejected[0]["reject_reason"] == "incidental_year_answer"


def test_employee_count_question_is_accepted():
    from scripts.clean_verified_questions import clean_questions

    text = "Apple (AAPL) fiscal 2025 Item 1 Business. As of September 27, 2025, the Company had approximately 164,000 full-time equivalent employees."
    questions = [
        {
            "question_id": "good_employees",
            "question": "How many full-time equivalent employees did Apple have as of September 27, 2025?",
            "answer": "164,000",
            "normalized_answer": "164000 employees",
            "answer_aliases": ["164000", "164,000 employees"],
            "question_type": "numeric_fact",
            "ticker": "AAPL",
            "company": "Apple",
            "year": "2025",
            "source_section": "Item 1",
            "source_type": "narrative",
            "gold_evidence_text": "approximately 164,000 full-time equivalent employees",
            "relevant_chunk_ids": ["employees"],
        }
    ]
    accepted, rejected = clean_questions(questions, [_chunk("employees", text)])

    assert accepted
    assert not rejected
    assert accepted[0]["question_quality"] == "research_valid"


def test_services_net_sales_question_is_accepted():
    from scripts.clean_verified_questions import clean_questions

    text = "Apple (AAPL) fiscal 2025 Products and Services Performance table. Services net sales were $109.158 billion in 2025, $96.169 billion in 2024, and $85.200 billion in 2023. Unit: USD millions."
    questions = [
        {
            "question_id": "good_services",
            "question": "What were Apple's Services net sales in fiscal 2025?",
            "answer": "$109.158 billion",
            "normalized_answer": "109158 million USD",
            "answer_aliases": ["$109,158 million", "109158 million"],
            "question_type": "numeric_fact",
            "ticker": "AAPL",
            "company": "Apple",
            "year": "2025",
            "source_section": "Item 7",
            "source_type": "table_row",
            "gold_evidence_text": "Services net sales were $109.158 billion",
            "relevant_chunk_ids": ["services"],
        }
    ]
    accepted, rejected = clean_questions(questions, [_chunk("services", text)])

    assert accepted
    assert not rejected


def test_plausible_negative_question_is_accepted():
    from scripts.clean_verified_questions import clean_questions

    questions = [
        {
            "question_id": "good_negative",
            "question": "Did Apple report a Cloud Infrastructure reportable segment in fiscal 2025?",
            "answer": "not enough information",
            "normalized_answer": "not enough information",
            "question_type": "negative",
            "ticker": "AAPL",
            "company": "Apple",
            "year": "2025",
            "expected_refusal": True,
            "relevant_chunk_ids": [],
        }
    ]
    accepted, rejected = clean_questions(questions, [])

    assert accepted
    assert not rejected


def test_cleaning_report_is_generated(tmp_path: Path):
    from scripts.clean_verified_questions import clean_questions, write_cleaning_outputs

    text = "Apple (AAPL) fiscal 2025 Products and Services Performance table. Services net sales were $109.158 billion in 2025. Unit: USD millions."
    questions = [
        {
            "question_id": "good_services",
            "question": "What were Apple's Services net sales in fiscal 2025?",
            "answer": "$109.158 billion",
            "normalized_answer": "109158 million USD",
            "answer_aliases": ["$109,158 million"],
            "question_type": "numeric_fact",
            "ticker": "AAPL",
            "company": "Apple",
            "year": "2025",
            "source_section": "Item 7",
            "source_type": "table_row",
            "gold_evidence_text": "Services net sales were $109.158 billion",
            "relevant_chunk_ids": ["services"],
        }
    ]
    accepted, rejected = clean_questions(questions, [_chunk("services", text)])
    out = tmp_path / "questions_research.jsonl"
    write_cleaning_outputs(Path("input.jsonl"), Path("chunks.jsonl"), out, accepted, rejected)

    assert out.exists()
    assert (tmp_path / "questions_research.csv").exists()
    assert (tmp_path / "rejected_questions_research.jsonl").exists()
    assert (tmp_path / "question_cleaning_report.md").exists()
    assert "accepted research_valid count" in (tmp_path / "question_cleaning_report.md").read_text(encoding="utf-8")
