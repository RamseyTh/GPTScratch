from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.answer_extraction import normalize_numeric_phrase, numeric_candidates
from src.question_validation import validate_question_rows
from src.utils import compact_text, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an evidence-grounded research question set from chunks.")
    parser.add_argument("--chunks", default="outputs/chunks/chunks.jsonl")
    parser.add_argument("--out", default="data/questions/research_questions.jsonl")
    parser.add_argument("--validated-out", default="outputs/questions/validated_questions.jsonl")
    parser.add_argument("--target", type=int, default=120)
    args = parser.parse_args()

    chunks = read_jsonl(args.chunks)
    questions = build_questions(chunks, target=args.target)
    write_jsonl(args.out, questions)
    valid, rejected, _ = validate_question_rows(questions, chunks)
    write_jsonl(args.validated_out, valid)
    write_jsonl(Path(args.validated_out).parent / "rejected_questions.jsonl", rejected)
    print(f"wrote questions: {len(questions)} to {args.out}")
    print(f"valid questions: {len(valid)}")
    print(f"rejected questions: {len(rejected)}")
    if len(valid) < 50:
        print("WARNING: fewer than 50 valid questions; this benchmark is a smoke test only.")


def build_questions(chunks: list[dict], target: int = 120) -> list[dict]:
    questions: list[dict] = []
    questions.extend(_numeric_questions(chunks, target=35))
    questions.extend(_text_questions(chunks, target=25))
    questions.extend(_risk_questions(chunks, target=25))
    questions.extend(_explanation_questions(chunks, target=25))
    questions.extend(_negative_questions(chunks, target=15))
    return questions[:target]


def _numeric_questions(chunks: list[dict], target: int) -> list[dict]:
    rows = [chunk for chunk in chunks if chunk.get("source_type") == "table_row" and chunk.get("retrievable", True)]
    questions = []
    for chunk in rows:
        candidates = numeric_candidates(str(chunk.get("text", "")))
        if not candidates:
            continue
        answer = candidates[0]
        row_label = str(chunk.get("row_label") or "the reported row")
        company = chunk.get("company") or chunk.get("ticker")
        year = str(chunk.get("year") or "2025")
        qid = f"{chunk.get('ticker', 'UNK')}_{year}_NUM_{len(questions)+1:03d}"
        questions.append(
            {
                "question_id": qid,
                "question": f"What was {company}'s {row_label} in fiscal {year}?",
                "answer": answer,
                "normalized_answer": normalize_numeric_phrase(answer),
                "answer_aliases": [answer, normalize_numeric_phrase(answer)],
                "ticker": chunk.get("ticker", ""),
                "company": company,
                "year": year,
                "question_type": "numeric_fact",
                "source_section": chunk.get("section_id", ""),
                "source_type": "table_row",
                "gold_evidence_text": answer,
                "relevant_chunk_ids": [chunk.get("chunk_id")],
                "expected_refusal": False,
                "row_label": row_label,
            }
        )
        if len(questions) >= target:
            break
    return questions


def _text_questions(chunks: list[dict], target: int) -> list[dict]:
    rows = [chunk for chunk in chunks if chunk.get("source_type") == "narrative" and chunk.get("section_id") == "Item 1" and chunk.get("retrievable", True)]
    questions = []
    for chunk in rows:
        company = chunk.get("company") or chunk.get("ticker")
        year = str(chunk.get("year") or "2025")
        answer = _first_good_sentence(chunk.get("text", ""))
        if not answer:
            continue
        questions.append(_question(chunk, f"{chunk.get('ticker', 'UNK')}_{year}_TXT_{len(questions)+1:03d}", f"What business information did {company} report in fiscal {year}?", answer, "text_fact"))
        if len(questions) >= target:
            break
    return questions


def _risk_questions(chunks: list[dict], target: int) -> list[dict]:
    rows = [chunk for chunk in chunks if chunk.get("section_id") == "Item 1A" and chunk.get("retrievable", True)]
    questions = []
    for chunk in rows:
        company = chunk.get("company") or chunk.get("ticker")
        year = str(chunk.get("year") or "2025")
        answer = _first_good_sentence(chunk.get("text", ""))
        if not answer:
            continue
        questions.append(_question(chunk, f"{chunk.get('ticker', 'UNK')}_{year}_RISK_{len(questions)+1:03d}", f"What risk factor did {company} discuss in fiscal {year}?", answer, "risk_factor"))
        if len(questions) >= target:
            break
    return questions


def _explanation_questions(chunks: list[dict], target: int) -> list[dict]:
    rows = [chunk for chunk in chunks if chunk.get("section_id") == "Item 7" and chunk.get("retrievable", True)]
    questions = []
    for chunk in rows:
        text = str(chunk.get("text", ""))
        if not re.search(r"increased|decreased|driven|primarily|due to", text, re.I):
            continue
        company = chunk.get("company") or chunk.get("ticker")
        year = str(chunk.get("year") or "2025")
        answer = _first_good_sentence(text)
        questions.append(_question(chunk, f"{chunk.get('ticker', 'UNK')}_{year}_EXP_{len(questions)+1:03d}", f"What drove a reported change for {company} in fiscal {year}?", answer, "explanation"))
        if len(questions) >= target:
            break
    return questions


def _negative_questions(chunks: list[dict], target: int) -> list[dict]:
    companies = []
    for chunk in chunks:
        key = (chunk.get("ticker", ""), chunk.get("company", ""), str(chunk.get("year") or "2025"))
        if key[0] and key not in companies:
            companies.append(key)
    unsupported = {
        "AAPL": "Cloud Infrastructure reportable segment",
        "NVDA": "consumer banking segment",
        "GOOG": "iPhone net sales",
        "MSFT": "automotive leasing segment",
        "AMZN": "GPU hardware segment",
        "META": "retail grocery segment",
        "TSLA": "search advertising segment",
    }
    questions = []
    for ticker, company, year in companies:
        subject = unsupported.get(ticker, "consumer banking segment")
        questions.append(
            {
                "question_id": f"{ticker}_{year}_NEG_{len(questions)+1:03d}",
                "question": f"Did {company or ticker} report a {subject} in fiscal {year}?",
                "answer": "not enough information",
                "normalized_answer": "not enough information",
                "answer_aliases": ["not enough information", "insufficient information"],
                "ticker": ticker,
                "company": company,
                "year": year,
                "question_type": "negative",
                "source_section": "",
                "source_type": "",
                "gold_evidence_text": "",
                "relevant_chunk_ids": [],
                "expected_refusal": True,
            }
        )
        if len(questions) >= target:
            break
    return questions


def _question(chunk: dict, question_id: str, text: str, answer: str, qtype: str) -> dict:
    return {
        "question_id": question_id,
        "question": text,
        "answer": answer,
        "normalized_answer": answer.lower(),
        "answer_aliases": [answer],
        "ticker": chunk.get("ticker", ""),
        "company": chunk.get("company", ""),
        "year": str(chunk.get("year") or "2025"),
        "question_type": qtype,
        "source_section": chunk.get("section_id", ""),
        "source_type": chunk.get("source_type", "narrative"),
        "gold_evidence_text": compact_text(answer, 240),
        "relevant_chunk_ids": [chunk.get("chunk_id")],
        "expected_refusal": False,
    }


def _first_good_sentence(text: str) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", str(text)):
        sentence = sentence.strip()
        if 8 <= len(sentence.split()) <= 45 and not sentence.lower().startswith("item "):
            return sentence
    return compact_text(str(text), 220)


if __name__ == "__main__":
    main()
