from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import normalize_answer
from src.question_validation import classify_question_quality, semantic_rejection_reasons
from src.utils import compact_text, ensure_dir, read_jsonl, write_jsonl


DEFAULT_QUESTIONS = [
    Path("data/questions/questions_verified.remapped.jsonl"),
    Path("data/questions/questions_verified.jsonl"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean verified questions into a research-quality financial QA benchmark.")
    parser.add_argument("--questions", default=None, help="Verified JSONL input. Defaults to remapped verified questions if available.")
    parser.add_argument("--chunks", default="outputs/chunks/chunks.jsonl")
    parser.add_argument("--out", default="data/questions/questions_research.jsonl")
    args = parser.parse_args()

    question_path = resolve_questions(args.questions)
    chunks_path = Path(args.chunks)
    out_path = Path(args.out)

    questions = read_jsonl(question_path)
    chunks = read_jsonl(chunks_path)
    accepted, rejected = clean_questions(questions, chunks)
    write_cleaning_outputs(question_path, chunks_path, out_path, accepted, rejected)

    print("Cleaned verified questions:")
    print(f"  input: {question_path}")
    print(f"  input questions: {len(questions)}")
    print(f"  accepted research_valid: {len(accepted)}")
    print(f"  rejected/structural_only: {len(rejected)}")
    print(f"  output: {out_path}")


def resolve_questions(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Question file not found: {path}")
        return path
    for path in DEFAULT_QUESTIONS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No verified question file found. Expected data/questions/questions_verified.remapped.jsonl "
        "or data/questions/questions_verified.jsonl."
    )


def clean_questions(questions: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunks if chunk.get("chunk_id")}
    seen: set[str] = set()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for raw_row in questions:
        row = standardize_question(raw_row)
        reasons = semantic_rejection_reasons(row, chunks_by_id, seen)
        quality, reject_reason = classify_question_quality(row, chunks_by_id, seen)
        enriched = {
            **row,
            "question_quality": quality,
            "reject_reason": reject_reason,
            "semantic_rejection_reasons": reasons,
        }
        if quality == "research_valid":
            enriched["reject_reason"] = ""
            accepted.append(enriched)
        else:
            rejected.append(enriched)
        seen.add(question_key(row))
    return accepted, rejected


def standardize_question(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    ticker = str(out.get("ticker") or "").upper()
    company_by_ticker = {
        "AAPL": "Apple",
        "AMZN": "Amazon",
        "GOOG": "Alphabet",
        "GOOGL": "Alphabet",
        "META": "Meta",
        "MSFT": "Microsoft",
        "NVDA": "NVIDIA",
        "TSLA": "Tesla",
    }
    if not out.get("year"):
        out["year"] = str(out.get("fiscal_year") or out.get("filing_year") or "")
    if (not out.get("company") or str(out.get("company")).lower() == "unknown") and ticker in company_by_ticker:
        out["company"] = company_by_ticker[ticker]
    if out.get("answer") and not out.get("normalized_answer"):
        out["normalized_answer"] = str(out["answer"])
    if not out.get("answer_aliases") and out.get("question_type") != "negative" and out.get("answer"):
        out["answer_aliases"] = [str(out["answer"])]
    return out


def question_key(row: dict[str, Any]) -> str:
    return str(row.get("question_id") or normalize_answer(row.get("question", "")))


def write_cleaning_outputs(
    question_path: Path,
    chunks_path: Path,
    out_path: Path,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> None:
    out_dir = ensure_dir(out_path.parent)
    write_jsonl(out_path, accepted)
    write_jsonl(out_dir / "rejected_questions_research.jsonl", rejected)
    write_csv(out_dir / "questions_research.csv", accepted)
    write_summary_csv(out_dir / "question_cleaning_summary.csv", accepted, rejected)
    write_report(out_dir / "question_cleaning_report.md", question_path, chunks_path, accepted, rejected)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "question_id",
        "question",
        "answer",
        "normalized_answer",
        "answer_aliases",
        "ticker",
        "company",
        "year",
        "question_type",
        "source_section",
        "source_type",
        "gold_evidence_text",
        "relevant_chunk_ids",
        "expected_refusal",
        "question_quality",
        "reject_reason",
    ]
    fieldnames = preferred + sorted({key for row in rows for key in row.keys()} - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def write_summary_csv(path: Path, accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = [
        {"category": "total", "value": "input_questions", "count": len(accepted) + len(rejected)},
        {"category": "total", "value": "accepted_research_valid", "count": len(accepted)},
        {"category": "total", "value": "rejected_or_structural", "count": len(rejected)},
    ]
    for reason, count in Counter(str(row.get("reject_reason") or "unknown") for row in rejected).most_common():
        rows.append({"category": "rejected_by_reason", "value": reason, "count": count})
    for qtype, count in Counter(str(row.get("question_type") or "unknown") for row in accepted).most_common():
        rows.append({"category": "accepted_by_question_type", "value": qtype, "count": count})
    for ticker, count in Counter(str(row.get("ticker") or "unknown") for row in accepted).most_common():
        rows.append({"category": "accepted_by_ticker", "value": ticker, "count": count})
    for company, count in Counter(str(row.get("company") or "unknown") for row in accepted).most_common():
        rows.append({"category": "accepted_by_company", "value": company, "count": count})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "value", "count"])
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, question_path: Path, chunks_path: Path, accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    rejected_by_reason = Counter(str(row.get("reject_reason") or "unknown") for row in rejected)
    accepted_by_type = Counter(str(row.get("question_type") or "unknown") for row in accepted)
    accepted_by_ticker = Counter(str(row.get("ticker") or "unknown") for row in accepted)
    accepted_by_company = Counter(str(row.get("company") or "unknown") for row in accepted)
    lines = [
        "# Question Cleaning Report",
        "",
        f"- input question file: `{question_path}`",
        f"- chunks file: `{chunks_path}`",
        f"- input question count: {len(accepted) + len(rejected)}",
        f"- accepted research_valid count: {len(accepted)}",
        f"- rejected count: {len(rejected)}",
        "",
        "## Rejected Count By Reason",
        *_counter_lines(rejected_by_reason),
        "",
        "## Accepted Count By Question Type",
        *_counter_lines(accepted_by_type),
        "",
        "## Accepted Count By Ticker",
        *_counter_lines(accepted_by_ticker),
        "",
        "## Accepted Count By Company",
        *_counter_lines(accepted_by_company),
        "",
        "## Examples Of Rejected TOC/Item-Number Questions",
        *_example_lines([row for row in rejected if row.get("reject_reason") in {"toc_item_number_answer", "table_of_contents_question", "page_number_answer", "incidental_year_answer"}]),
        "",
        "## Examples Of Accepted High-Quality Questions",
        *_example_lines(accepted),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _counter_lines(counter: Counter) -> list[str]:
    if not counter:
        return ["- none"]
    return [f"- {key}: {count}" for key, count in counter.most_common()]


def _example_lines(rows: list[dict[str, Any]], limit: int = 8) -> list[str]:
    if not rows:
        return ["- none"]
    lines: list[str] = []
    for row in rows[:limit]:
        reason = row.get("reject_reason")
        suffix = f" ({reason})" if reason else ""
        lines.append(f"- `{row.get('question_id', '')}`{suffix}: {compact_text(str(row.get('question', '')), 180)}")
        lines.append(f"  answer: {compact_text(str(row.get('answer', '')), 120)}")
    return lines


def csv_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


if __name__ == "__main__":
    main()
