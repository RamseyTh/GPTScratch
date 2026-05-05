from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.question_validation import benchmark_invalid_reasons, question_balance, validate_question_rows
from src.utils import ensure_dir, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate evidence-grounded financial QA questions.")
    parser.add_argument("--questions", default="data/questions/questions.jsonl")
    parser.add_argument("--chunks", default="outputs/chunks/chunks.jsonl")
    parser.add_argument("--out", default="outputs/questions/validated_questions.jsonl")
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    chunks = read_jsonl(args.chunks)
    valid, rejected, audit = validate_question_rows(questions, chunks)

    out_path = Path(args.out)
    out_dir = ensure_dir(out_path.parent)
    write_jsonl(out_path, valid)
    write_jsonl(out_dir / "rejected_questions.jsonl", rejected)
    with (out_dir / "question_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question_type", "accepted", "reasons"])
        writer.writeheader()
        writer.writerows(audit)
    _write_evidence_audit(out_dir / "evidence_audit.md", valid, rejected)

    print(f"valid questions: {len(valid)}")
    print(f"rejected questions: {len(rejected)}")
    print(f"question balance: {question_balance(valid)}")
    invalid_reasons = benchmark_invalid_reasons(valid)
    if invalid_reasons:
        print(f"WARNING: benchmark is not research-valid: {', '.join(invalid_reasons)}")


def _write_evidence_audit(path: Path, valid: list[dict], rejected: list[dict]) -> None:
    lines = ["# Evidence Audit", "", f"Valid questions: {len(valid)}", f"Rejected questions: {len(rejected)}", ""]
    for row in rejected[:20]:
        lines.append(f"- `{row.get('question_id', '')}` rejected: {', '.join(row.get('rejection_reasons', []))}")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
