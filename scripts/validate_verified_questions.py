"""Remap verified question evidence IDs to the current chunk file."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import normalize_answer
from src.utils import compact_text, ensure_dir, read_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remap verified question evidence to the current chunk IDs.")
    parser.add_argument("--questions", default="data/questions/questions_verified.jsonl")
    parser.add_argument("--chunks", default="outputs/chunks/chunks.jsonl")
    parser.add_argument("--out", default="data/questions/questions_verified.remapped.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = remap_verified_questions(args.questions, args.chunks, args.out)
    print(f"Questions processed: {result['num_questions']}")
    print(f"Evidence valid: {result['evidence_valid_count']}")
    print(f"Evidence missing: {result['evidence_missing_count']}")
    print(f"Evidence match rate: {result['evidence_match_rate']:.4f}")
    print(f"Remapped questions: {result['remapped_count']}")
    print(f"Output: {result['out_path']}")
    if result["evidence_match_rate"] < 0.95:
        print("WARNING: evidence match rate is below 0.95; evidence quality is weak for research reporting.")


def remap_verified_questions(questions_path: str | Path, chunks_path: str | Path, out_path: str | Path) -> dict:
    questions_path = resolve_project_path(questions_path)
    chunks_path = resolve_project_path(chunks_path)
    out_path = resolve_project_path(out_path)
    questions = read_jsonl(questions_path)
    chunks = read_jsonl(chunks_path)
    if not questions:
        raise FileNotFoundError(f"No questions found: {questions_path}")
    if not chunks:
        raise FileNotFoundError(f"No chunks found: {chunks_path}")

    chunks_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunks}
    remapped: list[dict] = []
    summary_rows: list[dict] = []
    notes = Counter()

    for question in questions:
        row = dict(question)
        expected_refusal = bool(row.get("expected_refusal", False))
        current_ids = [str(item) for item in row.get("relevant_chunk_ids", []) if item]
        existing_ids = [chunk_id for chunk_id in current_ids if chunk_id in chunks_by_id]

        if expected_refusal or row.get("question_type") == "negative":
            row["evidence_valid"] = True
            row.setdefault("validation_notes", "negative question; evidence chunk not required")
            remapped.append(row)
            summary_rows.append(_summary_row(row, "negative", existing_ids))
            notes["negative"] += 1
            continue

        if existing_ids and _question_supported_by_chunks(row, [chunks_by_id[chunk_id] for chunk_id in existing_ids]):
            row["evidence_valid"] = True
            row["gold_chunk_id"] = existing_ids[0]
            row.setdefault("validation_notes", "existing relevant_chunk_ids valid")
            remapped.append(row)
            summary_rows.append(_summary_row(row, "existing_valid", existing_ids))
            notes["existing_valid"] += 1
            continue

        match, score, method = find_best_evidence_chunk(row, chunks)
        if match:
            chunk_id = str(match.get("chunk_id"))
            row["relevant_chunk_ids"] = [chunk_id]
            row["gold_chunk_id"] = chunk_id
            row["evidence_valid"] = True
            row["validation_notes"] = f"relevant_chunk_ids remapped from gold_evidence_text ({method})"
            row["evidence_match_score"] = score
            remapped.append(row)
            summary_rows.append(_summary_row(row, f"remapped_{method}", [chunk_id], score))
            notes["remapped"] += 1
            continue

        row["evidence_valid"] = False
        row["validation_notes"] = "evidence not found in current chunks"
        row["evidence_match_score"] = score
        remapped.append(row)
        summary_rows.append(_summary_row(row, "missing", [], score))
        notes["missing"] += 1

    write_jsonl(out_path, remapped)
    report_dir = ensure_dir(out_path.parent)
    summary_path = report_dir / "evidence_remap_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["question_id", "question_type", "ticker", "year", "status", "matched_chunk_ids", "score", "evidence_valid"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    report_path = report_dir / "evidence_remap_report.md"
    report_path.write_text(render_report(questions_path, chunks_path, out_path, notes, summary_rows), encoding="utf-8")

    answerable_count = sum(1 for row in remapped if row.get("question_type") != "negative" and not row.get("expected_refusal", False))
    negative_count = len(remapped) - answerable_count
    valid_count = sum(1 for row in remapped if row.get("evidence_valid"))
    missing_count = sum(1 for row in remapped if row.get("evidence_valid") is False)
    matched_answerable = answerable_count - missing_count
    evidence_match_rate = matched_answerable / answerable_count if answerable_count else 1.0
    return {
        "num_questions": len(remapped),
        "answerable_questions": answerable_count,
        "negative_questions": negative_count,
        "evidence_valid_count": valid_count,
        "evidence_missing_count": missing_count,
        "evidence_match_rate": evidence_match_rate,
        "remapped_count": notes["remapped"],
        "out_path": str(out_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
    }


def find_best_evidence_chunk(question: dict, chunks: list[dict]) -> tuple[dict | None, float, str]:
    evidence = str(question.get("gold_evidence_text") or "")
    if not evidence:
        return None, 0.0, "no_evidence"
    candidates = _metadata_candidates(question, chunks)
    best_chunk: dict | None = None
    best_score = 0.0
    best_method = "none"
    for chunk in candidates:
        text = str(chunk.get("text") or "")
        score, method = evidence_match_score(evidence, text)
        score += _metadata_bonus(question, chunk)
        if score > best_score:
            best_chunk = chunk
            best_score = score
            best_method = method
    if best_chunk and best_score >= 0.62:
        return best_chunk, round(best_score, 4), best_method
    return None, round(best_score, 4), best_method


def _metadata_candidates(question: dict, chunks: list[dict]) -> list[dict]:
    ticker = str(question.get("ticker") or "").upper()
    year = str(question.get("year") or "")
    source_type = str(question.get("source_type") or "")
    filtered = [
        chunk
        for chunk in chunks
        if (not ticker or not chunk.get("ticker") or str(chunk.get("ticker")).upper() == ticker)
        and (not year or not chunk.get("year") or str(chunk.get("year")) == year)
    ]
    if source_type:
        typed = [chunk for chunk in filtered if str(chunk.get("source_type") or "") == source_type]
        if typed:
            filtered = typed
    return sorted(filtered or chunks, key=lambda chunk: _metadata_penalty(question, chunk))


def evidence_match_score(evidence: str, text: str) -> tuple[float, str]:
    if evidence and evidence in text:
        return 1.0, "exact"
    normalized_evidence = normalize_answer(evidence)
    normalized_text = normalize_answer(text)
    if normalized_evidence and normalized_evidence in normalized_text:
        return 0.95, "normalized"
    evidence_tokens = set(normalized_evidence.split())
    text_tokens = set(normalized_text.split())
    overlap = len(evidence_tokens & text_tokens) / max(1, len(evidence_tokens))
    sequence = SequenceMatcher(None, normalized_evidence, normalized_text[: max(len(normalized_evidence) * 3, 300)]).ratio()
    return max(overlap * 0.8, sequence), "fuzzy"


def _metadata_bonus(question: dict, chunk: dict) -> float:
    bonus = 0.0
    for field in ("ticker", "year"):
        expected = str(question.get(field) or "").lower()
        actual = str(chunk.get(field) or "").lower()
        if expected and actual and expected == actual:
            bonus += 0.04
    section = str(question.get("source_section") or "").lower()
    if section and section in str(chunk.get("section_id") or "").lower():
        bonus += 0.03
    source_type = str(question.get("source_type") or "").lower()
    if source_type and source_type == str(chunk.get("source_type") or "").lower():
        bonus += 0.03
    return bonus


def _metadata_penalty(question: dict, chunk: dict) -> tuple[int, int, int]:
    ticker_mismatch = int(bool(question.get("ticker")) and bool(chunk.get("ticker")) and str(question.get("ticker")).upper() != str(chunk.get("ticker")).upper())
    year_mismatch = int(bool(question.get("year")) and bool(chunk.get("year")) and str(question.get("year")) != str(chunk.get("year")))
    source_mismatch = int(bool(question.get("source_type")) and bool(chunk.get("source_type")) and str(question.get("source_type")) != str(chunk.get("source_type")))
    return ticker_mismatch, year_mismatch, source_mismatch


def _question_supported_by_chunks(question: dict, chunks: list[dict]) -> bool:
    evidence = str(question.get("gold_evidence_text") or "")
    for chunk in chunks:
        score, _ = evidence_match_score(evidence, str(chunk.get("text") or ""))
        if score >= 0.80:
            return True
    return False


def _summary_row(question: dict, status: str, chunk_ids: list[str], score: float | None = None) -> dict:
    return {
        "question_id": question.get("question_id", ""),
        "question_type": question.get("question_type", ""),
        "ticker": question.get("ticker", ""),
        "year": question.get("year", ""),
        "status": status,
        "matched_chunk_ids": ";".join(chunk_ids),
        "score": "" if score is None else score,
        "evidence_valid": question.get("evidence_valid"),
    }


def render_report(questions_path: Path, chunks_path: Path, out_path: Path, notes: Counter, rows: list[dict]) -> str:
    total = len(rows)
    negative = sum(1 for row in rows if row["status"] == "negative")
    answerable = total - negative
    valid = sum(1 for row in rows if row.get("evidence_valid") is True)
    missing = sum(1 for row in rows if row.get("evidence_valid") is False)
    matched_answerable = answerable - missing
    match_rate = matched_answerable / answerable if answerable else 1.0
    examples = [row for row in rows if row["status"] == "missing"][:10]
    lines = [
        "# Evidence Remap Report",
        "",
        f"- questions: `{questions_path}`",
        f"- chunks: `{chunks_path}`",
        f"- output: `{out_path}`",
        f"- total questions: {total}",
        f"- answerable questions: {answerable}",
        f"- negative questions: {negative}",
        f"- evidence matched count: {matched_answerable}",
        f"- evidence missing count: {missing}",
        f"- evidence match rate: {match_rate:.4f}",
        f"- evidence valid rows: {valid}",
        f"- remapped: {notes['remapped']}",
        f"- existing valid: {notes['existing_valid']}",
        "",
    ]
    if match_rate < 0.95:
        lines.extend(["## Warning", "Evidence match rate is below 0.95; evidence quality is weak.", ""])
    lines.append("## Missing Evidence Examples")
    if not examples:
        lines.append("No missing evidence examples.")
    for row in examples:
        lines.append(f"- `{row['question_id']}` ({row['ticker']} {row['year']}): score={row['score']}")
    return "\n".join(lines) + "\n"


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
