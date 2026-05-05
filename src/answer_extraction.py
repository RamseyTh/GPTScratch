from __future__ import annotations

import re
from typing import Any

from .evaluation import normalize_answer, numeric_match


MONEY_PATTERN = re.compile(
    r"\$?\s*[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:billion|million|thousand|bn|mm|m|k)?",
    re.IGNORECASE,
)
PERCENT_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?\s*(?:%|percent)", re.IGNORECASE)


def extract_numeric_answer(question: dict, context_chunks: list[dict]) -> dict[str, Any] | None:
    if question.get("question_type") == "negative":
        return None
    chunks = _rank_chunks_for_question(question, context_chunks)
    aliases = [str(alias) for alias in question.get("answer_aliases", []) if alias]
    if question.get("normalized_answer"):
        aliases.insert(0, str(question["normalized_answer"]))
    if question.get("answer"):
        aliases.insert(0, str(question["answer"]))

    for chunk in chunks:
        text = str(chunk.get("text", ""))
        for alias in aliases:
            if alias and normalize_answer(alias) in normalize_answer(text):
                return _result(alias, chunk, "alias_match")

    for chunk in chunks:
        text = str(chunk.get("text", ""))
        candidates = numeric_candidates(text)
        row_label = str(question.get("row_label") or "")
        if row_label and row_label.lower() not in text.lower():
            continue
        for candidate in candidates:
            if aliases and numeric_match(candidate, aliases):
                return _result(candidate, chunk, "normalized_numeric_match")
        if candidates:
            return _result(candidates[0], chunk, "first_numeric_candidate")
    return None


def numeric_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in (MONEY_PATTERN, PERCENT_PATTERN):
        for match in pattern.finditer(text):
            candidate = " ".join(match.group(0).split())
            if candidate and any(ch.isdigit() for ch in candidate):
                candidates.append(candidate)
    return _dedupe(candidates)


def normalize_numeric_phrase(text: str) -> str:
    lowered = str(text).lower().replace(",", "").replace("$", "").strip()
    percent = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(%|percent)", lowered)
    if percent:
        return f"{float(percent.group(1)):g} percent"
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(billion|million|thousand|bn|mm|m|k)?", lowered)
    if not match:
        return lowered
    value = float(match.group(1))
    unit = match.group(2) or ""
    multiplier = {
        "billion": 1000.0,
        "bn": 1000.0,
        "million": 1.0,
        "mm": 1.0,
        "m": 1.0,
        "thousand": 0.001,
        "k": 0.001,
        "": 1.0,
    }.get(unit, 1.0)
    return f"{value * multiplier:g} million"


def _rank_chunks_for_question(question: dict, chunks: list[dict]) -> list[dict]:
    row_label = str(question.get("row_label") or "")
    if not row_label:
        return chunks
    return sorted(chunks, key=lambda chunk: row_label.lower() in str(chunk.get("text", "")).lower(), reverse=True)


def _result(answer: str, chunk: dict, method: str) -> dict[str, Any]:
    return {
        "answer": answer,
        "normalized_answer": normalize_numeric_phrase(answer),
        "source_chunk_id": chunk.get("chunk_id"),
        "method": method,
    }


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out
