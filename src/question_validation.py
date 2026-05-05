from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

from .evaluation import answer_coverage, extract_numeric_values, normalize_answer


VAGUE_QUESTION_PHRASES = (
    "what is one important fact",
    "why does this section matter",
    "what does the company disclose",
    "summarize this section",
)


BALANCE_TARGETS = {
    "numeric_fact": 20,
    "text_fact": 15,
    "risk_factor": 15,
    "explanation": 15,
    "negative": 10,
}


def validate_question_rows(questions: list[dict], chunks: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    chunks_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunks}
    seen: set[str] = set()
    valid: list[dict] = []
    rejected: list[dict] = []
    audit: list[dict] = []
    for row in questions:
        reasons = question_rejection_reasons(row, chunks_by_id, seen)
        question_key = _question_key(row)
        seen.add(question_key)
        audit.append(
            {
                "question_id": row.get("question_id", ""),
                "question_type": row.get("question_type", ""),
                "accepted": not reasons,
                "reasons": "; ".join(reasons),
            }
        )
        if reasons:
            rejected.append({**row, "rejection_reasons": reasons})
        else:
            valid.append(row)
    return valid, rejected, audit


def question_balance(valid_questions: list[dict]) -> dict[str, int]:
    return dict(Counter(str(row.get("question_type", "")) for row in valid_questions))


def benchmark_invalid_reasons(valid_questions: list[dict], min_questions: int = 50) -> list[str]:
    reasons: list[str] = []
    if len(valid_questions) < min_questions:
        reasons.append(f"valid_questions<{min_questions}")
    counts = question_balance(valid_questions)
    for qtype, target in BALANCE_TARGETS.items():
        if counts.get(qtype, 0) < target:
            reasons.append(f"{qtype}<{target}")
    return reasons


def question_rejection_reasons(row: dict, chunks_by_id: dict[str, dict], seen: set[str]) -> list[str]:
    reasons: list[str] = []
    question = str(row.get("question") or "")
    answer = str(row.get("answer") or "")
    qtype = str(row.get("question_type") or "")
    key = _question_key(row)
    normalized_question = normalize_answer(question)

    if key in seen:
        reasons.append("duplicate_question")
    if any(normalize_answer(phrase) in normalized_question for phrase in VAGUE_QUESTION_PHRASES):
        reasons.append("vague_question")
    if answer.lower() in {"", "todo", "tbd", "placeholder", "n/a"}:
        reasons.append("placeholder_or_empty_answer")
    if qtype not in {"numeric_fact", "text_fact", "risk_factor", "explanation", "negative"}:
        reasons.append("invalid_question_type")

    expected_refusal = bool(row.get("expected_refusal", False))
    relevant_ids = [str(item) for item in row.get("relevant_chunk_ids", []) if item]
    if qtype == "negative":
        if not expected_refusal:
            if relevant_ids:
                return reasons
            reasons.append("negative_expected_refusal_required")
        if expected_refusal and normalize_answer(answer) != "not enough information":
            reasons.append("negative_answer_must_be_not_enough_information")
        if expected_refusal and relevant_ids:
            reasons.append("unsupported_negative_should_not_have_relevant_chunks")
        if _is_absurd_negative(question):
            reasons.append("negative_question_not_plausible")
        return reasons

    for field in ("ticker", "company", "year"):
        if not str(row.get(field) or "").strip():
            reasons.append(f"{field}_required")
    if not answer:
        reasons.append("answer_required")
    if not str(row.get("normalized_answer") or "").strip():
        reasons.append("normalized_answer_required")
    if not (row.get("answer_aliases") or []):
        reasons.append("answer_aliases_required")
    if not str(row.get("gold_evidence_text") or "").strip():
        reasons.append("gold_evidence_required")
    if not relevant_ids:
        reasons.append("relevant_chunk_ids_required")
    missing = [chunk_id for chunk_id in relevant_ids if chunk_id not in chunks_by_id]
    if missing:
        reasons.append("relevant_chunk_id_missing")
    if relevant_ids and not missing and not _evidence_supported(row, relevant_ids, chunks_by_id):
        reasons.append("evidence_not_supported_by_relevant_chunks")

    if qtype == "numeric_fact":
        if not extract_numeric_values(answer):
            reasons.append("numeric_answer_required")
        if not _has_unit_when_applicable(answer):
            reasons.append("numeric_units_required")
        if row.get("source_type") not in {"table_row", "table_summary"}:
            reasons.append("numeric_source_type_should_be_table")
    return reasons


def _question_key(row: dict) -> str:
    return str(row.get("question_id") or normalize_answer(row.get("question", "")))


def _evidence_supported(row: dict, relevant_ids: list[str], chunks_by_id: dict[str, dict]) -> bool:
    evidence = str(row.get("gold_evidence_text") or "")
    answer = str(row.get("answer") or "")
    aliases = [str(alias) for alias in row.get("answer_aliases", []) if alias]
    for chunk_id in relevant_ids:
        text = str(chunks_by_id[chunk_id].get("text", ""))
        if answer_coverage(text, {"gold_evidence_text": evidence, "answer": answer, "answer_aliases": aliases}):
            return True
        if _fuzzy_contains(text, evidence):
            return True
    return False


def _fuzzy_contains(text: str, evidence: str, threshold: float = 0.82) -> bool:
    normalized_text = normalize_answer(text)
    normalized_evidence = normalize_answer(evidence)
    if not normalized_evidence:
        return False
    if normalized_evidence in normalized_text:
        return True
    if len(normalized_evidence) > len(normalized_text):
        return SequenceMatcher(None, normalized_text, normalized_evidence).ratio() >= threshold
    window = len(normalized_evidence)
    step = max(1, window // 4)
    for start in range(0, max(1, len(normalized_text) - window + 1), step):
        if SequenceMatcher(None, normalized_text[start : start + window], normalized_evidence).ratio() >= threshold:
            return True
    return False


def _has_unit_when_applicable(answer: str) -> bool:
    lowered = answer.lower()
    if "%" in lowered or "percent" in lowered:
        return True
    if any(unit in lowered for unit in ("million", "billion", "thousand", "$")):
        return True
    values = extract_numeric_values(answer)
    return bool(values) and abs(values[0]) < 100


def _is_absurd_negative(question: str) -> bool:
    normalized = normalize_answer(question)
    absurd_terms = ("banana", "unicorn", "volcano", "dragon")
    return any(term in normalized for term in absurd_terms)
