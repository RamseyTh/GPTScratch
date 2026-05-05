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

SEC_ITEM_ANSWERS = {
    "1",
    "1a",
    "1b",
    "1c",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "7a",
    "8",
    "9",
    "9a",
    "9b",
    "9c",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
}

TOC_MARKERS = (
    "table of contents",
    "part i",
    "part ii",
    "part iii",
    "part iv",
    "item 1 business",
    "item 1a risk factors",
    "item 7 management",
    "item 8 financial statements",
    "item 9b other information",
    "exhibit and financial statement schedules",
    "form 10-k summary",
    "signatures",
)

TEMPLATE_QUESTION_PHRASES = (
    "what value did",
    "what is one important fact",
    "what does the company disclose",
    "why does this section matter",
    "summarize this section",
)


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


def semantic_rejection_reasons(row: dict, chunks_by_id: dict[str, dict], seen: set[str] | None = None) -> list[str]:
    """Return semantic cleaning reasons for noisy verified benchmark rows."""
    seen = seen or set()
    reasons: list[str] = []
    question = str(row.get("question") or "")
    answer = str(row.get("answer") or "")
    qtype = str(row.get("question_type") or "")
    evidence = _combined_evidence(row, chunks_by_id)
    normalized_question = normalize_answer(question)
    normalized_answer = normalize_answer(answer)
    normalized_evidence = normalize_answer(evidence)

    if _question_key(row) in seen:
        reasons.append("duplicate_question")
    if not question or not qtype:
        reasons.append("malformed_question")
    if any(normalize_answer(phrase) in normalized_question for phrase in TEMPLATE_QUESTION_PHRASES):
        reasons.append("vague_question")
    if normalized_answer in SEC_ITEM_ANSWERS:
        reasons.append("toc_item_number_answer" if _toc_like(evidence) or "what value did" in normalized_question else "not_financial_or_operational")
    if _toc_like(evidence) or _toc_like(question):
        reasons.append("table_of_contents_question")
    if _page_number_answer(answer, evidence):
        reasons.append("page_number_answer")
    if _incidental_year_answer(answer, row, question, evidence):
        reasons.append("incidental_year_answer")
    if qtype != "negative" and not evidence:
        reasons.append("missing_evidence")
    if qtype != "negative" and evidence and not _answer_supported(row, evidence):
        reasons.append("unsupported_answer")
    if _boilerplate_evidence(evidence):
        reasons.append("boilerplate_evidence")
    if qtype == "numeric_fact" and "not_financial_or_operational" not in reasons:
        if not _meaningful_numeric_subject(question, answer, evidence):
            reasons.append("not_financial_or_operational")
    return _dedupe_reasons(reasons)


def classify_question_quality(row: dict, chunks_by_id: dict[str, dict], seen: set[str] | None = None) -> tuple[str, str]:
    """Classify a verified question as research-valid, structural-only, or reject."""
    reasons = semantic_rejection_reasons(row, chunks_by_id, seen)
    if not reasons:
        return "research_valid", ""
    structural_priority = [
        "toc_item_number_answer",
        "table_of_contents_question",
        "page_number_answer",
        "incidental_year_answer",
        "boilerplate_evidence",
    ]
    for reason in structural_priority:
        if reason in reasons:
            return "structural_only", reason
    return "reject", reasons[0]


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


def _combined_evidence(row: dict, chunks_by_id: dict[str, dict]) -> str:
    pieces = [str(row.get("gold_evidence_text") or "")]
    for chunk_id in row.get("relevant_chunk_ids") or []:
        chunk = chunks_by_id.get(str(chunk_id))
        if chunk:
            pieces.append(str(chunk.get("text") or ""))
    return " ".join(piece for piece in pieces if piece).strip()


def _toc_like(text: str) -> bool:
    normalized = normalize_answer(text)
    if not normalized:
        return False
    item_hits = len(__import__("re").findall(r"\bitem\s+\d+[a-z]?\b", str(text), __import__("re").I))
    marker_hits = sum(1 for marker in TOC_MARKERS if normalize_answer(marker) in normalized)
    return marker_hits >= 2 or "table of contents" in normalized or item_hits >= 4


def _page_number_answer(answer: str, evidence: str) -> bool:
    normalized = normalize_answer(answer)
    if not normalized.isdigit():
        return False
    value = int(normalized)
    return value <= 400 and _toc_like(evidence)


def _incidental_year_answer(answer: str, row: dict, question: str, evidence: str) -> bool:
    normalized = normalize_answer(answer)
    if not normalized.isdigit() or len(normalized) != 4:
        return False
    year = int(normalized)
    if year < 1900 or year > 2100:
        return False
    normalized_question = normalize_answer(question)
    normalized_evidence = normalize_answer(evidence)
    return normalized == str(row.get("year") or "") or "fiscal year" in normalized_question or "form 10 k" in normalized_evidence


def _answer_supported(row: dict, evidence: str) -> bool:
    answer = str(row.get("answer") or "")
    aliases = [str(alias) for alias in row.get("answer_aliases") or [] if alias]
    if answer_coverage(evidence, {"gold_evidence_text": row.get("gold_evidence_text", ""), "answer": answer, "answer_aliases": aliases}):
        return True
    normalized_evidence = normalize_answer(evidence)
    return any(normalize_answer(alias) in normalized_evidence for alias in [answer, *aliases] if alias)


def _boilerplate_evidence(evidence: str) -> bool:
    normalized = normalize_answer(evidence)
    return any(phrase in normalized for phrase in ("forward looking statements", "documents incorporated by reference", "where you can find more information"))


def _meaningful_numeric_subject(question: str, answer: str, evidence: str) -> bool:
    normalized_answer = normalize_answer(answer)
    if normalized_answer in SEC_ITEM_ANSWERS:
        return False
    if not extract_numeric_values(answer):
        return False
    text = normalize_answer(f"{question} {evidence}")
    good_terms = (
        "sales",
        "revenue",
        "income",
        "margin",
        "employees",
        "cash",
        "expense",
        "assets",
        "liabilities",
        "capital",
        "percent",
        "growth",
    )
    return any(term in text for term in good_terms)


def _dedupe_reasons(reasons: list[str]) -> list[str]:
    out: list[str] = []
    for reason in reasons:
        if reason not in out:
            out.append(reason)
    return out
