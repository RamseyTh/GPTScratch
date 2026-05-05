from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir, write_jsonl


REFUSAL_PHRASES = (
    "not enough information",
    "cannot determine",
    "can't determine",
    "not provided",
    "not in the context",
    "insufficient information",
    "unknown",
)


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[$,]", "", text)
    text = re.sub(r"[^a-z0-9.\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(answer) for answer in answers)


def token_f1(prediction: str, answers: list[str]) -> float:
    return max((_token_f1_single(prediction, answer) for answer in answers), default=0.0)


def _token_f1_single(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(answer).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_numeric_values(text: str) -> list[float]:
    lowered = str(text).lower().replace(",", "").replace("$", "")
    pattern = r"([-+]?\d+(?:\.\d+)?)\s*(billion|million|thousand|bn|mm|m|k)?"
    values = []
    for number, unit in re.findall(pattern, lowered):
        value = float(number)
        multiplier = {
            "billion": 1_000_000_000,
            "bn": 1_000_000_000,
            "million": 1_000_000,
            "mm": 1_000_000,
            "m": 1_000_000,
            "thousand": 1_000,
            "k": 1_000,
            "": 1.0,
        }.get(unit, 1.0)
        values.append(value * multiplier)
    return values


def numeric_match(prediction: str, answers: list[str], rel_tol: float = 1e-3) -> bool:
    pred_values = extract_numeric_values(prediction)
    gold_values = [value for answer in answers for value in extract_numeric_values(answer)]
    if not pred_values or not gold_values:
        return False
    for pred in pred_values:
        for gold in gold_values:
            if math.isclose(pred, gold, rel_tol=rel_tol, abs_tol=max(1.0, abs(gold) * rel_tol)):
                return True
    return False


def numeric_answer_available(answers: list[str]) -> bool:
    return bool([value for answer in answers for value in extract_numeric_values(answer)])


def is_refusal(text: str) -> bool:
    normalized = normalize_answer(text)
    return any(phrase in normalized for phrase in REFUSAL_PHRASES)


def answer_coverage(retrieved_context: str, question: dict) -> bool:
    haystack = normalize_answer(retrieved_context)
    candidates = []
    if question.get("gold_evidence_text"):
        candidates.append(str(question["gold_evidence_text"]))
    if question.get("answer"):
        candidates.append(str(question["answer"]))
    aliases = question.get("answer_aliases") or []
    candidates.extend(str(alias) for alias in aliases)
    for candidate in candidates:
        normalized = normalize_answer(candidate)
        if normalized and normalized in haystack:
            return True
    return False


def answer_coverage_for_results(results: list[dict], question: dict, k: int = 3) -> bool:
    context = "\n".join(str(row.get("text", "")) for row in results[:k])
    return answer_coverage(context, question)


def source_accuracy(prediction_row: dict, question: dict) -> bool | None:
    metas = prediction_row.get("retrieved_metadata") or []
    expected_ticker = str(question.get("ticker") or "").upper()
    expected_year = str(question.get("year") or "")
    if not expected_ticker and not expected_year:
        return None
    if not metas:
        return False
    for meta in metas:
        ticker_ok = not expected_ticker or str(meta.get("ticker", "")).upper() == expected_ticker
        year_ok = not expected_year or str(meta.get("year", "")) == expected_year
        if ticker_ok and year_ok:
            return True
    return False


def source_accuracy_for_results(results: list[dict], question: dict, k: int = 3) -> bool | None:
    row = {"retrieved_metadata": [result.get("metadata", {}) for result in results[:k]]}
    return source_accuracy(row, question)


def table_row_recall_for_results(results: list[dict], question: dict, k: int = 3) -> bool | None:
    if question.get("question_type") != "numeric_fact":
        return None
    for result in results[:k]:
        metadata = result.get("metadata", {})
        if metadata.get("source_type") == "table_row":
            return True
    return False


def evaluate_predictions(predictions: list[dict], questions: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    question_by_id = {row.get("question_id"): row for row in questions}
    details = []
    grouped: dict[str, list[dict]] = defaultdict(list)

    for prediction in predictions:
        question = question_by_id.get(prediction.get("question_id"), {})
        answers = [str(question.get("answer", prediction.get("gold_answer", "")))]
        answers.extend(str(alias) for alias in question.get("answer_aliases", []) if alias)
        pred_text = str(prediction.get("prediction", ""))
        expected_refusal = bool(question.get("expected_refusal", False))
        numeric_available = numeric_answer_available(answers)
        extracted_numeric_answer = prediction.get("extracted_numeric_answer")
        numeric_generation_accuracy = numeric_match(pred_text, answers) if numeric_available else None
        numeric_extraction_accuracy = numeric_match(str(extracted_numeric_answer or ""), answers) if numeric_available and extracted_numeric_answer else None
        numeric_accuracy = None
        if numeric_available:
            numeric_accuracy = bool(numeric_generation_accuracy) or bool(numeric_extraction_accuracy)
        detail = {
            **prediction,
            "exact_match": exact_match(pred_text, answers),
            "token_f1": token_f1(pred_text, answers),
            "numeric_available": numeric_available,
            "numeric_generation_accuracy": numeric_generation_accuracy,
            "numeric_extraction_accuracy": numeric_extraction_accuracy,
            "numeric_accuracy": numeric_accuracy,
            "expected_refusal": expected_refusal,
            "refusal_correct": is_refusal(pred_text) == expected_refusal if expected_refusal else None,
            "hallucinated_on_negative": (not is_refusal(pred_text)) if expected_refusal else None,
            "answer_coverage_at_1": answer_coverage("\n".join(prediction.get("retrieved_texts", [])[:1]), question)
            if prediction.get("retrieved_texts")
            else None,
            "answer_coverage_at_3": answer_coverage(prediction.get("retrieved_context", ""), question)
            if prediction.get("retrieved_context")
            else None,
            "source_accuracy": source_accuracy(prediction, question),
            "source_accuracy_at_3": source_accuracy(prediction, question),
            "table_row_recall_at_3": _table_row_recall_from_prediction(prediction, question),
        }
        details.append(detail)
        grouped[str(prediction.get("system"))].append(detail)

    rows = []
    baseline_perplexity = None
    for system, system_rows in grouped.items():
        row = _summarize_system(system, system_rows)
        rows.append(row)
        if system == "baseline_gpt":
            baseline_perplexity = row["average_gold_answer_perplexity"]
    for row in rows:
        if baseline_perplexity is None or pd.isna(row["average_gold_answer_perplexity"]):
            row["perplexity_delta_vs_baseline"] = float("nan")
        else:
            row["perplexity_delta_vs_baseline"] = row["average_gold_answer_perplexity"] - baseline_perplexity
    return pd.DataFrame(rows).sort_values("system"), details


def _summarize_system(system: str, rows: list[dict]) -> dict[str, Any]:
    numeric_rows = [row for row in rows if row.get("numeric_available")]
    negative_rows = [row for row in rows if row.get("expected_refusal")]
    coverage_1_rows = [row for row in rows if row.get("answer_coverage_at_1") is not None]
    coverage_rows = [row for row in rows if row.get("answer_coverage_at_3") is not None]
    source_rows = [row for row in rows if row.get("source_accuracy") is not None]
    table_row_rows = [row for row in rows if row.get("table_row_recall_at_3") is not None]
    return {
        "system": system,
        "num_questions": len(rows),
        "exact_match": _mean(row["exact_match"] for row in rows),
        "token_f1": _mean(row["token_f1"] for row in rows),
        "numeric_accuracy": _mean(row["numeric_accuracy"] for row in numeric_rows) if numeric_rows else float("nan"),
        "numeric_generation_accuracy": _mean(row["numeric_generation_accuracy"] for row in numeric_rows) if numeric_rows else float("nan"),
        "numeric_extraction_accuracy": _mean(row["numeric_extraction_accuracy"] for row in numeric_rows) if numeric_rows else float("nan"),
        "refusal_accuracy": _mean(row["refusal_correct"] for row in negative_rows) if negative_rows else float("nan"),
        "hallucination_rate": _mean(row["hallucinated_on_negative"] for row in negative_rows) if negative_rows else float("nan"),
        "average_generation_latency": _mean(row.get("generation_latency_seconds") for row in rows),
        "average_retrieval_latency": _mean(row.get("retrieval_latency_seconds") for row in rows),
        "average_total_latency": _mean(row.get("total_latency_seconds") for row in rows),
        "average_prompt_tokens": _mean(row.get("prompt_tokens") for row in rows),
        "average_completion_tokens": _mean(row.get("completion_tokens") for row in rows),
        "average_gold_answer_loss": _mean(row.get("gold_answer_loss") for row in rows),
        "average_gold_answer_perplexity": _mean(row.get("gold_answer_perplexity") for row in rows),
        "answer_coverage_at_1": _mean(row.get("answer_coverage_at_1") for row in coverage_1_rows) if coverage_1_rows else float("nan"),
        "answer_coverage_at_3": _mean(row.get("answer_coverage_at_3") for row in coverage_rows) if coverage_rows else float("nan"),
        "source_accuracy": _mean(row.get("source_accuracy") for row in source_rows) if source_rows else float("nan"),
        "source_accuracy_at_3": _mean(row.get("source_accuracy_at_3") for row in source_rows) if source_rows else float("nan"),
        "table_row_recall_at_3": _mean(row.get("table_row_recall_at_3") for row in table_row_rows) if table_row_rows else float("nan"),
    }


def build_run_metadata(
    questions: list[dict],
    predictions: list[dict],
    summary: pd.DataFrame,
    question_path: str | Path,
    limit: int | None = None,
    min_research_questions: int = 50,
    expected_systems: list[str] | None = None,
    question_source: str | None = None,
) -> dict:
    num_questions = len(questions)
    invalid_reasons: list[str] = []
    path_name = Path(question_path).name
    question_source = question_source or _infer_question_source(question_path)
    is_sample_file = path_name == "sample_questions.jsonl"
    if num_questions < min_research_questions:
        invalid_reasons.append(f"num_questions<{min_research_questions}")
    if question_source not in {"verified", "verified_remapped"}:
        invalid_reasons.append("question_source_not_verified")
    if is_sample_file:
        invalid_reasons.append("question_file_is_sample_questions")

    type_counts = Counter(str(row.get("question_type", "")) for row in questions if row.get("question_type"))
    ticker_counts = Counter(str(row.get("ticker", "")) for row in questions if row.get("ticker"))
    company_counts = Counter(str(row.get("company", "")) for row in questions if row.get("company"))
    for qtype in ("numeric_fact", "text_fact", "risk_factor", "explanation", "negative"):
        if 0 < type_counts.get(qtype, 0) < 5:
            invalid_reasons.append(f"{qtype}<5")
    missing_evidence_rate = _missing_evidence_rate(questions)

    summary_by_system = {str(row["system"]): row for _, row in summary.iterrows()}
    oracle = summary_by_system.get("oracle_gpt")
    rag = summary_by_system.get("rag_gpt_tfidf_top3")
    oracle_cov = _safe_float(oracle.get("answer_coverage_at_3")) if oracle is not None else float("nan")
    rag_cov = _safe_float(rag.get("answer_coverage_at_3")) if rag is not None else float("nan")
    if oracle is None or math.isnan(oracle_cov) or oracle_cov < 0.95:
        invalid_reasons.append("oracle_answer_coverage<0.95")

    expected = expected_systems or sorted({str(row.get("system")) for row in predictions})
    counts = Counter(str(row.get("system")) for row in predictions)
    for system in expected:
        if counts.get(system, 0) != num_questions:
            invalid_reasons.append(f"missing_predictions:{system}")

    is_smoke_test = bool(limit is not None) or num_questions < min_research_questions
    return {
        "num_questions": num_questions,
        "is_smoke_test": is_smoke_test,
        "valid_for_research": len(invalid_reasons) == 0,
        "invalid_reasons": invalid_reasons,
        "min_research_questions": min_research_questions,
        "question_file": str(question_path),
        "question_source": question_source,
        "limit": limit,
        "question_type_counts": dict(type_counts),
        "ticker_counts": dict(ticker_counts),
        "company_counts": dict(company_counts),
        "missing_evidence_rate": missing_evidence_rate,
        "oracle_answer_coverage_at_3": oracle_cov if not math.isnan(oracle_cov) else None,
        "rag_answer_coverage_at_3": rag_cov if not math.isnan(rag_cov) else None,
        "retrieval_quality": "weak" if math.isnan(rag_cov) or rag_cov < 0.70 else "acceptable",
    }


def _infer_question_source(question_path: str | Path) -> str:
    name = Path(question_path).name
    if name == "questions_verified.remapped.jsonl":
        return "verified_remapped"
    if name == "questions_verified.jsonl":
        return "verified"
    if name == "questions.jsonl":
        return "default"
    if name == "sample_questions.jsonl":
        return "sample"
    return "custom"


def _missing_evidence_rate(questions: list[dict]) -> float:
    answerable = [row for row in questions if not row.get("expected_refusal", False)]
    if not answerable:
        return 0.0
    missing = 0
    for row in answerable:
        evidence_valid = row.get("evidence_valid")
        if evidence_valid is False:
            missing += 1
        elif not row.get("gold_evidence_text") or not row.get("relevant_chunk_ids"):
            missing += 1
    return missing / len(answerable)


def oracle_sanity_check(summary: pd.DataFrame, run_metadata: dict, latency_warning_seconds: float = 5.0) -> dict:
    baseline = _summary_row(summary, "baseline_gpt")
    oracle = _summary_row(summary, "oracle_gpt")
    reasons: list[str] = []
    if oracle is None:
        reasons.append("oracle_system_missing")
    if baseline is None:
        reasons.append("baseline_system_missing")
    oracle_coverage = _safe_float(oracle.get("answer_coverage_at_3")) if oracle is not None else float("nan")
    if math.isnan(oracle_coverage) or oracle_coverage < 0.95:
        reasons.append("oracle_answer_coverage<0.95")
    baseline_ppl = float("nan")
    oracle_ppl = float("nan")
    baseline_f1 = float("nan")
    oracle_f1 = float("nan")
    if baseline is not None and oracle is not None:
        oracle_ppl = _safe_float(oracle.get("average_gold_answer_perplexity"))
        baseline_ppl = _safe_float(baseline.get("average_gold_answer_perplexity"))
        if not math.isnan(oracle_ppl) and not math.isnan(baseline_ppl) and oracle_ppl > baseline_ppl:
            reasons.append("oracle_perplexity_not_better_than_baseline")
        oracle_f1 = _safe_float(oracle.get("token_f1"))
        baseline_f1 = _safe_float(baseline.get("token_f1"))
        if not math.isnan(oracle_f1) and not math.isnan(baseline_f1) and oracle_f1 < baseline_f1:
            reasons.append("oracle_token_f1_below_baseline")
        oracle_latency = _safe_float(oracle.get("average_total_latency"))
        if not math.isnan(oracle_latency) and oracle_latency > latency_warning_seconds:
            reasons.append("oracle_latency_warning")
    if run_metadata.get("num_questions", 0) < run_metadata.get("min_research_questions", 50):
        reasons.append("oracle_question_count_below_minimum")
    hard_reasons = [reason for reason in reasons if reason != "oracle_latency_warning"]
    return {
        "passed": len(hard_reasons) == 0,
        "warning_only": reasons == ["oracle_latency_warning"],
        "reasons": reasons,
        "oracle_answer_coverage_at_3": oracle_coverage,
        "oracle_answer_coverage": oracle_coverage,
        "baseline_avg_gold_answer_perplexity": baseline_ppl,
        "oracle_avg_gold_answer_perplexity": oracle_ppl,
        "perplexity_delta_vs_baseline": oracle_ppl - baseline_ppl if not math.isnan(oracle_ppl) and not math.isnan(baseline_ppl) else float("nan"),
        "token_f1_delta_vs_baseline": oracle_f1 - baseline_f1 if not math.isnan(oracle_f1) and not math.isnan(baseline_f1) else float("nan"),
    }


VAGUE_QUESTION_PHRASES = (
    "what is one important fact",
    "why does this section matter",
    "what does the company disclose",
    "summarize this section",
)


def validate_question_rows(questions: list[dict], chunks: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    chunks_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunks}
    seen: set[str] = set()
    valid: list[dict] = []
    rejected: list[dict] = []
    audit: list[dict] = []
    for row in questions:
        reasons = _question_rejection_reasons(row, chunks_by_id, seen)
        question_id = str(row.get("question_id") or row.get("question") or "")
        if question_id:
            seen.add(question_id)
        audit_row = {
            "question_id": row.get("question_id", ""),
            "question_type": row.get("question_type", ""),
            "accepted": not reasons,
            "reasons": "; ".join(reasons),
        }
        audit.append(audit_row)
        if reasons:
            rejected.append({**row, "rejection_reasons": reasons})
        else:
            valid.append(row)
    return valid, rejected, audit


def question_balance(valid_questions: list[dict]) -> dict[str, int]:
    counts = Counter(str(row.get("question_type", "")) for row in valid_questions)
    return dict(counts)


def _question_rejection_reasons(row: dict, chunks_by_id: dict[str, dict], seen: set[str]) -> list[str]:
    reasons: list[str] = []
    question_id = str(row.get("question_id") or row.get("question") or "")
    question = str(row.get("question") or "")
    answer = str(row.get("answer") or "")
    qtype = str(row.get("question_type") or "")
    normalized_question = normalize_answer(question)
    if question_id in seen:
        reasons.append("duplicate_question")
    if any(normalize_answer(phrase) in normalized_question for phrase in VAGUE_QUESTION_PHRASES):
        reasons.append("vague_question")
    if answer.lower() in {"", "todo", "tbd", "placeholder", "n/a"}:
        reasons.append("placeholder_or_empty_answer")

    expected_refusal = bool(row.get("expected_refusal", False))
    relevant_ids = [str(item) for item in row.get("relevant_chunk_ids", []) if item]
    if qtype == "negative":
        if not expected_refusal:
            reasons.append("negative_expected_refusal_required")
        if normalize_answer(answer) != "not enough information":
            reasons.append("negative_answer_must_be_not_enough_information")
        if _is_absurd_negative(question):
            reasons.append("negative_question_not_plausible")
        return reasons

    if not answer:
        reasons.append("answer_required")
    if not str(row.get("gold_evidence_text") or "").strip():
        reasons.append("gold_evidence_required")
    if not relevant_ids:
        reasons.append("relevant_chunk_ids_required")
    if relevant_ids and not _evidence_supported(row, relevant_ids, chunks_by_id):
        reasons.append("evidence_not_supported_by_relevant_chunks")
    if qtype == "numeric_fact":
        aliases = row.get("answer_aliases") or []
        if not extract_numeric_values(answer):
            reasons.append("numeric_answer_required")
        if not _has_unit_when_applicable(answer):
            reasons.append("numeric_units_required")
        if not aliases:
            reasons.append("answer_aliases_required")
        if row.get("source_type") not in {"table_row", "table_summary"}:
            reasons.append("numeric_source_type_should_be_table")
    return reasons


def _evidence_supported(row: dict, relevant_ids: list[str], chunks_by_id: dict[str, dict]) -> bool:
    evidence = str(row.get("gold_evidence_text") or "")
    answer = str(row.get("answer") or "")
    aliases = [str(alias) for alias in row.get("answer_aliases", []) if alias]
    for chunk_id in relevant_ids:
        chunk = chunks_by_id.get(chunk_id)
        if not chunk:
            continue
        text = str(chunk.get("text", ""))
        if answer_coverage(text, {"gold_evidence_text": evidence, "answer": answer, "answer_aliases": aliases}):
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


def _table_row_recall_from_prediction(prediction: dict, question: dict) -> bool | None:
    if question.get("question_type") != "numeric_fact":
        return None
    for meta in prediction.get("retrieved_metadata") or []:
        if meta.get("source_type") == "table_row":
            return True
    return False


def _summary_row(summary: pd.DataFrame, system: str) -> dict | None:
    match = summary[summary["system"] == system]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric


def _mean(values) -> float:
    clean = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        clean.append(numeric)
    return sum(clean) / len(clean) if clean else float("nan")


def write_evaluation_outputs(run_dir: str | Path, report_dir: str | Path, summary: pd.DataFrame, details: list[dict]) -> None:
    run_dir = Path(run_dir)
    report_dir = Path(report_dir)
    ensure_dir(run_dir)
    ensure_dir(report_dir)
    summary.to_csv(run_dir / "evaluation_summary.csv", index=False)
    write_jsonl(run_dir / "evaluation_details.jsonl", details)
    summary.to_csv(report_dir / "comparison_table.csv", index=False)
