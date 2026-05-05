from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .evaluation import is_refusal
from .utils import compact_text, ensure_dir, read_json, read_jsonl


TRAINING_NOT_AVAILABLE = (
    "The submitted experiment uses provided trained weights at model/model_weights.pt. "
    "Because training logs were not provided, training time and convergence are reported as not available. "
    "The RAG variation changes inference only, so training time is unchanged relative to the baseline GPT."
)

ACADEMIC_HONESTY = (
    "Some scaffolding, debugging prompts, and documentation drafts were generated with OpenAI Codex/ChatGPT. "
    "All code was reviewed, edited, and tested by the author. The submitted experimental design, implementation choices, "
    "and results are the author's responsibility."
)


def build_report(run_id: str, outputs_dir: str | Path = "outputs") -> Path:
    outputs_dir = Path(outputs_dir)
    run_dir = outputs_dir / "runs" / run_id
    report_dir = ensure_dir(outputs_dir / "reports" / run_id)

    summary_path = run_dir / "evaluation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    details = read_jsonl(run_dir / "evaluation_details.jsonl")
    diagnostics = read_jsonl(run_dir / "retrieval_diagnostics.jsonl")
    config = read_json(run_dir / "run_config.json", default={}) or {}
    tokenizer_metadata = read_json(run_dir / "tokenizer_metadata.json", default={}) or config.get("tokenizer_metadata", {})

    comparison = summary.copy()
    comparison.to_csv(report_dir / "comparison_table.csv", index=False)
    _write_subset(comparison, report_dir / "latency_table.csv", [
        "system",
        "average_generation_latency",
        "average_retrieval_latency",
        "average_total_latency",
        "average_prompt_tokens",
        "average_completion_tokens",
    ])
    _write_subset(comparison, report_dir / "retrieval_table.csv", [
        "system",
        "answer_coverage_at_1",
        "answer_coverage_at_3",
        "source_accuracy_at_3",
        "table_row_recall_at_3",
        "average_retrieval_latency",
    ])
    taxonomy = write_failure_analysis(report_dir, details or predictions, diagnostics)

    report_text = render_report(run_id, summary, predictions, diagnostics, config, tokenizer_metadata, taxonomy)
    report_path = report_dir / "final_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def build_oracle_sanity_report(run_id: str, oracle_sanity: dict, summary: pd.DataFrame, outputs_dir: str | Path = "outputs") -> Path:
    report_dir = ensure_dir(Path(outputs_dir) / "reports" / run_id)
    lines = [
        "# Oracle Sanity Report",
        "",
        f"Passed: {oracle_sanity.get('passed')}",
        f"Reasons: {', '.join(oracle_sanity.get('reasons') or ['none'])}",
        f"Oracle answer coverage@3: {_fmt(oracle_sanity.get('oracle_answer_coverage_at_3', oracle_sanity.get('oracle_answer_coverage')))}",
        f"Baseline gold-answer perplexity: {_fmt(oracle_sanity.get('baseline_avg_gold_answer_perplexity'))}",
        f"Oracle gold-answer perplexity: {_fmt(oracle_sanity.get('oracle_avg_gold_answer_perplexity'))}",
        f"Perplexity delta vs baseline: {_fmt(oracle_sanity.get('perplexity_delta_vs_baseline'))}",
        f"Token F1 delta vs baseline: {_fmt(oracle_sanity.get('token_f1_delta_vs_baseline'))}",
        "",
        "## Metrics",
        _markdown_table(summary),
        "",
    ]
    path = report_dir / "oracle_sanity_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def render_report(
    run_id: str,
    summary: pd.DataFrame,
    predictions: list[dict],
    diagnostics: list[dict],
    config: dict,
    tokenizer_metadata: dict | None = None,
    taxonomy: list[dict] | None = None,
) -> str:
    baseline = _row(summary, "baseline_gpt")
    rag = _row(summary, "rag_gpt_tfidf_top3")
    oracle = _row(summary, "oracle_gpt")
    random_context = _row(summary, "random_context_gpt")
    metadata = config.get("run_metadata") or _fallback_metadata(predictions)
    tokenizer_metadata = tokenizer_metadata or {}
    invalid = not bool(metadata.get("valid_for_research"))
    strict_halt = invalid and bool(metadata.get("strict_research_report"))

    lines = [
        "# Final Report: Retrieval-Augmented Generation with Financial Filings",
        "",
        "## Abstract",
        _abstract(metadata, rag, config),
        "",
        "## Hypothesis",
        "Adding retrieved 10-K context should improve factual grounding and reduce gold-answer perplexity, while increasing inference latency.",
        "",
        "## Experiment Setup",
        _experimental_design(config),
        "",
        "## Dataset Validity",
        _dataset_section(metadata),
        "",
        "## Tokenizer and Checkpoint",
        _tokenizer_checkpoint_section(config, tokenizer_metadata),
        "",
        "## Chunking Summary",
        _chunking_summary(config),
        "",
        "## Research Validity Check",
        _research_validity_check(metadata),
        "",
        "## Retrieval Quality",
        _retrieval_analysis(summary, diagnostics, metadata),
        "",
        "## Generation Quality",
        _markdown_table(summary),
        "",
        "## RAG vs Baseline Conclusion",
        _rag_vs_baseline(metadata, baseline, rag),
        "",
        "## Latency Tradeoff",
        _latency_summary(baseline, rag),
        "",
        "## Ablations",
        _ablation_summary(run_id),
        "",
        "## Discussion",
        _discussion(metadata, baseline, rag, oracle, random_context),
        "",
        "## Training and Convergence",
        _training_summary(config),
        "",
        "## Limitations",
        _limitations(metadata),
        "",
    ]
    if strict_halt:
        lines.extend(
            [
                "## Report Halted Before Conclusion",
                "The strict research-report gate was enabled. Because this run is not valid for research, the report stops before making a conclusion section.",
                "",
            ]
        )
    else:
        lines.extend(["## Conclusion", _conclusion(metadata, baseline, rag), ""])
    lines.extend(
        [
            "## Failure Analysis",
            "See `failure_taxonomy.csv` and `failure_examples.md` for qualitative examples and failure labels.",
            _failure_summary(taxonomy or []),
            "",
            "## Reproducibility Tables",
            "- `comparison_table.csv`",
            "- `latency_table.csv`",
            "- `retrieval_table.csv`",
            "- `failure_taxonomy.csv`",
            "",
            "## Academic Honesty",
            ACADEMIC_HONESTY,
            "",
        ]
    )
    return "\n".join(lines)


def write_failure_analysis(report_dir: Path, details: list[dict], diagnostics: list[dict]) -> list[dict]:
    diagnostics_by_question = {row.get("question_id"): row for row in diagnostics}
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in details:
        grouped[str(row.get("question_id"))][str(row.get("system"))] = row
    taxonomy: list[dict] = []
    examples: list[str] = ["# Failure Examples", ""]
    wanted_examples = {
        "rag_succeeds_baseline_fails": None,
        "baseline_succeeds_rag_fails": None,
        "retrieval_miss": None,
        "retrieval_finds_evidence_generator_fails": None,
        "random_context_fails": None,
        "oracle_succeeds": None,
        "oracle_fails": None,
    }
    for question_id, systems in grouped.items():
        baseline = systems.get("baseline_gpt", {})
        rag = systems.get("rag_gpt_tfidf_top3", {})
        oracle = systems.get("oracle_gpt", {})
        random_context = systems.get("random_context_gpt", {})
        for system_name, row in systems.items():
            failure_type = classify_failure(row, diagnostics_by_question.get(question_id, {}))
            taxonomy.append(
                {
                    "question_id": question_id,
                    "system": system_name,
                    "failure_type": failure_type,
                    "exact_match": row.get("exact_match"),
                    "token_f1": row.get("token_f1"),
                    "numeric_accuracy": row.get("numeric_accuracy"),
                    "answer_coverage_at_3": row.get("answer_coverage_at_3"),
                    "noise_reason": diagnostics_by_question.get(question_id, {}).get("noise_reason", ""),
                }
            )
        _maybe_store_examples(wanted_examples, question_id, baseline, rag, oracle, random_context, diagnostics_by_question.get(question_id, {}))
    with (report_dir / "failure_taxonomy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question_id", "system", "failure_type", "exact_match", "token_f1", "numeric_accuracy", "answer_coverage_at_3", "noise_reason"],
        )
        writer.writeheader()
        writer.writerows(taxonomy)
    for label, item in wanted_examples.items():
        examples.append(f"## {label}")
        examples.append("")
        examples.append(item or "No example available in this run.")
        examples.append("")
    (report_dir / "failure_examples.md").write_text("\n".join(examples), encoding="utf-8")
    return taxonomy


def classify_failure(row: dict, diagnostic: dict | None = None) -> str:
    if row.get("exact_match") or row.get("numeric_accuracy") or (row.get("token_f1") or 0) >= 0.8:
        return "correct"
    diagnostic = diagnostic or {}
    noise = diagnostic.get("noise_reason") or ""
    if noise in {"wrong_section", "wrong_company", "boilerplate", "answer_split_across_chunks"}:
        return noise
    if row.get("system") == "rag_gpt_tfidf_top3" and row.get("answer_coverage_at_3") is False:
        return "retrieval_miss"
    if row.get("system") == "rag_gpt_tfidf_top3" and row.get("answer_coverage_at_3") is True:
        return "generator_hallucination"
    if row.get("expected_refusal") and row.get("refusal_correct") is False:
        return "generator_refusal_error"
    if row.get("numeric_available") and row.get("numeric_accuracy") is False:
        return "numeric_unit_error"
    if str(row.get("prediction", "")).startswith("[generation error"):
        return "tokenizer_or_generation_issue"
    return "retrieval_noise"


def _maybe_store_examples(slots: dict, question_id: str, baseline: dict, rag: dict, oracle: dict, random_context: dict, diagnostic: dict) -> None:
    if not baseline and not rag:
        return
    if slots["rag_succeeds_baseline_fails"] is None and _correct(rag) and not _correct(baseline):
        slots["rag_succeeds_baseline_fails"] = _example(question_id, baseline, rag, diagnostic)
    if slots["baseline_succeeds_rag_fails"] is None and _correct(baseline) and not _correct(rag):
        slots["baseline_succeeds_rag_fails"] = _example(question_id, baseline, rag, diagnostic)
    if slots["retrieval_miss"] is None and rag and rag.get("answer_coverage_at_3") is False:
        slots["retrieval_miss"] = _example(question_id, baseline, rag, diagnostic)
    if slots["retrieval_finds_evidence_generator_fails"] is None and rag and rag.get("answer_coverage_at_3") is True and not _correct(rag):
        slots["retrieval_finds_evidence_generator_fails"] = _example(question_id, baseline, rag, diagnostic)
    if slots["random_context_fails"] is None and random_context and not _correct(random_context):
        slots["random_context_fails"] = _example(question_id, baseline, random_context, diagnostic)
    if slots["oracle_succeeds"] is None and _correct(oracle):
        slots["oracle_succeeds"] = _example(question_id, baseline, oracle, diagnostic)
    if slots["oracle_fails"] is None and oracle and not _correct(oracle):
        slots["oracle_fails"] = _example(question_id, baseline, oracle, diagnostic)


def _example(question_id: str, baseline: dict, candidate: dict, diagnostic: dict) -> str:
    preview = compact_text(candidate.get("retrieved_context", ""), 500)
    return "\n".join(
        [
            f"- question_id: `{question_id}`",
            f"- question: {candidate.get('question', baseline.get('question', ''))}",
            f"- gold answer: {candidate.get('gold_answer', baseline.get('gold_answer', ''))}",
            f"- baseline prediction: {compact_text(baseline.get('prediction', ''), 220)}",
            f"- candidate system: {candidate.get('system')}",
            f"- candidate prediction: {compact_text(candidate.get('prediction', ''), 220)}",
            f"- retrieved preview: {preview}",
            f"- failure type: {classify_failure(candidate, diagnostic)}",
        ]
    )


def _correct(row: dict) -> bool:
    return bool(row) and (bool(row.get("exact_match")) or bool(row.get("numeric_accuracy")) or float(row.get("token_f1") or 0.0) >= 0.8)


def _abstract(metadata: dict, rag: dict, config: dict) -> str:
    if not metadata.get("valid_for_research"):
        return (
            "This run evaluates the local GPT/RAG pipeline, but the validity gates mark it as a smoke or diagnostic run. "
            "This run verifies pipeline execution. "
            "The pipeline executed successfully, but the benchmark is too small or otherwise invalid for research conclusions."
        )
    return (
        "This report compares Baseline GPT against RAG-GPT TF-IDF top-3 using the local assignment GPT checkpoint "
        f"`{config.get('checkpoint_path', 'model/model_weights.pt')}` on {metadata.get('num_questions')} verified questions. "
        f"RAG answer coverage@3 is {_fmt(rag.get('answer_coverage_at_3') if rag else None)}."
    )


def _experimental_design(config: dict) -> str:
    return "\n".join(
        [
            '- Baseline GPT (`baseline_gpt`): The assignment GPT receives only the question.',
            '- RAG-GPT TF-IDF top-3 (`rag_gpt_tfidf_top3`): The same assignment GPT receives retrieved Form 10-K chunks plus the question.',
            '- Oracle Evidence GPT (`oracle_gpt`): The same assignment GPT receives gold evidence, used as an upper bound.',
            '- Random Context GPT (`random_context_gpt`): Noise control using random chunks.',
            f"- All systems use the same GPT architecture and the same checkpoint at {config.get('checkpoint_path', 'model/model_weights.pt')}. The only difference between baseline and RAG is the inference-time retrieval context.",
            f"- retrieval method: {config.get('retrieval_method', 'tfidf')}",
            "- chunking method: section-aware narrative chunks, table rows, table summaries, and local context chunks",
        ]
    )


def _data_report(predictions: list[dict], config: dict, metadata: dict) -> str:
    questions = {row.get("question_id") for row in predictions}
    companies = sorted({str(meta.get("company") or "") for row in predictions for meta in row.get("retrieved_metadata", []) if meta.get("company")})
    chunking = config.get("chunking_summary", {})
    return "\n".join(
        [
            f"- run_id: {config.get('run_id')}",
            f"- number of questions: {len(questions) or metadata.get('num_questions')}",
            f"- number of chunks: {config.get('num_chunks')}",
            f"- chunk types: {chunking.get('by_source_type', {})}",
            f"- companies represented in retrieved chunks: {', '.join(companies[:12]) if companies else 'not available'}",
            f"- question source: {metadata.get('question_source')}",
            f"- research validity status: {metadata.get('valid_for_research')}",
        ]
    )


def _dataset_section(metadata: dict) -> str:
    missing_rate = metadata.get("missing_evidence_rate")
    success_rate = None
    try:
        success_rate = 1.0 - float(missing_rate)
    except (TypeError, ValueError):
        success_rate = None
    return "\n".join(
        [
            f"- question file: {metadata.get('question_file')}",
            f"- question source: {metadata.get('question_source')}",
            f"- number of questions: {metadata.get('num_questions')}",
            f"- question type counts: {metadata.get('question_type_counts')}",
            f"- ticker counts: {metadata.get('ticker_counts')}",
            f"- company counts: {metadata.get('company_counts')}",
            f"- evidence remapping success rate: {_fmt(success_rate)}",
            f"- missing evidence rate: {_fmt(missing_rate)}",
            f"- evidence quality: {'weak' if success_rate is not None and success_rate < 0.95 else 'acceptable'}",
            f"- smoke_test: {metadata.get('is_smoke_test')}",
            f"- valid_for_research: {metadata.get('valid_for_research')}",
            f"- invalid reasons: {metadata.get('invalid_reasons')}",
        ]
    )


def _tokenizer_checkpoint_section(config: dict, tokenizer_metadata: dict) -> str:
    model_config = config.get("model_config", {}) or {}
    tokenizer_vocab = tokenizer_metadata.get("vocab_size")
    checkpoint_vocab = model_config.get("vocab_size") or config.get("checkpoint_vocab_size")
    compatible = tokenizer_vocab is not None and checkpoint_vocab is not None and int(tokenizer_vocab) == int(checkpoint_vocab)
    return "\n".join(
        [
            f"- tokenizer dir: {config.get('tokenizer_dir', tokenizer_metadata.get('tokenizer_dir', 'model/hftokenizer'))}",
            f"- tokenizer class: {tokenizer_metadata.get('tokenizer_class', 'not available')}",
            f"- tokenizer vocab size: {tokenizer_vocab}",
            f"- checkpoint: {config.get('checkpoint_path', 'model/model_weights.pt')}",
            f"- checkpoint vocab size: {checkpoint_vocab}",
            f"- compatibility: {'PASS' if compatible else 'not available' if tokenizer_vocab is None or checkpoint_vocab is None else 'FAIL'}",
        ]
    )


def _chunking_summary(config: dict) -> str:
    stats = config.get("chunking_summary", {}) or {}
    by_type = stats.get("by_source_type", {}) or {}
    return "\n".join(
        [
            f"- total chunks: {stats.get('num_chunks', config.get('num_chunks'))}",
            f"- narrative chunks: {by_type.get('narrative', 0)}",
            f"- table_row chunks: {by_type.get('table_row', stats.get('table_row_count', 0))}",
            f"- table_summary chunks: {by_type.get('table_summary', stats.get('table_summary_count', 0))}",
            f"- local_context chunks: {by_type.get('local_context', 0)}",
            f"- retrievable chunks: {stats.get('retrievable_count')}",
            f"- non-retrievable chunks: {stats.get('non_retrievable_count')}",
        ]
    )


def _research_validity_check(metadata: dict) -> str:
    warning = ""
    if not metadata.get("valid_for_research"):
        warning = (
            "\n\nThis run is a smoke test or diagnostic run. It verifies that the pipeline executes, "
            "but it should not be used for final research conclusions."
        )
    fields = [
        "num_questions",
        "min_research_questions",
        "is_smoke_test",
        "valid_for_research",
        "invalid_reasons",
        "limit",
        "question_source",
        "question_type_counts",
        "ticker_counts",
        "company_counts",
        "missing_evidence_rate",
        "oracle_answer_coverage_at_3",
        "rag_answer_coverage_at_3",
        "retrieval_quality",
        "question_file",
        "chunks_file",
        "checkpoint_path",
        "tokenizer_dir",
    ]
    return "\n".join(f"- {field}: {metadata.get(field)}" for field in fields) + warning


def _rag_vs_baseline(metadata: dict, baseline: dict, rag: dict) -> str:
    if not metadata.get("valid_for_research"):
        return "This is a smoke or diagnostic run, so it should not be used for final RAG-vs-baseline conclusions."
    f1_delta = _num(rag.get("token_f1")) - _num(baseline.get("token_f1"))
    ppl_delta = _num(rag.get("average_gold_answer_perplexity")) - _num(baseline.get("average_gold_answer_perplexity"))
    numeric_delta = _num(rag.get("numeric_accuracy")) - _num(baseline.get("numeric_accuracy"))
    latency_delta = _num(rag.get("average_total_latency")) - _num(baseline.get("average_total_latency"))
    return "\n".join(
        [
            f"- token F1 delta: {f1_delta:.4f}",
            f"- gold-answer perplexity delta: {ppl_delta:.4f}",
            f"- numeric accuracy delta: {numeric_delta:.4f}",
            f"- latency delta: {latency_delta:.4f}s",
            f"- RAG answer coverage@3: {_fmt(rag.get('answer_coverage_at_3') if rag else None)}",
        ]
    )


def _training_summary(config: dict) -> str:
    return (
        "Training logs were not provided. The submitted experiment compares inference-time changes. "
        "RAG does not change training time; it adds retrieval overhead at inference."
    )


def _retrieval_analysis(summary: pd.DataFrame, diagnostics: list[dict], metadata: dict) -> str:
    rag = _row(summary, "rag_gpt_tfidf_top3")
    noise = Counter(str(row.get("noise_reason", "none")) for row in diagnostics)
    lines = [
        f"- answer_coverage@1: {_fmt(rag.get('answer_coverage_at_1') if rag else None)}",
        f"- answer_coverage@3: {_fmt(rag.get('answer_coverage_at_3') if rag else None)}",
        f"- source_accuracy@3: {_fmt(rag.get('source_accuracy_at_3') if rag else None)}",
        f"- table_row_recall@3: {_fmt(rag.get('table_row_recall_at_3') if rag else None)}",
        f"- noise reason counts: {dict(noise)}",
    ]
    if rag and _num(rag.get("answer_coverage_at_3")) < 0.70:
        lines.append("- Retrieval is the main bottleneck when answer coverage is below the research threshold. Improve chunking, metadata filtering, and evidence labels.")
    if not metadata.get("valid_for_research") and metadata.get("num_questions", 0) < metadata.get("min_research_questions", 50):
        lines.append("- The benchmark is too small to determine whether RAG improves the baseline.")
    return "\n".join(lines)


def _discussion(metadata: dict, baseline: dict, rag: dict, oracle: dict, random_context: dict) -> str:
    if not metadata.get("valid_for_research"):
        return "The benchmark is too small to determine whether RAG improves the baseline."
    oracle_ok = oracle and _num(oracle.get("answer_coverage_at_3")) >= 0.95 and _num(oracle.get("average_gold_answer_perplexity")) <= _num(baseline.get("average_gold_answer_perplexity"))
    rag_ok = rag and _num(rag.get("answer_coverage_at_3")) >= 0.70
    if oracle_ok and not rag_ok:
        return "Oracle context appears useful, but retrieved context is weaker; retrieval is the bottleneck."
    if not oracle_ok:
        return "Oracle context did not clearly improve generation; generation, prompt format, tokenizer, or model capacity may be the bottleneck."
    return "The run passes validity gates; interpret RAG quality using the retrieval, generation, and latency tables."


def _limitations(metadata: dict) -> str:
    limits = [
        "The local GPT checkpoint may have limited capacity for exact financial copying.",
        "Table extraction from flattened HTML can still miss complex row/column structure.",
        TRAINING_NOT_AVAILABLE,
    ]
    if not metadata.get("valid_for_research"):
        limits.insert(0, "The benchmark is not research-valid because the validity gates failed.")
    return "\n".join(f"- {item}" for item in limits)


def _conclusion(metadata: dict, baseline: dict, rag: dict) -> str:
    if not metadata.get("valid_for_research"):
        return (
            "This run is a smoke test or diagnostic run. It verifies that the pipeline executes, but it should not be used for final research conclusions.\n\n"
            "The pipeline executed successfully, but the benchmark is too small for conclusions.\n\n"
            "The benchmark is too small to determine whether RAG improves the baseline."
        )
    f1_delta = _num(rag.get("token_f1")) - _num(baseline.get("token_f1"))
    ppl_delta = _num(rag.get("average_gold_answer_perplexity")) - _num(baseline.get("average_gold_answer_perplexity"))
    if f1_delta > 0 or ppl_delta < 0:
        verdict = "RAG improved at least one primary generation metric in this research-valid run."
    else:
        verdict = "RAG did not improve the primary generation metrics in this research-valid run."
    return f"{verdict} Token F1 delta is {f1_delta:.4f} and gold-answer perplexity delta is {ppl_delta:.4f}."


def _latency_summary(baseline: dict, rag: dict) -> str:
    if not baseline or not rag:
        return "Latency comparison is unavailable because one of the required systems is missing."
    delta = _num(rag.get("average_total_latency")) - _num(baseline.get("average_total_latency"))
    return (
        f"- baseline total latency: {_num(baseline.get('average_total_latency')):.4f}s\n"
        f"- RAG total latency: {_num(rag.get('average_total_latency')):.4f}s\n"
        f"- latency delta vs baseline: {delta:.4f}s"
    )


def _ablation_summary(run_id: str, outputs_dir: str | Path = "outputs") -> str:
    report_dir = Path(outputs_dir) / "reports" / run_id
    paths = [
        report_dir / "ablation_retrieval_topk.csv",
        report_dir / "ablation_chunk_type.csv",
        report_dir / "ablation_retrieval_scope.csv",
        report_dir / "ablation_numeric_extraction.csv",
    ]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return "No ablation tables were generated for this run."
    return "\n".join(f"- `{path.name}`" for path in existing)


def _failure_summary(taxonomy: list[dict]) -> str:
    if not taxonomy:
        return "No failure taxonomy rows were generated."
    counts = Counter(row.get("failure_type", "unknown") for row in taxonomy)
    return f"Failure counts: {dict(counts)}"


def _markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "system",
        "exact_match",
        "token_f1",
        "numeric_accuracy",
        "refusal_accuracy",
        "answer_coverage_at_3",
        "average_gold_answer_perplexity",
        "perplexity_delta_vs_baseline",
        "average_total_latency",
    ]
    present = [col for col in columns if col in df.columns]
    small = df[present].copy()
    for col in present:
        if col != "system":
            small[col] = small[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    headers = small.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in small.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def _row(summary: pd.DataFrame, system: str) -> dict:
    match = summary[summary["system"] == system]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def _fallback_metadata(predictions: list[dict]) -> dict:
    num_questions = len({row.get("question_id") for row in predictions})
    return {
        "num_questions": num_questions,
        "min_research_questions": 50,
        "is_smoke_test": num_questions < 50,
        "valid_for_research": False,
        "invalid_reasons": ["run_metadata_missing"],
    }


def _write_subset(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    present = [col for col in columns if col in df.columns]
    df[present].to_csv(path, index=False)


def _fmt(value) -> str:
    try:
        if pd.isna(value):
            return "not available"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "not available"


def _num(value) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
