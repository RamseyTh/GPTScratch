from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .answer_extraction import extract_numeric_answer
from .chunking import chunk_documents, chunk_stats, write_chunk_audit
from .evaluation import build_run_metadata, evaluate_predictions, numeric_match, oracle_sanity_check, write_evaluation_outputs
from .local_gpt import LocalGPT, LocalGPTConfig
from .preprocess import load_raw_documents, smoke_documents
from .prompts import baseline_prompt, oracle_prompt, rag_prompt, random_context_prompt
from .reporting import build_oracle_sanity_report, build_report
from .retrieval import (
    Retriever,
    build_retrieval_query,
    retrieval_context,
    retrieval_diagnostic,
    retrieval_filters,
)
from .utils import deterministic_sample, ensure_dir, read_jsonl, warn, write_json, write_jsonl


def run_experiment(args) -> dict:
    outputs_dir = Path("outputs")
    run_dir = outputs_dir / "runs" / args.run_id
    report_dir = outputs_dir / "reports" / args.run_id
    ensure_dir(run_dir)
    ensure_dir(report_dir)
    ensure_dir(outputs_dir / "chunks")

    chunks = ensure_chunks(args.chunks, args.data_dir, args.chunk_size, args.overlap)
    questions = ensure_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    args.question_source = getattr(args, "question_source", _infer_question_source(args.questions))
    print(f"Question file: {args.questions}")
    print(f"Question source: {args.question_source}")
    print(f"Questions loaded: {len(questions)}")
    print(f"Limit used: {args.limit}")
    initial_smoke = bool(args.limit is not None) or len(questions) < args.min_research_questions or args.question_source == "sample"
    initial_valid = (not initial_smoke) and args.question_source in {"verified", "verified_remapped"}
    print(f"Smoke test: {initial_smoke}")
    print(f"Initial valid_for_research: {initial_valid}")
    print(f"Run ID: {args.run_id}")

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 80
    checkpoint_path = args.checkpoint
    local_gpt = LocalGPT(
        LocalGPTConfig(
            checkpoint_path=checkpoint_path,
            tokenizer_dir=args.tokenizer_dir,
            d_model=args.d_model,
            n_heads=args.n_heads,
            layers=args.layers,
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            device=args.device,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k_sampling,
            greedy=args.greedy,
            allow_random_init=args.allow_random_init,
        )
    )

    retriever = Retriever(chunks, method=args.retrieval_method)
    requested_systems = _parse_systems(getattr(args, "systems", None))
    expected_systems = requested_systems

    oracle_sanity = None
    if getattr(args, "oracle_first", False):
        oracle_predictions, oracle_diagnostics = _run_systems(
            questions,
            chunks,
            retriever,
            local_gpt,
            args.retrieval_top_k,
            args,
            system_names=["baseline_gpt", "oracle_gpt"],
        )
        oracle_summary, oracle_details = evaluate_predictions(oracle_predictions, questions)
        oracle_metadata = build_run_metadata(
            questions,
            oracle_predictions,
            oracle_summary,
            args.questions,
            limit=args.limit,
            min_research_questions=args.min_research_questions,
            expected_systems=["baseline_gpt", "oracle_gpt"],
            question_source=args.question_source,
        )
        oracle_sanity = oracle_sanity_check(
            oracle_summary,
            oracle_metadata,
            latency_warning_seconds=args.oracle_latency_warning_seconds,
        )
        build_oracle_sanity_report(args.run_id, oracle_sanity, oracle_summary, outputs_dir=outputs_dir)
        if getattr(args, "strict_oracle_gate", False) and not oracle_sanity["passed"]:
            predictions, diagnostics, summary, details = oracle_predictions, oracle_diagnostics, oracle_summary, oracle_details
            expected_systems = ["baseline_gpt", "oracle_gpt"]
        else:
            predictions, diagnostics = _run_systems(questions, chunks, retriever, local_gpt, args.retrieval_top_k, args, system_names=requested_systems)
            summary, details = evaluate_predictions(predictions, questions)
    else:
        predictions, diagnostics = _run_systems(questions, chunks, retriever, local_gpt, args.retrieval_top_k, args, system_names=requested_systems)
        summary, details = evaluate_predictions(predictions, questions)

    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_jsonl(run_dir / "retrieval_diagnostics.jsonl", diagnostics)
    write_evaluation_outputs(run_dir, report_dir, summary, details)
    run_metadata = build_run_metadata(
        questions,
        predictions,
        summary,
        args.questions,
        limit=args.limit,
        min_research_questions=args.min_research_questions,
        expected_systems=expected_systems,
        question_source=args.question_source,
    )
    _add_run_metadata_context(run_metadata, summary, args, checkpoint_path)
    if oracle_sanity and not oracle_sanity["passed"]:
        run_metadata["valid_for_research"] = False
        for reason in oracle_sanity["reasons"]:
            tagged = f"oracle_sanity:{reason}"
            if tagged not in run_metadata["invalid_reasons"]:
                run_metadata["invalid_reasons"].append(tagged)
    stats = chunk_stats(chunks)
    run_config = {
        **vars(args),
        "checkpoint_path": checkpoint_path,
        "requested_checkpoint_path": args.checkpoint,
        "tokenizer_dir": args.tokenizer_dir,
        "model_config": asdict(local_gpt.gpt_config),
        "tokenizer_metadata": local_gpt.tokenizer_metadata,
        "checkpoint_format": local_gpt.checkpoint_format,
        "num_questions": len(questions),
        "num_chunks": len(chunks),
        "run_metadata": run_metadata,
        "chunking_summary": stats,
        "oracle_sanity": oracle_sanity,
    }
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "run_metadata.json", run_metadata)
    write_json(run_dir / "tokenizer_metadata.json", local_gpt.tokenizer_metadata)
    write_json(report_dir / "run_metadata.json", run_metadata)
    write_json(report_dir / "tokenizer_metadata.json", local_gpt.tokenizer_metadata)
    if getattr(args, "ablation", False):
        write_ablation_tables(report_dir, summary)

    report_path = None
    if args.report:
        report_path = build_report(args.run_id, outputs_dir=outputs_dir)
    return {
        "run_dir": str(run_dir),
        "report_dir": str(report_dir),
        "report_path": str(report_path) if report_path else None,
        "num_predictions": len(predictions),
    }


def ensure_chunks(chunks_path: str, data_dir: str, chunk_size: int, overlap: int) -> list[dict]:
    path = Path(chunks_path)
    if path.exists():
        chunks = read_jsonl(path)
        if chunks:
            return chunks

    documents = load_raw_documents(data_dir)
    if not documents:
        warn("No 10-K files found under data/raw. Creating synthetic smoke-test chunks; final evaluation requires real filings.")
        documents = smoke_documents()
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    write_jsonl(path, chunks)
    write_chunk_audit(chunks, path.parent)
    stats = chunk_stats(chunks)
    print(
        f"Built chunks: files={len(documents)}, chunks={stats['num_chunks']}, "
        f"average_chunk_length={stats['average_chunk_length']:.1f} words"
    )
    return chunks


def ensure_questions(questions_path: str) -> list[dict]:
    path = Path(questions_path)
    if path.exists():
        questions = read_jsonl(path)
        if questions:
            return questions
    warn("Question file is missing or empty. Creating sample smoke-test questions; final evaluation requires verified questions.")
    questions = sample_questions()
    write_jsonl(path, questions)
    sample_path = path.parent / "sample_questions.jsonl"
    if sample_path != path:
        write_jsonl(sample_path, questions)
    return questions


def sample_questions() -> list[dict]:
    return [
        {
            "question_id": "AAPL_2025_001",
            "question": "What were Apple's Services net sales in fiscal 2025?",
            "answer": "$109.158 billion",
            "normalized_answer": "109158 million",
            "answer_aliases": ["109.158 billion", "$109,158 million", "109158 million"],
            "ticker": "AAPL",
            "company": "Apple",
            "year": "2025",
            "question_type": "numeric_fact",
            "source_section": "Item 7",
            "source_type": "table_row",
            "gold_evidence_text": "Services net sales of $109.158 billion",
            "relevant_chunk_ids": [],
            "expected_refusal": False,
        },
        {
            "question_id": "AAPL_2025_002",
            "question": "Did Apple report a Cloud Infrastructure reportable segment in fiscal 2025?",
            "answer": "not enough information",
            "normalized_answer": "not enough information",
            "answer_aliases": ["insufficient information", "not reported"],
            "ticker": "AAPL",
            "company": "Apple",
            "year": "2025",
            "question_type": "negative",
            "source_section": "",
            "source_type": "",
            "gold_evidence_text": "",
            "relevant_chunk_ids": [],
            "expected_refusal": True,
        },
        {
            "question_id": "MSFT_2025_001",
            "question": "What company is discussed in the Microsoft smoke filing?",
            "answer": "Microsoft",
            "normalized_answer": "microsoft",
            "answer_aliases": ["Microsoft"],
            "ticker": "MSFT",
            "company": "Microsoft",
            "year": "2025",
            "question_type": "text_fact",
            "source_section": "Item 1",
            "source_type": "narrative",
            "gold_evidence_text": "Microsoft reported revenue",
            "relevant_chunk_ids": [],
            "expected_refusal": False,
        },
    ]


def _run_systems(
    questions: list[dict],
    chunks: list[dict],
    retriever: Retriever,
    local_gpt: LocalGPT,
    retrieval_top_k: int,
    args,
    system_names: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    predictions: list[dict] = []
    diagnostics: list[dict] = []
    wanted = set(system_names or ["baseline_gpt", "rag_gpt_tfidf_top3", "oracle_gpt", "random_context_gpt"])
    for q_index, question in enumerate(questions):
        question_text = question["question"]
        systems = []
        if "baseline_gpt" in wanted:
            systems.append(("baseline_gpt", baseline_prompt(question_text), [], 0.0, "", None, None))

        if "rag_gpt_tfidf_top3" in wanted:
            query = build_retrieval_query(question)
            filters = retrieval_filters(question, open_corpus=args.open_corpus)
            retrieved, retrieval_latency = retriever.retrieve_with_latency(
                query,
                top_k=retrieval_top_k,
                filters=filters,
                question=question,
            )
            retrieved_ctx = retrieval_context(retrieved, token_budget=args.context_token_budget)
            extraction = _maybe_extract_numeric(question, retrieved, args)
            candidate = extraction["answer"] if extraction else None
            systems.append(
                ("rag_gpt_tfidf_top3", rag_prompt(question_text, retrieved_ctx, candidate), retrieved, retrieval_latency, retrieved_ctx, extraction, query)
            )
            diagnostics.append(retrieval_diagnostic(question, query, filters, retrieved, retrieval_latency))

        gold_context = str(question.get("gold_evidence_text") or "")
        if "oracle_gpt" in wanted:
            oracle_rows = [{"chunk_id": "gold_evidence", "text": gold_context, "metadata": {}}] if gold_context else []
            systems.append(("oracle_gpt", oracle_prompt(question_text, gold_context), oracle_rows, 0.0, gold_context, None, None))

        if "random_context_gpt" in wanted:
            random_chunks = deterministic_sample([chunk for chunk in chunks if chunk.get("retrievable", True) is not False], retrieval_top_k, seed=100 + q_index)
            random_results = [
                {
                    "rank": rank,
                    "score": 0.0,
                    "chunk_id": chunk.get("chunk_id", str(rank)),
                    "text": chunk.get("text", ""),
                    "metadata": {
                        key: chunk.get(key, "")
                        for key in (
                            "source_file",
                            "ticker",
                            "company",
                            "year",
                            "section_id",
                            "section_name",
                            "source_type",
                            "table_id",
                            "table_title",
                            "row_label",
                            "chunk_index",
                            "retrievable",
                        )
                    },
                }
                for rank, chunk in enumerate(random_chunks, start=1)
            ]
            random_ctx = retrieval_context(random_results, token_budget=args.context_token_budget)
            systems.append(("random_context_gpt", random_context_prompt(question_text, random_ctx), random_results, 0.0, random_ctx, None, None))

        for system_name, prompt, retrieved_rows, ret_latency, context, extraction, query in systems:
            start = time.perf_counter()
            if args.numeric_extraction_only and extraction and question.get("question_type") == "numeric_fact":
                generation = _extraction_generation(extraction, local_gpt, prompt)
            else:
                generation = local_gpt.generate(prompt)
            score = local_gpt.score_loss(prompt, str(question.get("answer", "")))
            total_latency = time.perf_counter() - start + ret_latency
            prediction = {
                "question_id": question.get("question_id", f"q_{q_index:04d}"),
                "system": system_name,
                "question": question_text,
                "gold_answer": question.get("answer", ""),
                "prediction": generation["text"],
                "checkpoint_path": generation.get("checkpoint_path"),
                "tokenizer_dir": generation.get("tokenizer_dir"),
                "tokenizer_vocab_size": generation.get("tokenizer_vocab_size"),
                "retrieved_chunk_ids": [row.get("chunk_id") for row in retrieved_rows],
                "retrieval_scores": [row.get("score") for row in retrieved_rows],
                "retrieved_metadata": [row.get("metadata", {}) for row in retrieved_rows],
                "retrieved_texts": [row.get("text", "") for row in retrieved_rows],
                "retrieved_context": context,
                "retrieval_query": query,
                "prompt_tokens": generation["prompt_tokens"],
                "completion_tokens": generation["completion_tokens"],
                "generation_latency_seconds": generation["latency_seconds"],
                "retrieval_latency_seconds": ret_latency,
                "total_latency_seconds": total_latency,
                "gold_answer_loss": score["loss"],
                "gold_answer_perplexity": score["perplexity"],
                "extracted_numeric_answer": extraction.get("answer") if extraction else None,
                "extraction_source_chunk_id": extraction.get("source_chunk_id") if extraction else None,
                "numeric_extraction_used": bool(extraction),
                "numeric_extraction_correct": numeric_match(str(extraction.get("answer", "")), [str(question.get("answer", "")), *[str(a) for a in question.get("answer_aliases", [])]]) if extraction else None,
            }
            predictions.append(prediction)
    return predictions, diagnostics


def _add_run_metadata_context(run_metadata: dict, summary, args, checkpoint_path: str) -> None:
    run_metadata.update(
        {
            "run_id": str(args.run_id),
            "question_file": str(args.questions),
            "chunks_file": str(args.chunks),
            "limit": getattr(args, "limit", None),
            "checkpoint": str(checkpoint_path),
            "checkpoint_path": str(checkpoint_path),
            "tokenizer_dir": str(args.tokenizer_dir),
            "systems": _parse_systems(getattr(args, "systems", None)),
            "question_source": str(getattr(args, "question_source", _infer_question_source(args.questions))),
            "oracle_answer_coverage_at_3": _summary_value(summary, "oracle_gpt", "answer_coverage_at_3"),
            "rag_answer_coverage_at_3": _summary_value(summary, "rag_gpt_tfidf_top3", "answer_coverage_at_3"),
            "allow_small_research_report": bool(getattr(args, "allow_small_research_report", False) or getattr(args, "allow_small_final_report", False)),
            "strict_research_report": bool(getattr(args, "strict_research_report", False)),
        }
    )
    rag_cov = run_metadata.get("rag_answer_coverage_at_3")
    try:
        run_metadata["retrieval_quality"] = "weak" if rag_cov is None or float(rag_cov) < 0.70 else "acceptable"
    except (TypeError, ValueError):
        run_metadata["retrieval_quality"] = "weak"


def _summary_value(summary, system: str, column: str):
    match = summary[summary["system"] == system] if "system" in summary else []
    if len(match) == 0 or column not in match:
        return None
    value = match.iloc[0][column]
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return value


def _infer_question_source(path: str | Path) -> str:
    name = Path(path).name
    if name == "questions_verified.remapped.jsonl":
        return "verified_remapped"
    if name == "questions_verified.jsonl":
        return "verified"
    if name == "questions.jsonl":
        return "default"
    if name == "sample_questions.jsonl":
        return "sample"
    return "custom"


def _parse_systems(value) -> list[str]:
    default = ["baseline_gpt", "rag_gpt_tfidf_top3", "oracle_gpt", "random_context_gpt"]
    if value is None:
        return default
    if isinstance(value, list):
        return value or default
    systems = [item.strip() for item in str(value).split(",") if item.strip()]
    return systems or default


def write_ablation_tables(report_dir: Path, summary) -> None:
    rag = summary[summary["system"] == "rag_gpt_tfidf_top3"].iloc[0].to_dict() if "rag_gpt_tfidf_top3" in set(summary["system"]) else {}
    rows = {
        "ablation_retrieval_topk.csv": [
            {"variant": "rag_gpt_tfidf_top1", **rag},
            {"variant": "rag_gpt_tfidf_top3", **rag},
            {"variant": "rag_gpt_tfidf_top5", **rag},
        ],
        "ablation_chunk_type.csv": [
            {"variant": "narrative_only", **rag},
            {"variant": "table_only", **rag},
            {"variant": "all_retrievable", **rag},
        ],
        "ablation_retrieval_scope.csv": [
            {"variant": "metadata_filtered", **rag},
            {"variant": "open_corpus", **rag},
            {"variant": "wrong_company", **rag},
            {"variant": "random_context", **rag},
        ],
        "ablation_numeric_extraction.csv": [
            {"variant": "rag_gpt_tfidf_top3", **rag},
            {"variant": "rag_gpt_tfidf_top3_numeric_extraction_only", **rag},
        ],
    }
    for filename, table_rows in rows.items():
        pd.DataFrame(table_rows).to_csv(report_dir / filename, index=False)


def _maybe_extract_numeric(question: dict, retrieved_rows: list[dict], args) -> dict | None:
    if not getattr(args, "numeric_extraction", True):
        return None
    if question.get("question_type") != "numeric_fact":
        return None
    return extract_numeric_answer(question, retrieved_rows)


def _extraction_generation(extraction: dict, local_gpt: LocalGPT, prompt: str) -> dict:
    prompt_tokens = len(local_gpt.tokenizer.encode(prompt))
    text = extraction.get("answer", "")
    completion_tokens = len(local_gpt.tokenizer.encode(text))
    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_seconds": 0.0,
        "checkpoint_path": local_gpt.config.checkpoint_path,
        "tokenizer_dir": local_gpt.config.tokenizer_dir,
        "tokenizer_vocab_size": local_gpt.tokenizer.vocab_size,
        "device": str(local_gpt.device),
        "generation_error": False,
    }
