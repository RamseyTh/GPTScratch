from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import Retriever, build_retrieval_query, retrieval_diagnostic, retrieval_filters
from src.utils import compact_text, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect retrieved chunks for a query or a question file.")
    parser.add_argument("--chunks", default="outputs/chunks/chunks.jsonl")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--method", default="tfidf", choices=["tfidf", "dense"])
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--year", default=None)
    parser.add_argument("--company", default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--show-gold", action="store_true")
    parser.add_argument("--open-corpus", action="store_true")
    args = parser.parse_args()

    chunks = read_jsonl(args.chunks)
    retriever = Retriever(chunks, method=args.method)
    if args.questions:
        questions = read_jsonl(args.questions)
        for question in questions[: args.sample or len(questions)]:
            query = build_retrieval_query(question)
            filters = retrieval_filters(question, open_corpus=args.open_corpus)
            results, latency = retriever.retrieve_with_latency(query, top_k=args.top_k, filters=filters, question=question)
            diag = retrieval_diagnostic(question, query, filters, results, latency)
            print(f"question_id={question.get('question_id')} coverage@3={diag['answer_coverage_at_3']} source_accuracy={diag['source_accuracy']} noise={diag['noise_reason']}")
            print(f"query={query}")
            if args.show_gold:
                print(f"gold={question.get('gold_evidence_text', '')}")
            _print_results(results)
        return

    if not args.query:
        raise SystemExit("Provide --query or --questions.")
    filters = {key: value for key, value in {"ticker": args.ticker, "year": args.year, "company": args.company}.items() if value}
    results = retriever.retrieve(args.query, top_k=args.top_k, filters=filters)
    _print_results(results)


def _print_results(results: list[dict]) -> None:
    for result in results:
        meta = result["metadata"]
        print(f"rank={result['rank']} score={result['score']:.4f} chunk_id={result['chunk_id']}")
        print(
            "source="
            f"{meta.get('source_file')} ticker={meta.get('ticker')} year={meta.get('year')} "
            f"section={meta.get('section_id')} type={meta.get('source_type')}"
        )
        print(compact_text(result["text"], 500))
        print()


if __name__ == "__main__":
    main()
