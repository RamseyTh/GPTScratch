from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.checkpoint import infer_checkpoint_vocab_size
from src.hftokenizer import HFTokenizer
from src.pipeline import PipelineRunner
from src.utils import question_source_for_path, resolve_question_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline GPT vs RAG-GPT over financial filing questions.")
    parser.add_argument("--quick-run", action="store_true", help="Run the clean default verified-question RAG workflow.")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--chunks", default="outputs/chunks/chunks.jsonl")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--checkpoint", default="model/model_weights.pt")
    parser.add_argument("--tokenizer-dir", default="model/hftokenizer")
    parser.add_argument("--run-id", default="run")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k-sampling", type=int, default=50)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--inspect-tokenizer", action="store_true", help="Inspect tokenizer and checkpoint compatibility, then exit.")
    parser.add_argument("--systems", default="baseline_gpt,rag_gpt_tfidf_top3,oracle_gpt,random_context_gpt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-research-questions", type=int, default=50)
    parser.add_argument("--allow-small-final-report", action="store_true")
    parser.add_argument("--allow-small-research-report", action="store_true")
    parser.add_argument("--strict-research-report", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--open-corpus", action="store_true", help="Disable ticker/year metadata filtering during retrieval.")
    parser.add_argument("--context-token-budget", type=int, default=700)
    parser.add_argument("--numeric-extraction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--numeric-extraction-only", action="store_true")
    parser.add_argument("--oracle-first", action="store_true")
    parser.add_argument("--strict-oracle-gate", action="store_true")
    parser.add_argument("--oracle-latency-warning-seconds", type=float, default=5.0)

    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--retrieval-method", default="tfidf", choices=["tfidf", "dense"])
    parser.add_argument("--retriever", dest="retrieval_method", choices=["tfidf", "dense"], default="tfidf")
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--top-k", dest="retrieval_top_k", type=int, default=3)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.inspect_tokenizer:
        inspect_tokenizer(args.tokenizer_dir, args.checkpoint)
        return
    runner = PipelineRunner.from_args(args)
    result = runner.run()
    print(f"Run complete: {result['run_dir']}")
    if result.get("report_path"):
        print(f"Report: {result['report_path']}")


def inspect_tokenizer(tokenizer_dir: str, checkpoint: str) -> None:
    tokenizer = HFTokenizer(tokenizer_dir)
    checkpoint_vocab_size = infer_checkpoint_vocab_size(checkpoint)
    print(f"checkpoint path: {checkpoint}")
    print(f"tokenizer dir: {tokenizer_dir}")
    print(f"tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"checkpoint vocab size: {checkpoint_vocab_size}")
    print(f"compatibility result: {'PASS' if tokenizer.vocab_size == checkpoint_vocab_size else 'FAIL'}")


def resolve_questions_path(path: str | None) -> str:
    resolved, _ = resolve_question_file(path)
    return str(resolved)


def question_source(path) -> str:
    return question_source_for_path(path, explicit=False)


if __name__ == "__main__":
    main()
