from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .checkpoint import infer_checkpoint_vocab_size
from .experiments import ensure_chunks, ensure_questions, run_experiment
from .hftokenizer import HFTokenizer
from .utils import ensure_dir, project_root, resolve_question_file


DEFAULT_SYSTEMS = ["baseline_gpt", "rag_gpt_tfidf_top3", "oracle_gpt", "random_context_gpt"]


@dataclass
class PipelineConfig:
    run_id: str
    quick_run: bool
    questions: Path | None
    chunks: Path
    checkpoint: Path
    tokenizer_dir: Path
    report: bool
    limit: int | None
    systems: list[str]
    top_k: int
    context_token_budget: int
    overwrite: bool
    resume: bool


class PipelineRunner:
    def __init__(self, config: PipelineConfig, args=None):
        self.config = config
        self.args = args or SimpleNamespace()
        self.question_source = "sample"
        self.questions_loaded = 0
        self.compatibility_pass = False

    @classmethod
    def from_args(cls, args) -> "PipelineRunner":
        systems = parse_systems(getattr(args, "systems", None))
        config = PipelineConfig(
            run_id=args.run_id,
            quick_run=bool(getattr(args, "quick_run", False)),
            questions=Path(args.questions) if args.questions else None,
            chunks=Path(args.chunks or "outputs/chunks/chunks.jsonl"),
            checkpoint=Path(args.checkpoint or "model/model_weights.pt"),
            tokenizer_dir=Path(args.tokenizer_dir or "model/hftokenizer"),
            report=bool(args.report),
            limit=args.limit,
            systems=systems,
            top_k=int(getattr(args, "retrieval_top_k", None) or 3),
            context_token_budget=int(getattr(args, "context_token_budget", 700)),
            overwrite=bool(getattr(args, "overwrite", False)),
            resume=bool(getattr(args, "resume", False)),
        )
        return cls(config, args=args)

    def run(self) -> dict:
        self.bootstrap_outputs()
        self.resolve_inputs()
        self.check_tokenizer_and_checkpoint()
        self.ensure_chunks()
        self.load_questions()
        self.print_startup_summary()
        summary = self.run_experiment()
        if self.config.report:
            self.write_report()
        return summary

    def bootstrap_outputs(self) -> None:
        ensure_dir("outputs/runs")
        ensure_dir("outputs/reports")
        ensure_dir("outputs/chunks")

    def resolve_inputs(self) -> None:
        question_path, source = resolve_question_file(self.config.questions)
        self.config.questions = question_path
        self.question_source = source
        self.args.questions = str(question_path)
        self.args.question_source = source
        self.args.chunks = str(self.config.chunks)
        self.args.checkpoint = str(self.config.checkpoint)
        self.args.tokenizer_dir = str(self.config.tokenizer_dir)
        self.args.retrieval_top_k = self.config.top_k
        self.args.context_token_budget = self.config.context_token_budget
        self.args.systems = self.config.systems

    def check_tokenizer_and_checkpoint(self) -> None:
        tokenizer = HFTokenizer(self.config.tokenizer_dir)
        checkpoint_vocab_size = infer_checkpoint_vocab_size(self.config.checkpoint)
        if tokenizer.vocab_size != checkpoint_vocab_size:
            raise ValueError(
                "Tokenizer/checkpoint compatibility failed. "
                f"Tokenizer vocab_size={tokenizer.vocab_size}; checkpoint vocab_size={checkpoint_vocab_size}."
            )
        self.compatibility_pass = True

    def ensure_chunks(self) -> None:
        ensure_chunks(str(self.config.chunks), self.args.data_dir, self.args.chunk_size, self.args.overlap)

    def load_questions(self) -> None:
        questions = ensure_questions(str(self.config.questions))
        if self.config.limit is not None:
            questions = questions[: self.config.limit]
        self.questions_loaded = len(questions)

    def run_experiment(self) -> dict:
        return run_experiment(self.args)

    def write_report(self) -> None:
        # run_experiment writes the report when args.report is true.
        return None

    def print_startup_summary(self) -> None:
        mode = "quick-run" if self.config.quick_run else "standard"
        systems = ", ".join(self.config.systems)
        print(f"Project root: {project_root()}")
        print(f"Run ID: {self.config.run_id}")
        print(f"Mode: {mode}")
        print(f"Checkpoint: {self.config.checkpoint}")
        print(f"Tokenizer: {self.config.tokenizer_dir}")
        print(f"Tokenizer/checkpoint compatibility: {'PASS' if self.compatibility_pass else 'FAIL'}")
        print(f"Questions: {self.config.questions}")
        print(f"Question source: {self.question_source}")
        print(f"Questions loaded: {self.questions_loaded}")
        print(f"Chunks: {self.config.chunks}")
        print(f"Systems: {systems}")
        print(f"Report: {'enabled' if self.config.report else 'disabled'}")


def parse_systems(value: str | list[str] | None) -> list[str]:
    if value is None:
        return list(DEFAULT_SYSTEMS)
    if isinstance(value, list):
        return value
    systems = [item.strip() for item in value.split(",") if item.strip()]
    return systems or list(DEFAULT_SYSTEMS)
