"""Small coordinator for the clean local GPT RAG workflow.

`main.py` owns argument parsing.  This module resolves defaults, checks that
the local tokenizer and checkpoint agree, and then hands execution to
`src.experiments`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .checkpoint import infer_checkpoint_vocab_size
from .hftokenizer import HFTokenizer
from .experiments import run_experiment as execute_experiment
from .reporting import build_chunk_ablation_report, build_report
from .utils import ensure_dir, read_jsonl, resolve_question_file

# Tests and older scripts monkeypatch `src.pipeline.run_experiment`, so keep a
# module-level alias while the public implementation uses PipelineRunner.
run_experiment = execute_experiment


DEFAULT_SYSTEMS = ["baseline_gpt", "rag_gpt_tfidf_top3", "oracle_gpt", "random_context_gpt"]


@dataclass
class PipelineConfig:
    """Configuration for a quick-run or report-only project execution."""

    run_id: str
    quick_run: bool = False
    report: bool = False
    report_only: bool = False
    questions: Path | None = None
    chunks: Path = Path("outputs/chunks/chunks.jsonl")
    checkpoint: Path = Path("model/model_weights.pt")
    tokenizer_dir: Path = Path("model/hftokenizer")
    systems: list[str] = field(default_factory=lambda: list(DEFAULT_SYSTEMS))
    top_k: int = 3
    context_token_budget: int = 700
    limit: int | None = None
    overwrite: bool = False
    resume: bool = False


class PipelineRunner:
    """Resolve inputs, validate model/tokenizer compatibility, and run experiments."""

    def __init__(self, config: PipelineConfig, args: Any | None = None):
        self.config = config
        self.args = args
        self.compatibility_pass = False
        self.question_file: Path | None = None
        self.question_source = ""
        self.num_questions = 0
        self.result: dict[str, Any] = {}

    @classmethod
    def from_args(cls, args) -> "PipelineRunner":
        """Build a runner from parsed CLI arguments."""
        systems = _parse_systems(getattr(args, "systems", None))
        resolved_questions = Path(args.questions) if getattr(args, "questions", None) else None
        config = PipelineConfig(
            run_id=args.run_id,
            quick_run=bool(getattr(args, "quick_run", False)),
            report=bool(getattr(args, "report", False)),
            report_only=bool(getattr(args, "report_only", False)),
            questions=resolved_questions,
            chunks=Path(args.chunks),
            checkpoint=Path(args.checkpoint),
            tokenizer_dir=Path(args.tokenizer_dir),
            systems=systems,
            top_k=int(getattr(args, "retrieval_top_k", 3)),
            context_token_budget=int(getattr(args, "context_token_budget", 700)),
            limit=getattr(args, "limit", None),
            overwrite=bool(getattr(args, "overwrite", False)),
            resume=bool(getattr(args, "resume", False)),
        )
        args.systems = systems
        return cls(config, args=args)

    def run(self) -> dict[str, Any]:
        """Coordinate input resolution, execution, and optional reporting."""
        self.bootstrap_outputs()
        self.resolve_inputs()
        if self.config.report_only:
            report_path = self.write_report()
            self.result = {"run_dir": str(Path("outputs") / "runs" / self.config.run_id), "report_path": str(report_path)}
            return self.summary()
        self.check_tokenizer_and_checkpoint()
        self.ensure_chunks()
        self.load_questions()
        self.print_startup_summary()
        self.run_experiment()
        if self.config.report:
            report_path = self.write_report()
            self.result["report_path"] = str(report_path)
        return self.summary()

    def bootstrap_outputs(self) -> None:
        """Create top-level output folders used by all workflows."""
        ensure_dir("outputs/runs")
        ensure_dir("outputs/reports")
        ensure_dir("outputs/chunks")
        ensure_dir("outputs/indexes")

    def resolve_inputs(self) -> None:
        """Resolve quick-run question defaults and pass them back to args."""
        path, source = resolve_question_file(self.config.questions)
        self.question_file = path
        self.question_source = source
        self.args.questions = str(path)
        self.args.question_source = source
        self.args.chunks = str(self.config.chunks)
        self.args.checkpoint = str(self.config.checkpoint)
        self.args.tokenizer_dir = str(self.config.tokenizer_dir)
        self.args.systems = self.config.systems

    def check_model_and_tokenizer(self) -> None:
        """Fail early if the HF tokenizer does not match the checkpoint vocab."""
        tokenizer = HFTokenizer(self.config.tokenizer_dir)
        checkpoint_vocab = infer_checkpoint_vocab_size(self.config.checkpoint)
        if tokenizer.vocab_size != checkpoint_vocab:
            raise ValueError(
                "Tokenizer/checkpoint mismatch. "
                f"The checkpoint expects vocab_size={checkpoint_vocab}, but {self.config.tokenizer_dir} "
                f"has vocab_size={tokenizer.vocab_size}."
            )
        self.compatibility_pass = True

    def check_tokenizer_and_checkpoint(self) -> None:
        """Backward-compatible name used by some tests and earlier prompts."""
        self.check_model_and_tokenizer()

    def ensure_chunks(self) -> None:
        """Let the experiment builder create standard chunks when needed."""
        if getattr(self.args, "experiment", "standard") == "chunk_ablation":
            return
        if not Path(self.args.chunks).exists():
            print(f"Chunks missing at {self.args.chunks}; they will be built from {self.args.data_dir}.")

    def load_questions(self) -> None:
        """Count questions after any limit so startup output is honest."""
        rows = read_jsonl(self.args.questions)
        self.num_questions = min(len(rows), self.config.limit) if self.config.limit else len(rows)

    def run_experiment(self) -> None:
        """Run the configured standard experiment or chunking ablation."""
        original_report = bool(getattr(self.args, "report", False))
        self.args.report = False
        try:
            self.result = run_experiment(self.args)
        finally:
            self.args.report = original_report

    def write_report(self) -> Path:
        """Rebuild the requested report from saved run artifacts."""
        if getattr(self.args, "experiment", "standard") == "chunk_ablation":
            return build_chunk_ablation_report(self.config.run_id)
        return build_report(self.config.run_id)

    def print_startup_summary(self) -> None:
        """Print a concise reproducibility summary before generation starts."""
        print(f"Project root: {Path.cwd()}")
        print(f"Run ID: {self.config.run_id}")
        print(f"Mode: {'quick-run' if self.config.quick_run else 'custom'}")
        print(f"Experiment: {getattr(self.args, 'experiment', 'standard')}")
        print(f"Checkpoint: {self.config.checkpoint}")
        print(f"Tokenizer: {self.config.tokenizer_dir}")
        print(f"Tokenizer/checkpoint compatibility: {'PASS' if self.compatibility_pass else 'FAIL'}")
        print(f"Questions: {self.question_file}")
        print(f"Question source: {self.question_source}")
        print(f"Questions loaded: {self.num_questions}")
        print(f"Chunks: {self.config.chunks}")
        print(f"Systems: {', '.join(self.config.systems)}")
        print(f"Report: {'enabled' if self.config.report else 'disabled'}")

    def summary(self) -> dict[str, Any]:
        """Return the experiment result in the shape expected by `main.py`."""
        return self.result or {
            "run_dir": str(Path("outputs") / "runs" / self.config.run_id),
            "report_path": None,
        }


def _parse_systems(value) -> list[str]:
    if value is None:
        return list(DEFAULT_SYSTEMS)
    if isinstance(value, list):
        return value or list(DEFAULT_SYSTEMS)
    systems = [item.strip() for item in str(value).split(",") if item.strip()]
    return systems or list(DEFAULT_SYSTEMS)
