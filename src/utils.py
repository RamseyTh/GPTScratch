"""Shared file, path, text, and benchmark-resolution helpers."""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write a list of dictionaries as CSV with a stable union of columns."""
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: str | Path, default: Any | None = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_jsonl_rows(path: str | Path) -> int:
    """Count non-empty rows in a JSONL file without parsing every object."""
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def safe_copy(src: str | Path, dst: str | Path) -> Path:
    """Copy a file after creating the destination directory."""
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists():
        raise FileNotFoundError(f"Input file not found: {src_path}")
    ensure_dir(dst_path.parent)
    shutil.copyfile(src_path, dst_path)
    return dst_path


def normalize_whitespace(text: str) -> str:
    lines = []
    blank = False
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t\r\f\v]+", " ", raw_line).strip()
        if not line:
            if not blank:
                lines.append("")
            blank = True
        else:
            lines.append(line)
            blank = False
    return "\n".join(lines).strip()


def normalize_text(text: str) -> str:
    """Compatibility alias for whitespace normalization."""
    return normalize_whitespace(text)


def simple_word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def compact_text(text: str, max_chars: int) -> str:
    text = normalize_whitespace(text).replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def deterministic_sample(items: list[Any], k: int, seed: int = 13) -> list[Any]:
    if k <= 0 or not items:
        return []
    rng = random.Random(seed)
    if len(items) <= k:
        shuffled = list(items)
        rng.shuffle(shuffled)
        return shuffled
    return rng.sample(items, k)


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def resolve_question_file(explicit_path: str | Path | None = None) -> tuple[Path, str]:
    """Resolve the question file used by the main experiment.

    Cleaned research questions are preferred when present. Verified remapped
    questions remain the main fallback because their evidence IDs are aligned
    to the current chunk file.
    """
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return path, question_source_for_path(path, explicit=True)
        if path.name in {"questions_research.validated.jsonl", "questions_research.jsonl"}:
            for fallback, source in _default_question_candidates():
                if fallback.exists():
                    print(
                        f"WARNING: Requested question file {path} was not found. "
                        f"Using {fallback} as a diagnostic fallback."
                    )
                    return fallback, source
        return path, question_source_for_path(path, explicit=True)
    for path, source in _default_question_candidates():
        if path.exists():
            return path, source
    return Path("data/questions/sample_questions.jsonl"), "sample"


def _default_question_candidates() -> list[tuple[Path, str]]:
    return [
        (Path("data/questions/questions_research.validated.jsonl"), "research_validated"),
        (Path("data/questions/questions_research.jsonl"), "research"),
        (Path("data/questions/questions_verified.remapped.jsonl"), "verified_remapped"),
        (Path("data/questions/questions_verified.jsonl"), "verified"),
        (Path("data/questions/questions.jsonl"), "fallback"),
        (Path("data/questions/sample_questions.jsonl"), "sample"),
    ]


def question_source_for_path(path: str | Path, explicit: bool = False) -> str:
    name = Path(path).name
    if name == "questions_research.validated.jsonl":
        return "research_validated"
    if name == "questions_research.jsonl":
        return "research"
    if name == "questions_verified.remapped.jsonl":
        return "verified_remapped"
    if name == "questions_verified.jsonl":
        return "verified"
    if name == "questions.jsonl":
        return "fallback"
    if name == "sample_questions.jsonl":
        return "sample"
    return "explicit" if explicit else "custom"
