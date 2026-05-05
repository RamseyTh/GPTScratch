from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, read_jsonl, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install verified financial QA benchmark artifacts.")
    parser.add_argument("--verified-jsonl", default=None)
    parser.add_argument("--verified-csv", default=None)
    parser.add_argument("--evidence-audit", default=None)
    parser.add_argument("--validation-report", default=None)
    parser.add_argument("--source-dir", default=None, help="Directory containing uploaded verified question files.")
    parser.add_argument("--out-dir", default="data/questions")
    parser.add_argument("--list-candidates", action="store_true", help="List discovered candidate files and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be installed without copying files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing installed artifacts.")
    parser.add_argument("--project-root", default=None, help="Project root. Defaults to this repository root.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else ROOT
    if args.list_candidates:
        print_candidate_report(project_root, args.source_dir)
        return
    summary = install_verified_questions(
        verified_jsonl=args.verified_jsonl,
        verified_csv=args.verified_csv,
        evidence_audit=args.evidence_audit,
        validation_report=args.validation_report,
        out_dir=args.out_dir,
        source_dir=args.source_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        project_root=project_root,
    )
    print_install_summary(summary)


def install_verified_questions(
    verified_jsonl: str | None = None,
    verified_csv: str | None = None,
    evidence_audit: str | None = None,
    validation_report: str | None = None,
    out_dir: str | Path = "data/questions",
    source_dir: str | Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    project_root: str | Path | None = None,
) -> dict:
    project_root = Path(project_root or ROOT)
    out_dir = resolve_project_path(out_dir, project_root)
    if not dry_run:
        ensure_dir(out_dir)
    jsonl_path = resolve_input(
        verified_jsonl,
        "questions_verified.jsonl",
        source_dir=source_dir,
        project_root=project_root,
        required=True,
    )
    if jsonl_path is None:
        raise missing_required_error("questions_verified.jsonl", project_root, source_dir)

    copied: dict[str, Path] = {}
    copied["questions_verified_jsonl"] = copy_file(jsonl_path, out_dir / "questions_verified.jsonl", dry_run=dry_run, overwrite=overwrite)
    copied["questions_jsonl"] = copy_file(jsonl_path, out_dir / "questions.jsonl", dry_run=dry_run, overwrite=overwrite)

    for label, user_path, filename in (
        ("questions_verified_csv", verified_csv, "questions_verified.csv"),
        ("evidence_audit", evidence_audit, "evidence_audit.md"),
        ("validation_report", validation_report, "validation_report.md"),
    ):
        source = resolve_input(user_path, filename, source_dir=source_dir, project_root=project_root, required=False)
        if source is not None:
            copied[label] = copy_file(source, out_dir / filename, dry_run=dry_run, overwrite=overwrite)
        else:
            print(f"WARNING: optional file not found: {filename}")

    rows = read_jsonl(jsonl_path)
    type_counts = Counter(str(row.get("question_type", "")) for row in rows)
    ticker_counts = Counter(str(row.get("ticker", "")) for row in rows if row.get("ticker"))
    refusal_counts = Counter(str(bool(row.get("expected_refusal", False))).lower() for row in rows)

    distribution_path = out_dir / "question_distribution.csv"
    if not dry_run:
        with distribution_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["field", "value", "count"])
            writer.writeheader()
            for key, count in sorted(type_counts.items()):
                writer.writerow({"field": "question_type", "value": key, "count": count})
            for key, count in sorted(ticker_counts.items()):
                writer.writerow({"field": "ticker", "value": key, "count": count})
            for key, count in sorted(refusal_counts.items()):
                writer.writerow({"field": "expected_refusal", "value": key, "count": count})

    summary = {
        "num_questions": len(rows),
        "question_type_counts": dict(type_counts),
        "ticker_counts": dict(ticker_counts),
        "expected_refusal_counts": dict(refusal_counts),
        "source_paths": {
            "questions_verified_jsonl": str(jsonl_path),
        },
        "output_paths": {label: str(path) for label, path in copied.items()},
        "question_distribution_csv": str(distribution_path),
        "dry_run": dry_run,
    }
    if not dry_run:
        write_json(out_dir / "install_summary.json", summary)
    return summary


def copy_file(source: Path, destination: Path, dry_run: bool = False, overwrite: bool = False) -> Path:
    if not dry_run:
        ensure_dir(destination.parent)
    if source.resolve() != destination.resolve():
        if destination.exists() and overwrite:
            print(f"Overwriting existing file: {destination}")
        if not dry_run:
            shutil.copyfile(source, destination)
    return destination


def resolve_input(
    path: str | None,
    filename: str,
    source_dir: str | Path | None = None,
    project_root: Path = ROOT,
    required: bool = False,
) -> Path | None:
    locations = search_locations(project_root, source_dir)
    if path:
        for candidate in explicit_path_candidates(path, filename, project_root, source_dir):
            if candidate.exists():
                return candidate
        # Treat a missing basename as a discovery hint, so the common command
        # --verified-jsonl questions_verified.jsonl still works when files are
        # in /mnt/data, data/questions, or another searched upload folder.
        if Path(path).name != filename or Path(path).parent != Path("."):
            if required:
                raise missing_required_error(filename, project_root, source_dir, explicit_path=path)
            print(f"WARNING: optional file not found: {path}")
            return None
    for base in locations:
        candidate = base / filename
        if candidate.exists():
            return candidate
    if required:
        raise missing_required_error(filename, project_root, source_dir, explicit_path=path)
    return None


def explicit_path_candidates(path: str | Path, filename: str, project_root: Path, source_dir: str | Path | None) -> list[Path]:
    raw = Path(path).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([Path.cwd() / raw, project_root / raw])
        if source_dir:
            source = resolve_project_path(source_dir, project_root)
            candidates.append(source / raw)
        # If only a basename was provided, also search for the expected file
        # name in every discovery location.
        if raw.name == filename and raw.parent == Path("."):
            candidates.extend(base / filename for base in search_locations(project_root, source_dir))
    return _unique_paths(candidates)


def search_locations(project_root: Path = ROOT, source_dir: str | Path | None = None) -> list[Path]:
    locations = [
        resolve_project_path(source_dir, project_root) if source_dir else None,
        Path.cwd(),
        project_root,
        Path("/mnt/data"),
        project_root / "data" / "questions",
        project_root / "outputs" / "questions" / "verified",
        project_root.parent,
        Path.home() / "Downloads",
    ]
    return _unique_paths(path for path in locations if path is not None)


def candidate_files(project_root: Path = ROOT, source_dir: str | Path | None = None) -> list[Path]:
    patterns = ["*questions*.jsonl", "*verified*.jsonl", "evidence_audit.md", "validation_report.md"]
    found: list[Path] = []
    for base in search_locations(project_root, source_dir):
        for pattern in patterns:
            found.extend(base.glob(pattern))
    return _unique_paths(found)


def print_candidate_report(project_root: Path = ROOT, source_dir: str | Path | None = None) -> None:
    print("Searched locations:")
    for location in search_locations(project_root, source_dir):
        print(f"- {location}")
    print("Found candidate files:")
    candidates = candidate_files(project_root, source_dir)
    if not candidates:
        print("- none")
    for candidate in candidates:
        print(f"- {candidate}")


def missing_required_error(filename: str, project_root: Path, source_dir: str | Path | None, explicit_path: str | None = None) -> FileNotFoundError:
    lines = [f"Required verified question file not found: {filename}"]
    if explicit_path:
        lines.append(f"Requested path: {explicit_path}")
    lines.append("")
    lines.append("Searched locations:")
    for location in search_locations(project_root, source_dir):
        lines.append(f"- {location}")
    lines.append("")
    lines.append("Found candidate files:")
    candidates = candidate_files(project_root, source_dir)
    if candidates:
        lines.extend(f"- {candidate}" for candidate in candidates)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Copy questions_verified.jsonl into the repo root or pass --source-dir /path/to/uploaded/files.")
    return FileNotFoundError("\n".join(lines))


def resolve_project_path(path: str | Path | None, project_root: Path = ROOT) -> Path:
    if path is None:
        return project_root
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return project_root / path


def _unique_paths(paths) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path).expanduser()
        key = resolved.resolve() if resolved.exists() else resolved.absolute()
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def print_install_summary(summary: dict) -> None:
    paths = summary["output_paths"]
    print("Verified question installation plan:" if summary.get("dry_run") else "Installed verified questions:")
    print(f"  total: {summary['num_questions']}")
    print(f"  jsonl: {paths.get('questions_verified_jsonl')}")
    print(f"  default questions: {paths.get('questions_jsonl')}")
    print(f"  csv: {paths.get('questions_verified_csv', 'not installed')}")
    print(f"  evidence audit: {paths.get('evidence_audit', 'not installed')}")
    print(f"  validation report: {paths.get('validation_report', 'not installed')}")
    print(f"  question type counts: {summary['question_type_counts']}")
    print(f"  ticker counts: {summary['ticker_counts']}")
    print(f"  expected_refusal counts: {summary['expected_refusal_counts']}")


if __name__ == "__main__":
    main()
