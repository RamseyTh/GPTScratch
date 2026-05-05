from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import sample_questions
from src.utils import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an editable JSONL question template.")
    parser.add_argument("--out", default="data/questions/sample_questions.jsonl")
    args = parser.parse_args()
    write_jsonl(args.out, sample_questions())
    print(f"Wrote sample questions to {args.out}")


if __name__ == "__main__":
    main()

