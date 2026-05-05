"""Rebuild standard or chunk-ablation reports from saved outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting import build_chunk_ablation_report, build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild final report files from saved run outputs.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment", default="standard", choices=["standard", "chunk_ablation"])
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()
    if args.experiment == "chunk_ablation":
        report_path = build_chunk_ablation_report(args.run_id, outputs_dir=args.outputs_dir)
    else:
        report_path = build_report(args.run_id, outputs_dir=args.outputs_dir)
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
