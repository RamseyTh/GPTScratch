from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hftokenizer import HFTokenizer, resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Hugging Face tokenizer from local model data.")
    parser.add_argument("--data", default="data/model/data.txt")
    parser.add_argument("--out-dir", default="model/hftokenizer")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--model-max-length", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_path = resolve_project_path(args.data)
    out_dir = resolve_project_path(args.out_dir)
    if not data_path.exists():
        raise SystemExit(f"Training data not found: {data_path}")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} already exists. Pass --overwrite to replace it.")

    tokenizer = HFTokenizer.train_from_file(
        data_path=data_path,
        output_dir=out_dir,
        vocab_size=args.vocab_size,
        model_max_length=args.model_max_length,
    )
    print(f"training data path: {data_path}")
    print(f"output directory: {out_dir}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"model max length: {tokenizer.metadata['model_max_length']}")
    print("saved files:")
    for filename in (
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ):
        print(f"  {out_dir / filename}")


if __name__ == "__main__":
    main()
