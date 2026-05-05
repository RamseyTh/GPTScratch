from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import infer_checkpoint_vocab_size
from src.hftokenizer import HFTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the Hugging Face tokenizer and checkpoint compatibility.")
    parser.add_argument("--tokenizer-dir", default="model/hftokenizer")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sample", default="Apple reported Services net sales in 2025.")
    args = parser.parse_args()

    tokenizer = HFTokenizer(args.tokenizer_dir)
    metadata = tokenizer.metadata
    print(f"tokenizer dir: {metadata['tokenizer_dir']}")
    print(f"tokenizer class: {metadata['tokenizer_class']}")
    print(f"vocab size: {metadata['vocab_size']}")
    print(f"model max length: {metadata['model_max_length']}")
    print(f"eos token: {metadata['eos_token']}")
    print(f"eos token id: {metadata['eos_token_id']}")
    print(f"bos token: {metadata['bos_token']}")
    print(f"unk token: {metadata['unk_token']}")
    print("first 20 vocab entries:")
    for idx, token in tokenizer.first_vocab_entries(20):
        print(f"  {idx}: {token!r}")

    if args.checkpoint:
        checkpoint_vocab_size = infer_checkpoint_vocab_size(args.checkpoint)
        print(f"checkpoint vocab size: {checkpoint_vocab_size}")
        print(f"compatibility: {'PASS' if checkpoint_vocab_size == tokenizer.vocab_size else 'FAIL'}")

    ids = tokenizer.encode(args.sample)
    decoded = tokenizer.decode(ids)
    print(f"sample text: {args.sample}")
    print(f"encoded IDs: {ids}")
    print(f"decoded text: {decoded}")


if __name__ == "__main__":
    main()

