from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import (
    checkpoint_format,
    checkpoint_parameter_shapes,
    extract_state_dict,
    infer_gpt_config,
    load_checkpoint_object,
)
from src.hftokenizer import HFTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GPT checkpoint architecture and tokenizer compatibility.")
    parser.add_argument("--checkpoint", default="model/model_weights.pt")
    parser.add_argument("--tokenizer-dir", default="model/hftokenizer")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    args = parser.parse_args()

    checkpoint = load_checkpoint_object(args.checkpoint)
    state_dict = extract_state_dict(checkpoint)
    config = infer_gpt_config(
        state_dict,
        checkpoint=checkpoint,
        d_model=args.d_model,
        n_heads=args.n_heads,
        layers=args.layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
    )
    tokenizer = HFTokenizer(args.tokenizer_dir)

    print(f"checkpoint path: {args.checkpoint}")
    print(f"checkpoint format: {checkpoint_format(checkpoint)}")
    print(f"checkpoint vocab size: {config.vocab_size}")
    print("embedding/output layer shapes:")
    for name, shape in checkpoint_parameter_shapes(state_dict).items():
        print(f"  {name}: {shape}")
    print("inferred architecture:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  layers: {config.layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"tokenizer dir: {args.tokenizer_dir}")
    print(f"tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"compatibility: {'PASS' if tokenizer.vocab_size == config.vocab_size else 'FAIL'}")


if __name__ == "__main__":
    main()
