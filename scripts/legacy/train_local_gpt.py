from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gpt import GPTModel
from src.hftokenizer import HFTokenizer, resolve_project_path
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPTModel using the project HF tokenizer.")
    parser.add_argument("--data", default="data/model/data.txt")
    parser.add_argument("--tokenizer-dir", default="model/hftokenizer")
    parser.add_argument("--out", default="model/model_weights.pt")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data_path = resolve_project_path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Training data not found: {data_path}")
    tokenizer = HFTokenizer(args.tokenizer_dir)
    text = data_path.read_text(encoding="utf-8", errors="ignore")
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= args.max_seq_len + 1:
        raise SystemExit("Training data is too short for the requested max sequence length.")

    device = choose_device(args.device)
    model = GPTModel(args.d_model, args.n_heads, args.layers, tokenizer.vocab_size, args.max_seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    data = torch.tensor(token_ids, dtype=torch.long)
    run_id = args.run_id or time.strftime("train_%Y%m%d_%H%M%S")
    train_dir = ensure_dir(Path("outputs") / "training" / run_id)
    log_path = train_dir / "train_log.jsonl"
    curve_path = train_dir / "train_curve.csv"
    summary_path = train_dir / "train_summary.json"

    start_time = time.perf_counter()
    initial_loss = None
    final_loss = None
    tokens_seen = 0
    with log_path.open("w", encoding="utf-8") as log_f, curve_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=["step", "loss", "perplexity", "elapsed_seconds"])
        writer.writeheader()
        for step in range(1, args.steps + 1):
            x, y = sample_batch(data, args.batch_size, args.max_seq_len, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            if initial_loss is None:
                initial_loss = loss_value
            final_loss = loss_value
            tokens_seen += args.batch_size * args.max_seq_len
            elapsed = time.perf_counter() - start_time
            row = {
                "step": step,
                "loss": loss_value,
                "perplexity": math.exp(min(loss_value, 50.0)),
                "elapsed_seconds": elapsed,
            }
            writer.writerow(row)
            log_f.write(json.dumps(row) + "\n")

    out_path = resolve_project_path(args.out)
    ensure_dir(out_path.parent)
    torch.save(model.state_dict(), out_path)
    elapsed = time.perf_counter() - start_time
    summary = {
        "training_data": str(data_path),
        "tokenizer_dir": args.tokenizer_dir,
        "vocab_size": tokenizer.vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "layers": args.layers,
        "max_seq_len": args.max_seq_len,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_perplexity": math.exp(min(final_loss or 0.0, 50.0)),
        "training_time_seconds": elapsed,
        "tokens_per_second": tokens_seen / elapsed if elapsed else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved model weights to {out_path}")
    print(f"Saved training summary to {summary_path}")


def sample_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[start : start + seq_len] for start in starts]).to(device)
    y = torch.stack([data[start + 1 : start + seq_len + 1] for start in starts]).to(device)
    return x, y


def choose_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
