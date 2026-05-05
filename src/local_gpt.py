"""Wrap the assignment GPTModel for generation and perplexity scoring."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint import GPTConfig, load_model_from_checkpoint, select_device
from .hftokenizer import HFTokenizer


@dataclass
class LocalGPTConfig:
    checkpoint_path: str = "model/model_weights.pt"
    tokenizer_dir: str = "model/hftokenizer"
    d_model: int | None = None
    n_heads: int | None = None
    layers: int | None = None
    vocab_size: int | None = None
    max_seq_len: int | None = None
    device: str = "auto"
    max_new_tokens: int = 80
    temperature: float = 1.0
    top_k: int | None = 50
    greedy: bool = True
    allow_random_init: bool = False


class LocalGPT:
    """Local-only inference interface for the course GPT checkpoint."""

    def __init__(self, config: LocalGPTConfig):
        self.config = config
        self.device = select_device(config.device)
        self.tokenizer = HFTokenizer(config.tokenizer_dir)

        checkpoint_path = Path(config.checkpoint_path)
        vocab_size = config.vocab_size
        if not checkpoint_path.exists() and config.allow_random_init:
            vocab_size = self.tokenizer.vocab_size

        self.model, self.model_config, self.checkpoint_format = load_model_from_checkpoint(
            checkpoint_path,
            device=config.device,
            allow_random_init=config.allow_random_init,
            d_model=config.d_model,
            n_heads=config.n_heads,
            layers=config.layers,
            vocab_size=vocab_size,
            max_seq_len=config.max_seq_len,
        )
        self.model.eval()
        self._validate_tokenizer_checkpoint()
        self.tokenizer_metadata = self._build_tokenizer_metadata()

    @property
    def gpt_config(self) -> GPTConfig:
        return self.model_config

    def _validate_tokenizer_checkpoint(self) -> None:
        checkpoint_vocab_size = self.model_config.vocab_size
        tokenizer_vocab_size = self.tokenizer.vocab_size
        print(f"checkpoint path: {self.config.checkpoint_path}")
        print(f"tokenizer dir: {self.config.tokenizer_dir}")
        print(f"tokenizer vocab size: {tokenizer_vocab_size}")
        print(f"checkpoint vocab size: {checkpoint_vocab_size}")
        if checkpoint_vocab_size != tokenizer_vocab_size:
            print("compatibility result: FAIL")
            raise ValueError(
                "Tokenizer/checkpoint mismatch. "
                f"The checkpoint expects vocab_size={checkpoint_vocab_size}, "
                f"but {self.config.tokenizer_dir} has vocab_size={tokenizer_vocab_size}. "
                "Provide the tokenizer used to train model/model_weights.pt."
            )
        print("compatibility result: PASS")

    def _build_tokenizer_metadata(self) -> dict:
        metadata = self.tokenizer.metadata
        metadata.update(
            {
                "checkpoint_path": self.config.checkpoint_path,
                "checkpoint_vocab_size": self.model_config.vocab_size,
                "compatible_with_checkpoint": self.model_config.vocab_size == self.tokenizer.vocab_size,
            }
        )
        return metadata

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> dict:
        """Generate a short autoregressive completion from a prompt."""
        start = time.perf_counter()
        max_new = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.eos_token_id or 0]
        prompt_ids = prompt_ids[-self.model_config.max_seq_len :]
        generated: list[int] = []

        try:
            with torch.no_grad():
                ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                for _ in range(max_new):
                    ids_cond = ids[:, -self.model_config.max_seq_len :]
                    logits = self.model(ids_cond)
                    next_id = self._decode_next_token(logits[:, -1, :])
                    generated.append(next_id)
                    next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                    ids = torch.cat([ids, next_tensor], dim=1)
                    if self.tokenizer.eos_token_id is not None and next_id == self.tokenizer.eos_token_id:
                        break
            text = self.tokenizer.decode(generated)
            error = False
        except Exception as exc:
            text = f"[generation error: {exc}]"
            error = True

        latency = time.perf_counter() - start
        return {
            "text": text,
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(generated),
            "latency_seconds": latency,
            "checkpoint_path": self.config.checkpoint_path,
            "tokenizer_dir": self.config.tokenizer_dir,
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
            "device": str(self.device),
            "generation_error": error,
        }

    def score_loss(self, prompt: str, target: str) -> dict:
        """Compute next-token loss/perplexity for a gold target suffix."""
        prompt_ids = self.tokenizer.encode(prompt)
        target_ids = self.tokenizer.encode(target)
        if not target_ids:
            return {"loss": 0.0, "perplexity": 1.0, "num_tokens": 0}

        full_ids = prompt_ids + target_ids
        if len(full_ids) < 2:
            return {"loss": 0.0, "perplexity": 1.0, "num_tokens": 0}

        crop_start = max(0, len(full_ids) - self.model_config.max_seq_len)
        cropped = full_ids[crop_start:]
        if len(cropped) < 2:
            return {"loss": 0.0, "perplexity": 1.0, "num_tokens": 0}

        input_ids = torch.tensor([cropped[:-1]], dtype=torch.long, device=self.device)
        labels = torch.tensor([cropped[1:]], dtype=torch.long, device=self.device)
        label_positions = list(range(crop_start + 1, crop_start + len(cropped)))
        target_mask = torch.tensor(
            [position >= len(prompt_ids) for position in label_positions],
            dtype=torch.bool,
            device=self.device,
        )
        if not bool(target_mask.any()):
            return {"loss": float("nan"), "perplexity": float("nan"), "num_tokens": 0}

        with torch.no_grad():
            logits = self.model(input_ids)
            losses = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
            selected = losses[target_mask]
            loss = float(selected.mean().item())
        return {
            "loss": loss,
            "perplexity": float(math.exp(min(loss, 50.0))),
            "num_tokens": int(target_mask.sum().item()),
        }

    def _decode_next_token(self, logits: torch.Tensor) -> int:
        if self.config.greedy or self.config.temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())
        logits = logits / max(self.config.temperature, 1e-6)
        if self.config.top_k is not None and self.config.top_k > 0:
            k = min(self.config.top_k, logits.size(-1))
            values, indices = torch.topk(logits, k)
            probs = torch.softmax(values, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            return int(indices.gather(-1, sampled).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
