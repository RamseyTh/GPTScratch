"""Hugging Face tokenizer wrapper used by the local GPT checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer

from .utils import project_root


@dataclass
class HFTokenizerConfig:
    tokenizer_dir: str = "model/hftokenizer"
    train_data: str = "data/model/data.txt"
    vocab_size: int = 10000
    model_max_length: int = 1024


class HFTokenizer:
    """Thin project wrapper around the saved `model/hftokenizer` folder."""

    def __init__(self, tokenizer_dir: str | Path = "model/hftokenizer"):
        self.tokenizer_dir = resolve_project_path(tokenizer_dir)
        if not self.tokenizer_dir.exists():
            raise FileNotFoundError(
                f"Tokenizer directory not found: {self.tokenizer_dir}\n"
                "Provide the Hugging Face tokenizer folder at model/hftokenizer."
            )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        except Exception:
            # Some tokenizers runtimes cannot parse older tokenizer.json files.
            # The same HF folder can still be loaded from vocab.json/merges.txt.
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir, use_fast=False)
        self._ensure_endoftext_specials()

    @classmethod
    def train_from_file(
        cls,
        data_path: str | Path = "data/model/data.txt",
        output_dir: str | Path = "model/hftokenizer",
        vocab_size: int = 10000,
        model_max_length: int = 1024,
    ) -> "HFTokenizer":
        """Train a GPT-2 style tokenizer from local text and save it."""
        data_path = resolve_project_path(data_path)
        output_dir = resolve_project_path(output_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")

        base = AutoTokenizer.from_pretrained("gpt2")
        base.model_max_length = model_max_length
        base.eos_token = "<|endoftext|>"
        base.bos_token = "<|endoftext|>"
        base.unk_token = "<|endoftext|>"
        tokenizer = base.train_new_from_iterator(
            _line_iterator(data_path),
            vocab_size=vocab_size,
            limit_alphabet=500,
            new_special_tokens=["<|endoftext|>"],
        )
        tokenizer.model_max_length = model_max_length
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.bos_token = "<|endoftext|>"
        tokenizer.unk_token = "<|endoftext|>"
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        return cls(output_dir)

    def encode(self, text: str) -> list[int]:
        """Encode text to model token IDs without adding special tokens."""
        return list(self.tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode model token IDs back to text."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return int(getattr(self.tokenizer, "vocab_size", len(self.tokenizer)))

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def metadata(self) -> dict:
        return {
            "tokenizer_dir": str(self.tokenizer_dir),
            "tokenizer_class": type(self.tokenizer).__name__,
            "vocab_size": self.vocab_size,
            "model_max_length": self.tokenizer.model_max_length,
            "eos_token": self.tokenizer.eos_token,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token": self.tokenizer.bos_token,
            "unk_token": self.tokenizer.unk_token,
        }

    def first_vocab_entries(self, n: int = 20) -> list[tuple[int, str]]:
        vocab = self.tokenizer.get_vocab()
        return sorted((idx, token) for token, idx in vocab.items())[:n]

    def _ensure_endoftext_specials(self) -> None:
        vocab = self.tokenizer.get_vocab()
        if "<|endoftext|>" in vocab:
            self.tokenizer.eos_token = "<|endoftext|>"
            self.tokenizer.bos_token = "<|endoftext|>"
            self.tokenizer.unk_token = "<|endoftext|>"


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    rooted = project_root() / path
    return rooted if rooted.exists() or not path.exists() else path


def _line_iterator(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line
