from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_tiny_hf_tokenizer(tokenizer_dir: Path, vocab_size: int | None = None) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokens = [
        "<|endoftext|>",
        "Question",
        "Answer",
        "Context",
        "Gold",
        "evidence",
        "using",
        "the",
        "not",
        "enough",
        "information",
        "hello",
        "world",
        "Apple",
        "reported",
        "Services",
        "net",
        "sales",
        "in",
        "2025",
        "Microsoft",
        "revenue",
        "was",
        "10",
        "billion",
        ".",
        ":",
        "$",
        "What",
        "were",
    ]
    if vocab_size is not None:
        while len(tokens) < vocab_size:
            tokens.append(f"tok_{len(tokens)}")
    vocab = {token: index for index, token in enumerate(tokens)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        model_max_length=64,
    )
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(tokenizer_dir)
    return tokenizer_dir


@pytest.fixture
def tiny_tokenizer_dir(tmp_path: Path) -> Path:
    return build_tiny_hf_tokenizer(tmp_path / "hftokenizer")


def pytest_collection_modifyitems(items):
    items.sort(key=lambda item: str(item.path))
