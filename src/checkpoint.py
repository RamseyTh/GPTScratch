from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .gpt import GPTModel


@dataclass
class GPTConfig:
    d_model: int
    n_heads: int
    layers: int
    vocab_size: int
    max_seq_len: int

    def to_kwargs(self) -> dict[str, int]:
        return asdict(self)


def select_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint_object(checkpoint_path: str | Path) -> Any:
    return torch.load(Path(checkpoint_path), map_location="cpu")


def checkpoint_format(checkpoint: Any) -> str:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return key
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return "raw_state_dict"
    return "unknown"


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return normalize_state_dict(value)
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return normalize_state_dict(checkpoint)
    raise ValueError("Could not find a model state_dict in checkpoint.")


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = str(key)
        for prefix in ("module.", "_orig_mod."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        normalized[clean_key] = value
    return normalized


def checkpoint_parameter_shapes(state_dict: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...] | None]:
    return {
        "word_embeddings.weight": tuple(state_dict["word_embeddings.weight"].shape)
        if "word_embeddings.weight" in state_dict
        else None,
        "position_embeddings.weight": tuple(state_dict["position_embeddings.weight"].shape)
        if "position_embeddings.weight" in state_dict
        else None,
        "fc_out.weight": tuple(state_dict["fc_out.weight"].shape) if "fc_out.weight" in state_dict else None,
        "fc_out.bias": tuple(state_dict["fc_out.bias"].shape) if "fc_out.bias" in state_dict else None,
    }


def infer_checkpoint_vocab_size_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int | None:
    embedding_vocab = int(state_dict["word_embeddings.weight"].shape[0]) if "word_embeddings.weight" in state_dict else None
    output_vocab = int(state_dict["fc_out.weight"].shape[0]) if "fc_out.weight" in state_dict else None
    if embedding_vocab is not None and output_vocab is not None and embedding_vocab != output_vocab:
        raise ValueError(
            "Checkpoint has inconsistent vocabulary sizes: "
            f"word_embeddings.weight={embedding_vocab}, fc_out.weight={output_vocab}."
        )
    return embedding_vocab if embedding_vocab is not None else output_vocab


def infer_checkpoint_vocab_size(checkpoint_path: str | Path) -> int:
    checkpoint = load_checkpoint_object(checkpoint_path)
    state_dict = extract_state_dict(checkpoint)
    vocab_size = infer_checkpoint_vocab_size_from_state_dict(state_dict)
    if vocab_size is None:
        raise ValueError("Could not infer checkpoint vocab_size from word_embeddings.weight or fc_out.weight.")
    return vocab_size


def infer_gpt_config(
    state_dict: dict[str, torch.Tensor],
    checkpoint: Any | None = None,
    d_model: int | None = None,
    n_heads: int | None = None,
    layers: int | None = None,
    vocab_size: int | None = None,
    max_seq_len: int | None = None,
) -> GPTConfig:
    embedded = state_dict.get("word_embeddings.weight")
    positioned = state_dict.get("position_embeddings.weight")

    inferred_vocab_size = infer_checkpoint_vocab_size_from_state_dict(state_dict)
    inferred_d_model = int(embedded.shape[1]) if embedded is not None else None
    inferred_max_seq_len = int(positioned.shape[0]) if positioned is not None else None

    layer_ids = {
        int(match.group(1))
        for key in state_dict
        for match in [re.match(r"layers\.(\d+)\.", key)]
        if match is not None
    }
    inferred_layers = max(layer_ids) + 1 if layer_ids else None

    raw_config = _checkpoint_config(checkpoint)
    raw_heads = raw_config.get("n_heads") or raw_config.get("num_heads") or raw_config.get("heads")

    final_d_model = d_model or inferred_d_model or _int_or_none(raw_config.get("d_model"))
    final_vocab_size = vocab_size or inferred_vocab_size or _int_or_none(raw_config.get("vocab_size"))
    final_max_seq_len = (
        max_seq_len
        or inferred_max_seq_len
        or _int_or_none(raw_config.get("max_seq_len"))
        or _int_or_none(raw_config.get("block_size"))
    )
    final_layers = layers or inferred_layers or _int_or_none(raw_config.get("layers")) or _int_or_none(raw_config.get("n_layers"))
    final_n_heads = n_heads or _int_or_none(raw_heads)
    if final_n_heads is None and final_d_model is not None and final_d_model % 8 == 0:
        final_n_heads = 8

    missing = [
        name
        for name, value in (
            ("d_model", final_d_model),
            ("n_heads", final_n_heads),
            ("layers", final_layers),
            ("vocab_size", final_vocab_size),
            ("max_seq_len", final_max_seq_len),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Could not infer GPT config fields: "
            + ", ".join(missing)
            + ". Provide explicit CLI args such as --d-model, --n-heads, --layers, --vocab-size, --max-seq-len."
        )
    if final_d_model % final_n_heads != 0:
        raise ValueError(f"d_model={final_d_model} must be divisible by n_heads={final_n_heads}.")
    return GPTConfig(
        d_model=int(final_d_model),
        n_heads=int(final_n_heads),
        layers=int(final_layers),
        vocab_size=int(final_vocab_size),
        max_seq_len=int(final_max_seq_len),
    )


def create_model(config: GPTConfig) -> GPTModel:
    return GPTModel(
        d_model=config.d_model,
        n_heads=config.n_heads,
        layers=config.layers,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
    )


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
    allow_random_init: bool = False,
    d_model: int | None = None,
    n_heads: int | None = None,
    layers: int | None = None,
    vocab_size: int | None = None,
    max_seq_len: int | None = None,
) -> tuple[GPTModel, GPTConfig, str]:
    path = Path(checkpoint_path)
    torch_device = select_device(device)
    if not path.exists():
        if not allow_random_init:
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Train one with: python scripts/train_local_gpt.py --data data/model/data.txt "
                "--tokenizer-dir model/hftokenizer --out model/model_weights.pt"
            )
        config = _random_init_config(d_model, n_heads, layers, vocab_size, max_seq_len)
        model = create_model(config).to(torch_device)
        model.eval()
        return model, config, "random_init"

    checkpoint = load_checkpoint_object(path)
    state_dict = extract_state_dict(checkpoint)
    config = infer_gpt_config(
        state_dict,
        checkpoint=checkpoint,
        d_model=d_model,
        n_heads=n_heads,
        layers=layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )
    model = create_model(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(torch_device)
    model.eval()
    return model, config, checkpoint_format(checkpoint)


def tokenizer_checkpoint_compatible(checkpoint_vocab_size: int, tokenizer_vocab_size: int) -> bool:
    return int(checkpoint_vocab_size) == int(tokenizer_vocab_size)


def _checkpoint_config(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    for key in ("config", "model_config", "hparams", "args"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _random_init_config(
    d_model: int | None,
    n_heads: int | None,
    layers: int | None,
    vocab_size: int | None,
    max_seq_len: int | None,
) -> GPTConfig:
    final_d_model = d_model or 64
    final_n_heads = n_heads or (8 if final_d_model % 8 == 0 else 1)
    if final_d_model % final_n_heads != 0:
        raise ValueError(f"d_model={final_d_model} must be divisible by n_heads={final_n_heads}.")
    return GPTConfig(
        d_model=final_d_model,
        n_heads=final_n_heads,
        layers=layers or 2,
        vocab_size=vocab_size or 128,
        max_seq_len=max_seq_len or 128,
    )

