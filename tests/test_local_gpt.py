from __future__ import annotations

import math


def test_tiny_local_gpt_generates_and_scores(tmp_path, tiny_tokenizer_dir):
    import torch

    from src.gpt import GPTModel
    from src.hftokenizer import HFTokenizer
    from src.local_gpt import LocalGPT, LocalGPTConfig

    tokenizer = HFTokenizer(tiny_tokenizer_dir)
    checkpoint_path = tmp_path / "tiny.pt"
    torch.save(
        GPTModel(d_model=16, n_heads=8, layers=1, vocab_size=tokenizer.vocab_size, max_seq_len=16).state_dict(),
        checkpoint_path,
    )

    model = LocalGPT(
        LocalGPTConfig(
            checkpoint_path=str(checkpoint_path),
            tokenizer_dir=str(tiny_tokenizer_dir),
            max_new_tokens=3,
            device="cpu",
        )
    )
    generated = model.generate("Question: hello Answer:")
    assert generated["generation_error"] is False
    assert generated["completion_tokens"] >= 1
    assert generated["tokenizer_dir"] == str(tiny_tokenizer_dir)
    assert generated["tokenizer_vocab_size"] == tokenizer.vocab_size

    scored = model.score_loss("Question: hello Answer:", "world")
    assert scored["num_tokens"] > 0
    assert math.isfinite(scored["loss"])
    assert math.isfinite(scored["perplexity"])


def test_random_init_uses_hf_tokenizer_vocab(tmp_path, tiny_tokenizer_dir):
    from src.hftokenizer import HFTokenizer
    from src.local_gpt import LocalGPT, LocalGPTConfig

    tokenizer = HFTokenizer(tiny_tokenizer_dir)
    model = LocalGPT(
        LocalGPTConfig(
            checkpoint_path=str(tmp_path / "missing.pt"),
            tokenizer_dir=str(tiny_tokenizer_dir),
            d_model=16,
            n_heads=8,
            layers=1,
            max_seq_len=16,
            max_new_tokens=1,
            device="cpu",
            allow_random_init=True,
        )
    )

    assert model.gpt_config.vocab_size == tokenizer.vocab_size
