from __future__ import annotations


def test_checkpoint_inference_and_loading(tmp_path):
    import torch

    from src.checkpoint import (
        checkpoint_parameter_shapes,
        extract_state_dict,
        infer_checkpoint_vocab_size,
        infer_gpt_config,
        load_checkpoint_object,
        load_model_from_checkpoint,
        tokenizer_checkpoint_compatible,
    )
    from src.gpt import GPTModel

    checkpoint_path = tmp_path / "tiny.pt"
    model = GPTModel(d_model=16, n_heads=8, layers=2, vocab_size=32, max_seq_len=24)
    torch.save(model.state_dict(), checkpoint_path)

    checkpoint = load_checkpoint_object(checkpoint_path)
    state_dict = extract_state_dict(checkpoint)
    config = infer_gpt_config(state_dict, checkpoint=checkpoint)
    shapes = checkpoint_parameter_shapes(state_dict)

    assert config.vocab_size == 32
    assert config.d_model == 16
    assert config.max_seq_len == 24
    assert config.layers == 2
    assert config.n_heads == 8
    assert shapes["word_embeddings.weight"] == (32, 16)
    assert shapes["fc_out.weight"] == (32, 16)
    assert infer_checkpoint_vocab_size(checkpoint_path) == 32
    assert tokenizer_checkpoint_compatible(32, 32)
    assert not tokenizer_checkpoint_compatible(32, 31)

    loaded_model, loaded_config, fmt = load_model_from_checkpoint(checkpoint_path, device="cpu")
    loaded_model.load_state_dict(state_dict)
    assert loaded_config == config
    assert fmt == "raw_state_dict"


def test_checkpoint_tokenizer_mismatch_fails_in_local_gpt(tmp_path, tiny_tokenizer_dir):
    import pytest
    import torch

    from src.gpt import GPTModel
    from src.local_gpt import LocalGPT, LocalGPTConfig

    checkpoint_path = tmp_path / "mismatch.pt"
    torch.save(GPTModel(d_model=16, n_heads=8, layers=1, vocab_size=7, max_seq_len=16).state_dict(), checkpoint_path)

    config = LocalGPTConfig(
        checkpoint_path=str(checkpoint_path),
        tokenizer_dir=str(tiny_tokenizer_dir),
        device="cpu",
        max_new_tokens=1,
    )
    with pytest.raises(ValueError, match="Tokenizer/checkpoint mismatch"):
        LocalGPT(config)
