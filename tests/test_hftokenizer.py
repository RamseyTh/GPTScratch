from __future__ import annotations

import sys


def test_hf_tokenizer_loads_encodes_and_decodes(tiny_tokenizer_dir):
    from src.hftokenizer import HFTokenizer

    tokenizer = HFTokenizer(tiny_tokenizer_dir)
    ids = tokenizer.encode("Apple reported Services net sales in 2025.")
    decoded = tokenizer.decode(ids)

    assert ids
    assert all(isinstance(token_id, int) for token_id in ids)
    assert isinstance(decoded, str)
    assert tokenizer.vocab_size > 0
    assert tokenizer.eos_token_id is not None
    assert tokenizer.metadata["tokenizer_dir"] == str(tiny_tokenizer_dir)


def test_inspect_tokenizer_script_reports_compatibility(tmp_path, tiny_tokenizer_dir, monkeypatch, capsys):
    import torch

    from src.gpt import GPTModel
    from src.hftokenizer import HFTokenizer
    from scripts.inspect_tokenizer import main as inspect_main

    tokenizer = HFTokenizer(tiny_tokenizer_dir)
    checkpoint_path = tmp_path / "tiny.pt"
    torch.save(
        GPTModel(d_model=16, n_heads=8, layers=1, vocab_size=tokenizer.vocab_size, max_seq_len=16).state_dict(),
        checkpoint_path,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/inspect_tokenizer.py",
            "--tokenizer-dir",
            str(tiny_tokenizer_dir),
            "--checkpoint",
            str(checkpoint_path),
        ],
    )
    inspect_main()
    output = capsys.readouterr().out

    assert "tokenizer class:" in output
    assert "vocab size:" in output
    assert "compatibility: PASS" in output
    assert "encoded IDs:" in output
    assert "decoded text:" in output
