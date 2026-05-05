from __future__ import annotations

import ast
from pathlib import Path


def test_key_modules_have_docstrings():
    for path in (
        "src/pipeline.py",
        "src/local_gpt.py",
        "src/retrieval.py",
        "src/evaluation.py",
        "src/reporting.py",
        "src/hftokenizer.py",
        "src/checkpoint.py",
    ):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        assert ast.get_docstring(tree), f"{path} is missing a module docstring"
