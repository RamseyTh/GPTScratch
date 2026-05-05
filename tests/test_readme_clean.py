from __future__ import annotations

from pathlib import Path


def test_readme_is_clean_and_grading_focused():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in (
        "python main.py --quick-run",
        "model/model_weights.pt",
        "model/hftokenizer",
        "outputs/runs/{run_id}",
        "outputs/reports/{run_id}",
        "Academic Honesty",
    ):
        assert expected in readme

    for forbidden in (
        "GPT-5",
        "GPT-4o",
        "OpenAI API key is required",
        "pad-and-mask",
        "allow-vocab-mismatch",
        "Audit existing pilot outputs",
    ):
        assert forbidden not in readme
