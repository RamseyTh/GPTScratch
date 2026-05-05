from __future__ import annotations

from pathlib import Path


def test_readme_describes_verified_research_pipeline():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "python main.py \\" in readme
    assert "--quick-run" in readme
    assert "questions_verified" in readme
    assert "model/hftokenizer" in readme
    assert "scripts/install_verified_questions.py" in readme
    assert "scripts/validate_verified_questions.py" in readme
    assert "Any run using `--limit` is a smoke test" in readme
    assert "Academic Honesty" in readme
    assert "Some scaffolding, debugging prompts, and documentation drafts were generated with OpenAI Codex/ChatGPT" in readme
    assert "GPT-5" not in readme
    assert "OpenAI API key is required" not in readme
    assert "Audit existing pilot outputs" not in readme
