from __future__ import annotations

import sys


def test_main_pipeline_does_not_import_openai(monkeypatch):
    sys.modules.pop("openai", None)

    import main  # noqa: F401
    import src.pipeline  # noqa: F401

    assert "openai" not in sys.modules
