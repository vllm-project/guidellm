"""Checks for the custom backend sample in the backends guide."""

from __future__ import annotations

from pathlib import Path

import pytest

_GUIDE = Path(__file__).resolve().parents[3] / "docs" / "en" / "guides" / "backends.md"


def _custom_backend_section() -> str:
    """Return the English custom-backend section, excluding the pages around it."""
    text = _GUIDE.read_text(encoding="utf-8")
    start = text.index("## Custom Backends and Plugins")
    end = text.index("## Expanding Backend Support")
    return text[start:end]


@pytest.mark.regression
def test_custom_backend_sample_can_target_a_local_server() -> None:
    """
    The plugin sample must not send ``default-model`` to the tokenizer or server.

    An empty model is resolved from ``GET /v1/models``. The sample command names
    the served or Hugging Face id and passes ``--tokenizer`` when that id is not
    a local path or Hub repo.

    ## WRITTEN BY AI ##
    """
    section = _custom_backend_section()

    assert 'default="default-model"' not in section
    assert "/v1/models" in section
    assert "model=<served-or-hf-id>" in section
    assert "--tokenizer" in section
    assert "kind=huggingface_auto,model=<local-path-or-hf-repo>" in section
