"""Offline HuggingFace tokenizer hub-id resolution for E2E.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from guidellm.data.tokenizers.huggingface import (
    HuggingFaceTokenizer,
    HuggingFaceTokenizerArgs,
)
from tests.fixtures.tokenizers import MINIMAL_TOKENIZER_DIR, seed_hub_cache_for_model

# Unique id so a developer machine's real ``gpt2`` hub cache cannot shadow the fixture.
_HUB_MODEL_ID = "guidellm-test-tokenizer"


@pytest.mark.sanity
@pytest.mark.timeout(30)
def test_huggingface_tokenizer_loads_hub_id_from_seeded_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    GuideLLM resolves a hub model id from a seeded local HF cache.

    Does not contact HuggingFace Hub: the minimal fixture files are copied into
    a hub-style cache under a temporary ``HF_HOME`` with offline env vars set.

    ## WRITTEN BY AI ##
    """
    assert MINIMAL_TOKENIZER_DIR.is_dir(), f"Missing fixture: {MINIMAL_TOKENIZER_DIR}"
    seed_hub_cache_for_model(tmp_path, model_id=_HUB_MODEL_ID)

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    # Drop any cached env that might point at a real Hub session.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    # huggingface_hub reads cache roots at import time; force the seeded tmp cache.
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr("huggingface_hub.constants.HF_HOME", str(tmp_path))

    tokenizer = HuggingFaceTokenizer(
        HuggingFaceTokenizerArgs(
            model=_HUB_MODEL_ID,
            load_kwargs={"local_files_only": True},
        )
    )
    loaded = tokenizer()
    assert loaded is not None
    assert loaded.vocab_size < 5000
    assert loaded.encode("hello") == loaded.encode("hello")
    # Second call must reuse the cached instance.
    assert tokenizer() is loaded
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


@pytest.mark.smoke
@pytest.mark.timeout(30)
def test_huggingface_tokenizer_loads_from_vendored_path() -> None:
    """
    GuideLLM loads the vendored tokenizer directory without Hub access.

    ## WRITTEN BY AI ##
    """
    tokenizer = HuggingFaceTokenizer(
        HuggingFaceTokenizerArgs(
            model=str(MINIMAL_TOKENIZER_DIR),
            load_kwargs={"local_files_only": True},
        )
    )
    loaded = tokenizer()
    assert len(loaded.encode("hello world")) > 0
    assert loaded.vocab_size < 5000
