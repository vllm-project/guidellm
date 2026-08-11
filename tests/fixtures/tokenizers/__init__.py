"""Shared tokenizer fixtures for offline tests."""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = [
    "MINIMAL_TOKENIZER_DIR",
    "seed_hub_cache_for_model",
]

# Small BPE tokenizer trained on Faker-style synthetic text (no Hub download).
# Vocab size 1024; encode/decode matches GuideLLM synthetic_text prompt sizing
# (Faker text -> encode -> truncate -> decode). Regenerated with the `tokenizers`
# + `transformers` libraries from a Faker corpus when fixtures need updating.
MINIMAL_TOKENIZER_DIR = Path(__file__).resolve().parent / "minimal"


def seed_hub_cache_for_model(
    hf_home: Path,
    model_id: str = "guidellm-test-tokenizer",
    source: Path | None = None,
    revision: str = "a" * 40,
) -> Path:
    """
    Populate a Hugging Face hub cache layout so ``from_pretrained(model_id)``
    resolves offline from ``HF_HOME``.

    Uses a hex revision id so ``huggingface_hub`` accepts the snapshot layout.
    The vendored fixture includes a minimal ``config.json`` (``model_type=gpt2``)
    so ``AutoTokenizer`` can resolve a tokenizer class without Hub access.

    :param hf_home: Directory to use as ``HF_HOME``
    :param model_id: Hub model id used for the cache directory name
    :param source: Local tokenizer directory to copy; defaults to the minimal fixture
    :param revision: Snapshot / refs revision name (40-char hex preferred)
    :return: Path to the seeded snapshot directory
    """
    source_dir = source or MINIMAL_TOKENIZER_DIR
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer fixture missing: {source_dir}")

    repo_cache = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshot_dir = repo_cache / "snapshots" / revision
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, snapshot_dir / path.name)

    refs_dir = repo_cache / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(revision, encoding="utf-8")
    return snapshot_dir
