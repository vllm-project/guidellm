"""Shared tokenizer fixtures for offline tests."""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = [
    "GPT2_TOKENIZER_DIR",
    "seed_hub_cache_for_model",
]

# Vendored GPT-2 tokenizer files (no Hub download required).
GPT2_TOKENIZER_DIR = Path(__file__).resolve().parent / "gpt2"


def seed_hub_cache_for_model(
    hf_home: Path,
    model_id: str = "gpt2",
    source: Path | None = None,
    revision: str = "local-offline",
) -> Path:
    """
    Populate a Hugging Face hub cache layout so ``from_pretrained(model_id)``
    resolves offline from ``HF_HOME``.

    :param hf_home: Directory to use as ``HF_HOME``
    :param model_id: Hub model id (e.g. ``gpt2``)
    :param source: Local tokenizer directory to copy; defaults to vendored gpt2
    :param revision: Snapshot / refs revision name
    :return: Path to the seeded snapshot directory
    """
    source_dir = source or GPT2_TOKENIZER_DIR
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
