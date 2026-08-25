from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from guidellm.schemas.data.entrypoints import DataTokenizerArgs


@DataTokenizerArgs.register(["huggingface_auto", "hf_auto"])
class HuggingFaceTokenizerArgs(DataTokenizerArgs):
    """Model for Hugging Face tokenizer arguments."""

    kind: Literal["huggingface_auto", "hf_auto"] = Field(
        default="huggingface_auto",
        description="Type identifier for the HuggingFace tokenizer.",
    )
    load_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        examples=[{"use_fast": True, "revision": "main"}],
        description=(
            "Optional additional arguments to pass to the HuggingFace tokenizer's "
            "from_pretrained method, such as 'use_fast' or 'revision'."
        ),
    )
