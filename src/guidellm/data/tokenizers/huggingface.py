from __future__ import annotations

from typing import Any, Literal

from pydantic import Field
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from guidellm.data.schemas import DataTokenizerArgs
from guidellm.data.tokenizers.tokenizer import DataTokenizer, TokenizerRegistry

__all__ = ["HuggingFaceTokenizer", "HuggingFaceTokenizerArgs"]


@DataTokenizerArgs.register(["huggingface_auto", "hf_auto"])
class HuggingFaceTokenizerArgs(DataTokenizerArgs):
    kind: Literal["huggingface_auto", "hf_auto"] = "huggingface_auto"
    load_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional additional arguments to pass to the HuggingFace tokenizer's "
            "from_pretrained method, such as 'use_fast' or 'revision'."
        ),
    )


@TokenizerRegistry.register(["huggingface_auto", "hf_auto"])
class HuggingFaceTokenizer(DataTokenizer):
    def __init__(
        self,
        config: HuggingFaceTokenizerArgs,
    ) -> None:
        if config.model is None:
            raise ValueError("The 'name' field must be provided")

        self._config = config
        self._tokenizer: None | PreTrainedTokenizerBase = None

    def __call__(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is not None:
            return self._tokenizer
        else:
            from_pretrained = AutoTokenizer.from_pretrained(
                self._config.model,
                **self._config.load_kwargs,
            )
            self._tokenizer = from_pretrained
            return from_pretrained
