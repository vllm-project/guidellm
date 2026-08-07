from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from guidellm.data.tokenizers.tokenizer import DataTokenizer, TokenizerRegistry
from guidellm.schemas.data.tokenizers import HuggingFaceTokenizerArgs

__all__ = ["HuggingFaceTokenizer", "HuggingFaceTokenizerArgs"]


@TokenizerRegistry.register(["huggingface_auto", "hf_auto"])
class HuggingFaceTokenizer(DataTokenizer):
    """Tokenizer for Hugging Face models."""

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
