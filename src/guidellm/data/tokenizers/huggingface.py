from __future__ import annotations

from pathlib import Path

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from guidellm.data.tokenizers.tokenizer import DataTokenizer, TokenizerRegistry
from guidellm.schemas.data.tokenizers import HuggingFaceTokenizerArgs

__all__ = ["HuggingFaceTokenizer"]


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
            self._check_name(self._config.model)
            from_pretrained = AutoTokenizer.from_pretrained(
                self._config.model,
                **self._config.load_kwargs,
            )
            self._tokenizer = from_pretrained
            return from_pretrained

    @staticmethod
    def _check_name(name: str | None) -> None:
        """
        Reject a name that can be neither a local path nor a Hugging Face repo id.

        The tokenizer defaults to the server's model name, which for servers such
        as Ollama (``qwen3:4b``) is not a Hugging Face id. Without this check the
        user gets two chained Hugging Face tracebacks and no hint of the fix.

        :param name: The configured tokenizer name or path
        :raises ValueError: If ``name`` is not an existing path and not a valid
            Hugging Face repo id
        """
        if name is None or Path(name).exists():
            return
        try:
            validate_repo_id(name)
        except HFValidationError as err:
            raise ValueError(
                f"Cannot load a tokenizer for {name!r}: it is neither an existing "
                "local path nor a valid Hugging Face repo id. The tokenizer defaults "
                "to the model name reported by the server, which for servers such as "
                "Ollama is not a Hugging Face id. Pass one explicitly, for example "
                "--tokenizer kind=huggingface_auto,model=Qwen/Qwen3-4B "
                f"(Hugging Face said: {err})"
            ) from None
