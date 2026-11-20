"""Count output tokens from response text when the server reports no usage.

Without a count, a response's output token metrics fall back to the request's
target output length, which overstates throughput when generation stops early.
Counting runs in the main process so that workers never import a tokenizer.
"""

from __future__ import annotations

import asyncio

from guidellm.data.tokenizers import DataTokenizer, TokenizerRegistry
from guidellm.logger import logger
from guidellm.schemas import GenerationResponse
from guidellm.schemas.data import DataTokenizerArgs

__all__ = ["OutputTokenCounter"]


class OutputTokenCounter:
    """Fill in missing output token counts with the benchmark's tokenizer."""

    def __init__(
        self,
        config: DataTokenizerArgs,
        tokenizer: DataTokenizer | None,
        load_error: Exception | None = None,
    ):
        """
        :param config: The resolved tokenizer configuration for the benchmark
        :param tokenizer: The loaded tokenizer, or None to leave counts unset
        :param load_error: Why the tokenizer could not be loaded, if it failed
        """
        self._config = config
        self._tokenizer = tokenizer
        self._load_error = load_error
        self._warned = False

    @classmethod
    async def load(cls, config: DataTokenizerArgs) -> OutputTokenCounter:
        """
        Load the tokenizer before the benchmark starts so loading never delays a
        response.

        :param config: The resolved tokenizer configuration for the benchmark
        :return: A counter, which leaves counts unset and warns once if the
            tokenizer failed to load
        """
        try:
            tokenizer = TokenizerRegistry.create(config)
            await asyncio.to_thread(tokenizer)
        except Exception as err:  # noqa: BLE001
            return cls(config, None, err)

        return cls(config, tokenizer)

    async def __call__(self, response: GenerationResponse | None) -> None:
        """
        Count the response's output tokens if the server did not report them.

        :param response: The response, updated in place
        """
        if response is None or response.output_metrics.text_tokens is not None:
            return

        text = (response.reasoning_text or "") + (response.text or "")
        if not text:
            return

        if self._tokenizer is None:
            if not self._warned:
                self._warned = True
                logger.warning(
                    "The server returned no token usage and tokenizer '{}' could "
                    "not be loaded ({}); output token counts fall back to the "
                    "requested output length.",
                    self._config.model,
                    self._load_error,
                )
            return

        encoded = await asyncio.to_thread(
            self._tokenizer().encode, text, add_special_tokens=False
        )
        response.output_metrics.text_tokens = len(encoded)
        if not self._warned:
            self._warned = True
            logger.warning(
                "The server returned no token usage; counting output tokens from "
                "the response text with tokenizer '{}'.",
                self._config.model,
            )
