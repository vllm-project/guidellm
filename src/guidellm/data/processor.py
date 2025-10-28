from __future__ import annotations

from typing import Any

from transformers import (  # type: ignore[import]
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

__all__ = ["ProcessorFactory"]


class ProcessorFactory:
    def __init__(
        self,
        processor: str | PreTrainedTokenizerBase,
        processor_args: dict[str, Any] | None = None,
    ) -> None:
        self.processor = processor
        self.processor_args = processor_args or {}

    def __call__(self) -> PreTrainedTokenizerBase:
        if isinstance(self.processor, PreTrainedTokenizerBase):
            return self.processor
        else:
            from_pretrained = AutoTokenizer.from_pretrained(
                self.processor,
                **(self.processor_args or {}),
            )
            self.processor = from_pretrained
            return from_pretrained
