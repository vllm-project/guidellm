from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase  # type: ignore[import]

__all__ = [
    "check_load_processor",
]


def check_load_processor(
    processor: str | Path | PreTrainedTokenizerBase | None,
    processor_args: dict[str, Any] | None,
    error_msg: str,
) -> PreTrainedTokenizerBase:
    if processor is None:
        raise ValueError(f"Processor/Tokenizer is required for {error_msg}.")

    try:
        if isinstance(processor, str | Path):
            loaded = AutoTokenizer.from_pretrained(
                processor,
                **(processor_args or {}),
            )
        else:
            loaded = processor
    except Exception as err:
        raise ValueError(
            f"Failed to load processor/Tokenizer for {error_msg}."
        ) from err

    if not isinstance(loaded, PreTrainedTokenizerBase):
        raise ValueError(f"Invalid processor/Tokenizer for {error_msg}.")

    return loaded
