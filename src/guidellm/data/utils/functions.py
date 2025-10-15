from typing import Literal

__all__ = ["text_stats"]


def text_stats(
    text: str,
) -> dict[Literal["type", "text", "num_chars", "num_words"], str | int]:
    """Compute basic text statistics."""
    num_chars = len(text)
    num_words = len(text.split())

    return {
        "type": "text",
        "text": text,
        "num_chars": num_chars,
        "num_words": num_words,
    }
