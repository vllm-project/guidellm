from typing import Literal

__all__ = ["text_stats"]


def text_stats(
    text: str,
) -> dict[Literal["num_chars", "num_words"], int]:
    """Compute basic text statistics."""
    num_chars = len(text)
    num_words = len(text.split())

    return {
        "num_chars": num_chars,
        "num_words": num_words,
    }
