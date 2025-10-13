"""
Text processing utilities for content manipulation and formatting operations.

Provides comprehensive text processing capabilities including cleaning, filtering,
splitting, loading from various sources, and formatting utilities. Supports loading
text from URLs, compressed files, package resources, and local files with automatic
encoding detection. Includes specialized formatting for display values and text
wrapping operations for consistent presentation across the system.
"""

from __future__ import annotations

import gzip
import re
import textwrap
from pathlib import Path
from typing import Any

import ftfy
import httpx
from loguru import logger

from guidellm.settings import settings
from guidellm.utils.console import Colors

__all__ = [
    "MAX_PATH_LENGTH",
    "EndlessTextCreator",
    "camelize_str",
    "clean_text",
    "filter_text",
    "format_value_display",
    "is_punctuation",
    "load_text",
    "split_text",
    "split_text_list_by_length",
]

MAX_PATH_LENGTH: int = 4096


def format_value_display(
    value: float,
    label: str,
    units: str = "",
    total_characters: int | None = None,
    digits_places: int | None = None,
    decimal_places: int | None = None,
) -> str:
    """
    Format a numeric value with units and label for consistent display output.

    Creates standardized display strings for metrics and measurements with
    configurable precision, width, and color formatting. Supports both
    fixed-width and variable-width output for tabular displays.

    :param value: Numeric value to format and display
    :param label: Descriptive label for the value
    :param units: Units string to append after the value
    :param total_characters: Total width for right-aligned output formatting
    :param digits_places: Total number of digits for numeric formatting
    :param decimal_places: Number of decimal places for numeric precision
    :return: Formatted string with value, units, and colored label
    """
    if decimal_places is None and digits_places is None:
        formatted_number = f"{value:.0f}"
    elif digits_places is None:
        formatted_number = f"{value:.{decimal_places}f}"
    elif decimal_places is None:
        formatted_number = f"{value:>{digits_places}f}"
    else:
        formatted_number = f"{value:>{digits_places}.{decimal_places}f}"

    result = f"{formatted_number}{units} [{Colors.info}]{label}[/{Colors.info}]"

    if total_characters is not None:
        total_characters += len(Colors.info) * 2 + 5

        if len(result) < total_characters:
            result = result.rjust(total_characters)

    return result


def split_text_list_by_length(
    text_list: list[Any],
    max_characters: int | list[int],
    pad_horizontal: bool = True,
    pad_vertical: bool = True,
) -> list[list[str]]:
    """
    Split text strings into wrapped lines with specified maximum character limits.

    Processes each string in the input list by wrapping text to fit within character
    limits, with optional padding for consistent formatting in tabular displays.
    Supports different character limits per string and uniform padding across results.

    :param text_list: List of strings to process and wrap
    :param max_characters: Maximum characters per line, either single value or
        per-string limits
    :param pad_horizontal: Right-align lines within their character limits
    :param pad_vertical: Pad shorter results to match the longest wrapped result
    :return: List of wrapped line lists, one per input string
    :raises ValueError: If max_characters list length doesn't match text_list length
    """
    if not isinstance(max_characters, list):
        max_characters = [max_characters] * len(text_list)

    if len(max_characters) != len(text_list):
        raise ValueError(
            f"max_characters must be a list of the same length as text_list, "
            f"but got {len(max_characters)} and {len(text_list)}"
        )

    result: list[list[str]] = []
    for index, text in enumerate(text_list):
        lines = textwrap.wrap(text, max_characters[index])
        result.append(lines)

    if pad_vertical:
        max_lines = max(len(lines) for lines in result)
        for lines in result:
            while len(lines) < max_lines:
                lines.append(" ")

    if pad_horizontal:
        for index in range(len(result)):
            lines = result[index]
            max_chars = max_characters[index]
            new_lines = []
            for line in lines:
                new_lines.append(line.rjust(max_chars))
            result[index] = new_lines

    return result


def filter_text(
    text: str,
    filter_start: str | int | None = None,
    filter_end: str | int | None = None,
) -> str:
    """
    Extract text substring using start and end markers or indices.

    Filters text content by locating string markers or using numeric indices
    to extract specific portions. Supports flexible filtering for content
    extraction and preprocessing operations.

    :param text: Source text to filter and extract from
    :param filter_start: Starting marker string or index position
    :param filter_end: Ending marker string or index position
    :return: Filtered text substring between specified boundaries
    :raises ValueError: If filter indices are invalid or markers not found
    """
    filter_start_index = -1
    filter_end_index = -1

    if filter_start and isinstance(filter_start, str):
        filter_start_index = text.index(filter_start)
    elif filter_start:
        if not isinstance(filter_start, int):
            raise ValueError(f"Invalid filter start index: {filter_start}")
        filter_start_index = filter_start

    if filter_end and isinstance(filter_end, str):
        filter_end_index = text.index(filter_end)
    elif filter_end:
        if not isinstance(filter_end, int):
            raise ValueError(f"Invalid filter end index: {filter_end}")
        filter_end_index = filter_end

    if filter_start_index > -1:
        text = text[filter_start_index:]
    if filter_end_index > -1:
        text = text[:filter_end_index]

    return text


def clean_text(text: str) -> str:
    """
    Normalize text by fixing encoding issues and standardizing whitespace.

    Applies Unicode normalization and whitespace standardization for consistent
    text processing. Removes excessive whitespace and fixes common encoding problems.

    :param text: Raw text string to clean and normalize
    :return: Cleaned text with normalized encoding and whitespace
    """
    return re.sub(r"\s+", " ", ftfy.fix_text(text)).strip()


def split_text(text: str, split_punctuation: bool = False) -> list[str]:
    """
    Split text into tokens with optional punctuation separation.

    Tokenizes text into words and optionally separates punctuation marks
    for detailed text analysis and processing operations.

    :param text: Text string to tokenize and split
    :param split_punctuation: Separate punctuation marks as individual tokens
    :return: List of text tokens
    """
    text = clean_text(text)

    if split_punctuation:
        return re.findall(r"[\w]+|[.,!?;]", text)

    return text.split()


def load_text(data: str | Path, encoding: str | None = None) -> str:
    """
    Load text content from various sources including URLs, files, and package data.

    Supports loading from HTTP/FTP URLs, local files, compressed archives, package
    resources, and raw text strings. Automatically detects source type and applies
    appropriate loading strategy with encoding support.

    :param data: Source location or raw text - URL, file path, package resource
        identifier, or text content
    :param encoding: Character encoding for file reading operations
    :return: Loaded text content as string
    :raises FileNotFoundError: If local file path does not exist
    :raises httpx.HTTPStatusError: If URL request fails
    """
    logger.debug("Loading text: {}", data)

    if not data:
        return ""

    # check URLs
    if isinstance(data, str) and data.strip().startswith(("http", "ftp")):
        with httpx.Client(timeout=settings.request_timeout) as client:
            response = client.get(data.strip())
            response.raise_for_status()
            return response.text

    # check gzipped files
    if isinstance(data, str) and data.endswith(".gz"):
        with gzip.open(data, "rt", encoding=encoding) as file:
            return file.read()

    # check if it's raw text by not being a path
    if isinstance(data, str) and (
        len(data) > MAX_PATH_LENGTH or not Path(data).exists()
    ):
        return data

    # assume local file
    if not isinstance(data, Path):
        data = Path(data)

    if not data.exists() or not data.is_file():
        raise FileNotFoundError(f"File not found: {data}")

    return data.read_text(encoding=encoding)


def is_punctuation(text: str) -> bool:
    """
    Check if a single character is a punctuation mark.

    Identifies punctuation characters by excluding alphanumeric characters
    and whitespace from single-character strings.

    :param text: Single character string to test
    :return: True if the character is punctuation, False otherwise
    """
    return len(text) == 1 and not text.isalnum() and not text.isspace()


def camelize_str(snake_case_string: str) -> str:
    return (words := snake_case_string.split("_"))[0].lower() + "".join(
        word.capitalize() for word in words[1:]
    )


class EndlessTextCreator:
    """
    Infinite text generator for load testing and content creation operations.

    Provides deterministic text generation by cycling through preprocessed word
    tokens from source content. Supports filtering and punctuation handling for
    realistic text patterns in benchmarking scenarios.

    Example:
    ::
        creator = EndlessTextCreator("path/to/source.txt")
        generated = creator.create_text(start=0, length=100)
        more_text = creator.create_text(start=50, length=200)
    """

    def __init__(
        self,
        data: str | Path,
        filter_start: str | int | None = None,
        filter_end: str | int | None = None,
    ):
        """
        Initialize text creator with source content and optional filtering.

        :param data: Source text location or content - file path, URL, or raw text
        :param filter_start: Starting marker or index for content filtering
        :param filter_end: Ending marker or index for content filtering
        """
        self.data = data
        self.text = load_text(data)
        self.filtered_text = filter_text(self.text, filter_start, filter_end)
        self.words = split_text(self.filtered_text, split_punctuation=True)

    def create_text(self, start: int, length: int) -> str:
        """
        Generate text by cycling through word tokens from the specified position.

        Creates deterministic text sequences by selecting consecutive tokens from
        the preprocessed word list, wrapping around when reaching the end.
        Maintains proper spacing and punctuation formatting.

        :param start: Starting position in the token sequence
        :param length: Number of tokens to include in generated text
        :return: Generated text string with proper spacing and punctuation
        """
        text = ""

        for counter in range(length):
            index = (start + counter) % len(self.words)
            add_word = self.words[index]

            if counter != 0 and not is_punctuation(add_word):
                text += " "

            text += add_word

        return text
