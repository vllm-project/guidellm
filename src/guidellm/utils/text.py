import gzip
import re
import textwrap
from importlib.resources import as_file, files  # type: ignore[attr-defined]
from pathlib import Path
from typing import Any, Optional, Union

import ftfy
import httpx
from loguru import logger

from guidellm import data as package_data
from guidellm.config import settings

__all__ = [
    "EndlessTextCreator",
    "camelize_str",
    "clean_text",
    "filter_text",
    "is_puncutation",
    "load_text",
    "split_text",
    "split_text_list_by_length",
]

MAX_PATH_LENGTH = 4096


def split_text_list_by_length(
    text_list: list[Any],
    max_characters: Union[int, list[int]],
    pad_horizontal: bool = True,
    pad_vertical: bool = True,
) -> list[list[str]]:
    """
    Split a list of strings into a list of strings,
    each with a maximum length of max_characters

    :param text_list: the list of strings to split
    :param max_characters: the maximum length of each string
    :param pad_horizontal: whether to pad the strings horizontally, defaults to True
    :param pad_vertical: whether to pad the strings vertically, defaults to True
    :return: a list of strings
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
    filter_start: Optional[Union[str, int]] = None,
    filter_end: Optional[Union[str, int]] = None,
) -> str:
    """
    Filter text by start and end strings or indices

    :param text: the text to filter
    :param filter_start: the start string or index to filter from
    :param filter_end: the end string or index to filter to
    :return: the filtered text
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
    return re.sub(r"\s+", " ", ftfy.fix_text(text)).strip()


def split_text(text: str, split_punctuation: bool = False) -> list[str]:
    text = clean_text(text)

    if split_punctuation:
        return re.findall(r"[\w]+|[.,!?;]", text)

    return text.split()


def load_text(data: Union[str, Path], encoding: Optional[str] = None) -> str:
    """
    Load an HTML file from a path or URL

    :param data: the path or URL to load the HTML file from
    :type data: Union[str, Path]
    :param encoding: the encoding to use when reading the file
    :type encoding: str
    :return: the HTML content
    :rtype: str
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

    # check package data
    if isinstance(data, str) and data.startswith("data:"):
        resource_path = files(package_data).joinpath(data[5:])
        with (
            as_file(resource_path) as resource_file,
            gzip.open(resource_file, "rt", encoding=encoding) as file,
        ):
            return file.read()

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


def is_puncutation(text: str) -> bool:
    """
    Check if the text is a punctuation

    :param text: the text to check
    :type text: str
    :return: True if the text is a punctuation, False otherwise
    :rtype: bool
    """
    return len(text) == 1 and not text.isalnum() and not text.isspace()


def camelize_str(snake_case_string: str) -> str:
    return (words := snake_case_string.split("_"))[0].lower() + "".join(
        word.capitalize() for word in words[1:]
    )


class EndlessTextCreator:
    def __init__(
        self,
        data: Union[str, Path],
        filter_start: Optional[Union[str, int]] = None,
        filter_end: Optional[Union[str, int]] = None,
    ):
        self.data = data
        self.text = load_text(data)
        self.filtered_text = filter_text(self.text, filter_start, filter_end)
        self.words = split_text(self.filtered_text, split_punctuation=True)

    def create_text(self, start: int, length: int) -> str:
        text = ""

        for counter in range(length):
            index = (start + counter) % len(self.words)
            add_word = self.words[index]

            if counter != 0 and not is_puncutation(add_word):
                text += " "

            text += add_word

        return text
