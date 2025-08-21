from __future__ import annotations

import gzip
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest

from guidellm.utils.text import (
    MAX_PATH_LENGTH,
    EndlessTextCreator,
    clean_text,
    filter_text,
    format_value_display,
    is_puncutation,
    load_text,
    split_text,
    split_text_list_by_length,
)


def test_max_path_length():
    """Test that MAX_PATH_LENGTH is correctly defined."""
    assert isinstance(MAX_PATH_LENGTH, int)
    assert MAX_PATH_LENGTH == 4096


class TestFormatValueDisplay:
    """Test suite for format_value_display."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        (
            "value",
            "label",
            "units",
            "total_characters",
            "digits_places",
            "decimal_places",
            "expected",
        ),
        [
            (42.0, "test", "", None, None, None, "42 [info]test[/info]"),
            (42.5, "test", "ms", None, None, 1, "42.5ms [info]test[/info]"),
            (42.123, "test", "", None, 5, 2, " 42.12 [info]test[/info]"),
            (
                42.0,
                "test",
                "ms",
                30,
                None,
                0,
                "                    42ms [info]test[/info]",
            ),
        ],
    )
    def test_invocation(
        self,
        value,
        label,
        units,
        total_characters,
        digits_places,
        decimal_places,
        expected,
    ):
        """Test format_value_display with various parameters."""
        result = format_value_display(
            value=value,
            label=label,
            units=units,
            total_characters=total_characters,
            digits_places=digits_places,
            decimal_places=decimal_places,
        )
        assert label in result
        assert units in result
        value_check = (
            str(int(value))
            if decimal_places == 0
            else (
                f"{value:.{decimal_places}f}"
                if decimal_places is not None
                else str(value)
            )
        )
        assert value_check in result or str(value) in result

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("value", "label"),
        [
            (None, "test"),
            (42.0, None),
            ("not_number", "test"),
        ],
    )
    def test_invocation_with_none_values(self, value, label):
        """Test format_value_display with None/invalid inputs still works."""
        result = format_value_display(value, label)
        assert isinstance(result, str)
        if label is not None:
            assert str(label) in result
        if value is not None:
            assert str(value) in result


class TestSplitTextListByLength:
    """Test suite for split_text_list_by_length."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        (
            "text_list",
            "max_characters",
            "pad_horizontal",
            "pad_vertical",
            "expected_structure",
        ),
        [
            (
                ["hello world", "test"],
                5,
                False,
                False,
                [["hello", "world"], ["test"]],
            ),
            (
                ["short", "longer text"],
                [5, 10],
                True,
                True,
                [[" short"], ["longer", "text"]],
            ),
            (
                ["a", "b", "c"],
                10,
                True,
                True,
                [["         a"], ["         b"], ["         c"]],
            ),
        ],
    )
    def test_invocation(
        self,
        text_list,
        max_characters,
        pad_horizontal,
        pad_vertical,
        expected_structure,
    ):
        """Test split_text_list_by_length with various parameters."""
        result = split_text_list_by_length(
            text_list, max_characters, pad_horizontal, pad_vertical
        )
        assert len(result) == len(text_list)
        if pad_vertical:
            max_lines = max(len(lines) for lines in result)
            assert all(len(lines) == max_lines for lines in result)

    @pytest.mark.sanity
    def test_invalid_max_characters_length(self):
        """Test split_text_list_by_length with mismatched max_characters length."""
        error_msg = "max_characters must be a list of the same length"
        with pytest.raises(ValueError, match=error_msg):
            split_text_list_by_length(["a", "b"], [5, 10, 15])

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("text_list", "max_characters"),
        [
            (None, 5),
            (["test"], None),
            (["test"], []),
        ],
    )
    def test_invalid_invocation(self, text_list, max_characters):
        """Test split_text_list_by_length with invalid inputs."""
        with pytest.raises((TypeError, ValueError)):
            split_text_list_by_length(text_list, max_characters)


class TestFilterText:
    """Test suite for filter_text."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("text", "filter_start", "filter_end", "expected"),
        [
            ("hello world test", "world", None, "world test"),
            ("hello world test", None, "world", "hello "),
            ("hello world test", "hello", "test", "hello world "),
            ("hello world test", 6, 11, "world test"),
            ("hello world test", 0, 5, "hello"),
            ("hello world test", None, None, "hello world test"),
        ],
    )
    def test_invocation(self, text, filter_start, filter_end, expected):
        """Test filter_text with various start and end markers."""
        result = filter_text(text, filter_start, filter_end)
        assert result == expected

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("text", "filter_start", "filter_end"),
        [
            ("hello", "notfound", None),
            ("hello", None, "notfound"),
            ("hello", "invalid_type", None),
            ("hello", None, "invalid_type"),
        ],
    )
    def test_invalid_invocation(self, text, filter_start, filter_end):
        """Test filter_text with invalid markers."""
        with pytest.raises((ValueError, TypeError)):
            filter_text(text, filter_start, filter_end)


class TestCleanText:
    """Test suite for clean_text."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("hello    world", "hello world"),
            ("  hello\n\nworld  ", "hello world"),
            ("hello\tworld\r\ntest", "hello world test"),
            ("", ""),
            ("   ", ""),
        ],
    )
    def test_invocation(self, text, expected):
        """Test clean_text with various whitespace scenarios."""
        result = clean_text(text)
        assert result == expected

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "text",
        [
            None,
            123,
        ],
    )
    def test_invalid_invocation(self, text):
        """Test clean_text with invalid inputs."""
        with pytest.raises((TypeError, AttributeError)):
            clean_text(text)


class TestSplitText:
    """Test suite for split_text."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("text", "split_punctuation", "expected"),
        [
            ("hello world", False, ["hello", "world"]),
            ("hello, world!", True, ["hello", ",", "world", "!"]),
            ("test.example", False, ["test.example"]),
            ("test.example", True, ["test", ".", "example"]),
            ("", False, []),
        ],
    )
    def test_invocation(self, text, split_punctuation, expected):
        """Test split_text with various punctuation options."""
        result = split_text(text, split_punctuation)
        assert result == expected

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "text",
        [
            None,
            123,
        ],
    )
    def test_invalid_invocation(self, text):
        """Test split_text with invalid inputs."""
        with pytest.raises((TypeError, AttributeError)):
            split_text(text)


class TestLoadText:
    """Test suite for load_text."""

    @pytest.mark.smoke
    def test_empty_data(self):
        """Test load_text with empty data."""
        result = load_text("")
        assert result == ""

    @pytest.mark.smoke
    def test_raw_text(self):
        """Test load_text with raw text that's not a file."""
        long_text = "a" * (MAX_PATH_LENGTH + 1)
        result = load_text(long_text)
        assert result == long_text

    @pytest.mark.smoke
    def test_local_file(self):
        """Test load_text with local file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            test_content = "test file content"
            tmp.write(test_content)
            tmp.flush()

            result = load_text(tmp.name)
            assert result == test_content

            Path(tmp.name).unlink()

    @pytest.mark.smoke
    def test_gzipped_file(self):
        """Test load_text with gzipped file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as tmp:
            test_content = "test gzipped content"
            with gzip.open(tmp.name, "wt") as gzf:
                gzf.write(test_content)

            result = load_text(tmp.name)
            assert result == test_content

            Path(tmp.name).unlink()

    @pytest.mark.smoke
    @patch("httpx.Client")
    def test_url_loading(self, mock_client):
        """Test load_text with HTTP URL."""
        mock_response = Mock()
        mock_response.text = "url content"
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        result = load_text("http://example.com/test.txt")
        assert result == "url content"

    @pytest.mark.smoke
    @patch("guidellm.utils.text.files")
    @patch("guidellm.utils.text.as_file")
    def test_package_data_loading(self, mock_as_file, mock_files):
        """Test load_text with package data."""
        mock_resource = Mock()
        mock_files.return_value.joinpath.return_value = mock_resource

        mock_file = Mock()
        mock_file.read.return_value = "package data content"
        mock_as_file.return_value.__enter__.return_value = mock_file

        with patch("gzip.open") as mock_gzip:
            mock_gzip.return_value.__enter__.return_value = mock_file
            result = load_text("data:test.txt")
            assert result == "package data content"

    @pytest.mark.sanity
    def test_nonexistent_file(self):
        """Test load_text with nonexistent file returns the path as raw text."""
        result = load_text("/nonexistent/path/file.txt")
        assert result == "/nonexistent/path/file.txt"

    @pytest.mark.sanity
    @patch("httpx.Client")
    def test_url_error(self, mock_client):
        """Test load_text with HTTP error."""
        mock_client.return_value.__enter__.return_value.get.side_effect = (
            httpx.HTTPStatusError("HTTP error", request=None, response=None)
        )

        with pytest.raises(httpx.HTTPStatusError):
            load_text("http://example.com/error.txt")


class TestIsPunctuation:
    """Test suite for is_puncutation."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (".", True),
            (",", True),
            ("!", True),
            ("?", True),
            (";", True),
            ("a", False),
            ("1", False),
            (" ", False),
            ("ab", False),
            ("", False),
        ],
    )
    def test_invocation(self, text, expected):
        """Test is_puncutation with various characters."""
        result = is_puncutation(text)
        assert result == expected

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "text",
        [
            None,
            123,
        ],
    )
    def test_invalid_invocation(self, text):
        """Test is_puncutation with invalid inputs."""
        with pytest.raises((TypeError, AttributeError)):
            is_puncutation(text)


class TestEndlessTextCreator:
    """Test suite for EndlessTextCreator."""

    @pytest.fixture(
        params=[
            {
                "data": "hello world test",
                "filter_start": None,
                "filter_end": None,
            },
            {
                "data": "hello world test",
                "filter_start": "world",
                "filter_end": None,
            },
            {"data": "one two three four", "filter_start": 0, "filter_end": 9},
        ],
        ids=["no_filter", "string_filter", "index_filter"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for EndlessTextCreator."""
        constructor_args = request.param
        instance = EndlessTextCreator(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test EndlessTextCreator signatures and methods."""
        assert hasattr(EndlessTextCreator, "__init__")
        assert hasattr(EndlessTextCreator, "create_text")
        instance = EndlessTextCreator("test")
        assert hasattr(instance, "data")
        assert hasattr(instance, "text")
        assert hasattr(instance, "filtered_text")
        assert hasattr(instance, "words")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test EndlessTextCreator initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, EndlessTextCreator)
        assert instance.data == constructor_args["data"]
        assert isinstance(instance.text, str)
        assert isinstance(instance.filtered_text, str)
        assert isinstance(instance.words, list)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("data", "filter_start", "filter_end"),
        [
            ("test", "notfound", None),
        ],
    )
    def test_invalid_initialization_values(self, data, filter_start, filter_end):
        """Test EndlessTextCreator with invalid initialization values."""
        with pytest.raises((TypeError, ValueError)):
            EndlessTextCreator(data, filter_start, filter_end)

    @pytest.mark.smoke
    def test_initialization_with_none(self):
        """Test EndlessTextCreator handles None data gracefully."""
        instance = EndlessTextCreator(None)
        assert isinstance(instance, EndlessTextCreator)
        assert instance.data is None

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("start", "length", "expected_length"),
        [
            (0, 5, 5),
            (2, 3, 3),
            (0, 0, 0),
        ],
    )
    def test_create_text(self, valid_instances, start, length, expected_length):
        """Test EndlessTextCreator.create_text."""
        instance, constructor_args = valid_instances
        result = instance.create_text(start, length)
        assert isinstance(result, str)
        if length > 0 and instance.words:
            assert len(result) > 0

    @pytest.mark.smoke
    def test_create_text_cycling(self):
        """Test EndlessTextCreator.create_text cycling behavior."""
        instance = EndlessTextCreator("one two three")
        result1 = instance.create_text(0, 3)
        result2 = instance.create_text(3, 3)
        assert isinstance(result1, str)
        assert isinstance(result2, str)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("start", "length"),
        [
            ("invalid", 5),
            (0, "invalid"),
        ],
    )
    def test_create_text_invalid(self, valid_instances, start, length):
        """Test EndlessTextCreator.create_text with invalid inputs."""
        instance, constructor_args = valid_instances
        with pytest.raises((TypeError, ValueError)):
            instance.create_text(start, length)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("start", "length", "min_length"),
        [
            (-1, 5, 0),
            (0, -1, 0),
        ],
    )
    def test_create_text_edge_cases(self, valid_instances, start, length, min_length):
        """Test EndlessTextCreator.create_text with edge cases."""
        instance, constructor_args = valid_instances
        result = instance.create_text(start, length)
        assert isinstance(result, str)
        assert len(result) >= min_length
