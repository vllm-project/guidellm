"""Tests for PreprocessStrategyArgs registry models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.data.schemas import (
    ConcatenatePreprocessStrategyArgs,
    ErrorPreprocessStrategyArgs,
    IgnorePreprocessStrategyArgs,
    PadPreprocessStrategyArgs,
    PreprocessStrategyArgs,
)


class TestPreprocessStrategyArgsRegistry:
    """Polymorphic validation and defaults for preprocess strategies.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_ignore_dispatch(self):
        """Kind ignore dispatches to IgnorePreprocessStrategyArgs.

        ## WRITTEN BY AI ##
        """
        result = PreprocessStrategyArgs.model_validate(
            {"kind": "ignore", "prompt_tokens": 10, "output_tokens": 5}
        )
        assert isinstance(result, IgnorePreprocessStrategyArgs)
        assert result.count_prefix is False

    @pytest.mark.smoke
    def test_concatenate_dispatch_with_delimiter(self):
        """Kind concatenate includes delimiter and decodes escapes.

        ## WRITTEN BY AI ##
        """
        result = PreprocessStrategyArgs.model_validate(
            {
                "kind": "concatenate",
                "prompt_tokens": 10,
                "output_tokens": 5,
                "delimiter": "\\n\\n",
            }
        )
        assert isinstance(result, ConcatenatePreprocessStrategyArgs)
        assert result.delimiter == "\n\n"

    @pytest.mark.smoke
    def test_pad_dispatch_with_pad(self):
        """Kind pad includes pad character.

        ## WRITTEN BY AI ##
        """
        result = PreprocessStrategyArgs.model_validate(
            {
                "kind": "pad",
                "prompt_tokens": 10,
                "output_tokens": 5,
                "pad": " ",
            }
        )
        assert isinstance(result, PadPreprocessStrategyArgs)
        assert result.pad == " "

    @pytest.mark.smoke
    def test_error_dispatch(self):
        """Kind error dispatches to ErrorPreprocessStrategyArgs.

        ## WRITTEN BY AI ##
        """
        result = PreprocessStrategyArgs.model_validate(
            {"kind": "error", "prompt_tokens": 10, "output_tokens": 5}
        )
        assert isinstance(result, ErrorPreprocessStrategyArgs)

    @pytest.mark.sanity
    def test_count_prefix_and_token_fields(self):
        """Shared token fields and count_prefix validate on the base.

        ## WRITTEN BY AI ##
        """
        result = PreprocessStrategyArgs.model_validate(
            {
                "kind": "ignore",
                "prompt_tokens": 100,
                "prompt_tokens_stdev": 10,
                "prompt_tokens_min": 80,
                "prompt_tokens_max": 120,
                "output_tokens": 50,
                "prefix_tokens_max": 20,
                "count_prefix": True,
            }
        )
        assert result.prompt_tokens == 100
        assert result.prefix_tokens_max == 20
        assert result.count_prefix is True

    @pytest.mark.sanity
    def test_missing_kind_rejected(self):
        """Missing kind fails validation.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            PreprocessStrategyArgs.model_validate(
                {"prompt_tokens": 10, "output_tokens": 5}
            )

    @pytest.mark.sanity
    def test_pad_specific_field_not_on_ignore(self):
        """Pad-only fields are rejected on ignore strategy.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            PreprocessStrategyArgs.model_validate(
                {
                    "kind": "ignore",
                    "prompt_tokens": 10,
                    "output_tokens": 5,
                    "pad": "X",
                }
            )

    @pytest.mark.regression
    def test_registered_names(self):
        """All four strategy kinds are registered.

        ## WRITTEN BY AI ##
        """
        names = set(PreprocessStrategyArgs.registered_names())
        assert {"ignore", "concatenate", "pad", "error"} <= names
