"""Regression coverage for preprocessing strategy validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.schemas.data import PreprocessStrategyArgs


@pytest.mark.regression
@pytest.mark.parametrize("padding", [{}, {"pad": ""}, {"pad": "\\\n"}])
def test_pad_strategy_rejects_empty_padding(padding):
    """Reject padding that cannot extend a short prompt before processing it.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError, match="pad"):
        PreprocessStrategyArgs.model_validate(
            {
                "kind": "pad",
                "prompt_tokens": 10,
                "output_tokens": 5,
                **padding,
            }
        )
