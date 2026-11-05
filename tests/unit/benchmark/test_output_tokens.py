"""Tests for counting output tokens when the server reports no usage."""

from unittest.mock import Mock, patch

import pytest

from guidellm.benchmark.output_tokens import OutputTokenCounter
from guidellm.schemas import GenerationResponse, UsageMetrics
from guidellm.schemas.data.tokenizers import HuggingFaceTokenizerArgs

CONFIG = HuggingFaceTokenizerArgs(model="test-tokenizer")


def _tokenizer_factory(error: Exception | None = None):
    tokenizer = Mock()
    tokenizer.encode = Mock(side_effect=lambda text, **_: text.split())
    factory = Mock(side_effect=error, return_value=tokenizer)
    return factory, tokenizer


async def _load_counter(factory) -> OutputTokenCounter:
    with patch(
        "guidellm.benchmark.output_tokens.TokenizerRegistry.create",
        return_value=factory,
    ):
        return await OutputTokenCounter.load(CONFIG)


def _response(text_tokens: int | None) -> GenerationResponse:
    return GenerationResponse(
        request_id="test",
        request_args=None,
        text=" three",
        reasoning_text="one two",
        output_metrics=UsageMetrics(text_tokens=text_tokens),
    )


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_counts_text_with_tokenizer_when_usage_missing():
    """
    Without usage, output tokens come from the response text, not the request
    target, so early-stopped streams are not reported at the target length.

    ## WRITTEN BY AI ##
    """
    factory, tokenizer = _tokenizer_factory()
    counter = await _load_counter(factory)
    response = _response(text_tokens=None)

    await counter(response)

    assert response.output_metrics.text_tokens == 3
    tokenizer.encode.assert_called_once_with("one two three", add_special_tokens=False)


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_usage_wins_over_tokenizer_count():
    """
    When the server reports usage, its count is kept and the tokenizer is not used.

    ## WRITTEN BY AI ##
    """
    factory, tokenizer = _tokenizer_factory()
    counter = await _load_counter(factory)
    response = _response(text_tokens=7)

    await counter(response)

    assert response.output_metrics.text_tokens == 7
    tokenizer.encode.assert_not_called()


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_tokenizer_load_failure_keeps_previous_behavior():
    """
    If the tokenizer cannot be loaded, the output token count is left unset, as
    before, instead of failing the benchmark.

    ## WRITTEN BY AI ##
    """
    factory, _tokenizer = _tokenizer_factory(error=OSError("no such model"))
    counter = await _load_counter(factory)
    response = _response(text_tokens=None)

    await counter(response)

    assert response.output_metrics.text_tokens is None
