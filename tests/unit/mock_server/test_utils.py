from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from guidellm.mock_server.utils import (
    MockTokenizer,
    create_fake_tokens_str,
)
from tests.fixtures.tokenizers import MINIMAL_TOKENIZER_DIR


def test_mock_tokenizer_returns_raw_token_ids() -> None:
    """Test the mock exposes raw token IDs through its encoding interfaces.

    ## WRITTEN BY AI ##
    """
    tokenizer = MockTokenizer()
    messages = [{"role": "user", "content": "hello world"}]

    token_ids = tokenizer("hello world")

    assert token_ids == tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize("hello world")
    )
    assert tokenizer.encode("hello world") == token_ids
    assert tokenizer.apply_chat_template(messages) == "hello world"
    assert isinstance(tokenizer.apply_chat_template(messages, tokenize=True), list)


@pytest.mark.regression
@pytest.mark.parametrize(
    "tokenizer_kind",
    ["huggingface", "mock"],
    ids=("huggingface", "mock"),
)
def test_fake_token_chunks_are_decoded_correctly(tokenizer_kind) -> None:
    """Test generated chunks are display text matching the requested token count.

    ## WRITTEN BY AI ##
    """
    tokenizer = (
        AutoTokenizer.from_pretrained(MINIMAL_TOKENIZER_DIR)
        if tokenizer_kind == "huggingface"
        else MockTokenizer()
    )
    source_text = "Score each cause. Quality matters."
    num_tokens = len(tokenizer.tokenize(source_text))
    fake = Mock()
    fake.text.return_value = source_text

    chunks = create_fake_tokens_str(num_tokens, tokenizer, fake=fake)

    assert len(chunks) == num_tokens
    assert "".join(chunks) == source_text


@pytest.mark.regression
def test_fake_token_chunks_support_zero_tokens() -> None:
    """Test zero generated tokens produce no chunks.

    ## WRITTEN BY AI ##
    """
    assert create_fake_tokens_str(0, MockTokenizer()) == []
