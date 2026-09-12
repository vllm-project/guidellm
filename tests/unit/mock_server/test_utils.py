from guidellm.mock_server.utils import MockTokenizer


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
