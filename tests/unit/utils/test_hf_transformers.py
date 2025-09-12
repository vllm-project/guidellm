import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from transformers import PreTrainedTokenizerBase

from guidellm.utils.hf_transformers import check_load_processor

class DummyTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        pass

@patch("guidellm.utils.hf_transformers.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
def test_check_load_processor_with_tokenizer_instance(mock_from_pretrained):
    tokenizer = DummyTokenizer()
    result = check_load_processor(tokenizer, None, "test")
    assert isinstance(result, PreTrainedTokenizerBase)

@patch("guidellm.utils.hf_transformers.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
def test_check_load_processor_with_tokenizer_name(mock_from_pretrained):
    result = check_load_processor("bert-base-uncased", None, "test")
    assert isinstance(result, PreTrainedTokenizerBase)

@patch("guidellm.utils.hf_transformers.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
def test_check_load_processor_with_tokenizer_path(mock_from_pretrained, tmp_path):
    result = check_load_processor(tmp_path, None, "test")
    assert isinstance(result, PreTrainedTokenizerBase)

def test_check_load_processor_none_raises():
    with pytest.raises(ValueError, match="Processor/Tokenizer is required"):
        check_load_processor(None, None, "test")

def test_check_load_processor_invalid_type_raises():
    with pytest.raises(ValueError, match="Invalid processor/Tokenizer"):
        check_load_processor(123, None, "test")