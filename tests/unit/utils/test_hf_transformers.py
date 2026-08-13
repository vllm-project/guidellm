import pytest
import transformers
from transformers import PreTrainedTokenizerBase

from guidellm.utils.hf_transformers import check_load_processor


class DummyTokenizer(PreTrainedTokenizerBase):
    pass


def test_processor_is_none():
    with pytest.raises(ValueError, match="Processor/Tokenizer is required for test."):
        check_load_processor(None, None, "test")


def test_processor_not_isinstance():
    with pytest.raises(ValueError, match="Invalid processor/Tokenizer for test."):
        check_load_processor(123, None, "test")  # type: ignore


def test_processor_load_by_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    tokenizer = check_load_processor(tmp_path, None, "test")
    assert isinstance(tokenizer, PreTrainedTokenizerBase)


def test_processor_load_error(monkeypatch):
    def raise_error(*args, **kwargs):
        raise RuntimeError("test error")

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", raise_error)
    with pytest.raises(
        ValueError, match="Failed to load processor/Tokenizer for test."
    ):
        check_load_processor("gpt2", None, "test")
