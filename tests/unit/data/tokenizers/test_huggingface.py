"""
Unit tests for guidellm.data.tokenizers.huggingface module.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest

from guidellm.data.tokenizers.huggingface import HuggingFaceTokenizer
from guidellm.schemas.data import HuggingFaceTokenizerArgs
from tests.fixtures.tokenizers import MINIMAL_TOKENIZER_DIR


class TestHuggingFaceTokenizerArgs:
    """Tests for HuggingFaceTokenizerArgs schema.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_default_kind(self):
        """HuggingFaceTokenizerArgs defaults kind to 'huggingface_auto'.

        ### WRITTEN BY AI ###
        """
        args = HuggingFaceTokenizerArgs(model="gpt2")
        assert args.kind == "huggingface_auto"

    @pytest.mark.smoke
    def test_model_field_optional(self):
        """model field is optional at schema level (validated at runtime).

        ### WRITTEN BY AI ###
        """
        args = HuggingFaceTokenizerArgs()
        assert args.model is None

    @pytest.mark.smoke
    def test_load_kwargs_defaults_empty(self):
        """load_kwargs defaults to empty dict.

        ### WRITTEN BY AI ###
        """
        args = HuggingFaceTokenizerArgs(model="gpt2")
        assert args.load_kwargs == {}

    @pytest.mark.sanity
    def test_custom_load_kwargs(self):
        """load_kwargs accepts custom values.

        ### WRITTEN BY AI ###
        """
        args = HuggingFaceTokenizerArgs(
            model="gpt2", load_kwargs={"use_fast": False, "revision": "main"}
        )
        assert args.load_kwargs == {"use_fast": False, "revision": "main"}

    @pytest.mark.regression
    def test_serialization(self):
        """HuggingFaceTokenizerArgs serializes correctly.

        ### WRITTEN BY AI ###
        """
        args = HuggingFaceTokenizerArgs(model="gpt2", load_kwargs={"use_fast": True})
        dumped = args.model_dump()
        assert dumped["kind"] == "huggingface_auto"
        assert dumped["model"] == "gpt2"
        assert dumped["load_kwargs"] == {"use_fast": True}


class TestHuggingFaceTokenizer:
    """Tests for HuggingFaceTokenizer implementation.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_construction_requires_model(self):
        """HuggingFaceTokenizer raises ValueError if model is None.

        ### WRITTEN BY AI ###
        """
        config = HuggingFaceTokenizerArgs()
        with pytest.raises(ValueError, match="must be provided"):
            HuggingFaceTokenizer(config)

    @pytest.mark.smoke
    def test_construction_with_model(self):
        """HuggingFaceTokenizer constructs successfully with model.

        ### WRITTEN BY AI ###
        """
        config = HuggingFaceTokenizerArgs(model="gpt2")
        tokenizer = HuggingFaceTokenizer(config)
        assert tokenizer is not None

    @pytest.mark.sanity
    def test_lazy_loading_not_called_on_init(self):
        """Tokenizer is not loaded during construction.

        ### WRITTEN BY AI ###
        """
        config = HuggingFaceTokenizerArgs(model="gpt2")
        tokenizer = HuggingFaceTokenizer(config)
        assert tokenizer._tokenizer is None

    @pytest.mark.slow
    @pytest.mark.sanity
    def test_lazy_loading_on_call(self):
        """Tokenizer is loaded on first call from the vendored fixture.

        ### WRITTEN BY AI ###
        """
        config = HuggingFaceTokenizerArgs(
            model=str(MINIMAL_TOKENIZER_DIR),
            load_kwargs={"local_files_only": True},
        )
        tokenizer = HuggingFaceTokenizer(config)

        # First call loads
        result = tokenizer()
        assert result is not None
        assert tokenizer._tokenizer is not None

    @pytest.mark.slow
    @pytest.mark.sanity
    def test_caching(self):
        """Second call returns cached tokenizer.

        ### WRITTEN BY AI ###
        """
        config = HuggingFaceTokenizerArgs(
            model=str(MINIMAL_TOKENIZER_DIR),
            load_kwargs={"local_files_only": True},
        )
        tokenizer = HuggingFaceTokenizer(config)

        first_call = tokenizer()
        second_call = tokenizer()
        assert first_call is second_call

    @pytest.mark.slow
    @pytest.mark.regression
    def test_load_kwargs_passed_through(self, monkeypatch: pytest.MonkeyPatch):
        """load_kwargs are passed to AutoTokenizer.from_pretrained.

        ### WRITTEN BY AI ###
        """
        captured: dict = {}

        def fake_from_pretrained(model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs
            return object()

        monkeypatch.setattr(
            "guidellm.data.tokenizers.huggingface.AutoTokenizer.from_pretrained",
            fake_from_pretrained,
        )
        config = HuggingFaceTokenizerArgs(
            model=str(MINIMAL_TOKENIZER_DIR),
            load_kwargs={"local_files_only": True, "use_fast": False},
        )
        tokenizer = HuggingFaceTokenizer(config)
        result = tokenizer()
        assert result is not None
        assert captured["model"] == str(MINIMAL_TOKENIZER_DIR)
        assert captured["kwargs"]["local_files_only"] is True
        assert captured["kwargs"]["use_fast"] is False
