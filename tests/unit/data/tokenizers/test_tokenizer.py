"""
Unit tests for guidellm.data.tokenizers.tokenizer module.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest

import guidellm.data.tokenizers  # noqa: F401 — ensures tokenizers are registered
from guidellm.data.tokenizers import HuggingFaceTokenizerArgs, TokenizerRegistry
from guidellm.data.tokenizers.huggingface import HuggingFaceTokenizer


class TestTokenizerRegistry:
    """Tests for TokenizerRegistry factory.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_huggingface_auto_registered(self):
        """'huggingface_auto' kind is registered in TokenizerRegistry.

        ### WRITTEN BY AI ###
        """
        tokenizer_cls = TokenizerRegistry.get_registered_object("huggingface_auto")
        assert tokenizer_cls is HuggingFaceTokenizer

    @pytest.mark.smoke
    def test_hf_auto_alias_registered(self):
        """'hf_auto' alias is registered in TokenizerRegistry.

        ### WRITTEN BY AI ###
        """
        tokenizer_cls = TokenizerRegistry.get_registered_object("hf_auto")
        assert tokenizer_cls is HuggingFaceTokenizer

    @pytest.mark.sanity
    def test_unknown_kind_raises(self):
        """TokenizerRegistry.create raises ValueError for unknown kind.

        ### WRITTEN BY AI ###
        """
        config = HuggingFaceTokenizerArgs(model="test")
        # Monkey-patch kind to something not registered
        object.__setattr__(config, "kind", "nonexistent_tokenizer_kind")

        with pytest.raises(ValueError, match="not registered"):
            TokenizerRegistry.create(config)

    @pytest.mark.smoke
    def test_registry_returns_class(self):
        """TokenizerRegistry.get_registered_object returns a class, not an instance.

        ### WRITTEN BY AI ###
        """
        tokenizer_cls = TokenizerRegistry.get_registered_object("huggingface_auto")
        assert tokenizer_cls is HuggingFaceTokenizer
        assert isinstance(tokenizer_cls, type)
