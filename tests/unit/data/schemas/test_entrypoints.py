"""
Unit tests for guidellm.data.schemas.entrypoints module.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# Import subclasses to ensure registry is populated
import guidellm.data.deserializers  # noqa: F401
import guidellm.data.finalizers  # noqa: F401
import guidellm.data.loaders  # noqa: F401
import guidellm.data.preprocessors  # noqa: F401
from guidellm.data.deserializers.huggingface import HuggingFaceDataArgs
from guidellm.data.finalizers.generative import GenerativeRequestFinalizerConfig
from guidellm.data.loaders.torch import TorchDataLoaderArgs
from guidellm.data.preprocessors.mappers import GenerativeColumnMapperArgs
from guidellm.data.schemas.entrypoints import (
    DataArgs,
    DataFinalizerArgs,
    DataLoaderArgs,
    DataPreprocessorArgs,
    DataTokenizerArgs,
)


class TestDataLoaderArgsRegistry:
    """Tests for DataLoaderArgs base class and registry.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_pytorch_registered(self):
        """TorchDataLoaderArgs is registered under 'pytorch'.

        ### WRITTEN BY AI ###
        """
        from guidellm.data.loaders.loader import DataLoaderRegistry

        obj = DataLoaderRegistry.get_registered_object("pytorch")
        assert obj is not None

    @pytest.mark.smoke
    def test_torch_loader_args_defaults(self):
        """TorchDataLoaderArgs has correct defaults.

        ### WRITTEN BY AI ###
        """
        args = TorchDataLoaderArgs()
        assert args.kind == "pytorch"
        assert args.samples == -1
        assert args.num_workers == 1
        assert args.shuffle is False

    @pytest.mark.sanity
    def test_torch_loader_args_kind_field(self):
        """TorchDataLoaderArgs kind field is always 'pytorch'.

        ### WRITTEN BY AI ###
        """
        args = TorchDataLoaderArgs(samples=100)
        assert args.kind == "pytorch"
        assert args.samples == 100

    @pytest.mark.sanity
    def test_data_loader_args_schema_discriminator(self):
        """DataLoaderArgs schema_discriminator is 'kind'.

        ### WRITTEN BY AI ###
        """
        assert DataLoaderArgs.schema_discriminator == "kind"


class TestDataArgsPolymorphicDispatch:
    """Tests for DataArgs base class polymorphic validation.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_huggingface_dispatch(self):
        """DataArgs.model_validate dispatches to HuggingFaceDataArgs.

        ### WRITTEN BY AI ###
        """
        result = DataArgs.model_validate(
            {"kind": "huggingface", "source": "my_dataset"}
        )
        assert isinstance(result, HuggingFaceDataArgs)
        assert result.kind == "huggingface"
        assert result.source == "my_dataset"

    @pytest.mark.sanity
    def test_unknown_kind_raises(self):
        """DataArgs.model_validate raises for unknown kind.

        ### WRITTEN BY AI ###
        """
        with pytest.raises((ValidationError, ValueError)):
            DataArgs.model_validate({"kind": "nonexistent_kind"})

    @pytest.mark.sanity
    def test_data_args_schema_discriminator(self):
        """DataArgs schema_discriminator is 'kind'.

        ### WRITTEN BY AI ###
        """
        assert DataArgs.schema_discriminator == "kind"


class TestDataPreprocessorArgsRegistry:
    """Tests for DataPreprocessorArgs base class and registry.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_generative_column_mapper_dispatch(self):
        """DataPreprocessorArgs dispatches to GenerativeColumnMapperArgs.

        ### WRITTEN BY AI ###
        """
        result = DataPreprocessorArgs.model_validate(
            {"kind": "generative_column_mapper"}
        )
        assert isinstance(result, GenerativeColumnMapperArgs)
        assert result.kind == "generative_column_mapper"

    @pytest.mark.smoke
    def test_pooling_column_mapper_dispatch(self):
        """DataPreprocessorArgs dispatches to GenerativeColumnMapperArgs.

        ### WRITTEN BY AI ###
        """
        result = DataPreprocessorArgs.model_validate({"kind": "pooling_column_mapper"})
        assert isinstance(result, GenerativeColumnMapperArgs)

    @pytest.mark.sanity
    def test_data_preprocessor_args_schema_discriminator(self):
        """DataPreprocessorArgs schema_discriminator is 'kind'.

        ### WRITTEN BY AI ###
        """
        assert DataPreprocessorArgs.schema_discriminator == "kind"


class TestDataFinalizerArgsRegistry:
    """Tests for DataFinalizerArgs base class and registry.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_generative_dispatch(self):
        """DataFinalizerArgs dispatches to GenerativeRequestFinalizerConfig.

        ### WRITTEN BY AI ###
        """
        result = DataFinalizerArgs.model_validate({"kind": "generative"})
        assert isinstance(result, GenerativeRequestFinalizerConfig)
        assert result.kind == "generative"

    @pytest.mark.sanity
    def test_generative_finalizer_config_defaults(self):
        """GenerativeRequestFinalizerConfig has correct defaults.

        ### WRITTEN BY AI ###
        """
        config = GenerativeRequestFinalizerConfig()
        assert config.kind == "generative"

    @pytest.mark.sanity
    def test_data_finalizer_args_schema_discriminator(self):
        """DataFinalizerArgs schema_discriminator is 'kind'.

        ### WRITTEN BY AI ###
        """
        assert DataFinalizerArgs.schema_discriminator == "kind"


class TestDataTokenizerArgsRegistry:
    """Tests for DataTokenizerArgs base class and registry.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_huggingface_auto_registered(self):
        """HuggingFaceTokenizerArgs is registered under 'huggingface_auto'.

        ### WRITTEN BY AI ###
        """
        from guidellm.data.tokenizers import TokenizerRegistry

        obj = TokenizerRegistry.get_registered_object("huggingface_auto")
        assert obj is not None

    @pytest.mark.smoke
    def test_hf_auto_alias_registered(self):
        """HuggingFaceTokenizerArgs is registered under 'hf_auto' alias.

        ### WRITTEN BY AI ###
        """
        from guidellm.data.tokenizers import TokenizerRegistry

        obj = TokenizerRegistry.get_registered_object("hf_auto")
        assert obj is not None

    @pytest.mark.smoke
    def test_tokenizer_args_defaults(self):
        """HuggingFaceTokenizerArgs has correct defaults.

        ### WRITTEN BY AI ###
        """
        from guidellm.data.tokenizers import HuggingFaceTokenizerArgs

        args = HuggingFaceTokenizerArgs(model="gpt2")
        assert args.kind == "huggingface_auto"
        assert args.model == "gpt2"
        assert args.load_kwargs == {}

    @pytest.mark.sanity
    def test_polymorphic_dispatch(self):
        """DataTokenizerArgs.model_validate dispatches correctly.

        ### WRITTEN BY AI ###
        """
        result = DataTokenizerArgs.model_validate(
            {"kind": "huggingface_auto", "model": "gpt2"}
        )
        from guidellm.data.tokenizers import HuggingFaceTokenizerArgs

        assert isinstance(result, HuggingFaceTokenizerArgs)
        assert result.model == "gpt2"

    @pytest.mark.sanity
    def test_data_tokenizer_args_schema_discriminator(self):
        """DataTokenizerArgs schema_discriminator is 'kind'.

        ### WRITTEN BY AI ###
        """
        assert DataTokenizerArgs.schema_discriminator == "kind"
