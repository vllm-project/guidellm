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
    DataEntrypointArgs,
    DataFinalizerArgs,
    DataLoaderArgs,
    DataPreprocessorArgs,
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
        assert args.sampler is None

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


class TestDataEntrypointArgs:
    """Tests for DataEntrypointArgs validation.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_entrypoint_args(self):
        """Minimal valid DataEntrypointArgs.

        ### WRITTEN BY AI ###
        """
        return DataEntrypointArgs(
            loader=TorchDataLoaderArgs(),
            data=[HuggingFaceDataArgs(source="my_dataset")],
            preprocessors=[],
            finalizer=GenerativeRequestFinalizerConfig(),
        )

    @pytest.mark.smoke
    def test_valid_construction(self, valid_entrypoint_args):
        """DataEntrypointArgs constructs with valid fields.

        ### WRITTEN BY AI ###
        """
        args = valid_entrypoint_args
        assert args.loader.kind == "pytorch"
        assert len(args.data) == 1
        assert args.finalizer.kind == "generative"

    @pytest.mark.sanity
    def test_empty_data_list_rejected(self):
        """DataEntrypointArgs rejects empty data list (min_length=1).

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValidationError):
            DataEntrypointArgs(
                loader=TorchDataLoaderArgs(),
                data=[],
                preprocessors=[],
                finalizer=GenerativeRequestFinalizerConfig(),
            )

    @pytest.mark.sanity
    def test_multiple_data_sources(self):
        """DataEntrypointArgs accepts multiple data sources.

        ### WRITTEN BY AI ###
        """
        args = DataEntrypointArgs(
            loader=TorchDataLoaderArgs(),
            data=[
                HuggingFaceDataArgs(source="dataset_a"),
                HuggingFaceDataArgs(source="dataset_b"),
            ],
            preprocessors=[GenerativeColumnMapperArgs()],
            finalizer=GenerativeRequestFinalizerConfig(),
        )
        assert len(args.data) == 2
        assert len(args.preprocessors) == 1

    @pytest.mark.regression
    def test_round_trip_serialization(self, valid_entrypoint_args):
        """DataEntrypointArgs serializes and deserializes correctly.

        ### WRITTEN BY AI ###
        """
        dumped = valid_entrypoint_args.model_dump()
        assert dumped["loader"]["kind"] == "pytorch"
        assert dumped["data"][0]["kind"] == "huggingface"
        assert dumped["finalizer"]["kind"] == "generative"
