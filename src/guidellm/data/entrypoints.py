from pathlib import Path
from random import Random
from typing import Any

from guidellm.data import builders
from guidellm.data.deserializers import DatasetDeserializerFactory
from guidellm.data.finalizers import DatasetFinalizer, FinalizerRegistry
from guidellm.data.loaders import DataLoader, DataLoaderRegistry
from guidellm.data.preprocessors import (
    DataDependentPreprocessor,
    DatasetPreprocessor,
    PreprocessorRegistry,
)
from guidellm.data.schemas import (
    DataArgs,
    DataFinalizerArgs,
    DataLoaderArgs,
    DataPreprocessorArgs,
    DatasetType,
    DataTokenizerArgs,
    PreprocessStrategyArgs,
)
from guidellm.data.tokenizers import TokenizerRegistry
from guidellm.utils.console import Console
from guidellm.utils.mixins import InfoMixin


async def create_data_loader(
    loader_config: DataLoaderArgs,
    data_config: list[DataArgs],
    tokenizer_config: DataTokenizerArgs,
    column_mapper_config: DataPreprocessorArgs,
    preprocessors_config: list[DataPreprocessorArgs],
    finalizer_config: DataFinalizerArgs,
    random_seed: int = 42,
    console: Console | None = None,
) -> DataLoader:
    """
    Factory function to create a DataLoader instance based on provided configurations.

    :param loader_config: Configuration for the data loader.
    :param data_config: List of configurations for dataset deserialization.
    :param tokenizer_config: Configuration for the tokenizer factory.
    :param column_mapper_config: Configuration for the column mapping preprocessor.
    :param preprocessors_config: List of configurations for additional preprocessors.
    :param finalizer_config: Configuration for the dataset finalizer.
    :param random_seed: Seed for random operations to ensure reproducibility.
    :param console: Optional Console instance for logging and progress display.
    :return: An instance of DataLoader configured according to the provided arguments.
    """
    rng = Random(random_seed)

    tokenizer_factory = TokenizerRegistry.create(tokenizer_config)

    console_step = (
        console.print_update_step(title="Deserializing datasets from configuration")
        if console
        else None
    )

    datasets: list[DatasetType] = [
        DatasetDeserializerFactory.deserialize(
            config=args,
            processor_factory=tokenizer_factory,
            random_seed=rng.getrandbits(32),
        )
        for args in data_config
    ]

    if console_step:
        console_step.finish(
            title=f"{len(datasets)} datasets resolved",
            details=[args.model_dump(mode="json") for args in data_config],
            status_level="success",
        )

    console_step = (
        console.print_update_step(
            title="Initializing preprocessors from configuration",
        )
        if console
        else None
    )

    preproc_configs = [column_mapper_config] + preprocessors_config
    preprocessors: list[DatasetPreprocessor | DataDependentPreprocessor] = [
        PreprocessorRegistry.create(pre) for pre in preproc_configs
    ]

    if console_step:
        console_step.finish(
            title=f"{len(preprocessors)} preprocessors resolved",
            details=[pre.model_dump(mode="json") for pre in preproc_configs],
            status_level="success",
        )

    console_step = (
        console.print_update_step(
            title="Initializing finalizer from configuration",
        )
        if console
        else None
    )

    finalizer: DatasetFinalizer = FinalizerRegistry.create(finalizer_config)

    if console_step:
        console_step.finish(
            title="Finalizer resolved",
            details=finalizer_config.model_dump(mode="json"),
            status_level="success",
        )

    console_step = (
        console.print_update_step(
            title="Initializing request loader from configuration",
        )
        if console
        else None
    )

    data_loader = DataLoaderRegistry.create(
        config=loader_config,
        datasets=datasets,
        preprocessors=preprocessors,
        finalizer=finalizer,
        random_seed=rng.getrandbits(32),
    )

    if console_step:
        samples = loader_config.samples if loader_config.samples > 0 else "inf"
        console_step.finish(
            title=(f"Request loader resolved with {samples} unique requests"),
            details=InfoMixin.extract_from_obj(data_loader),
            status_level="success",
        )

    return data_loader


def process_dataset(
    data: DataArgs | dict[str, Any],
    output_path: str | Path,
    tokenizer: DataTokenizerArgs | dict[str, Any],
    strategy: PreprocessStrategyArgs | dict[str, Any],
    data_column_mapper: DataPreprocessorArgs | dict[str, Any] | None = None,
    data_loader: DataLoaderArgs | dict[str, Any] | None = None,
    push_to_hub: bool = False,
    hub_dataset_id: str | None = None,
    random_seed: int = 42,
) -> None:
    """
    Main method to process and save a dataset with sampled prompt/output token counts.

    :param data: Dataset source configuration (``DataArgs`` or equivalent dict).
    :param output_path: File path to save the processed dataset.
    :param tokenizer: Tokenizer configuration (``DataTokenizerArgs`` or dict).
    :param strategy: Preprocess strategy configuration including token targets and
        short-prompt handling (``PreprocessStrategyArgs`` or dict).
    :param data_column_mapper: Optional column mapping configuration.
    :param data_loader: Optional data loader configuration. ``samples`` limits how
        many processed rows are written; ``shuffle`` and ``num_workers`` are ignored.
    :param push_to_hub: Whether to push to Hugging Face Hub.
    :param hub_dataset_id: Dataset ID on Hugging Face Hub.
    :param random_seed: Seed for random sampling.
    :raises ValueError: If the output path is invalid or pushing conditions unmet.
    """
    data_config = DataArgs.model_validate(data)
    tokenizer_config = DataTokenizerArgs.model_validate(tokenizer)
    strategy_config = PreprocessStrategyArgs.model_validate(strategy)
    column_mapper_config = DataPreprocessorArgs.model_validate(
        data_column_mapper
        if data_column_mapper is not None
        else {"kind": "generative_column_mapper"}
    )
    loader_config = DataLoaderArgs.model_validate(
        data_loader if data_loader is not None else {"kind": "pytorch"}
    )
    builders.process_dataset(
        data_config,
        output_path,
        tokenizer_config,
        strategy_config,
        column_mapper_config,
        push_to_hub,
        hub_dataset_id,
        random_seed,
        loader_config,
    )
