from pathlib import Path
from random import Random
from typing import Any

from transformers import PreTrainedTokenizerBase

from guidellm.data import builders
from guidellm.data.builders import ShortPromptStrategy
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

    # Extract branch specs from data configs (if any synthetic text
    # config has branches defined, pass them to the data loader for
    # graph construction with sub-agent branches)
    branch_specs: list[dict[str, Any]] = []
    for dc in data_config:
        if hasattr(dc, "branches") and dc.branches:
            branch_specs = [b.model_dump() for b in dc.branches]
            break

    data_loader = DataLoaderRegistry.create(
        config=loader_config,
        datasets=datasets,
        preprocessors=preprocessors,
        finalizer=finalizer,
        random_seed=rng.getrandbits(32),
        branch_specs=branch_specs or None,
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
    data: dict,
    output_path: str | Path,
    processor: str | Path | PreTrainedTokenizerBase,
    config: str | Path,
    processor_args: dict[str, Any] | None = None,
    data_args: dict[str, Any] | None = None,
    data_column_mapper: dict[str, str] | None = None,
    short_prompt_strategy: ShortPromptStrategy = ShortPromptStrategy.IGNORE,
    pad_char: str | None = None,
    concat_delimiter: str | None = None,
    include_prefix_in_token_count: bool = False,
    push_to_hub: bool = False,
    hub_dataset_id: str | None = None,
    random_seed: int = 42,
) -> None:
    """
    Main method to process and save a dataset with sampled prompt/output token counts.

    :param data: Path or identifier for dataset input.
    :param output_path: File path to save the processed dataset.
    :param processor: Tokenizer object or its config.
    :param config: PreprocessDatasetConfig string or file path.
    :param processor_args: Optional processor arguments.
    :param data_args: Optional data loading arguments.
    :param data_column_mapper: Optional column mapping dictionary.
    :param short_prompt_strategy: Strategy for handling short prompts.
    :param pad_char: Character used when padding short prompts.
    :param concat_delimiter: Delimiter for concatenation strategy.
    :param include_prefix_in_token_count:
        Whether to include prefix in prompt token count, simplifying the token counts.
        When True, prefix trimming is disabled and the prefix is kept as-is. The prefix
        token count is subtracted from the prompt token budget instead.
    :param push_to_hub: Whether to push to Hugging Face Hub.
    :param hub_dataset_id: Dataset ID on Hugging Face Hub.
    :param random_seed: Seed for random sampling.
    :raises ValueError: If the output path is invalid or pushing conditions unmet.
    """
    data_config = DataArgs.model_validate(data)
    builders.process_dataset(
        data_config,
        output_path,
        processor,
        config,
        processor_args,
        data_args,
        data_column_mapper,
        short_prompt_strategy,
        pad_char,
        concat_delimiter,
        include_prefix_in_token_count,
        push_to_hub,
        hub_dataset_id,
        random_seed,
    )
