import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import Dataset
from loguru import logger
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers import DatasetDeserializerFactory
from guidellm.data.preprocessors import GenerativeColumnMapper, PreprocessorRegistry
from guidellm.data.schemas import (
    DataArgs,
    DataLoaderArgs,
    DataPreprocessorArgs,
    DataTokenizerArgs,
    PreprocessStrategyArgs,
    PromptTooShortError,
)
from guidellm.data.tokenizers import TokenizerRegistry
from guidellm.utils.hf_datasets import SUPPORTED_TYPES, save_dataset_to_file
from guidellm.utils.random import IntegerRangeSampler

__all__ = [
    "PromptTooShortError",
    "process_dataset",
    "push_dataset_to_hub",
]


def _validate_output_suffix(output_path: str | Path) -> None:
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported file suffix '{suffix}' in output_path '{output_path}'. "
            f"Only {SUPPORTED_TYPES} are supported."
        )


def process_dataset(
    data: DataArgs,
    output_path: str | Path,
    tokenizer: DataTokenizerArgs,
    strategy: PreprocessStrategyArgs,
    data_column_mapper: DataPreprocessorArgs,
    push_to_hub: bool,
    hub_dataset_id: str | None,
    random_seed: int,
    data_loader: DataLoaderArgs | None = None,
) -> None:
    """
    Main method to process and save a dataset with sampled prompt/output token counts.

    :param data_loader: Optional loader config. When ``samples`` is greater than 0,
        processing stops after that many successfully processed rows. ``shuffle`` and
        ``num_workers`` are ignored for preprocessing.
    """
    _validate_output_suffix(output_path)
    logger.info(
        "Starting dataset conversion | Input: {} | Output: {}", data, output_path
    )

    # samples > 0 caps successfully processed output rows; -1 means unlimited.
    # shuffle / num_workers on TorchDataLoaderArgs are ignored for preprocess.
    max_samples = data_loader.samples if data_loader is not None else -1

    # Load tokenizer
    tokenizer_factory = TokenizerRegistry.create(tokenizer)
    loaded_tokenizer = tokenizer_factory()

    # Load dataset
    dataset = DatasetDeserializerFactory.deserialize(
        config=data,
        processor_factory=tokenizer_factory,
        random_seed=random_seed,
    )
    # Setup column mapper
    column_mapper: GenerativeColumnMapper = PreprocessorRegistry.create(  # type: ignore[assignment]
        config=data_column_mapper
    )
    column_mapper.setup_data(
        datasets=[dataset],
    )

    # Extract column names from mapper
    prompt_column, prefix_column, output_column = _extract_column_names(column_mapper)

    # Create token samplers
    prompt_token_sampler, output_token_sampler = _create_token_samplers(
        strategy,
        random_seed,
    )

    # Process dataset
    dataset_iterator = iter(dataset)
    processed_prompts = []

    for row in dataset_iterator:
        processed_row = _process_single_row(
            row=row,
            prompt_column=prompt_column,
            prefix_column=prefix_column,
            prompt_token_sampler=prompt_token_sampler,
            output_token_sampler=output_token_sampler,
            tokenizer=loaded_tokenizer,
            strategy=strategy,
            dataset_iterator=dataset_iterator,
            output_column=output_column,
        )
        if processed_row is not None:
            processed_prompts.append(processed_row)
            if max_samples > 0 and len(processed_prompts) >= max_samples:
                break

    _finalize_processed_dataset(
        processed_prompts,
        output_path,
        push_to_hub,
        hub_dataset_id,
    )


def _extract_column_names(
    column_mapper: GenerativeColumnMapper,
) -> tuple[str, str | None, str]:
    """
    Extract column names for prompt, prefix, and output from column mapper.

    :param column_mapper: Initialized column mapper.
    :return: Tuple of (prompt_column, prefix_column, output_column).
    :raises ValueError: If column mapper is not properly initialized.
    """
    if column_mapper.datasets_column_mappings is None:
        raise ValueError("Column mapper not properly initialized")

    try:
        text_mappings = column_mapper.datasets_column_mappings[("text_column", 0)]
        prompt_column = text_mappings[0][1]
    except KeyError as err:
        raise ValueError("Could not find text column in dataset") from err

    try:
        prefix_mappings = column_mapper.datasets_column_mappings[("prefix_column", 0)]
        prefix_column = prefix_mappings[0][1]
    except (KeyError, IndexError):
        prefix_column = None

    try:
        output_mappings = column_mapper.datasets_column_mappings[
            ("output_tokens_count_column", 0)
        ]
        output_column = output_mappings[0][1]
    except (KeyError, IndexError):
        output_column = "output_tokens_count"

    return prompt_column, prefix_column, output_column


def _create_token_samplers(
    strategy: PreprocessStrategyArgs,
    random_seed: int,
) -> tuple[Iterator[int], Iterator[int]]:
    """
    Create token samplers for prompt and output tokens.

    :param strategy: Preprocess strategy with token count settings.
    :param random_seed: Seed for random sampling.
    :return: Tuple of (prompt_sampler, output_sampler).
    """
    prompt_token_sampler = iter(
        IntegerRangeSampler(
            average=strategy.prompt_tokens,
            variance=strategy.prompt_tokens_stdev,
            min_value=strategy.prompt_tokens_min,
            max_value=strategy.prompt_tokens_max,
            random_seed=random_seed,
        )
    )

    output_token_sampler = iter(
        IntegerRangeSampler(
            average=strategy.output_tokens,
            variance=strategy.output_tokens_stdev,
            min_value=strategy.output_tokens_min,
            max_value=strategy.output_tokens_max,
            random_seed=random_seed,
        )
    )

    return prompt_token_sampler, output_token_sampler


def _process_dataset_row(
    row: dict[str, Any],
    prompt_column: str,
    prefix_column: str | None,
    output_column: str,
    target_output_len: int,
    prompt_text: str,
    prefix_text: str | None,
    tokens: list[int],
) -> dict[str, Any]:
    """
    Create a processed row from the processed prompt/prefix data.

    :param row: Original dataset row.
    :param prompt_column: Name of prompt column.
    :param prefix_column: Name of prefix column or None.
    :param output_column: Name of output tokens count column.
    :param target_output_len: Target output token length.
    :param prompt_text: Processed prompt text.
    :param prefix_text: Processed prefix text or None.
    :param tokens: Tokenized prompt.
    :return: Processed row dictionary.
    """
    processed_row = row.copy()
    processed_row[prompt_column] = prompt_text
    if prefix_column and prefix_text:
        processed_row[prefix_column] = prefix_text
    processed_row["prompt_tokens_count"] = len(tokens)
    processed_row[output_column] = target_output_len
    return processed_row


def _process_single_row(
    row: dict[str, Any],
    prompt_column: str,
    prefix_column: str | None,
    prompt_token_sampler: Iterator[int],
    output_token_sampler: Iterator[int],
    tokenizer: PreTrainedTokenizerBase,
    strategy: PreprocessStrategyArgs,
    dataset_iterator: Iterator[dict[str, Any]],
    output_column: str,
) -> dict[str, Any] | None:
    """
    Process a single row from the dataset.

    :param strategy: Preprocess strategy controlling token targets and short prompts.
    :return: Processed row dictionary or None if row should be skipped.
    """
    # Extract prompt and prefix
    prompt_text: str = row.get(prompt_column, "")
    prefix_text: str | None = row.get(prefix_column) if prefix_column else None

    # Sample target prompt token count
    target_prompt_len = next(prompt_token_sampler)
    count_adjustment = 0

    # Handle prefix
    if prefix_text:
        # Apply prefix_tokens_max limit if set (strict maximum)
        if strategy.prefix_tokens_max is not None:
            prefix_tokens_list = tokenizer.encode(prefix_text)
            if len(prefix_tokens_list) > strategy.prefix_tokens_max:
                prefix_text = tokenizer.decode(  # type: ignore[assignment]
                    prefix_tokens_list[: strategy.prefix_tokens_max]
                )

        # Count prefix tokens toward prompt if enabled
        if strategy.count_prefix:
            count_adjustment = len(tokenizer.encode(prefix_text))

    if target_prompt_len == 0:
        logger.warning("zero prompt size requested; skipping row")
        return None
    elif count_adjustment > 0:
        adjusted_prompt_len = target_prompt_len - count_adjustment
        if adjusted_prompt_len <= 0:
            logger.warning(
                "The prefix exceeds target output length with "
                "count_prefix enabled; Using prompt size"
                "of 1; skipping row"
            )
            return None
        target_prompt_len = adjusted_prompt_len

    # Handle short prompts via strategy encapsulation
    handled_prompt = strategy.handle_short_prompt(
        current_prompt=prompt_text,
        min_prompt_tokens=target_prompt_len,
        tokenizer=tokenizer,
        dataset_iterator=dataset_iterator,
        prompt_column=prompt_column,
    )
    if handled_prompt is None:
        return None
    prompt_text = handled_prompt

    # Trim long prompts
    tokens = tokenizer.encode(prompt_text)
    if len(tokens) > target_prompt_len:
        prompt_text = tokenizer.decode(tokens[:target_prompt_len])  # type: ignore[assignment]
        tokens = tokenizer.encode(prompt_text)

    # Sample output token count
    target_output_len = next(output_token_sampler)

    # Create processed row
    return _process_dataset_row(
        row=row,
        prompt_column=prompt_column,
        prefix_column=prefix_column,
        output_column=output_column,
        target_output_len=target_output_len,
        prompt_text=prompt_text,
        prefix_text=prefix_text,
        tokens=tokens,
    )


def _finalize_processed_dataset(
    processed_prompts: list[dict[str, Any]],
    output_path: str | Path,
    push_to_hub: bool,
    hub_dataset_id: str | None,
) -> None:
    """
    Finalize the processed dataset by saving and optionally pushing to hub.

    :param processed_prompts: List of processed row dictionaries.
    :param output_path: Path to save the dataset.
    :param push_to_hub: Whether to push to Hugging Face Hub.
    :param hub_dataset_id: Dataset ID on Hugging Face Hub.
    """
    if not processed_prompts:
        logger.error("No prompts remained after processing")
        return

    logger.info("Generated processed dataset with {} prompts", len(processed_prompts))

    processed_dataset = Dataset.from_list(processed_prompts)
    save_dataset_to_file(processed_dataset, output_path)
    logger.info("Conversion completed. Dataset saved to: {}", output_path)

    if push_to_hub:
        push_dataset_to_hub(hub_dataset_id, processed_dataset)
        logger.info("Pushed dataset to: {}", hub_dataset_id)


def push_dataset_to_hub(
    hub_dataset_id: str | None,
    processed_dataset: Dataset,
) -> None:
    """
    Pushes the processed dataset to Hugging Face Hub using HF_TOKEN.

    :param hub_dataset_id: Identifier on the Hub to push to.
    :param processed_dataset: HuggingFace Dataset object.
    :raises ValueError: If hub_dataset_id or HF_TOKEN is not available.
    """

    hf_token = os.environ.get("HF_TOKEN")
    if not hub_dataset_id or not hf_token:
        raise ValueError(
            "hub_dataset_id and HF_TOKEN env var must be provided when push_to_hub"
            " is True"
        )
    processed_dataset.push_to_hub(hub_dataset_id, token=hf_token)
