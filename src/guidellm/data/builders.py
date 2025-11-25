import os
from collections.abc import Callable, Iterator
from enum import Enum
from pathlib import Path
from typing import Any, cast

from datasets import Dataset
from loguru import logger
from transformers import PreTrainedTokenizerBase

from guidellm.data.config import load_config
from guidellm.data.deserializers import (
    DatasetDeserializerFactory,
)
from guidellm.data.preprocessors import GenerativeColumnMapper
from guidellm.data.schemas import PreprocessDatasetConfig
from guidellm.utils import IntegerRangeSampler, check_load_processor
from guidellm.utils.hf_datasets import SUPPORTED_TYPES, save_dataset_to_file


class PromptTooShortError(Exception):
    pass


class ShortPromptStrategy(str, Enum):
    IGNORE = "ignore"
    CONCATENATE = "concatenate"
    PAD = "pad"
    ERROR = "error"


class ShortPromptStrategyHandler:
    """Handler class for short prompt strategies."""

    @staticmethod
    def handle_ignore(
        current_prompt: str,
        min_prompt_tokens: int,
        tokenizer: PreTrainedTokenizerBase,
        **_kwargs,
    ) -> str | None:
        """
        Ignores prompts that are shorter than the required minimum token length.

        :param current_prompt: The input prompt string.
        :param min_prompt_tokens: Minimum required token count.
        :param tokenizer: Tokenizer used to count tokens.
        :return: The prompt if it meets the length, otherwise None.
        """

        if len(tokenizer.encode(current_prompt)) < min_prompt_tokens:
            logger.warning("Prompt too short, ignoring")
            return None
        return current_prompt

    @staticmethod
    def handle_concatenate(
        current_prompt: str,
        min_prompt_tokens: int,
        dataset_iterator: Iterator[dict[str, Any]],
        prompt_column: str,
        tokenizer: PreTrainedTokenizerBase,
        concat_delimiter: str,
        **_kwargs,
    ) -> str | None:
        """
        Concatenates prompts until the minimum token requirement is met.

        :param current_prompt: The initial prompt.
        :param min_prompt_tokens: Target minimum token length.
        :param dataset_iterator: Iterator to fetch more prompts.
        :param prompt_column: Column key for prompt extraction.
        :param tokenizer: Tokenizer used to count tokens.
        :param concat_delimiter: Delimiter to use between prompts.
        :return: Concatenated prompt or None if not enough data.
        """

        tokens_len = len(tokenizer.encode(current_prompt))
        while tokens_len < min_prompt_tokens:
            try:
                next_row = next(dataset_iterator)
            except StopIteration:
                logger.warning(
                    "Could not concatenate enough prompts to reach minimum "
                    "length, ignoring"
                )
                return None
            current_prompt += concat_delimiter + next_row[prompt_column]
            tokens_len = len(tokenizer.encode(current_prompt))
        return current_prompt

    @staticmethod
    def handle_pad(
        current_prompt: str,
        min_prompt_tokens: int,
        tokenizer: PreTrainedTokenizerBase,
        pad_char: str,
        pad_multiplier: int = 2,
        **_kwargs,
    ) -> str:
        """
        Pads the prompt with a character until it reaches the minimum token length.

        :param current_prompt: The input prompt.
        :param min_prompt_tokens: Desired minimum token count.
        :param tokenizer: Tokenizer used to count tokens.
        :param pad_char: Character used for padding.
        :param pad_multiplier: Multiplier for padding character length.
        :return: Padded prompt string.
        """
        tokens = tokenizer.encode(current_prompt)
        pad_count = 1
        prompt = current_prompt
        while len(tokens) < min_prompt_tokens:
            prompt += pad_char * pad_count
            tokens = tokenizer.encode(prompt)
            pad_count *= pad_multiplier
        return prompt

    @staticmethod
    def handle_error(
        current_prompt: str,
        min_prompt_tokens: int,
        tokenizer: PreTrainedTokenizerBase,
        **_kwargs,
    ) -> str | None:
        """
        Raises an error if the prompt is too short.

        :param current_prompt: The input prompt.
        :param min_prompt_tokens: Required token count.
        :param tokenizer: Tokenizer used to count tokens.
        :return: The input prompt if valid.
        :raises PromptTooShortError: If the prompt is too short.
        """

        prompt_len = len(tokenizer.encode(current_prompt))
        if prompt_len < min_prompt_tokens:
            raise PromptTooShortError(
                f"Found too short prompt: {current_prompt}, with length: {prompt_len}. "
                f"Minimum length required: {min_prompt_tokens}.",
            )
        return current_prompt

    @classmethod
    def get_strategy_handler(cls, strategy: ShortPromptStrategy) -> Callable[..., Any]:
        """
        Get the handler for a specific strategy.

        :param strategy: The short prompt strategy to get the handler for.
        :return: The handler callable for the specified strategy.
        """
        return cast("Callable[..., Any]", STRATEGY_HANDLERS[strategy])


# Initialize STRATEGY_HANDLERS after class definition to allow method references
STRATEGY_HANDLERS = {
    ShortPromptStrategy.IGNORE: ShortPromptStrategyHandler.handle_ignore,
    ShortPromptStrategy.CONCATENATE: ShortPromptStrategyHandler.handle_concatenate,
    ShortPromptStrategy.PAD: ShortPromptStrategyHandler.handle_pad,
    ShortPromptStrategy.ERROR: ShortPromptStrategyHandler.handle_error,
}


def _validate_output_suffix(output_path: str | Path) -> None:
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported file suffix '{suffix}' in output_path '{output_path}'. "
            f"Only {SUPPORTED_TYPES} are supported."
        )


def parse_synthetic_config(
    config_input: str | Path,
) -> PreprocessDatasetConfig:
    """
    Parse PreprocessDatasetConfig from string or file path.

    Reuses SyntheticTextDatasetDeserializer's parsing logic to support:
    - JSON strings
    - Key=value pairs
    - File paths (.json, .yaml, .yml, .config)

    :param config_input: String or path to config.
    :return: Parsed PreprocessDatasetConfig instance.
    :raises ValueError: If the format is not recognized or parsing fails.
    """
    config = load_config(config_input, PreprocessDatasetConfig)

    if config is not None:
        return config

    raise ValueError(
        f"Could not parse config from input: {config_input}. "
        "Expected JSON string, key=value pairs, or file path "
        "(.json, .yaml, .yml, .config)"
    )


def process_dataset(
    data: str | Path,
    output_path: str | Path,
    processor: str | Path | PreTrainedTokenizerBase,
    config: str | Path,
    processor_args: dict[str, Any] | None,
    data_args: dict[str, Any] | None,
    data_column_mapper: dict[str, str] | None,
    short_prompt_strategy: ShortPromptStrategy,
    pad_char: str | None,
    concat_delimiter: str | None,
    include_prefix_in_token_count: bool,
    push_to_hub: bool,
    hub_dataset_id: str | None,
    random_seed: int,
) -> None:
    """
    Main method to process and save a dataset with sampled prompt/output token counts.
    """
    _validate_output_suffix(output_path)
    logger.info(
        f"Starting dataset conversion | Input: {data} | Output: {output_path}"
    )

    # Parse config
    config_obj = parse_synthetic_config(config)

    # Load tokenizer
    tokenizer = check_load_processor(
        processor,
        processor_args,
        "dataset conversion.",
    )

    # Load dataset
    dataset = DatasetDeserializerFactory.deserialize(
        data=data,
        processor_factory=lambda: tokenizer,
        random_seed=random_seed,
        **(data_args or {}),
    )

    # Setup column mapper
    column_mapper = GenerativeColumnMapper(
        column_mappings=data_column_mapper  # type: ignore[arg-type]
    )
    column_mapper.setup_data(
        datasets=[dataset],
        data_args=[data_args or {}],
    )

    # Extract column names from mapper
    prompt_column, prefix_column, output_column = _extract_column_names(column_mapper)

    # Create token samplers
    prompt_token_sampler, output_token_sampler, prefix_tokens_max = (
        _create_token_samplers(
            config_obj,
            random_seed,
        )
    )

    # Process dataset
    dataset_iterator = iter(dataset)
    processed_prompts = []
    prompt_handler = ShortPromptStrategyHandler.get_strategy_handler(
        short_prompt_strategy
    )

    for row in dataset_iterator:
        processed_row = _process_single_row(
            row=row,
            prompt_column=prompt_column,
            prefix_column=prefix_column,
            prompt_token_sampler=prompt_token_sampler,
            output_token_sampler=output_token_sampler,
            tokenizer=tokenizer,
            prompt_handler=prompt_handler,
            dataset_iterator=dataset_iterator,
            include_prefix_in_token_count=include_prefix_in_token_count,
            pad_char=pad_char,
            concat_delimiter=concat_delimiter,
            output_column=output_column,
            prefix_tokens_max=prefix_tokens_max,
        )
        if processed_row is not None:
            processed_prompts.append(processed_row)

        # Finalize
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

    text_mappings = column_mapper.datasets_column_mappings.get("text_column", [])
    if not text_mappings:
        raise ValueError("Could not find text column in dataset")
    prompt_column = text_mappings[0][1]

    prefix_mappings = column_mapper.datasets_column_mappings.get("prefix_column", [])
    prefix_column = prefix_mappings[0][1] if prefix_mappings else None

    output_mappings = column_mapper.datasets_column_mappings.get(
        "output_tokens_count_column", []
    )
    output_column = (
        output_mappings[0][1] if output_mappings else "output_tokens_count"
    )

    return prompt_column, prefix_column, output_column


def _create_token_samplers(
    config_obj: PreprocessDatasetConfig,
    random_seed: int,
) -> tuple[Iterator[int], Iterator[int], int | None]:
    """
    Create token samplers for prompt, output, and prefix tokens.

    :param config_obj: Configuration object with token settings.
    :param prefix_tokens: Optional single prefix token count.
    :param random_seed: Seed for random sampling.
    :return: Tuple of (prompt_sampler, output_sampler, prefix_tokens_max).
        prefix_sampler is None when prefix_tokens is not provided.
        prefix_tokens_max is the maximum prefix token limit from config.
    """
    prompt_token_sampler = iter(
        IntegerRangeSampler(
            average=config_obj.prompt_tokens,
            variance=config_obj.prompt_tokens_stdev,
            min_value=config_obj.prompt_tokens_min,
            max_value=config_obj.prompt_tokens_max,
            random_seed=random_seed,
        )
    )

    output_token_sampler = iter(
        IntegerRangeSampler(
            average=config_obj.output_tokens,
            variance=config_obj.output_tokens_stdev,
            min_value=config_obj.output_tokens_min,
            max_value=config_obj.output_tokens_max,
            random_seed=random_seed,
        )
    )

    return prompt_token_sampler, output_token_sampler, config_obj.prefix_tokens_max


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
    :param target_prompt_len: Target prompt token length.
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
    prompt_handler: Callable,
    dataset_iterator: Iterator[dict[str, Any]],
    include_prefix_in_token_count: bool,
    pad_char: str | None,
    concat_delimiter: str | None,
    output_column: str,
    prefix_tokens_max: int | None,
) -> dict[str, Any] | None:
    """
    Process a single row from the dataset.

    :param include_prefix_in_token_count: When True, includes prefix tokens in the
        prompt token count calculation. When False, prefix tokens are not counted
        toward prompt tokens.
    :param prefix_tokens_max: Maximum prefix token limit. If set, the prefix will be
        trimmed if it exceeds this limit.
    :return: Processed row dictionary or None if row should be skipped.
    """
    # Extract prompt and prefix
    prompt_text = row.get(prompt_column, "")
    prefix_text = row.get(prefix_column) if prefix_column else None

    # Sample target prompt token count
    target_prompt_len = next(prompt_token_sampler)
    count_adjustment = 0

    # Handle prefix
    if prefix_text:
        # Apply prefix_tokens_max limit if set (strict maximum)
        if prefix_tokens_max is not None:
            prefix_tokens_list = tokenizer.encode(prefix_text)
            if len(prefix_tokens_list) > prefix_tokens_max:
                prefix_text = tokenizer.decode(
                    prefix_tokens_list[:prefix_tokens_max]
                )

        # Count prefix tokens toward prompt if enabled
        if include_prefix_in_token_count:
            count_adjustment = len(tokenizer.encode(prefix_text))

    if target_prompt_len == 0:
        logger.warning("zero prompt size requested; skipping row")
        return None
    elif count_adjustment > 0:
        adjusted_prompt_len = target_prompt_len - count_adjustment
        if adjusted_prompt_len <= 0:
            logger.warning("The prefix exceeds target output length with "
                           "--include-prefix-in-token-count enabled; Using prompt size"
                            "of 1; skipping row")
            return None
        target_prompt_len = adjusted_prompt_len

    # Handle short prompts
    prompt_text = prompt_handler(
        current_prompt=prompt_text,
        min_prompt_tokens=target_prompt_len,
        dataset_iterator=dataset_iterator,
        prompt_column=prompt_column,
        tokenizer=tokenizer,
        pad_char=pad_char,
        concat_delimiter=concat_delimiter,
    )
    if prompt_text is None:
        return None

    # Trim long prompts
    tokens = tokenizer.encode(prompt_text)
    if len(tokens) > target_prompt_len:
        prompt_text = tokenizer.decode(tokens[:target_prompt_len])
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

    logger.info(f"Generated processed dataset with {len(processed_prompts)} prompts")

    processed_dataset = Dataset.from_list(processed_prompts)
    save_dataset_to_file(processed_dataset, output_path)
    logger.info(f"Conversion completed. Dataset saved to: {output_path}")

    if push_to_hub:
        push_dataset_to_hub(hub_dataset_id, processed_dataset)
        logger.info(f"Pushed dataset to: {hub_dataset_id}")


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
