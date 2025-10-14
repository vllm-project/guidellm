import json
import os
from collections.abc import Callable, Iterator
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset
from loguru import logger
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase

from guidellm.utils import IntegerRangeSampler, check_load_processor
from guidellm.utils.hf_datasets import SUPPORTED_TYPES, save_dataset_to_file


class PromptTooShortError(Exception):
    pass


class ShortPromptStrategy(str, Enum):
    IGNORE = "ignore"
    CONCATENATE = "concatenate"
    PAD = "pad"
    ERROR = "error"


def handle_ignore_strategy(
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


def handle_concatenate_strategy(
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
                "Could not concatenate enough prompts to reach minimum length, ignoring"
            )
            return None
        current_prompt += concat_delimiter + next_row[prompt_column]
        tokens_len = len(tokenizer.encode(current_prompt))
    return current_prompt


def handle_pad_strategy(
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


def handle_error_strategy(
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


STRATEGY_HANDLERS: dict[ShortPromptStrategy, Callable] = {
    ShortPromptStrategy.IGNORE: handle_ignore_strategy,
    ShortPromptStrategy.CONCATENATE: handle_concatenate_strategy,
    ShortPromptStrategy.PAD: handle_pad_strategy,
    ShortPromptStrategy.ERROR: handle_error_strategy,
}


class TokensConfig(BaseModel):
    average: int = Field(
        description="The average number of tokens.",
        gt=0,
    )
    stdev: int | None = Field(
        description="The standard deviation of the tokens.",
        gt=0,
        default=None,
    )
    min: int | None = Field(
        description="The minimum number of tokens.",
        gt=0,
        default=None,
    )
    max: int | None = Field(
        description="The maximum number of tokens.",
        gt=0,
        default=None,
    )

    @staticmethod
    def parse_str(data: str | Path) -> "TokensConfig":
        """
        Parses a string or path into a TokensConfig object. Supports:
        - JSON string
        - key=value pairs
        - file path to .yaml/.config

        :param data: String or path containing configuration.
        :return: Parsed TokensConfig instance.
        :raises ValueError: If the format is not recognized.
        """

        if (
            isinstance(data, Path)
            or data.strip().endswith(".config")
            or data.strip().endswith(".yaml")
        ):
            return TokensConfig.parse_config_file(data)

        if data.strip().startswith("{"):
            return TokensConfig.parse_json(data)

        if data.count("=") > 1:
            return TokensConfig.parse_key_value_pairs(data)

        raise ValueError(
            f"Unsupported data format. Expected JSON or key-value pairs, got {data}"
        )

    @staticmethod
    def parse_json(data: str) -> "TokensConfig":
        config_dict = json.loads(data.strip())

        return TokensConfig(**config_dict)

    @staticmethod
    def parse_key_value_pairs(data: str) -> "TokensConfig":
        config_dict = {}
        items = data.strip().split(",")
        for item in items:
            key, value = item.split("=")
            config_dict[key.strip()] = (
                int(value.strip()) if value.strip().isnumeric() else value.strip()
            )

        return TokensConfig(**config_dict)  # type: ignore[arg-type]

    @staticmethod
    def parse_config_file(data: str | Path) -> "TokensConfig":
        with Path(data).open("r") as file:
            config_dict = yaml.safe_load(file)

        return TokensConfig(**config_dict)


def _validate_output_suffix(output_path: str | Path) -> None:
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported file suffix '{suffix}' in output_path '{output_path}'. "
            f"Only {SUPPORTED_TYPES} are supported."
        )


def process_dataset(
    data: str | Path,
    output_path: str | Path,
    processor: str | Path | PreTrainedTokenizerBase,
    prompt_tokens: str | Path,
    output_tokens: str | Path,
    processor_args: dict[str, Any] | None = None,
    data_args: dict[str, Any] | None = None,
    short_prompt_strategy: ShortPromptStrategy = ShortPromptStrategy.IGNORE,
    pad_char: str | None = None,
    concat_delimiter: str | None = None,
    push_to_hub: bool = False,
    hub_dataset_id: str | None = None,
    random_seed: int = 42,
) -> None:
    """
    Main method to process and save a dataset with sampled prompt/output token counts.

    :param data: Path or identifier for dataset input.
    :param output_path: File path to save the processed dataset.
    :param processor: Tokenizer object or its config.
    :param prompt_tokens: Prompt token config string or file.
    :param output_tokens: Output token config string or file.
    :param processor_args: Optional processor arguments.
    :param data_args: Optional data loading arguments.
    :param short_prompt_strategy: Strategy for handling short prompts.
    :param pad_char: Character used when padding short prompts.
    :param concat_delimiter: Delimiter for concatenation strategy.
    :param push_to_hub: Whether to push to Hugging Face Hub.
    :param hub_dataset_id: Dataset ID on Hugging Face Hub.
    :param random_seed: Seed for random sampling.
    :raises ValueError: If output path is invalid or pushing conditions unmet.
    """

    _validate_output_suffix(output_path)
    logger.info(
        f"Starting dataset conversion | Input: {data} | Output directory: {output_path}"
    )

    dataset, column_mappings = None, None
    tokenizer = check_load_processor(
        processor,
        processor_args,
        "dataset conversion.",
    )
    prompt_column = column_mappings.get("prompt_column")
    output_column = column_mappings.get(
        "output_tokens_count_column", "output_tokens_count"
    )

    prompt_tokens_cfg = TokensConfig.parse_str(prompt_tokens)
    output_tokens_cfg = TokensConfig.parse_str(output_tokens)

    prompt_token_sampler = iter(
        IntegerRangeSampler(
            average=prompt_tokens_cfg.average,
            variance=prompt_tokens_cfg.stdev,
            min_value=prompt_tokens_cfg.min,
            max_value=prompt_tokens_cfg.max,
            random_seed=random_seed,
        )
    )

    output_token_sampler = iter(
        IntegerRangeSampler(
            average=output_tokens_cfg.average,
            variance=output_tokens_cfg.stdev,
            min_value=output_tokens_cfg.min,
            max_value=output_tokens_cfg.max,
            random_seed=random_seed,
        )
    )

    dataset_iterator = iter(dataset)
    processed_prompts = []
    prompt_handler = STRATEGY_HANDLERS[short_prompt_strategy]

    for prompt_row in dataset_iterator:
        prompt_text = prompt_row[prompt_column]
        target_prompt_len = next(prompt_token_sampler)

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
            continue

        tokens = tokenizer.encode(prompt_text)
        if len(tokens) > target_prompt_len:
            prompt_text = tokenizer.decode(tokens[:target_prompt_len])

        processed_prompt = prompt_row.copy()
        processed_prompt[prompt_column] = prompt_text
        processed_prompt["prompt_tokens_count"] = target_prompt_len
        processed_prompt[output_column] = next(output_token_sampler)

        processed_prompts.append(processed_prompt)

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
