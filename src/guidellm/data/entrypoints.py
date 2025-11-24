from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

from guidellm.data import builders
from guidellm.data.builders import ShortPromptStrategy


def process_dataset(
    data: str | Path,
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
    builders.process_dataset(
        data, output_path, processor, config, processor_args, data_args,
        data_column_mapper, short_prompt_strategy, pad_char, concat_delimiter,
        include_prefix_in_token_count, push_to_hub, hub_dataset_id, random_seed,
    )
