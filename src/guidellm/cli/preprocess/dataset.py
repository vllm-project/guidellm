"""Dataset preprocessing command."""

from __future__ import annotations

import click

import guidellm.utils.cli as cli_tools
from guidellm.data import ShortPromptStrategy, process_dataset

__all__ = ["dataset"]


@click.command(
    "dataset",
    help=(
        "Process a dataset to have specific prompt and output token sizes. "
        "Supports multiple strategies for handling prompts and optional "
        "Hugging Face Hub upload.\n\n"
        "DATA: Path to the input dataset or dataset ID.\n\n"
        "OUTPUT_PATH: Path to save the processed dataset, including file suffix."
    ),
    context_settings={"auto_envvar_prefix": "GUIDELLM"},
)
@click.argument(
    "data",
    callback=cli_tools.parse_arguments,
    required=True,
)
@click.argument(
    "output_path",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, resolve_path=True),
    required=True,
)
@click.option(
    "--processor",
    type=str,
    required=True,
    help="Processor or tokenizer name for calculating token counts.",
)
@click.option(
    "--config",
    type=str,
    required=True,
    help=(
        "PreprocessDatasetConfig as JSON string, key=value pairs, "
        "or file path (.json, .yaml, .yml, .config). "
        "Example: 'prompt_tokens=100,output_tokens=50,prefix_tokens_max=10'"
        ' or \'{"prompt_tokens": 100, "output_tokens": 50, '
        '"prefix_tokens_max": 10}\''
    ),
)
@click.option(
    "--processor-args",
    default=None,
    callback=cli_tools.parse_arguments,
    help="JSON string of arguments to pass to the processor constructor.",
)
@click.option(
    "--data-args",
    callback=cli_tools.parse_arguments,
    help="JSON string of arguments to pass to dataset creation.",
)
@click.option(
    "--data-column-mapper",
    default={"kind": "generative_column_mapper"},
    callback=cli_tools.parse_arguments,
    help="JSON string of column mappings to apply to the dataset.",
)
@click.option(
    "--short-prompt-strategy",
    type=click.Choice([s.value for s in ShortPromptStrategy]),
    default=ShortPromptStrategy.IGNORE.value,
    show_default=True,
    help="Strategy for handling prompts shorter than target length.",
)
@click.option(
    "--pad-char",
    type=str,
    default="",
    callback=cli_tools.decode_escaped_str,
    help="Character to pad short prompts with when using 'pad' strategy.",
)
@click.option(
    "--concat-delimiter",
    type=str,
    default="",
    help=(
        "Delimiter for concatenating short prompts (used with 'concatenate' strategy)."
    ),
)
@click.option(
    "--include-prefix-in-token-count",
    is_flag=True,
    default=False,
    help="Include prefix tokens in prompt token count calculation.",
)
@click.option(
    "--push-to-hub",
    is_flag=True,
    help="Push the processed dataset to Hugging Face Hub.",
)
@click.option(
    "--hub-dataset-id",
    type=str,
    default=None,
    help=("Hugging Face Hub dataset ID for upload (required if --push-to-hub is set)."),
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducible token sampling.",
)
def dataset(
    data,
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
):
    process_dataset(
        data=data,
        output_path=output_path,
        processor=processor,
        config=config,
        processor_args=processor_args,
        data_args=data_args,
        data_column_mapper=data_column_mapper,
        short_prompt_strategy=short_prompt_strategy,
        pad_char=pad_char,
        concat_delimiter=concat_delimiter,
        include_prefix_in_token_count=include_prefix_in_token_count,
        push_to_hub=push_to_hub,
        hub_dataset_id=hub_dataset_id,
        random_seed=random_seed,
    )
