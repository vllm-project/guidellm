"""Dataset preprocessing command."""

from __future__ import annotations

import click
from pydantic import ValidationError

import guidellm.utils.cli as cli_tools
from guidellm.cli.preprocess.args import PreprocessDatasetArgs
from guidellm.data import ShortPromptStrategy, process_dataset
from guidellm.data.schemas import DataArgs
from guidellm.utils.click_pydantic import (
    format_validation_errors,
    registry_options_from_model,
)

__all__ = ["dataset"]


@click.command(
    "dataset",
    help=(
        "Process a dataset to have specific prompt and output token sizes. "
        "Supports multiple strategies for handling prompts and optional "
        "Hugging Face Hub upload.\n\n"
        "DATA: Dataset descriptor (kind=<type>,...). "
        "Supports the same data kinds as ``guidellm run --data``.\n\n"
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
@registry_options_from_model(model=PreprocessDatasetArgs)
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
def dataset(
    data,
    output_path,
    config,
    short_prompt_strategy,
    pad_char,
    concat_delimiter,
    include_prefix_in_token_count,
    push_to_hub,
    hub_dataset_id,
    **kwargs,
):
    ctx = click.get_current_context()

    try:
        data_config = DataArgs.model_validate(data)
    except ValidationError as err:
        raise format_validation_errors(ctx, err, base_class=DataArgs) from err

    try:
        args = PreprocessDatasetArgs.model_validate(kwargs)
    except ValidationError as err:
        raise format_validation_errors(
            ctx, err, base_class=PreprocessDatasetArgs
        ) from err

    process_dataset(
        data=data_config,
        output_path=output_path,
        tokenizer=args.tokenizer,
        config=config,
        data_column_mapper=args.data_column_mapper,
        short_prompt_strategy=short_prompt_strategy,
        pad_char=pad_char,
        concat_delimiter=concat_delimiter,
        include_prefix_in_token_count=include_prefix_in_token_count,
        push_to_hub=push_to_hub,
        hub_dataset_id=hub_dataset_id,
        random_seed=args.seed.value,  # type: ignore[attr-defined]
    )
