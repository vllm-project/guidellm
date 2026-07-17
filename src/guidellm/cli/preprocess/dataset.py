"""Dataset preprocessing command."""

from __future__ import annotations

import click
from pydantic import ValidationError

import guidellm.utils.cli as cli_tools
from guidellm.cli.preprocess.args import PreprocessDatasetArgs
from guidellm.data import process_dataset
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
@registry_options_from_model(model=PreprocessDatasetArgs)
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
        strategy=args.strategy,
        data_column_mapper=args.data_column_mapper,
        data_loader=args.data_loader,
        push_to_hub=push_to_hub,
        hub_dataset_id=hub_dataset_id,
        random_seed=args.seed.value,  # type: ignore[attr-defined]
    )
