from __future__ import annotations

import click
from pydantic import ValidationError

import guidellm.utils.cli as cli_tools
from guidellm.benchmark.schemas.entrypoints import BenchmarkArgs

__all__ = [
    "run",
]


@click.command(
    "run",
    help=(
        "Run a benchmark against a generative model. "
        "Supports multiple backends, data sources, strategies, and output formats. "
        "Configuration can be loaded from a scenario file or specified via options."
    ),
)
def run(**kwargs):  # noqa: C901, PLR0915
    ctx = click.get_current_context()
    # Only set CLI args that differ from click defaults
    kwargs = cli_tools.set_if_not_default(ctx, **kwargs)

    try:
        args = BenchmarkArgs.create(**kwargs)
    except ValidationError as err:
        # Translate pydantic validation error to click argument error
        errs = err.errors(include_url=False, include_context=True, include_input=True)
        first_loc = errs[0]["loc"]
        top_field = str(first_loc[0]) if first_loc else ""
        param_name = "--" + top_field.replace("_", "-")
        raise click.BadParameter(
            cli_tools.format_validation_errors(errs), ctx=ctx, param_hint=param_name
        ) from err

    click.echo(f"Running benchmark with args: {args}")
