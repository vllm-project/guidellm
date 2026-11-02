"""GuideLLM command for writing benchmark configuration files."""

from __future__ import annotations

import click
import yaml
from click.utils import LazyFile
from pydantic import ValidationError

import guidellm.utils.cli as cli_tools
from guidellm.schemas.benchmark import BenchmarkScenario
from guidellm.utils.click_pydantic import (
    RegistryAwareCommand,
    format_validation_errors,
)
from guidellm.utils.typing import BLANK

from .common import common_run_options

__all__ = ["config"]


@click.command(
    "config",
    cls=RegistryAwareCommand,
    help=(
        "Create a YAML benchmark configuration from the same options accepted "
        "by the run command."
    ),
)
@click.argument("output", type=click.File("w"), default="-")
@common_run_options
def config(output: LazyFile, **kwargs) -> None:
    """Write the configured benchmark scenario to ``PATH``."""
    ctx = click.get_current_context()
    kwargs = cli_tools.set_if_not_default(ctx, **kwargs)
    kwargs.pop("disable_console", None)
    kwargs.pop("disable_console_interactive", None)

    try:
        args = BenchmarkScenario.create(
            spec=kwargs.get("spec", {}),
            benchmarks=kwargs.get("benchmarks") or BLANK,
            metadata={"labels": dict(kwargs.get("labels", []))},
            scenario=kwargs.get("config"),
        )
    except ValidationError as err:
        raise format_validation_errors(ctx, err, base_class=BenchmarkScenario) from err

    # ``exclude_defaults`` can drop required fields from registry-backed union
    # models while serializing. ``exclude_unset`` keeps those fields and omits
    # values supplied by model defaults instead.
    serialized = args.model_dump(mode="json", exclude_unset=True)
    output.write(yaml.safe_dump(serialized, sort_keys=False))
