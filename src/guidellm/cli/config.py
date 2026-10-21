"""GuideLLM command for writing benchmark configuration files."""

from __future__ import annotations

from pathlib import Path

import click
import yaml
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
@click.argument(
    "path",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
)
@common_run_options
def config(path: Path, **kwargs) -> None:
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
    if serialized.get("benchmarks") == [None]:
        serialized.pop("benchmarks")
    metadata = serialized.get("metadata")
    if isinstance(metadata, dict) and metadata.get("labels") == {}:
        metadata.pop("labels")
        if not metadata:
            serialized.pop("metadata")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(serialized, sort_keys=False),
        encoding="utf-8",
    )
