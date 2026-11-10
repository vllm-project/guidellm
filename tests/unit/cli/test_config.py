"""Tests for the ``guidellm config`` CLI command."""

from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner

from guidellm.__main__ import cli
from guidellm.schemas.benchmark import BenchmarkScenario


def _option_names(command: click.Command) -> set[str]:
    return {
        option
        for parameter in command.params
        if isinstance(parameter, click.Option)
        for option in parameter.opts
    }


@pytest.mark.regression
def test_config_and_run_share_options():
    """The config command exposes every option supported by run.

    ## WRITTEN BY AI ##
    """
    assert _option_names(cli.commands["config"]) == _option_names(cli.commands["run"])


@pytest.mark.regression
def test_config_writes_benchmark_scenario(tmp_path: Path):
    """Config writes a reloadable YAML representation of the benchmark scenario.

    ## WRITTEN BY AI ##
    """
    path = tmp_path / "benchmark.yaml"
    result = CliRunner().invoke(
        cli,
        [
            "config",
            str(path),
            "--backend",
            "kind=openai_http,target=http://localhost:8000",
            "--profile",
            "kind=constant,rate=1",
            "--data",
            "kind=synthetic_text,prompt_tokens=8",
            "--label",
            "environment=test",
        ],
    )

    assert result.exit_code == 0, result.output
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert scenario["spec"]["backend"]["target"] == "http://localhost:8000"
    assert scenario["spec"]["data"][0]["prompt_tokens"] == 8
    assert "metrics" not in scenario["spec"]
    assert "benchmarks" not in scenario
    assert scenario["metadata"]["labels"] == {"environment": "test"}
    loaded = BenchmarkScenario.create(scenario=path)
    assert loaded.spec.backend.target == "http://localhost:8000"


@pytest.mark.regression
def test_config_merges_multiple_scenarios_and_preserves_includes(tmp_path: Path):
    """Config overlays scenario files and records each input in metadata.

    ## WRITTEN BY AI ##
    """
    first_path = tmp_path / "base.yaml"
    first_metadata = {
        "name": "base",
        "description": "Base scenario",
        "labels": {"environment": "test"},
    }
    first_path.write_text(
        yaml.safe_dump(
            {
                "spec": {
                    "backend": {
                        "kind": "openai_http",
                        "target": "http://localhost:8000",
                    },
                    "profile": {"kind": "constant", "rate": 1},
                    "data": [{"kind": "synthetic_text", "prompt_tokens": 8}],
                },
                "metadata": first_metadata,
            }
        ),
        encoding="utf-8",
    )

    overlay_path = tmp_path / "overlay.yaml"
    overlay_metadata = {
        "name": "overlay",
        "description": "Overlay scenario",
        "labels": {"owner": "benchmark-team"},
    }
    overlay_path.write_text(
        yaml.safe_dump(
            {
                "spec": {
                    "backend": {"kind": "openai_http", "model": "test-model"},
                    "profile": {"kind": "constant", "rate": 2},
                },
                "metadata": overlay_metadata,
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "merged.yaml"
    result = CliRunner().invoke(
        cli,
        [
            "config",
            str(output_path),
            "--config",
            str(first_path),
            "--config",
            str(overlay_path),
        ],
    )

    assert result.exit_code == 0, result.output
    scenario = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert scenario["spec"]["backend"]["target"] == "http://localhost:8000"
    assert scenario["spec"]["backend"]["model"] == "test-model"
    assert scenario["spec"]["profile"]["rate"] == [2.0]
    assert scenario["spec"]["data"][0]["prompt_tokens"] == 8
    assert scenario["metadata"]["includes"] == [
        {"scenario": first_path.name, "metadata": first_metadata},
        {"scenario": overlay_path.name, "metadata": overlay_metadata},
    ]

    loaded = BenchmarkScenario.create(scenario=output_path)
    assert loaded.spec.backend.target == "http://localhost:8000"
    assert loaded.spec.backend.model == "test-model"
