import filecmp
import json
import os

import pytest
import yaml
from trio import Path

from guidellm.benchmark import reimport_benchmarks_report
from guidellm.benchmark.outputs.serialized import (
    JSONBenchmarkOutputArgs,
    YAMLBenchmarkOutputArgs,
)
from guidellm.benchmark.schemas import (
    BenchmarkScenario,
    GenerativeBenchmarksReport,
)


@pytest.fixture
def sample_report() -> GenerativeBenchmarksReport:
    """
    Create a valid report that can be loaded by reimport_benchmarks_report.

    ## WRITTEN BY AI ##
    """
    config = BenchmarkScenario.model_validate(
        {
            "spec": {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000/v1",
                    "model": "test-model",
                },
                "data": [{"kind": "huggingface", "source": "test_data.jsonl"}],
                "profile": {"kind": "constant", "rate": 10.0},
            },
        }
    )
    return GenerativeBenchmarksReport(config=config)


@pytest.fixture
def saved_json_report(
    tmp_path: Path, sample_report: GenerativeBenchmarksReport
) -> Path:
    """
    Persist a sample report as JSON for from-file entrypoint tests.

    ## WRITTEN BY AI ##
    """
    return sample_report.save_file(tmp_path / "benchmarks.json")


@pytest.fixture
def saved_yaml_report(
    tmp_path: Path, sample_report: GenerativeBenchmarksReport
) -> Path:
    """
    Persist a sample report as YAML for from-file entrypoint tests.

    ## WRITTEN BY AI ##
    """
    return sample_report.save_file(tmp_path / "benchmarks.yaml")


@pytest.fixture
def report_directory(tmp_path: Path, sample_report: GenerativeBenchmarksReport) -> Path:
    """
    Directory containing the default benchmarks.json report file.

    ## WRITTEN BY AI ##
    """
    sample_report.save_file(tmp_path / "benchmarks.json")
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_loads_json_report(
    saved_json_report: Path, sample_report: GenerativeBenchmarksReport
):
    """
    Loading a JSON report returns the saved configuration and benchmark list.

    ## WRITTEN BY AI ##
    """
    report, output_results = await reimport_benchmarks_report(saved_json_report, [])

    assert output_results == {}
    assert report.config == sample_report.config
    assert report.benchmarks == []
    assert report.metadata.version == 2


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_loads_yaml_report(
    saved_yaml_report: Path, sample_report: GenerativeBenchmarksReport
):
    """
    Loading a YAML report returns the saved configuration and benchmark list.

    ## WRITTEN BY AI ##
    """
    report, output_results = await reimport_benchmarks_report(saved_yaml_report, [])

    assert output_results == {}
    assert report.config == sample_report.config
    assert report.benchmarks == []


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_loads_report_from_directory(
    report_directory: Path, sample_report: GenerativeBenchmarksReport
):
    """
    A directory path resolves to benchmarks.json before loading the report.

    ## WRITTEN BY AI ##
    """
    report, output_results = await reimport_benchmarks_report(report_directory, [])

    assert output_results == {}
    assert report.config == sample_report.config


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_console_output(capfd, saved_json_report: Path):
    """
    Console output is produced when the console output format is requested.

    ## WRITTEN BY AI ##
    """
    os.environ["COLUMNS"] = "180"

    _, output_results = await reimport_benchmarks_report(
        saved_json_report,
        [{"kind": "console"}],
    )

    out, err = capfd.readouterr()
    combined = out + err
    assert "Import of old benchmarks complete" in combined
    assert "Run Summary Info" in combined
    assert output_results == {"console": "printed to console"}


@pytest.mark.asyncio
@pytest.mark.regression
async def test_reexports_json(saved_json_report: Path, tmp_path: Path):
    """
    Re-exporting to JSON writes a file with the same report content.

    ## WRITTEN BY AI ##
    """
    exported_file = tmp_path / "benchmarks_reexported.json"

    report, output_results = await reimport_benchmarks_report(
        saved_json_report,
        [{"kind": "json", "path": exported_file}],
    )

    assert exported_file.exists()
    assert output_results["json"] == exported_file
    assert filecmp.cmp(saved_json_report, exported_file, shallow=False)

    reloaded = GenerativeBenchmarksReport.load_file(exported_file)
    assert reloaded.config == report.config
    assert reloaded.benchmarks == report.benchmarks


@pytest.mark.asyncio
@pytest.mark.regression
async def test_reexports_yaml(saved_yaml_report: Path, tmp_path: Path):
    """
    Re-exporting to YAML writes a file with the same report content.

    ## WRITTEN BY AI ##
    """
    exported_file = tmp_path / "benchmarks_reexported.yaml"

    report, output_results = await reimport_benchmarks_report(
        saved_yaml_report,
        [{"kind": "yaml", "path": exported_file}],
    )

    assert exported_file.exists()
    assert output_results["yaml"] == exported_file
    assert filecmp.cmp(saved_yaml_report, exported_file, shallow=False)

    reloaded = GenerativeBenchmarksReport.load_file(exported_file)
    assert reloaded.config == report.config
    assert reloaded.benchmarks == report.benchmarks


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_multiple_output_formats(saved_json_report: Path, tmp_path: Path):
    """
    Multiple output formats are finalized and returned in a single call.

    ## WRITTEN BY AI ##
    """
    exported_json = tmp_path / "benchmarks.json"
    exported_yaml = tmp_path / "benchmarks.yaml"

    report, output_results = await reimport_benchmarks_report(
        saved_json_report,
        [
            {"kind": "console"},
            {"kind": "json", "path": exported_json},
            {"kind": "yaml", "path": exported_yaml},
        ],
    )

    assert set(output_results) == {"console", "json", "yaml"}
    assert output_results["console"] == "printed to console"
    assert output_results["json"] == exported_json
    assert output_results["yaml"] == exported_yaml
    assert exported_json.exists()
    assert exported_yaml.exists()
    assert json.loads(exported_json.read_text()) == json.loads(
        saved_json_report.read_text()
    )
    parsed_yaml = yaml.safe_load(exported_yaml.read_text())
    assert isinstance(parsed_yaml, dict)
    assert parsed_yaml["config"] == json.loads(saved_json_report.read_text())["config"]
    assert report.benchmarks == []


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_outputs_accepts_tuple_of_args(saved_json_report: Path, tmp_path: Path):
    """
    Output arguments may be provided as validated BenchmarkOutputArgs instances.

    ## WRITTEN BY AI ##
    """
    exported_file = tmp_path / "benchmarks.json"

    _, output_results = await reimport_benchmarks_report(
        saved_json_report,
        (
            JSONBenchmarkOutputArgs(path=exported_file),
            YAMLBenchmarkOutputArgs(path=tmp_path / "benchmarks.yaml"),
        ),
    )

    assert set(output_results) == {"json", "yaml"}
    assert exported_file.exists()
    assert (tmp_path / "benchmarks.yaml").exists()
