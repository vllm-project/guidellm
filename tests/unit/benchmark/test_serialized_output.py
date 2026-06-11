import json
from pathlib import Path

import pytest
import yaml

from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.outputs.serialized import GenerativeBenchmarkerSerialized
from guidellm.benchmark.schemas import (
    BenchmarkGenerativeTextArgs,
    GenerativeBenchmarksReport,
)


@pytest.fixture
def minimal_report() -> GenerativeBenchmarksReport:
    """
    Create a minimal GenerativeBenchmarksReport for testing serialization.

    ## WRITTEN BY AI ##
    """
    args = BenchmarkGenerativeTextArgs.model_validate(
        {
            "backend_kwargs": {
                "kind": "openai_http",
                "target": "http://localhost:8000/v1",
                "model": "test-model",
            },
            "data": [{"kind": "huggingface", "source": "test_data.jsonl"}],
            "profile": {"kind": "sweep", "rate": [10.0]},
            "tokenizer": {"kind": "huggingface_auto", "model": "test-model"},
            "data_column_mapper": {"kind": "generative_column_mapper"},
            "data_preprocessors": [],
            "data_finalizer": {"kind": "generative"},
            "data_loader": {"kind": "pytorch"},
        }
    )
    return GenerativeBenchmarksReport(args=args)


@pytest.fixture
def file_path_report(tmp_path: Path) -> GenerativeBenchmarksReport:
    """
    Create a report whose benchmark args contain nested Path values.

    ## WRITTEN BY AI ##
    """
    args = BenchmarkGenerativeTextArgs.model_validate(
        {
            "backend_kwargs": {
                "kind": "openai_http",
                "target": "http://localhost:8000/v1",
                "model": "test-model",
            },
            "profile": {"kind": "sweep"},
            "data": [
                {
                    "kind": "json_file",
                    "path": tmp_path / "data.jsonl",
                    "load_kwargs": {"split": "train"},
                }
            ],
            "tokenizer": {"kind": "huggingface_auto", "model": "test-model"},
            "data_column_mapper": {"kind": "generative_column_mapper"},
            "data_preprocessors": [],
            "data_finalizer": {"kind": "generative"},
            "data_loader": {"kind": "pytorch"},
            "output_dir": tmp_path / "outputs",
        }
    )
    return GenerativeBenchmarksReport(args=args)


@pytest.fixture
def report_with_extras() -> GenerativeBenchmarksReport:
    """
    Create a GenerativeBenchmarksReport with output_extras set.

    ## WRITTEN BY AI ##
    """
    extras = {"tag": "prod-test", "hardware": "A100", "metadata": {"count": 8}}
    args = BenchmarkGenerativeTextArgs(
        backend_kwargs={
            "type": "openai_http",
            "target": "http://localhost:8000/v1",
            "model": "test-model",
        },
        data=["test_data.jsonl"],
        output_extras=extras,
    )
    return GenerativeBenchmarksReport(args=args, extras=extras)


class TestResolveFormatType:
    """
    Tests that resolve() propagates the format key correctly.

    ## WRITTEN BY AI ##
    """

    def test_resolve_yaml_alias_sets_format_type(self, tmp_path: Path):
        """
        Resolving with the bare 'yaml' alias should produce a handler
        with format_type='yaml'.

        ## WRITTEN BY AI ##
        """
        resolved = GenerativeBenchmarkerOutput.resolve(
            outputs=["yaml"], output_dir=tmp_path
        )
        handler = resolved["yaml"]
        assert isinstance(handler, GenerativeBenchmarkerSerialized)
        assert handler.format_type == "yaml"

    def test_resolve_json_alias_sets_format_type(self, tmp_path: Path):
        """
        Resolving with the bare 'json' alias should produce a handler
        with format_type='json'.

        ## WRITTEN BY AI ##
        """
        resolved = GenerativeBenchmarkerOutput.resolve(
            outputs=["json"], output_dir=tmp_path
        )
        handler = resolved["json"]
        assert isinstance(handler, GenerativeBenchmarkerSerialized)
        assert handler.format_type == "json"

    def test_resolve_explicit_yaml_filename(self, tmp_path: Path):
        """
        Resolving with an explicit filename like 'results.yaml' should produce
        a handler whose output_path includes the filename.

        ## WRITTEN BY AI ##
        """
        resolved = GenerativeBenchmarkerOutput.resolve(
            outputs=["results.yaml"], output_dir=tmp_path
        )
        handler = resolved["yaml"]
        assert isinstance(handler, GenerativeBenchmarkerSerialized)
        assert handler.output_path == tmp_path / "results.yaml"


class TestFinalizeFormats:
    """
    Tests that finalize() writes the correct file format.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.asyncio
    async def test_finalize_yaml_writes_yaml_file(
        self, tmp_path: Path, minimal_report: GenerativeBenchmarksReport
    ):
        """
        A handler with format_type='yaml' should write a .yaml file
        containing valid YAML content.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerSerialized(
            output_path=tmp_path, format_type="yaml"
        )
        result_path = await handler.finalize(minimal_report)

        assert result_path.suffix == ".yaml"
        assert result_path.name == "benchmarks.yaml"
        assert result_path.exists()

        content = result_path.read_text()
        # JSON is valid YAML, so verify the output is not JSON
        # to confirm YAML was actually written
        with pytest.raises(json.JSONDecodeError):
            json.loads(content)
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)
        assert "args" in parsed

    @pytest.mark.asyncio
    async def test_finalize_json_writes_json_file(
        self, tmp_path: Path, minimal_report: GenerativeBenchmarksReport
    ):
        """
        A handler with format_type='json' should write a .json file
        containing valid JSON content.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerSerialized(
            output_path=tmp_path, format_type="json"
        )
        result_path = await handler.finalize(minimal_report)

        assert result_path.suffix == ".json"
        assert result_path.name == "benchmarks.json"
        assert result_path.exists()

        content = result_path.read_text()
        parsed = json.loads(content)
        assert isinstance(parsed, dict)
        assert "args" in parsed

    @pytest.mark.asyncio
    async def test_finalize_json_serializes_nested_paths(
        self, tmp_path: Path, file_path_report: GenerativeBenchmarksReport
    ):
        """
        JSON report serialization should convert nested Path values to strings.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerSerialized(
            output_path=tmp_path, format_type="json"
        )
        result_path = await handler.finalize(file_path_report)

        parsed = json.loads(result_path.read_text())
        assert parsed["args"]["data"][0]["path"] == str(tmp_path / "data.jsonl")
        assert parsed["args"]["output_dir"] == str(tmp_path / "outputs")

    @pytest.mark.asyncio
    async def test_finalize_yaml_serializes_nested_paths(
        self, tmp_path: Path, file_path_report: GenerativeBenchmarksReport
    ):
        """
        YAML report serialization should emit nested Path values as safe strings.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerSerialized(
            output_path=tmp_path, format_type="yaml"
        )
        result_path = await handler.finalize(file_path_report)

        parsed = yaml.safe_load(result_path.read_text())
        assert parsed["args"]["data"][0]["path"] == str(tmp_path / "data.jsonl")
        assert parsed["args"]["output_dir"] == str(tmp_path / "outputs")

    @pytest.mark.asyncio
    async def test_finalize_explicit_path_respects_extension(
        self, tmp_path: Path, minimal_report: GenerativeBenchmarksReport
    ):
        """
        When output_path is an explicit file (not a directory), finalize
        should use that path directly regardless of format_type.

        ## WRITTEN BY AI ##
        """
        explicit_path = tmp_path / "my_report.yaml"
        handler = GenerativeBenchmarkerSerialized(output_path=explicit_path)
        result_path = await handler.finalize(minimal_report)

        assert result_path == explicit_path
        assert result_path.exists()

        content = result_path.read_text()
        # JSON is valid YAML, so verify the output is not JSON
        # to confirm YAML was actually written
        with pytest.raises(json.JSONDecodeError):
            json.loads(content)
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)
        assert "args" in parsed


@pytest.mark.smoke
class TestExtrasInReport:
    """
    Tests that the extras field is present and correct in serialized output.

    ## WRITTEN BY AI ##
    """

    def test_report_extras_none_by_default(
        self, minimal_report: GenerativeBenchmarksReport
    ):
        """
        extras field should be None when not supplied to the report.

        ## WRITTEN BY AI ##
        """
        assert minimal_report.extras is None

    def test_report_extras_stored(self, report_with_extras: GenerativeBenchmarksReport):
        """
        extras field should hold the dict passed during construction.

        ## WRITTEN BY AI ##
        """
        assert report_with_extras.extras == {
            "tag": "prod-test",
            "hardware": "A100",
            "metadata": {"count": 8},
        }


@pytest.mark.sanity
class TestExtrasSerializedToFile:
    """
    Tests that extras appear correctly in JSON and YAML output files.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.asyncio
    async def test_extras_present_in_json_output(
        self, tmp_path: Path, report_with_extras: GenerativeBenchmarksReport
    ):
        """
        extras dict should appear under the 'extras' key in the written JSON file.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerSerialized(
            output_path=tmp_path, format_type="json"
        )
        result_path = await handler.finalize(report_with_extras)

        parsed = json.loads(result_path.read_text())
        assert "extras" in parsed
        assert parsed["extras"] == {
            "tag": "prod-test",
            "hardware": "A100",
            "metadata": {"count": 8},
        }

    @pytest.mark.asyncio
    async def test_extras_present_in_yaml_output(
        self, tmp_path: Path, report_with_extras: GenerativeBenchmarksReport
    ):
        """
        extras dict should appear under the 'extras' key in the written YAML file.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerSerialized(
            output_path=tmp_path, format_type="yaml"
        )
        result_path = await handler.finalize(report_with_extras)

        parsed = yaml.safe_load(result_path.read_text())
        assert "extras" in parsed
        assert parsed["extras"] == {
            "tag": "prod-test",
            "hardware": "A100",
            "metadata": {"count": 8},
        }

    @pytest.mark.asyncio
    async def test_extras_null_in_json_when_not_set(
        self, tmp_path: Path, minimal_report: GenerativeBenchmarksReport
    ):
        """
        extras should serialize as null in JSON when not provided.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerSerialized(
            output_path=tmp_path, format_type="json"
        )
        result_path = await handler.finalize(minimal_report)

        parsed = json.loads(result_path.read_text())
        assert "extras" in parsed
        assert parsed["extras"] is None
