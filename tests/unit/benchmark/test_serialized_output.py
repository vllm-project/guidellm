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
    args = BenchmarkGenerativeTextArgs(
        target="http://localhost:8000/v1",
        data=["test_data.jsonl"],
    )
    return GenerativeBenchmarksReport(args=args)


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
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)
        assert "args" in parsed
