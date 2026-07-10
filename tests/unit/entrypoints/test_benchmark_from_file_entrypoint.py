import filecmp
import os
import unittest
from pathlib import Path

import pytest

from guidellm.benchmark import reimport_benchmarks_report
from guidellm.benchmark.schemas import BenchmarkScenario, GenerativeBenchmarksReport

# Set to true to re-write the expected output.
REGENERATE_ARTIFACTS = False


@pytest.fixture
def get_test_asset_dir():
    def _() -> Path:
        return Path(__file__).parent / "assets"

    return _


@pytest.fixture
def cleanup():
    to_delete: list[Path] = []
    yield to_delete
    for item in to_delete:
        if item.exists():
            item.unlink()  # Deletes the file


@pytest.mark.skip(reason="currently broken")
def test_display_entrypoint_json(capfd, get_test_asset_dir):
    generic_test_display_entrypoint(
        "benchmarks_stripped.json",
        capfd,
        get_test_asset_dir,
    )


@pytest.mark.skip(reason="currently broken")
def test_display_entrypoint_yaml(capfd, get_test_asset_dir):
    generic_test_display_entrypoint(
        "benchmarks_stripped.yaml",
        capfd,
        get_test_asset_dir,
    )


def generic_test_display_entrypoint(filename, capfd, get_test_asset_dir):
    os.environ["COLUMNS"] = "180"  # CLI output depends on terminal width.
    asset_dir = get_test_asset_dir()
    reimport_benchmarks_report(asset_dir / filename, None)
    out, err = capfd.readouterr()
    expected_output_path = asset_dir / "benchmarks_stripped_output.txt"
    if REGENERATE_ARTIFACTS:
        expected_output_path.write_text(out)
        # Fail to prevent accidentally leaving regeneration mode on
        pytest.fail("Test bypassed to regenerate output")
    else:
        with expected_output_path.open(encoding="utf_8") as file:
            expected_output = file.read()
        assert out == expected_output


@pytest.mark.skip(reason="currently broken")
def test_reexporting_benchmark(get_test_asset_dir, cleanup):
    asset_dir = get_test_asset_dir()
    source_file = asset_dir / "benchmarks_stripped.json"
    exported_file = asset_dir / "benchmarks_reexported.json"
    # If you need to inspect the output to see why it failed, comment out
    # the cleanup statement.
    cleanup.append(exported_file)
    if exported_file.exists():
        exported_file.unlink()
    reimport_benchmarks_report(source_file, exported_file)
    # The reexported file should exist and be identical to the source.
    assert exported_file.exists()
    assert filecmp.cmp(source_file, exported_file, shallow=False)


@pytest.mark.asyncio
async def test_reimport_with_console_format_does_not_raise(tmp_path):
    """
    Regression test for https://github.com/vllm-project/guidellm/issues/925

    `reimport_benchmarks_report` used to unconditionally attach an output path
    to every requested format, but console output does not accept a path,
    raising a pydantic ValidationError.
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
    report = GenerativeBenchmarksReport(config=config)
    report_file = tmp_path / "benchmarks.json"
    report.save_file(report_file)

    _, results = await reimport_benchmarks_report(
        report_file, tmp_path, ("console", "json")
    )

    assert results["console"] == "printed to console"
    assert results["json"].exists()


if __name__ == "__main__":
    unittest.main()
