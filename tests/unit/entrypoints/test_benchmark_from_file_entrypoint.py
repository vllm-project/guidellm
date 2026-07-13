import filecmp
import os
import unittest
from pathlib import Path

import pytest

from guidellm.benchmark import reimport_benchmarks_report

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
    reimport_benchmarks_report(source_file, {"kind": "json", "path": exported_file})
    # The reexported file should exist and be identical to the source.
    assert exported_file.exists()
    assert filecmp.cmp(source_file, exported_file, shallow=False)


if __name__ == "__main__":
    unittest.main()
