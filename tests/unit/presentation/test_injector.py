from pathlib import Path

import pytest
from pydantic import BaseModel

from guidellm.config import settings
from guidellm.presentation.injector import create_report, inject_data


class ExampleModel(BaseModel):
    name: str
    version: str


@pytest.mark.smoke
def test_inject_data():
    html = "<head><script>window.runInfo = {};</script></head>"
    expected_html = (
        "<head><script>"
        "window.runInfo ="
        '{ "model": { "name": "neuralmagic/Qwen2.5-7B-quantized.w8a8" } };'
        "</script></head>"
    )
    js_data = {
        "window.runInfo = {};": "window.runInfo ="
        '{ "model": { "name": "neuralmagic/Qwen2.5-7B-quantized.w8a8" } };'
    }
    result = inject_data(
        js_data,
        html,
    )
    assert result == expected_html


@pytest.mark.smoke
def test_create_report_to_file(tmpdir):
    js_data = {
        "window.runInfo = {};": "window.runInfo ="
        '{ "model": { "name": "neuralmagic/Qwen2.5-7B-quantized.w8a8" } };'
    }
    html_content = "<head><script>window.runInfo = {};</script></head>"
    expected_html_content = (
        "<head><script>"
        "window.runInfo ="
        '{ "model": { "name": "neuralmagic/Qwen2.5-7B-quantized.w8a8" } };'
        "</script></head>"
    )

    mock_html_path = tmpdir.join("template.html")
    mock_html_path.write(html_content)
    settings.report_generation.source = str(mock_html_path)

    output_path = tmpdir.join("output.html")
    result_path = create_report(js_data, str(output_path))
    result_content = result_path.read_text()

    assert result_path == output_path
    assert result_content == expected_html_content


@pytest.mark.smoke
def test_create_report_with_file_nested_in_dir(tmpdir):
    js_data = {
        "window.runInfo = {};": "window.runInfo ="
        '{ "model": { "name": "neuralmagic/Qwen2.5-7B-quantized.w8a8" } };'
    }
    html_content = "<head><script>window.runInfo = {};</script></head>"
    expected_html_content = (
        "<head><script>"
        "window.runInfo ="
        '{ "model": { "name": "neuralmagic/Qwen2.5-7B-quantized.w8a8" } };'
        "</script></head>"
    )

    output_dir = tmpdir.mkdir("output_dir")
    mock_html_path = tmpdir.join("template.html")
    mock_html_path.write(html_content)
    settings.report_generation.source = str(mock_html_path)

    output_path = Path(output_dir) / "report.html"
    result_path = create_report(js_data, str(output_path))

    with Path(result_path).open("r") as file:
        result_content = file.read()

    assert result_path == output_path
    assert result_content == expected_html_content
