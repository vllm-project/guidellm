import base64
import json
from pathlib import Path

import pytest
import yaml

from guidellm.backends.openai.request_handlers import AudioRequestHandler
from guidellm.benchmark.schemas.report import GenerativeBenchmarksReport
from guidellm.scheduler import ConcurrentStrategy
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    RequestInfo,
)
from guidellm.schemas.backends import OpenAIHTTPBackendArgs
from tests.unit.benchmark.html_report_fixtures import (
    make_benchmark,
    make_scenario,
    report,
)


@pytest.fixture
def audio_report() -> tuple[GenerativeBenchmarksReport, bytes]:
    """Build a report from an actual multipart request and transcription response.

    ## WRITTEN BY AI ##
    """
    audio = b"\x00\xff\x80" * 4096
    request = GenerationRequest(
        request_id="audio",
        columns={
            "audio_column": [
                {"audio": audio, "file_name": "sample.wav", "mimetype": "audio/wav"}
            ]
        },
    )
    handler = AudioRequestHandler()
    arguments = handler.format(request)
    response = handler.compile_non_streaming(
        request, arguments, {"text": "The transcription remains available."}
    )
    info = RequestInfo(request_id=request.request_id, status="completed")
    info.timings.request_start = 1.0
    info.timings.request_end = 2.0
    info.timings.resolve_end = 2.0
    stats = response.compile_stats(request, info)
    benchmark = make_benchmark(
        strategy=ConcurrentStrategy(streams=1), rps=1.0, tps=0.0, requests=[stats]
    )
    return report(benchmark), audio


@pytest.mark.regression
@pytest.mark.parametrize("format_type", ["json", "yaml"])
def test_media_omission_preserves_transcription_and_independent_lossless_output(
    tmp_path: Path, audio_report, format_type: str
):
    """Omit multipart media without changing the source report or a later output.

    ## WRITTEN BY AI ##
    """
    source, audio = audio_report
    before = source.model_dump(mode="json")
    omitted_path = source.save_file(
        tmp_path / f"omitted.{format_type}", include_media=False
    )
    lossless_path = source.save_file(tmp_path / f"lossless.{format_type}")
    omitted = yaml.safe_load(omitted_path.read_text())
    lossless = yaml.safe_load(lossless_path.read_text())
    omitted_stats = omitted["benchmarks"][0]["requests"]["successful"][0]
    lossless_stats = lossless["benchmarks"][0]["requests"]["successful"][0]
    encoded = base64.urlsafe_b64encode(audio).decode()

    assert json.loads(lossless_stats["request_args"])["files"]["file"][1] == encoded
    assert encoded not in omitted_path.read_text()
    assert json.loads(omitted_stats["request_args"])["files"]["file"] == {
        "omitted": True,
        "serialized_characters": len(encoded),
        "filename": "sample.wav",
        "media_type": "audio/wav",
    }
    assert omitted_stats["output"] == lossless_stats["output"]
    assert omitted_stats["output"] == "The transcription remains available."
    assert source.model_dump(mode="json") == before
    restored = GenerativeBenchmarksReport.load_file(omitted_path)
    assert (
        restored.benchmarks[0].requests.successful[0].output == omitted_stats["output"]
    )


@pytest.mark.regression
@pytest.mark.parametrize(
    ("content", "path"),
    [
        (
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAEC"}},
            ("image_url", "url"),
        ),
        (
            {"type": "input_audio", "input_audio": {"data": "AAEC", "format": "wav"}},
            ("input_audio", "data"),
        ),
        (
            {"type": "input_file", "file_data": "AAEC", "filename": "x.pdf"},
            ("file_data",),
        ),
        ({"type": "audio", "data": "AAEC"}, ("data",)),
        ({"b64_json": "AAEC"}, ("b64_json",)),
    ],
)
def test_media_omission_covers_inline_payloads_in_configuration(
    tmp_path: Path, content, path
):
    """Remove explicit inline encodings even when configuration retains a copy.

    ## WRITTEN BY AI ##
    """
    scenario = make_scenario()
    scenario.spec.backend = OpenAIHTTPBackendArgs(
        target="http://localhost:8000",
        model="test-model",
        extras=GenerationRequestArguments(
            body={
                "content": [content],
                "text": "Keep this prompt.",
                "remote": "https://example.com/image.png",
                "user_data": {"type": {"custom": "tag"}, "data": "ordinary text"},
            }
        ),
    )
    source = report(scenario=scenario)
    saved = source.save_file(tmp_path / "report.json", include_media=False)
    body = json.loads(saved.read_text())["config"]["spec"]["backend"]["extras"]["body"]
    metadata = body["content"][0]
    for key in path:
        metadata = metadata[key]

    assert metadata["omitted"] is True
    assert metadata["size_bytes"] == 3
    assert "AAEC" not in saved.read_text()
    assert body["text"] == "Keep this prompt."
    assert body["remote"] == "https://example.com/image.png"
    assert body["user_data"] == {"type": {"custom": "tag"}, "data": "ordinary text"}


@pytest.mark.regression
def test_media_omission_preserves_opaque_arguments(tmp_path: Path, audio_report):
    """Leave an external backend's non-JSON argument description intact.

    ## WRITTEN BY AI ##
    """
    source, _ = audio_report
    source.benchmarks[0].requests.successful[0].request_args = "an opaque description"
    saved = source.save_file(tmp_path / "report.json", include_media=False)
    restored = GenerativeBenchmarksReport.load_file(saved)
    assert (
        restored.benchmarks[0].requests.successful[0].request_args
        == "an opaque description"
    )
