import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from guidellm.extras.audio import encode_audio


@pytest.fixture
def sample_audio_tensor():
    sample_rate = 16000
    t = torch.linspace(0, 1, sample_rate)
    return 0.3 * torch.sin(2 * np.pi * 440 * t).unsqueeze(0)


@pytest.fixture
def sample_wav_file(sample_audio_tensor):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"fake_wav_content")
        temp_path = Path(f.name)
    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def real_wav_file():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        with wave.open(f.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


def test_encode_audio_with_tensor_input(sample_audio_tensor):
    result = encode_audio(
        audio=sample_audio_tensor,
        sample_rate=16000,
        audio_format="mp3",
        bitrate="64k",
    )

    assert result["type"] == "audio_file"
    assert isinstance(result["audio"], bytes)
    assert result["format"] == "mp3"
    assert result["mimetype"] == "audio/mp3"
    assert result["audio_samples"] == 16000
    assert result["audio_seconds"] == 1.0
    assert isinstance(result["audio_bytes"], int)
    assert result["audio_bytes"] > 0


def test_encode_audio_with_numpy_array(sample_audio_tensor):
    numpy_audio = sample_audio_tensor.numpy()

    result = encode_audio(audio=numpy_audio, sample_rate=16000)

    assert result["type"] == "audio_file"
    assert isinstance(result["audio"], bytes)
    assert result["audio_bytes"] > 0


def test_encode_audio_with_real_file_path(real_wav_file):
    result = encode_audio(audio=real_wav_file, sample_rate=16000, max_duration=1.0)

    assert result["type"] == "audio_file"
    assert isinstance(result["audio"], bytes)
    assert result["format"] == "mp3"
    assert result["mimetype"] == "audio/mp3"
    assert result["file_name"] == Path(real_wav_file).name
    assert result["audio_bytes"] > 0
    assert result["audio_seconds"] <= 1.0


def test_encode_audio_with_dict_input_complete():
    audio_dict = {"data": torch.randn(1, 16000), "sample_rate": 16000}

    result = encode_audio(audio=audio_dict)

    assert result["type"] == "audio_file"
    assert result["audio_bytes"] > 0
    assert result["audio_samples"] == 16000
    assert result["audio_seconds"] == 1.0


@patch("httpx.get")
@patch("guidellm.extras.audio._encode_audio")
def test_encode_audio_with_url(mock_http_get, sample_audio_tensor):
    # mock http get response
    mock_response = MagicMock()
    mock_response.content = b"fake_audio_content"
    mock_response.raise_for_status = MagicMock()
    mock_http_get.return_value = mock_response

    # mock decode - return sample audio tensor
    with patch("guidellm.extras.audio._decode_audio") as mock_decoder:
        mock_audio_result = MagicMock()
        mock_audio_result.data = sample_audio_tensor
        mock_audio_result.sample_rate = 16000
        mock_decoder.return_value = mock_audio_result

        result = encode_audio(audio="https://example.com/audio.wav", sample_rate=16000)
        assert result["type"] == "audio_file"


def test_encode_audio_with_max_duration(sample_audio_tensor):
    long_audio = torch.randn(1, 32000)

    result = encode_audio(audio=long_audio, sample_rate=16000, max_duration=1.0)

    assert result["audio_seconds"] == 1.0


def test_encode_audio_different_formats(sample_audio_tensor):
    formats = ["mp3", "wav", "flac"]

    for fmt in formats:
        result = encode_audio(
            audio=sample_audio_tensor, sample_rate=16000, audio_format=fmt
        )

        assert result["format"] == fmt
        assert result["mimetype"] == f"audio/{fmt}"
        assert result["audio_bytes"] > 0


def test_encode_audio_resampling(sample_audio_tensor):
    original_rate = 16000
    target_rate = 8000

    result = encode_audio(
        audio=sample_audio_tensor,
        sample_rate=original_rate,
        encode_sample_rate=target_rate,
    )

    assert "audio_samples" in result


def test_encode_audio_error_handling():
    with pytest.raises(ValueError):
        encode_audio(audio=123)

    with pytest.raises(ValueError):
        encode_audio(audio=torch.randn(1, 16000), sample_rate=None)

    with pytest.raises(ValueError):
        encode_audio(audio="/nonexistent/path/audio.wav")


def test_audio_quality_preservation(sample_audio_tensor):
    result = encode_audio(
        audio=sample_audio_tensor,
        sample_rate=16000,
        audio_format="mp3",
        bitrate="128k",
    )

    assert len(result["audio"]) > 1000


def test_end_to_end_audio_processing(sample_audio_tensor):
    original_samples = sample_audio_tensor.shape[1]
    original_duration = original_samples / 16000

    result = encode_audio(
        audio=sample_audio_tensor,
        sample_rate=16000,
        audio_format="mp3",
        bitrate="64k",
        max_duration=0.5,
    )

    assert result["type"] == "audio_file"
    assert isinstance(result["audio"], bytes)
    assert result["format"] == "mp3"
    assert result["audio_samples"] == 16000
    assert result["audio_seconds"] == min(original_duration, 0.5)
