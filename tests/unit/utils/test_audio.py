import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from guidellm.utils import audio as _audio_mod


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
    result = _audio_mod.encode_audio(
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
    """
    Numpy array input has no codec metadata, so format falls back to WAV.

    ## WRITTEN BY AI ##
    """
    numpy_audio = sample_audio_tensor.numpy()

    result = _audio_mod.encode_audio(audio=numpy_audio, sample_rate=16000)

    assert result["type"] == "audio_file"
    assert isinstance(result["audio"], bytes)
    assert result["format"] == "wav"
    assert result["audio_bytes"] > 0


def test_encode_audio_with_real_file_path(real_wav_file):
    """
    WAV file should be detected as pcm codec and re-encoded as WAV.

    ## WRITTEN BY AI ##
    """
    result = _audio_mod.encode_audio(
        audio=real_wav_file, sample_rate=16000, max_duration=1.0
    )

    assert result["type"] == "audio_file"
    assert isinstance(result["audio"], bytes)
    assert result["format"] == "wav"
    assert result["mimetype"] == "audio/wav"
    assert result["file_name"] == Path(real_wav_file).name
    assert result["audio_bytes"] > 0
    assert result["audio_seconds"] <= 1.0


def test_encode_audio_with_dict_input_complete():
    """
    Dict with raw 'data' key (float tensor) has no codec, falls back to WAV.

    ## WRITTEN BY AI ##
    """
    audio_dict = {"data": torch.randn(1, 16000), "sample_rate": 16000}

    result = _audio_mod.encode_audio(audio=audio_dict)

    assert result["type"] == "audio_file"
    assert result["format"] == "wav"
    assert result["audio_bytes"] > 0
    assert result["audio_samples"] == 16000
    assert result["audio_seconds"] == 1.0


@patch("guidellm.utils.audio._encode_audio")
@patch("guidellm.utils.audio._decode_audio")
def test_encode_audio_with_url(mock_decoder, mock_encode, sample_audio_tensor):
    """
    URL input: _decode_audio returns codec, format is derived from it.

    ## WRITTEN BY AI ##
    """
    mock_audio_result = MagicMock()
    mock_audio_result.sample_rate = 16000
    mock_audio_result.duration_seconds = 1.0
    mock_decoder.return_value = (mock_audio_result, "pcm_s16le")
    mock_encode.return_value = b"encoded_data"

    result = _audio_mod.encode_audio(
        audio="https://example.com/audio.wav", sample_rate=16000
    )
    assert result["type"] == "audio_file"
    assert result["format"] == "wav"
    assert result["file_name"] == "audio.wav"


def test_encode_audio_with_max_duration(sample_audio_tensor):
    long_audio = torch.randn(1, 32000)

    result = _audio_mod.encode_audio(
        audio=long_audio, sample_rate=16000, max_duration=1.0, audio_format="wav"
    )

    assert result["audio_seconds"] == 1.0


def test_encode_audio_different_formats(sample_audio_tensor):
    formats = ["mp3", "wav", "flac"]

    for fmt in formats:
        result = _audio_mod.encode_audio(
            audio=sample_audio_tensor, sample_rate=16000, audio_format=fmt
        )

        assert result["format"] == fmt
        assert result["mimetype"] == f"audio/{fmt}"
        assert result["audio_bytes"] > 0


def test_encode_audio_resampling(sample_audio_tensor):
    original_rate = 16000
    target_rate = 8000

    result = _audio_mod.encode_audio(
        audio=sample_audio_tensor,
        sample_rate=original_rate,
        encode_sample_rate=target_rate,
        audio_format="wav",
    )

    assert "audio_samples" in result


def test_encode_audio_error_handling():
    with pytest.raises(ValueError):
        _audio_mod.encode_audio(audio=123)

    with pytest.raises(ValueError):
        _audio_mod.encode_audio(audio=torch.randn(1, 16000), sample_rate=None)

    with pytest.raises(ValueError):
        _audio_mod.encode_audio(audio="/nonexistent/path/audio.wav")


def test_audio_quality_preservation(sample_audio_tensor):
    result = _audio_mod.encode_audio(
        audio=sample_audio_tensor,
        sample_rate=16000,
        audio_format="mp3",
        bitrate="128k",
    )

    assert len(result["audio"]) > 1000


def test_end_to_end_audio_processing(sample_audio_tensor):
    original_samples = sample_audio_tensor.shape[1]
    original_duration = original_samples / 16000

    result = _audio_mod.encode_audio(
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


# --- Codec detection tests ---


def test_codec_to_format_mapping():
    """
    _codec_to_format should map known codec strings to container formats.

    ## WRITTEN BY AI ##
    """
    assert _audio_mod._codec_to_format("pcm_s16le") == "wav"
    assert _audio_mod._codec_to_format("pcm_f32le") == "wav"
    assert _audio_mod._codec_to_format("flac") == "flac"
    assert _audio_mod._codec_to_format("mp3") == "mp3"
    assert _audio_mod._codec_to_format("vorbis") == "ogg"
    assert _audio_mod._codec_to_format("aac") == "aac"
    assert _audio_mod._codec_to_format("opus") == "ogg"
    assert _audio_mod._codec_to_format("unknown_codec") is None


def test_codec_detection_from_wav_file(real_wav_file):
    """
    Decoding a WAV file should detect a pcm_* codec and re-encode as WAV.

    ## WRITTEN BY AI ##
    """
    result = _audio_mod.encode_audio(audio=real_wav_file, max_duration=1.0)

    assert result["format"] == "wav"
    assert result["mimetype"] == "audio/wav"


def test_codec_detection_from_wav_bytes(sample_audio_tensor):
    """
    Encoding to WAV then decoding from raw bytes should detect pcm codec
    and re-encode as WAV without needing explicit audio_format.

    ## WRITTEN BY AI ##
    """
    # First encode as WAV to get valid WAV bytes
    wav_result = _audio_mod.encode_audio(
        audio=sample_audio_tensor, sample_rate=16000, audio_format="wav"
    )
    wav_bytes = wav_result["audio"]

    # Now feed raw bytes back -- codec detection should find pcm_*
    result = _audio_mod.encode_audio(audio=wav_bytes)

    assert result["format"] == "wav"
    assert result["mimetype"] == "audio/wav"


def test_codec_detection_from_flac_bytes(sample_audio_tensor):
    """
    Encoding to FLAC then decoding from raw bytes should detect flac codec
    and re-encode as FLAC.

    ## WRITTEN BY AI ##
    """
    flac_result = _audio_mod.encode_audio(
        audio=sample_audio_tensor, sample_rate=16000, audio_format="flac"
    )
    flac_bytes = flac_result["audio"]

    result = _audio_mod.encode_audio(audio=flac_bytes)

    assert result["format"] == "flac"
    assert result["mimetype"] == "audio/flac"


def test_codec_detection_from_mp3_bytes(sample_audio_tensor):
    """
    Encoding to MP3 then decoding from raw bytes should detect mp3 codec
    and re-encode as MP3.

    ## WRITTEN BY AI ##
    """
    mp3_result = _audio_mod.encode_audio(
        audio=sample_audio_tensor, sample_rate=16000, audio_format="mp3"
    )
    mp3_bytes = mp3_result["audio"]

    result = _audio_mod.encode_audio(audio=mp3_bytes)

    assert result["format"] == "mp3"
    assert result["mimetype"] == "audio/mp3"


def test_explicit_format_overrides_codec(real_wav_file):
    """
    An explicit audio_format kwarg should override the codec detected
    from the source.

    ## WRITTEN BY AI ##
    """
    result = _audio_mod.encode_audio(
        audio=real_wav_file,
        sample_rate=16000,
        audio_format="mp3",
        max_duration=1.0,
    )

    assert result["format"] == "mp3"
    assert result["mimetype"] == "audio/mp3"


@patch("guidellm.utils.audio._decode_audio")
def test_pcm16_append_b64_chunks_rejects_unknown_dict_keys(mock_decode):
    """## WRITTEN BY AI ##"""
    mock_decode.side_effect = AssertionError("_decode_audio should not run")
    with pytest.raises(ValueError, match="audio_column dict"):
        _audio_mod.pcm16_append_b64_chunks({"foo": 1})


@patch("guidellm.utils.audio._decode_audio")
def test_pcm16_append_b64_chunks_splits_into_multiple_base64_chunks(mock_decode):
    """## WRITTEN BY AI ##"""
    mock_decode.return_value = MagicMock()
    mock_decode.return_value.data = torch.zeros(1, 5000)
    mock_decode.return_value.sample_rate = 16000

    out = _audio_mod.pcm16_append_b64_chunks({"audio": b"x"}, chunk_samples=3200)

    assert len(out) == 2
    assert all(isinstance(chunk_b64, str) for chunk_b64 in out)


@patch("guidellm.utils.audio._decode_audio")
def test_pcm16_append_b64_chunks_empty_wave_raises(mock_decode):
    """## WRITTEN BY AI ##"""
    mock_decode.return_value = MagicMock()
    mock_decode.return_value.data = torch.zeros(1, 0)
    mock_decode.return_value.sample_rate = 16000

    with pytest.raises(ValueError, match="no PCM"):
        _audio_mod.pcm16_append_b64_chunks({"audio": b"x"})


@patch("guidellm.utils.audio._decode_audio")
def test_pcm16_append_b64_chunks_downmixes_stereo(mock_decode):
    """## WRITTEN BY AI ##"""
    mock_decode.return_value = MagicMock()
    mock_decode.return_value.data = torch.randn(2, 200)
    mock_decode.return_value.sample_rate = 16000

    out = _audio_mod.pcm16_append_b64_chunks({"audio": b"x"}, chunk_samples=100)

    assert len(out) >= 1


@patch("guidellm.utils.audio._decode_audio")
def test_pcm16_append_b64_chunks_audio_dict_passes_outer_sample_rate(mock_decode):
    """## WRITTEN BY AI ##"""
    mock_decode.return_value = MagicMock()
    mock_decode.return_value.data = torch.zeros(1, 100)
    mock_decode.return_value.sample_rate = 16000

    _audio_mod.pcm16_append_b64_chunks({"audio": b"x", "sample_rate": 8000})

    mock_decode.assert_called_once()
    assert mock_decode.call_args.kwargs.get("sample_rate") == 8000


@patch("guidellm.utils.audio._decode_audio")
def test_pcm16_append_b64_chunks_sampling_rate_alias(mock_decode):
    """## WRITTEN BY AI ##"""
    mock_decode.return_value = MagicMock()
    mock_decode.return_value.data = torch.zeros(1, 50)
    mock_decode.return_value.sample_rate = 16000

    _audio_mod.pcm16_append_b64_chunks({"audio": b"x", "sampling_rate": 44100})

    assert mock_decode.call_args.kwargs.get("sample_rate") == 44100


@patch("guidellm.utils.audio._decode_audio")
def test_pcm16_append_b64_chunks_invalid_decoder_sample_rate_raises(mock_decode):
    """## WRITTEN BY AI ##"""
    mock_decode.return_value = MagicMock()
    mock_decode.return_value.data = torch.zeros(1, 10)
    mock_decode.return_value.sample_rate = 0

    with pytest.raises(ValueError, match="invalid sample_rate"):
        _audio_mod.pcm16_append_b64_chunks({"audio": b"x"})
