from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import torch

# CRITICAL: Use 'import ... as libs' pattern to preserve lazy loading
# This defers errors until attributes are actually accessed
import guidellm.extras.audio as libs
from guidellm.logger import logger

__all__ = [
    "encode_audio",
    "is_url",
]


def is_url(text: Any) -> bool:
    return isinstance(text, str) and text.startswith(("http://", "https://"))


# Maps torchcodec codec names to encoder-compatible container format strings.
# PCM variants (pcm_s16le, pcm_f32le, etc.) are handled separately via prefix.
_CODEC_TO_FORMAT: dict[str, str] = {
    "flac": "flac",
    "mp3": "mp3",
    "vorbis": "ogg",
    "aac": "aac",
    "opus": "ogg",
}

_wav_fallback_state = {"warned": False}


def _codec_to_format(codec: str) -> str | None:
    """
    Map a torchcodec codec name to an encoder-compatible container format.

    :param codec: Codec string from ``AudioDecoder.metadata.codec``.
    :return: Container format string, or None if the codec is unrecognized.
    """
    if codec.startswith("pcm_"):
        return "wav"
    return _CODEC_TO_FORMAT.get(codec)


def encode_audio(
    audio: libs.AudioDecoder
    | bytes
    | str
    | Path
    | np.ndarray
    | torch.Tensor
    | dict[str, Any],
    sample_rate: int | None = None,
    file_name: str | None = None,
    encode_sample_rate: int = 16000,
    max_duration: float | None = None,
    mono: bool = True,
    audio_format: str | None = None,
    bitrate: str = "64k",
) -> dict[
    Literal[
        "type",
        "audio",
        "format",
        "mimetype",
        "audio_samples",
        "audio_seconds",
        "audio_bytes",
        "file_name",
    ],
    str | int | float | bytes | None,
]:
    """
    Decode audio (if necessary) and re-encode to specified format.

    If ``audio_format`` is not provided, the format is detected from the
    source codec (via torchcodec metadata). When detection is not possible
    (e.g. raw float tensors), falls back to WAV with a one-time warning.

    :param audio: Audio input in any supported form.
    :param sample_rate: Sample rate hint for raw decoded audio.
    :param file_name: Override file name in output metadata. Defaults to a
        name derived from the resolved format.
    :param encode_sample_rate: Target sample rate for the encoded output.
    :param max_duration: Truncate audio to this duration in seconds.
    :param mono: Convert to mono if True.
    :param audio_format: Target encoding format. If None, detected from source.
    :param bitrate: Bitrate for lossy formats like mp3.
    :return: Dict containing encoded audio bytes and metadata.
    """
    samples, source_codec = _decode_audio(
        audio, sample_rate=sample_rate, max_duration=max_duration
    )

    # Resolve format: explicit > codec-detected > WAV fallback
    if audio_format is None:
        if source_codec:
            audio_format = _codec_to_format(source_codec)
        if audio_format is None:
            audio_format = "wav"
            if not _wav_fallback_state["warned"]:
                _wav_fallback_state["warned"] = True
                logger.warning(
                    "Could not detect source audio codec; "
                    "falling back to WAV encoding. "
                    "Set audio_format explicitly to suppress this warning."
                )

    bitrate_val = (
        int(bitrate.rstrip("k")) * 1000 if bitrate.endswith("k") else int(bitrate)
    )
    format_val = audio_format.lower()

    encoded_audio = _encode_audio(
        samples=samples,
        resample_rate=encode_sample_rate,
        bitrate=bitrate_val,
        audio_format=format_val,
        mono=mono,
    )

    # Use source file name when available, otherwise build from resolved format
    if isinstance(audio, str | Path):
        resolved_file_name = get_file_name(audio)
    elif file_name is not None:
        resolved_file_name = file_name
    else:
        resolved_file_name = f"audio.{format_val}"

    return {
        "type": "audio_file",
        "audio": encoded_audio,
        "file_name": resolved_file_name,
        "format": audio_format,
        "mimetype": f"audio/{format_val}",
        "audio_samples": samples.sample_rate,
        "audio_seconds": samples.duration_seconds,
        "audio_bytes": len(encoded_audio),
    }


def _decode_audio(  # noqa: C901, PLR0912, PLR0915
    audio: libs.AudioDecoder
    | bytes
    | str
    | Path
    | np.ndarray
    | torch.Tensor
    | dict[str, Any],
    sample_rate: int | None = None,
    max_duration: float | None = None,
) -> tuple[libs.AudioSamples, str | None]:
    """
    Decode audio from various input types into AudioSamples.

    :return: Tuple of (decoded samples, detected codec string or None).
    """
    # If input is a dict, unwrap it into a function call
    if isinstance(audio, dict):
        sample_rate = audio.get("sample_rate", audio.get("sampling_rate", sample_rate))
        if "data" not in audio and "url" not in audio:
            raise ValueError(
                f"Audio dict must contain either 'data' or 'url' keys, got {audio}"
            )
        audio_data = audio["data"] if "data" in audio else audio.get("url")
        if audio_data is None:
            raise ValueError(
                f"Audio dict must contain either 'data' or 'url' keys, got {audio}"
            )
        return _decode_audio(
            audio=audio_data,
            sample_rate=sample_rate,
            max_duration=max_duration,
        )

    # Convert numpy array to torch tensor and re-call
    if isinstance(audio, np.ndarray):
        return _decode_audio(
            audio=torch.from_numpy(audio),
            sample_rate=sample_rate,
            max_duration=max_duration,
        )

    data: torch.Tensor | bytes
    codec: str | None = None

    # HF datasets return AudioDecoder for audio column
    if isinstance(audio, libs.AudioDecoder):
        codec = audio.metadata.codec
        samples = audio.get_samples_played_in_range(stop_seconds=max_duration)
    elif isinstance(audio, torch.Tensor):
        # If float stream assume decoded audio -- no codec info available
        if torch.is_floating_point(audio):
            if sample_rate is None:
                raise ValueError("Sample rate must be set for decoded audio")

            full_duration = audio.shape[1] / sample_rate
            # If max_duration is set, trim the audio to that duration
            if max_duration is not None:
                num_samples = int(max_duration * sample_rate)
                duration = min(max_duration, full_duration)
                data = audio[:, :num_samples]
            else:
                duration = full_duration
                data = audio

            samples = libs.AudioSamples(
                data=data,
                pts_seconds=0.0,
                duration_seconds=duration,
                sample_rate=sample_rate,
            )
        # If bytes tensor assume encoded audio
        elif audio.dtype == torch.uint8:
            decoder = libs.AudioDecoder(
                source=audio,
                sample_rate=sample_rate,
            )
            codec = decoder.metadata.codec
            samples = decoder.get_samples_played_in_range(stop_seconds=max_duration)

        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

    # If bytes, assume encoded audio
    elif isinstance(audio, bytes):
        decoder = libs.AudioDecoder(
            source=audio,
            sample_rate=sample_rate,
        )
        codec = decoder.metadata.codec
        samples = decoder.get_samples_played_in_range(stop_seconds=max_duration)

    # If str or Path, assume file path or URL to encoded audio
    elif isinstance(audio, str | Path):
        if isinstance(audio, str) and is_url(audio):
            response = httpx.get(audio)
            response.raise_for_status()
            data = response.content
        else:
            if not Path(audio).exists():
                raise ValueError(f"Audio file does not exist: {audio}")
            data = Path(audio).read_bytes()
        decoder = libs.AudioDecoder(
            source=data,
        )
        codec = decoder.metadata.codec
        samples = decoder.get_samples_played_in_range(stop_seconds=max_duration)
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    return samples, codec


def _encode_audio(
    samples: libs.AudioSamples,
    resample_rate: int | None = None,
    bitrate: int = 64000,
    audio_format: str = "wav",
    mono: bool = True,
) -> bytes:
    encoder = libs.AudioEncoder(
        samples=samples.data,
        sample_rate=samples.sample_rate,
    )

    audio_tensor = encoder.to_tensor(
        format=audio_format,
        bit_rate=bitrate if audio_format == "mp3" else None,
        num_channels=1 if mono else None,
        sample_rate=resample_rate,
    )

    return audio_tensor.numpy().tobytes()


def get_file_name(path: Path | str) -> str:
    """Get file name from path."""
    return Path(path).name
