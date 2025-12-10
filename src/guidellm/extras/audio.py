from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import torch

try:
    from torchcodec import AudioSamples
    from torchcodec.decoders import AudioDecoder
    from torchcodec.encoders import AudioEncoder
except ImportError as e:
    raise ImportError("Please install guidellm[audio] to use audio features") from e

__all__ = [
    "encode_audio",
    "is_url",
]


def is_url(text: Any) -> bool:
    return isinstance(text, str) and text.startswith(("http://", "https://"))


def encode_audio(
    audio: AudioDecoder
    | bytes
    | str
    | Path
    | np.ndarray
    | torch.Tensor
    | dict[str, Any],
    b64encode: bool = False,
    sample_rate: int | None = None,
    file_name: str = "audio.wav",
    encode_sample_rate: int = 16000,
    max_duration: float | None = None,
    mono: bool = True,
    audio_format: str = "mp3",
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
    """Decode audio (if necessary) and re-encode to specified format."""
    samples = _decode_audio(audio, sample_rate=sample_rate, max_duration=max_duration)

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

    return {
        "type": "audio_base64" if b64encode else "audio_file",
        "audio": (
            base64.b64encode(encoded_audio).decode("utf-8")
            if b64encode
            else encoded_audio
        ),
        "file_name": get_file_name(audio)
        if isinstance(audio, str | Path)
        else file_name,
        "format": audio_format,
        "mimetype": f"audio/{format_val}",
        "audio_samples": samples.sample_rate,
        "audio_seconds": samples.duration_seconds,
        "audio_bytes": len(encoded_audio),
    }


def _decode_audio(  # noqa: C901, PLR0912
    audio: AudioDecoder
    | bytes
    | str
    | Path
    | np.ndarray
    | torch.Tensor
    | dict[str, Any],
    sample_rate: int | None = None,
    max_duration: float | None = None,
) -> AudioSamples:
    """Decode audio from various input types into AudioSamples."""
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

    samples: AudioSamples

    data: torch.Tensor | bytes
    # HF datasets return AudioDecoder for audio column
    if isinstance(audio, AudioDecoder):
        samples = audio.get_samples_played_in_range(stop_seconds=max_duration)
    elif isinstance(audio, torch.Tensor):
        # If float stream assume decoded audio
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

            samples = AudioSamples(
                data=data,
                pts_seconds=0.0,
                duration_seconds=duration,
                sample_rate=sample_rate,
            )
        # If bytes tensor assume encoded audio
        elif audio.dtype == torch.uint8:
            decoder = AudioDecoder(
                source=audio,
                sample_rate=sample_rate,
            )
            samples = decoder.get_samples_played_in_range(stop_seconds=max_duration)

        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

    # If bytes, assume encoded audio
    elif isinstance(audio, bytes):
        decoder = AudioDecoder(
            source=audio,
            sample_rate=sample_rate,
        )
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
        decoder = AudioDecoder(
            source=data,
        )
        samples = decoder.get_samples_played_in_range(stop_seconds=max_duration)
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    return samples


def _encode_audio(
    samples: AudioSamples,
    resample_rate: int | None = None,
    bitrate: int = 64000,
    audio_format: str = "mp3",
    mono: bool = True,
) -> bytes:
    encoder = AudioEncoder(
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
