from __future__ import annotations

import base64
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
    "pcm16_append_b64_chunks",
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
            logger.debug("Falling back to WAV audio formatting")

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
        "type": "audio_file",
        "audio": encoded_audio,
        "file_name": get_file_name(audio)
        if isinstance(audio, str | Path)
        else file_name,
        "format": audio_format,
        "mimetype": f"audio/{format_val}",
        "audio_samples": samples.sample_rate,
        "audio_seconds": samples.duration_seconds,
        "audio_bytes": len(encoded_audio),
    }


def decode_audio(audio: bytes):
    audio_samples, _ = _decode_audio(audio)
    # torchcodec decodes audio on CPU, so .data is always
    # a CPU torch.Tensor. .cpu() is a no-op on CPU tensors.
    return audio_samples.data.cpu().numpy()


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


# Decoded float waveforms are nominally in [-1.0, 1.0]; clip before scaling to int16.
_PCM16_WAVE_CLIP_MIN = -1.0
_PCM16_WAVE_CLIP_MAX = 1.0
# Symmetric int16 positive peak (2**15 - 1); standard float[-1, 1] -> PCM16 mapping.
_PCM16_FLOAT_TO_INT16_SCALE = 32767.0
_BYTES_PER_PCM16_SAMPLE = 2


def _sample_rate_hint_from_audio_column_dict(d: dict[str, Any]) -> int | None:
    """Return ``sample_rate`` / ``sampling_rate`` from an audio column dict."""
    hint = d.get("sample_rate", d.get("sampling_rate"))
    if (
        hint is not None
        and not isinstance(hint, bool)
        and isinstance(hint, int | float)
        and hint > 0
    ):
        return int(round(float(hint)))
    return None


def _require_positive_sample_rate(sr_raw: Any) -> float:
    if isinstance(sr_raw, bool) or not isinstance(sr_raw, int | float) or sr_raw <= 0:
        raise ValueError(
            "Decoded audio has invalid sample_rate "
            f"{sr_raw!r}; expected a positive number"
        )
    return float(sr_raw)


def pcm16_append_b64_chunks(
    audio_item: dict[str, Any] | bytes,
    *,
    target_sample_rate: int = 16000,
    chunk_samples: int = 3200,
) -> list[str]:
    """
    Decode audio to base64-encoded PCM16 mono chunks for realtime ``append`` events.

    Matches vLLM ``input_audio_buffer.append`` (PCM16 mono at ``target_sample_rate``
    Hz), split into ``chunk_samples``-frame segments.
    Equivalent conversion flow to vLLM's realtime microphone client example, but
    generalized for dataset/file inputs used by GuideLLM benchmarks.
    """
    # Accept common audio column shapes used in GuideLLM datasets.
    if isinstance(audio_item, dict):
        if "audio" in audio_item:
            decode_sr = _sample_rate_hint_from_audio_column_dict(audio_item)
            samples = _decode_audio(
                audio_item["audio"],
                sample_rate=decode_sr,
            )
        elif "data" in audio_item or "url" in audio_item:
            samples = _decode_audio(audio_item)
        else:
            raise ValueError(
                "audio_column dict must include 'audio', 'data', or 'url' "
                "(same shapes as encode_audio / _decode_audio); "
                f"got keys {list(audio_item)!r}"
            )
    else:
        samples = _decode_audio(audio_item)

    # Ensure channel-first shape, then downmix to mono for realtime PCM input.
    data = samples.data
    if data.dim() == 1:
        data = data.unsqueeze(0)
    if data.shape[0] > 1:
        data = data.mean(dim=0, keepdim=True)

    # Realtime endpoint expects 16 kHz PCM16 mono.
    sr = _require_positive_sample_rate(samples.sample_rate)
    if sr != target_sample_rate:
        t_in = data.shape[1]
        t_out = max(1, int(round(t_in * target_sample_rate / sr)))
        data = torch.nn.functional.interpolate(
            data.unsqueeze(0),
            size=t_out,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

    # Convert float waveform to signed little-endian PCM16 bytes.
    wave = data.squeeze(0)
    pcm_i16 = (
        wave.clamp(_PCM16_WAVE_CLIP_MIN, _PCM16_WAVE_CLIP_MAX)
        * _PCM16_FLOAT_TO_INT16_SCALE
    ).round().to(torch.int16)
    buf = pcm_i16.cpu().numpy().tobytes()

    # Split PCM bytes into chunk-sized base64 payloads for append events.
    chunk_bytes = max(1, chunk_samples) * _BYTES_PER_PCM16_SAMPLE
    out: list[str] = []
    for i in range(0, len(buf), chunk_bytes):
        pcm_chunk = buf[i : i + chunk_bytes]
        if pcm_chunk:
            out.append(base64.b64encode(pcm_chunk).decode("ascii"))
    if not out:
        raise ValueError("Decoded audio produced no PCM data")
    return out
