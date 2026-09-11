from __future__ import annotations

import base64

import pytest

from guidellm.mock_server.models import ChatMessage
from guidellm.mock_server.multimodal import (
    LOSSY_BYTES_PER_SECOND,
    WAV_BYTES_PER_SECOND,
    InvalidContentPartError,
    MultimodalPromptStats,
    accumulate_multimodal_content,
    estimate_audio_seconds,
)
from guidellm.schemas.mock_server.config import MockServerConfig


def _config(**kwargs) -> MockServerConfig:
    defaults = {
        "image_tokens": 576,
        "video_tokens": 1024,
        "audio_tokens_per_second": 25.0,
    }
    defaults.update(kwargs)
    return MockServerConfig(**defaults)


def _b64_audio(num_bytes: int) -> str:
    return base64.b64encode(b"\x00" * num_bytes).decode("utf-8")


class TestEstimateAudioSeconds:
    """Test suite for estimate_audio_seconds."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("num_bytes", "format_hint", "expected"),
        [
            (WAV_BYTES_PER_SECOND, "wav", 1.0),
            (WAV_BYTES_PER_SECOND, "audio/wav", 1.0),
            (WAV_BYTES_PER_SECOND, "sample.WAV", 1.0),
            (WAV_BYTES_PER_SECOND, "pcm", 1.0),
            (LOSSY_BYTES_PER_SECOND, "mp3", 1.0),
            (LOSSY_BYTES_PER_SECOND, "audio/flac", 1.0),
            (LOSSY_BYTES_PER_SECOND, None, 1.0),
            (0, "wav", 0.0),
        ],
    )
    def test_invocation(self, num_bytes, format_hint, expected):
        """Test duration estimates across formats and byte counts.

        ## WRITTEN BY AI ##
        """
        assert estimate_audio_seconds(num_bytes, format_hint) == expected


class TestMultimodalPromptStats:
    """Test suite for MultimodalPromptStats."""

    @pytest.mark.smoke
    def test_total_tokens(self):
        """Test total_tokens sums the present modalities and treats None as 0.

        ## WRITTEN BY AI ##
        """
        assert MultimodalPromptStats().total_tokens == 0
        stats = MultimodalPromptStats(
            image_tokens=576, video_tokens=1024, audio_tokens=25, audio_seconds=1.0
        )
        assert stats.total_tokens == 576 + 1024 + 25
        assert MultimodalPromptStats(image_tokens=10).total_tokens == 10

    @pytest.mark.smoke
    def test_prompt_tokens_details_empty(self):
        """Test details are None for text-only requests.

        ## WRITTEN BY AI ##
        """
        assert MultimodalPromptStats().prompt_tokens_details(42) is None

    @pytest.mark.smoke
    def test_prompt_tokens_details_full(self):
        """Test details include text tokens and only the present modalities.

        ## WRITTEN BY AI ##
        """
        stats = MultimodalPromptStats(
            image_tokens=576, video_tokens=1024, audio_tokens=25, audio_seconds=1.0
        )
        assert stats.prompt_tokens_details(42) == {
            "prompt_tokens": 42,
            "image_tokens": 576,
            "video_tokens": 1024,
            "audio_tokens": 25,
            "seconds": 1.0,
        }
        assert MultimodalPromptStats(image_tokens=576).prompt_tokens_details(7) == {
            "prompt_tokens": 7,
            "image_tokens": 576,
        }


class TestAccumulateMultimodalContent:
    """Test suite for accumulate_multimodal_content."""

    @pytest.mark.smoke
    def test_text_only(self):
        """Test string and text-part content produce no multimodal charges.

        ## WRITTEN BY AI ##
        """
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content=[{"type": "text", "text": "Hi"}]),
            ChatMessage(role="assistant", content=None),
        ]
        stats = accumulate_multimodal_content(messages, _config())
        assert stats.image_tokens is None
        assert stats.video_tokens is None
        assert stats.audio_tokens is None
        assert stats.audio_seconds is None
        assert stats.total_tokens == 0

    @pytest.mark.smoke
    def test_mixed_modalities(self):
        """Test image, video, and audio parts accumulate across messages.

        ## WRITTEN BY AI ##
        """
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Describe these"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                    {"type": "image_url", "image_url": {"url": "http://x/img.png"}},
                    {"type": "video_url", "video_url": {"url": "http://x/clip.mp4"}},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": _b64_audio(WAV_BYTES_PER_SECOND * 2),
                            "format": "wav",
                        },
                    },
                ],
            ),
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": _b64_audio(LOSSY_BYTES_PER_SECOND),
                            "format": "mp3",
                        },
                    },
                ],
            ),
        ]
        stats = accumulate_multimodal_content(messages, _config())
        assert stats.image_tokens == 2 * 576
        assert stats.video_tokens == 1024
        assert stats.audio_seconds == 3.0
        assert stats.audio_tokens == 75
        assert stats.total_tokens == 2 * 576 + 1024 + 75

    @pytest.mark.sanity
    def test_configurable_charges(self):
        """Test token charges follow the configured heuristic values.

        ## WRITTEN BY AI ##
        """
        config = _config(image_tokens=10, video_tokens=20, audio_tokens_per_second=2.0)
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
                    {"type": "video_url", "video_url": {"url": "http://x/a.mp4"}},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": _b64_audio(WAV_BYTES_PER_SECOND // 2),
                            "format": "wav",
                        },
                    },
                ],
            ),
        ]
        stats = accumulate_multimodal_content(messages, config)
        assert stats.image_tokens == 10
        assert stats.video_tokens == 20
        assert stats.audio_seconds == 0.5
        assert stats.audio_tokens == 1  # ceil(0.5 * 2.0)

    @pytest.mark.sanity
    def test_audio_format_default_lossy(self):
        """Test audio without a format field uses the lossy byte rate.

        ## WRITTEN BY AI ##
        """
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "input_audio",
                        "input_audio": {"data": _b64_audio(LOSSY_BYTES_PER_SECOND)},
                    },
                ],
            ),
        ]
        stats = accumulate_multimodal_content(messages, _config())
        assert stats.audio_seconds == 1.0
        assert stats.audio_tokens == 25

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("part", "match"),
        [
            ({"type": "ref_image"}, "Unsupported content part type"),
            ({"text": "no type"}, "Unsupported content part type"),
            ({"type": "text"}, "string 'text' field"),
            ({"type": "text", "text": 5}, "string 'text' field"),
            ({"type": "image_url"}, "string 'url' field"),
            ({"type": "image_url", "image_url": "http://x/a.png"}, "string 'url'"),
            ({"type": "image_url", "image_url": {"url": ""}}, "non-empty 'url'"),
            ({"type": "video_url", "video_url": {}}, "string 'url' field"),
            ({"type": "input_audio"}, "base64 string 'data' field"),
            ({"type": "input_audio", "input_audio": {"data": ""}}, "non-empty"),
            (
                {"type": "input_audio", "input_audio": {"data": "@@not-base64@@"}},
                "invalid base64",
            ),
            (
                {
                    "type": "input_audio",
                    "input_audio": {"data": "AAAA", "format": 3},
                },
                "non-string 'format'",
            ),
        ],
    )
    def test_invalid_parts(self, part, match):
        """Test malformed content parts raise InvalidContentPartError.

        ## WRITTEN BY AI ##
        """
        messages = [ChatMessage(role="user", content=[part])]
        with pytest.raises(InvalidContentPartError, match=match):
            accumulate_multimodal_content(messages, _config())
