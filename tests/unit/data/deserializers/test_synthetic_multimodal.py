"""Unit tests for synthetic_image / synthetic_video deserializers."""

from __future__ import annotations

import base64
import hashlib
import io
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import imageio
import pytest
from PIL import Image

from guidellm.data.deserializers import (
    DatasetDeserializerFactory,
    SyntheticImageDataArgs,
    SyntheticImageDataset,
    SyntheticImageDatasetDeserializer,
    SyntheticVideoDataArgs,
    SyntheticVideoDataset,
    SyntheticVideoDatasetDeserializer,
)
from guidellm.data.schemas import DataArgs
from guidellm.utils.vision import synthesize_image, synthesize_video


def _mock_tokenizer() -> Mock:
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
    tokenizer.decode.side_effect = lambda tokens, skip_special_tokens=False: " ".join(
        f"tok_{t}" for t in tokens
    )
    return tokenizer


def _decode_data_url(data_url: str) -> bytes:
    _, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded)


# ---------------------------------------------------------------------------
# synthesize_image
# ---------------------------------------------------------------------------


class TestSynthesizeImage:
    @pytest.mark.smoke
    @pytest.mark.parametrize("fmt", ["jpeg", "png"])
    @pytest.mark.parametrize(("width", "height"), [(640, 480), (1280, 720), (256, 256)])
    def test_decoded_dims_match(self, fmt: str, width: int, height: int):
        """## WRITTEN BY AI ##"""
        out = synthesize_image(width, height, image_format=fmt, seed=0, row_index=0)
        decoded = _decode_data_url(out["image"])
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (width, height)
        assert out["image_pixels"] == width * height

    @pytest.mark.smoke
    def test_image_bytes_match_payload(self):
        """## WRITTEN BY AI ##"""
        out = synthesize_image(640, 480, seed=0, row_index=0)
        decoded = _decode_data_url(out["image"])
        assert out["image_bytes"] == len(decoded)

    @pytest.mark.smoke
    def test_reproducible_same_seed_row_index(self):
        """## WRITTEN BY AI ##"""
        a = synthesize_image(320, 240, seed=99, row_index=7)
        b = synthesize_image(320, 240, seed=99, row_index=7)
        assert a["image"] == b["image"]

    @pytest.mark.smoke
    def test_row_index_changes_payload(self):
        """## WRITTEN BY AI ##"""
        a = synthesize_image(320, 240, seed=99, row_index=0)
        b = synthesize_image(320, 240, seed=99, row_index=1)
        assert a["image"] != b["image"]

    @pytest.mark.sanity
    def test_seed_changes_payload(self):
        """## WRITTEN BY AI ##"""
        a = synthesize_image(320, 240, seed=1, row_index=0)
        b = synthesize_image(320, 240, seed=2, row_index=0)
        assert a["image"] != b["image"]

    @pytest.mark.sanity
    @pytest.mark.parametrize("content", ["gradient", "noise", "solid", "checkerboard"])
    def test_content_modes_produce_valid_images(self, content: str):
        """## WRITTEN BY AI ##"""
        out = synthesize_image(64, 64, content=content, seed=3, row_index=0)
        decoded = _decode_data_url(out["image"])
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (64, 64)
        assert out["image_bytes"] > 0

    @pytest.mark.sanity
    def test_byte_uniqueness_gradient_1000_rows(self):
        """1000 gradient rows with the same seed must all be byte-different.

        ## WRITTEN BY AI ##
        """
        hashes = set()
        for i in range(1000):
            out = synthesize_image(128, 128, content="gradient", seed=17, row_index=i)
            hashes.add(hashlib.sha256(out["image"].encode()).hexdigest())
        assert len(hashes) == 1000

    @pytest.mark.regression
    def test_unsupported_format_raises(self):
        """## WRITTEN BY AI ##"""
        with pytest.raises(ValueError, match="format"):
            synthesize_image(64, 64, image_format="webp", seed=0)

    @pytest.mark.regression
    def test_unsupported_content_raises(self):
        """## WRITTEN BY AI ##"""
        with pytest.raises(ValueError, match="content"):
            synthesize_image(64, 64, content="zebra", seed=0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# synthesize_video
# ---------------------------------------------------------------------------


class TestSynthesizeVideo:
    @pytest.mark.smoke
    @pytest.mark.parametrize("frames", [4, 6, 12])
    @pytest.mark.parametrize("fps", [1.0, 2.0])
    def test_decoded_frame_count_and_seconds_match(self, frames: int, fps: float):
        """## WRITTEN BY AI ##"""
        out = synthesize_video(320, 240, frames=frames, fps=fps, seed=5, row_index=0)
        decoded = _decode_data_url(out["video"])
        # Write to temp file and read back via imageio's ffmpeg reader to
        # check decoded frame count and dims.
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(decoded)
            path = f.name
        try:
            reader: Any = imageio.get_reader(path, "ffmpeg")  # type: ignore[arg-type]
            decoded_frames = [frame for frame in reader]  # noqa: C416
            assert len(decoded_frames) == frames
            assert decoded_frames[0].shape == (240, 320, 3)
            reader.close()
        finally:
            Path(path).unlink()

        assert out["video_frames"] == frames
        assert out["video_seconds"] == pytest.approx(frames / fps)

    @pytest.mark.smoke
    def test_video_bytes_match_payload(self):
        """## WRITTEN BY AI ##"""
        out = synthesize_video(320, 240, frames=4, fps=1, seed=5, row_index=0)
        decoded = _decode_data_url(out["video"])
        assert out["video_bytes"] == len(decoded)

    @pytest.mark.smoke
    def test_reproducible_same_seed_row_index(self):
        """## WRITTEN BY AI ##"""
        a = synthesize_video(160, 120, frames=3, fps=1, seed=42, row_index=2)
        b = synthesize_video(160, 120, frames=3, fps=1, seed=42, row_index=2)
        assert a["video"] == b["video"]

    @pytest.mark.smoke
    def test_row_index_changes_payload(self):
        """## WRITTEN BY AI ##"""
        a = synthesize_video(160, 120, frames=3, fps=1, seed=42, row_index=0)
        b = synthesize_video(160, 120, frames=3, fps=1, seed=42, row_index=1)
        assert a["video"] != b["video"]

    @pytest.mark.sanity
    def test_byte_uniqueness_gradient_video(self):
        """200 gradient clips with same seed must all be byte-different.

        ## WRITTEN BY AI ##
        """
        hashes = set()
        for i in range(200):
            out = synthesize_video(
                64, 64, frames=2, fps=1, content="gradient", seed=8, row_index=i
            )
            hashes.add(hashlib.sha256(out["video"].encode()).hexdigest())
        assert len(hashes) == 200

    @pytest.mark.regression
    def test_unsupported_format_raises(self):
        """## WRITTEN BY AI ##"""
        with pytest.raises(ValueError, match="format"):
            synthesize_video(64, 64, frames=2, video_format="webm", seed=0)

    @pytest.mark.regression
    def test_unsupported_content_raises(self):
        """## WRITTEN BY AI ##"""
        with pytest.raises(ValueError, match="content"):
            synthesize_video(64, 64, frames=2, content="solid", seed=0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestSyntheticImageConfig:
    @pytest.mark.smoke
    def test_resolution_resolves_to_width_height(self):
        """## WRITTEN BY AI ##"""
        cfg = SyntheticImageDataArgs(resolution="720p", text_tokens=50)
        assert cfg.width == 1280
        assert cfg.height == 720

    @pytest.mark.sanity
    def test_aspect_ratio_overrides_width(self):
        """## WRITTEN BY AI ##"""
        cfg = SyntheticImageDataArgs(
            resolution="720p", aspect_ratio="4:3", text_tokens=50
        )
        # 720 * 4 / 3 = 960
        assert cfg.height == 720
        assert cfg.width == 960

    @pytest.mark.sanity
    def test_prompt_tokens_alias_accepted(self):
        """## WRITTEN BY AI ##"""
        cfg = SyntheticImageDataArgs.model_validate(
            {
                "kind": "synthetic_image",
                "width": 640,
                "height": 480,
                "prompt_tokens": 50,
            }
        )
        assert cfg.text_tokens == 50

    @pytest.mark.regression
    def test_missing_dims_raises(self):
        """## WRITTEN BY AI ##"""
        with pytest.raises(ValueError):
            SyntheticImageDataArgs(text_tokens=10)

    @pytest.mark.regression
    def test_unknown_resolution_raises(self):
        """## WRITTEN BY AI ##"""
        with pytest.raises(ValueError, match="resolution"):
            SyntheticImageDataArgs(resolution="9000p", text_tokens=10)


# ---------------------------------------------------------------------------
# Deserializer from typed config + 10-row pull
# ---------------------------------------------------------------------------


class TestSyntheticImageDeserializer:
    @pytest.mark.smoke
    def test_pull_10_rows_from_config(self):
        """## WRITTEN BY AI ##"""
        d = SyntheticImageDatasetDeserializer()
        ds = d(
            config=SyntheticImageDataArgs(
                resolution="480p",
                text_tokens=20,
                output_tokens=8,
                seed=11,
            ),
            processor_factory=_mock_tokenizer,
            random_seed=42,
        )
        assert isinstance(ds, SyntheticImageDataset)

        rows = []
        it = iter(ds)
        for _ in range(10):
            rows.append(next(it))

        assert len(rows) == 10
        for row in rows:
            assert "image" in row
            assert row["image"]["image_pixels"] == 854 * 480
            assert row["image"]["image_bytes"] > 0
            assert row["prompt_tokens_count_0"] > 0
            assert row["output_tokens_count_0"] > 0

        # All 10 image payloads must be byte-different (cache-bust guarantee).
        digests = {
            hashlib.sha256(r["image"]["image"].encode()).hexdigest() for r in rows
        }
        assert len(digests) == 10

    @pytest.mark.sanity
    def test_factory_dispatch_via_explicit_kind(self):
        """## WRITTEN BY AI ##"""
        config = DataArgs.model_validate(
            {
                "kind": "synthetic_image",
                "width": 320,
                "height": 240,
                "text_tokens": 15,
                "output_tokens": 4,
            }
        )
        ds = DatasetDeserializerFactory.deserialize(
            config=config,
            processor_factory=_mock_tokenizer,
            random_seed=42,
        )
        assert isinstance(ds, SyntheticImageDataset)

    @pytest.mark.regression
    def test_images_per_request_emits_indexed_columns(self):
        """## WRITTEN BY AI ##"""
        d = SyntheticImageDatasetDeserializer()
        ds = d(
            config=SyntheticImageDataArgs(
                width=64,
                height=64,
                images_per_request=3,
                text_tokens=5,
                output_tokens=2,
            ),
            processor_factory=_mock_tokenizer,
            random_seed=42,
        )
        row = next(iter(ds))
        assert "image_0" in row
        assert "image_1" in row
        assert "image_2" in row
        # All three images in the same row should be byte-different.
        digests = {row[f"image_{i}"]["image"] for i in range(3)}
        assert len(digests) == 3


class TestSyntheticVideoDeserializer:
    @pytest.mark.smoke
    def test_pull_10_rows_from_config(self):
        """## WRITTEN BY AI ##"""
        d = SyntheticVideoDatasetDeserializer()
        ds = d(
            config=SyntheticVideoDataArgs(
                width=320,
                height=240,
                frames=4,
                fps=1,
                text_tokens=10,
                output_tokens=4,
                seed=17,
            ),
            processor_factory=_mock_tokenizer,
            random_seed=42,
        )
        assert isinstance(ds, SyntheticVideoDataset)

        rows = []
        it = iter(ds)
        for _ in range(10):
            rows.append(next(it))

        assert len(rows) == 10
        for row in rows:
            assert "video" in row
            assert row["video"]["video_frames"] == 4
            assert row["video"]["video_seconds"] == 4.0
            assert row["video"]["video_bytes"] > 0

        digests = {
            hashlib.sha256(r["video"]["video"].encode()).hexdigest() for r in rows
        }
        assert len(digests) == 10

    @pytest.mark.sanity
    def test_factory_dispatch_via_explicit_kind(self):
        """## WRITTEN BY AI ##"""
        config = DataArgs.model_validate(
            {
                "kind": "synthetic_video",
                "width": 160,
                "height": 120,
                "frames": 3,
                "fps": 1,
                "text_tokens": 10,
                "output_tokens": 4,
            }
        )
        ds = DatasetDeserializerFactory.deserialize(
            config=config,
            processor_factory=_mock_tokenizer,
            random_seed=42,
        )
        assert isinstance(ds, SyntheticVideoDataset)

    @pytest.mark.smoke
    def test_video_config_via_json(self):
        """## WRITTEN BY AI ##"""
        cfg = SyntheticVideoDataArgs.model_validate(
            {
                "kind": "synthetic_video",
                "width": 320,
                "height": 240,
                "frames": 4,
                "fps": 1,
                "text_tokens": 10,
                "video_bitrate": "200k",
            }
        )
        assert cfg.width == 320
        assert cfg.frames == 4
        assert cfg.video_bitrate == "200k"


# ---------------------------------------------------------------------------
# End-to-end reproducibility across the deserializer (not just the helpers)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_full_dataset_reproducible_with_same_seed():
    """Two datasets with the same seed must produce identical per-row sha256.

    ## WRITTEN BY AI ##
    """
    d = SyntheticImageDatasetDeserializer()
    config = SyntheticImageDataArgs(
        width=128,
        height=128,
        text_tokens=10,
        output_tokens=2,
        seed=999,
    )
    ds_a = d(config=config, processor_factory=_mock_tokenizer, random_seed=42)
    ds_b = d(config=config, processor_factory=_mock_tokenizer, random_seed=42)

    digests_a = []
    digests_b = []
    it_a, it_b = iter(ds_a), iter(ds_b)
    for _ in range(10):
        digests_a.append(
            hashlib.sha256(next(it_a)["image"]["image"].encode()).hexdigest()
        )
        digests_b.append(
            hashlib.sha256(next(it_b)["image"]["image"].encode()).hexdigest()
        )
    assert digests_a == digests_b
