import base64
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from guidellm.extras.vision import (
    encode_image,
    encode_video,
    get_file_format,
    resize_image,
)


@pytest.fixture
def sample_jpeg_file():
    # Create a valid JPEG image
    rng = np.random.default_rng(42)
    img_array = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name, format="JPEG", quality=95)
        temp_path = Path(f.name)

    yield temp_path
    # Clean up
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_image_array(sample_jpeg_file) -> np.ndarray:
    img = Image.open(sample_jpeg_file)
    return np.array(img)


@pytest.fixture
def sample_image_bytes(sample_jpeg_file) -> bytes:
    with Path.open(sample_jpeg_file, "rb") as f:
        return f.read()


# Fixture for common test video
@pytest.fixture
def sample_video_file():
    """Create a temporary video file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"sample video content for testing")
        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


def test_encode_image_base64(sample_image_bytes: bytes):
    result = encode_image(sample_image_bytes, encode_type="base64")
    assert result["type"] == "image_base64"
    assert "image" in result
    assert result["image_bytes"] > 0
    assert result["image_pixels"] > 0


def test_encode_image_url():
    result = encode_image(image="https://example.com/vision.jpg", encode_type="url")
    assert result["type"] == "image_url"
    assert result["image"] == "https://example.com/vision.jpg"


def test_resize_image(sample_image_array: np.ndarray):
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(sample_image_array)

    original_height, original_width = sample_image_array.shape[:2]
    new_width, new_height = 100, 100

    resized_image = resize_image(
        pil_image,  # Pass PIL Image instead of numpy array
        width=new_width,
        height=new_height,
    )
    assert isinstance(resized_image, Image.Image)
    assert resized_image.size == (new_width, new_height)


def test_get_file_format(sample_jpeg_file):
    file_format = get_file_format(sample_jpeg_file)
    assert file_format == "jpg"


def test_encode_video_with_fixture(sample_video_file):
    result = encode_video(video=sample_video_file, encode_type="base64")

    assert result["type"] == "video_base64"
    assert result["video"].startswith("data:video/mp4;base64,")
    assert result["video_bytes"] == 32  # Length of "sample video content for testing"


def test_encode_video_with_url_base64():
    """Test encoding a video URL with base64 encoding"""
    test_url = "https://example.com/video.mp4"
    mock_video_content = b"fake video content"

    with patch("httpx.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = mock_video_content
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = encode_video(video=test_url, encode_type="base64")

        mock_get.assert_called_once_with(test_url)
        assert result["type"] == "video_base64"
        assert result["video"].startswith("data:video/unknown;base64,")
        assert result["video_bytes"] == len(mock_video_content)
        assert result["video_frames"] is None
        assert result["video_seconds"] is None


def test_encode_video_with_url_url_encoding():
    """Test encoding a video URL with url encoding"""
    test_url = "https://example.com/video.mp4"
    result = encode_video(video=test_url, encode_type="url")

    assert result["type"] == "video_url"
    assert result["video"] == test_url
    assert result["video_frames"] is None
    assert result["video_seconds"] is None
    assert result["video_bytes"] is None


def test_encode_video_with_base64_string():
    """Test encoding an already base64 encoded video string"""
    test_video_bytes = b"fake video content"
    base64_video = base64.b64encode(test_video_bytes).decode("utf-8")
    data_url = f"data:video/mp4;base64,{base64_video}"

    result = encode_video(video=data_url, encode_type="base64")

    assert result["type"] == "video_base64"
    assert result["video"] == data_url
    assert result["video_bytes"] == len(base64_video) * 3 // 4
    assert result["video_frames"] is None
    assert result["video_seconds"] is None


def test_encode_video_with_file_path(sample_video_file):
    result = encode_video(video=sample_video_file, encode_type="base64")

    assert result["type"] == "video_base64"
    assert result["video"].startswith("data:video/mp4;base64,")
    assert result["video_bytes"] == len(b"sample video content for testing")
    assert result["video_frames"] is None
    assert result["video_seconds"] is None

    # Verify base64 encoding is correct
    base64_part = result["video"].split(",", 1)[1]
    decoded_bytes = base64.b64decode(base64_part)
    assert decoded_bytes == b"sample video content for testing"


def test_encode_video_with_path_object():
    """Test encoding a video from Path object"""
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f:
        try:
            mp4_content = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
            f.write(mp4_content)
            f.flush()
            temp_path = Path(f.name)

            result = encode_video(video=temp_path, encode_type="base64")

            assert result["type"] == "video_base64"
            assert result["video"].startswith("data:video/avi;base64,")
            assert result["video_bytes"] == len(mp4_content)
            assert result["video_frames"] is None
            assert result["video_seconds"] is None
        finally:
            if temp_path.exists():
                temp_path.unlink()


def test_encode_video_with_raw_bytes():
    """Test encoding video from raw bytes"""
    video_bytes = b"raw video bytes content"

    result = encode_video(video=video_bytes, encode_type="base64")

    assert result["type"] == "video_base64"
    assert result["video"].startswith("data:video/unknown;base64,")
    assert result["video_bytes"] == len(video_bytes)
    assert result["video_frames"] is None
    assert result["video_seconds"] is None

    # Verify base64 encoding
    base64_part = result["video"].split(",", 1)[1]
    decoded_bytes = base64.b64decode(base64_part)
    assert decoded_bytes == video_bytes


def test_encode_video_url_with_http_error():
    """Test URL encoding when HTTP request fails"""
    test_url = "https://example.com/video.mp4"

    with patch("httpx.get") as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response

        with pytest.raises(Exception, match="HTTP Error"):
            encode_video(video=test_url, encode_type="base64")


def test_encode_video_with_none_encode_type():
    """Test encoding with None encode_type"""
    video_bytes = b"test video"

    result = encode_video(video=video_bytes, encode_type=None)

    # Should default to base64 encoding
    assert result["type"] == "video_base64"
    assert result["video"].startswith("data:video/unknown;base64,")


def test_encode_video_with_unsupported_type():
    """Test encoding with unsupported video type"""
    with pytest.raises(ValueError, match="Unsupported video type"):
        encode_video(video=123, encode_type="base64")  # int is not supported


def test_encode_video_file_not_found():
    """Test encoding with non-existent file path"""
    non_existent_path = "/path/that/does/not/exist/video.mp4"

    with pytest.raises(FileNotFoundError):
        encode_video(video=non_existent_path, encode_type="base64")


def test_encode_video_base64_correctness():
    """Test that base64 encoding is mathematically correct"""
    # Use a known input to verify base64 encoding
    test_bytes = b"Hello World"
    expected_base64 = base64.b64encode(test_bytes).decode("utf-8")

    result = encode_video(video=test_bytes, encode_type="base64")

    base64_part = result["video"].split(",", 1)[1]
    assert base64_part == expected_base64
    assert result["video_bytes"] == len(test_bytes)


def test_encode_video_data_url_format():
    """Test that data URL format is correct"""
    video_bytes = b"test video data"

    result = encode_video(video=video_bytes, encode_type="base64")

    assert result["video"].startswith("data:video/unknown;base64,")
    # Verify the format is exactly as expected
    parts = result["video"].split(",", 1)
    assert len(parts) == 2
    assert parts[0] == "data:video/unknown;base64"
    assert base64.b64decode(parts[1]) == video_bytes


# Additional test for edge cases
def test_encode_video_empty_bytes():
    """Test encoding empty video bytes"""
    result = encode_video(video=b"", encode_type="base64")

    assert result["type"] == "video_base64"
    assert result["video"] == "data:video/unknown;base64,"
    assert result["video_bytes"] == 0


def test_encode_video_large_content():
    """Test encoding with larger video content"""
    large_content = b"x" * 1024 * 1024  # 1MB of data

    result = encode_video(video=large_content, encode_type="base64")

    assert result["type"] == "video_base64"
    assert result["video_bytes"] == len(large_content)
    # Verify we can decode it back
    base64_part = result["video"].split(",", 1)[1]
    decoded = base64.b64decode(base64_part)
    assert decoded == large_content
