import base64
import io
from unittest.mock import patch

import httpx2
import pytest
from PIL import Image

from guidellm.utils.vision import encode_image


@pytest.mark.regression
@pytest.mark.parametrize(
    ("resize_kwargs", "expected_size"),
    [
        ({"width": 40}, (40, 20)),
        ({"height": 30}, (60, 30)),
        ({"width": 40, "height": 30}, (40, 30)),
        ({"width": 160, "max_height": 20}, (40, 20)),
        ({"max_size": 40}, (40, 20)),
    ],
)
def test_encode_image_url_preserves_resize_options(resize_kwargs, expected_size):
    """Downloaded images honor the same resize options as bytes. ## WRITTEN BY AI ##"""
    image_buffer = io.BytesIO()
    Image.new("RGB", (80, 40), color="red").save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    url = "https://example.com/image.png"
    response = httpx2.Response(
        200, content=image_bytes, request=httpx2.Request("GET", url)
    )

    with patch("guidellm.utils.vision.httpx2.get", return_value=response):
        result = encode_image(url, **resize_kwargs)
    local_result = encode_image(image_bytes, **resize_kwargs)

    decoded = Image.open(io.BytesIO(base64.b64decode(result["image"].split(",", 1)[1])))
    assert decoded.size == expected_size
    assert result["image_pixels"] == expected_size[0] * expected_size[1]
    assert result == local_result


@pytest.mark.regression
@pytest.mark.parametrize(
    ("image_size", "resize_kwargs", "expected_size"),
    [
        ((1000, 1), {"width": 100}, (100, 1)),
        ((1, 1000), {"height": 100}, (1, 100)),
        ((1000, 1), {"max_size": 100}, (100, 1)),
        ((1, 1000), {"max_size": 100}, (1, 100)),
        ((1000, 1), {"max_width": 100}, (100, 1)),
        ((1, 1000), {"max_height": 100}, (1, 100)),
        ((1000, 1), {"width": 100, "max_height": 10}, (100, 1)),
        ((1, 1000), {"height": 100, "max_width": 10}, (1, 100)),
    ],
)
def test_encode_image_preserves_nonzero_dimensions(
    image_size, resize_kwargs, expected_size
):
    """Downscaling thin images retains at least one pixel. ## WRITTEN BY AI ##"""
    result = encode_image(Image.new("RGB", image_size), **resize_kwargs)

    decoded = Image.open(io.BytesIO(base64.b64decode(result["image"].split(",", 1)[1])))
    assert decoded.size == expected_size
    assert result["image_pixels"] == expected_size[0] * expected_size[1]


@pytest.mark.regression
@pytest.mark.parametrize(
    "dimension", ["width", "height", "max_size", "max_width", "max_height"]
)
@pytest.mark.parametrize("value", [0, -1])
def test_encode_image_rejects_nonpositive_dimensions(dimension, value):
    """Invalid dimensions must not be rounded up to one pixel. ## WRITTEN BY AI ##"""
    with pytest.raises(ValueError):
        encode_image(Image.new("RGB", (100, 50)), **{dimension: value})
