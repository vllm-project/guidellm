import base64
import io
from unittest.mock import patch

import httpx
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
    response = httpx.Response(
        200, content=image_bytes, request=httpx.Request("GET", url)
    )

    with patch("guidellm.utils.vision.httpx.get", return_value=response):
        result = encode_image(url, **resize_kwargs)
    local_result = encode_image(image_bytes, **resize_kwargs)

    decoded = Image.open(io.BytesIO(base64.b64decode(result["image"].split(",", 1)[1])))
    assert decoded.size == expected_size
    assert result["image_pixels"] == expected_size[0] * expected_size[1]
    assert result == local_result
