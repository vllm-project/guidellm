import base64
import io
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

from guidellm.utils.vision import encode_image, encode_video, image_dict_to_pil


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


@pytest.mark.regression
@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("media_kind", ["image", "pil", "video"])
def test_remote_media_follows_redirects(httpx_mock, status_code, media_kind):
    """Encode the final media payload after relative and absolute redirects.

    ## WRITTEN BY AI ##
    """
    if media_kind == "video":
        payload = b"encoded-video-fixture"
    else:
        buffer = io.BytesIO()
        Image.new("RGB", (8, 4), color="red").save(buffer, format="PNG")
        payload = buffer.getvalue()

    url = f"https://media.example/{media_kind}"
    final_url = f"https://cdn.example/{media_kind}"
    httpx_mock.add_response(
        url=url, status_code=status_code, headers={"Location": "/asset"}
    )
    httpx_mock.add_response(
        url="https://media.example/asset",
        status_code=302,
        headers={"Location": final_url},
    )
    httpx_mock.add_response(url=final_url, content=payload)

    if media_kind == "image":
        assert encode_image(url, width=4) == encode_image(payload, width=4)
    elif media_kind == "pil":
        decoded = image_dict_to_pil({"image": url})
        assert decoded.size == (8, 4)
        assert decoded.getpixel((0, 0)) == (255, 0, 0)
    elif media_kind == "video":
        result = encode_video(url)
        assert base64.b64decode(result["video"].split(",", 1)[1]) == payload
        assert result["video_bytes"] == len(payload)
    assert len(httpx_mock.get_requests()) == 3


@pytest.mark.regression
def test_redirect_destination_error_is_reported(httpx_mock):
    """Keep HTTP error reporting for the final redirected response.

    ## WRITTEN BY AI ##
    """
    url = "https://media.example/image"
    httpx_mock.add_response(url=url, status_code=302, headers={"Location": "/missing"})
    httpx_mock.add_response(url="https://media.example/missing", status_code=404)

    with pytest.raises(httpx.HTTPStatusError) as error:
        encode_image(url)
    assert error.value.response.status_code == 404
