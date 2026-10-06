import base64
import io
import wave

import numpy as np
import pytest
from datasets import Dataset
from PIL import Image

from guidellm.data.preprocessors.encoders import MediaEncoder
from guidellm.data.preprocessors.mappers import GenerativeColumnMapper
from guidellm.schemas.data.preprocessors import (
    GenerativeColumnMapperArgs,
    MediaEncoderArgs,
)


@pytest.mark.regression
@pytest.mark.parametrize("shape", [(4, 5, 3), (1, 1)])
def test_encode_image_array_from_formatted_dataset(shape):
    """Encode both multi-element and single-pixel all-zero image arrays.

    ## WRITTEN BY AI ##
    """
    dataset = Dataset.from_dict({"image": [np.zeros(shape, dtype=np.uint8)]})
    dataset = dataset.with_format("numpy", dtype=np.uint8)
    mapper = GenerativeColumnMapper(GenerativeColumnMapperArgs())
    mapper.setup_data([dataset])
    encoder = MediaEncoder(MediaEncoderArgs())

    turns = encoder(mapper([{"dataset": dataset[0]}]))

    assert len(turns[0]["image_column"]) == 1
    image = turns[0]["image_column"][0]
    assert image["image_pixels"] == shape[0] * shape[1]
    payload = base64.b64decode(image["image"].split(",", 1)[1])
    with Image.open(io.BytesIO(payload)) as decoded:
        assert decoded.size == (shape[1], shape[0])


@pytest.mark.regression
@pytest.mark.parametrize("format_type", ["numpy", "torch"])
def test_encode_audio_array_from_formatted_dataset(format_type):
    """Encode raw audio arrays without evaluating their samples as booleans.

    ## WRITTEN BY AI ##
    """
    dataset = Dataset.from_dict({"audio": [np.zeros((1, 1600), dtype=np.float32)]})
    dataset = dataset.with_format(format_type)
    mapper = GenerativeColumnMapper(GenerativeColumnMapperArgs())
    mapper.setup_data([dataset])
    encoder = MediaEncoder(
        MediaEncoderArgs(audio_kwargs={"sample_rate": 16000, "audio_format": "wav"})
    )

    turns = encoder(mapper([{"dataset": dataset[0]}]))

    assert len(turns[0]["audio_column"]) == 1
    audio = turns[0]["audio_column"][0]
    assert audio["audio_samples"] == 1600
    with wave.open(io.BytesIO(audio["audio"]), "rb") as decoded:
        assert decoded.getnframes() == 1600
        assert decoded.getframerate() == 16000


@pytest.mark.regression
@pytest.mark.parametrize("column", ["image_column", "audio_column", "video_column"])
def test_empty_media_placeholders_are_skipped(column):
    """Keep missing and empty media placeholders out of the encoded turn.

    ## WRITTEN BY AI ##
    """
    encoder = MediaEncoder(MediaEncoderArgs())

    assert encoder([{column: [None, "", b"", {}, []]}]) == [{column: []}]
