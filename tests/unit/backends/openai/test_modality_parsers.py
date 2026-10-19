"""Tests for registered non-text output parsers."""

from __future__ import annotations

import base64

import pytest

from guidellm.backends.openai import request_handlers as _request_handlers
from guidellm.backends.openai.modality_parsers import ModalityParserRegistry


@pytest.mark.regression
def test_audio_parser_is_registered_and_ignores_content_parts():
    """Register audio parsing while leaving content-list items unclaimed.

    :return: None.
    ## WRITTEN BY AI ##
    """
    parser_type = ModalityParserRegistry.get_registered_object("audio")

    assert parser_type is not None
    assert parser_type is _request_handlers._AudioOutputParser
    assert parser_type().parse_content_part({"type": "audio"}) is None
    assert parser_type().parse_dedicated_field(None) is None


@pytest.mark.regression
def test_audio_parser_normalizes_dedicated_audio_field():
    """Decode audio bytes and preserve optional response metadata.

    :return: None.
    ## WRITTEN BY AI ##
    """
    parser_type = ModalityParserRegistry.get_registered_object("audio")
    payload = b"audio-packet"
    assert parser_type is not None

    parsed = parser_type().parse_dedicated_field(
        {
            "id": "audio-1",
            "data": base64.b64encode(payload).decode("ascii"),
            "format": "wav",
            "transcript": "spoken words",
            "expires_at": 123,
        }
    )

    assert parsed == {
        "modality": "audio",
        "data": payload,
        "format": "wav",
        "transcript": "spoken words",
        "id": "audio-1",
        "expires_at": 123,
    }


@pytest.mark.regression
@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("audio-object", "not a dictionary"),
        ({}, "missing a non-empty string 'data'"),
        ({"data": "%%%"}, "not valid base64"),
        (
            {"data": base64.b64encode(b"").decode("ascii")},
            "missing a non-empty string 'data'",
        ),
    ],
)
def test_audio_parser_rejects_malformed_dedicated_fields(value, message):
    """Reject malformed dedicated audio objects instead of dropping them.

    :param value: Raw message.audio value to parse.
    :param message: Expected diagnostic fragment.
    :return: None.
    ## WRITTEN BY AI ##
    """
    parser_type = ModalityParserRegistry.get_registered_object("audio")
    assert parser_type is not None

    with pytest.raises(ValueError, match=message):
        parser_type().parse_dedicated_field(value)
