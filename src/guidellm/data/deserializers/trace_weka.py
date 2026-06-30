"""
The WEKA trace format and data arguments.

TODO
"""

from typing import Literal

from datasets import Features
from faker import Faker
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializerFactory,
)
from guidellm.data.deserializers.trace_common import (
    TraceDataArgs,
    TraceDatasetDeserializer,
    TraceFormatBase,
    TraceFormatRegistry,
    decode_prompt,
    generate_token_ids,
)
from guidellm.data.schemas import DataArgs

__all__ = ["WEKATraceFormatArgs"]


DatasetDeserializerFactory.register_decorator(TraceDatasetDeserializer, "weka")


@DataArgs.register("weka")
class WEKATraceFormatArgs(TraceDataArgs):
    kind: Literal["weka"] = Field(
        default="weka",
        description="Type identifier for the WEKA trace format.",
    )


@TraceFormatRegistry.register("weka")
class WEKATraceFormat(TraceFormatBase):
    """TODO"""

    def __init__(self) -> None:
        pass

    def required_columns(
        self,
        config: WEKATraceFormatArgs,  # noqa: ARG002
    ) -> Features:
        return []

    def validate_row(
        self,
        config: WEKATraceFormatArgs,  # noqa: ARG002
        row: dict,  # noqa: ARG002
    ) -> None:
        return

    def create_prompt(
        self,
        config: WEKATraceFormatArgs,
        row: dict,
        processor: PreTrainedTokenizerBase,
        faker: Faker,
    ) -> str:
        token_ids = generate_token_ids(
            row[config.prompt_tokens_column], processor, faker
        )
        return decode_prompt(processor, token_ids)
