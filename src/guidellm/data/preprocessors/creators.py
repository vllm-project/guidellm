from __future__ import annotations

from typing import Any, Literal

from datasets import Dataset, IterableDataset
from jinja2 import Template

from guidellm.data.formatters import JinjaEnvironmentMixin
from guidellm.data.objects import (
    GenerationRequest,
    GenerationRequestPayload,
    GenerativeDatasetArgs,
    GenerativeRequestType,
)
from guidellm.data.preprocessors.objects import DatasetPreprocessor

__all__ = ["GenerativeRequestCreator"]


class GenerativeRequestCreator(DatasetPreprocessor, JinjaEnvironmentMixin):
    def __init__(
        self,
        request_type: GenerativeRequestType | str = "text_completions",
        request_template: str | Template | None = None,
        payload_extras: dict[str, Any] | None = None,
        payload_defaults: dict[str, Any] | None = None,
        environment_extras: dict[str, Any] | None = None,
    ):
        self.datasets: list[Dataset | IterableDataset] | None = None
        self.data_args: list[GenerativeDatasetArgs] | None = None

        self.request_type = request_type
        self.request_template = request_template
        self.payload_extras = payload_extras or {}
        self.payload_defaults = payload_defaults or {
            "json": {
                "stream": True,
                "stream_options": {
                    "include_usage": True,
                },
            }
        }
        self.environment_extras = environment_extras or {}
        self.jinja_template: Template | None = None

    def init_data(
        self,
        datasets: list[Dataset | IterableDataset],
        data_args: list[GenerativeDatasetArgs],
    ):
        self.datasets = datasets
        self.data_args = data_args

        self.create_environment(**self.environment_extras)
        self.jinja_template = (
            self.template_from_source(self.request_template)
            if self.request_template
            else self.template_from_registry(self.request_type)
        )

    def __call__(
        self, item: dict[str, Any]
    ) -> dict[Literal["request"], GenerationRequest]:
        if self.request_template is None:
            raise ValueError("GenerativeRequestCreator not initialized with data.")

        stats = (
            {"prompt_tokens": item["prompt_tokens_count_column"]}
            if "prompt_tokens_count_column" in item
            else {}
        )
        constraints = (
            {"output_tokens": item["output_tokens_count_column"]}
            if "output_tokens_count_column" in item
            else {}
        )
        payload = {}
        payload.update(self.payload_defaults)
        payload.update(self.payload_extras)
        payload.update(self.jinja_template.render(**item))

        return {
            "request": GenerationRequest(
                request_type=self.request_type,
                payload=GenerationRequestPayload(**payload),
                stats=stats,
                constraints=constraints,
            )
        }
