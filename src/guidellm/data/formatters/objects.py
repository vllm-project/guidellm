from __future__ import annotations

from typing import Any, Literal

from datasets import Dataset, IterableDataset
from jinja2 import Template

from guidellm.data.formatters import JinjaEnvironmentMixin
from guidellm.data.objects import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerativeDatasetArgs,
    GenerativeRequestType,
)
from guidellm.data.preprocessors.objects import DatasetPreprocessor

__all__ = ["GenerativeRequestFormatter"]


class GenerativeRequestFormatter(DatasetPreprocessor, JinjaEnvironmentMixin):
    def __init__(
        self,
        request_type: GenerativeRequestType | str = "text_completions",
        request_template: str | Template | None = None,
        request_extras: dict[str, Any] | GenerationRequestArguments | None = None,
        request_defaults: dict[str, Any] | GenerationRequestArguments | None = None,
        environment_extras: dict[str, Any] | None = None,
    ):
        self.datasets: list[Dataset | IterableDataset] | None = None
        self.data_args: list[GenerativeDatasetArgs] | None = None

        self.request_type = request_type
        self.request_template = request_template
        self.request_extras = request_extras or {}
        self.request_defaults = request_defaults or {
            "stream": True,
            "json_body": {
                "stream": True,
                "stream_options": {
                    "include_usage": True,
                },
            },
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
        if self.jinja_template is None:
            raise ValueError("GenerativeRequestCreator not initialized with data.")

        stats = {}
        if "prompt_tokens_count" in item:
            count = item["prompt_tokens_count"][0]
            stats["prompt_tokens"] = count
            item["prompt_tokens_count"] = count
        if "output_tokens_count" in item:
            count = item["output_tokens_count"][0]
            stats["output_tokens"] = count
            item["output_tokens_count"] = count

        return {
            "request": {
                "request_type": self.request_type,
                "arguments": GenerationRequestArguments.model_combine_dict(
                    self.request_defaults,
                    self.request_extras,
                    self.jinja_template.render(
                        **item,
                        request_defaults=self.request_defaults,
                        request_extras=self.request_extras,
                    ),
                ),
                "stats": stats,
            }
        }
