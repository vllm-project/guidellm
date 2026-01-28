from __future__ import annotations

import json
from collections.abc import Sequence
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "DatasetSettings",
    "Environment",
    "LoggingSettings",
    "Settings",
    "print_config",
    "reload_settings",
    "settings",
]


class Environment(str, Enum):
    """
    Enum for the supported environments
    """

    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


ENV_REPORT_MAPPING = {
    Environment.PROD: "https://blog.vllm.ai/guidellm/ui/v0.5.0/index.html",
    Environment.STAGING: "https://blog.vllm.ai/guidellm/ui/release/v0.4.0/index.html",
    Environment.DEV: "https://blog.vllm.ai/guidellm/ui/dev/index.html",
    Environment.LOCAL: "http://localhost:3000/index.html",
}


class LoggingSettings(BaseModel):
    """
    Logging settings for the application
    """

    disabled: bool = False
    clear_loggers: bool = True
    console_log_level: str = "WARNING"
    log_file: str | None = None
    log_file_level: str | None = None


class DatasetSettings(BaseModel):
    """
    Dataset settings for the application
    """

    preferred_data_columns: list[str] = Field(
        default_factory=lambda: [
            "prompt",
            "instruction",
            "input",
            "inputs",
            "question",
            "context",
            "text",
            "content",
            "body",
            "data",
        ]
    )
    preferred_data_splits: list[str] = Field(
        default_factory=lambda: ["test", "tst", "validation", "val", "train"]
    )


class ReportGenerationSettings(BaseModel):
    """
    Report generation settings for the application
    """

    source: str = ""


class Settings(BaseSettings):
    """
    All the settings are powered by pydantic_settings and could be
    populated from the .env file.

    The format to populate the settings is next

    ```sh
    export GUIDELLM__LOGGING__DISABLED=true
    ```
    """

    model_config = SettingsConfigDict(
        env_prefix="GUIDELLM__",
        env_nested_delimiter="__",
        extra="ignore",
        validate_default=True,
        env_file=".env",
    )

    # general settings
    env: Environment = Environment.PROD
    default_async_loop_sleep: float = 10e-5
    logging: LoggingSettings = LoggingSettings()
    default_sweep_number: int = 10

    # HTTP settings
    request_follow_redirects: bool = True
    request_timeout: int = 60 * 5  # 5 minutes
    request_http2: bool = True

    # Scheduler settings
    mp_context_type: Literal["spawn", "fork", "forkserver"] | None = "fork"
    mp_serialization: Literal["dict", "sequence"] | None = "dict"
    mp_encoding: (
        Literal["msgpack", "msgspec"]
        | None
        | list[Literal["msgpack", "msgspec"] | None]
    ) = ["msgspec", "msgpack", None]
    mp_messaging_object: Literal["queue", "manager_queue", "pipe"] = "queue"
    mp_requests_send_buffer_size: int = 1
    mp_poll_interval: float = 0.1
    mp_max_pending_buffer_percent: float = 0.5
    mp_max_worker_buffer_percent: float = 0.2
    max_concurrency: int = 512
    max_worker_processes: int = 10
    scheduler_start_delay_non_distributed: float = 1.0
    constraint_error_window_size: float = 30
    constraint_error_min_processed: float = 30

    # Data settings
    dataset: DatasetSettings = DatasetSettings()

    # Request/stats settings
    preferred_prompt_tokens_source: Literal["request", "response"] = "response"
    preferred_output_tokens_source: Literal["request", "response"] = "response"
    preferred_backend: Literal["openai"] = "openai"
    preferred_route: Literal["text_completions", "chat_completions"] = (
        "chat_completions"
    )

    # Report settings
    report_generation: ReportGenerationSettings = ReportGenerationSettings()

    # Output settings
    table_border_char: str = "="
    table_headers_border_char: str = "-"
    table_column_separator_char: str = "|"

    @model_validator(mode="after")
    @classmethod
    def set_default_source(cls, values):
        if not values.report_generation.source:
            values.report_generation.source = ENV_REPORT_MAPPING.get(values.env)
        return values

    def generate_env_file(self) -> str:
        """
        Generate the .env file from the current settings
        """
        return Settings._recursive_generate_env(
            self,
            self.model_config["env_prefix"],  # type: ignore  # noqa: PGH003
            self.model_config["env_nested_delimiter"],  # type: ignore  # noqa: PGH003
        )

    @staticmethod
    def _recursive_generate_env(model: BaseModel, prefix: str, delimiter: str) -> str:
        env_file = ""
        add_models = []
        for key, value in model.model_dump().items():
            if isinstance(value, BaseModel):
                # add nested properties to be processed after the current level
                add_models.append((key, value))
                continue

            dict_values = (
                {
                    f"{prefix}{key.upper()}{delimiter}{sub_key.upper()}": sub_value
                    for sub_key, sub_value in value.items()
                }
                if isinstance(value, dict)
                else {f"{prefix}{key.upper()}": value}
            )

            for tag, sub_value in dict_values.items():
                if isinstance(sub_value, Sequence) and not isinstance(sub_value, str):
                    value_str = ",".join(f'"{item}"' for item in sub_value)
                    env_file += f"{tag}=[{value_str}]\n"
                elif isinstance(sub_value, dict):
                    value_str = json.dumps(sub_value)
                    env_file += f"{tag}={value_str}\n"
                elif not sub_value:
                    env_file += f"{tag}=\n"
                else:
                    env_file += f'{tag}="{sub_value}"\n'

        for key, value in add_models:
            env_file += Settings._recursive_generate_env(
                value, f"{prefix}{key.upper()}{delimiter}", delimiter
            )
        return env_file


settings = Settings()


def reload_settings():
    """
    Reload the settings from the environment variables
    """
    new_settings = Settings()
    settings.__dict__.update(new_settings.__dict__)


def print_config():
    """
    Print the current configuration settings
    """
    print(f"Settings: \n{settings.generate_env_file()}")  # noqa: T201
