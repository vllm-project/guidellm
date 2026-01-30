from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import ValidationError

from guidellm.data.schemas import DataConfig, DataNotSupportedError
from guidellm.utils import arg_string

ConfigT = TypeVar("ConfigT", bound=DataConfig)


def load_config(config: Any, config_class: type[ConfigT]) -> ConfigT | None:
    # Try file path first
    if (loaded_config := _load_config_file(config, config_class)) is not None:
        return loaded_config

    # Try dict parsing next
    if (loaded_config := _load_config_dict(config, config_class)) is not None:
        return loaded_config

    # Try string parsing
    if (loaded_config := _load_config_str(config, config_class)) is not None:
        return loaded_config

    return None


def _load_config_dict(data: Any, config_class: type[ConfigT]) -> ConfigT | None:
    if not isinstance(data, dict | list):
        return None

    try:
        return config_class.model_validate(data)
    except ValidationError:
        return None


def _load_config_file(data: Any, config_class: type[ConfigT]) -> ConfigT | None:
    if (not isinstance(data, str) and not isinstance(data, Path)) or (
        not Path(data).is_file()
    ):
        return None

    data_path = Path(data) if isinstance(data, str) else data
    error = None

    if Path(data).is_file() and data_path.suffix.lower() == ".json":
        try:
            return config_class.model_validate_json(data_path.read_text())
        except Exception as err:  # noqa: BLE001
            error = err

    if Path(data).is_file() and data_path.suffix.lower() in {
        ".yaml",
        ".yml",
        ".config",
    }:
        try:
            return config_class.model_validate(yaml.safe_load(data_path.read_text()))
        except Exception as err:  # noqa: BLE001
            error = err

    err_message = (
        f"Unsupported file {data_path} for "
        f"{config_class.__name__}, expected .json, "
        f".yaml, .yml, or .config"
    )

    if error is not None:
        err_message += f" with error: {error}"
        raise DataNotSupportedError(err_message) from error
    raise DataNotSupportedError(err_message)


def _load_config_str(data: str, config_class: type[ConfigT]) -> ConfigT | None:
    if not isinstance(data, str):
        return None

    err_message = (
        f"Unsupported string data for {config_class.__name__}, "
        f"expected JSON or key-value pairs, got {data}"
    )

    try:
        data_parsed = yaml.safe_load(data)
    except yaml.YAMLError:
        data_parsed = data

    # If no change from YAML parsing, try arg_string parsing
    if data_parsed == data:
        try:
            data_parsed = arg_string.loads(data_parsed)
        except arg_string.ArgStringParseError as e:
            raise DataNotSupportedError(err_message) from e

    try:
        return config_class.model_validate(data_parsed)
    except ValidationError as err:
        raise DataNotSupportedError(err_message) from err
