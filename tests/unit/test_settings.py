import os

import pytest

from guidellm.settings import (
    DatasetSettings,
    LoggingSettings,
    Settings,
    print_config,
    reload_settings,
    settings,
)


@pytest.mark.smoke
def test_default_settings(mocker):
    """
    Default settings should load without a remote HTML report source.

    ## WRITTEN BY AI ##
    """
    mocker.patch.dict(
        "os.environ",
        {k: v for k, v in os.environ.items() if not k.startswith("GUIDELLM__")},
        clear=True,
    )
    loaded = Settings(_env_file=None)
    assert loaded.logging == LoggingSettings()
    assert "report_generation" not in Settings.model_fields


@pytest.mark.smoke
def test_settings_from_env_variables(mocker):
    """
    Nested logging env vars should still apply.

    ## WRITTEN BY AI ##
    """
    mocker.patch.dict(
        "os.environ",
        {
            "GUIDELLM__logging__disabled": "true",
        },
    )

    loaded = Settings()
    assert loaded.logging.disabled is True


@pytest.mark.sanity
def test_logging_settings():
    logging_settings = LoggingSettings(
        disabled=True,
        console_log_level="DEBUG",
        log_file="app.log",
        log_file_level="ERROR",
    )
    assert logging_settings.disabled is True
    assert logging_settings.console_log_level == "DEBUG"
    assert logging_settings.log_file == "app.log"
    assert logging_settings.log_file_level == "ERROR"


@pytest.mark.sanity
def test_generate_env_file(mocker):
    mocker.patch.dict(
        "os.environ",
        {k: v for k, v in os.environ.items() if not k.startswith("GUIDELLM__")},
        clear=True,
    )
    loaded = Settings(_env_file=None)
    env_file_content = loaded.generate_env_file()
    assert "GUIDELLM__LOGGING__DISABLED" in env_file_content
    assert "REPORT_GENERATION" not in env_file_content


@pytest.mark.sanity
def test_reload_settings(mocker):
    mocker.patch.dict(
        "os.environ",
        {
            "GUIDELLM__logging__disabled": "false",
        },
    )
    reload_settings()
    assert settings.logging.disabled is False


@pytest.mark.sanity
def test_print_config(capsys):
    print_config()
    captured = capsys.readouterr()
    assert "Settings:" in captured.out
    assert "GUIDELLM__LOGGING__DISABLED" in captured.out


@pytest.mark.sanity
def test_dataset_settings_defaults():
    dataset_settings = DatasetSettings()
    assert dataset_settings.preferred_data_columns == [
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
    assert dataset_settings.preferred_data_splits == [
        "test",
        "tst",
        "validation",
        "val",
        "train",
    ]


@pytest.mark.sanity
def test_table_properties_defaults(mocker):
    mocker.patch.dict(
        "os.environ",
        {k: v for k, v in os.environ.items() if not k.startswith("GUIDELLM__")},
        clear=True,
    )
    loaded = Settings(_env_file=None)
    assert loaded.table_border_char == "="
    assert loaded.table_headers_border_char == "-"
    assert loaded.table_column_separator_char == "|"


@pytest.mark.sanity
def test_settings_with_env_variables(mocker):
    mocker.patch.dict(
        "os.environ",
        {
            "GUIDELLM__DATASET__PREFERRED_DATA_COLUMNS": '["custom_column"]',
            "GUIDELLM__TABLE_BORDER_CHAR": "*",
        },
    )
    loaded = Settings()
    assert loaded.dataset.preferred_data_columns == ["custom_column"]
    assert loaded.table_border_char == "*"
