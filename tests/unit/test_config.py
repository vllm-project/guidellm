import pytest

from guidellm.config import (
    DatasetSettings,
    Environment,
    LoggingSettings,
    OpenAISettings,
    ReportGenerationSettings,
    Settings,
    print_config,
    reload_settings,
    settings,
)


@pytest.mark.smoke
def test_default_settings():
    settings = Settings()
    assert settings.env == Environment.PROD
    assert settings.logging == LoggingSettings()
    assert settings.openai == OpenAISettings()
    assert (
        settings.report_generation.source
        == "https://blog.vllm.ai/guidellm/ui/v0.3.0/index.html"
    )


@pytest.mark.smoke
def test_settings_from_env_variables(mocker):
    mocker.patch.dict(
        "os.environ",
        {
            "GUIDELLM__env": "dev",
            "GUIDELLM__logging__disabled": "true",
            "GUIDELLM__OPENAI__API_KEY": "test_key",
            "GUIDELLM__OPENAI__BASE_URL": "http://test.url",
            "GUIDELLM__REPORT_GENERATION__SOURCE": "http://custom.url",
        },
    )

    settings = Settings()
    assert settings.env == Environment.DEV
    assert settings.logging.disabled is True
    assert settings.openai.api_key == "test_key"
    assert settings.openai.base_url == "http://test.url"
    assert settings.report_generation.source == "http://custom.url"


@pytest.mark.smoke
def test_report_generation_default_source():
    settings = Settings(env=Environment.LOCAL)
    assert settings.report_generation.source == "http://localhost:3000/index.html"

    settings = Settings(env=Environment.DEV)
    assert (
        settings.report_generation.source
        == "https://blog.vllm.ai/guidellm/ui/dev/index.html"
    )

    settings = Settings(env=Environment.STAGING)
    assert (
        settings.report_generation.source
        == "https://blog.vllm.ai/guidellm/ui/release/v0.3.0/index.html"
    )

    settings = Settings(env=Environment.PROD)
    assert (
        settings.report_generation.source
        == "https://blog.vllm.ai/guidellm/ui/v0.3.0/index.html"
    )


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
def test_openai_settings():
    openai_settings = OpenAISettings(api_key="test_api_key", base_url="http://test.api")
    assert openai_settings.api_key == "test_api_key"
    assert openai_settings.base_url == "http://test.api"


def test_report_generation_settings():
    report_settings = ReportGenerationSettings(source="http://custom.report")
    assert report_settings.source == "http://custom.report"


@pytest.mark.sanity
def test_generate_env_file():
    settings = Settings()
    env_file_content = settings.generate_env_file()
    assert "GUIDELLM__LOGGING__DISABLED" in env_file_content
    assert "GUIDELLM__OPENAI__API_KEY" in env_file_content


@pytest.mark.sanity
def test_reload_settings(mocker):
    mocker.patch.dict(
        "os.environ",
        {
            "GUIDELLM__env": "staging",
            "GUIDELLM__logging__disabled": "false",
        },
    )
    reload_settings()
    assert settings.env == Environment.STAGING
    assert settings.logging.disabled is False


@pytest.mark.sanity
def test_print_config(capsys):
    print_config()
    captured = capsys.readouterr()
    assert "Settings:" in captured.out
    assert "GUIDELLM__LOGGING__DISABLED" in captured.out
    assert "GUIDELLM__OPENAI__API_KEY" in captured.out


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
def test_openai_settings_defaults():
    openai_settings = OpenAISettings()
    assert openai_settings.api_key is None
    assert openai_settings.bearer_token is None
    assert openai_settings.organization is None
    assert openai_settings.project is None
    assert openai_settings.base_url == "http://localhost:8000"
    assert openai_settings.max_output_tokens == 16384


@pytest.mark.sanity
def test_table_properties_defaults():
    settings = Settings()
    assert settings.table_border_char == "="
    assert settings.table_headers_border_char == "-"
    assert settings.table_column_separator_char == "|"


@pytest.mark.sanity
def test_settings_with_env_variables(mocker):
    mocker.patch.dict(
        "os.environ",
        {
            "GUIDELLM__DATASET__PREFERRED_DATA_COLUMNS": '["custom_column"]',
            "GUIDELLM__OPENAI__API_KEY": "env_api_key",
            "GUIDELLM__TABLE_BORDER_CHAR": "*",
            "GUIDELLM__OPENAI__HEADERS": '{"Authorization": "Bearer env-token"}',
            "GUIDELLM__OPENAI__VERIFY": "false",
        },
    )
    settings = Settings()
    assert settings.dataset.preferred_data_columns == ["custom_column"]
    assert settings.openai.api_key == "env_api_key"
    assert settings.table_border_char == "*"
    assert settings.openai.headers == {"Authorization": "Bearer env-token"}
    assert settings.openai.verify is False
