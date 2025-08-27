import pytest

from guidellm.backend import OpenAIHTTPBackend
from guidellm.settings import settings


@pytest.mark.smoke
def test_openai_http_backend_default_initialization():
    backend = OpenAIHTTPBackend()
    assert backend.verify is True


@pytest.mark.smoke
def test_openai_http_backend_custom_ssl_verification():
    backend = OpenAIHTTPBackend(verify=False)
    assert backend.verify is False


@pytest.mark.smoke
def test_openai_http_backend_custom_headers_override():
    # Set a default api_key, which would normally create an Authorization header
    settings.openai.api_key = "default-api-key"

    # Set custom headers that override the default Authorization and add a new header
    openshift_token = "Bearer sha256~xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    override_headers = {
        "Authorization": openshift_token,
        "Custom-Header": "Custom-Value",
    }

    # Initialize the backend
    backend = OpenAIHTTPBackend(headers=override_headers)

    # Check that the override headers are used
    assert backend.headers["Authorization"] == openshift_token
    assert backend.headers["Custom-Header"] == "Custom-Value"
    assert len(backend.headers) == 2

    # Reset the settings
    settings.openai.api_key = None
    settings.openai.headers = None


@pytest.mark.smoke
def test_openai_http_backend_kwarg_headers_override_settings():
    # Set headers via settings (simulating environment variables)
    settings.openai.headers = {"Authorization": "Bearer settings-token"}

    # Set different headers via kwargs (simulating --backend-args)
    override_headers = {
        "Authorization": "Bearer kwargs-token",
        "Custom-Header": "Custom-Value",
    }

    # Initialize the backend with kwargs
    backend = OpenAIHTTPBackend(headers=override_headers)

    # Check that the kwargs headers took precedence
    assert backend.headers["Authorization"] == "Bearer kwargs-token"
    assert backend.headers["Custom-Header"] == "Custom-Value"
    assert len(backend.headers) == 2

    # Reset the settings
    settings.openai.headers = None


@pytest.mark.smoke
def test_openai_http_backend_remove_header_with_none():
    # Set a default api_key, which would normally create an Authorization header
    settings.openai.api_key = "default-api-key"

    # Set a custom header and explicitly set Authorization to None to remove it
    override_headers = {
        "Authorization": None,
        "Custom-Header": "Custom-Value",
    }

    # Initialize the backend
    backend = OpenAIHTTPBackend(headers=override_headers)

    # Check that the Authorization header is removed and the custom header is present
    assert "Authorization" not in backend.headers
    assert backend.headers["Custom-Header"] == "Custom-Value"
    assert len(backend.headers) == 1

    # Reset the settings
    settings.openai.api_key = None
    settings.openai.headers = None
