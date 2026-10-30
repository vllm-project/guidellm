"""Tests for bounded server-info collection and configuration filtering."""

import asyncio
import gzip
import json

import httpx2
import pytest

from guidellm.collectors import Collector, VLLMServerInfoCollector
from guidellm.collectors import vllm_server_info as module
from guidellm.collectors.vllm_server_info import _redact_server_config
from guidellm.schemas.benchmark import VLLMServerInfoCollectorArgs

URL = "http://localhost:8000/server_info"


@pytest.mark.smoke
def test_registered_collector():
    """Resolve arguments through the same registry used by the child process.

    ## WRITTEN BY AI ##
    """
    collector = Collector.resolve(VLLMServerInfoCollectorArgs(url=URL))
    assert isinstance(collector, VLLMServerInfoCollector)
    assert collector.redact_hosts is True


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_collects_config_after_each_benchmark():
    """Fetch once on stop, omit environments, and discard the previous result.

    ## WRITTEN BY AI ##
    """
    requests = []

    def respond(request):
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "vllm_config": {
                    "model": "example/model",
                    "max_model_len": len(requests),
                },
                "vllm_env": {"SECRET": "private-value"},
                "system_env": {"SECRET": "private-value"},
            },
        )

    collector = VLLMServerInfoCollector(URL, transport=httpx2.MockTransport(respond))
    for count in (1, 2):
        await collector.start()
        assert collector.result(0, 1) == {}
        assert len(requests) == count - 1
        await collector.stop()
        result = collector.result(0, 1)
        assert result["vllm_config"]["max_model_len"] == count
        assert result["source"] == URL
        assert result["collected_at"] > 0
        assert "vllm_env" not in result
        assert "system_env" not in result
        assert "private-value" not in json.dumps(result)
    assert len(requests) == 2
    assert all(
        dict(request.url.params) == {"config_format": "json"} for request in requests
    )


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(404, text="private-value"),
        httpx2.Response(401, text="private-value"),
        httpx2.Response(500, text="private-value"),
        httpx2.Response(302, headers={"Location": "http://private-value/"}),
        httpx2.Response(200, text="private-value"),
        httpx2.Response(200, json=[]),
        httpx2.Response(200, json={"vllm_config": "legacy config"}),
        httpx2.Response(200, json={"system_env": {"SECRET": "private-value"}}),
    ],
)
async def test_failed_capture_has_safe_error_and_does_not_follow_redirects(response):
    """Keep a result on failure without exposing response data or following redirects.

    ## WRITTEN BY AI ##
    """
    requests = []

    def respond(request):
        requests.append(request)
        return response

    collector = VLLMServerInfoCollector(URL, transport=httpx2.MockTransport(respond))
    result = await collector.collect()
    assert "error" in result
    assert "vllm_config" not in result
    assert "private-value" not in json.dumps(result)
    assert len(requests) == 1


class _Chunks(httpx2.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.parametrize("compressed", [False, True])
async def test_limits_decoded_body_size(compressed):
    """Reject oversized chunked and compressed bodies even without Content-Length.

    ## WRITTEN BY AI ##
    """
    body = json.dumps(
        {"vllm_config": {"model": "a" * module._SERVER_CONFIG_MAX_BYTES}}
    ).encode()
    if compressed:
        response = httpx2.Response(
            200,
            headers={"Content-Encoding": "gzip"},
            stream=_Chunks([gzip.compress(body)]),
        )
    else:
        response = httpx2.Response(200, stream=_Chunks([body[:100], body[100:]]))
    collector = VLLMServerInfoCollector(
        URL, transport=httpx2.MockTransport(lambda _: response)
    )
    result = await collector.collect()
    assert result["error"] == "ValueError"
    assert "vllm_config" not in result


@pytest.mark.regression
@pytest.mark.asyncio
async def test_total_timeout_cancels_trickling_response(monkeypatch):
    """Bound total time even if individual response chunks arrive successfully.

    ## WRITTEN BY AI ##
    """
    closed = asyncio.Event()

    class SlowStream(httpx2.AsyncByteStream):
        async def __aiter__(self):
            while True:
                yield b" "
                await asyncio.sleep(0.01)

        async def aclose(self):
            closed.set()

    monkeypatch.setattr(module, "_SERVER_CONFIG_TIMEOUT", 0.05)
    collector = VLLMServerInfoCollector(
        URL,
        transport=httpx2.MockTransport(
            lambda _: httpx2.Response(200, stream=SlowStream())
        ),
    )
    result = await asyncio.wait_for(collector.collect(), timeout=1)
    assert result["error"] == "TimeoutError"
    assert closed.is_set()


@pytest.mark.regression
@pytest.mark.asyncio
async def test_failure_does_not_reuse_previous_snapshot():
    """A failed second capture must not appear to have the first run's config.

    ## WRITTEN BY AI ##
    """
    responses = iter(
        [httpx2.Response(200, json={"vllm_config": {}}), httpx2.Response(404)]
    )
    collector = VLLMServerInfoCollector(
        URL, transport=httpx2.MockTransport(lambda _: next(responses))
    )
    await collector.start()
    await collector.stop()
    assert collector.result(0, 1)["vllm_config"] == {}
    await collector.start()
    await collector.stop()
    assert "vllm_config" not in collector.result(1, 2)
    assert "error" in collector.result(1, 2)


@pytest.mark.regression
@pytest.mark.parametrize("redact_hosts", [True, False])
def test_host_filtering_is_independent_of_credentials(redact_hosts):
    """Apply host policy to fields and URLs while always filtering credentials.

    ## WRITTEN BY AI ##
    """
    payload = {
        "host": "worker.internal",
        "endpoint": "https://user:password@worker.internal:8443/path?max_tokens=32",
        "api_key": "private-value",
        "max_tokens": 4096,
    }
    result = _redact_server_config(payload, redact_hosts=redact_hosts)
    assert result["api_key"] == "[REDACTED]"
    assert result["max_tokens"] == 4096
    assert result["host"] == ("[REDACTED]" if redact_hosts else "worker.internal")
    assert result["endpoint"] == (
        "https://[REDACTED]/path?max_tokens=32"
        if redact_hosts
        else "https://[REDACTED]@worker.internal:8443/path?max_tokens=32"
    )
    assert _redact_server_config(result, redact_hosts=redact_hosts) == result


@pytest.mark.regression
class TestServerConfigRedaction:
    """Concrete credential patterns must not erase useful benchmark settings."""

    @pytest.mark.parametrize(
        "key",
        [
            "HF_TOKEN",
            "OPENAI_API_KEY",
            "VLLM_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "WANDB_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "ssl_keyfile_password",
            "PASSWD",
            "DB_PASS",
            "PRIVATE_KEY",
            "AZURE_CLIENT_CREDENTIAL",
            "COOKIE",
            "BEARER",
            "GITHUB_PAT",
            "proxy-auth",
            "client_credentials",
            "apiKey",
            "SSH_KEY",
            "plugin.encryption-key",
            "service_signing_key",
            "LICENSE_KEY",
            "tls_client_key",
            "PASSPHRASE",
            "db_pwd",
        ],
    )
    def test_credential_fields(self, key):
        """Filter reviewer examples, including nested dictionaries inside lists.

        ## WRITTEN BY AI ##
        """
        payload = {"plugins": [{key: "private-value"}]}
        assert _redact_server_config(payload) == {"plugins": [{key: "[REDACTED]"}]}
        assert payload == {"plugins": [{key: "private-value"}]}

    @pytest.mark.parametrize("value", [None, False, True])
    def test_unset_credentials_and_boolean_switches(self, value):
        """Preserve the difference between absent credentials and enabled switches.

        ## WRITTEN BY AI ##
        """
        assert _redact_server_config({"hf_token": value}) == {"hf_token": value}

    def test_useful_settings_are_preserved(self):
        """Do not confuse token counts, cache keys or compiler passes with secrets.

        ## WRITTEN BY AI ##
        """
        payload = {
            "max_tokens": 2048,
            "max_num_batched_tokens": 4096,
            "tokenizer": "test-model",
            "tokenizer_mode": "auto",
            "cache_key": "sha256",
            "monkey_patch": False,
            "bypass": True,
            "pass_config": {"enable_fusion": True},
            "enable_auth": True,
            "auth_method": "oauth2",
            "credential_provider": "default",
            "ssh_key_path": "/keys/id_ed25519",
            "encryption_key_id": "key-1",
            "signing_key_algorithm": "ed25519",
            "license_key_required": True,
            "client_key_file": "/keys/client.pem",
            "signature_algorithm": "sha256",
            "pwd_length": 32,
            "hostname_override_enabled": False,
            "VLLM_USE_V1": True,
            "pip_packages": (
                "torch==2.13.0\nvllm==0.30.0\ngoogle-auth==2.40.0\nsecretstorage==3.3.3"
            ),
            "cuda_runtime_version": "13.0",
            "env_vars": "CUDA_VERSION=13.0\nVLLM_USE_V1=1",
        }
        assert _redact_server_config(payload) == payload

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("postgres://user:pw@host/db", "postgres://[REDACTED]@host/db"),
            ("https://token@host/path", "https://[REDACTED]@host/path"),
            ("redis://user:p%40ss@[::1]:6379/0", "redis://[REDACTED]@[::1]:6379/0"),
            ("postgres://u:p@ss@host/db", "postgres://[REDACTED]@host/db"),
            ("https://host/path@revision", "https://host/path@revision"),
            ("https://host/path?max_tokens=32", "https://host/path?max_tokens=32"),
            (
                "https://host:8000/path?revision=main",
                "https://host:8000/path?revision=main",
            ),
            ("redis://user:pa%2Fss@cache:6379/0", "redis://[REDACTED]@cache:6379/0"),
            ("redis://user:pa/ss@cache:6379/0", "[REDACTED]"),
            ("redis://user:pa?ss@cache:6379/0", "[REDACTED]"),
            ("redis://user:pa#ss@cache:6379/0", "[REDACTED]"),
            ("endpoint=redis://user:pa/ss@cache:6379/0 retries=2", "[REDACTED]"),
            ("https://host:8000/path@revision", "[REDACTED]"),
            ("https://[::1]/path@revision", "[REDACTED]"),
            (
                "primary=https://u:pw@a/ secondary=https://u:pw@b/",
                "primary=https://[REDACTED]@a/ secondary=https://[REDACTED]@b/",
            ),
        ],
    )
    def test_url_userinfo(self, value, expected):
        """Remove URL credentials without removing hosts, paths or numeric settings.

        ## WRITTEN BY AI ##
        """
        assert _redact_server_config({"endpoint": value}, redact_hosts=False) == {
            "endpoint": expected
        }

    @pytest.mark.parametrize(
        "value",
        [
            "HF_TOKEN=hf_placeholder",
            "export HF_TOKEN = 'two word secret'",
            "--api-key=sk_placeholder",
            "serve --api-key 'two word secret' --port 8000",
            "CUDA_VERSION=13.0\nVLLM_API_KEY=placeholder\nVLLM_USE_V1=1",
            "https://host/path?token=placeholder&max_tokens=32",
            "Authorization: Bearer placeholder",
            "proxy-authorization: Basic placeholder\nAccept: application/json",
            "https://host/path?sig=placeholder&max_tokens=32",
            "https://host/path?X-Amz-Signature=placeholder&X-Amz-Expires=60",
            "--signing-key placeholder",
            "PASSPHRASE=placeholder",
        ],
    )
    def test_sensitive_free_form_text(self, value):
        """Drop sensitive free-form text as a whole rather than parse shell syntax.

        ## WRITTEN BY AI ##
        """
        assert _redact_server_config({"diagnostic": value}) == {
            "diagnostic": "[REDACTED]"
        }

    def test_argument_lists(self):
        """Filter flag values, assignments and nested lists without mutation.

        ## WRITTEN BY AI ##
        """
        payload = [
            "--api-key",
            "two word secret",
            "--max-tokens",
            "32",
            "HF_TOKEN=placeholder",
            ["--private-key", {"value": "secret"}],
            "--password=placeholder",
            "--tokenizer",
            "test-model",
        ]
        result = _redact_server_config(payload)
        assert result == [
            "--api-key",
            "[REDACTED]",
            "--max-tokens",
            "32",
            "[REDACTED]",
            ["--private-key", "[REDACTED]"],
            "[REDACTED]",
            "--tokenizer",
            "test-model",
        ]
        assert payload[1] == "two word secret"
        assert _redact_server_config(result) == result

    def test_environment_records(self):
        """Use an environment record's name to filter its value without mutation.

        ## WRITTEN BY AI ##
        """
        payload = {
            "env": [
                {"name": "VLLM_API_KEY", "value": "private-value"},
                {"name": "SERVICE_SIGNING_KEY", "value": "private-signing-key"},
                {"name": "VLLM_ATTENTION_BACKEND", "value": "FLASH_ATTN"},
                {"name": "CUDA_VISIBLE_DEVICES", "value": "0,1"},
                {"name": "HF_HOME", "value": "/cache/huggingface"},
                {"name": None, "value": "ordinary-value"},
                {"name": ["ordinary-name"], "value": "ordinary-value"},
            ]
        }
        result = _redact_server_config(payload)
        assert result["env"][:2] == [
            {"name": "VLLM_API_KEY", "value": "[REDACTED]"},
            {"name": "SERVICE_SIGNING_KEY", "value": "[REDACTED]"},
        ]
        assert result["env"][2:] == payload["env"][2:]
        assert payload["env"][0]["value"] == "private-value"
        assert _redact_server_config(result) == result

    def test_non_string_keys(self):
        """Preserve non-JSON keys while still filtering the values below them.

        ## WRITTEN BY AI ##
        """
        payload = {1: {"API_KEY": "private-value"}, None: "ordinary-value"}
        assert _redact_server_config(payload) == {
            1: {"API_KEY": "[REDACTED]"},
            None: "ordinary-value",
        }
        assert payload[1] == {"API_KEY": "private-value"}

    @pytest.mark.parametrize(
        "value",
        [
            "Authorization is enabled",
            "Content-Type: application/json",
            "https://host/path?signature_algorithm=sha256&max_tokens=32",
            "https://host/path?X-Amz-Expires=60",
        ],
    )
    def test_non_sensitive_text(self, value):
        """Keep ordinary headers and query settings available for comparison.

        ## WRITTEN BY AI ##
        """
        assert _redact_server_config(value, redact_hosts=False) == value

    @pytest.mark.parametrize("value", ["localhost", "127.0.0.1", "::1", "", None])
    def test_local_or_unset_hosts(self, value):
        """Keep canonical local hosts and unset values useful for reproduction.

        ## WRITTEN BY AI ##
        """
        payload = dict.fromkeys(
            ["host", "VLLM_EC_SIDE_CHANNEL_HOST", "hostname", "node_ip", "master_addr"],
            value,
        )
        assert _redact_server_config(payload) == payload

    @pytest.mark.parametrize(
        "value",
        [
            "private.internal",
            "10.0.0.7",
            "2001:db8::1",
            "localhost.private.internal",
            "0:0:0:0:0:0:0:1",
        ],
    )
    def test_other_hosts(self, value):
        """Conservatively filter noncanonical and non-local host fields.

        ## WRITTEN BY AI ##
        """
        payload = dict.fromkeys(
            ["host", "VLLM_EC_SIDE_CHANNEL_HOST", "hostname", "node_ip", "master_addr"],
            value,
        )
        assert _redact_server_config(payload) == dict.fromkeys(payload, "[REDACTED]")
