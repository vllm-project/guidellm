"""Unit tests for vLLM Python backend shared utilities."""

from __future__ import annotations

import logging
import os
import sys

import pytest

from guidellm.backends.vllm_python.common import (
    prepare_vllm_benchmark_logging,
    vllm_benchmark_engine_config,
)


class TestPrepareVllmBenchmarkLogging:
    @pytest.mark.smoke
    def test_sets_default_env_without_overriding_user_value(self, monkeypatch):
        """Existing VLLM_LOGGING_LEVEL is preserved. ## WRITTEN BY AI ##"""
        monkeypatch.setenv("VLLM_LOGGING_LEVEL", "ERROR")
        prepare_vllm_benchmark_logging()
        assert os.environ["VLLM_LOGGING_LEVEL"] == "ERROR"

    @pytest.mark.sanity
    def test_sets_error_level_by_default(self, monkeypatch):
        """Default benchmark logging level is ERROR. ## WRITTEN BY AI ##"""
        monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
        prepare_vllm_benchmark_logging()
        assert os.environ["VLLM_LOGGING_LEVEL"] == "ERROR"

    @pytest.mark.sanity
    def test_lowers_configured_vllm_logger(self, monkeypatch):
        """Configured vLLM loggers are quieted in-process. ## WRITTEN BY AI ##"""
        monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
        vllm_logger = logging.getLogger("vllm")
        handler = logging.StreamHandler()
        vllm_logger.addHandler(handler)
        vllm_logger.setLevel(logging.INFO)
        handler.setLevel(logging.INFO)

        prepare_vllm_benchmark_logging("ERROR")

        assert vllm_logger.level == logging.ERROR
        assert handler.level == logging.ERROR
        vllm_logger.removeHandler(handler)

    @pytest.mark.regression
    def test_configure_vllm_root_logger_exception_does_not_propagate(self, monkeypatch):
        """_configure_vllm_root_logger errors are swallowed gracefully. ## WRITTEN BY AI ##"""
        monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
        fake_logger_module = type(sys)("vllm.logger")

        def _raise():
            raise RuntimeError("vLLM API changed")

        fake_logger_module._configure_vllm_root_logger = _raise
        monkeypatch.setitem(sys.modules, "vllm.logger", fake_logger_module)

        # Should not raise even though _configure_vllm_root_logger blows up.
        prepare_vllm_benchmark_logging("ERROR")

    @pytest.mark.sanity
    def test_reconfigures_vllm_root_logger_when_already_imported(self, monkeypatch):
        """Late prepare re-applies vLLM logging after import. ## WRITTEN BY AI ##"""
        monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
        fake_logger_module = type(sys)("vllm.logger")
        called = {"value": False}
        fake_logger_module._configure_vllm_root_logger = lambda: called.update(
            value=True
        )
        monkeypatch.setitem(sys.modules, "vllm.logger", fake_logger_module)

        prepare_vllm_benchmark_logging("ERROR")

        assert called["value"] is True


class TestVllmBenchmarkEngineConfig:
    @pytest.mark.smoke
    def test_disable_log_stats_default(self):
        """disable_log_stats defaults to True for benchmarks. ## WRITTEN BY AI ##"""
        config = vllm_benchmark_engine_config({"tensor_parallel_size": 1})
        assert config["disable_log_stats"] is True
        assert config["tensor_parallel_size"] == 1

    @pytest.mark.sanity
    def test_disable_log_stats_user_override(self):
        """User-provided disable_log_stats is preserved. ## WRITTEN BY AI ##"""
        config = vllm_benchmark_engine_config({"disable_log_stats": False})
        assert config["disable_log_stats"] is False
