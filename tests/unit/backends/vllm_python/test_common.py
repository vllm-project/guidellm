"""Unit tests for vLLM Python backend shared utilities."""

from __future__ import annotations

import pytest

from guidellm.backends.vllm_python.common import vllm_benchmark_engine_config


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
