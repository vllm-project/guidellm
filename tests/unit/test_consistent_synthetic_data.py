"""
CLI tests for the --consistent-synthetic-data flag.
"""

import inspect
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from guidellm.__main__ import run
from guidellm.benchmark.entrypoints import benchmark_generative_text
from guidellm.benchmark.scenario import GenerativeTextScenario
from guidellm.request.loader import GenerativeRequestLoader


class TestCLIConsistentSyntheticDataFlag:
    def test_cli_help_includes_flag(self):
        """Test that the help output includes the new flag."""
        runner = CliRunner()
        result = runner.invoke(run, ["--help"])

        assert result.exit_code == 0
        assert "--consistent-synthetic-data" in result.output

    @patch("guidellm.__main__.benchmark_with_scenario")
    def test_cli_flag_default_behavior(self, mock_benchmark):
        """Test that the flag defaults to False when not specified."""
        mock_benchmark.return_value = None

        runner = CliRunner()
        runner.invoke(
            run,
            [
                "--target",
                "http://localhost:8000",
                "--data",
                '{"prompt_tokens": 50, "output_tokens": 25}',
                "--rate-type",
                "concurrent",
                "--rate",
                "1,2",
            ],
        )

        mock_benchmark.assert_called_once()

        call_args = mock_benchmark.call_args
        scenario = call_args[1]["scenario"]

        assert hasattr(scenario, "consistent_synthetic_data")
        assert scenario.consistent_synthetic_data is False

    @patch("guidellm.__main__.benchmark_with_scenario")
    def test_cli_flag_enabled(self, mock_benchmark):
        """Test that the flag can be enabled via CLI."""
        mock_benchmark.return_value = None

        runner = CliRunner()
        runner.invoke(
            run,
            [
                "--target",
                "http://localhost:8000",
                "--data",
                '{"prompt_tokens": 50, "output_tokens": 25}',
                "--rate-type",
                "concurrent",
                "--rate",
                "1,2",
                "--consistent-synthetic-data",
            ],
        )

        mock_benchmark.assert_called_once()

        call_args = mock_benchmark.call_args
        scenario = call_args[1]["scenario"]

        assert hasattr(scenario, "consistent_synthetic_data")
        assert scenario.consistent_synthetic_data is True

    def test_cli_scenario_override_behavior(self):
        """Test that CLI flag can be processed alongside scenario parameters."""
        scenario = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="concurrent",
            rate=[1.0, 2.0],
            consistent_synthetic_data=False,
        )
        assert scenario.consistent_synthetic_data is False

        overridden_scenario = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="concurrent",
            rate=[1.0, 2.0],
            consistent_synthetic_data=True,
        )
        assert overridden_scenario.consistent_synthetic_data is True

        # Test that the CLI processing can handle the flag
        with patch("guidellm.__main__.benchmark_with_scenario") as mock_benchmark:
            mock_benchmark.return_value = None

            runner = CliRunner()
            runner.invoke(
                run,
                [
                    "--target",
                    "http://localhost:8000",
                    "--data",
                    '{"prompt_tokens": 50, "output_tokens": 25}',
                    "--rate-type",
                    "concurrent",
                    "--rate",
                    "1,2",
                    "--consistent-synthetic-data",
                ],
            )

            mock_benchmark.assert_called_once()

            call_args = mock_benchmark.call_args
            scenario = call_args[1]["scenario"]

            assert scenario.consistent_synthetic_data is True


class TestCLIConsistentSyntheticDataIntegration:
    """Integration tests for CLI with the new flag."""

    @patch("guidellm.backend.Backend.create")
    @patch("guidellm.benchmark.entrypoints.benchmark_generative_text")
    def test_cli_to_function_parameter_flow(
        self, mock_benchmark_func, mock_backend_create
    ):
        """Test that CLI flag flows to the benchmark function."""
        mock_backend_create.return_value = Mock()
        mock_benchmark_func.return_value = (Mock(), None)

        runner = CliRunner()
        runner.invoke(
            run,
            [
                "--target",
                "http://localhost:8000",
                "--data",
                '{"prompt_tokens": 50, "output_tokens": 25}',
                "--rate-type",
                "concurrent",
                "--rate",
                "1,2",
                "--consistent-synthetic-data",
                "--disable-console-outputs",
            ],
        )

        mock_benchmark_func.assert_called_once()

        call_args = mock_benchmark_func.call_args
        call_kwargs = call_args[1]

        assert "consistent_synthetic_data" in call_kwargs
        assert call_kwargs["consistent_synthetic_data"] is True


class TestCLIConsistentSyntheticDataRegressionPrevention:
    """Regression tests to ensure the flag doesn't break existing functionality."""

    @patch("guidellm.__main__.benchmark_with_scenario")
    def test_existing_cli_commands_still_work(self, mock_benchmark):
        """Test that existing CLI commands continue to work without the flag."""
        mock_benchmark.return_value = None

        existing_command = [
            "--target",
            "http://localhost:8000",
            "--data",
            '{"prompt_tokens": 100, "output_tokens": 50}',
            "--rate-type",
            "concurrent",
            "--rate",
            "1,2,4,8",
            "--max-requests",
            "1000",
            "--random-seed",
            "42",
        ]

        runner = CliRunner()
        runner.invoke(run, existing_command)

        mock_benchmark.assert_called_once()

        call_args = mock_benchmark.call_args
        scenario = call_args[1]["scenario"]

        assert hasattr(scenario, "consistent_synthetic_data")
        assert scenario.consistent_synthetic_data is False

    @patch("guidellm.__main__.benchmark_with_scenario")
    def test_all_existing_flags_work_with_new_flag(self, mock_benchmark):
        """Test that the new flag works alongside all existing flags."""
        mock_benchmark.return_value = None

        # Test with many existing flags
        comprehensive_command = [
            "--target",
            "http://localhost:8000",
            "--backend-type",
            "openai_http",
            "--model",
            "test-model",
            "--data",
            '{"prompt_tokens": 100, "output_tokens": 50}',
            "--rate-type",
            "concurrent",
            "--rate",
            "1,2,4",
            "--max-requests",
            "100",
            "--max-seconds",
            "60",
            "--random-seed",
            "42",
            "--warmup-percent",
            "0.1",
            "--cooldown-percent",
            "0.1",
            "--consistent-synthetic-data",
            "--disable-progress",
            "--disable-console-outputs",
        ]

        runner = CliRunner()
        runner.invoke(run, comprehensive_command)

        mock_benchmark.assert_called_once()

        call_args = mock_benchmark.call_args
        scenario = call_args[1]["scenario"]

        assert scenario.target == "http://localhost:8000"
        assert scenario.backend_type == "openai_http"
        assert scenario.model == "test-model"
        assert scenario.rate_type == "concurrent"
        assert scenario.rate == [1.0, 2.0, 4.0]
        assert scenario.max_requests == 100
        assert scenario.max_seconds == 60
        assert scenario.random_seed == 42
        assert scenario.warmup_percent == 0.1
        assert scenario.cooldown_percent == 0.1

        assert scenario.consistent_synthetic_data is True


class TestConsistentSyntheticDataIntegration:
    """Integration tests for the complete consistent_synthetic_data flow."""

    def test_scenario_to_entrypoint_parameter_flow(self):
        """Test that the flag flows from scenario to entrypoint function."""
        scenario = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="concurrent",
            rate=[1.0, 2.0],
            consistent_synthetic_data=True,
        )

        assert scenario.consistent_synthetic_data is True

        scenario_vars = vars(scenario)
        assert "consistent_synthetic_data" in scenario_vars
        assert scenario_vars["consistent_synthetic_data"] is True

    def test_benchmark_generative_text_accepts_parameter(self):
        """Test that benchmark_generative_text function accepts the new parameter."""
        sig = inspect.signature(benchmark_generative_text)
        params = list(sig.parameters.keys())

        assert "consistent_synthetic_data" in params

        param = sig.parameters["consistent_synthetic_data"]
        assert param.default is False

    def test_end_to_end_parameter_passing_mock(self):
        """Test that the parameter flows through the entire system (mocked version)."""

        with patch(
            "guidellm.benchmark.entrypoints.benchmark_generative_text"
        ) as mock_benchmark_func:
            mock_benchmark_func.return_value = (Mock(), None)

            scenario = GenerativeTextScenario(
                target="http://localhost:8000",
                data='{"prompt_tokens": 50, "output_tokens": 25}',
                rate_type="concurrent",
                rate=[1.0, 2.0],
                consistent_synthetic_data=True,
            )

            scenario_vars = vars(scenario)

            assert "consistent_synthetic_data" in scenario_vars
            assert scenario_vars["consistent_synthetic_data"] is True

            mock_benchmark_func(**scenario_vars)

            mock_benchmark_func.assert_called_once()
            call_kwargs = mock_benchmark_func.call_args[1]
            assert "consistent_synthetic_data" in call_kwargs
            assert call_kwargs["consistent_synthetic_data"] is True

    def test_complete_flow_with_real_components(self):
        """Test the complete flow using real components."""
        loader_disabled = GenerativeRequestLoader(
            data='{"prompt_tokens": 80, "output_tokens": 40, "samples": 10}',
            data_args=None,
            processor="gpt2",
            processor_args=None,
            iter_type="infinite",
            shuffle=False,
            random_seed=42,
            consistent_synthetic_data=False,  # Flag disabled
        )

        loader_enabled = GenerativeRequestLoader(
            data='{"prompt_tokens": 80, "output_tokens": 40, "samples": 10}',
            data_args=None,
            processor="gpt2",
            processor_args=None,
            iter_type="infinite",
            shuffle=False,
            random_seed=42,
            consistent_synthetic_data=True,  # Flag enabled
        )

        disabled_prompts = []
        enabled_prompts = []

        for _iteration in range(2):
            iteration_prompts = []
            for i, request in enumerate(loader_disabled):
                if i >= 3:
                    break
                iteration_prompts.append(request.content[:50])
            disabled_prompts.append(iteration_prompts)

        for _iteration in range(2):
            iteration_prompts = []
            for i, request in enumerate(loader_enabled):
                if i >= 3:
                    break
                iteration_prompts.append(request.content[:50])
            enabled_prompts.append(iteration_prompts)

        assert disabled_prompts[0] != disabled_prompts[1], (
            "Expected different prompts with flag disabled"
        )

        assert enabled_prompts[0] == enabled_prompts[1], (
            "Expected same prompts with flag enabled"
        )

    def test_flag_only_affects_synthetic_data(self):
        """Test that the flag only affects synthetic data, not other data types."""
        non_synthetic_data = [
            {
                "prompt": "Test prompt 1",
                "prompt_tokens_count": 5,
                "output_tokens_count": 10,
            },
            {
                "prompt": "Test prompt 2",
                "prompt_tokens_count": 4,
                "output_tokens_count": 8,
            },
            {
                "prompt": "Test prompt 3",
                "prompt_tokens_count": 6,
                "output_tokens_count": 12,
            },
        ]

        loader_flag_enabled = GenerativeRequestLoader(
            data=non_synthetic_data,
            data_args=None,
            processor="gpt2",
            processor_args=None,
            iter_type="finite",
            random_seed=42,
            consistent_synthetic_data=True,
        )

        loader_flag_disabled = GenerativeRequestLoader(
            data=non_synthetic_data,
            data_args=None,
            processor="gpt2",
            processor_args=None,
            iter_type="finite",
            random_seed=42,
            consistent_synthetic_data=False,  # Flag disabled
        )

        enabled_requests = []
        disabled_requests = []

        for i, request in enumerate(loader_flag_enabled):
            if i >= 3:
                break
            enabled_requests.append(request.content)

        for i, request in enumerate(loader_flag_disabled):
            if i >= 3:
                break
            disabled_requests.append(request.content)

        assert enabled_requests == disabled_requests
        expected_prompts = {"Test prompt 1", "Test prompt 2", "Test prompt 3"}
        actual_prompts = set(enabled_requests)
        assert actual_prompts == expected_prompts


class TestConsistentSyntheticDataErrorHandling:
    def test_invalid_synthetic_data_handled_gracefully(self):
        """Test that invalid synthetic data is handled gracefully."""
        # Test with malformed JSON
        with pytest.raises((ValueError, TypeError, AttributeError)):
            GenerativeRequestLoader(
                data='{"invalid_json": malformed}',
                data_args=None,
                processor="gpt2",
                processor_args=None,
                random_seed=42,
                consistent_synthetic_data=True,
            )

    def test_flag_with_different_iter_types(self):
        """Test that the flag behaves correctly with different iter_types."""
        loader_finite = GenerativeRequestLoader(
            data='{"prompt_tokens": 50, "output_tokens": 25, "samples": 5}',
            data_args=None,
            processor="gpt2",
            processor_args=None,
            iter_type="finite",
            random_seed=42,
            consistent_synthetic_data=True,
        )

        requests = list(loader_finite)
        assert len(requests) == 5

        loader_infinite = GenerativeRequestLoader(
            data='{"prompt_tokens": 50, "output_tokens": 25, "samples": 5}',
            data_args=None,
            processor="gpt2",
            processor_args=None,
            iter_type="infinite",
            random_seed=42,
            consistent_synthetic_data=True,
        )

        first_requests = []
        for i, request in enumerate(loader_infinite):
            if i >= 3:
                break
            first_requests.append(request.content)

        assert len(first_requests) == 3


class TestGenerativeTextScenarioConsistentSyntheticData:
    """Test suite for the consistent_synthetic_data field in GenerativeTextScenario."""

    def test_default_field_value(self):
        """Test that consistent_synthetic_data defaults to False."""
        scenario = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="concurrent",
            rate=[1.0, 2.0, 4.0],
        )

        assert hasattr(scenario, "consistent_synthetic_data")
        assert scenario.consistent_synthetic_data is False

    def test_explicit_field_values(self):
        """Test that consistent_synthetic_data can be explicitly set."""
        # Test explicit False
        scenario_false = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="concurrent",
            rate=[1.0, 2.0, 4.0],
            consistent_synthetic_data=False,
        )
        assert scenario_false.consistent_synthetic_data is False

        # Test explicit True
        scenario_true = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="concurrent",
            rate=[1.0, 2.0, 4.0],
            consistent_synthetic_data=True,
        )
        assert scenario_true.consistent_synthetic_data is True

    def test_field_validation(self):
        """
        Test that the field accepts boolean values and converts common representations.
        """
        # Valid boolean values should work
        scenario_true = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="concurrent",
            rate=[1.0, 2.0, 4.0],
            consistent_synthetic_data=True,
        )
        assert scenario_true.consistent_synthetic_data is True

        scenario_false = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="concurrent",
            rate=[1.0, 2.0, 4.0],
            consistent_synthetic_data=False,
        )
        assert scenario_false.consistent_synthetic_data is False

        scenario_str_true = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="concurrent",
            rate=[1.0, 2.0, 4.0],
            consistent_synthetic_data="true",  # type: ignore[arg-type]
        )
        assert scenario_str_true.consistent_synthetic_data is True

        scenario_int_one = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="concurrent",
            rate=[1.0, 2.0, 4.0],
            consistent_synthetic_data=1,  # type: ignore[arg-type]
        )
        assert scenario_int_one.consistent_synthetic_data is True

        scenario_int_zero = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="concurrent",
            rate=[1.0, 2.0, 4.0],
            consistent_synthetic_data=0,  # type: ignore[arg-type]
        )
        assert scenario_int_zero.consistent_synthetic_data is False

        with pytest.raises(ValidationError):
            GenerativeTextScenario(
                target="http://localhost:8000",
                data='{"prompt_tokens": 50, "output_tokens": 25}',
                rate_type="concurrent",
                rate=[1.0, 2.0, 4.0],
                consistent_synthetic_data=None,  # type: ignore[arg-type]
            )

        with pytest.raises(ValidationError):
            GenerativeTextScenario(
                target="http://localhost:8000",
                data='{"prompt_tokens": 50, "output_tokens": 25}',
                rate_type="concurrent",
                rate=[1.0, 2.0, 4.0],
                consistent_synthetic_data=[],  # type: ignore[arg-type]
            )

        with pytest.raises(ValidationError):
            GenerativeTextScenario(
                target="http://localhost:8000",
                data='{"prompt_tokens": 50, "output_tokens": 25}',
                rate_type="concurrent",
                rate=[1.0, 2.0, 4.0],
                consistent_synthetic_data={},  # type: ignore[arg-type]
            )

    def test_field_with_all_scenario_types(self):
        """Test that the field works with different rate types."""
        # Test with concurrent
        scenario_concurrent = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="concurrent",
            rate=[1.0, 2.0, 4.0],
            consistent_synthetic_data=True,
        )
        assert scenario_concurrent.consistent_synthetic_data is True

        # Test with throughput
        scenario_throughput = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="throughput",
            consistent_synthetic_data=True,
        )
        assert scenario_throughput.consistent_synthetic_data is True

        # Test with sweep
        scenario_sweep = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="sweep",
            rate=[5.0],
            consistent_synthetic_data=True,
        )
        assert scenario_sweep.consistent_synthetic_data is True

        # Test with synchronous
        scenario_sync = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 50, "output_tokens": 25}',
            rate_type="synchronous",
            consistent_synthetic_data=True,
        )
        assert scenario_sync.consistent_synthetic_data is True


class TestGenerativeTextScenarioBackwardCompatibility:
    """Test suite to ensure backward compatibility."""

    def test_existing_scenarios_still_work(self):
        """Test that existing scenario configurations continue to work."""
        scenario = GenerativeTextScenario(
            target="http://localhost:8000",
            data='{"prompt_tokens": 100, "output_tokens": 50}',
            rate_type="concurrent",
            rate=[1.0, 2.0, 4.0, 8.0],
            random_seed=123,
            max_requests=1000,
        )

        assert scenario.target == "http://localhost:8000"
        assert scenario.rate_type == "concurrent"
        assert scenario.rate == [1.0, 2.0, 4.0, 8.0]
        assert scenario.random_seed == 123
        assert scenario.max_requests == 1000

        assert scenario.consistent_synthetic_data is False


class TestGenerativeRequestLoaderConsistentSyntheticData:
    """Test suite for the consistent_synthetic_data flag functionality."""

    @patch("guidellm.dataset.synthetic.SyntheticDatasetCreator.is_supported")
    def test_iterator_preservation_with_flag_disabled(self, mock_is_supported):
        """Test that iterator is preserved when flag is disabled (default behavior)."""
        mock_is_supported.return_value = True

        loader = GenerativeRequestLoader(
            data='{"prompt_tokens": 50, "output_tokens": 25, "samples": 10}',
            data_args=None,
            processor="gpt2",
            processor_args=None,
            iter_type="infinite",
            random_seed=42,
            consistent_synthetic_data=False,  # Flag disabled
        )

        iter1 = loader._get_dataset_iter(0)
        assert iter1 is not None

        iter2 = loader._get_dataset_iter(1)
        assert iter2 is not None

    @patch("guidellm.dataset.synthetic.SyntheticDatasetCreator.is_supported")
    def test_iterator_reset_with_flag_enabled(self, mock_is_supported):
        """Test that iterator is reset when flag is enabled for synthetic data."""
        mock_is_supported.return_value = True

        loader = GenerativeRequestLoader(
            data='{"prompt_tokens": 50, "output_tokens": 25, "samples": 10}',
            data_args=None,
            processor="gpt2",
            processor_args=None,
            iter_type="infinite",
            random_seed=42,
            consistent_synthetic_data=True,  # Flag enabled
        )

        iter1 = loader._get_dataset_iter(0)
        assert iter1 is not None

        iter2 = loader._get_dataset_iter(1)
        assert iter2 is not None

    def test_finite_iter_type_unaffected(self):
        """Test that finite iter_type is unaffected by the flag."""
        loader = GenerativeRequestLoader(
            data='{"prompt_tokens": 50, "output_tokens": 25, "samples": 5}',
            data_args=None,
            processor="gpt2",
            processor_args=None,
            iter_type="finite",  # Finite type
            random_seed=42,
            consistent_synthetic_data=True,
        )

        iter1 = loader._get_dataset_iter(0)
        assert iter1 is not None

        iter2 = loader._get_dataset_iter(1)
        assert iter2 is None

    def test_consistent_prompts_with_flag_enabled(self):
        """Integration test: verify consistent prompts across multiple iterations."""
        loader = GenerativeRequestLoader(
            data='{"prompt_tokens": 80, "output_tokens": 40, "samples": 5}',
            data_args=None,
            processor="gpt2",
            processor_args=None,
            iter_type="infinite",
            shuffle=False,
            random_seed=42,
            consistent_synthetic_data=True,
        )

        first_iteration_prompts = []
        for i, request in enumerate(loader):
            if i >= 3:
                break
            first_iteration_prompts.append(request.content)

        second_iteration_prompts = []
        for i, request in enumerate(loader):
            if i >= 3:
                break
            second_iteration_prompts.append(request.content)

        assert first_iteration_prompts == second_iteration_prompts
        assert len(first_iteration_prompts) == 3
        assert len(second_iteration_prompts) == 3

    def test_different_prompts_with_flag_disabled(self):
        """
        Integration test: verify different prompts across iterations when flag is
        disabled.
        """
        loader = GenerativeRequestLoader(
            data='{"prompt_tokens": 80, "output_tokens": 40, "samples": 10}',
            data_args=None,
            processor="gpt2",
            processor_args=None,
            iter_type="infinite",
            shuffle=False,
            random_seed=42,
            consistent_synthetic_data=False,
        )

        first_iteration_prompts = []
        for i, request in enumerate(loader):
            if i >= 3:
                break
            first_iteration_prompts.append(request.content)

        second_iteration_prompts = []
        for i, request in enumerate(loader):
            if i >= 3:
                break
            second_iteration_prompts.append(request.content)

        assert first_iteration_prompts != second_iteration_prompts
        assert len(first_iteration_prompts) == 3
        assert len(second_iteration_prompts) == 3
