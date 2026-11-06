"""
Centralized benchmark argument schemas for GuideLLM.
"""

from guidellm.schemas.benchmark.collectors import (
    DEFAULT_VLLM_METRICS,
    CollectorArgs,
    PrometheusCollectorArgs,
    PrometheusCounterSeries,
    PrometheusGaugeSeries,
    PrometheusHistogramSeries,
    PrometheusSummary,
    VLLMServerInfoCollectorArgs,
)
from guidellm.schemas.benchmark.entrypoints import (
    BenchmarkArgs,
    BenchmarkMetadata,
    BenchmarkScenario,
    GenerativeMetricsArgs,
    MetricsArgs,
    args_model_config,
    default_kind,
    default_kind_list,
    merge_scenario_files,
)
from guidellm.schemas.benchmark.goodput import GoodputSLO
from guidellm.schemas.benchmark.outputs import (
    BenchmarkOutputArgs,
    ConsoleBenchmarkOutputArgs,
    CSVBenchmarkOutputArgs,
    HTMLBenchmarkOutputArgs,
    JSONBenchmarkOutputArgs,
    PlotBenchmarkOutputArgs,
    YAMLBenchmarkOutputArgs,
)
from guidellm.schemas.benchmark.profiles import (
    AsyncProfileArgs,
    ConcurrentProfileArgs,
    GoodputProfileArgs,
    KneeProfileArgs,
    ProfileArgs,
    ReplayProfileArgs,
    SweepProfileArgs,
    SynchronousProfileArgs,
    ThroughputProfileArgs,
)
from guidellm.schemas.benchmark.random import RandomArgs, StaticRandomArgs
from guidellm.schemas.benchmark.scenarios import SCENARIO_DIR, get_builtin_scenarios
from guidellm.schemas.benchmark.transient import TransientPhaseConfig
from guidellm.schemas.benchmark.warnings import (
    MetricRef,
    WarningCondition,
    WarningRuleArgs,
)

__all__ = [
    "DEFAULT_VLLM_METRICS",
    "SCENARIO_DIR",
    "AsyncProfileArgs",
    "BenchmarkArgs",
    "BenchmarkMetadata",
    "BenchmarkOutputArgs",
    "BenchmarkScenario",
    "CSVBenchmarkOutputArgs",
    "CollectorArgs",
    "ConcurrentProfileArgs",
    "ConsoleBenchmarkOutputArgs",
    "GenerativeMetricsArgs",
    "GoodputProfileArgs",
    "GoodputSLO",
    "HTMLBenchmarkOutputArgs",
    "JSONBenchmarkOutputArgs",
    "KneeProfileArgs",
    "MetricRef",
    "MetricsArgs",
    "PlotBenchmarkOutputArgs",
    "ProfileArgs",
    "PrometheusCollectorArgs",
    "PrometheusCounterSeries",
    "PrometheusGaugeSeries",
    "PrometheusHistogramSeries",
    "PrometheusSummary",
    "RandomArgs",
    "ReplayProfileArgs",
    "StaticRandomArgs",
    "SweepProfileArgs",
    "SynchronousProfileArgs",
    "ThroughputProfileArgs",
    "TransientPhaseConfig",
    "VLLMServerInfoCollectorArgs",
    "WarningCondition",
    "WarningRuleArgs",
    "YAMLBenchmarkOutputArgs",
    "args_model_config",
    "default_kind",
    "default_kind_list",
    "get_builtin_scenarios",
    "merge_scenario_files",
]
