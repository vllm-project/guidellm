"""
Centralized benchmark argument schemas for GuideLLM.
"""

from guidellm.schemas.benchmark.entrypoints import (
    BenchmarkArgs,
    BenchmarkMetadata,
    BenchmarkScenario,
    GenerativeMetricsArgs,
    MetricsArgs,
    args_model_config,
    default_kind,
    default_kind_list,
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
    ProfileArgs,
    ReplayProfileArgs,
    SweepProfileArgs,
    SynchronousProfileArgs,
    ThroughputProfileArgs,
)
from guidellm.schemas.benchmark.random import RandomArgs, StaticRandomArgs
from guidellm.schemas.benchmark.scenarios import SCENARIO_DIR, get_builtin_scenarios
from guidellm.schemas.benchmark.transient import TransientPhaseConfig

__all__ = [
    "SCENARIO_DIR",
    "AsyncProfileArgs",
    "BenchmarkArgs",
    "BenchmarkMetadata",
    "BenchmarkOutputArgs",
    "BenchmarkScenario",
    "CSVBenchmarkOutputArgs",
    "ConcurrentProfileArgs",
    "ConsoleBenchmarkOutputArgs",
    "GenerativeMetricsArgs",
    "GoodputProfileArgs",
    "GoodputSLO",
    "HTMLBenchmarkOutputArgs",
    "JSONBenchmarkOutputArgs",
    "MetricsArgs",
    "PlotBenchmarkOutputArgs",
    "ProfileArgs",
    "RandomArgs",
    "ReplayProfileArgs",
    "StaticRandomArgs",
    "SweepProfileArgs",
    "SynchronousProfileArgs",
    "ThroughputProfileArgs",
    "TransientPhaseConfig",
    "YAMLBenchmarkOutputArgs",
    "args_model_config",
    "default_kind",
    "default_kind_list",
    "get_builtin_scenarios",
]
