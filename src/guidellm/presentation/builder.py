from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from guidellm.benchmark import GenerativeBenchmark

from guidellm.presentation.data_models import BenchmarkDatum, RunInfo, WorkloadDetails


class UIDataBuilder:
    def __init__(self, benchmarks: list["GenerativeBenchmark"]):
        self.benchmarks = benchmarks

    def build_run_info(self):
        return RunInfo.from_benchmarks(self.benchmarks)

    def build_workload_details(self):
        return WorkloadDetails.from_benchmarks(self.benchmarks)

    def build_benchmarks(self):
        return [BenchmarkDatum.from_benchmark(b) for b in self.benchmarks]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_info": self.build_run_info().model_dump(),
            "workload_details": self.build_workload_details().model_dump(),
            "benchmarks": [b.model_dump() for b in self.build_benchmarks()],
        }
