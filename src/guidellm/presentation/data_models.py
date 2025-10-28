import random
from collections import defaultdict
from math import ceil
from typing import TYPE_CHECKING

from pydantic import BaseModel, computed_field

if TYPE_CHECKING:
    from guidellm.benchmark import GenerativeBenchmark

from guidellm.utils import DistributionSummary


class Bucket(BaseModel):
    value: float | int
    count: int

    @staticmethod
    def from_data(
        data: list[float] | list[int],
        bucket_width: float | None = None,
        n_buckets: int | None = None,
    ) -> tuple[list["Bucket"], float]:
        if not data:
            return [], 1.0

        min_v = min(data)
        max_v = max(data)
        range_v = (1 + max_v) - min_v

        if bucket_width is None:
            if n_buckets is None:
                n_buckets = 10
            bucket_width = range_v / n_buckets
        else:
            n_buckets = ceil(range_v / bucket_width)

        bucket_counts: defaultdict[float | int, int] = defaultdict(int)
        for val in data:
            idx = int((val - min_v) // bucket_width)
            if idx >= n_buckets:
                idx = n_buckets - 1
            bucket_start = min_v + idx * bucket_width
            bucket_counts[bucket_start] += 1

        buckets = [
            Bucket(value=start, count=count)
            for start, count in sorted(bucket_counts.items())
        ]
        return buckets, bucket_width


class Model(BaseModel):
    name: str
    size: int


class Dataset(BaseModel):
    name: str


class RunInfo(BaseModel):
    model: Model
    task: str
    timestamp: float
    dataset: Dataset

    @classmethod
    def from_benchmarks(cls, benchmarks: list["GenerativeBenchmark"]):
        model = benchmarks[0].benchmarker.backend.get("model", "N/A")
        timestamp = max(
            bm.run_stats.start_time for bm in benchmarks if bm.start_time is not None
        )
        return cls(
            model=Model(name=model or "", size=0),
            task="N/A",
            timestamp=timestamp,
            dataset=Dataset(name="N/A"),
        )


class Distribution(BaseModel):
    statistics: DistributionSummary | None = None
    buckets: list[Bucket]
    bucket_width: float


class TokenDetails(BaseModel):
    samples: list[str]
    token_distributions: Distribution


class Server(BaseModel):
    target: str


class RequestOverTime(BaseModel):
    num_benchmarks: int
    requests_over_time: Distribution


class WorkloadDetails(BaseModel):
    prompts: TokenDetails
    generations: TokenDetails
    requests_over_time: RequestOverTime
    rate_type: str
    server: Server

    @classmethod
    def from_benchmarks(cls, benchmarks: list["GenerativeBenchmark"]):
        target = benchmarks[0].benchmarker.backend.get("target", "N/A")
        rate_type = benchmarks[0].scheduler.strategy.type_
        successful_requests = [
            req for bm in benchmarks for req in bm.requests.successful
        ]
        sample_indices = random.sample(
            range(len(successful_requests)), min(5, len(successful_requests))
        )
        sample_prompts = [
            req.request_args.replace("\n", " ").replace('"', "'")
            if (req := successful_requests[i]).request_args
            else ""
            for i in sample_indices
        ]
        sample_outputs = [
            req.output.replace("\n", " ").replace('"', "'")
            if (req := successful_requests[i]).output
            else ""
            for i in sample_indices
        ]

        prompt_tokens = [
            float(req.prompt_tokens) if req.prompt_tokens is not None else -1
            for bm in benchmarks
            for req in bm.requests.successful
        ]
        output_tokens = [
            float(req.output_tokens) if req.output_tokens is not None else -1
            for bm in benchmarks
            for req in bm.requests.successful
        ]

        prompt_token_buckets, _prompt_token_bucket_width = Bucket.from_data(
            prompt_tokens, 1
        )
        output_token_buckets, _output_token_bucket_width = Bucket.from_data(
            output_tokens, 1
        )

        prompt_token_stats = DistributionSummary.from_values(prompt_tokens)
        output_token_stats = DistributionSummary.from_values(output_tokens)
        prompt_token_distributions = Distribution(
            statistics=prompt_token_stats, buckets=prompt_token_buckets, bucket_width=1
        )
        output_token_distributions = Distribution(
            statistics=output_token_stats, buckets=output_token_buckets, bucket_width=1
        )

        min_start_time = benchmarks[0].start_time

        all_req_times = [
            req.info.timings.request_start - min_start_time
            for bm in benchmarks
            for req in bm.requests.successful
            if req.info.timings.request_start is not None
        ]
        number_of_buckets = len(benchmarks)
        request_over_time_buckets, bucket_width = Bucket.from_data(
            all_req_times, None, number_of_buckets
        )
        request_over_time_distribution = Distribution(
            buckets=request_over_time_buckets, bucket_width=bucket_width
        )
        return cls(
            prompts=TokenDetails(
                samples=sample_prompts, token_distributions=prompt_token_distributions
            ),
            generations=TokenDetails(
                samples=sample_outputs, token_distributions=output_token_distributions
            ),
            requests_over_time=RequestOverTime(
                requests_over_time=request_over_time_distribution,
                num_benchmarks=number_of_buckets,
            ),
            rate_type=rate_type,
            server=Server(target=target),
        )


class TabularDistributionSummary(DistributionSummary):
    """
    Same fields as `DistributionSummary`, but adds a ready-to-serialize/iterate
    `percentile_rows` helper.
    """

    @computed_field
    def percentile_rows(self) -> list[dict[str, str | float]]:
        rows = [
            {"percentile": name, "value": value}
            for name, value in self.percentiles.model_dump().items()
        ]
        return list(
            filter(lambda row: row["percentile"] in ["p50", "p90", "p95", "p99"], rows)
        )

    @classmethod
    def from_distribution_summary(
        cls, distribution: DistributionSummary
    ) -> "TabularDistributionSummary":
        return cls(**distribution.model_dump())


class BenchmarkDatum(BaseModel):
    requests_per_second: float
    itl: TabularDistributionSummary
    ttft: TabularDistributionSummary
    throughput: TabularDistributionSummary
    time_per_request: TabularDistributionSummary

    @classmethod
    def from_benchmark(cls, bm: "GenerativeBenchmark"):
        return cls(
            requests_per_second=bm.metrics.requests_per_second.successful.mean,
            itl=TabularDistributionSummary.from_distribution_summary(
                bm.metrics.inter_token_latency_ms.successful
            ),
            ttft=TabularDistributionSummary.from_distribution_summary(
                bm.metrics.time_to_first_token_ms.successful
            ),
            throughput=TabularDistributionSummary.from_distribution_summary(
                bm.metrics.output_tokens_per_second.successful
            ),
            time_per_request=TabularDistributionSummary.from_distribution_summary(
                bm.metrics.request_latency.successful
            ),
        )
