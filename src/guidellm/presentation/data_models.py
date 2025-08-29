import httpx
import random
from collections import defaultdict
from math import ceil
from typing import TYPE_CHECKING, Any, Optional, Union

from pydantic import BaseModel, computed_field

if TYPE_CHECKING:
    from guidellm.benchmark.benchmark import GenerativeBenchmark

from guidellm.dataset.file import FileDatasetCreator
from guidellm.dataset.hf_datasets import HFDatasetsCreator
from guidellm.dataset.in_memory import InMemoryDatasetCreator
from guidellm.dataset.synthetic import SyntheticDatasetConfig, SyntheticDatasetCreator
from guidellm.objects.statistics import DistributionSummary
from guidellm.preprocess.dataset import TokensConfig



class Bucket(BaseModel):
    value: Union[float, int]
    count: int

    @staticmethod
    def from_data(
        data: Union[list[float], list[int]],
        bucket_width: Optional[float] = None,
        n_buckets: Optional[int] = None,
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

        bucket_counts: defaultdict[Union[float, int], int] = defaultdict(int)
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

    @classmethod
    def from_data(cls, request_loader: Any):
        creators = [
            InMemoryDatasetCreator,
            SyntheticDatasetCreator,
            FileDatasetCreator,
            HFDatasetsCreator,
        ]
        dataset_name = ""
        data = request_loader.data
        data_args = request_loader.data_args
        processor = request_loader.processor
        processor_args = request_loader.processor_args
        
        for creator in creators:
            if creator.is_supported(data, None):
                random_seed = 42
                dataset = creator.handle_create(data, data_args, processor, processor_args, random_seed)
                dataset_name = creator.extract_dataset_name(dataset)
                if dataset_name is None or dataset_name == "":
                    if creator == SyntheticDatasetCreator:
                        data_dict = SyntheticDatasetConfig.parse_str(data)
                        dataset_name = data_dict.source
                    if creator == FileDatasetCreator or isinstance(creator, HFDatasetsCreator):
                        dataset_name = data
                    if creator == InMemoryDatasetCreator:
                        dataset_name = "In-memory"
                break
        return cls(
            name=dataset_name or ""
        )


class RunInfo(BaseModel):
    model: Model
    task: str
    timestamp: float
    dataset: Dataset

    @classmethod
    def from_benchmarks(cls, benchmarks: list["GenerativeBenchmark"]):
        model = benchmarks[0].worker.backend_model or "N/A"
        timestamp = max(
            bm.run_stats.start_time for bm in benchmarks if bm.start_time is not None
        )
        response = httpx.get(f"https://huggingface.co/api/models/{model}")
        modelJson = response.json()
            
        return cls(
            model=Model(name=model, size=modelJson.get("usedStorage", 0)),
            task="N/A",
            timestamp=timestamp,
            dataset=Dataset.from_data(benchmarks[0].request_loader),
        )
    

class Distribution(BaseModel):
    statistics: Optional[DistributionSummary] = None
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
        target = benchmarks[0].worker.backend_target
        rate_type = benchmarks[0].args.profile.type_
        successful_requests = [
            req for bm in benchmarks for req in bm.requests.successful
        ]
        sample_indices = random.sample(
            range(len(successful_requests)), min(5, len(successful_requests))
        )
        sample_prompts = [
            successful_requests[i].prompt.replace("\n", " ").replace('"', "'")
            for i in sample_indices
        ]
        sample_outputs = [
            successful_requests[i].output.replace("\n", " ").replace('"', "'")
            for i in sample_indices
        ]

        prompt_tokens = [
            float(req.prompt_tokens)
            for bm in benchmarks
            for req in bm.requests.successful
        ]
        output_tokens = [
            float(req.output_tokens)
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

        min_start_time = benchmarks[0].run_stats.start_time

        all_req_times = [
            req.start_time - min_start_time
            for bm in benchmarks
            for req in bm.requests.successful
            if req.start_time is not None
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
    def percentile_rows(self) -> list[dict[str, float]]:
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
    tpot: TabularDistributionSummary
    ttft: TabularDistributionSummary
    throughput: TabularDistributionSummary
    time_per_request: TabularDistributionSummary

    @classmethod
    def from_benchmark(cls, bm: "GenerativeBenchmark"):
        return cls(
            requests_per_second=bm.metrics.requests_per_second.successful.mean,
            tpot=TabularDistributionSummary.from_distribution_summary(
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
