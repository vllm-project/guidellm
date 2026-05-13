"""Programmatic API for multi-image benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from guidellm.data.generators.multi_image import generate_synthetic_images
from guidellm.data.schemas import MultiImageDatasetConfig

__all__ = ["MultiImageBenchmark", "MultiImageBenchmarkResults"]


@dataclass
class MultiImageBenchmarkResults:
    """Results from multi-image benchmark comparing multiple frame counts."""

    results: dict[int, Any]  # {image_count: benchmark_result}

    def ttft_by_count(self) -> dict[int, float]:
        """Return mean TTFT (ms) for each image count."""
        ttft = {}
        for img_count, result in self.results.items():
            if hasattr(result, "requests") and result.requests and hasattr(result.requests, "stats"):
                if hasattr(result.requests.stats, "ttft_ms"):
                    ttft[img_count] = result.requests.stats.ttft_ms.mean
        return ttft

    def itl_by_count(self) -> dict[int, float]:
        """Return mean ITL (ms) for each image count."""
        itl = {}
        for img_count, result in self.results.items():
            if hasattr(result, "requests") and result.requests and hasattr(result.requests, "stats"):
                if hasattr(result.requests.stats, "itl_ms"):
                    itl[img_count] = result.requests.stats.itl_ms.mean
        return itl


class MultiImageBenchmark:
    """
    Benchmark latency impact of multiple images per request.

    Example:
        bench = MultiImageBenchmark(
            image_counts=[1, 2, 5],
            prompt_tokens=256,
            output_tokens=128,
        )
        config_dict = bench.get_configs()
        # Use configs with benchmark runner
    """

    def __init__(
        self,
        image_counts: list[int],
        prompt_tokens: int = 256,
        output_tokens: int = 128,
        image_size: str = "720p",
        random_seed: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize multi-image benchmark configuration.

        Args:
            image_counts: List of image counts to benchmark (e.g., [1, 2, 5])
            prompt_tokens: Average prompt token count
            output_tokens: Average output token count
            image_size: Image resolution ("720p")
            random_seed: Random seed for reproducible image generation
            **kwargs: Additional arguments for MultiImageDatasetConfig
        """
        self.image_counts = sorted(image_counts)
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.image_size = image_size
        self.random_seed = random_seed
        self.kwargs = kwargs

    def get_configs(self) -> dict[int, MultiImageDatasetConfig]:
        """
        Get MultiImageDatasetConfig for each image count.

        Returns:
            Dict mapping image_count to MultiImageDatasetConfig
        """
        configs = {}
        for img_count in self.image_counts:
            configs[img_count] = MultiImageDatasetConfig(
                prompt_tokens=self.prompt_tokens,
                output_tokens=self.output_tokens,
                images_per_request=img_count,
                image_size=self.image_size,
                **self.kwargs,
            )
        return configs

    def generate_images(self, img_count: int) -> tuple[list[dict], int, int]:
        """
        Generate synthetic images for a given count.

        Args:
            img_count: Number of images to generate

        Returns:
            Tuple of (images_list, total_pixels, total_bytes)
        """
        return generate_synthetic_images(
            num_images=img_count,
            image_size=self.image_size,
            seed=self.random_seed,
        )

    def get_image_stats(self, img_count: int) -> dict[str, int]:
        """
        Get image statistics (pixels, bytes) for a given count.

        Args:
            img_count: Number of images

        Returns:
            Dict with 'total_pixels' and 'total_bytes'
        """
        _, total_pixels, total_bytes = self.generate_images(img_count)
        return {
            "image_count": img_count,
            "total_pixels": total_pixels,
            "total_bytes": total_bytes,
            "pixels_per_image": (total_pixels // img_count) if img_count > 0 else 0,
            "bytes_per_image": (total_bytes // img_count) if img_count > 0 else 0,
        }
