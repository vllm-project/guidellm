from typing import Any, Protocol, TypeVar, runtime_checkable

from guidellm.schemas.request import GenerationRequest, UsageMetrics
from guidellm.utils.registry import RegistryMixin

DataT_co = TypeVar("DataT_co", covariant=True)


@runtime_checkable
class DatasetFinalizer(Protocol[DataT_co]):
    """
    Protocol for finalizing dataset rows into a desired data type.
    """

    def __call__(self, item: dict[str, Any]) -> DataT_co: ...


class FinalizerRegistry(RegistryMixin[type[DatasetFinalizer]]):
    pass


@FinalizerRegistry.register("generative")
class GenerativeRequestFinalizer(DatasetFinalizer[GenerationRequest]):
    """
    Finalizer that converts dataset rows into GenerationRequest objects,
    aggregating usage metrics from the provided columns.
    """

    def __call__(  # noqa: C901 PLR0912
        self, columns: dict[str, Any]
    ) -> GenerationRequest:
        input_metrics = UsageMetrics()
        output_metrics = UsageMetrics()

        # Sum prompt token column
        if prompt_tokens := sum(
            count for count in columns.get("prompt_tokens_count_column", []) if count
        ):
            input_metrics.text_tokens = prompt_tokens

        # Sum output token column
        if output_tokens := sum(
            count for count in columns.get("output_tokens_count_column", []) if count
        ):
            output_metrics.text_tokens = output_tokens

        # Count words in prefixes
        for prefix in columns.get("prefix_column", []):
            if not prefix:
                continue

            input_metrics.add_text_metrics(prefix)

        # Count words in text prompts
        for text in columns.get("text_column", []):
            if not text:
                continue

            input_metrics.add_text_metrics(text)

        # Count pixels and bytes in images
        for image in columns.get("image_column", []):
            if not image:
                continue

            if (image_pixels := image.get("image_pixels")) is not None:
                input_metrics.image_pixels = (
                    input_metrics.image_pixels or 0
                ) + image_pixels
            if (image_bytes := image.get("image_bytes")) is not None:
                input_metrics.image_bytes = (
                    input_metrics.image_bytes or 0
                ) + image_bytes

        # Count frames, seconds, and bytes in videos
        for video in columns.get("video_column", []):
            if not video:
                continue

            if (video_frames := video.get("video_frames")) is not None:
                input_metrics.video_frames = (
                    input_metrics.video_frames or 0
                ) + video_frames
            if (video_seconds := video.get("video_seconds")) is not None:
                input_metrics.video_seconds = (
                    input_metrics.video_seconds or 0.0
                ) + video_seconds
            if (video_bytes := video.get("video_bytes")) is not None:
                input_metrics.video_bytes = (
                    input_metrics.video_bytes or 0
                ) + video_bytes

        # Count samples, seconds, and bytes in audio
        for audio in columns.get("audio_column", []):
            if not audio:
                continue

            if (audio_samples := audio.get("audio_samples")) is not None:
                input_metrics.audio_samples = (
                    input_metrics.audio_samples or 0
                ) + audio_samples
            if (audio_seconds := audio.get("audio_seconds")) is not None:
                input_metrics.audio_seconds = (
                    input_metrics.audio_seconds or 0.0
                ) + audio_seconds
            if (audio_bytes := audio.get("audio_bytes")) is not None:
                input_metrics.audio_bytes = (
                    input_metrics.audio_bytes or 0
                ) + audio_bytes

        return GenerationRequest(
            columns=columns,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )


@FinalizerRegistry.register("embeddings")
class EmbeddingsRequestFinalizer(DatasetFinalizer[GenerationRequest]):
    """
    Finalizer that converts dataset rows into embeddings GenerationRequest objects.

    Much simpler than GenerativeRequestFinalizer since embeddings only need
    a text input field. Collects text from 'text_column' and creates a request
    with basic token/word counting.

    Example:
    ::
        finalizer = EmbeddingsRequestFinalizer()
        row = {"text_column": ["This is a test sentence"]}
        request = finalizer(row)
        # request.body["input"] == "This is a test sentence"
    """

    def __call__(self, columns: dict[str, Any]) -> GenerationRequest:
        """
        Convert dataset row to embeddings request.

        :param columns: Dict with 'text_column' containing text strings
        :return: GenerationRequest configured for embeddings
        """
        input_metrics = UsageMetrics()
        texts = []

        # Collect all text inputs
        for text in columns.get("text_column", []):
            if not text:
                continue

            texts.append(text)
            input_metrics.add_text_metrics(text)

        # For embeddings, input is a single text or list of texts
        if not texts:
            raise ValueError("No text found in dataset row for embeddings")

        # Create GenerationRequest with columns and metrics
        return GenerationRequest(
            columns=columns,
            input_metrics=input_metrics,
            output_metrics=UsageMetrics(),  # Embeddings have no output
        )
