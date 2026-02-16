from __future__ import annotations

from guidellm.schemas import GenerationRequest

__all__ = ["EmbeddingsRequestCollator", "GenerativeRequestCollator"]


class GenerativeRequestCollator:
    """
    Collator for generative (chat/completion) requests.

    Currently enforces batch size of 1 - batching not yet supported.
    """

    def __call__(self, batch: list) -> GenerationRequest:
        if len(batch) != 1:
            raise NotImplementedError(
                f"Batch size greater than 1 is not currently supported. "
                f"Got batch size: {len(batch)}"
            )

        return batch[0]


class EmbeddingsRequestCollator:
    """
    Collator for embeddings requests.

    Simple pass-through that enforces batch size of 1. Embeddings requests
    are already properly formatted by the EmbeddingsRequestFinalizer.
    """

    def __call__(self, batch: list) -> GenerationRequest:
        """
        Collate batch of embeddings requests.

        :param batch: List of GenerationRequest objects (should be length 1)
        :return: Single GenerationRequest
        :raises NotImplementedError: If batch size > 1
        """
        if len(batch) != 1:
            raise NotImplementedError(
                f"Batch size greater than 1 is not currently supported. "
                f"Got batch size: {len(batch)}"
            )

        return batch[0]
