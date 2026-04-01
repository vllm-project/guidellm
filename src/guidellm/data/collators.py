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

    Wraps single requests in a list for scheduler compatibility.
    Embeddings are single-turn, so each conversation contains one request.
    """

    def __call__(self, batch: list) -> list:
        """
        Collate batch of embeddings requests.

        :param batch: List of GenerationRequest objects (should be length 1)
        :return: List containing single GenerationRequest (for scheduler)
        :raises NotImplementedError: If batch size > 1
        """
        if len(batch) != 1:
            raise NotImplementedError(
                f"Batch size greater than 1 is not currently supported. "
                f"Got batch size: {len(batch)}"
            )

        # Return as list for scheduler (expects Iterable[Iterable[Request]])
        return [batch[0]]
