from __future__ import annotations

from guidellm.schemas import GenerationRequest

__all__ = ["GenerativeRequestCollator"]


class GenerativeRequestCollator:
    def __call__(self, batch: list) -> GenerationRequest:
        if len(batch) != 1:
            raise NotImplementedError(
                f"Batch size greater than 1 is not currently supported. "
                f"Got batch size: {len(batch)}"
            )

        return batch[0]
