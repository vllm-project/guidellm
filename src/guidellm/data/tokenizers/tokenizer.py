from __future__ import annotations

from typing import Protocol, runtime_checkable

from transformers import PreTrainedTokenizerBase

from guidellm.data.schemas import DataTokenizerArgs
from guidellm.utils.registry import RegistryMixin

__all__ = ["DataTokenizer", "TokenizerRegistry"]


@runtime_checkable
class DataTokenizer(Protocol):
    def __init__(self, config: DataTokenizerArgs) -> None: ...

    def __call__(self) -> PreTrainedTokenizerBase: ...


class TokenizerRegistry(RegistryMixin[type[DataTokenizer]]):
    @classmethod
    def create(cls, config: DataTokenizerArgs) -> DataTokenizer:
        """
        Factory method to create a DatasetTokenizer instance based on configuration.

        :param config: A DataTokenizerArgs object containing the configuration.
        """
        kind = config.kind
        tokenizer_cls = cls.get_registered_object(kind)

        if tokenizer_cls is None:
            raise ValueError(
                f"DataTokenizer type '{kind}' is not registered."
                f"Available types: {list(cls.registry.keys()) if cls.registry else []}"
            )

        return tokenizer_cls(config)
