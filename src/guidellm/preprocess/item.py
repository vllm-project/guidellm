from collections.abc import Sequence
from typing import Generic, Optional, TypeVar, Union

from pydantic import Field

from guidellm.objects.pydantic import StandardBaseModel

PromptT = TypeVar("PromptT")


class Item(StandardBaseModel, Generic[PromptT]):
    """
    Represents a single item in a dataset, containing a prompt and its associated metadata.
    """

    value: PromptT = Field(
        description="The prompt text or data for the item.",
        examples=[
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
        ],
    )
    prompt_tokens: Optional[int] = Field(
        default=None, gt=0, description="Number of tokens in the prompt"
    )
    output_tokens: Optional[int] = Field(
        default=None, gt=0, description="Number of tokens in the output"
    )


class ItemList(Sequence[Item[PromptT]]):
    """
    Represents a list of items, each containing a prompt and its metadata.
    """

    def __init__(self, *items: Item[PromptT], shared_prefix: Optional[PromptT] = None):
        self.shared_prefix: Optional[PromptT] = shared_prefix
        self._items: list[Item[PromptT]] = list(items)

    def __getitem__(self, key) -> Union[Item[PromptT], Sequence[Item[PromptT]]]:
        return self._items[key]

    def __len__(self) -> int:
        return len(self._items)

    @classmethod
    def from_lists(
        cls,
        prompts: list[PromptT],
        prompts_tokens: list[Optional[int]],
        outputs_tokens: list[Optional[int]],
    ) -> "ItemList":
        return cls(
            *[
                Item(value=prompt, output_tokens=in_t, prompt_tokens=out_t)
                for prompt, in_t, out_t in zip(
                    prompts, prompts_tokens, outputs_tokens, strict=True
                )
            ]
        )
