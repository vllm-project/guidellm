from __future__ import annotations

from collections import defaultdict
from typing import Any

from guidellm.data.preprocessors.preprocessor import (
    DatasetPreprocessor,
    PreprocessorRegistry,
)

__all__ = ["TurnPivot"]


@PreprocessorRegistry.register("turn_pivot")
class TurnPivot(DatasetPreprocessor):
    """
    Swaps the turn and batch dimensions in a multi-turn dataset.

    Example:
    ::
        # Input: 2 turns, each with 2 batches
        turns = [
            {"prompt": ["P1a", "P1b"], "output_tokens_count": [11, 12]},  # Turn 1
            {"prompt": ["P2a", "P2b"], "output_tokens_count": [21, 22]},  # Turn 2
        ]

        preprocessor = TurnPivot()
        turns = preprocessor(turns)

        # Resulting turns:
        # turns[0] = {"prompt": ["P1a", "P2a"], "output_tokens_count": [11, 21]}
        # turns[1] = {"prompt": ["P1b", "P2b"], "output_tokens_count": [12, 22]}
    """

    def __call__(self, items: list[dict[str, list[Any]]]) -> list[dict[str, list[Any]]]:
        new_turns: list[dict[str, list[Any]]] = []
        for turn in items:
            for column_name, values in turn.items():
                for i, value in enumerate(values):
                    if len(new_turns) <= i:
                        new_turns.append(defaultdict(list))
                    new_turns[i][column_name].append(value)

        return [dict(turn) for turn in new_turns]
