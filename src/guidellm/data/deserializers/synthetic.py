from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from random import Random
from typing import Any

import numpy as np
from datasets import DatasetInfo, Features, IterableDataset, Value
from datasets.iterable_dataset import _BaseExamplesIterable
from faker import Faker
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationParentRef,
    ConversationTurnData,
)
from guidellm.schemas import RequestSettings
from guidellm.schemas.data.deserializers import (
    DEFAULT_SYNTHETIC_TOOLS,
    BranchSpec,
    SyntheticTextDataArgs,
    SyntheticTextPrefixBucketConfig,
)
from guidellm.settings import settings
from guidellm.utils.imports import json
from guidellm.utils.random import FloatRangeSampler, IntegerRangeSampler

__all__ = [
    "DEFAULT_SYNTHETIC_TOOLS",
    "BranchSpec",
    "SyntheticTextDataArgs",
    "SyntheticTextDataset",
    "SyntheticTextDatasetDeserializer",
    "SyntheticTextPrefixBucketConfig",
]


def _integer_range_sampler(
    average: int | None,
    variance: int | None,
    min_value: int | None,
    max_value: int | None,
    random_seed: int,
) -> Iterator[int] | None:
    """Build an ``IntegerRangeSampler`` iterator, or ``None`` if average is unset."""
    if average is None:
        return None
    return iter(
        IntegerRangeSampler(
            average=average,
            variance=variance,
            min_value=min_value,
            max_value=max_value,
            random_seed=random_seed,
        )
    )


class _SyntheticTextExamplesIterable(_BaseExamplesIterable):
    """Custom examples iterable for synthetic text generation."""

    def __init__(
        self,
        config: SyntheticTextDataArgs,
        processor: PreTrainedTokenizerBase,
        random_seed: int,
    ):
        super().__init__()
        self.config = config
        self.processor = processor
        self.random_seed = random_seed
        self.iteration_count = 0

    def __iter__(self) -> Iterator[tuple[int, dict[str, Any]]]:  # noqa: C901, PLR0915
        iter_random_seed = self.random_seed + self.iteration_count
        self.iteration_count += 1

        faker = Faker()
        faker.seed_instance(iter_random_seed)
        prompt_tokens_sampler: Iterator[int] = iter(
            IntegerRangeSampler(
                average=self.config.prompt_tokens,
                variance=self.config.prompt_tokens_stdev,
                min_value=self.config.prompt_tokens_min,
                max_value=self.config.prompt_tokens_max,
                random_seed=iter_random_seed,
            )
        )
        output_tokens_sampler = _integer_range_sampler(
            average=self.config.output_tokens,
            variance=self.config.output_tokens_stdev,
            min_value=self.config.output_tokens_min,
            max_value=self.config.output_tokens_max,
            random_seed=iter_random_seed + 1,  # ensure diff dist from prompts
        )
        first_prompt_tokens_sampler = _integer_range_sampler(
            average=self.config.first_prompt_tokens,
            variance=self.config.first_prompt_tokens_stdev,
            min_value=self.config.first_prompt_tokens_min,
            max_value=self.config.first_prompt_tokens_max,
            random_seed=iter_random_seed + 4,
        )
        first_output_tokens_sampler = _integer_range_sampler(
            average=self.config.first_output_tokens,
            variance=self.config.first_output_tokens_stdev,
            min_value=self.config.first_output_tokens_min,
            max_value=self.config.first_output_tokens_max,
            random_seed=iter_random_seed + 5,
        )
        delay_sampler = (
            iter(
                FloatRangeSampler(
                    average=self.config.delay,
                    variance=self.config.delay_stdev,
                    min_value=self.config.delay_min,
                    max_value=self.config.delay_max,
                    # ensure diff dist from prompts and outputs
                    random_seed=iter_random_seed + 2,
                )
            )
            if self.config.delay is not None
            else None
        )

        # Create a shared prefix if specified
        rand = Random(iter_random_seed + 3)
        prefix_iter = self._create_prefix_iter(faker, rand)
        samples_count = 0

        # Resolve tool definitions for client-side tool-call turns
        tool_call_turns_set = set(self.config.tool_call_turns)
        tools_defs: list[dict[str, Any]] | None = None
        if tool_call_turns_set:
            tools_defs = self.config.tools or DEFAULT_SYNTHETIC_TOOLS

        # Optional sampler for variable-length tool responses
        tool_response_sampler: Iterator[int] | None = None
        if self.config.tool_response_tokens is not None:
            tool_response_sampler = iter(
                IntegerRangeSampler(
                    average=self.config.tool_response_tokens,
                    variance=self.config.tool_response_tokens_stdev,
                    min_value=self.config.tool_response_tokens_min,
                    max_value=self.config.tool_response_tokens_max,
                    random_seed=iter_random_seed + 2,
                )
            )

        while True:
            delay = next(delay_sampler) if delay_sampler is not None else None
            row = self._create_conversation_row(
                faker=faker,
                samples_count=samples_count,
                prompt_tokens_sampler=prompt_tokens_sampler,
                output_tokens_sampler=output_tokens_sampler,
                first_prompt_tokens_sampler=first_prompt_tokens_sampler,
                first_output_tokens_sampler=first_output_tokens_sampler,
                delay=delay,
                prefix=next(prefix_iter),
                tools_defs=tools_defs,
                tool_response_sampler=tool_response_sampler,
                iter_random_seed=iter_random_seed,
            )
            # Count logical main turns, client-tool injection nodes (added by
            # the shared finalizer expander), and branch turns.
            client_tool_extras = sum(
                1 for i in tool_call_turns_set if i < self.config.turns
            )
            samples_count += (
                self.config.turns
                + client_tool_extras
                + sum(b.turns for b in self.config.branches)
            )
            yield samples_count, row

    @staticmethod
    def _sample_turn_tokens(
        turn_index: int,
        main_sampler: Iterator[int] | None,
        first_sampler: Iterator[int] | None,
    ) -> int | None:
        """Sample token count for a turn, using first_* on turn 0 when configured.

        Returns ``None`` when neither a first-turn nor main sampler applies
        (e.g. ``output_tokens`` omitted and no ``first_output_tokens``).
        """
        if turn_index == 0 and first_sampler is not None:
            return next(first_sampler)
        if main_sampler is not None:
            return next(main_sampler)
        return None

    @staticmethod
    def _sample_required_turn_tokens(
        turn_index: int,
        main_sampler: Iterator[int],
        first_sampler: Iterator[int] | None,
    ) -> int:
        """Sample a required prompt token count for a turn."""
        count = _SyntheticTextExamplesIterable._sample_turn_tokens(
            turn_index=turn_index,
            main_sampler=main_sampler,
            first_sampler=first_sampler,
        )
        if count is None:
            raise ValueError("prompt token sampler produced no value")
        return count

    def _create_conversation_row(  # noqa: C901 PLR0912 PLR0915
        self,
        faker: Faker,
        samples_count: int,
        prompt_tokens_sampler: Iterator[int],
        output_tokens_sampler: Iterator[int] | None,
        first_prompt_tokens_sampler: Iterator[int] | None,
        first_output_tokens_sampler: Iterator[int] | None,
        delay: float | None,
        prefix: str,
        tools_defs: list[dict[str, Any]] | None,
        tool_response_sampler: Iterator[int] | None,
        iter_random_seed: int,
    ) -> dict[str, Any]:
        """
        Build a ``conversation_turns`` payload for linear or branched graphs.

        Client ``tool_call_turns`` are emitted as a single logical turn that
        still carries ``tools_column`` and ``tool_response_column``. The shared
        finalizer expander splits those into tool-call + injection nodes and
        rewrites parent refs. ``BranchSpec.at_turn`` remains a logical
        conversation index.

        Token sizes are sampled independently per turn from the main
        distribution. Turn 0 of the main chain and of each branch may use
        ``first_*`` overrides (branch first inherits parent first when unset).

        :param faker: Seeded Faker for prompt text.
        :param samples_count: Counter used to uniquify prompts.
        :param prompt_tokens_sampler: Main prompt token sampler.
        :param output_tokens_sampler: Main output token sampler, if configured.
        :param first_prompt_tokens_sampler: Optional parent first-turn prompt sampler.
        :param first_output_tokens_sampler: Optional parent first-turn output sampler.
        :param delay: Optional requeue delay for main turns.
        :param prefix: Optional system prefix applied to the first main turn.
        :param tools_defs: Tool definitions for client tool-call turns.
        :param tool_response_sampler: Optional sampler for tool response size.
        :param iter_random_seed: Seed base for per-branch first-turn samplers.
        :return: A dataset row with a JSON ``conversation_turns`` column.
        """
        tool_call_turns = set(self.config.tool_call_turns)
        server_tool_call_turns = set(self.config.server_tool_call_turns)
        turn_settings = (
            RequestSettings(requeue_delay=delay) if delay is not None else None
        )

        turns: list[ConversationTurnData] = []

        for turn_idx in range(self.config.turns):
            parents: list[ConversationParentRef] = []
            if turn_idx > 0:
                parents.append(
                    ConversationParentRef(
                        parent_node_id=f"main_{turn_idx - 1}",
                        history_context="full",
                    )
                )
            for b_idx, branch in enumerate(self.config.branches):
                if branch.at_turn + branch.merge_after == turn_idx:
                    parents.append(
                        ConversationParentRef(
                            parent_node_id=f"branch_{b_idx}_{branch.turns - 1}",
                            history_context="last",
                        )
                    )

            prompt_tokens_count = self._sample_required_turn_tokens(
                turn_index=turn_idx,
                main_sampler=prompt_tokens_sampler,
                first_sampler=first_prompt_tokens_sampler,
            )
            output_tokens_count = self._sample_turn_tokens(
                turn_index=turn_idx,
                main_sampler=output_tokens_sampler,
                first_sampler=first_output_tokens_sampler,
            )
            text = self._create_prompt(
                prompt_tokens_count,
                faker,
                f"{self.iteration_count} {samples_count} m{turn_idx} ",
            )
            columns: dict[str, list[Any]] = {
                "text_column": [text],
                "prompt_tokens_count_column": [prompt_tokens_count],
            }
            if turn_idx == 0 and prefix:
                columns["prefix_column"] = [prefix]
            if output_tokens_count is not None:
                columns["output_tokens_count_column"] = [output_tokens_count]
            if turn_idx in server_tool_call_turns:
                columns["turn_type_column"] = ["server_tool_call"]

            if turn_idx in tool_call_turns:
                tools_raw = json.dumps(tools_defs or DEFAULT_SYNTHETIC_TOOLS)
                columns["tools_column"] = [
                    tools_raw.decode() if isinstance(tools_raw, bytes) else tools_raw
                ]
                if tool_response_sampler is not None:
                    tr_tokens = next(tool_response_sampler)
                    body = self._create_prompt(tr_tokens, faker)
                    response_raw = json.dumps({"result": body})
                    tool_response = (
                        response_raw.decode()
                        if isinstance(response_raw, bytes)
                        else response_raw
                    )
                else:
                    tool_response = settings.default_synthetic_tool_response
                columns["tool_response_column"] = [tool_response]

            turns.append(
                ConversationTurnData(
                    node_id=f"main_{turn_idx}",
                    agent_id="default",
                    parents=parents,
                    columns=columns,
                    settings=turn_settings,
                )
            )

        for b_idx, branch in enumerate(self.config.branches):
            # Branch-local first_* overrides; otherwise inherit parent first_* samplers
            branch_first_prompt = _integer_range_sampler(
                average=branch.first_prompt_tokens,
                variance=branch.first_prompt_tokens_stdev,
                min_value=branch.first_prompt_tokens_min,
                max_value=branch.first_prompt_tokens_max,
                random_seed=iter_random_seed + 10 + b_idx * 2,
            )
            branch_first_output = _integer_range_sampler(
                average=branch.first_output_tokens,
                variance=branch.first_output_tokens_stdev,
                min_value=branch.first_output_tokens_min,
                max_value=branch.first_output_tokens_max,
                random_seed=iter_random_seed + 11 + b_idx * 2,
            )
            resolved_first_prompt = (
                branch_first_prompt
                if branch_first_prompt is not None
                else first_prompt_tokens_sampler
            )
            resolved_first_output = (
                branch_first_output
                if branch_first_output is not None
                else first_output_tokens_sampler
            )

            for t in range(branch.turns):
                if t == 0:
                    parents = [
                        ConversationParentRef(
                            parent_node_id=f"main_{branch.at_turn}",
                            history_context="new",
                        )
                    ]
                else:
                    parents = [
                        ConversationParentRef(
                            parent_node_id=f"branch_{b_idx}_{t - 1}",
                            history_context="full",
                        )
                    ]

                branch_prompt_tokens = self._sample_required_turn_tokens(
                    turn_index=t,
                    main_sampler=prompt_tokens_sampler,
                    first_sampler=resolved_first_prompt,
                )
                branch_output_tokens = self._sample_turn_tokens(
                    turn_index=t,
                    main_sampler=output_tokens_sampler,
                    first_sampler=resolved_first_output,
                )

                branch_columns: dict[str, list[Any]] = {
                    "text_column": [
                        self._create_prompt(
                            branch_prompt_tokens,
                            faker,
                            f"{self.iteration_count} {samples_count} b{b_idx}_{t} ",
                        )
                    ],
                    "prompt_tokens_count_column": [branch_prompt_tokens],
                }
                if branch_output_tokens is not None:
                    branch_columns["output_tokens_count_column"] = [
                        branch_output_tokens
                    ]

                turns.append(
                    ConversationTurnData(
                        node_id=f"branch_{b_idx}_{t}",
                        agent_id=branch.agent_id,
                        parents=parents,
                        columns=branch_columns,
                    )
                )

        graph_data = ConversationGraphData(turns=turns)
        payload = json.dumps(graph_data.model_dump(mode="json"))
        return {
            "conversation_turns": (
                payload.decode() if isinstance(payload, bytes) else payload
            )
        }

    @property
    def is_typed(self) -> bool:
        return True

    @property
    def features(self) -> Features:
        return Features({"conversation_turns": Value("large_string")})

    @property
    def num_shards(self) -> int:
        return 1

    def shuffle_data_sources(
        self,
        generator: np.random.Generator,  # noqa: ARG002
    ) -> _SyntheticTextExamplesIterable:
        """Return self since synthetic data doesn't have fixed sources to shuffle."""
        return self

    def shard_data_sources(
        self,
        num_shards: int,  # noqa: ARG002
        index: int,  # noqa: ARG002
        contiguous: bool = True,  # noqa: ARG002
    ) -> _SyntheticTextExamplesIterable:
        """Return self since synthetic data generation is infinite and stateless."""
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state from a state dict."""
        self.iteration_count = state_dict.get("iteration_count", 0)

    def _init_state_dict(self) -> dict:
        """Initialize the state dict for the iterable."""
        self._state_dict = {"iteration_count": self.iteration_count}
        return self._state_dict

    def _create_prompt(
        self, prompt_tokens_count: int, faker: Faker, unique: str = ""
    ) -> str:
        prompt_token_ids: list[int] = []
        avg_chars_per_token = 5
        margin_of_safety = 1.5
        attempts = 0

        while len(prompt_token_ids) < prompt_tokens_count:
            attempts += 1
            num_chars = int(
                prompt_tokens_count * avg_chars_per_token * margin_of_safety * attempts
            )
            text = unique + faker.text(max_nb_chars=num_chars)
            prompt_token_ids = self.processor.encode(text)

        return self.processor.decode(  # type: ignore[return-value]
            prompt_token_ids[:prompt_tokens_count], skip_special_tokens=True
        )

    def _create_prefix_iter(self, faker: Faker, rand: Random) -> Iterator[str]:
        if not self.config.prefix_buckets:
            while True:
                yield ""

        # Increase weights to ensure an integer number of samples per per-prefix
        least_common_prefix_count = math.lcm(
            *(bucket.prefix_count for bucket in self.config.prefix_buckets)
        )
        unnorm_weights = [
            least_common_prefix_count * bucket.bucket_weight // bucket.prefix_count
            for bucket in self.config.prefix_buckets
        ]
        # Use GCD to reduce the weights to smallest integer ratio
        common_divisor = math.gcd(*unnorm_weights)

        # Create prefix list maintaining the correct distribution
        prefixes = []
        for bucket, weight in zip(
            self.config.prefix_buckets, unnorm_weights, strict=False
        ):
            bucket_prefixes = [
                self._create_prompt(bucket.prefix_tokens, faker)
                for _ in range(bucket.prefix_count)
            ]
            sample_count = weight // common_divisor
            prefixes.extend(bucket_prefixes * sample_count)

        while True:
            yield rand.choice(prefixes)


class SyntheticTextDataset(IterableDataset):
    def __init__(
        self,
        config: SyntheticTextDataArgs,
        processor: PreTrainedTokenizerBase,
        random_seed: int = 42,
    ):
        self.config = config
        self.processor = processor
        self.random_seed = random_seed

        # Create the examples iterable
        ex_iterable = _SyntheticTextExamplesIterable(
            config=config,
            processor=processor,
            random_seed=random_seed,
        )

        # Initialize parent with proper ex_iterable
        super().__init__(
            ex_iterable=ex_iterable,
            info=DatasetInfo(
                description="Synthetic text dataset generator",
                features=ex_iterable.features,
            ),
        )

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset iteration."""
        if isinstance(self._ex_iterable, _SyntheticTextExamplesIterable):
            self._ex_iterable.iteration_count = epoch


@DatasetDeserializerFactory.register("synthetic_text")
class SyntheticTextDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: SyntheticTextDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> IterableDataset:
        return SyntheticTextDataset(
            config=config,
            processor=processor_factory(),
            random_seed=random_seed,
        )
